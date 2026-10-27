#!/usr/bin/env python3
"""Convert a saved pySLAM map (slam_state/map.json, C++-core format) plus the KITTI-layout
images it was built from into a nerfstudio dataset for splatfacto.

Streams map.json (the inner "map" field is a JSON string inside JSON; 5+ GB is fine):
memory use stays in the low hundreds of MB.

    uv pip install ijson numpy
    map2nerfstudio.py --map slam_state/map.json --seq kitti/sequences/02 --out ns_data \
        [--max-images 600] [--use-right] [--drop-blurriest 0.2] [--min-obs 2] [--crop-margin 3]
        [--odom kitti/sequences/01/odom.csv]
        [--traj trajectory_final.txt --traj-step-m 0.05 --traj-step-deg 5]

Output:
    <out>/images/        copies of the selected rectified frames (l_XXXXXX.png [, r_XXXXXX.png])
    <out>/transforms.json  OPENCV model, zero distortion, camera-to-world in OpenGL convention
    <out>/sparse_pc.ply  map points (xyz + rgb) in the same world frame, used to init Gaussians
    <out>/keyframes.tsv  img_id, timestamp, Twc (OpenCV convention, 3x4 row-major), for reference
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

try:
    import ijson
except ImportError:
    sys.exit("pip install ijson")


# ---------------------------------------------------------------- streaming unescape
CHUNK = 32 << 20
_TOKEN = re.compile(rb'\\(?:u[0-9a-fA-F]{4}|[^u])|"')
_SIMPLE = {b'"': b'"', b"\\": b"\\", b"/": b"/", b"b": b"\b", b"f": b"\f",
           b"n": b"\n", b"r": b"\r", b"t": b"\t"}


class JsonStringReader:
    """File-like reader returning the decoded contents of one JSON string value.

    `f` must be positioned just after the opening quote. Reading stops at the closing quote.
    """

    def __init__(self, f, chunk=CHUNK):
        self.f, self.chunk = f, chunk
        self.carry = b""
        self.out = bytearray()
        self.done = False
        self.raw_read = 0

    def _fill(self):
        raw = self.f.read(self.chunk)
        self.raw_read += len(raw)
        data = self.carry + raw
        self.carry = b""
        if not raw and not data:
            raise ValueError("unterminated JSON string")
        # fast path: every backslash in the chunk is part of an escaped quote (the common case)
        if b"\x00" not in data and data.count(b"\\") == data.count(b'\\"'):
            tmp = data.replace(b'\\"', b"\x00")
            end = tmp.find(b'"')
            if end >= 0:
                tmp = tmp[:end]
                self.done = True
            self.out += tmp.replace(b"\x00", b'"')
            return
        pieces, prev = [], 0
        for m in _TOKEN.finditer(data):
            tok = m.group()
            pieces.append(data[prev:m.start()])
            if tok == b'"':
                self.done = True
                prev = None
                break
            if tok[1:2] == b"u":
                pieces.append(chr(int(tok[2:], 16)).encode("utf-8"))
            else:
                esc = _SIMPLE.get(tok[1:2])
                if esc is None:
                    raise ValueError(f"bad escape {tok!r}")
                pieces.append(esc)
            prev = m.end()
        if prev is not None:
            tail = data[prev:]
            bs = tail.find(b"\\")
            if bs >= 0 and raw:  # dangling incomplete escape: carry it to the next chunk
                pieces.append(tail[:bs])
                self.carry = tail[bs:]
            else:
                pieces.append(tail)
        self.out += b"".join(pieces)

    def read(self, n=-1):
        while not self.done and (n < 0 or len(self.out) < n):
            self._fill()
        if n < 0 or n >= len(self.out):
            res, self.out = bytes(self.out), bytearray()
        else:
            res = bytes(self.out[:n])
            del self.out[:n]
        return res


def open_inner_map(path):
    """Return (outer file handle, reader over the decoded inner map JSON)."""
    f = open(path, "rb")
    head = f.read(1 << 16)
    key = b'"map":'
    i = head.find(key)
    if i < 0:
        sys.exit(f'{path}: no "map" key in the first 64 KB')
    j = i + len(key)
    while head[j:j + 1] in (b" ", b"\t", b"\n", b"\r"):
        j += 1
    if head[j:j + 1] == b"{":
        sys.exit("map is a plain JSON object (Python-core format); this tool expects the C++-core string form")
    if head[j:j + 1] != b'"':
        sys.exit(f"unexpected token after \"map\": {head[j:j+20]!r}")
    f.seek(j + 1)
    return f, JsonStringReader(f)


def stream_items(path, prefix):
    f, reader = open_inner_map(path)
    try:
        for item in ijson.items(reader, prefix, use_float=True):
            yield item, reader.raw_read
    finally:
        f.close()


# ---------------------------------------------------------------- helpers
def read_camera_txt(p):
    d = {}
    for line in Path(p).read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = float(v)
    return d


def write_ply(path, xyz, rgb):
    n = len(xyz)
    arr = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode()
    with open(path, "wb") as fo:
        fo.write(header)
        fo.write(arr.tobytes())


def progress(tag, n, raw, total, t0):
    dt = time.time() - t0
    print(f"\r  {tag}: {n} items, {raw / 1e9:.2f}/{total / 1e9:.2f} GB read, {raw / 1e6 / max(dt, 1e-6):.0f} MB/s",
          end="", flush=True)


def quat_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def add_traj_frames(a, seq, sel, kf_ids):
    """Add tracked non-keyframes from trajectory_final.txt (t x y z qx qy qz qw, Twc), spaced
    at least --traj-step-m or --traj-step-deg from the previously kept frame."""
    import cv2

    times = np.atleast_1d(np.loadtxt(seq / "times.txt"))
    traj = np.atleast_2d(np.loadtxt(a.traj))
    cand = []
    for row in traj:
        i = int(np.argmin(np.abs(times - row[0])))
        if abs(times[i] - row[0]) > 1e-3 or i in kf_ids:
            continue
        Twc = np.eye(4)
        Twc[:3, :3], Twc[:3, 3] = quat_to_R(row[4:8]), row[1:4]
        if np.allclose(Twc, np.eye(4), atol=1e-9) and i > 1:
            continue  # pySLAM writes identity while tracking is lost
        cand.append(dict(img_id=i, ts=float(row[0]), Twc=Twc))
    # pySLAM occasionally writes a pose metres away for a frame or two; drop poses that sit far
    # from the median of their neighbours
    if len(cand) > 5:
        Cc = np.array([c["Twc"][:3, 3] for c in cand])
        pad = np.pad(Cc, ((2, 2), (0, 0)), mode="edge")
        med = np.median(np.stack([pad[k:k + len(Cc)] for k in range(5)]), axis=0)
        ok = np.linalg.norm(Cc - med, axis=1) < 0.15
        if not ok.all():
            print(f"trajectory: dropped {np.sum(~ok)} poses that jump away from their neighbours")
        cand = [c for c, k in zip(cand, ok) if k]
    for c in cand:
        g = cv2.imread(str(seq / "image_0" / f"{c['img_id']:06d}.png"), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        c["lap"] = float(cv2.Laplacian(g, cv2.CV_64F).var())
    if a.drop_blurriest > 0 and cand:
        thr = np.percentile([c["lap"] for c in cand], 100 * a.drop_blurriest)
        cand = [c for c in cand if c["lap"] >= thr]
    merged = sorted(sel + cand, key=lambda k: k["img_id"])
    sel_ids = {k["img_id"] for k in sel}
    kept, last = [], None
    for k in merged:
        if k["img_id"] in sel_ids:
            kept.append(k)
            last = k
            continue
        if last is not None:
            d = np.linalg.norm(k["Twc"][:3, 3] - last["Twc"][:3, 3])
            c = (np.trace(last["Twc"][:3, :3].T @ k["Twc"][:3, :3]) - 1) / 2
            if d < a.traj_step_m and np.degrees(np.arccos(np.clip(c, -1, 1))) < a.traj_step_deg:
                continue
        kept.append(k)
        last = k
    print(f"trajectory {a.traj}: {len(traj)} poses, added {len(kept) - len(sel)} non-keyframes "
          f"(step {a.traj_step_m} m / {a.traj_step_deg} deg), {len(kept)} frames total")
    return kept


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="slam_state/map.json")
    ap.add_argument("--seq", required=True, help="KITTI sequence dir the map was built from (image_0, camera.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=0, help="uniformly subsample keyframes to at most N (0 = all)")
    ap.add_argument("--use-right", action="store_true", help="also add right images (pose = left + baseline along x)")
    ap.add_argument("--drop-blurriest", type=float, default=0.0,
                    help="drop this fraction of keyframes with the lowest Laplacian variance (pySLAM's is_blurry "
                         "flag uses a fixed threshold and can mark everything)")
    ap.add_argument("--min-obs", type=int, default=0, help="drop map points with fewer observations")
    ap.add_argument("--crop-margin", type=float, default=3.0,
                    help="drop map points farther than this [m] outside the camera bounding box (<0 disables)")
    ap.add_argument("--swap-rb", action="store_true", help="swap red/blue of point colors if they look wrong")
    ap.add_argument("--odom", help="odom.csv from bag2kitti, for a path-length scale check")
    ap.add_argument("--link", action="store_true", help="symlink images instead of copying")
    ap.add_argument("--traj", help="pySLAM trajectory_final.txt (TUM format): also add tracked non-keyframes, "
                                   "for runs where pySLAM made few keyframes")
    ap.add_argument("--traj-step-m", type=float, default=0.05, help="min translation between added frames")
    ap.add_argument("--traj-step-deg", type=float, default=5.0, help="or min rotation between added frames")
    a = ap.parse_args()

    seq, out = Path(a.seq), Path(a.out)
    cam = read_camera_txt(seq / "camera.txt")
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    W, H, b = int(cam["width"]), int(cam["height"]), cam["baseline_m"]
    n_img = len(np.atleast_1d(np.loadtxt(seq / "times.txt")))
    total = Path(a.map).stat().st_size
    print(f"map {a.map} ({total / 1e9:.2f} GB), seq {seq} ({n_img} frames), {W}x{H} fx {fx:.3f} b {b:.6f} m")

    # ---- pass 1: keyframes
    print("pass 1/2: keyframes")
    kfs, t0 = [], time.time()
    for k, raw in stream_items(a.map, "keyframes.item"):
        T = np.array(k["pose"], dtype=np.float64).reshape(4, 4)
        kfs.append(dict(id=k.get("id"), img_id=int(k["img_id"]), ts=float(k.get("timestamp", 0.0)),
                        Tcw=T, blurry=bool(k.get("is_blurry", False)), lap=k.get("laplacian_var")))
        if len(kfs) % 50 == 0:
            progress("keyframes", len(kfs), raw, total, t0)
    progress("keyframes", len(kfs), total, total, t0)
    print()
    if not kfs:
        sys.exit("no keyframes found")

    # ---- pass 2: points
    print("pass 2/2: map points")
    xyz, rgb, t0, n_seen = [], [], time.time(), 0
    for p, raw in stream_items(a.map, "points.item"):
        n_seen += 1
        if a.min_obs and p.get("_num_observations", a.min_obs) < a.min_obs:
            continue
        if p.get("_is_bad", False):
            continue
        xyz.append(p["pt"])
        rgb.append(p.get("color", [128, 128, 128]))
        if n_seen % 20000 == 0:
            progress("points", n_seen, raw, total, t0)
    progress("points", n_seen, total, total, t0)
    print()
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.clip(np.asarray(rgb, dtype=np.float64).reshape(-1, 3), 0, 255).astype(np.uint8)
    if a.swap_rb:
        rgb = rgb[:, ::-1].copy()
    fin = np.isfinite(xyz).all(1)
    xyz, rgb = xyz[fin], rgb[fin]
    n_finite = len(xyz)

    # ---- sanity checks on poses
    kfs.sort(key=lambda k: k["img_id"])
    Rerr = max(np.abs(k["Tcw"][:3, :3] @ k["Tcw"][:3, :3].T - np.eye(3)).max() for k in kfs)
    bad_ids = [k["img_id"] for k in kfs if not (0 <= k["img_id"] < n_img)]
    if Rerr > 1e-3:
        sys.exit(f"pose rotation not orthonormal (err {Rerr:.2e}); pose layout is not what this tool expects")
    if bad_ids:
        sys.exit(f"{len(bad_ids)} keyframe img_ids outside 0..{n_img - 1} (e.g. {bad_ids[:5]}); wrong --seq?")
    for k in kfs:
        k["Twc"] = np.linalg.inv(k["Tcw"])
    C = np.array([k["Twc"][:3, 3] for k in kfs])
    path_len = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
    if a.crop_margin >= 0 and len(xyz):
        lo, hi = C.min(0) - a.crop_margin, C.max(0) + a.crop_margin
        keep = ((xyz >= lo) & (xyz <= hi)).all(1)
        xyz, rgb = xyz[keep], rgb[keep]
        print(f"cropped map points to camera bbox +/- {a.crop_margin} m: {len(xyz)} of {n_finite} kept")
    laps = np.array([k["lap"] if k["lap"] is not None else np.nan for k in kfs], float)
    print(f"keyframes {len(kfs)} (img_id {kfs[0]['img_id']}..{kfs[-1]['img_id']}), "
          f"pySLAM is_blurry {sum(k['blurry'] for k in kfs)}, rot err {Rerr:.1e}")
    if np.isfinite(laps).any():
        q = np.nanpercentile(laps, [5, 25, 50, 75, 95])
        print("laplacian_var percentiles 5/25/50/75/95: " + " ".join(f"{v:.1f}" for v in q))
    print(f"keyframe path length {path_len:.2f} m, camera bbox {np.ptp(C, axis=0).round(2)} m")
    print(f"map points {len(xyz)} kept of {n_seen}, bbox {np.ptp(xyz, axis=0).round(2) if len(xyz) else '-'} m")

    if a.odom and (seq / "times_ns.txt").exists():
        tns = np.loadtxt(seq / "times_ns.txt", dtype=np.int64)
        o = np.loadtxt(a.odom, delimiter=",", skiprows=1)
        t_kf = np.array([tns[k["img_id"]] for k in kfs], dtype=np.float64)
        o = o[(o[:, 0] >= t_kf[0] - 1e9) & (o[:, 0] <= t_kf[-1] + 1e9)]
        if len(o) > 10:
            ox = np.interp(t_kf, o[:, 0], o[:, 1])
            oy = np.interp(t_kf, o[:, 0], o[:, 2])
            yaw = np.unwrap(2 * np.arctan2(o[:, 6], o[:, 7]))  # planar yaw from qz, qw
            oyaw = np.interp(t_kf, o[:, 0], yaw)
            ol = float(np.hypot(np.diff(ox), np.diff(oy)).sum())
            print(f"path length at keyframe times: SLAM {path_len:.2f} m, odom {ol:.2f} m, ratio {path_len / ol:.3f}")
            # scale from straight segments: displacement over ~2 s with little rotation
            ratios = []
            for i0 in range(len(kfs)):
                i1 = np.searchsorted(t_kf, t_kf[i0] + 2e9)
                if i1 >= len(kfs):
                    break
                d_o = np.hypot(ox[i1] - ox[i0], oy[i1] - oy[i0])
                if d_o < 0.3 or abs(oyaw[i1] - oyaw[i0]) > np.radians(5):
                    continue
                ratios.append(np.linalg.norm(C[i1] - C[i0]) / d_o)
            if ratios:
                r = np.percentile(ratios, [10, 50, 90])
                print(f"straight-segment scale SLAM/odom over {len(ratios)} windows: "
                      f"median {r[1]:.3f} (p10 {r[0]:.3f}, p90 {r[2]:.3f})")
            else:
                print("no straight segments >= 0.3 m found for a scale check")

    # ---- select keyframes
    sel = list(kfs)
    if a.drop_blurriest > 0 and np.isfinite(laps).any():
        thr = np.nanpercentile(laps, 100 * a.drop_blurriest)
        sel = [k for k in kfs if k["lap"] is None or k["lap"] >= thr]
        print(f"dropped {len(kfs) - len(sel)} blurriest keyframes (laplacian_var < {thr:.1f})")
    if a.max_images and len(sel) > a.max_images:
        idx = np.round(np.linspace(0, len(sel) - 1, a.max_images)).astype(int)
        sel = [sel[i] for i in sorted(set(idx))]
    print(f"selected {len(sel)} keyframes" + (" (+ right images)" if a.use_right else ""))
    if not sel:
        sys.exit("no keyframes selected")

    # ---- extra non-keyframes from the final trajectory
    if a.traj:
        sel = add_traj_frames(a, seq, sel, {k["img_id"] for k in kfs})

    # ---- drop isolated pose glitches: a frame far from both neighbours while they are close
    sel.sort(key=lambda k: k["img_id"])
    Cs = np.array([k["Twc"][:3, 3] for k in sel])
    bad = [i for i in range(1, len(sel) - 1)
           if min(np.linalg.norm(Cs[i] - Cs[i - 1]), np.linalg.norm(Cs[i] - Cs[i + 1])) > 0.3
           and np.linalg.norm(Cs[i + 1] - Cs[i - 1]) < 0.3]
    if bad:
        print(f"dropped {len(bad)} frames with glitch poses: img_id {[sel[i]['img_id'] for i in bad]}")
        sel = [k for i, k in enumerate(sel) if i not in set(bad)]

    # ---- write dataset
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera axes -> OpenGL (nerfstudio)
    T_lr = np.eye(4)
    T_lr[0, 3] = b  # right camera centre is +b along left camera x
    frames, rows = [], []

    def put(src, name):
        dst = img_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if a.link:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

    for k in sel:
        i = k["img_id"]
        src_l = seq / "image_0" / f"{i:06d}.png"
        if not src_l.exists():
            sys.exit(f"missing {src_l}")
        put(src_l, f"l_{i:06d}.png")
        frames.append({"file_path": f"images/l_{i:06d}.png", "transform_matrix": (k["Twc"] @ flip).tolist()})
        rows.append(f"{i}\t{k['ts']:.6f}\t" + " ".join(f"{v:.9g}" for v in k["Twc"][:3].ravel()))
        if a.use_right:
            src_r = seq / "image_1" / f"{i:06d}.png"
            put(src_r, f"r_{i:06d}.png")
            frames.append({"file_path": f"images/r_{i:06d}.png",
                           "transform_matrix": (k["Twc"] @ T_lr @ flip).tolist()})

    write_ply(out / "sparse_pc.ply", xyz, rgb)
    tf = {"camera_model": "OPENCV", "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": W, "h": H,
          "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
          "ply_file_path": "sparse_pc.ply", "frames": frames}
    (out / "transforms.json").write_text(json.dumps(tf, indent=1))
    (out / "keyframes.tsv").write_text("img_id\ttimestamp\tTwc_3x4_row_major_opencv\n" + "\n".join(rows) + "\n")
    print(f"wrote {out}/transforms.json ({len(frames)} frames), sparse_pc.ply ({len(xyz)} points), images/")


if __name__ == "__main__":
    main()
