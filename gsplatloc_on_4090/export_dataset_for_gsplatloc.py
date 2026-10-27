#!/usr/bin/env python3
"""Export one of our nerfstudio datasets (map2nerfstudio.py / merge_legs.py / stereo_densify.py
output) in the folder layout GSplatLoc's train.py and loc_inference.py read.

    export_dataset_for_gsplatloc.py --data ns_data/seq04_merged \
        --out bench/gsplatloc_for_4090/data/oct04_both_legs_left_only_half_res --scale 0.5 [--use-right]

GSplatLoc (github be2rlab/gsplatloc) is built on the original 3DGS code. What that implies:
  * It reads  <out>/images/*.png  and  <out>/sparse/0/{cameras.txt, images.txt, points3D.ply}.
  * Its renderer and its localizer both assume the principal point is the exact image centre.
    Ours is 10 to 12 px off, so every image is cropped here to put it at the centre.
  * With --eval it holds out every 8th image (index % 8 == 2 after sorting by file name) as the
    test set, and loc_inference.py localizes exactly those. File names here are numbered so that
    sorting gives the order we want.
  * With --use-right, the right image of a held-out left image is left out of the dataset, and no
    right image is ever held out. Otherwise the test would be trivial (twin 12 cm away).
Poses stay metric (metres), so its reported errors in cm are real.

Also writes <out>/split.tsv (name, train/test, original frame, side, camera-to-world pose) for
scoring its output with our own tools later.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL (nerfstudio) <-> OpenCV camera axes
HOLD, PHASE = 8, 2  # GSplatLoc: test image if index % 8 == 2


def rot_to_quat(R):
    """Rotation matrix -> (qw, qx, qy, qz), COLMAP order."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def read_ply_xyzrgb(path):
    raw = Path(path).read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    n = int([ln for ln in head.split(b"\n") if ln.startswith(b"element vertex")][0].split()[-1])
    dt = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
    if b"binary_little_endian" not in head or len(body) != n * dt.itemsize:
        raise SystemExit(f"{path}: unexpected ply layout")
    d = np.frombuffer(body, dt)
    return d["xyz"], d["rgb"]


def write_ply_with_normals(path, xyz, rgb):
    dt = np.dtype([("xyz", "<f4", 3), ("n", "<f4", 3), ("rgb", "u1", 3)])
    d = np.zeros(len(xyz), dt)
    d["xyz"], d["rgb"] = xyz, rgb
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(xyz))
    Path(path).write_bytes(head.encode() + d.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="nerfstudio dataset folder (transforms.json, images/, dense_pc.ply)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="resize factor before cropping (0.5 for the 2K recording)")
    ap.add_argument("--use-right", action="store_true", help="also export right images for training (never as test images)")
    ap.add_argument("--max-frame", type=int, help="ignore frames after this original frame id")
    ap.add_argument("--voxel", type=float, default=0.05, help="thin the starting point cloud to one point per voxel [m]")
    ap.add_argument("--max-points", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    tf = json.loads((data / "transforms.json").read_text())
    by_id = {}
    for f in tf["frames"]:
        side, idx = Path(f["file_path"]).stem.split("_")
        if a.max_frame is not None and int(idx) > a.max_frame:
            continue
        by_id.setdefault(int(idx), {})[side] = f
    ids = sorted(i for i in by_id if "l" in by_id[i])

    # intrinsics after resize, then a crop that puts the principal point at the image centre
    s = a.scale
    W, H = round(tf["w"] * s), round(tf["h"] * s)
    fx, fy, cx, cy = tf["fl_x"] * s, tf["fl_y"] * s, tf["cx"] * s, tf["cy"] * s
    hw = int(np.floor(min(cx, W - cx))) // 2 * 2
    hh = int(np.floor(min(cy, H - cy))) // 2 * 2
    x0, y0 = int(round(cx - hw)), int(round(cy - hh))
    Wc, Hc = 2 * hw, 2 * hh
    off = (cx - x0 - hw, cy - y0 - hh)
    print(f"source {tf['w']}x{tf['h']} -> resized {W}x{H} -> cropped {Wc}x{Hc}; focal {fx:.3f} px; "
          f"principal point is {off[0]:+.2f}, {off[1]:+.2f} px from the centre after cropping")

    # ordered export list. Left images take the held-out slots; a right image never does.
    items = []  # (side, frame id)
    for i in ids:
        k = len(items)
        if k % HOLD == PHASE:
            items.append(("l", i))  # test image; its right twin is not exported
            continue
        items.append(("l", i))
        if a.use_right and "r" in by_id[i] and (k + 1) % HOLD != PHASE:
            items.append(("r", i))

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for old in (out / "images").glob("*.png"):
        old.unlink()
    img_lines, split = [], ["name\trole\tframe\tside\tTwc_3x4_row_major_opencv"]
    n_test = 0
    for k, (side, i) in enumerate(items):
        f = by_id[i][side]
        name = f"{k:05d}_{side}_{i:06d}.png"
        img = cv2.imread(str(data / f["file_path"]))
        if s != 1.0:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "images" / name), img[y0:y0 + Hc, x0:x0 + Wc])
        Twc = np.array(f["transform_matrix"]) @ FLIP
        Tcw = np.linalg.inv(Twc)
        q, t = rot_to_quat(Tcw[:3, :3]), Tcw[:3, 3]
        img_lines.append(f"{k + 1} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} "
                         f"{t[0]:.9g} {t[1]:.9g} {t[2]:.9g} 1 {name}\n\n")
        role = "test" if k % HOLD == PHASE else "train"
        n_test += role == "test"
        split.append(f"{name}\t{role}\t{i}\t{side}\t" + " ".join(f"{v:.9g}" for v in Twc[:3].ravel()))
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(items)} images")
    (out / "sparse/0/cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {Wc} {Hc} {fx:.6f} {fy:.6f} {Wc / 2:.3f} {Hc / 2:.3f}\n")
    (out / "sparse/0/images.txt").write_text(
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "".join(img_lines))
    (out / "split.tsv").write_text("\n".join(split) + "\n")

    # starting point cloud: the dense stereo cloud, thinned
    src = data / tf.get("ply_file_path", "dense_pc.ply")
    xyz, rgb = read_ply_xyzrgb(src)
    key = np.floor(xyz / a.voxel).astype(np.int64)
    key = (key[:, 0] + 100000) * 10**12 + (key[:, 1] + 100000) * 10**6 + (key[:, 2] + 100000)
    _, first = np.unique(key, return_index=True)
    if len(first) > a.max_points:
        first = np.random.default_rng(a.seed).choice(first, a.max_points, replace=False)
    write_ply_with_normals(out / "sparse/0/points3D.ply", xyz[first], rgb[first])

    n_r = sum(1 for sd, _ in items if sd == "r")
    print(f"wrote {out}: {len(items)} images ({len(items) - n_r} left, {n_r} right), "
          f"{len(items) - n_test} train / {n_test} test, {len(first)} starting points (from {len(xyz)} in {src.name})")
    (out / "HOW_TO_RUN.txt").write_text(
        f"Made by export_dataset_for_gsplatloc.py from {data}\n"
        f"{len(items)} images, {Wc}x{Hc}, focal {fx:.3f} px, {len(items) - n_test} train / {n_test} test, metric poses.\n\n"
        "Inside the GSplatLoc repo, with this folder at data/<name>:\n\n"
        "  python train.py -s data/<name> -m output/<name> --eval --iterations 15000 \\\n"
        "      --save_iterations 15000 --test_iterations 15000 --data_device cpu\n"
        "  python loc_inference.py -m output/<name> --data_device cuda\n\n"
        "--data_device cuda at localization is needed because training used cpu (the script compares a GPU\n"
        "render with the query image). README_SETUP_AND_RUN.md describes a variant that also saves per-query results.\n"
        "--eval is required: without it there are no test images and nothing gets localized.\n"
        "--iterations must be given: the code's built-in default is 2000, not the 30000 the README says.\n"
        "Results print at the end and are saved in output/<name>/error_logs/.\n")


if __name__ == "__main__":
    main()
