#!/usr/bin/env python3
"""Add dense stereo depth to a nerfstudio dataset made by map2nerfstudio.py.

For every left frame in transforms.json (images/l_XXXXXX.png) this computes an SGBM disparity
from the rectified pair in the KITTI sequence, converts it to metric depth (bf / d), and:
  * writes depth/<name>.png (uint16, millimetres) for every frame, left and right (nerfstudio
    requires depth on all frames or none), and sets depth_file_path,
  * back-projects valid pixels into world space and merges them into a voxel-downsampled,
    colour-averaged point cloud dense_pc.ply, which replaces ply_file_path for Gaussian init.

    uv pip install opencv-python-headless numpy
    stereo_densify.py --data ns_data/seq02 --seq kitti/sequences/02 [--max-depth 6] [--voxel 0.02] [--scale 0.5]

transforms.json is backed up to transforms_sparse.json on first run.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np


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


class VoxelAccumulator:
    """Running per-voxel mean of position and colour."""

    def __init__(self, voxel):
        self.v = voxel
        self.keys = np.empty(0, np.int64)
        self.sum = np.empty((0, 6), np.float64)
        self.cnt = np.empty(0, np.int64)
        self.pend = []

    @staticmethod
    def _key(ijk):
        ijk = ijk + (1 << 20)
        return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]

    def add(self, xyz, rgb):
        ijk = np.floor(xyz / self.v).astype(np.int64)
        self.pend.append((self._key(ijk), np.hstack([xyz, rgb]).astype(np.float64)))
        if sum(len(k) for k, _ in self.pend) > 20_000_000:
            self.flush()

    def flush(self):
        if not self.pend:
            return
        keys = np.concatenate([self.keys] + [k for k, _ in self.pend])
        vals = np.concatenate([self.sum] + [v for _, v in self.pend])
        cnts = np.concatenate([self.cnt] + [np.ones(len(k), np.int64) for k, _ in self.pend])
        self.pend = []
        uk, inv = np.unique(keys, return_inverse=True)
        s = np.zeros((len(uk), 6))
        np.add.at(s, inv, vals)
        c = np.bincount(inv, weights=cnts, minlength=len(uk)).astype(np.int64)
        self.keys, self.sum, self.cnt = uk, s, c

    def result(self, min_count):
        self.flush()
        m = self.cnt >= min_count
        mean = self.sum[m] / self.cnt[m, None]
        return mean[:, :3].astype(np.float32), np.clip(mean[:, 3:], 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="nerfstudio dataset dir (transforms.json)")
    ap.add_argument("--seq", required=True, help="KITTI sequence dir with image_0, image_1, camera.txt")
    ap.add_argument("--scale", type=float, default=0.5, help="SGBM working resolution scale (speed vs detail)")
    ap.add_argument("--min-depth", type=float, default=0.25)
    ap.add_argument("--max-depth", type=float, default=6.0, help="drop depths beyond this [m] (stereo error grows ~z^2)")
    ap.add_argument("--voxel", type=float, default=0.02, help="point cloud voxel size [m]")
    ap.add_argument("--min-views", type=int, default=2, help="keep voxels hit by at least this many pixels")
    ap.add_argument("--stride", type=int, default=2, help="use every Nth valid pixel for the point cloud")
    ap.add_argument("--no-depth-files", action="store_true", help="only build the point cloud")
    a = ap.parse_args()

    data, seq = Path(a.data), Path(a.seq)
    tf_path = data / "transforms.json"
    backup = data / "transforms_sparse.json"
    if not backup.exists():
        shutil.copy2(tf_path, backup)
    tf = json.loads(backup.read_text())
    cam = read_camera_txt(seq / "camera.txt")
    fx, fy, cx, cy, bf = cam["fx"], cam["fy"], cam["cx"], cam["cy"], cam["bf"]
    W, H = int(cam["width"]), int(cam["height"])

    s = a.scale
    w, h = int(round(W * s)), int(round(H * s))
    max_disp = bf * s / a.min_depth
    num_disp = int(np.ceil(max_disp / 16.0)) * 16
    bs = 5
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=bs,
                                 P1=8 * 3 * bs * bs, P2=32 * 3 * bs * bs, disp12MaxDiff=1,
                                 uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    print(f"{len(tf['frames'])} frames, SGBM at {w}x{h}, numDisparities {num_disp}, "
          f"depth {a.min_depth}-{a.max_depth} m, voxel {a.voxel} m")

    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    rays = np.stack([(uu - cx) / fx, (vv - cy) / fy], -1)
    acc = VoxelAccumulator(a.voxel)
    (data / "depth").mkdir(exist_ok=True)
    t0, n_left, fills = time.time(), 0, []

    def disparity(A, B):
        ga = cv2.cvtColor(cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(B, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        d = sgbm.compute(ga, gb).astype(np.float32) / 16.0
        return cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) / s  # full-res disparity

    for fr in tf["frames"]:
        name = Path(fr["file_path"]).name
        side, idx = name[:2], name[2:8]
        if side not in ("l_", "r_"):
            sys.exit(f"unexpected image name {name}; expected l_XXXXXX.png / r_XXXXXX.png")
        L = cv2.imread(str(seq / "image_0" / f"{idx}.png"))
        R = cv2.imread(str(seq / "image_1" / f"{idx}.png"))
        if L is None or R is None:
            sys.exit(f"missing image pair {idx} in {seq}")
        if side == "l_":
            disp = disparity(L, R)
            img = L
        else:  # right-view disparity: match on horizontally mirrored images, mirror back
            disp = disparity(R[:, ::-1].copy(), L[:, ::-1].copy())[:, ::-1]
            img = R
        depth = np.zeros_like(disp)
        ok = disp > 0
        depth[ok] = bf / disp[ok]
        depth[(depth < a.min_depth) | (depth > a.max_depth)] = 0
        valid = depth > 0
        fills.append(valid.mean())

        if not a.no_depth_files:
            dpath = f"depth/{name}"
            cv2.imwrite(str(data / dpath), np.round(depth * 1000).astype(np.uint16))
            fr["depth_file_path"] = dpath

        if side == "l_":  # point cloud from left views only (right views add near-duplicates)
            Twc = np.array(fr["transform_matrix"]) @ flip  # back to OpenCV camera axes
            vmask = valid.copy()
            if a.stride > 1:
                sub = np.zeros_like(vmask)
                sub[::a.stride, ::a.stride] = True
                vmask &= sub
            z = depth[vmask]
            pc = np.column_stack([rays[vmask] * z[:, None], z])
            pw = pc @ Twc[:3, :3].T + Twc[:3, 3]
            acc.add(pw, img[vmask][:, ::-1])
            n_left += 1
        n_done = len(fills)
        if n_done % 20 == 0:
            print(f"\r  {n_done} frames, mean valid depth {np.mean(fills) * 100:.0f}%, "
                  f"{n_done / (time.time() - t0):.1f} fr/s", end="", flush=True)
    print()
    if not n_left:
        sys.exit("no left frames (images/l_*.png) in transforms.json")

    xyz, rgb = acc.result(a.min_views)
    write_ply(data / "dense_pc.ply", xyz, rgb)
    tf["ply_file_path"] = "dense_pc.ply"
    if not a.no_depth_files:
        tf["depth_unit_scale_factor"] = 1e-3
    tf_path.write_text(json.dumps(tf, indent=1))
    print(f"valid depth per frame: median {np.median(fills) * 100:.0f}%, min {np.min(fills) * 100:.0f}%")
    print(f"dense_pc.ply: {len(xyz)} points (voxel {a.voxel} m, >= {a.min_views} hits), "
          f"bbox {np.ptp(xyz, axis=0).round(2) if len(xyz) else '-'} m")
    print(f"updated {tf_path} (original kept as {backup.name})")


if __name__ == "__main__":
    main()
