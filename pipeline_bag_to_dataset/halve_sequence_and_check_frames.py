#!/usr/bin/env python3
"""Write a half-resolution copy of a KITTI-layout sequence and, in the same pass, measure every
stereo pair so corrupted frames can be found (bag2kitti.py does not detect them).

Per frame, on the half-size images: stereo ORB median |dy| (rows should agree to about 1 px; the
rolled/seamed frames of the Oct 4 bag gave about 15 px), number of stereo matches, Laplacian
variance of the left image (sharpness), and how much the left/right thumbnails differ from their
neighbours compared with how much the neighbours differ from each other.

usage: uv run python halve_sequence_and_check_frames.py kitti/sequences/07 kitti/sequences/08
Writes <dst>/image_0, image_1 (real files, same frame numbers), image_2/3 symlinks, camera.txt,
calib.txt, times*.txt, odom/imu symlinks, frame_quality.csv, corrupted_frames.txt, log.txt."""
import os, sys
from multiprocessing import Pool

import cv2
import numpy as np

src, dst = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
cam = {k.strip(): float(v) for k, v in (l.split(":") for l in open(f"{src}/camera.txt") if ":" in l)}
W, H = int(cam["width"]) // 2, int(cam["height"]) // 2
names = sorted(os.listdir(f"{src}/image_0"))


def work(name):
    cv2.setNumThreads(1)
    out = []
    grays = []
    for side in ("image_0", "image_1"):
        img = cv2.imread(f"{src}/{side}/{name}")
        if img is None:
            return name, 99.0, 0, 0.0, None, None
        small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{dst}/{side}/{name}", small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        grays.append(g)
        out.append(cv2.resize(g, (48, 27), interpolation=cv2.INTER_AREA).astype(np.float32))
    orb = cv2.ORB_create(800)
    kl, dl = orb.detectAndCompute(grays[0], None)
    kr, dr = orb.detectAndCompute(grays[1], None)
    med, n = 99.0, 0
    if dl is not None and dr is not None and len(kl) > 10 and len(kr) > 10:
        ms = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dl, dr)
        d = np.array([(kl[m.queryIdx].pt[0] - kr[m.trainIdx].pt[0], kl[m.queryIdx].pt[1] - kr[m.trainIdx].pt[1]) for m in ms])
        if len(d):
            d = d[(d[:, 0] > -2) & (d[:, 0] < 0.3 * W)]  # plausible disparity only
            n = len(d)
            if n >= 10:
                med = float(np.median(np.abs(d[:, 1])))
    lap = float(cv2.Laplacian(grays[0], cv2.CV_64F).var())
    return name, med, n, lap, out[0], out[1]


if __name__ == "__main__":
    for side in ("image_0", "image_1"):
        os.makedirs(f"{dst}/{side}", exist_ok=True)
    with Pool(18) as pool:
        res = []
        for i, r in enumerate(pool.imap(work, names, chunksize=8)):
            res.append(r)
            if i % 500 == 0:
                print(f"{i}/{len(names)}", flush=True)

    def neighbour_ratio(th):
        """mean |frame - neighbours| over mean |prev - next|: large when one frame is the odd one out."""
        r = np.ones(len(th))
        for i in range(1, len(th) - 1):
            if th[i] is None or th[i - 1] is None or th[i + 1] is None:
                continue
            a = np.abs(th[i] - th[i - 1]).mean()
            b = np.abs(th[i] - th[i + 1]).mean()
            c = np.abs(th[i - 1] - th[i + 1]).mean()
            r[i] = min(a, b) / (c + 1.0)
        return r

    rl = neighbour_ratio([r[4] for r in res])
    rr = neighbour_ratio([r[5] for r in res])
    with open(f"{dst}/frame_quality.csv", "w") as f:
        f.write("frame,stereo_dy_px,stereo_matches,laplacian_var,left_odd_ratio,right_odd_ratio\n")
        for i, r in enumerate(res):
            f.write(f"{i},{r[1]:.3f},{r[2]},{r[3]:.2f},{rl[i]:.3f},{rr[i]:.3f}\n")

    # half-size intrinsics for 2x2 area averaging: pixel centres move by a quarter pixel
    fx, fy = cam["fx"] / 2, cam["fy"] / 2
    cx, cy = cam["cx"] / 2 - 0.25, cam["cy"] / 2 - 0.25
    bf = cam["bf"] / 2
    with open(f"{dst}/camera.txt", "w") as f:
        f.write(f"fx: {fx:.6f}\nfy: {fy:.6f}\ncx: {cx:.6f}\ncy: {cy:.6f}\nbf: {bf:.6f}\n"
                f"width: {W}\nheight: {H}\nbaseline_m: {cam['baseline_m']:.6f}\n")
    with open(f"{dst}/calib.txt", "w") as f:
        f.write(f"P0: {fx:.12e} 0 {cx:.12e} 0 0 {fy:.12e} {cy:.12e} 0 0 0 1 0\n")
        f.write(f"P1: {fx:.12e} 0 {cx:.12e} {-bf:.12e} 0 {fy:.12e} {cy:.12e} 0 0 0 1 0\n")
    for name in ("times.txt", "times_ns.txt"):
        open(f"{dst}/{name}", "w").write(open(f"{src}/{name}").read())
    rel = os.path.relpath(src, dst)
    for name, target in (("image_2", "image_0"), ("image_3", "image_1"),
                         ("odom.csv", f"{rel}/odom.csv"), ("imu.csv", f"{rel}/imu.csv")):
        if not os.path.lexists(f"{dst}/{name}"):
            os.symlink(target, f"{dst}/{name}")
    open(f"{dst}/log.txt", "w").write(
        f"half-resolution copy of {src} ({W}x{H}, INTER_AREA), same frame numbers\n"
        f"fx {fx:.4f} fy {fy:.4f} cx {cx:.4f} cy {cy:.4f} bf {bf:.4f}\n")
    dy = np.array([r[1] for r in res])
    print(f"done: {len(res)} frames, {W}x{H}; stereo dy median {np.median(dy):.2f} p99 {np.percentile(dy, 99):.2f}")
