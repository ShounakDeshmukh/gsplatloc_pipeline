#!/usr/bin/env python3
"""Merge two nerfstudio datasets built from two pySLAM maps of one out-and-back recording when
only ONE in-place spin separates the legs (merge_legs.py needs two).

Start: the spin's rotation centre is the same 3D point in both maps, and the heading change across
the untracked part of the spin comes from the gyro (scale and sign calibrated against the SLAM
rotation on the tracked parts of the same spin; wheel odom is printed for comparison).
Refine: one rotation about that centre (3 DOF) that makes the stereo point clouds of the two legs
agree (rotation-only ICP; walls fix yaw, the floor fixes tilt). The centre itself is not moved.

    merge_legs_one_spin.py --a ns_data/seq10 --a-traj <A>/trajectory_final.txt \
        --b ns_data/seq11 --b-traj <B>/trajectory_final.txt --b-offset 5115 \
        --seq kitti/sequences/09 --out ns_data/seq09_merged

--seq is the whole recording; leg A is its first frames, leg B's frame i is frame i + --b-offset.
"""
import argparse
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from merge_legs import Odom, load_traj


def rodrigues(v):
    return cv2.Rodrigues(np.asarray(v, float).reshape(3, 1))[0]


def signed_angle(R, axis):
    """Rotation angle of R [rad], signed by the given axis."""
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return np.arctan2(np.linalg.norm(w) / 2, (np.trace(R) - 1) / 2) * (1 if w @ axis >= 0 else -1)


def cloud_job(args):
    seq, idx, T, cam, max_depth = args
    cv2.setNumThreads(1)
    L = cv2.imread(f"{seq}/image_0/{idx:06d}.png", cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(f"{seq}/image_1/{idx:06d}.png", cv2.IMREAD_GRAYSCALE)
    sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5, P1=8 * 25, P2=32 * 25,
                               uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                               mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = sg.compute(L, R).astype(np.float32) / 16
    v, u = np.mgrid[4:d.shape[0]:8, 4:d.shape[1]:8]
    dd = d[v, u]
    ok = dd > cam["bf"] / max_depth
    z = cam["bf"] / dd[ok]
    ok2 = z > 0.5
    z = z[ok2]
    x = (u[ok][ok2] - cam["cx"]) * z / cam["fx"]
    y = (v[ok][ok2] - cam["cy"]) * z / cam["fy"]
    P = np.stack([x, y, z], 1)
    return (P @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def leg_cloud(seq, poses, cam, max_depth, voxel, step_m=0.25, step_deg=10):
    ids, last = [], None
    for i in sorted(poses):
        T = poses[i]
        if last is None or np.linalg.norm(T[:3, 3] - last[:3, 3]) > step_m or \
                np.degrees(np.arccos(np.clip((np.trace(last[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1))) > step_deg:
            ids.append(i)
            last = T
    with Pool(16) as pool:
        P = np.vstack(pool.map(cloud_job, [(seq, i, poses[i], cam, max_depth) for i in ids], chunksize=4))
    key = np.floor(P / voxel).astype(np.int64)
    _, first, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return P[first[cnt >= 2]].astype(np.float64), len(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--a", "--a-traj", "--b", "--b-traj", "--seq"):
        ap.add_argument(k, required=True)
    ap.add_argument("--b-offset", type=int, required=True)
    ap.add_argument("--out")
    ap.add_argument("--still-tol", type=float, default=0.03)
    ap.add_argument("--max-depth", type=float, default=4.0)
    ap.add_argument("--voxel", type=float, default=0.04)
    ap.add_argument("--no-refine", action="store_true")
    a = ap.parse_args()

    seq = Path(a.seq)
    cam = {k.strip(): float(v) for k, v in (ln.split(":") for ln in open(seq / "camera.txt") if ":" in ln)}
    times = np.atleast_1d(np.loadtxt(seq / "times.txt"))
    tns = np.loadtxt(seq / "times_ns.txt", dtype=np.int64).astype(np.float64)
    od = Odom(seq / "odom.csv", tns)
    pa = load_traj(a.a_traj, times)
    pb = load_traj(a.b_traj, times, a.b_offset)
    fa, fb = max(pa), min(pb)
    print(f"A: {len(pa)} tracked frames ({min(pa)}..{fa}), B: {len(pb)} ({fb}..{max(pb)})")

    # the in-place spin around the gap
    s0 = fa
    while s0 > 0 and np.hypot(od.x[s0 - 1] - od.x[fa], od.y[s0 - 1] - od.y[fa]) < a.still_tol:
        s0 -= 1
    s1 = fb
    while s1 + 1 < len(times) and np.hypot(od.x[s1 + 1] - od.x[fa], od.y[s1 + 1] - od.y[fa]) < a.still_tol:
        s1 += 1
    ia = [i for i in range(s0, fa + 1) if i in pa]
    ib = [i for i in range(fb, s1 + 1) if i in pb]
    print(f"spin: frames {s0}..{s1}, odom turn {np.degrees(od.yaw[s1] - od.yaw[s0]):.1f} deg; "
          f"tracked in A {len(ia)} frames, in B {len(ib)} frames; untracked gap {fa}..{fb}")

    # spin axis in the camera frame (up), from both tracked parts
    acc = np.zeros(3)
    for poses, ids in ((pa, ia), (pb, ib)):
        for i, j in zip(ids, ids[5:]):
            R = poses[i][:3, :3].T @ poses[j][:3, :3]
            w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            if np.linalg.norm(w) > 2 * np.sin(np.radians(2)):
                acc += w if w[1] < 0 else -w
    axis = acc / np.linalg.norm(acc)
    # lever arm camera -> rotation centre: p_i + R_i l constant within each tracked part
    rows, rhs = [], []
    for s, (poses, ids) in enumerate(((pa, ia), (pb, ib))):
        for i in ids:
            A = np.zeros((3, 9))
            A[:, :3] = poses[i][:3, :3]
            A[:, 3 + 3 * s:6 + 3 * s] = -np.eye(3)
            rows.append(A)
            rhs.append(-poses[i][:3, 3])
    A, y = np.vstack(rows), np.concatenate(rhs)
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    rms = float(np.sqrt(((A @ sol - y).reshape(-1, 3) ** 2).sum(1).mean()))
    l = sol[:3] - axis * (sol[:3] @ axis)
    ca = np.mean([pa[i][:3, 3] + pa[i][:3, :3] @ l for i in ia], 0)
    cb = np.mean([pb[i][:3, 3] + pb[i][:3, :3] @ l for i in ib], 0)
    print(f"spin axis in camera frame {axis.round(4)} (tilt from -y {np.degrees(np.arccos(-axis[1])):.2f} deg); "
          f"lever arm {l.round(4)} m (|l| {np.linalg.norm(l) * 100:.1f} cm), fit rms {rms * 100:.2f} cm")

    # heading across the gap: gyro about the vertical IMU axis, calibrated on the tracked parts
    imu = np.loadtxt(seq / "imu.csv", delimiter=",", skiprows=1)
    g_axis = int(np.argmax(np.abs(imu[:, 1:4].mean(0))))  # the accelerometer axis that carries gravity
    still = np.interp(imu[:, 0], np.loadtxt(seq / "odom.csv", delimiter=",", skiprows=1)[:, 0],
                      np.abs(np.loadtxt(seq / "odom.csv", delimiter=",", skiprows=1)[:, 8:10]).sum(1)) < 1e-6
    bias = float(np.median(imu[still, 4 + g_axis]))
    gyaw = np.interp(tns, imu[:, 0], np.cumsum((imu[:, 4 + g_axis] - bias) * np.gradient(imu[:, 0]) / 1e9))
    sl, gy, oy = [], [], []
    for poses, ids in ((pa, ia), (pb, ib)):
        for i in ids:
            sl.append(signed_angle(poses[ids[0]][:3, :3].T @ poses[i][:3, :3], axis))
            gy.append(gyaw[i] - gyaw[ids[0]])
            oy.append(od.yaw[i] - od.yaw[ids[0]])
    sl, gy, oy = np.array(sl), np.array(gy), np.array(oy)
    kg, ko = (sl @ gy) / (gy @ gy), (sl @ oy) / (oy @ oy)
    print(f"gyro axis {'xyz'[g_axis]}, bias {np.degrees(bias):.3f} deg/s from {still.sum()} still samples")
    print(f"SLAM rotation on tracked spin parts = {kg:.4f} x gyro (rms {np.degrees(np.sqrt(np.mean((sl - kg * gy) ** 2))):.2f} deg), "
          f"= {ko:.4f} x odom (rms {np.degrees(np.sqrt(np.mean((sl - ko * oy) ** 2))):.2f} deg)")
    th_g, th_o = kg * (gyaw[fb] - gyaw[fa]), ko * (od.yaw[fb] - od.yaw[fa])
    print(f"heading change across the gap {fa}->{fb}: gyro {np.degrees(th_g):.2f} deg, odom {np.degrees(th_o):.2f} deg")

    R_ab = pa[fa][:3, :3] @ rodrigues(axis * th_g) @ pb[fb][:3, :3].T
    t_ab = ca - R_ab @ cb
    R0 = R_ab.copy()

    if not a.no_refine:
        Pa, na = leg_cloud(str(seq), pa, cam, a.max_depth, a.voxel)
        Pb, nb = leg_cloud(str(seq), pb, cam, a.max_depth, a.voxel)
        print(f"stereo clouds: A {len(Pa)} points from {na} frames, B {len(Pb)} from {nb} frames")
        tree = cKDTree(Pa)

        def stats(R, t, tag):
            d, _ = tree.query(Pb @ R.T + t, distance_upper_bound=0.5)
            ok = np.isfinite(d)
            print(f"  {tag}: {ok.mean() * 100:.0f}% of B within 0.5 m of A; of those median {np.median(d[ok]) * 100:.1f} cm, "
                  f"p90 {np.percentile(d[ok], 90) * 100:.1f} cm; within 5 cm {np.mean(d[ok] < 0.05) * 100:.0f}%")

        stats(R_ab, t_ab, "before refinement")
        for it in range(80):
            thr = max(0.5 * 0.94 ** it, 0.06)
            Q = Pb @ R_ab.T + t_ab
            d, j = tree.query(Q, distance_upper_bound=thr)
            ok = np.isfinite(d)
            X, Y = Q[ok] - ca, Pa[j[ok]] - ca
            U, _, Vt = np.linalg.svd(X.T @ Y)
            Rd = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
            R_ab = Rd @ R_ab
            t_ab = ca - R_ab @ cb
        stats(R_ab, t_ab, "after refinement ")
        Rc = R_ab @ R0.T
        up_a = pa[fa][:3, :3] @ axis
        yaw_c = signed_angle(Rc, up_a)
        rv = cv2.Rodrigues(Rc)[0].ravel()
        print(f"refinement rotated B by {np.degrees(np.linalg.norm(rv)):.2f} deg in total: "
              f"{np.degrees(rv @ up_a):.2f} deg about the vertical (heading), "
              f"{np.degrees(np.linalg.norm(rv - up_a * (rv @ up_a))):.2f} deg of tilt")
        Q = Pb @ R_ab.T + t_ab
        d, j = tree.query(Q, distance_upper_bound=0.06)
        ok = np.isfinite(d)
        print(f"mean leftover offset of matched points (not applied): {((Pa[j[ok]] - Q[ok]).mean(0) * 100).round(1)} cm")
        # agreement by distance from the tie point: a heading error would show as growth with distance
        d, j = tree.query(Q, distance_upper_bound=0.3)
        r = np.linalg.norm(Q - ca, axis=1)
        up = pa[fa][:3, :3] @ axis
        for r0 in range(0, int(r.max()) + 1, 6):
            m = (r >= r0) & (r < r0 + 6) & np.isfinite(d)
            if m.sum() > 200:
                off = Pa[j[m]] - Q[m]
                print(f"  {r0:2d}..{r0 + 6:2d} m from the spin: {m.sum():6d} matched points, median {np.median(d[m]) * 100:.1f} cm, "
                      f"within 5 cm {np.mean(d[m] < 0.05) * 100:.0f}%, mean vertical offset {(off @ up).mean() * 100:+.1f} cm")

    T_ab = np.eye(4)
    T_ab[:3, :3], T_ab[:3, 3] = R_ab, t_ab
    up_a = pa[fa][:3, :3] @ axis
    for name, poses, T0 in (("A", pa, np.eye(4)), ("B", pb, T_ab)):
        h = np.array([(T0 @ T)[:3, 3] @ up_a for T in poses.values()])
        print(f"camera height along A's vertical, leg {name}: median {np.median(h) * 100:.1f} cm, "
              f"p5..p95 {np.percentile(h, 5) * 100:.1f}..{np.percentile(h, 95) * 100:.1f} cm")
    e = (T_ab @ pb[max(pb)])[:3, 3] - pa[min(pa)][:3, 3]
    print(f"end of leg B relative to start of leg A: {np.linalg.norm(e):.2f} m "
          f"(odom, which drifts: {np.hypot(od.x[max(pb)] - od.x[min(pa)], od.y[max(pb)] - od.y[min(pa)]):.2f} m)")
    if not a.out:
        return

    # ---- write merged dataset (same layout as merge_legs.py)
    A, B, out = Path(a.a), Path(a.b), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    ta = json.loads((A / ("transforms_sparse.json" if (A / "transforms_sparse.json").exists() else "transforms.json")).read_text())
    tb = json.loads((B / ("transforms_sparse.json" if (B / "transforms_sparse.json").exists() else "transforms.json")).read_text())
    frames = []
    for old in (out / "images").glob("*.png"):
        old.unlink()
    for f in ta["frames"]:
        shutil.copy2(A / f["file_path"], out / f["file_path"])
        frames.append({"file_path": f["file_path"], "transform_matrix": f["transform_matrix"]})
    have = {f["file_path"] for f in frames}
    for f in tb["frames"]:
        side, idx = Path(f["file_path"]).stem.split("_")
        name = f"images/{side}_{int(idx) + a.b_offset:06d}.png"
        if name in have:
            continue
        shutil.copy2(B / f["file_path"], out / name)
        frames.append({"file_path": name, "transform_matrix": (T_ab @ np.array(f["transform_matrix"])).tolist()})
    frames.sort(key=lambda f: (Path(f["file_path"]).stem.split("_")[1], f["file_path"]))

    def read_ply(p):
        raw = p.read_bytes()
        head, body = raw.split(b"end_header\n", 1)
        n = int([ln for ln in head.split(b"\n") if ln.startswith(b"element vertex")][0].split()[-1])
        dt = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
        if b"binary_little_endian" not in head or len(body) != n * dt.itemsize:
            sys.exit(f"{p}: unexpected ply layout")
        d = np.frombuffer(body, dt)
        return head, d["xyz"].astype(np.float64), d["rgb"]

    head, xa, cola = read_ply(A / "sparse_pc.ply")
    _, xb, colb = read_ply(B / "sparse_pc.ply")
    xyz = np.vstack([xa, xb @ R_ab.T + t_ab]).astype("<f4")
    d = np.zeros(len(xyz), np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)]))
    d["xyz"], d["rgb"] = xyz, np.vstack([cola, colb])
    head = b"\n".join(ln if not ln.startswith(b"element vertex") else b"element vertex %d" % len(xyz)
                      for ln in head.split(b"\n"))
    (out / "sparse_pc.ply").write_bytes(head + b"end_header\n" + d.tobytes())

    rows = (A / "keyframes.tsv").read_text().strip().split("\n")
    seen = {r.split("\t")[0] for r in rows[1:]}
    for r in (B / "keyframes.tsv").read_text().strip().split("\n")[1:]:
        i, ts, P = r.split("\t")
        i = str(int(i) + a.b_offset)
        if i in seen:
            continue
        T = np.eye(4)
        T[:3] = np.array(P.split(), float).reshape(3, 4)
        rows.append(f"{i}\t{times[int(i)]:.6f}\t" + " ".join(f"{v:.9g}" for v in (T_ab @ T)[:3].ravel()))
    rows = [rows[0]] + sorted(rows[1:], key=lambda r: int(r.split("\t")[0]))
    (out / "keyframes.tsv").write_text("\n".join(rows) + "\n")
    ta["frames"] = frames
    ta["ply_file_path"] = "sparse_pc.ply"
    (out / "transforms.json").write_text(json.dumps(ta, indent=1))
    np.savetxt(out / "T_a_b.txt", T_ab, fmt="%.9f")
    print(f"wrote {out}/transforms.json ({len(frames)} frames), sparse_pc.ply ({len(xyz)} points), T_a_b.txt")


if __name__ == "__main__":
    main()
