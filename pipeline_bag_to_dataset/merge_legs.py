#!/usr/bin/env python3
"""Merge two nerfstudio datasets built from two pySLAM maps of the same bag (leg A, leg B) when
the maps share no tracked frames because tracking was lost while the robot spun in place.

The robot's rotation centre does not move during an in-place spin, so each spin that separates
the legs gives one 3D point known in both maps. Two such points plus the common vertical
(the spin axis, fixed in the camera frame) fix the rigid transform B -> A. No scale is fitted:
both maps are metric from the same stereo rig.

    merge_legs.py --a ns_data/seq04 --a-traj <A>/trajectory_final.txt \
                  --b ns_data/seq05 --b-traj <B>/trajectory_final.txt --b-offset 1676 \
                  --seq kitti/sequences/04 --out ns_data/seq04_merged

--seq is the sequence that leg A used; leg B's frame i is frame i + --b-offset of it.
Output images are named by --seq frame index, so stereo_densify.py runs with the same --seq.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def quat_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_traj(path, times, offset=0):
    """trajectory_final.txt -> {frame idx in --seq numbering: Twc}. Identity rows (lost) are skipped."""
    out = {}
    local = times[offset:] - times[offset]
    for row in np.atleast_2d(np.loadtxt(path)):
        i = int(np.argmin(np.abs(local - row[0])))
        if abs(local[i] - row[0]) > 1e-3:
            continue
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = quat_to_R(row[4:8]), row[1:4]
        if i > 1 and np.allclose(T, np.eye(4), atol=1e-9):
            continue
        out[i + offset] = T
    return out


class Odom:
    def __init__(self, path, tns):
        o = np.loadtxt(path, delimiter=",", skiprows=1)
        self.x = np.interp(tns, o[:, 0], o[:, 1])
        self.y = np.interp(tns, o[:, 0], o[:, 2])
        self.yaw = np.interp(tns, o[:, 0], np.unwrap(2 * np.arctan2(o[:, 6], o[:, 7])))

    def still_runs(self, tol=0.02, min_turn=np.radians(20)):
        """Maximal frame ranges [i0, i1] where the robot stays within tol m and turns >= min_turn."""
        runs, i0, n = [], 0, len(self.x)
        while i0 < n:
            i1 = i0
            while i1 + 1 < n and np.hypot(self.x[i1 + 1] - self.x[i0], self.y[i1 + 1] - self.y[i0]) < tol:
                i1 += 1
            if np.ptp(self.yaw[i0:i1 + 1]) >= min_turn:
                runs.append((i0, i1))
            i0 = i1 + 1
        return runs


def spin_axis(poses, runs):
    """Rotation axis of the in-place spins, expressed in the camera frame (unit, pointing up)."""
    acc = np.zeros(3)
    for i0, i1 in runs:
        ids = [i for i in range(i0, i1 + 1) if i in poses]
        for a, b in zip(ids, ids[5:]):
            R = poses[a][:3, :3].T @ poses[b][:3, :3]
            w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            if np.linalg.norm(w) > 2 * np.sin(np.radians(3)):
                acc += w if w[1] < 0 else -w  # camera y points down: keep the half-space y < 0 (up)
    if not np.linalg.norm(acc):
        sys.exit("no usable spin found to estimate the vertical axis")
    return acc / np.linalg.norm(acc)


def lever_arm(maps, runs, min_turn=np.radians(20)):
    """Camera -> rotation-centre offset l (camera frame): p_i + R_i l is constant during a spin."""
    rows, rhs, n_seg, segs = [], [], 0, []
    for poses in maps:
        for i0, i1 in runs:
            ids = [i for i in range(i0, i1 + 1) if i in poses]
            if len(ids) < 8:
                continue
            R0 = poses[ids[0]][:3, :3]
            turn = max(np.arccos(np.clip((np.trace(R0.T @ poses[i][:3, :3]) - 1) / 2, -1, 1)) for i in ids)
            if turn < min_turn:
                continue
            segs.append((poses, ids))
    for s, (poses, ids) in enumerate(segs):
        for i in ids:
            A = np.zeros((3, 3 + 3 * len(segs)))
            A[:, :3] = poses[i][:3, :3]
            A[:, 3 + 3 * s:6 + 3 * s] = -np.eye(3)
            rows.append(A)
            rhs.append(-poses[i][:3, 3])
    if not rows:
        sys.exit("no spin with enough tracked frames to estimate the camera lever arm")
    A, y = np.vstack(rows), np.concatenate(rhs)
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    res = (A @ sol - y).reshape(-1, 3)
    return sol[:3], float(np.sqrt((res ** 2).sum(1).mean())), len(segs), len(rows)


def centre(poses, ids, l):
    c = np.array([poses[i][:3, 3] + poses[i][:3, :3] @ l for i in ids])
    return c.mean(0), float(np.linalg.norm(c - c.mean(0), axis=1).max()) if len(c) > 1 else 0.0


def tie(pa, pb, od, run, l, name):
    """One tie point: rotation centre during still-run `run`, seen in map A and map B."""
    i0, i1 = run
    ia = [i for i in range(i0, i1 + 1) if i in pa]
    ib = [i for i in range(i0, i1 + 1) if i in pb]
    if len(ia) < 3 or len(ib) < 3:
        return None
    ca, sa = centre(pa, ia, l)
    cb, sb = centre(pb, ib, l)
    print(f"tie {name}: frames {i0}..{i1}, A tracked {len(ia)} (spread {sa * 100:.1f} cm), "
          f"B tracked {len(ib)} (spread {sb * 100:.1f} cm), turn {np.degrees(np.ptp(od.yaw[i0:i1 + 1])):.0f} deg")
    return ca, cb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-traj", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-traj", required=True)
    ap.add_argument("--b-offset", type=int, required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out")
    ap.add_argument("--max-height-err", type=float, default=0.04,
                    help="drop frames whose camera is this far [m] off the common camera plane (bad pose)")
    ap.add_argument("--still-tol", type=float, default=0.02, help="odom drift [m] allowed within one in-place spin")
    a = ap.parse_args()

    seq = Path(a.seq)
    times = np.atleast_1d(np.loadtxt(seq / "times.txt"))
    tns = np.loadtxt(seq / "times_ns.txt", dtype=np.int64).astype(np.float64)
    od = Odom(seq / "odom.csv", tns)
    pa = load_traj(a.a_traj, times)
    pb = load_traj(a.b_traj, times, a.b_offset) if a.b_traj != "-" else {}
    print(f"A: {len(pa)} tracked frames ({min(pa)}..{max(pa)})" + (f", B: {len(pb)} ({min(pb)}..{max(pb)})" if pb else ""))

    runs = od.still_runs(a.still_tol)
    print("in-place spins (frame range, turn deg): " +
          ", ".join(f"{i0}..{i1} {np.degrees(np.ptp(od.yaw[i0:i1 + 1])):.0f}" for i0, i1 in runs))
    maps = [pa] + ([pb] if pb else [])
    axis = spin_axis(pa, runs)
    l, rms, nseg, nobs = lever_arm(maps, runs)
    l -= axis * (l @ axis)  # the component along the spin axis is not observable; drop it
    print(f"spin axis in camera frame {axis.round(4)} (tilt from -y {np.degrees(np.arccos(-axis[1])):.2f} deg)")
    print(f"camera -> rotation centre lever arm {l.round(4)} m (|l| {np.linalg.norm(l) * 100:.1f} cm), "
          f"fit rms {rms * 100:.2f} cm over {nseg} spins / {nobs} frames")
    if not pb:
        return

    ties = [t for t in (tie(pa, pb, od, r, l, f"{k}") for k, r in enumerate(runs)) if t]
    if len(ties) < 2:
        sys.exit(f"need 2 spins tracked in both maps, found {len(ties)}")
    ties = [ties[0], ties[-1]] if len(ties) > 2 else ties
    (ca0, cb0), (ca1, cb1) = ties
    ua = np.mean([T[:3, :3] @ axis for T in pa.values()], 0)
    ub = np.mean([T[:3, :3] @ axis for T in pb.values()], 0)
    ua, ub = ua / np.linalg.norm(ua), ub / np.linalg.norm(ub)

    def basis(u, d):
        d = d - u * (d @ u)
        d /= np.linalg.norm(d)
        return np.stack([d, u, np.cross(d, u)], 1)

    # The floor is flat and the camera height is fixed, so each map's camera centres lie on one
    # plane. The path is too narrow to give roll, but its slope along the corridor is well
    # determined: tilt each map's vertical so that slope is zero, and match the plane heights.
    def level(poses, u, c0, c1):
        m = (c0 + c1) / 2
        C = np.array([T[:3, 3] for T in poses.values()]) - m
        for _ in range(3):
            d = (c1 - c0) - u * ((c1 - c0) @ u)
            d /= np.linalg.norm(d)
            x, h = C @ d, C @ u
            ok = np.ones(len(C), bool)
            for _ in range(3):
                beta, alpha = np.polyfit(x[ok], h[ok], 1)
                r = h - (alpha + beta * x)
                ok = np.abs(r) < max(3 * r[ok].std(), 0.01)
            u = u - beta * d
            u /= np.linalg.norm(u)
        h = C @ u
        return u, float(np.median(h[ok])), float(h[ok].std()), m

    ua0, ub0 = ua, ub
    ua, ha, sa, ma = level(pa, ua, ca0, ca1)
    ub, hb, sb, mb = level(pb, ub, cb0, cb1)
    print(f"levelling: A vertical moved {np.degrees(np.arccos(min(1, ua @ ua0))):.2f} deg, B {np.degrees(np.arccos(min(1, ub @ ub0))):.2f} deg; "
          f"camera height scatter A {sa * 100:.1f} cm, B {sb * 100:.1f} cm")
    R = basis(ua, ca1 - ca0) @ basis(ub, cb1 - cb0).T
    t = ma - R @ mb
    t += ua * (ha - hb)  # put both camera planes at the same height
    T_ab = np.eye(4)
    T_ab[:3, :3], T_ab[:3, 3] = R, t
    da, db = np.linalg.norm(ca1 - ca0), np.linalg.norm(cb1 - cb0)
    print(f"distance between the two spin centres: A {da:.3f} m, B {db:.3f} m, difference {abs(da - db) * 100:.1f} cm "
          f"({abs(da - db) / da * 100:.2f}%)")
    print(f"height change between spin centres: A {(ca1 - ca0) @ ua * 100:.1f} cm, B {(cb1 - cb0) @ ub * 100:.1f} cm")
    for name, poses, T0 in (("A", pa, np.eye(4)), ("B", pb, T_ab)):
        h = np.array([(T0 @ T)[:3, 3] @ ua for T in poses.values()])
        print(f"camera height in merged frame, leg {name}: median {np.median(h) * 100:.1f} cm, "
              f"p5..p95 {np.percentile(h, 5) * 100:.1f}..{np.percentile(h, 95) * 100:.1f} cm")
    h_ref = float(np.median([T[:3, 3] @ ua for T in pa.values()]))
    for k, (ca, cb) in enumerate(ties):
        print(f"tie {k} residual after alignment {np.linalg.norm(R @ cb + t - ca) * 100:.1f} cm")
    # independent check: heading change across each gap, SLAM (after alignment) vs odom
    for i0, i1 in runs:
        ia = [i for i in range(i0, i1 + 1) if i in pa]
        ib = [i for i in range(i0, i1 + 1) if i in pb]
        if not ia or not ib:
            continue
        i, j = (ia[-1], ib[0]) if ia[-1] < ib[0] else (ib[-1], ia[0])
        Ti, Tj = (pa[i], T_ab @ pb[j]) if i in pa and ia[-1] < ib[0] else (T_ab @ pb[i], pa[j])
        Rrel = Ti[:3, :3].T @ Tj[:3, :3]
        w = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])
        ang = np.degrees(np.arctan2(np.linalg.norm(w) / 2, (np.trace(Rrel) - 1) / 2)) * np.sign(w @ axis)
        oy = np.degrees(od.yaw[j] - od.yaw[i])
        print(f"heading change frames {i} -> {j}: SLAM {ang:.1f} deg, odom {oy:.1f} deg "
              f"(difference {((ang - oy + 180) % 360) - 180:.1f} deg)")
    if not a.out:
        return

    # ---- write merged dataset
    A, B, out = Path(a.a), Path(a.b), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    ta = json.loads((A / ("transforms_sparse.json" if (A / "transforms_sparse.json").exists() else "transforms.json")).read_text())
    tb = json.loads((B / ("transforms_sparse.json" if (B / "transforms_sparse.json").exists() else "transforms.json")).read_text())
    frames, n_off = [], 0

    def off_plane(M):
        return abs(np.asarray(M)[:3, 3] @ ua - h_ref) > a.max_height_err + b_half

    b_half = 0.0  # right cameras sit on the same plane (baseline is horizontal), no allowance needed
    for old in (out / "images").glob("*.png"):
        old.unlink()
    for f in ta["frames"]:
        if off_plane(f["transform_matrix"]):
            n_off += 1
            continue
        shutil.copy2(A / f["file_path"], out / f["file_path"])
        frames.append({"file_path": f["file_path"], "transform_matrix": f["transform_matrix"]})
    have = {f["file_path"] for f in frames}
    for f in tb["frames"]:
        side, idx = Path(f["file_path"]).stem.split("_")
        name = f"images/{side}_{int(idx) + a.b_offset:06d}.png"
        if name in have:
            continue  # frame already posed by leg A
        if off_plane(T_ab @ np.array(f["transform_matrix"])):
            n_off += 1
            continue
        shutil.copy2(B / f["file_path"], out / name)
        frames.append({"file_path": name, "transform_matrix": (T_ab @ np.array(f["transform_matrix"])).tolist()})
    print(f"dropped {n_off} images more than {a.max_height_err * 100:.0f} cm off the camera plane")
    kept_ids = {int(Path(f["file_path"]).stem.split("_")[1]) for f in frames}
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

    head, xa, ca = read_ply(A / "sparse_pc.ply")
    _, xb, cb = read_ply(B / "sparse_pc.ply")
    xyz = np.vstack([xa, xb @ R.T + t]).astype("<f4")
    rgb = np.vstack([ca, cb])
    d = np.zeros(len(xyz), np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)]))
    d["xyz"], d["rgb"] = xyz, rgb
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
    rows = [rows[0]] + sorted((r for r in rows[1:] if int(r.split("\t")[0]) in kept_ids),
                              key=lambda r: int(r.split("\t")[0]))
    (out / "keyframes.tsv").write_text("\n".join(rows) + "\n")
    ta["frames"] = frames
    ta["ply_file_path"] = "sparse_pc.ply"
    (out / "transforms.json").write_text(json.dumps(ta, indent=1))
    np.savetxt(out / "T_a_b.txt", T_ab, fmt="%.9f")
    print(f"wrote {out}/transforms.json ({len(frames)} frames), "
          f"sparse_pc.ply ({len(xyz)} points), T_a_b.txt")


if __name__ == "__main__":
    main()
