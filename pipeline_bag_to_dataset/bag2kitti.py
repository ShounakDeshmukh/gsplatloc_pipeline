#!/usr/bin/env python3
"""Convert a ROS 2 bag (sqlite3 or mcap) with ZED stereo images to the KITTI odometry
layout used by pySLAM's KITTI_DATASET loader. No ROS install needed.

    pip install rosbags opencv-python numpy

    bag2kitti.py <bag_dir> --list
    bag2kitti.py <bag_dir> <out_dir> [--left T] [--right T] [--left-info T] [--right-info T]
                 [--imu T] [--odom T] [--zed-conf SNxxxx.conf] [--slop 0.002] [--stride 1]

Topics are auto-detected when not given (Image or CompressedImage with 'left'/'right'
in the name, rectified topics preferred; camera_info from the same namespace).
If camera_info shows distortion or a non-identity R, images are rectified here using
camera_info (K, D, R, P). Intrinsics and bf are taken from the P matrices.
Without camera_info, pass --zed-conf (the factory SN<serial>.conf): images are rectified
with cv2.stereoRectify (alpha 0, zero disparity), picking the resolution block by image width.
The baseline-direction variant with the smallest row error on sampled frames is used.

Output matches zed_oc_recorder: image_0/, image_1/, image_2 -> image_0, image_3 -> image_1,
times.txt, times_ns.txt, camera.txt, calib.txt, [imu.csv], [tf_static.txt], log.txt
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
INFO_TYPE = "sensor_msgs/msg/CameraInfo"
IMU_TYPE = "sensor_msgs/msg/Imu"
ODOM_TYPE = "nav_msgs/msg/Odometry"
TF_TYPE = "tf2_msgs/msg/TFMessage"


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def decode(msg, msgtype):
    if msgtype == "sensor_msgs/msg/CompressedImage":
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot decode compressed image (format '{msg.format}')")
        return img
    enc = msg.encoding.lower()
    if enc in ("yuv422", "uyvy", "yuv422_yuy2", "yuyv", "yuy2"):
        h, w = msg.height, msg.width
        a = np.frombuffer(msg.data, np.uint8).reshape(h, msg.step)[:, : w * 2].reshape(h, w, 2)
        code = cv2.COLOR_YUV2BGR_UYVY if enc in ("yuv422", "uyvy") else cv2.COLOR_YUV2BGR_YUYV
        return cv2.cvtColor(a, code)
    ch = {"bgra8": 4, "rgba8": 4, "8uc4": 4, "bgr8": 3, "rgb8": 3, "8uc3": 3, "mono8": 1, "8uc1": 1}.get(enc)
    if ch is None:
        raise ValueError(f"unsupported encoding '{msg.encoding}'")
    h, w = msg.height, msg.width
    a = np.frombuffer(msg.data, np.uint8).reshape(h, msg.step)[:, : w * ch].reshape(h, w, ch)
    if enc in ("bgra8", "8uc4"):
        return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(a, cv2.COLOR_RGBA2BGR)
    if enc == "rgb8":
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    if ch == 1:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    return a.copy()


def list_topics(reader):
    print(f"duration {(reader.end_time - reader.start_time) / 1e9:.1f} s")
    for c in sorted(reader.connections, key=lambda c: c.topic):
        print(f"  {c.topic:60s} {c.msgtype:36s} {c.msgcount}")


def autodetect(reader, side):
    cands = [c.topic for c in reader.connections if c.msgtype in IMAGE_TYPES and side in c.topic.lower()]
    cands = [t for t in cands if "depth" not in t and "disparity" not in t and "confidence" not in t]
    if not cands:
        return None
    def score(t):
        return (("rect" in t) * 4 + ("color" in t) * 2 + ("gray" not in t) - ("raw" in t) * 4
                - ("compressed" in t) * 1 - len(t) / 1000)
    return max(cands, key=score)


def info_topic_for(reader, image_topic):
    base = image_topic[: -len("/compressed")] if image_topic.endswith("/compressed") else image_topic
    parent = base.rsplit("/", 1)[0]
    topics = {c.topic for c in reader.connections if c.msgtype == INFO_TYPE}
    for t in (parent + "/camera_info", base + "/camera_info"):
        if t in topics:
            return t
    return None


def first_msg(reader, topic):
    conns = [c for c in reader.connections if c.topic == topic]
    for conn, _, raw in reader.messages(connections=conns):
        return reader.deserialize(raw, conn.msgtype)
    raise SystemExit(f"no messages on {topic}")


def rect_maps(info):
    K = np.array(info.k, float).reshape(3, 3)
    D = np.array(info.d, float)
    R = np.array(info.r, float).reshape(3, 3)
    P = np.array(info.p, float).reshape(3, 4)
    needs = bool(np.any(np.abs(D) > 1e-9)) or not np.allclose(R, np.eye(3), atol=1e-9)
    if not needs:
        return P, None
    model = info.distortion_model
    if model not in ("plumb_bob", "rational_polynomial"):
        raise SystemExit(f"distortion model '{model}' not supported")
    size = (info.width, info.height)
    m = cv2.initUndistortRectifyMap(K, D, R, P[:, :3], size, cv2.CV_32FC1)
    return P, m


def read_zed_conf(path):
    sec, d = None, {}
    for line in open(path):
        line = line.strip()
        if line.startswith("["):
            sec = line.strip("[]").upper()
        elif "=" in line and sec:
            k, v = line.split("=", 1)
            try:
                d[(sec, k.strip().upper())] = float(v)
            except ValueError:
                pass
    return d


ZED_RES = {2208: "2K", 1920: "FHD", 1280: "HD", 672: "VGA"}


def zed_rect_variants(conf, size):
    w, h = size
    res = ZED_RES.get(w)
    if res is None:
        raise SystemExit(f"image width {w} is not a ZED mode (2208/1920/1280/672)")
    def cam(side):
        g = lambda k: conf[(f"{side}_CAM_{res}", k)]
        K = np.array([[g("FX"), 0, g("CX")], [0, g("FY"), g("CY")], [0, 0, 1]])
        D = np.array([g("K1"), g("K2"), g("P1"), g("P2"), g("K3")])
        return K, D
    KL, DL = cam("LEFT"); KR, DR = cam("RIGHT")
    st = lambda k, dflt=0.0: conf.get(("STEREO", k), dflt)
    b = st("BASELINE")
    ty = st(f"TY_{res}", st("TY")); tz = st(f"TZ_{res}", st("TZ"))
    rv = np.array([st(f"RX_{res}"), st(f"CV_{res}"), st(f"RZ_{res}")])
    R, _ = cv2.Rodrigues(rv)
    out = []
    for name, T, Rm in (("T=(+b,ty,tz)", [b, ty, tz], R), ("T=(+b,0,0) zed-open-capture", [b, 0, 0], R),
                        ("T=(-b,-ty,-tz)", [-b, -ty, -tz], R), ("T=(-b,-ty,-tz) R^T", [-b, -ty, -tz], R.T)):
        T = np.array(T, float).reshape(3, 1) / 1000.0
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(KL, DL, KR, DR, size, Rm, T,
                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=size)
        ml = cv2.initUndistortRectifyMap(KL, DL, R1, P1[:, :3], size, cv2.CV_32FC1)
        mr = cv2.initUndistortRectifyMap(KR, DR, R2, P2[:, :3], size, cv2.CV_32FC1)
        out.append((name, P1, abs(P2[0, 3]), ml, mr))
    return res, out


def row_error(left, right):
    """Sub-pixel vertical misalignment: LK-track corners from left to right, forward-backward checked."""
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(gl, 1500, 0.01, 8)
    if p0 is None or len(p0) < 30:
        return None
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    p0 = cv2.cornerSubPix(gl, p0.astype(np.float32), (5, 5), (-1, -1), crit)
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=crit)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(gl, gr, p0, None, **lk)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(gr, gl, p1, None, **lk)
    a, b, back = p0.reshape(-1, 2), p1.reshape(-1, 2), pb.reshape(-1, 2)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(a - back, axis=1) < 0.3)
    ok &= (a[:, 0] - b[:, 0] > -1) & (np.abs(a[:, 1] - b[:, 1]) < 20)
    dy = np.abs(a[ok, 1] - b[ok, 1])
    return (float(np.median(dy)), int(len(dy)), float(np.percentile(dy, 90))) if len(dy) >= 30 else None


def sample_pairs(reader, left_t, right_t, n=6, slop_ns=2_000_000):
    """Return up to n decoded (L, R) pairs spread over the bag."""
    conns = [c for c in reader.connections if c.topic in (left_t, right_t)]
    span = reader.end_time - reader.start_time
    targets = [reader.start_time + span * (i + 1) // (n + 1) for i in range(n)]
    pairs, buf, ti = [], {left_t: {}, right_t: {}}, 0
    for conn, t, raw in reader.messages(connections=conns):
        if ti >= len(targets):
            break
        if t < targets[ti]:
            continue
        msg = reader.deserialize(raw, conn.msgtype)
        s = stamp_ns(msg)
        other = right_t if conn.topic == left_t else left_t
        m = min(buf[other], key=lambda k: abs(k - s), default=None)
        if m is not None and abs(m - s) <= slop_ns:
            om, ot = buf[other].pop(m)
            pair = (msg, conn.msgtype, om, ot) if conn.topic == left_t else (om, ot, msg, conn.msgtype)
            pairs.append((decode(pair[0], pair[1]), decode(pair[2], pair[3])))
            buf = {left_t: {}, right_t: {}}
            ti += 1
        else:
            buf[conn.topic][s] = (msg, conn.msgtype)
    return pairs


def rect_check(left, right):
    e = row_error(left, right)
    if e is None:
        return "rect check: too few tracked corners on this frame"
    med, n, p90 = e
    flag = "  <-- BAD, rows not aligned" if med > 0.5 else "  (ok)"
    return f"rect check: {n} corners, |dy| median {med:.3f} px, p90 {p90:.3f} px{flag}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--left"); ap.add_argument("--right")
    ap.add_argument("--left-info"); ap.add_argument("--right-info")
    ap.add_argument("--imu", help="IMU topic, e.g. /zed/zed_node/imu/data")
    ap.add_argument("--odom", help="nav_msgs/Odometry topic to dump to odom.csv, e.g. /odom")
    ap.add_argument("--zed-conf", help="ZED factory SN<serial>.conf, used when there is no camera_info")
    ap.add_argument("--slop", type=float, default=0.002, help="max L/R stamp difference [s]")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth pair")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    bag = Path(a.bag)
    if bag.is_dir() and not (bag / "metadata.yaml").exists():
        paths = sorted(bag.glob("*.mcap"))  # bare split files, no metadata.yaml
        if not paths:
            raise SystemExit(f"{bag}: no metadata.yaml and no .mcap files")
        print(f"no metadata.yaml, reading {len(paths)} .mcap file(s) as one recording")
    else:
        paths = [bag]
    with AnyReader(paths, default_typestore=typestore) as reader:
        if a.list or not a.out:
            list_topics(reader)
            return

        left_t = a.left or autodetect(reader, "left")
        right_t = a.right or autodetect(reader, "right")
        if not left_t or not right_t:
            list_topics(reader)
            raise SystemExit("could not detect left/right image topics, pass --left/--right")
        li_t = None if a.zed_conf else (a.left_info or info_topic_for(reader, left_t))
        ri_t = None if a.zed_conf else (a.right_info or info_topic_for(reader, right_t))
        print(f"left  {left_t}\nright {right_t}")
        calib_src = ""
        if li_t and ri_t:
            print(f"info  {li_t}, {ri_t}")
            li, ri = first_msg(reader, li_t), first_msg(reader, ri_t)
            Pl, mapl = rect_maps(li)
            Pr, mapr = rect_maps(ri)
            fx, fy, cx, cy = Pl[0, 0], Pl[1, 1], Pl[0, 2], Pl[1, 2]
            bf = -Pr[0, 3]
            if bf <= 0:
                raise SystemExit(f"right P[0,3] = {Pr[0, 3]}; expected -fx*baseline. Wrong topic pairing?")
            if abs(Pr[1, 2] - cy) > 0.5 or abs(Pr[0, 0] - fx) > 0.5:
                print("WARNING: left/right P differ in fx or cy; images may not be a rectified pair")
            w, h = li.width, li.height
            calib_src = f"camera_info {li_t} {ri_t}"
        elif a.zed_conf:
            first = decode(first_msg(reader, left_t), next(c.msgtype for c in reader.connections if c.topic == left_t))
            h, w = first.shape[:2]
            res, variants = zed_rect_variants(read_zed_conf(a.zed_conf), (w, h))
            samples = sample_pairs(reader, left_t, right_t, slop_ns=int(a.slop * 1e9))
            if not samples:
                raise SystemExit("no L/R pairs found for calibration check; try a larger --slop")
            best = None
            for name, P1, bfv, ml, mr in variants:
                errs = [row_error(cv2.remap(L, *ml, cv2.INTER_LINEAR), cv2.remap(R, *mr, cv2.INTER_LINEAR))
                        for L, R in samples]
                errs = [e for e in errs if e]
                med = float(np.median([e[0] for e in errs])) if errs else float("inf")
                cnt = int(sum(e[1] for e in errs))
                print(f"  {name:30s} |dy| median {med:.3f} px over {len(errs)}/{len(samples)} frames, {cnt} corners")
                if best is None or med < best[0] - 0.02:
                    best = (med, cnt, name, P1, bfv, ml, mr)
            med, _, name, P1, bf, mapl, mapr = best
            if not np.isfinite(med):
                raise SystemExit("rectification check found no matches on sampled frames (dark or textureless?)")
            fx, fy, cx, cy = P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2]
            calib_src = f"zed-conf {a.zed_conf} [{res}] {name}, sampled |dy| median {med:.3f} px"
            print(f"using {name}")
        else:
            raise SystemExit("no camera_info topics; pass --zed-conf SN<serial>.conf (or --left-info/--right-info)")
        rectify = mapl is not None or mapr is not None
        print(f"{w}x{h}  fx {fx:.4f} fy {fy:.4f} cx {cx:.4f} cy {cy:.4f} bf {bf:.4f} "
              f"(baseline {bf / fx:.6f} m)  rectify here: {rectify}")

        out = Path(a.out)
        if (out / "image_0").exists() and any((out / "image_0").iterdir()):
            raise SystemExit(f"{out}/image_0 is not empty")
        (out / "image_0").mkdir(parents=True, exist_ok=True)
        (out / "image_1").mkdir(parents=True, exist_ok=True)
        for link, tgt in (("image_2", "image_0"), ("image_3", "image_1")):
            p = out / link
            if p.is_symlink() or p.exists():
                p.unlink()
            os.symlink(tgt, p)
        (out / "camera.txt").write_text(
            f"fx: {fx:.6f}\nfy: {fy:.6f}\ncx: {cx:.6f}\ncy: {cy:.6f}\nbf: {bf:.6f}\n"
            f"width: {w}\nheight: {h}\nbaseline_m: {bf / fx:.6f}\n")
        (out / "calib.txt").write_text(
            f"P0: {fx:.12e} 0 {cx:.12e} 0 0 {fy:.12e} {cy:.12e} 0 0 0 1 0\n"
            f"P1: {fx:.12e} 0 {cx:.12e} {-bf:.12e} 0 {fy:.12e} {cy:.12e} 0 0 0 1 0\n")

        topics = {left_t, right_t}
        if a.imu:
            topics.add(a.imu)
        if a.odom:
            topics.add(a.odom)
        tf_conns = [c for c in reader.connections if c.topic == "/tf_static" and c.msgtype == TF_TYPE]
        conns = [c for c in reader.connections if c.topic in topics] + tf_conns

        pool = ThreadPoolExecutor(a.threads)
        pending = []
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

        def write_pair(i, lm, lt, rm, rt):
            L, R = decode(lm, lt), decode(rm, rt)
            if mapl is not None:
                L = cv2.remap(L, *mapl, cv2.INTER_LINEAR)
            if mapr is not None:
                R = cv2.remap(R, *mapr, cv2.INTER_LINEAR)
            ok = cv2.imwrite(str(out / "image_0" / f"{i:06d}.png"), L, params)
            ok &= cv2.imwrite(str(out / "image_1" / f"{i:06d}.png"), R, params)
            if not ok:
                raise IOError(f"write failed for frame {i}")
            return (L, R) if i == 0 else None

        slop_ns = int(a.slop * 1e9)
        buf = {left_t: {}, right_t: {}}  # stamp -> (msg, msgtype)
        n_pairs = n_kept = n_unmatched = n_backwards = 0
        last_emit = -1
        t0 = None
        times = open(out / "times.txt", "w")
        times_ns = open(out / "times_ns.txt", "w")
        imu_f = None
        if a.imu:
            imu_f = open(out / "imu.csv", "w")
            imu_f.write("t_ns,ax,ay,az,gx_rad_s,gy_rad_s,gz_rad_s\n")
        odom_f = None
        if a.odom:
            odom_f = open(out / "odom.csv", "w")
            odom_f.write("t_ns,x,y,z,qx,qy,qz,qw,vx,wz\n")
        tf_lines = set()
        first_pair = None

        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.msgtype == TF_TYPE:
                for t in msg.transforms:
                    tr, q = t.transform.translation, t.transform.rotation
                    tf_lines.add(f"{t.header.frame_id} -> {t.child_frame_id}: t [{tr.x:.6f} {tr.y:.6f} {tr.z:.6f}] "
                                 f"q_xyzw [{q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}]")
                continue
            if conn.topic == a.odom:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                odom_f.write(f"{stamp_ns(msg)},{p.x:.6f},{p.y:.6f},{p.z:.6f},{q.x:.6f},{q.y:.6f},{q.z:.6f},{q.w:.6f},"
                             f"{msg.twist.twist.linear.x:.6f},{msg.twist.twist.angular.z:.6f}\n")
                continue
            if conn.topic == a.imu:
                la, av = msg.linear_acceleration, msg.angular_velocity
                imu_f.write(f"{stamp_ns(msg)},{la.x:.6f},{la.y:.6f},{la.z:.6f},{av.x:.6f},{av.y:.6f},{av.z:.6f}\n")
                continue
            s = stamp_ns(msg)
            other = right_t if conn.topic == left_t else left_t
            ob = buf[other]
            match = min(ob, key=lambda k: abs(k - s), default=None)
            if match is not None and abs(match - s) <= slop_ns:
                om, ot = ob.pop(match)
                if conn.topic == left_t:
                    lm, lt, rm, rt, st = msg, conn.msgtype, om, ot, s
                else:
                    lm, lt, rm, rt, st = om, ot, msg, conn.msgtype, match
                if st <= last_emit:
                    n_backwards += 1
                    continue
                last_emit = st
                n_pairs += 1
                if (n_pairs - 1) % a.stride:
                    continue
                if t0 is None:
                    t0 = st
                times.write(f"{(st - t0) / 1e9:.6f}\n")
                times_ns.write(f"{st}\n")
                pending.append(pool.submit(write_pair, n_kept, lm, lt, rm, rt))
                n_kept += 1
                if len(pending) > 4 * a.threads:
                    r = pending.pop(0).result()
                    first_pair = first_pair or r
                if n_kept % 100 == 0:
                    print(f"\r{n_kept} pairs", end="", flush=True)
            else:
                buf[conn.topic][s] = (msg, conn.msgtype)
            for b in buf.values():  # evict stale unmatched entries
                for k in [k for k in b if s - k > 1_000_000_000]:
                    del b[k]
                    n_unmatched += 1

        for f in pending:
            r = f.result()
            first_pair = first_pair or r
        pool.shutdown()
        times.close(); times_ns.close()
        if imu_f:
            imu_f.close()
        if odom_f:
            odom_f.close()
        n_unmatched += sum(len(b) for b in buf.values())
        if tf_lines:
            (out / "tf_static.txt").write_text("\n".join(sorted(tf_lines)) + "\n")

        check = rect_check(*first_pair) if first_pair else "rect check: no pairs"
        summary = (f"pairs {n_pairs}, written {n_kept} (stride {a.stride}), unmatched singles {n_unmatched}, "
                   f"out-of-order dropped {n_backwards}")
        dur = (last_emit - t0) / 1e9 if t0 is not None else 0
        print(f"\n{summary}\n{dur:.1f} s, {n_kept / dur if dur else 0:.1f} fps written\n{check}")
        (out / "log.txt").write_text(
            f"bag {a.bag}\nleft {left_t}\nright {right_t}\ncalib {calib_src}\nimu {a.imu}\nodom {a.odom}\n"
            f"rectified here: {rectify}\n{summary}\n{check}\nt0_ns {t0}\n")
        if n_kept == 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
