#!/usr/bin/env python3
"""Make a KITTI-layout sub-sequence of symlinks: either a sequence minus its corrupted frames, or
one leg (a frame range) of a sequence that SLAM has to run in two parts.

usage (Oct 10 recording, from ~/ros2_bag):
  uv run python make_subsequence.py kitti/sequences/08 kitti/sequences/09 --drop kitti/sequences/08/corrupted_frames.txt
  uv run python make_subsequence.py kitti/sequences/09 kitti/sequences/10 --first 0 --last 4800
  uv run python make_subsequence.py kitti/sequences/09 kitti/sequences/11 --first 5115 --last 10925

Writes <dst>/image_0, image_1 (relative symlinks to the real image files), image_2/3 symlinks,
camera.txt, calib.txt, times.txt (restarted at 0), times_ns.txt, odom/imu symlinks, frame_map.txt
(new index, source index, and the source's own frame_map columns), log.txt, and the empty
kitti/poses/<NN>.txt that pySLAM's KITTI loader needs. The leg boundaries (--first/--last) are a
manual choice: cut out the stretch where tracking breaks (e.g. a spin onto a blank wall); the
second leg's --first is the --b-offset of merge_legs_one_spin.py."""
import argparse
import os
import shutil


def link_target(src, rel, dst_depth):
    """Relative target for a link in dst to src/rel; reuses src's own link if it is one."""
    p = os.path.join(src, rel)
    if os.path.islink(p):
        return os.readlink(p)
    return os.path.join(*([".."] * dst_depth), os.path.basename(src), rel)


ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--drop", help="file with frame numbers to leave out (whitespace separated)")
ap.add_argument("--first", type=int, help="first source frame to keep")
ap.add_argument("--last", type=int, help="last source frame to keep (inclusive)")
args = ap.parse_args()
src, dst = args.src.rstrip("/"), args.dst.rstrip("/")
sname, dname = "seq" + os.path.basename(src), "seq" + os.path.basename(dst)

times = open(f"{src}/times.txt").read().split()
times_ns = open(f"{src}/times_ns.txt").read().split()
n = len(times)
keep = range(n)
if args.first is not None or args.last is not None:
    keep = range(args.first or 0, (n - 1 if args.last is None else args.last) + 1)
if args.drop:
    drop = {int(x) for x in open(args.drop).read().split()}
    keep = [i for i in keep if i not in drop]
keep = list(keep)

src_map = {}
if os.path.exists(f"{src}/frame_map.txt"):
    lines = open(f"{src}/frame_map.txt").read().splitlines()
    src_header = lines[0].split()[2:]
    for l in lines[1:]:
        c = l.split()
        src_map[int(c[0])] = c[1:]
else:
    src_header = []

if os.path.exists(dst):
    raise SystemExit(f"{dst} exists; remove it first")
for side in ("image_0", "image_1"):
    os.makedirs(f"{dst}/{side}")
    for j, i in enumerate(keep):
        os.symlink(link_target(src, f"{side}/{i:06d}.png", 2), f"{dst}/{side}/{j:06d}.png")
os.symlink("image_0", f"{dst}/image_2")
os.symlink("image_1", f"{dst}/image_3")
for f in ("camera.txt", "calib.txt"):
    shutil.copy(f"{src}/{f}", f"{dst}/{f}")
for f in ("imu.csv", "odom.csv"):
    if os.path.lexists(f"{src}/{f}"):
        os.symlink(link_target(src, f, 1), f"{dst}/{f}")

t0 = float(times[keep[0]])
with open(f"{dst}/times.txt", "w") as f:
    f.writelines(f"{float(times[i]) - t0:.6f}\n" for i in keep)
with open(f"{dst}/times_ns.txt", "w") as f:
    f.writelines(f"{times_ns[i]}\n" for i in keep)
with open(f"{dst}/frame_map.txt", "w") as f:
    f.write(" ".join(["#", f"{dname}_idx", f"{sname}_idx"] + src_header) + "\n")
    f.writelines(" ".join([str(j), str(i)] + src_map.get(i, [])) + "\n" for j, i in enumerate(keep))

what = []
if args.first is not None or args.last is not None:
    what.append(f"{sname} frames {keep[0]}..{keep[-1]}")
if args.drop:
    what.append(f"minus the frames in {args.drop}")
with open(f"{dst}/log.txt", "w") as f:
    f.write(f"sub-sequence of {src}: {' '.join(what) or 'all frames'}; {len(keep)} frames, symlinks\n")
poses = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(dst))), "poses")
os.makedirs(poses, exist_ok=True)
open(os.path.join(poses, os.path.basename(dst) + ".txt"), "a").close()
print(f"wrote {dst}: {len(keep)} of {n} frames of {src}")
