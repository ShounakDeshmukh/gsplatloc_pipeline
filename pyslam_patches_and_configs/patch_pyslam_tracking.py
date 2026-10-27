#!/usr/bin/env python3
"""Guard pySLAM's homography fallback against <4 matches (the C++ core has no such check and
cv::findHomography throws). Run inside the container after each start; idempotent.
usage: python ~/qualcomm_evk_loc_demo/pyslam_patches_and_configs/patch_pyslam_tracking.py [~/pyslam]"""
import os, re, sys

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/pyslam")
path = os.path.join(root, "pyslam/slam/tracking.py")
src = open(path).read()
if "len(idxs_cur) >= 4" in src:
    sys.exit("already patched")
pat = re.compile(r"(\n(\s+)\(f_cur\.is_blurry or f_ref\.is_blurry\)\n)")
out, n = pat.subn(lambda m: f"{m.group(1)}{m.group(2)}and len(idxs_cur) >= 4\n", src)
if n != 2:
    sys.exit(f"expected 2 call sites, found {n}: not patching")
open(path, "w").write(out)
print(f"patched {path} ({n} sites)")
