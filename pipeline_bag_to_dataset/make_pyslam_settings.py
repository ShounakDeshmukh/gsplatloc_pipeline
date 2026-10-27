#!/usr/bin/env python3
"""Write a pySLAM camera settings YAML from a bag2kitti camera.txt.
    python3 make_pyslam_settings.py <seq>/camera.txt <seq>/times.txt <template.yaml> <out.yaml>
"""
import re, sys
import numpy as np
cam_txt, times_txt, template, out = sys.argv[1:5]
c = dict(l.split(": ") for l in open(cam_txt).read().splitlines() if ": " in l)
t = np.loadtxt(times_txt)
fps = round(1.0 / np.median(np.diff(t)), 1)
vals = {"fx": c["fx"], "fy": c["fy"], "cx": c["cx"], "cy": c["cy"],
        "width": str(int(float(c["width"]))), "height": str(int(float(c["height"]))),
        "fps": str(fps), "bf": c["bf"]}
s = open(template).read()
for k, v in vals.items():
    s, n = re.subn(rf"^Camera\.{k}:.*$", f"Camera.{k}: {v}", s, flags=re.M)
    assert n == 1, k
open(out, "w").write(s)
print(f"wrote {out}: {vals}")
