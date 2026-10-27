# Trimmed headless pySLAM Docker image

This image runs the stereo SLAM step of the pipeline (the ZED 2i sequence becomes a camera pose for
every frame plus a sparse map). It replaces the 157 GB `pyslam_cuda` image from
[rosdocker](https://github.com/luigifreda/rosdocker). It holds only what our runs use:

- **SLAM:** ORB2 features, DBoW3 loop closure, g2o, GTSAM, and pySLAM's C++ core;
- **headless:** no GUI, no CUDA.

| | Old `pyslam_cuda` | This image |
|---|---|---|
| Disk usage | 157 GB | 11.3 GB (3.9 GB compressed) |
| CUDA | yes | no (not used by stereo ORB SLAM) |
| Runs over SSH | only with a hand-typed `docker run` and per-start steps | `run_pyslam_headless.sh`, no per-start steps |
| 4,801 frames at 1104x621 | not timed | 6 min 15 s |

## What is in it and why

The old image's own seq 10 run was traced: every Python module and every shared library it loaded
was recorded. A clean Python environment was then grown package by package until the same run
finished. The result is `requirements_pyslam_headless_pinned.txt`, 122 pinned packages.

- **pySLAM version:** pinned at commit `a5ff256`, the same commit as the old image.
- **How the C++ parts are built:** with pySLAM's own build scripts, so the compiler flags match the
  old image. That covers OpenCV 4.13 + contrib, GTSAM, g2o, DBoW2/3, ORB2 features, the C++ core
  and pySLAM's `cpp/` modules.
- **Differences from the old image:**
  - no CUDA: OpenCV without its CUDA modules, the CPU build of torch 2.8.0, and `faiss-cpu`
    1.14.2 instead of `faiss-gpu-cu12` 1.14.1;
  - only the Python packages pySLAM actually imports. torch, Open3D and PyQt5 are still in,
    because pySLAM imports them on start-up even in headless mode.
- **Our two patches are applied at build time.** Both live in `../pyslam_patches_and_configs/`:
  - the tracking crash guard;
  - headless Bundle Adjust + map save.
- **The ORB vocabulary is included,** checked by md5 against the old image's copy. pySLAM would
  otherwise download it from Google Drive at the start of every run.
- **Built for this machine's CPU:** GTSAM and the C++ core are compiled with `-march=native`, as
  pySLAM's scripts do. Build the image on the machine (or CPU family) that will run it.

## Build

From the repository root. It needs internet and takes about 1.5 hours the first time (the OpenCV
source build is most of it):

```
docker build -t pyslam_trimmed_headless -f pyslam_docker_image_trimmed_headless/Dockerfile .
```

The build checks itself at the end. It fails if:
- any library we compiled cannot find a library it links to;
- the main pySLAM modules do not import;
- pySLAM's C++ core is not active.

## Run

```
pyslam_docker_image_trimmed_headless/run_pyslam_headless.sh <pySLAM config.yaml> <output folder>
```

- **Config:** a normal pySLAM `config.yaml` with absolute dataset and settings paths, made from
  `../pyslam_patches_and_configs/pyslam_config_templates/` (main README, step 4).
- **Output folder:**
  - `slam_state/` (`map.json` and the rest of the saved map);
  - `headless.log`;
  - the trajectory folder, if the config's `SAVE_TRAJECTORY.output_folder` points inside it.
- **What the script handles:** it maps `~/ros2_bag` into the container and runs as your own user,
  so the output files belong to you. It needs no display.
- **When it is done:** the run ends by itself (Bundle Adjust, map save, `trajectory_final.txt`,
  exit). Success is the line `HEADLESS: MAP SAVED` in `headless.log`; the script prints the
  `HEADLESS:` lines at the end.

## Validation

On the reference sequence (4,801 frames, 1104x621) four runs of this image agreed with the old
157 GB image to 1 to 7 cm median per frame. One of the four lost tracking briefly and recovered
with its heading about 3 degrees off, ending 2 m away: check every run (main README, step 5).
