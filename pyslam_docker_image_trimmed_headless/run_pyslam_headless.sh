#!/usr/bin/env bash
# Run headless pySLAM in the trimmed image on one config. Works over SSH (no display needed).
#
# usage: run_pyslam_headless.sh <pySLAM config.yaml> <output folder> [image]
#   <pySLAM config.yaml>  e.g. ~/ros2_bag/pyslam_cfg/config_seq10.yaml (paths inside must be absolute)
#   <output folder>       gets slam_state/ (map.json etc.), the trajectory folder if the config's
#                         SAVE_TRAJECTORY.output_folder points inside it, and headless.log
#   [image]               default pyslam_trimmed_headless
#
# What it replaces from the old per-container-start steps: the slam_state symlink (made here),
# both patches and the evo plot backend (baked into the image), sourcing the pySLAM venv (on PATH).
# The run ends by itself: bundle adjustment, map.json save, trajectory_final.txt, exit.
# Success = "HEADLESS: MAP SAVED" in headless.log.
set -euo pipefail
CONFIG="$(readlink -f "$1")"
OUT="$(readlink -f "$2")"
IMAGE="${3:-pyslam_trimmed_headless}"
mkdir -p "$OUT/slam_state"

# SYSTEM_STATE.folder_path is relative to the pySLAM root (/opt/pyslam); point it at $OUT/slam_state
STATE_REL=$(awk '/^SYSTEM_STATE:/{f=1} f && /folder_path:/{print $2; exit}' "$CONFIG")
[ -n "$STATE_REL" ] || { echo "no SYSTEM_STATE.folder_path in $CONFIG"; exit 1; }

docker run --rm --net=host --ipc=host --shm-size=16g -u "$(id -u):$(id -g)" \
    -v "$HOME/ros2_bag:$HOME/ros2_bag" -v "$OUT:$OUT" -v "$(dirname "$CONFIG"):$(dirname "$CONFIG")" \
    --name "pyslam_headless_$(date +%H%M%S)" "$IMAGE" bash -c "
set -e
mkdir -p /opt/pyslam/\$(dirname '$STATE_REL')
rm -rf '/opt/pyslam/$STATE_REL' && ln -s '$OUT/slam_state' '/opt/pyslam/$STATE_REL'
cd /opt/pyslam && python -u main_slam.py --headless -c '$CONFIG'
" > "$OUT/headless.log" 2>&1 || true

grep -a "HEADLESS:" "$OUT/headless.log" || echo "no HEADLESS lines: the run did not finish, see $OUT/headless.log"
