# Trimmed pySLAM image: stereo SLAM, headless, exactly the parts our runs use.
#
# What our runs use (measured 2026-10-22 by tracing every module and shared library a full seq 10
# run loaded in the original 157 GB rosdocker image, then finding the smallest Python set that runs):
#   ORB2 features (orbslam2_features), DBoW3 loop closure (pydbow3), pySLAM's C++ core (cpp_core),
#   g2o, GTSAM (+ gtsam_factors), pySLAM's cpp/ modules, OpenCV 4.13 + contrib built from source,
#   and the Python packages in requirements_pyslam_headless_pinned.txt.
# Left out: CUDA, MonoGS / Gaussian splatting, every learned feature, depth and segmentation model,
#   TensorFlow, Open3D, Pangolin and the GUI, the git submodules, pip caches and build trees.
#
# Faithfulness: pySLAM is pinned at the commit of the original image (a5ff256) and every C++ part
# is built with pySLAM's own build scripts, so the compiler flags match the original image. The
# differences are: no CUDA (OpenCV without its CUDA modules, torch CPU build, faiss-cpu 1.14.2
# instead of faiss-gpu-cu12 1.14.1), and only the Python packages pySLAM actually imports.
# Our two pySLAM patches (crash guard; headless bundle adjust + save) are applied at build time.
#
# Like pySLAM's scripts, GTSAM and the C++ core are compiled with -march=native: build the image on
# the machine (or CPU family) that will run it.
#
# Build (from the repository root, about 1 to 1.5 hours, needs internet):
#   docker build -t pyslam_trimmed_headless -f pyslam_docker_image_trimmed_headless/Dockerfile .
# Run: see run_pyslam_headless.sh next to this file.

ARG UBUNTU_VERSION=20.04

########################################################################################################
FROM ubuntu:${UBUNTU_VERSION} AS builder
ARG DEBIAN_FRONTEND=noninteractive
ARG PYSLAM_COMMIT=a5ff2562eb929ed9a08420f528a120a3cca65585
ARG PYTHON_VERSION=3.11.9
ARG PYENV_TAG=v2.5.0
ENV TZ=Etc/UTC

# pySLAM's scripts call sudo and lsb_release; as root sudo is a no-op
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo lsb-release git curl wget ca-certificates gnupg software-properties-common \
        build-essential cmake ninja-build pkg-config unzip rsync file \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11.9, the version of the original image, built by pyenv's python-build
RUN git clone --depth 1 --branch ${PYENV_TAG} https://github.com/pyenv/pyenv.git /tmp/pyenv \
    && /tmp/pyenv/plugins/python-build/bin/python-build ${PYTHON_VERSION} /opt/python \
    && rm -rf /tmp/pyenv \
    && /opt/python/bin/python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:/opt/python/bin:$PATH

# pySLAM at the original image's commit, without submodules (none of them is used)
RUN git clone https://github.com/luigifreda/pyslam.git /opt/pyslam \
    && cd /opt/pyslam && git checkout ${PYSLAM_COMMIT}

# System packages exactly as pySLAM installs them (no CUDA present, so its CUDA steps are skipped)
RUN cd /opt/pyslam && bash scripts/install_system_packages.sh && rm -rf /var/lib/apt/lists/*

# Python packages, pinned. PIP_CONSTRAINT also holds every pip call inside pySLAM's scripts
# (for example `pip3 install --upgrade numpy` in the OpenCV script) to these versions.
COPY pyslam_docker_image_trimmed_headless/requirements_pyslam_headless_pinned.txt /opt/requirements_pyslam_headless_pinned.txt
RUN grep -v -E "^(--|torch|torchvision)" /opt/requirements_pyslam_headless_pinned.txt > /opt/pip_constraints.txt \
    && pip install --no-cache-dir --upgrade pip==26.2.1 setuptools==84.0.0 wheel==0.48.0 \
    && pip install --no-cache-dir -r /opt/requirements_pyslam_headless_pinned.txt
ENV PIP_CONSTRAINT=/opt/pip_constraints.txt PIP_NO_CACHE_DIR=1

# OpenCV 4.13.0 + contrib (non-free on) from source, pySLAM's script and flags; CUDA off
RUN cd /opt/pyslam && bash scripts/install_opencv_local.sh \
    && python -c "import cv2; print('cv2', cv2.__version__, cv2.__file__)"

# pySLAM's helper bash_utils.sh is bash-only; RUN defaults to /bin/sh (dash). Set here, not at the
# top, so the cached OpenCV layer above stays valid.
SHELL ["/bin/bash", "-c"]
# The C++ parts our runs load, each with pySLAM's own build script and options
RUN cd /opt/pyslam && . ./bash_utils.sh \
    && EXT="-DWITH_PYTHON_INTERP_CHECK=ON -DOpenCV_DIR=/opt/pyslam/thirdparty/opencv/install/lib/cmake/opencv4 -DCMAKE_POLICY_VERSION_MINIMUM=3.5" \
    && bash scripts/install_json_nlohmann.sh $EXT \
    && bash scripts/install_qhull.sh $EXT \
    && (cd thirdparty/orbslam2_features && ./build.sh $EXT) \
    && (cd thirdparty/g2opy && ./build.sh $EXT) \
    && (cd thirdparty/pydbow3 && ./build.sh $EXT) \
    && (cd thirdparty/pydbow2 && ./build.sh $EXT) \
    && (cd thirdparty/pyibow && ./build.sh $EXT)
RUN cd /opt/pyslam \
    && EXT="-DWITH_PYTHON_INTERP_CHECK=ON -DOpenCV_DIR=/opt/pyslam/thirdparty/opencv/install/lib/cmake/opencv4 -DCMAKE_POLICY_VERSION_MINIMUM=3.5" \
    && bash scripts/install_gtsam.sh $EXT \
    && python -c "import gtsam; print('gtsam', gtsam.__file__)"
RUN cd /opt/pyslam \
    && EXT="-DWITH_PYTHON_INTERP_CHECK=ON -DOpenCV_DIR=/opt/pyslam/thirdparty/opencv/install/lib/cmake/opencv4 -DCMAKE_POLICY_VERSION_MINIMUM=3.5" \
    && (cd cpp && bash build.sh $EXT) \
    && bash build_cpp_core.sh

# pySLAM on the path (the original used `pip install -e .`, which only adds this path)
RUN echo /opt/pyslam > $(python -c "import site; print(site.getsitepackages()[0])")/pyslam_source.pth

# Our two patches (pyslam_patches_and_configs/ in this repository)
COPY pyslam_patches_and_configs/patch_pyslam_tracking.py pyslam_patches_and_configs/patch_pyslam_headless_bundle_adjust_and_save.py /opt/pyslam_patches/
RUN python /opt/pyslam_patches/patch_pyslam_tracking.py /opt/pyslam \
    && python /opt/pyslam_patches/patch_pyslam_headless_bundle_adjust_and_save.py /opt/pyslam

# ORB vocabulary for DBoW3 loop closure. pySLAM otherwise downloads it from Google Drive at the start
# of every run (the container is removed after each run). md5 = the file in the original image.
RUN python -c "import gdown; gdown.download('https://drive.google.com/uc?id=13xmRtop_ow3aPtv3qCT5beG19_mlogqI', '/opt/pyslam/data/ORBvoc.dbow3', quiet=True)" \
    && echo "b24c59276b4b9bb44ad9342b93a05dc8  /opt/pyslam/data/ORBvoc.dbow3" | md5sum -c -

# Drop sources and build trees; keep install/ and lib/ folders (checked by ldd in the final stage)
RUN cd /opt/pyslam && rm -rf .git thirdparty/opencv/build thirdparty/opencv/opencv_contrib-* thirdparty/opencv/*.zip \
        thirdparty/opencv/3rdparty thirdparty/opencv/modules thirdparty/opencv/doc thirdparty/opencv/samples \
    && find thirdparty cpp pyslam/slam/cpp -type d -name build -prune -exec rm -rf {} + \
    && find / -xdev -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

# Runtime apt packages = owners of every system library our .so files link to
RUN find /opt/pyslam /opt/venv /opt/python -name "*.so*" -type f -print0 \
        | xargs -0 ldd 2>/dev/null | awk '/=> \// {print $3}' | sort -u \
        | grep -v "^/opt/" | xargs -r readlink -f | sort -u \
        | xargs -r dpkg -S 2>/dev/null | cut -d: -f1 | sort -u > /opt/runtime_apt_packages.txt \
    && wc -l /opt/runtime_apt_packages.txt

########################################################################################################
FROM ubuntu:${UBUNTU_VERSION}
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
COPY --from=builder /opt/runtime_apt_packages.txt /opt/runtime_apt_packages.txt
# libopengl0: PyOpenGL loads libOpenGL.so.0 with ctypes (main_slam.py always imports pyqtgraph.opengl
# through the plot module), so ldd above cannot see it. Found by comparing the libraries the old
# image's seq 10 run loaded with this image: the only other gaps were the CUDA libraries.
RUN apt-get update && apt-get install -y --no-install-recommends $(cat /opt/runtime_apt_packages.txt) ca-certificates libopengl0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/pyslam /opt/pyslam
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:/opt/python/bin:$PATH \
    PYTHONPATH= TF_CPP_MIN_LOG_LEVEL=3 MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen \
    HOME=/opt/home

# The gtsam Python module's RUNPATH points into gtsam_local/build (pruned above); the same libraries
# (libgtsam, libgtsam_unstable, libmetis-gtsam) are in gtsam_local/install/lib, so register that folder.
RUN echo /opt/pyslam/thirdparty/gtsam_local/install/lib > /etc/ld.so.conf.d/pyslam_gtsam_local.conf && ldconfig

# Every library we compiled must resolve (libpython3.11 is provided by the running interpreter).
# Pip wheels are not checked: their bundled libs resolve at import time, and the unresolved ones
# (PyQt5 3D/audio/PDF plugins, Open3D's CUDA torch ops, numba's TBB pool) are never loaded headless.
RUN S=/opt/venv/lib/python3.11/site-packages; \
    missing=$(find /opt/pyslam $S/gtsam $S/gtsam_unstable $S/cv2 -name "*.so*" -type f -print0 | xargs -0 ldd 2>/dev/null \
              | grep "not found" | grep -v "libpython3.11.so" | sort -u); \
    if [ -n "$missing" ]; then echo "$missing"; exit 1; fi

# Smoke test, in main_slam.py's import order (pyslam.config puts pySLAM's lib folders on sys.path):
# the C++ core must be on, and gtsam, gtsam_factors and pySLAM's cpp/ modules must import
RUN cd /opt/pyslam && python -c "import torch, cv2, numpy; from pyslam.config import Config; \
import gtsam, gtsam_factors, pyslam_utils; from pyslam.slam.slam import Slam; from pyslam.slam import USE_CPP; \
from pyslam.viz.slam_plot_drawer import SlamPlotDrawerThread; \
assert USE_CPP; print('pySLAM imports OK; C++ core on; cv2', cv2.__version__, 'numpy', numpy.__version__, 'torch', torch.__version__)"

# Writable by whatever uid runs the container (docker run -u $(id -u):$(id -g)); evo plots without a display.
# After the smoke test: it creates __pycache__ folders as root, and numba (pySLAM's JIT) fails to start
# when it cannot write its cache there.
RUN mkdir -p /opt/home /opt/pyslam/results /opt/pyslam/logs \
    && evo_config set plot_backend Agg \
    && chmod -R a+rwX /opt/home /opt/pyslam

WORKDIR /opt/pyslam
CMD ["bash"]
