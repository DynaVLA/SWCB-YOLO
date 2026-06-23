#!/usr/bin/env bash
# SWCB-YOLO environment setup.
#
# This repository contains the SWCB-YOLO research code: the AS-Swin2 and FS-DDA modules, the
# CA-Shape-IoU loss, the offline morphology-field pipeline, and the modified Ultralytics core
# files (nn/tasks.py, utils/loss.py, utils/metrics.py, cfg/default.yaml) that wire them in.
#
# It is built on top of Ultralytics 8.3.0. To keep the repository focused on the contribution,
# the unmodified parts of the Ultralytics data pipeline are obtained from the pinned upstream
# release and the SWCB-YOLO files are overlaid on top. This script automates that overlay.
#
# Usage:
#   bash setup.sh            # create a venv, install deps, build the overlay
#   bash setup.sh --no-venv  # install into the current environment
set -e

ULTRALYTICS_VERSION="8.3.0"
USE_VENV=1
[ "${1:-}" = "--no-venv" ] && USE_VENV=0

echo "[setup] SWCB-YOLO setup (Ultralytics base ${ULTRALYTICS_VERSION})"

if [ "${USE_VENV}" = "1" ]; then
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "[setup] created and activated virtualenv .venv"
fi

python -m pip install --upgrade pip

# 1) Install the pinned upstream package to obtain the complete, unmodified base
#    (data pipeline, engine, utils) as a known-good reference implementation.
python -m pip install "ultralytics==${ULTRALYTICS_VERSION}"

# 2) Install the remaining SWCB-YOLO runtime dependencies.
python -m pip install -r requirements.txt

# 3) Locate the installed upstream package and copy its `data` package into this repo, so the
#    SWCB-YOLO fork has the unmodified data pipeline available without vendoring it in git.
SITE_PKG="$(python -c 'import ultralytics, os; print(os.path.dirname(ultralytics.__file__))')"
echo "[setup] upstream ultralytics found at: ${SITE_PKG}"

if [ ! -f "ultralytics/data/__init__.py" ]; then
    echo "[setup] copying upstream data pipeline into ultralytics/data/"
    cp -r "${SITE_PKG}/data/." "ultralytics/data/"
fi

echo "[setup] done."
echo "[setup] Next:"
echo "  1. Edit data.yaml to point at your dataset."
echo "  2. (optional) python tools/prepare_ca_fields.py --labels-root <labels> --out-root <ca_fields> --imgsz 1280"
echo "  3. python train.py --model yaml/swcb_yolo.yaml --data data.yaml --imgsz 1280 --cashapeiou --ca-fields <ca_fields>/train"
