# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.3.0"

import os

# Set ENV Variables (place before imports)
os.environ["OMP_NUM_THREADS"] = "1"  # reduce CPU utilization during training

# The data Explorer is optional (it pulls heavy embedding dependencies) and may be absent in
# minimal installs; import it lazily so the core package always imports.
try:
    from ultralytics.data.explorer.explorer import Explorer
except Exception:  # pragma: no cover - Explorer is non-essential for SWCB-YOLO
    Explorer = None
from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
