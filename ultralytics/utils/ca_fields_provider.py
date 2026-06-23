# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Runtime bridge between the offline CA-Shape-IoU morphology caches and the training loss.

The morphological terms of CA-Shape-IoU (curvature / Voronoi / elongation) need, for each
ground-truth object matched to a prediction, the polygon-derived fields produced offline by
``ultralytics/data/ca_shape_fields.py``. Because the Ultralytics loss does not receive image
file paths directly, this module exposes a tiny process-global provider that:

  * lazily loads per-image ``.npz`` caches from a directory keyed by image stem, and
  * is enabled/disabled by the ``cashapeiou`` flag and a ``ca_fields`` path.

Design choice (robustness): if no cache directory is configured, or a particular image has no
cache entry, the provider returns ``None`` and the loss falls back to its plain CIoU base
term for those samples. Training therefore always proceeds; the morphological gains simply
require the caches to be present, exactly mirroring the paper's "offline, training-only"
description. This avoids forking the dataloader and keeps the detector runnable on the stock
Ultralytics data pipeline.

Typical wiring (done automatically when ``cashapeiou=True`` and ``ca_fields`` is set in the
config or environment):

    from ultralytics.utils.ca_fields_provider import configure_provider
    configure_provider("/path/to/ca_fields/train", imgsz=1280)
"""

import os
from pathlib import Path

import numpy as np

_PROVIDER = {"enabled": False, "dir": None, "imgsz": 1280, "cache": {}}


def configure_provider(fields_dir, imgsz=1280):
    """Enable the morphology provider and point it at a directory of ``.npz`` caches."""
    if fields_dir is None:
        _PROVIDER["enabled"] = False
        return
    p = Path(fields_dir)
    _PROVIDER["enabled"] = p.is_dir()
    _PROVIDER["dir"] = p if p.is_dir() else None
    _PROVIDER["imgsz"] = imgsz
    _PROVIDER["cache"].clear()


def configure_from_env():
    """Configure the provider from ``SWCB_CA_FIELDS`` / ``SWCB_CA_IMGSZ`` environment variables."""
    fields_dir = os.environ.get("SWCB_CA_FIELDS", None)
    imgsz = int(os.environ.get("SWCB_CA_IMGSZ", "1280"))
    if fields_dir:
        configure_provider(fields_dir, imgsz=imgsz)


def is_enabled():
    """Return True if a valid cache directory is configured."""
    return bool(_PROVIDER["enabled"] and _PROVIDER["dir"] is not None)


def get_objects_for_stem(stem):
    """Return the list of per-object morphology dicts for an image stem, or None if absent."""
    if not is_enabled():
        return None
    cache = _PROVIDER["cache"]
    if stem in cache:
        return cache[stem]
    npz = _PROVIDER["dir"] / f"{stem}.npz"
    if not npz.is_file():
        cache[stem] = None
        return None
    data = np.load(npz, allow_pickle=True)
    objects = list(data["objects"])
    cache[stem] = objects
    return objects


def get_imgsz():
    """Return the rasterization canvas size the caches were built with."""
    return _PROVIDER["imgsz"]
