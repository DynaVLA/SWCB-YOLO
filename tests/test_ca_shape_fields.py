# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Unit tests for the offline CA-Shape-IoU morphology pipeline.

These tests exercise the dependency-light parts of the loss (skeletonization, Bezier
curvature fitting, Voronoi field, elongation) and require only numpy / opencv / scikit-image /
scipy. They do NOT require PyTorch, so they run in CI without a GPU.

Run with::

    pytest tests/test_ca_shape_fields.py -v
"""

import numpy as np
import pytest

# The pipeline module imports cleanly without torch.
from ultralytics.data import ca_shape_fields as caf

skip_no_deps = pytest.mark.skipif(
    caf.cv2 is None or not caf._HAS_SKIMAGE,
    reason="requires opencv-python, scikit-image and scipy",
)


def test_bezier_fit_shapes():
    """A cubic Bezier fit returns four control points for any ordered point set."""
    pts = np.stack([np.linspace(0, 10, 20), np.linspace(0, 5, 20)], axis=1)
    ctrl = caf._fit_cubic_bezier(pts)
    assert ctrl.shape == (4, 2)


def test_straight_line_has_large_radius():
    """A straight skeleton has (near) infinite radius of curvature -> clamped to the max."""
    pts = np.stack([np.linspace(0, 100, 40), np.zeros(40)], axis=1)
    ctrl = caf._fit_cubic_bezier(pts)
    _, radius = caf._bezier_curvature(ctrl, num=32)
    assert radius.min() > 1e3  # essentially straight


def test_curved_line_has_finite_radius():
    """A clearly curved skeleton produces finite (small) radii where it bends."""
    t = np.linspace(0, np.pi, 40)
    pts = np.stack([50 + 40 * np.cos(t), 50 + 40 * np.sin(t)], axis=1)  # half circle, R=40
    ctrl = caf._fit_cubic_bezier(pts)
    _, radius = caf._bezier_curvature(ctrl, num=32)
    # the fitted radius should be on the order of the true arc radius, not the clamp ceiling
    assert radius.min() < 500


def test_ordered_skeleton_walk():
    """Greedy ordering returns every skeleton pixel exactly once."""
    skel = np.zeros((20, 20), dtype=bool)
    skel[10, 2:18] = True  # horizontal line
    ordered = caf._ordered_skeleton_points(skel)
    assert ordered.shape[0] == 16
    # consecutive points are adjacent along the line
    dx = np.abs(np.diff(ordered[:, 0]))
    assert dx.max() <= 1.5


@skip_no_deps
def test_process_polygon_crack():
    """A slender polygon yields a non-trivial skeleton and a high elongation ratio."""
    # a thin diagonal strip (elongated crack-like polygon), absolute px on a 256 canvas
    poly = np.array([[40, 40], [44, 44], [120, 120], [116, 124]], dtype=np.float64)
    out = caf._process_polygon(poly, imgsz=256)
    assert out is not None
    assert out["skeleton"].shape[1] == 2
    assert out["curve_pts"].shape == (32, 2)
    assert out["curve_radius"].shape == (32,)
    assert out["elongation"] > 1.0


@skip_no_deps
def test_process_polygon_degenerate_returns_none():
    """A near-zero-area polygon is rejected."""
    poly = np.array([[10, 10], [10, 10], [11, 11]], dtype=np.float64)
    out = caf._process_polygon(poly, imgsz=256)
    assert out is None


@skip_no_deps
def test_build_fields_for_label(tmp_path):
    """End-to-end: a polygon label file is converted to per-object morphology fields."""
    label = tmp_path / "img1.txt"
    label.write_text(
        "0 0.20 0.30 0.22 0.35 0.25 0.42 0.28 0.50 0.30 0.58 0.33 0.65\n"
        "2 0.10 0.10 0.40 0.10 0.40 0.40 0.10 0.40\n"
    )
    objs = caf.build_fields_for_label(str(label), imgsz=256)
    assert len(objs) == 2
    assert objs[0]["cls"] == 0 and objs[1]["cls"] == 2
    for o in objs:
        assert "skeleton" in o and "curve_radius" in o and "elongation" in o
