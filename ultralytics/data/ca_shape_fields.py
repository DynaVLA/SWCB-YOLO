# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Offline morphological field generation for the CA-Shape-IoU loss.

The CA-Shape-IoU loss is supervised by three quantities derived from the ground-truth
*polygon* masks rather than from the axis-aligned box:

  1. a medial-axis crack skeleton (used by the Voronoi distance term),
  2. a per-point radius of curvature obtained by cubic-Bezier fitting of the skeleton
     (used by the curvature-weight term), and
  3. an elongation (aspect-ratio) descriptor (used by the elongation term).

Following the paper, all of these are precomputed *offline*, once, before training, so they
add no inference cost. This script converts a directory of YOLO-format polygon labels into a
compact per-image ``.npz`` cache that the training loss reads.

Polygon label format (one object per line), identical to the Ultralytics segmentation
format::

    <class> x1 y1 x2 y2 ... xk yk        # all coordinates normalized to [0, 1]

Usage::

    python -m ultralytics.data.ca_shape_fields \
        --labels /path/to/labels/train \
        --out    /path/to/ca_fields/train \
        --imgsz  1280

The loss falls back gracefully to the plain CIoU base term when no cache is present, so the
detector still trains (without the morphological gains) on datasets that provide only boxes.
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a hard runtime dependency for this script
    cv2 = None

# medial-axis skeletonization (Lee 1994 thinning), provided by scikit-image
try:
    from skimage.morphology import skeletonize
    from scipy.ndimage import distance_transform_edt
    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    _HAS_SKIMAGE = False


# ---------------------------------------------------------------------------------------
# Bezier curvature
# ---------------------------------------------------------------------------------------
def _fit_cubic_bezier(points):
    """Least-squares fit of a single cubic Bezier (4 control points) to an ordered point set.

    Args:
        points (np.ndarray): (n, 2) ordered skeleton points.

    Returns:
        np.ndarray: (4, 2) control points P0..P3.
    """
    n = len(points)
    if n < 4:
        # not enough points: degenerate to endpoints repeated
        p0, p3 = points[0], points[-1]
        return np.stack([p0, p0, p3, p3], axis=0).astype(np.float64)

    # chord-length parameterization t in [0, 1]
    d = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=1))
    t = np.concatenate([[0.0], np.cumsum(d)])
    if t[-1] > 0:
        t = t / t[-1]

    # Bernstein basis for a cubic
    b0 = (1 - t) ** 3
    b1 = 3 * (1 - t) ** 2 * t
    b2 = 3 * (1 - t) * t ** 2
    b3 = t ** 3
    basis = np.stack([b0, b1, b2, b3], axis=1)  # (n, 4)

    # solve basis @ P = points (least squares) for the 4 control points
    ctrl, *_ = np.linalg.lstsq(basis, points, rcond=None)
    return ctrl.astype(np.float64)


def _bezier_curvature(ctrl, num=32):
    """Sample the analytic radius of curvature along a cubic Bezier.

    Args:
        ctrl (np.ndarray): (4, 2) control points.
        num (int): number of samples along t in [0, 1].

    Returns:
        tuple(np.ndarray, np.ndarray): sampled points (num, 2) and their radius of curvature (num,).
    """
    p0, p1, p2, p3 = ctrl
    t = np.linspace(0.0, 1.0, num).reshape(-1, 1)

    # first and second derivatives of a cubic Bezier
    d1 = 3 * (1 - t) ** 2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t ** 2 * (p3 - p2)
    d2 = 6 * (1 - t) * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)

    pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3

    # curvature kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2); R = 1 / kappa
    num_k = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den_k = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5 + 1e-9
    kappa = num_k / den_k
    radius = 1.0 / (kappa + 1e-9)
    radius = np.clip(radius, 0.0, 1e4)
    return pts, radius


# ---------------------------------------------------------------------------------------
# Skeleton + Voronoi distance field
# ---------------------------------------------------------------------------------------
def _ordered_skeleton_points(skel):
    """Return skeleton pixel coordinates ordered by a nearest-neighbour walk.

    A simple greedy ordering is sufficient for short, mostly non-branching crack skeletons
    and keeps the dependency footprint small.
    """
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(pts) <= 2:
        return pts

    remaining = pts.tolist()
    ordered = [remaining.pop(0)]
    while remaining:
        last = np.array(ordered[-1])
        rem = np.array(remaining)
        idx = int(np.argmin(((rem - last) ** 2).sum(axis=1)))
        ordered.append(remaining.pop(idx))
    return np.array(ordered, dtype=np.float64)


def _process_polygon(poly_xy, imgsz):
    """Build skeleton points, curvature samples and Voronoi distance map for one polygon.

    Args:
        poly_xy (np.ndarray): (k, 2) polygon vertices in absolute pixel coordinates.
        imgsz (int): square canvas size used to rasterize the mask.

    Returns:
        dict | None: per-object morphology fields, or None if the polygon is degenerate.
    """
    if cv2 is None or not _HAS_SKIMAGE:
        raise RuntimeError(
            "ca_shape_fields requires opencv-python, scikit-image and scipy. "
            "Install them with: pip install opencv-python scikit-image scipy"
        )

    mask = np.zeros((imgsz, imgsz), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_xy.astype(np.int32)], 1)
    if mask.sum() < 4:
        return None

    skel = skeletonize(mask.astype(bool))
    skel_pts = _ordered_skeleton_points(skel)
    if len(skel_pts) == 0:
        # fall back to the polygon centroid as a single skeleton point
        skel_pts = poly_xy.mean(axis=0, keepdims=True)

    # Bezier curvature along the skeleton
    ctrl = _fit_cubic_bezier(skel_pts)
    curve_pts, curve_radius = _bezier_curvature(ctrl, num=32)

    # Voronoi distance field = Euclidean distance to the nearest skeleton pixel,
    # evaluated inside the object's bounding region. We store the skeleton points and let
    # the loss compute point-to-skeleton distances on the fly (cheap, few points).
    x0, y0 = poly_xy.min(axis=0)
    x1, y1 = poly_xy.max(axis=0)
    w = max(float(x1 - x0), 1.0)
    h = max(float(y1 - y0), 1.0)
    elong = max(w / h, h / w)

    return {
        "skeleton": skel_pts.astype(np.float32),       # (s, 2) absolute px
        "curve_pts": curve_pts.astype(np.float32),     # (32, 2) absolute px
        "curve_radius": curve_radius.astype(np.float32),  # (32,)
        "bbox": np.array([x0, y0, x1, y1], dtype=np.float32),
        "elongation": np.float32(elong),
    }


def _read_polygon_label(line, imgsz):
    """Parse one polygon label line into (class_id, absolute polygon vertices)."""
    parts = line.strip().split()
    if len(parts) < 7:  # class + at least 3 vertices
        return None
    cls = int(float(parts[0]))
    coords = np.array(parts[1:], dtype=np.float64)
    if coords.size % 2 != 0:
        coords = coords[:-1]
    poly = coords.reshape(-1, 2) * imgsz  # normalized -> absolute px on an imgsz canvas
    return cls, poly


def build_fields_for_label(label_path, imgsz):
    """Build the list of per-object morphology fields for a single label file."""
    objects = []
    text = Path(label_path).read_text().splitlines()
    for line in text:
        parsed = _read_polygon_label(line, imgsz)
        if parsed is None:
            continue
        cls, poly = parsed
        fields = _process_polygon(poly, imgsz)
        if fields is None:
            continue
        fields["cls"] = np.int64(cls)
        objects.append(fields)
    return objects


def build_dataset_fields(labels_dir, out_dir, imgsz=1280):
    """Precompute morphology caches for every polygon label file in a directory.

    Each ``<stem>.txt`` produces a ``<stem>.npz`` containing an object array of per-object
    dictionaries. The cache is keyed by the label stem so the training dataloader can look it
    up by image name.
    """
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        print(f"[ca_shape_fields] no .txt labels found in {labels_dir}")
        return

    n_ok = 0
    for lf in label_files:
        objects = build_fields_for_label(lf, imgsz)
        np.savez_compressed(out_dir / f"{lf.stem}.npz", objects=np.array(objects, dtype=object))
        n_ok += 1
        if n_ok % 200 == 0:
            print(f"[ca_shape_fields] processed {n_ok}/{len(label_files)} labels")
    print(f"[ca_shape_fields] done: wrote {n_ok} caches to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Precompute CA-Shape-IoU morphology fields.")
    parser.add_argument("--labels", type=str, required=True, help="directory of YOLO polygon .txt labels")
    parser.add_argument("--out", type=str, required=True, help="output directory for .npz caches")
    parser.add_argument("--imgsz", type=int, default=1280, help="rasterization canvas size")
    args = parser.parse_args()
    build_dataset_fields(args.labels, args.out, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
