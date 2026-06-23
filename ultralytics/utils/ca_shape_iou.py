# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Differentiable morphological compensation terms for the CA-Shape-IoU loss.

This module implements the three training-only sub-penalties of Eq. (11)-(15) and
Algorithm 1 of the SWCB-YOLO paper, given the offline morphology fields produced by
``ultralytics/data/ca_shape_fields.py``:

  * ``L_Curve``    - curvature-weighted SmoothL1 on sampled box-boundary keypoints,
                     with the Bezier-derived adaptive weight ``w_i = 1 + beta * exp(-R_i / tau_c)``.
  * ``L_Voronoi``  - truncated squared distance from sampled predicted-box boundary points
                     to the ground-truth crack skeleton.
  * ``L_Ratio``    - elongation constraint, active only for strongly elongated targets
                     (gt aspect ratio > r_th), implemented with a smooth-L1 surrogate.

The total CA-Shape-IoU loss is ``L_Base + gamma * (L_Curve + L_Voronoi + L_Ratio)``, where the
base term is the CIoU alignment loss computed in ``BboxLoss``. All quantities derived from
polygon masks are precomputed offline, so these terms supervise the same four box parameters
as the base CIoU term and add no inference cost.

Hyperparameters follow the paper exactly: ``gamma = 0.5``, ``beta = 2``, ``tau_c = 5``,
``d_max = 10`` px, ``r_th = 8``, ``N = M = 16`` boundary samples.
"""

import torch
import torch.nn.functional as F

# paper hyperparameters (Algorithm 1)
GAMMA = 0.5      # morphological balance weight
BETA = 2.0       # curvature weight amplitude
TAU_C = 5.0      # curvature decay factor (distinct from attention temperature tau)
D_MAX = 10.0     # Voronoi distance truncation (px)
R_TH = 8.0       # elongation activation threshold
N_SAMPLE = 16    # boundary samples for the curvature term
M_SAMPLE = 16    # boundary samples for the Voronoi term


def _box_boundary_points(boxes, n_per_side):
    """Sample ``4 * n_per_side`` points uniformly along an axis-aligned box boundary.

    Args:
        boxes (torch.Tensor): (K, 4) boxes in xyxy format. Points are differentiable
            functions of the box corners, so gradients flow back to (cx, cy, w, h).
        n_per_side (int): samples per edge.

    Returns:
        torch.Tensor: (K, 4 * n_per_side, 2) boundary points.
    """
    x1, y1, x2, y2 = boxes.unbind(-1)  # each (K,)
    t = torch.linspace(0, 1, n_per_side, device=boxes.device, dtype=boxes.dtype)  # (n,)
    k = boxes.shape[0]
    t = t.view(1, -1).expand(k, -1)  # (K, n)

    def lerp(a, b):
        return a.view(-1, 1) * (1 - t) + b.view(-1, 1) * t  # (K, n)

    # top, right, bottom, left edges
    top_x, top_y = lerp(x1, x2), y1.view(-1, 1).expand(-1, n_per_side)
    right_x, right_y = x2.view(-1, 1).expand(-1, n_per_side), lerp(y1, y2)
    bot_x, bot_y = lerp(x2, x1), y2.view(-1, 1).expand(-1, n_per_side)
    left_x, left_y = x1.view(-1, 1).expand(-1, n_per_side), lerp(y2, y1)

    xs = torch.cat([top_x, right_x, bot_x, left_x], dim=1)  # (K, 4n)
    ys = torch.cat([top_y, right_y, bot_y, left_y], dim=1)  # (K, 4n)
    return torch.stack([xs, ys], dim=-1)  # (K, 4n, 2)


def curvature_loss(pred_boxes, gt_curve_pts, gt_curve_radius):
    """Curvature-weighted SmoothL1 between predicted boundary points and GT skeleton points.

    For each ground-truth curvature sample (with radius ``R_i``) the nearest predicted
    boundary point is matched, weighted by ``w_i = 1 + beta * exp(-R_i / tau_c)`` so that
    high-curvature regions receive up to ``1 + beta`` times the gradient (paper, Eq. (12)-(13)).

    Args:
        pred_boxes (torch.Tensor): (K, 4) predicted boxes (xyxy, px).
        gt_curve_pts (list[torch.Tensor]): per-box (32, 2) curvature sample points.
        gt_curve_radius (list[torch.Tensor]): per-box (32,) radii of curvature.

    Returns:
        torch.Tensor: scalar mean curvature loss.
    """
    if pred_boxes.numel() == 0:
        return pred_boxes.new_zeros(())
    pred_pts = _box_boundary_points(pred_boxes, N_SAMPLE)  # (K, 4N, 2)

    losses = []
    for i in range(pred_boxes.shape[0]):
        gpts = gt_curve_pts[i]
        if gpts is None or gpts.numel() == 0:
            continue
        radius = gt_curve_radius[i]
        w = 1.0 + BETA * torch.exp(-radius / TAU_C)  # (32,)
        # nearest predicted boundary point for each gt curvature sample
        d = torch.cdist(gpts, pred_pts[i])  # (32, 4N)
        idx = d.argmin(dim=1)  # (32,)
        matched = pred_pts[i][idx]  # (32, 2)
        sl1 = F.smooth_l1_loss(matched, gpts, reduction="none").sum(dim=-1)  # (32,)
        losses.append((w * sl1).mean())
    if not losses:
        return pred_boxes.new_zeros(())
    return torch.stack(losses).mean()


def voronoi_loss(pred_boxes, gt_skeletons):
    """Truncated squared distance from predicted boundary points to the GT crack skeleton.

    Implements ``L_Voronoi = (1/M) * sum min(d_j^2, d_max^2)`` (paper, Eq. (14)); the
    truncation prevents outlier boundary points from producing exploding early-training
    gradients.

    Args:
        pred_boxes (torch.Tensor): (K, 4) predicted boxes (xyxy, px).
        gt_skeletons (list[torch.Tensor]): per-box (s, 2) skeleton points.

    Returns:
        torch.Tensor: scalar mean Voronoi loss.
    """
    if pred_boxes.numel() == 0:
        return pred_boxes.new_zeros(())
    pred_pts = _box_boundary_points(pred_boxes, M_SAMPLE)  # (K, 4M, 2)

    losses = []
    for i in range(pred_boxes.shape[0]):
        skel = gt_skeletons[i]
        if skel is None or skel.numel() == 0:
            continue
        d = torch.cdist(pred_pts[i], skel)  # (4M, s)
        dmin = d.min(dim=1).values  # (4M,)
        d2 = torch.clamp(dmin ** 2, max=D_MAX ** 2)
        losses.append(d2.mean())
    if not losses:
        return pred_boxes.new_zeros(())
    return torch.stack(losses).mean()


def ratio_loss(pred_boxes, gt_boxes, gt_elong):
    """Elongation constraint, active only for strongly elongated ground-truth targets.

    For targets whose ground-truth aspect ratio exceeds ``r_th`` the term adds a smooth-L1
    surrogate of the relative width/height error; the gate depends only on the ground-truth
    box and is treated as a constant selector (paper, Eq. (15)).

    Args:
        pred_boxes (torch.Tensor): (K, 4) predicted boxes (xyxy, px).
        gt_boxes (torch.Tensor): (K, 4) ground-truth boxes (xyxy, px).
        gt_elong (torch.Tensor): (K,) ground-truth elongation max(w/h, h/w).

    Returns:
        torch.Tensor: scalar mean elongation loss.
    """
    if pred_boxes.numel() == 0:
        return pred_boxes.new_zeros(())
    pw = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1e-6)
    ph = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1e-6)
    gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-6)
    gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-6)

    gate = (gt_elong > R_TH).to(pred_boxes.dtype)  # (K,) non-differentiable selector
    if gate.sum() == 0:
        return pred_boxes.new_zeros(())

    rel_w = F.smooth_l1_loss(pw / gw, torch.ones_like(pw), reduction="none")
    rel_h = F.smooth_l1_loss(ph / gh, torch.ones_like(ph), reduction="none")
    term = (rel_w + rel_h) * gate
    return term.sum() / gate.sum().clamp(min=1.0)


def morphological_loss(pred_boxes, gt_boxes, fields):
    """Aggregate the three morphological terms for a batch of matched predictions.

    Args:
        pred_boxes (torch.Tensor): (K, 4) predicted boxes (xyxy, px).
        gt_boxes (torch.Tensor): (K, 4) ground-truth boxes (xyxy, px).
        fields (dict): batch morphology fields with keys ``skeleton``, ``curve_pts``,
            ``curve_radius`` and ``elongation`` (lists / tensors aligned to the K matches).

    Returns:
        torch.Tensor: scalar ``L_Morph = L_Curve + L_Voronoi + L_Ratio``.
    """
    l_curve = curvature_loss(pred_boxes, fields["curve_pts"], fields["curve_radius"])
    l_voronoi = voronoi_loss(pred_boxes, fields["skeleton"])
    l_ratio = ratio_loss(pred_boxes, gt_boxes, fields["elongation"])
    return l_curve + l_voronoi + l_ratio
