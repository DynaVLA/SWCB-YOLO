import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "ultralytics" / "utils" / "ca_shape_iou.py"
SPEC = importlib.util.spec_from_file_location("ca_shape_iou", MODULE_PATH)
ca_shape_iou = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ca_shape_iou)


def _sample_fields(device="cpu"):
    return {
        "skeleton": [torch.tensor([[5.0, 5.0], [6.0, 5.0]], device=device)],
        "curve_pts": [torch.tensor([[5.0, 5.0], [6.0, 5.0]], device=device)],
        "curve_radius": [torch.tensor([4.0, 8.0], device=device)],
        "elongation": torch.tensor([10.0], device=device),
    }


def test_normalized_terms_are_finite_and_voronoi_is_bounded():
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]], requires_grad=True)
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    fields = _sample_fields()

    curve = ca_shape_iou.curvature_loss(pred, gt, fields["curve_pts"], fields["curve_radius"])
    voronoi = ca_shape_iou.voronoi_loss(pred, fields["skeleton"])
    ratio = ca_shape_iou.ratio_loss(pred, gt, fields["elongation"])

    assert torch.isfinite(torch.stack([curve, voronoi, ratio])).all()
    assert 0.0 <= voronoi.item() <= 1.0


def test_detached_nearest_assignment_preserves_box_gradients():
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]], requires_grad=True)
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    fields = _sample_fields()

    loss = ca_shape_iou.curvature_loss(pred, gt, fields["curve_pts"], fields["curve_radius"])
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_per_term_weights_match_explicit_weighted_sum():
    pred = torch.tensor([[0.0, 0.0, 12.0, 8.0]], requires_grad=True)
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    fields = _sample_fields()

    curve = ca_shape_iou.curvature_loss(pred, gt, fields["curve_pts"], fields["curve_radius"])
    voronoi = ca_shape_iou.voronoi_loss(pred, fields["skeleton"])
    ratio = ca_shape_iou.ratio_loss(pred, gt, fields["elongation"])
    combined = ca_shape_iou.morphological_loss(
        pred,
        gt,
        fields,
        lambda_curve=0.5,
        lambda_voronoi=2.0,
        lambda_ratio=1.5,
    )

    expected = 0.5 * curve + 2.0 * voronoi + 1.5 * ratio
    assert torch.allclose(combined, expected)
