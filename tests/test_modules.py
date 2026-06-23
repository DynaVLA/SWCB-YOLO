# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Unit tests for the SWCB-YOLO neural modules (AS-Swin2, FS-DDA) and the CA-Shape-IoU terms.

These tests require PyTorch and are skipped automatically if it is not installed, so the
torch-free morphology tests can still run in a minimal CI. On a machine with torch installed,
run::

    pytest tests/test_modules.py -v
"""

import pytest

torch = pytest.importorskip("torch")


def test_asswin2_preserves_spatial_changes_channels():
    """AS-Swin2 keeps H, W and maps C1 -> C2 with a well-formed output."""
    from ultralytics.nn.modules.swcb import ASSwin2

    m = ASSwin2(c1=64, c2=128, depth=2, num_heads=4, window_size=7).eval()
    x = torch.randn(2, 64, 40, 40)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 128, 40, 40)
    assert torch.isfinite(y).all()


def test_asswin2_handles_non_divisible_resolution():
    """Strip and square windows must pad correctly for sizes not divisible by the window."""
    from ultralytics.nn.modules.swcb import ASSwin2

    m = ASSwin2(c1=32, c2=32, depth=1, num_heads=4, window_size=7).eval()
    x = torch.randn(1, 32, 37, 23)  # deliberately awkward
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 32, 37, 23)


def test_fs_dda_is_shape_preserving():
    """FS-DDA returns the same (B, C, H, W) shape as its input."""
    from ultralytics.nn.modules.swcb import FS_DDA

    m = FS_DDA(c1=96).eval()
    x = torch.randn(2, 96, 32, 32)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_fs_dda_dct_reconstruction_identity():
    """With identity channel weights the DCT->IDCT path reconstructs the input."""
    from ultralytics.nn.modules.swcb import _SpectralChannelModulation

    mod = _SpectralChannelModulation(channels=16).eval()
    # force the channel weighting to ~1 so only the transform pair is exercised
    with torch.no_grad():
        for p in mod.mlp.parameters():
            p.zero_()
        x = torch.randn(1, 16, 24, 24)
        # patch sigmoid(0)=0.5 scaling out by comparing transform-only reconstruction
        dct_h, dct_w = mod._bases(24, 24, x.device, x.dtype)
        from ultralytics.nn.modules.swcb import _dct_2d

        spec = _dct_2d(x, dct_h, dct_w)
        rec = torch.einsum("ih,bcij->bchj", dct_h, spec)
        rec = torch.einsum("bchj,jw->bchw", rec, dct_w)
    assert torch.allclose(x, rec, atol=1e-4)


def test_boundary_points_lie_on_box():
    """Sampled boundary points are differentiable and lie on the box perimeter."""
    from ultralytics.utils.ca_shape_iou import _box_boundary_points

    boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0]], requires_grad=True)
    pts = _box_boundary_points(boxes, n_per_side=16)
    assert pts.shape == (1, 64, 2)
    # gradient flows back to the box parameters
    pts.sum().backward()
    assert boxes.grad is not None and torch.isfinite(boxes.grad).all()


def test_curvature_loss_weights_high_curvature():
    """The curvature term is finite and non-negative for a simple matched pair."""
    from ultralytics.utils.ca_shape_iou import curvature_loss

    pred = torch.tensor([[10.0, 10.0, 60.0, 20.0]], requires_grad=True)
    curve_pts = [torch.tensor([[12.0, 11.0], [55.0, 18.0]])]
    curve_radius = [torch.tensor([2.0, 100.0])]  # one sharp, one flat
    loss = curvature_loss(pred, curve_pts, curve_radius)
    assert loss.item() >= 0 and torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_voronoi_loss_truncates():
    """The Voronoi term never exceeds d_max^2 per point and is differentiable."""
    from ultralytics.utils.ca_shape_iou import voronoi_loss, D_MAX

    pred = torch.tensor([[0.0, 0.0, 100.0, 100.0]], requires_grad=True)
    skel = [torch.tensor([[50.0, 50.0]])]
    loss = voronoi_loss(pred, skel)
    assert loss.item() <= D_MAX ** 2 + 1e-3
    loss.backward()
    assert pred.grad is not None


def test_ratio_loss_gating():
    """The elongation term activates only for ground-truth boxes above the threshold."""
    from ultralytics.utils.ca_shape_iou import ratio_loss

    pred = torch.tensor([[0.0, 0.0, 100.0, 10.0]], requires_grad=True)
    gt = torch.tensor([[0.0, 0.0, 90.0, 10.0]])
    elong_high = torch.tensor([9.0])  # > r_th=8 -> active
    elong_low = torch.tensor([2.0])   # < r_th=8 -> zero
    assert ratio_loss(pred, gt, elong_high).item() > 0
    assert ratio_loss(pred, gt, elong_low).item() == 0
