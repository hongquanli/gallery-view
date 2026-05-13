"""Pure-function tests of stitch_region.

Synthetic grids place tiny FOVs at known stage coordinates so we can assert
exact pixel layout, mean-blending, gap handling, and downsample math.
"""

import numpy as np

from gallery_view.stitch import FovCoord, stitch_region


def _flat(value: float, ny: int = 4, nx: int = 4) -> np.ndarray:
    return np.full((ny, nx), value, dtype=np.float32)


def test_empty_input_returns_none():
    assert stitch_region({}, [], pixel_um=1.0) is None


def test_single_fov_unchanged_at_factor_one():
    """One FOV at the origin with no downsample: mosaic equals that FOV exactly.

    With nx=ny=4 and target_longest_px=1024, factor=1 so no downsampling."""
    mip = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = stitch_region(
        {"0_0": mip},
        [FovCoord("0_0", x_mm=0.0, y_mm=0.0)],
        pixel_um=1.0,
        target_longest_px=1024,
    )
    assert result is not None
    # Mosaic shape matches the single FOV (no padding when only one tile).
    assert result.mip.shape == (4, 4)
    np.testing.assert_array_equal(result.mip, mip)


def test_two_fovs_side_by_side_lay_out_horizontally():
    """Two 4x4 FOVs centered at x=0 and x=4 um (pixel_um=1) place adjacent."""
    left = _flat(10.0)
    right = _flat(20.0)
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=4e-3, y_mm=0.0),  # 4 um right
    ]
    result = stitch_region(
        {"0_0": left, "0_1": right}, coords,
        pixel_um=1.0, target_longest_px=1024,
    )
    assert result is not None
    # Canvas width = 4 + 4 = 8 px; height = 4 px.
    assert result.mip.shape == (4, 8)
    np.testing.assert_array_equal(result.mip[:, :4], left)
    np.testing.assert_array_equal(result.mip[:, 4:], right)


def test_overlapping_fovs_mean_blend():
    """Two FOVs offset by half-width overlap by 50%; overlap pixels are mean."""
    a = _flat(10.0)   # 4x4
    b = _flat(20.0)   # 4x4
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=2e-3, y_mm=0.0),  # offset by 2 um (half width)
    ]
    result = stitch_region(
        {"0_0": a, "0_1": b}, coords,
        pixel_um=1.0, target_longest_px=1024,
    )
    assert result is not None
    # Canvas: 4 + 2 = 6 px wide. Cols 0-1 are A only, 2-3 are A+B blend
    # (mean=15), 4-5 are B only.
    assert result.mip.shape == (4, 6)
    np.testing.assert_array_equal(result.mip[:, 0:2], 10.0)
    np.testing.assert_allclose(result.mip[:, 2:4], 15.0)
    np.testing.assert_array_equal(result.mip[:, 4:6], 20.0)


def test_missing_fov_leaves_black_gap():
    """A FovCoord with no matching entry in fov_mips leaves that rect black."""
    a = _flat(10.0)
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=4e-3, y_mm=0.0),
    ]
    result = stitch_region(
        {"0_0": a}, coords,    # 0_1 missing
        pixel_um=1.0, target_longest_px=1024,
    )
    assert result is not None
    assert result.mip.shape == (4, 8)
    np.testing.assert_array_equal(result.mip[:, :4], a)
    np.testing.assert_array_equal(result.mip[:, 4:], 0.0)


def test_heterogeneous_shapes_returns_none():
    """If FOV MIPs aren't the same shape we bail rather than guess."""
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=4e-3, y_mm=0.0),
    ]
    result = stitch_region(
        {"0_0": _flat(1.0, 4, 4), "0_1": _flat(1.0, 4, 5)},
        coords, pixel_um=1.0,
    )
    assert result is None


def test_integer_downsample_factor():
    """Big FOVs trigger integer downsample to fit target_longest_px.

    Two 8x8 FOVs placed side-by-side -> canvas 8x16. With target=8, factor=2
    -> mosaic shape (4, 8)."""
    a = _flat(10.0, 8, 8)
    b = _flat(20.0, 8, 8)
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=8e-3, y_mm=0.0),
    ]
    result = stitch_region(
        {"0_0": a, "0_1": b}, coords,
        pixel_um=1.0, target_longest_px=8,
    )
    assert result is not None
    assert result.mip.shape == (4, 8)
    np.testing.assert_array_equal(result.mip[:, :4], 10.0)
    np.testing.assert_array_equal(result.mip[:, 4:], 20.0)


def test_negative_stage_coords_shift_to_origin():
    """Negative x_mm/y_mm shifts to a non-negative canvas origin."""
    a = _flat(10.0)
    coords = [FovCoord("0_0", x_mm=-1.0, y_mm=-1.0)]
    result = stitch_region(
        {"0_0": a}, coords, pixel_um=1.0, target_longest_px=1024,
    )
    assert result is not None
    np.testing.assert_array_equal(result.mip, a)


def test_flip_y_inverts_vertical_placement():
    """flip_y=True negates the y axis: y=0 ends up at the bottom of the canvas."""
    top_in_stage = _flat(10.0)
    bottom_in_stage = _flat(20.0)
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=0.0, y_mm=4e-3),
    ]
    no_flip = stitch_region(
        {"0_0": top_in_stage, "0_1": bottom_in_stage},
        coords, pixel_um=1.0, target_longest_px=1024, flip_y=False,
    )
    with_flip = stitch_region(
        {"0_0": top_in_stage, "0_1": bottom_in_stage},
        coords, pixel_um=1.0, target_longest_px=1024, flip_y=True,
    )
    assert no_flip is not None and with_flip is not None
    # Without flip, y=0 (top_in_stage, value 10) sits at the top of the canvas.
    # With flip, the y=0 tile sits at the bottom instead.
    assert no_flip.mip[0, 0] == 10.0
    assert with_flip.mip[0, 0] == 20.0


def test_percentiles_ignore_black_gaps():
    """Auto-contrast computed on covered pixels only — black gaps don't pull
    p1 toward 0."""
    a = _flat(100.0)
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=8e-3, y_mm=0.0),  # 8 um gap from origin (FOV is 4 px)
    ]
    result = stitch_region(
        {"0_0": a}, coords,  # 0_1 missing -> big black gap
        pixel_um=1.0, target_longest_px=1024,
    )
    assert result is not None
    # p1 should reflect the covered pixels' percentile (all 100s), not 0.
    assert result.p1 == 100.0
    assert result.p999 == 100.0


def test_factor_larger_than_tile_size_does_not_crash():
    """When target_longest_px forces factor > min(ny, nx), the stitcher
    should fall back to averaging the whole tile, not crash in reshape."""
    # Two 4x4 tiles tiled into an 80x80 canvas (large enough that
    # target_longest_px=4 forces factor=20 > 4).
    a = _flat(10.0, 4, 4)
    b = _flat(20.0, 4, 4)
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=76e-3, y_mm=0.0),  # 76 um apart so canvas is wide
    ]
    result = stitch_region(
        {"0_0": a, "0_1": b}, coords,
        pixel_um=1.0, target_longest_px=4,
    )
    assert result is not None
    # Each tile averaged to 1px; canvas downsampled to ~4 px wide.
    assert result.mip.shape[0] >= 1
    assert result.mip.shape[1] >= 1
    # Both averaged values should appear somewhere in the mosaic.
    flat = set(result.mip.flatten().tolist())
    assert 10.0 in flat or any(abs(v - 10.0) < 1e-6 for v in flat)
    assert 20.0 in flat or any(abs(v - 20.0) < 1e-6 for v in flat)


def test_zero_pixel_um_returns_none():
    """pixel_um <= 0 is invalid; return None instead of dividing by zero."""
    a = _flat(1.0)
    coords = [FovCoord("0_0", x_mm=0.0, y_mm=0.0)]
    assert stitch_region({"0_0": a}, coords, pixel_um=0.0) is None
    assert stitch_region({"0_0": a}, coords, pixel_um=-1.0) is None
