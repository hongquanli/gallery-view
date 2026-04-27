"""MIP math: accumulators, finalize, percentile, RGBA."""

import numpy as np
import pytest

from gallery_view import mips


def _gradient_stack(nz=4, ny=5, nx=6):
    """Predictable 3-D test stack: value = z*100 + y*10 + x."""
    z, y, x = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    return (z * 100 + y * 10 + x).astype(np.float32)


def test_new_axis_state_is_empty():
    state = mips.new_axis_state()
    assert state["z"] is None
    assert state["y_strips"] == []
    assert state["x_strips"] == []


def test_accumulate_one_slice_initialises_z():
    state = mips.new_axis_state()
    img = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mips.accumulate_axes(img, state)
    assert np.array_equal(state["z"], img)
    assert len(state["y_strips"]) == 1
    assert len(state["x_strips"]) == 1


def test_accumulate_z_mip_matches_numpy_max():
    stack = _gradient_stack()
    state = mips.new_axis_state()
    for z in range(stack.shape[0]):
        mips.accumulate_axes(stack[z], state)
    finalized = mips.finalize(state)
    np.testing.assert_array_equal(finalized["z"], stack.max(axis=0))


def test_finalize_y_and_x_match_numpy_max():
    stack = _gradient_stack()
    state = mips.new_axis_state()
    for z in range(stack.shape[0]):
        mips.accumulate_axes(stack[z], state)
    finalized = mips.finalize(state)
    np.testing.assert_array_equal(finalized["y"], stack.max(axis=1))
    np.testing.assert_array_equal(finalized["x"], stack.max(axis=2).T)


def test_finalize_returns_none_when_no_slices():
    assert mips.finalize(mips.new_axis_state()) is None


def test_axis_data_with_percentiles_returns_axismip_per_axis():
    stack = _gradient_stack()
    state = mips.new_axis_state()
    for z in range(stack.shape[0]):
        mips.accumulate_axes(stack[z], state)
    finalized = mips.finalize(state)
    out = mips.axis_data_with_percentiles(finalized)
    assert set(out.keys()) == {"z", "y", "x"}
    for ax_mip in out.values():
        assert ax_mip.mip.dtype == np.float32
        assert 0 <= ax_mip.p1 <= ax_mip.p999


def test_mip_to_rgba_clips_below_p1_and_above_p999():
    img = np.array([[0.0, 50.0, 200.0]], dtype=np.float32)
    rgba = mips.mip_to_rgba(img, p1=50.0, p999=200.0, color_rgb=(255, 0, 0))
    assert rgba.shape == (1, 3, 4)
    assert rgba.dtype == np.uint8
    # value 0 -> below p1 -> normalized to 0 -> red channel 0
    assert rgba[0, 0, 0] == 0
    # value 200 -> at p999 -> normalized to 1 -> red channel 255
    assert rgba[0, 2, 0] == 255
    # alpha always 255
    assert (rgba[..., 3] == 255).all()


def test_mip_to_rgba_returns_zeros_when_p1_equals_p999():
    img = np.array([[5.0, 5.0]], dtype=np.float32)
    rgba = mips.mip_to_rgba(img, p1=5.0, p999=5.0, color_rgb=(255, 0, 0))
    assert (rgba[..., :3] == 0).all()
