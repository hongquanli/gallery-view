"""OmeTiffHandler unit tests."""

import numpy as np
import pytest

from gallery_view.sources.ome_tiff import OmeTiffHandler


@pytest.fixture
def handler():
    return OmeTiffHandler()


def test_detect_returns_true_for_ome_tiff_acq(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq()
    assert handler.detect(str(folder)) is True


def test_detect_returns_false_for_other_formats(
    handler, make_single_tiff_acq
):
    folder = make_single_tiff_acq()
    assert handler.detect(str(folder)) is False


def test_build_populates_acquisition(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq(wavelengths=("488", "638"), nz=5)
    params = {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    acq = handler.build(str(folder), params)
    assert acq is not None
    assert [c.wavelength for c in acq.channels] == ["488", "638"]
    assert acq.handler is handler
    assert acq.fovs == ["0"]


def test_read_shape_matches_written(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq(nz=6, ny=12, nx=14)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert handler.read_shape(acq, "0") == (6, 12, 14)


def test_iter_z_slices_count_and_dtype(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq(nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    slices = list(handler.iter_z_slices(acq, "0", acq.channels[0]))
    assert len(slices) == 3
    assert slices[0].dtype == np.float32
    assert slices[0].shape == (4, 5)


def test_iter_z_slices_accepts_timepoint_kwarg(handler, make_ome_tiff_acq):
    """OME-TIFF has no <t>/ subdir but the protocol method takes the
    kwarg. Pass an arbitrary value and confirm slices come back unchanged."""
    folder = make_ome_tiff_acq(nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    slices = list(handler.iter_z_slices(acq, "0", acq.channels[0], timepoint="7"))
    assert len(slices) == 3


def test_cache_key_includes_timepoint(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq(wavelengths=("488",))
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    _, ch_id = handler.cache_key(acq, "0", acq.channels[0], timepoint="3")
    assert "/t3/" in ch_id


def test_cache_key_is_stable_and_distinct_per_channel(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq(wavelengths=("488", "561"))
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    k1 = handler.cache_key(acq, "0", acq.channels[0])
    k2 = handler.cache_key(acq, "0", acq.channels[0])
    k3 = handler.cache_key(acq, "0", acq.channels[1])
    assert k1 == k2
    assert k1 != k3


def test_load_full_stack_shape(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq(nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    stack = handler.load_full_stack(acq, "0", acq.channels[0])
    assert stack.shape == (3, 4, 5)


def test_channel_yaml_extras_returns_exposure_and_intensity(
    handler, make_ome_tiff_acq
):
    folder = make_ome_tiff_acq(wavelengths=("488",))
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    extras = handler.channel_yaml_extras(acq, acq.channels[0])
    assert extras["exposure_ms"] == 100.0
    assert extras["intensity"] == 25.0


def test_iter_full_channel_stacks_yields_independent_buffers(
    handler, make_ome_tiff_acq
):
    """Each channel stack must be its own contiguous buffer, not a slice
    into a shared parent. Otherwise napari layers all pin a single
    multi-GB OME-TIFF buffer for the lifetime of any one layer, and the
    memory footprint doesn't drop when the user closes the 3D viewer."""
    folder = make_ome_tiff_acq(wavelengths=("488", "561", "638"), nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    pairs = list(handler.iter_full_channel_stacks(acq, "0"))
    assert len(pairs) == 3
    _, s0 = pairs[0]
    _, s1 = pairs[1]
    _, s2 = pairs[2]
    assert not np.may_share_memory(s0, s1)
    assert not np.may_share_memory(s0, s2)
    # Each stack should own its data (no parent base).
    for _, s in pairs:
        assert s.base is None or s.base.base is None


def test_iter_full_channel_stacks_yields_correct_channels(
    handler, make_ome_tiff_acq
):
    folder = make_ome_tiff_acq(wavelengths=("488", "638"), nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    pairs = list(handler.iter_full_channel_stacks(acq, "0"))
    assert [c.wavelength for c, _ in pairs] == ["488", "638"]
    # Each per-channel slice has shape (nz, ny, nx)
    for _, stack in pairs:
        assert stack.shape == (3, 4, 5)


def test_channel_data_matches_label_when_yaml_order_is_unsorted(
    handler, make_ome_tiff_acq
):
    """Regression: yaml-order ≠ wavelength-sorted order. The conftest's
    gradient encodes the channel slot as a +1000-per-slot offset, so we can
    verify that a channel named "638" returns the data written into the
    638-slot, not whichever slot wavelength-sorted order would have placed
    it in. Catches the bug where channels were sorted by wavelength but the
    OME-TIFF page index still followed yaml order, silently swapping data."""
    folder = make_ome_tiff_acq(wavelengths=("638", "488"), nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    by_wl = {c.wavelength: c for c in acq.channels}
    s638 = next(handler.iter_z_slices(acq, "0", by_wl["638"]))
    s488 = next(handler.iter_z_slices(acq, "0", by_wl["488"]))
    # Slot 0 (638nm) was written with offset 0; slot 1 (488nm) with +1000.
    assert s638.min() < 1000 <= s488.min()
