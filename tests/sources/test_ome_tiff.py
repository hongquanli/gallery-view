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
    handler, make_multi_channel_tiff_acq, make_single_channel_tiff_acq
):
    folder = make_multi_channel_tiff_acq()
    assert handler.detect(str(folder)) is False
    _, folders = make_single_channel_tiff_acq()
    assert handler.detect(str(folders[0])) is False


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
