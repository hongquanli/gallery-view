"""SingleChannelTiffHandler unit tests."""

import numpy as np
import pytest

from gallery_view.sources.single_channel_tiff import SingleChannelTiffHandler


@pytest.fixture
def handler():
    return SingleChannelTiffHandler()


def test_detect_returns_true_for_single_channel_folder(
    handler, make_single_channel_tiff_acq
):
    _, folders = make_single_channel_tiff_acq(wavelengths=("488",))
    assert handler.detect(str(folders[0])) is True


def test_detect_returns_false_for_other_formats(
    handler, make_ome_tiff_acq, make_multi_channel_tiff_acq
):
    assert handler.detect(str(make_ome_tiff_acq())) is False
    assert handler.detect(str(make_multi_channel_tiff_acq())) is False


def test_build_populates_one_channel(handler, make_single_channel_tiff_acq):
    _, folders = make_single_channel_tiff_acq(wavelengths=("488",))
    acq = handler.build(str(folders[0]),
                        {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert len(acq.channels) == 1
    assert acq.channels[0].wavelength == "488"
    assert acq.fovs == ["0"]


def test_iter_z_slices_count(handler, make_single_channel_tiff_acq):
    _, folders = make_single_channel_tiff_acq(wavelengths=("561",), nz=3)
    acq = handler.build(str(folders[0]),
                        {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    slices = list(handler.iter_z_slices(acq, "0", acq.channels[0]))
    assert len(slices) == 3
    assert slices[0].dtype == np.float32


def test_cache_key_is_stable_per_channel_path(
    handler, make_single_channel_tiff_acq
):
    _, folders = make_single_channel_tiff_acq(wavelengths=("488",))
    acq = handler.build(str(folders[0]),
                        {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    k1 = handler.cache_key(acq, "0", acq.channels[0])
    k2 = handler.cache_key(acq, "0", acq.channels[0])
    assert k1 == k2


def test_load_full_stack_shape(handler, make_single_channel_tiff_acq):
    _, folders = make_single_channel_tiff_acq(wavelengths=("488",), nz=3, ny=4, nx=5)
    acq = handler.build(str(folders[0]),
                        {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    stack = handler.load_full_stack(acq, "0", acq.channels[0])
    assert stack.shape == (3, 4, 5)
