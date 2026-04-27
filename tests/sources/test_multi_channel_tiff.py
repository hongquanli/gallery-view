"""MultiChannelTiffHandler unit tests."""

import numpy as np
import pytest

from gallery_view.sources.multi_channel_tiff import MultiChannelTiffHandler


@pytest.fixture
def handler():
    return MultiChannelTiffHandler()


def test_detect_returns_true_for_multi_channel(handler, make_multi_channel_tiff_acq):
    folder = make_multi_channel_tiff_acq()
    assert handler.detect(str(folder)) is True


def test_detect_returns_false_for_other_formats(
    handler, make_ome_tiff_acq
):
    assert handler.detect(str(make_ome_tiff_acq())) is False


def test_build_populates_acquisition(handler, make_multi_channel_tiff_acq):
    folder = make_multi_channel_tiff_acq(wavelengths=("488", "638"))
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert [c.wavelength for c in acq.channels] == ["488", "638"]
    assert acq.fovs == ["0"]


def test_read_shape_matches_written(handler, make_multi_channel_tiff_acq):
    folder = make_multi_channel_tiff_acq(nz=5, ny=8, nx=10)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert handler.read_shape(acq, "0") == (5, 8, 10)


def test_iter_z_slices_count_and_order(handler, make_multi_channel_tiff_acq):
    folder = make_multi_channel_tiff_acq(nz=4, ny=4, nx=4)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    slices = list(handler.iter_z_slices(acq, "0", acq.channels[0]))
    assert len(slices) == 4
    # The synthetic stack has value = z*100 + y*10 + x; z=0 has min, z=3 max
    assert slices[0].max() < slices[3].max()


def test_cache_key_is_stable(handler, make_multi_channel_tiff_acq):
    folder = make_multi_channel_tiff_acq()
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert (
        handler.cache_key(acq, "0", acq.channels[0])
        == handler.cache_key(acq, "0", acq.channels[0])
    )


def test_load_full_stack_shape(handler, make_multi_channel_tiff_acq):
    folder = make_multi_channel_tiff_acq(nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    stack = handler.load_full_stack(acq, "0", acq.channels[0])
    assert stack.shape == (3, 4, 5)
