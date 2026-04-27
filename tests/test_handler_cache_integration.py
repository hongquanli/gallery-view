"""End-to-end: handler.cache_key + cache.save/load round-trip.

These tests catch drift between a handler's ``cache_key`` shape and what
``cache.save/load`` accepts — a class of bug that handler-only and
cache-only tests can both miss.
"""

import numpy as np
import pytest

from gallery_view import cache, mips


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "cache_mips"))
    return tmp_path / "cache_mips"


def _accumulate_full_stack(handler, acq, fov, channel):
    state = mips.new_axis_state()
    n = ny = nx = 0
    for slice_yx in handler.iter_z_slices(acq, fov, channel):
        if n == 0:
            ny, nx = slice_yx.shape
        mips.accumulate_axes(slice_yx, state)
        n += 1
    finalized = mips.finalize(state)
    return mips.axis_data_with_percentiles(finalized), (n, ny, nx)


def _roundtrip(handler, acq, fov, channel):
    src, ch_id = handler.cache_key(acq, fov, channel)
    channel_mips, shape_zyx = _accumulate_full_stack(handler, acq, fov, channel)
    cache.save(src, ch_id, channel_mips, shape_zyx)
    loaded, loaded_shape = cache.load(src, ch_id)
    assert loaded is not None
    assert loaded_shape == shape_zyx
    for ax in ("z", "y", "x"):
        np.testing.assert_array_equal(loaded[ax].mip, channel_mips[ax].mip)


def test_ome_tiff_handler_cache_roundtrip(make_ome_tiff_acq):
    from gallery_view.sources.ome_tiff import OmeTiffHandler
    folder = make_ome_tiff_acq(wavelengths=("488",), nz=4, ny=6, nx=7)
    handler = OmeTiffHandler()
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    _roundtrip(handler, acq, "0", acq.channels[0])


def test_single_tiff_handler_cache_roundtrip(make_single_tiff_acq):
    from gallery_view.sources.single_tiff import SingleTiffHandler
    folder = make_single_tiff_acq(wavelengths=("488",), nz=3, ny=5, nx=6)
    handler = SingleTiffHandler()
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    _roundtrip(handler, acq, "0", acq.channels[0])


def test_cache_keys_distinct_across_acquisitions(make_ome_tiff_acq):
    """Two acquisitions with the same channel but different paths must hash
    to different keys — otherwise drag-dropping the same data twice would
    overwrite the first cache entry."""
    from gallery_view.sources.ome_tiff import OmeTiffHandler
    folder_a = make_ome_tiff_acq(folder_name="25x_A1_2026-04-26_12-00-00.000000")
    folder_b = make_ome_tiff_acq(folder_name="25x_A2_2026-04-26_12-30-00.000000")
    handler = OmeTiffHandler()
    acq_a = handler.build(str(folder_a), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    acq_b = handler.build(str(folder_b), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    key_a = handler.cache_key(acq_a, "0", acq_a.channels[0])
    key_b = handler.cache_key(acq_b, "0", acq_b.channels[0])
    assert key_a != key_b
    # Hashed paths differ too.
    path_a = cache._cache_path(*key_a)
    path_b = cache._cache_path(*key_b)
    assert path_a != path_b


def test_cache_keys_distinct_across_timepoints(make_squid_single_tiff_acq):
    """Per-timepoint MIPs must hash to different cache files so they
    don't overwrite each other."""
    from gallery_view.sources.single_tiff import SingleTiffHandler
    handler = SingleTiffHandler()
    folder = make_squid_single_tiff_acq(nt=2)
    acq = handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    k_t0 = handler.cache_key(acq, acq.fovs[0], acq.channels[0], timepoint="0")
    k_t1 = handler.cache_key(acq, acq.fovs[0], acq.channels[0], timepoint="1")
    assert k_t0 != k_t1
    assert "/t0/" in k_t0[1]
    assert "/t1/" in k_t1[1]
