"""Cache supports region mosaics (single 'z' axis, region-keyed channel_id)."""

import numpy as np
import pytest

from gallery_view import cache
from gallery_view.types import AxisMip


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))
    return tmp_path / "mips"


def test_region_cache_roundtrip_single_z_axis():
    """A region mosaic saved with only 'z' loads back correctly."""
    mip = np.arange(24, dtype=np.float32).reshape(4, 6)
    data = {"z": AxisMip(mip=mip, p1=1.0, p999=22.0)}
    cache.save("/src/acq", "region:A5/t0/Fluorescence_488_nm_Ex", data)
    loaded, _ = cache.load("/src/acq", "region:A5/t0/Fluorescence_488_nm_Ex")
    assert loaded is not None
    assert set(loaded.keys()) == {"z"}
    np.testing.assert_array_equal(loaded["z"].mip, mip)
    assert loaded["z"].p1 == 1.0
    assert loaded["z"].p999 == 22.0


def test_fov_cache_still_requires_or_yields_all_axes():
    """Full FOV caches (all three axes saved) round-trip with all axes."""
    data = {
        ax: AxisMip(mip=np.zeros((3, 3), dtype=np.float32), p1=0.0, p999=1.0)
        for ax in ("z", "y", "x")
    }
    cache.save("/src/acq", "fov0/t0/Fluorescence_488_nm_Ex", data, (3, 3, 3))
    loaded, shape = cache.load("/src/acq", "fov0/t0/Fluorescence_488_nm_Ex")
    assert loaded is not None
    assert set(loaded.keys()) == {"z", "y", "x"}
    assert shape == (3, 3, 3)


def test_load_returns_none_when_no_axes_present(tmp_path, monkeypatch):
    """A .npz file with the right version but zero axes shouldn't masquerade
    as a valid cache."""
    import numpy as _np
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))
    (tmp_path / "mips").mkdir()
    path = cache._cache_path("/src/acq", "weird")
    _np.savez_compressed(path, version=_np.int32(cache.CACHE_VERSION))
    assert cache.load("/src/acq", "weird") == (None, None)


def test_region_and_fov_cache_keys_dont_collide(make_squid_single_tiff_acq):
    """region:<r>/... and fov<r>_<f>/... must hash to different files even
    when r happens to match."""
    fov_path = cache._cache_path("/src/acq", "fov0_0/t0/ch")
    region_path = cache._cache_path("/src/acq", "region:0/t0/ch")
    assert fov_path != region_path


def test_single_tiff_cache_key_region(make_squid_single_tiff_acq):
    """SingleTiffHandler.cache_key_region produces a region: prefix key
    that differs by region, timepoint, and channel."""
    from gallery_view.sources.single_tiff import SingleTiffHandler

    folder = make_squid_single_tiff_acq(regions=2, fovs_per_region=2, nt=2)
    handler = SingleTiffHandler()
    acq = handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    src_a, id_a = handler.cache_key_region(acq, "0", acq.channels[0], timepoint="0")
    src_b, id_b = handler.cache_key_region(acq, "1", acq.channels[0], timepoint="0")
    src_c, id_c = handler.cache_key_region(acq, "0", acq.channels[0], timepoint="1")
    src_d, id_d = handler.cache_key_region(acq, "0", acq.channels[1], timepoint="0")
    assert src_a == src_b == src_c == src_d == acq.path
    assert id_a.startswith("region:0/")
    assert id_b.startswith("region:1/")
    assert id_a != id_b != id_c != id_d


def test_ome_tiff_cache_key_region_not_implemented(make_ome_tiff_acq):
    from gallery_view.sources.ome_tiff import OmeTiffHandler

    handler = OmeTiffHandler()
    acq = handler.build(
        str(make_ome_tiff_acq()), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    with pytest.raises(NotImplementedError):
        handler.cache_key_region(acq, "0", acq.channels[0])


def test_stack_tiff_cache_key_region_not_implemented(make_stack_tiff_acq):
    from gallery_view.sources.stack_tiff import StackTiffHandler

    handler = StackTiffHandler()
    acq = handler.build(
        str(make_stack_tiff_acq()), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    with pytest.raises(NotImplementedError):
        handler.cache_key_region(acq, "current", acq.channels[0])
