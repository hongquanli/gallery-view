"""Cache I/O: .npz save/load round-trip + .lut.json sidecar."""

from pathlib import Path

import numpy as np
import pytest

from gallery_view import cache
from gallery_view.types import AxisMip


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    """Redirect the module's cache dir to a tmpdir for every test."""
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))
    return tmp_path / "mips"


def _make_axis_data():
    return {
        "z": AxisMip(mip=np.arange(20, dtype=np.float32).reshape(4, 5),
                    p1=1.0, p999=18.0),
        "y": AxisMip(mip=np.arange(15, dtype=np.float32).reshape(3, 5),
                    p1=0.5, p999=14.0),
        "x": AxisMip(mip=np.arange(12, dtype=np.float32).reshape(3, 4),
                    p1=0.5, p999=11.0),
    }


def test_load_returns_none_when_missing():
    mips, shape = cache.load("/nonexistent/path", "wl_488")
    assert mips is None and shape is None


def test_save_then_load_roundtrip():
    data = _make_axis_data()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))
    loaded, shape = cache.load("/some/src", "wl_488")
    assert shape == (10, 4, 5)
    for ax in ("z", "y", "x"):
        np.testing.assert_array_equal(loaded[ax].mip, data[ax].mip)
        assert loaded[ax].p1 == pytest.approx(data[ax].p1)
        assert loaded[ax].p999 == pytest.approx(data[ax].p999)


def test_load_returns_none_when_version_mismatch(isolated_cache_dir):
    data = _make_axis_data()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))
    # Corrupt: rewrite with a wrong version field
    npz_path = cache._cache_path("/some/src", "wl_488")
    payload = dict(np.load(npz_path))
    payload["version"] = np.int32(cache.CACHE_VERSION + 99)
    np.savez_compressed(npz_path, **payload)
    loaded, shape = cache.load("/some/src", "wl_488")
    assert loaded is None and shape is None


def test_save_lut_only_writes_sidecar_without_touching_npz():
    data = _make_axis_data()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))
    npz_path = cache._cache_path("/some/src", "wl_488")
    npz_mtime_before = npz_path.stat().st_mtime
    cache.save_lut_only(
        "/some/src",
        "wl_488",
        {"z": (data["z"].mip, 5.0, 50.0), "y": (data["y"].mip, 1.0, 10.0)},
    )
    assert npz_path.stat().st_mtime == npz_mtime_before
    sidecar = cache._lut_override_path("/some/src", "wl_488")
    assert sidecar.exists()


def test_lut_override_applies_on_top_of_cached_defaults():
    data = _make_axis_data()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))
    cache.save_lut_only(
        "/some/src",
        "wl_488",
        {"z": (data["z"].mip, 100.0, 200.0)},
    )
    loaded, _ = cache.load("/some/src", "wl_488")
    assert loaded["z"].p1 == 100.0
    assert loaded["z"].p999 == 200.0
    # Other axes keep their cached percentiles
    assert loaded["y"].p1 == pytest.approx(data["y"].p1)


def test_save_preserves_pre_existing_lut_override():
    """A fresh MIP recompute must NOT delete the user's saved LUT — the
    override is a user preference about contrast bounds, independent of
    the MIP arrays it applies to. Recomputing MIPs (e.g. cache version
    bump) shouldn't nuke saved contrast settings."""
    data = _make_axis_data()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))
    cache.save_lut_only("/some/src", "wl_488",
                        {"z": (data["z"].mip, 100.0, 200.0)})
    sidecar = cache._lut_override_path("/some/src", "wl_488")
    assert sidecar.exists()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))  # fresh compute
    # Sidecar must survive; load() applies it on top of the new percentiles.
    assert sidecar.exists()
    loaded, _ = cache.load("/some/src", "wl_488")
    assert loaded["z"].p1 == 100.0
    assert loaded["z"].p999 == 200.0


def test_clear_all_removes_cache_dir(isolated_cache_dir):
    data = _make_axis_data()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))
    assert isolated_cache_dir.exists()
    cache.clear_all()
    assert not isolated_cache_dir.exists()
