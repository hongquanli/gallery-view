"""LUT-override JSON edge cases."""

import pytest

from gallery_view import cache


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))
    return tmp_path / "mips"


def test_load_lut_override_returns_none_when_missing():
    assert cache._load_lut_override("/nope", "wl_488") is None


def test_load_lut_override_returns_none_for_malformed_json(isolated_cache_dir):
    isolated_cache_dir.mkdir(parents=True)
    p = cache._lut_override_path("/src", "wl_488")
    p.write_text("{not json")
    assert cache._load_lut_override("/src", "wl_488") is None


def test_load_lut_override_returns_none_when_keys_missing(isolated_cache_dir):
    isolated_cache_dir.mkdir(parents=True)
    p = cache._lut_override_path("/src", "wl_488")
    p.write_text('{"z": {"p1": 1.0}}')  # no p999
    assert cache._load_lut_override("/src", "wl_488") is None


def test_load_lut_override_partial_axes_ok(isolated_cache_dir):
    cache.save_lut_only(
        "/src",
        "wl_488",
        {"z": (None, 5.0, 50.0)},  # only Z
    )
    out = cache._load_lut_override("/src", "wl_488")
    assert out == {"z": (5.0, 50.0)}
