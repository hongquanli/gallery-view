"""SingleTiffHandler._load_coords reads coordinates.csv into
acq.extra['coords_by_region'] keyed by region, dedup'd to one row per
(region, fov) (z=0)."""

from gallery_view.sources.single_tiff import SingleTiffHandler


def test_load_coords_populates_extra(make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(
        regions=2, fovs_per_region=3, nz=2, write_coords=True,
    )
    handler = SingleTiffHandler()
    acq = handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    coords = handler._load_coords(acq)
    assert coords is not None
    assert acq.extra["coords_by_region"] is coords

    # 2 regions, 3 FOVs each — deduplicated across z.
    assert set(coords.keys()) == {"0", "1"}
    assert len(coords["0"]) == 3
    assert len(coords["1"]) == 3

    # Composite fov ids matching acq.fovs.
    fov_ids_r0 = {c.fov for c in coords["0"]}
    assert fov_ids_r0 == {"0_0", "0_1", "0_2"}

    # FOV 0 of region 0 is at (0, 0); fixture step is 800x600 um.
    c00 = next(c for c in coords["0"] if c.fov == "0_0")
    assert c00.x_mm == 0.0
    assert c00.y_mm == 0.0
    c02 = next(c for c in coords["0"] if c.fov == "0_2")
    assert c02.x_mm == 1.6  # 2 * 800 um = 1.6 mm


def test_load_coords_missing_file_returns_none(make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(
        regions=1, fovs_per_region=1, write_coords=False,
    )
    handler = SingleTiffHandler()
    acq = handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    assert handler._load_coords(acq) is None
    assert "coords_by_region" not in acq.extra


def test_load_coords_malformed_csv_returns_none(make_squid_single_tiff_acq, tmp_path):
    """Missing required columns -> None, no exception."""
    folder = make_squid_single_tiff_acq(
        regions=1, fovs_per_region=1, write_coords=True,
    )
    bad = folder / "0" / "coordinates.csv"
    bad.write_text("foo,bar\n1,2\n")  # overwrite with garbage
    handler = SingleTiffHandler()
    acq = handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    assert handler._load_coords(acq) is None


def test_load_coords_caches_result(make_squid_single_tiff_acq):
    """Second call returns the cached dict from acq.extra."""
    folder = make_squid_single_tiff_acq(
        regions=1, fovs_per_region=2, write_coords=True,
    )
    handler = SingleTiffHandler()
    acq = handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    first = handler._load_coords(acq)
    second = handler._load_coords(acq)
    assert first is second  # identity, not just equality
