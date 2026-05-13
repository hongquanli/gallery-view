"""SingleTiffHandler populates Acquisition.regions / selected_region."""

from gallery_view.sources.single_tiff import SingleTiffHandler


def test_single_region_squid_folder_has_one_region(make_squid_single_tiff_acq):
    """A folder with FOVs all under region '0' yields regions == ['0']."""
    folder = make_squid_single_tiff_acq(regions=1, fovs_per_region=2)
    acq = SingleTiffHandler().build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    assert acq is not None
    assert acq.regions == ["0"]
    assert acq.selected_region == "0"


def test_multi_region_numeric_regions_sort_numerically(make_squid_single_tiff_acq):
    """Numeric region ids sort as ints, not strings (so '10' comes after '2')."""
    folder = make_squid_single_tiff_acq(regions=12, fovs_per_region=1)
    acq = SingleTiffHandler().build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    assert acq.regions == [str(i) for i in range(12)]


def test_legacy_folder_gets_single_region(make_single_tiff_acq):
    """Legacy current_<fov>_<z>_<channel>.tiff folders have regions == ['0']."""
    folder = make_single_tiff_acq()
    acq = SingleTiffHandler().build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    assert acq.regions == ["0"]


def test_well_name_regions_order_matches_fovs_order(make_squid_single_tiff_acq):
    """Region order is derived from acq.fovs's order, so they never drift
    apart when the source format uses well-name regions."""
    folder = make_squid_single_tiff_acq(
        regions=3, fovs_per_region=2, with_well_prefix=True, well="A1",
    )
    from gallery_view.sources.single_tiff import SingleTiffHandler
    acq = SingleTiffHandler().build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    # The fixture writes prefix=A1_; the inner region ids are still "0", "1", "2".
    # Region order must match the order they appear in acq.fovs.
    seen = []
    for f in acq.fovs:
        r = f.split("_", 1)[0]
        if r not in seen:
            seen.append(r)
    assert acq.regions == seen


def test_fixture_writes_coordinates_csv(make_squid_single_tiff_acq):
    """The fixture writes coordinates.csv with one row per (region, fov, z)
    when ``write_coords=True``."""
    import csv

    folder = make_squid_single_tiff_acq(
        regions=2, fovs_per_region=3, nz=2,
        write_coords=True,
    )
    coords_path = folder / "0" / "coordinates.csv"
    assert coords_path.exists(), f"coordinates.csv missing under {folder}"

    with coords_path.open() as f:
        rows = list(csv.DictReader(f))
    # 2 regions * 3 fovs * 2 z = 12 rows
    assert len(rows) == 12
    assert {row["region"] for row in rows} == {"0", "1"}
    # FOVs within a region are 0..2; values come back as strings.
    assert {row["fov"] for row in rows} == {"0", "1", "2"}
