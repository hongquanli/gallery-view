"""SingleTiffHandler unit tests."""

import numpy as np
import pytest

from gallery_view.sources.single_tiff import (
    SingleTiffHandler,
    parse_legacy_filename,
    parse_squid_filename,
)


@pytest.fixture
def handler():
    return SingleTiffHandler()


def test_detect_returns_true_for_single_tiff_acq(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq()
    assert handler.detect(str(folder)) is True


def test_detect_returns_false_for_other_formats(
    handler, make_ome_tiff_acq
):
    assert handler.detect(str(make_ome_tiff_acq())) is False


def test_cache_key_includes_timepoint(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq(wavelengths=("488",))
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    _, ch_id = handler.cache_key(acq, "0", acq.channels[0], timepoint="2")
    assert "/t2/" in ch_id


def test_build_populates_acquisition(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq(wavelengths=("488", "638"))
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert [c.wavelength for c in acq.channels] == ["488", "638"]
    # Legacy folders synthesise region="0" so the composite FOV id is "0_0".
    assert acq.fovs == ["0_0"]


def test_read_shape_matches_written(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq(nz=5, ny=8, nx=10)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert handler.read_shape(acq, "0") == (5, 8, 10)


def test_iter_z_slices_count_and_order(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq(nz=4, ny=4, nx=4)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    slices = list(handler.iter_z_slices(acq, "0", acq.channels[0]))
    assert len(slices) == 4
    # The synthetic stack has value = z*100 + y*10 + x; z=0 has min, z=3 max
    assert slices[0].max() < slices[3].max()


def test_cache_key_is_stable(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq()
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert (
        handler.cache_key(acq, "0", acq.channels[0])
        == handler.cache_key(acq, "0", acq.channels[0])
    )


def test_load_full_stack_shape(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq(nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    stack = handler.load_full_stack(acq, "0", acq.channels[0])
    assert stack.shape == (3, 4, 5)


# ── filename parsers ────────────────────────────────────────────────────────


def test_parse_squid_filename_numeric_region():
    p = parse_squid_filename("0_1_2_Fluorescence_488_nm_Ex.tiff")
    assert p == {"region": "0", "fov": "1", "z": "2",
                 "channel": "Fluorescence_488_nm_Ex"}


def test_parse_squid_filename_well_as_region():
    """Squid puts the well in the region slot when wells are used."""
    p = parse_squid_filename("A1_0_2_Fluorescence_488_nm_Ex.tiff")
    assert p == {"region": "A1", "fov": "0", "z": "2",
                 "channel": "Fluorescence_488_nm_Ex"}


def test_parse_squid_filename_rejects_garbage():
    assert parse_squid_filename("not_a_match.tiff") is None
    assert parse_squid_filename("only_two_components.tiff") is None


def test_parse_legacy_filename_basic():
    p = parse_legacy_filename("current_3_5_Fluorescence_488_nm_Ex.tiff")
    assert p == {"fov": "3", "z": "5", "channel": "Fluorescence_488_nm_Ex"}


def test_parse_legacy_filename_rejects_squid():
    assert parse_legacy_filename("0_1_2_Fluorescence_488_nm_Ex.tiff") is None


def test_squid_regex_also_matches_legacy_filenames():
    """Important: squid regex's [^_]+ region accepts 'current' too. Detection
    code must try legacy first, then squid — never the other way around."""
    p = parse_squid_filename("current_0_0_Fluorescence_488_nm_Ex.tiff")
    assert p is not None
    assert p["region"] == "current"  # documents the overlap


# ── squid layout detection + multi-FOV discovery ────────────────────────────


def test_detect_squid_layout(handler, make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq()
    assert handler.detect(str(folder)) is True


def test_detect_squid_layout_with_well_prefix(handler, make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(with_well_prefix=True)
    assert handler.detect(str(folder)) is True


def test_legacy_layout_still_detects(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq()
    assert handler.detect(str(folder)) is True


def test_build_discovers_composite_fovs(handler, make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(regions=2, fovs_per_region=2)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert acq.fovs == ["0_0", "0_1", "1_0", "1_1"]
    assert acq.selected_fov == "0_0"


def test_legacy_build_uses_zero_region(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq()
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert acq.fovs == ["0_0"]


def test_build_discovers_multiple_timepoints(handler, make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(nt=3)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert acq.timepoints == ["0", "1", "2"]
    assert acq.selected_timepoint == "0"


def test_legacy_acq_has_single_timepoint(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq()
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert acq.timepoints == ["0"]
    assert acq.selected_timepoint == "0"
