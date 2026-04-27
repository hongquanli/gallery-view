"""OmeTiffHandler unit tests."""

import numpy as np
import pytest

from gallery_view.sources.ome_tiff import OmeTiffHandler, _build_page_map


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


# ── _build_page_map ───────────────────────────────────────────────────


def _ome_xml_with_planes(planes: list[tuple[int, int, int]]) -> str:
    """Wrap an explicit list of (T, Z, C) tuples into a minimal OME XML
    string; useful for testing storage-vs-logical orderings without
    actually writing a TIFF."""
    plane_tags = "".join(
        f'<Plane TheT="{t}" TheZ="{z}" TheC="{c}" />'
        for t, z, c in planes
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0">'
        f'<Pixels>{plane_tags}</Pixels>'
        '</Image></OME>'
    )


def test_build_page_map_returns_none_when_no_planes():
    assert _build_page_map(None) is None
    assert _build_page_map("") is None
    assert _build_page_map("<?xml version='1.0'?><OME/>") is None


def test_build_page_map_returns_none_for_malformed_xml():
    assert _build_page_map("<not xml") is None


def test_build_page_map_z_major_storage():
    """squid's typical ZCYX path: page = z*nc + c (channels-fast within Z)."""
    nz, nc = 3, 2
    planes = [(0, z, c) for z in range(nz) for c in range(nc)]
    pm = _build_page_map(_ome_xml_with_planes(planes))
    # Page 0 = (z=0, c=0); page 1 = (z=0, c=1); page 2 = (z=1, c=0); …
    assert pm[(0, 0, 0)] == 0
    assert pm[(0, 0, 1)] == 1
    assert pm[(0, 1, 0)] == 2
    assert pm[(0, 2, 1)] == 5


def test_iter_full_channel_stacks_shares_buffer_across_channels(
    handler, make_ome_tiff_acq
):
    """All channel stacks must share one underlying numpy buffer, otherwise
    we'd hold N copies of a multi-GB OME-TIFF in RAM (the original 30-GB
    blowup on a 4-channel XLight V3 file). Use ``np.shares_memory`` to
    confirm slices view the same parent."""
    folder = make_ome_tiff_acq(wavelengths=("488", "561", "638"), nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    pairs = list(handler.iter_full_channel_stacks(acq, "0"))
    assert len(pairs) == 3
    # First channel's stack must share storage with second and third.
    _, s0 = pairs[0]
    _, s1 = pairs[1]
    _, s2 = pairs[2]
    assert np.shares_memory(s0, s1)
    assert np.shares_memory(s0, s2)


def test_iter_full_channel_stacks_yields_correct_channels(
    handler, make_ome_tiff_acq
):
    folder = make_ome_tiff_acq(wavelengths=("488", "638"), nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    pairs = list(handler.iter_full_channel_stacks(acq, "0"))
    assert [c.wavelength for c, _ in pairs] == ["488", "638"]
    # Each per-channel slice has shape (nz, ny, nx)
    for _, stack in pairs:
        assert stack.shape == (3, 4, 5)


def test_build_page_map_c_major_storage_xlight_pattern():
    """XLight V3 pattern: each channel's full Z stack is contiguous, even
    though the file declares ``DimensionOrder='XYCZT'``. Pages 0..nz-1
    are c=0, then nz..2*nz-1 are c=1, etc. (i.e., page = c*nz + z)."""
    nz, nc = 5, 3
    planes = [(0, z, c) for c in range(nc) for z in range(nz)]
    pm = _build_page_map(_ome_xml_with_planes(planes))
    assert pm[(0, 0, 0)] == 0
    assert pm[(0, 1, 0)] == 1
    assert pm[(0, 0, 1)] == nz       # start of c=1 block
    assert pm[(0, 4, 1)] == 2 * nz - 1
    assert pm[(0, 0, 2)] == 2 * nz   # start of c=2 block
