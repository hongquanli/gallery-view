"""StackTiffHandler unit tests."""

import numpy as np
import pytest

from gallery_view.sources.stack_tiff import StackTiffHandler


@pytest.fixture
def handler():
    return StackTiffHandler()


def test_detect_returns_true_for_stack_acq(handler, make_stack_tiff_acq):
    folder = make_stack_tiff_acq()
    assert handler.detect(str(folder)) is True


def test_detect_returns_false_for_other_formats(
    handler, make_ome_tiff_acq, make_single_tiff_acq
):
    assert handler.detect(str(make_ome_tiff_acq())) is False
    assert handler.detect(str(make_single_tiff_acq())) is False


def test_build_populates_acquisition_in_file_order(handler, make_stack_tiff_acq):
    folder = make_stack_tiff_acq(wavelengths=("730", "488", "405"))
    acq = handler.build(str(folder), {})
    assert acq is not None
    # File order is preserved — same convention as OmeTiffHandler.
    assert [c.wavelength for c in acq.channels] == ["730", "488", "405"]
    assert acq.fovs == ["current_0"]
    assert acq.timepoints == ["0"]


def test_read_shape_uses_distinct_z_levels(handler, make_stack_tiff_acq):
    folder = make_stack_tiff_acq(nz=4, ny=6, nx=7)
    acq = handler.build(str(folder), {})
    assert handler.read_shape(acq, acq.selected_fov) == (4, 6, 7)


def test_iter_z_slices_yields_zsorted_float32(handler, make_stack_tiff_acq):
    folder = make_stack_tiff_acq(wavelengths=("730", "488"), nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {})
    slices = list(handler.iter_z_slices(acq, acq.selected_fov, acq.channels[0]))
    assert len(slices) == 3
    assert slices[0].dtype == np.float32
    assert slices[0].shape == (4, 5)


def test_channel_data_matches_label_when_file_order_is_unsorted(
    handler, make_stack_tiff_acq
):
    """Regression: file order != wavelength-sorted order. Conftest's
    +1000-per-slot gradient distinguishes channels; verifies that asking
    for channel "405" returns the data written into the 405-slot rather
    than the slot wavelength sorting would have placed it in."""
    folder = make_stack_tiff_acq(
        wavelengths=("730", "488", "405"), nz=2, ny=3, nx=4
    )
    acq = handler.build(str(folder), {})
    by_wl = {c.wavelength: c for c in acq.channels}
    s730 = next(handler.iter_z_slices(acq, acq.selected_fov, by_wl["730"]))
    s488 = next(handler.iter_z_slices(acq, acq.selected_fov, by_wl["488"]))
    s405 = next(handler.iter_z_slices(acq, acq.selected_fov, by_wl["405"]))
    # Slot 0 (730) was written with channel_offset=0; slot 1 (488) with
    # +1000; slot 2 (405) with +2000.
    assert s730.min() < 1000
    assert 1000 <= s488.min() < 2000
    assert s405.min() >= 2000


def test_load_full_stack_shape(handler, make_stack_tiff_acq):
    folder = make_stack_tiff_acq(nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {})
    stack = handler.load_full_stack(acq, acq.selected_fov, acq.channels[0])
    assert stack.shape == (3, 4, 5)


def test_iter_full_channel_stacks_pairs_channel_with_correct_data(
    handler, make_stack_tiff_acq
):
    folder = make_stack_tiff_acq(
        wavelengths=("730", "488", "405"), nz=2, ny=3, nx=4
    )
    acq = handler.build(str(folder), {})
    pairs = list(handler.iter_full_channel_stacks(acq, acq.selected_fov))
    assert [c.wavelength for c, _ in pairs] == ["730", "488", "405"]
    # Channel-offset baked into _gradient_3d: channel c gets +1000*c.
    for c_i, (_, stack) in enumerate(pairs):
        assert stack.shape == (2, 3, 4)
        assert 1000 * c_i <= stack.min() < 1000 * (c_i + 1) or c_i == 0


def test_multi_fov_discovery(handler, make_stack_tiff_acq):
    folder = make_stack_tiff_acq(n_fovs=3)
    acq = handler.build(str(folder), {})
    assert acq.fovs == ["current_0", "current_1", "current_2"]


def test_cache_key_distinct_per_channel_and_fov(handler, make_stack_tiff_acq):
    folder = make_stack_tiff_acq(wavelengths=("730", "488"), n_fovs=2)
    acq = handler.build(str(folder), {})
    k_a = handler.cache_key(acq, "current_0", acq.channels[0])
    k_b = handler.cache_key(acq, "current_0", acq.channels[1])
    k_c = handler.cache_key(acq, "current_1", acq.channels[0])
    assert k_a != k_b
    assert k_a != k_c


def test_channel_yaml_extras_pulls_from_configurations_xml(
    handler, make_stack_tiff_acq
):
    folder = make_stack_tiff_acq(wavelengths=("730",))
    acq = handler.build(str(folder), {})
    extras = handler.channel_yaml_extras(acq, acq.channels[0])
    # Conftest writes ExposureTime=50.0+i, IlluminationIntensity=25.0+i.
    assert extras["exposure_ms"] == 50.0
    assert extras["intensity"] == 25.0


def test_channel_yaml_extras_returns_empty_when_xml_missing(
    handler, make_stack_tiff_acq
):
    folder = make_stack_tiff_acq(write_configurations_xml=False)
    acq = handler.build(str(folder), {})
    assert handler.channel_yaml_extras(acq, acq.channels[0]) == {}


# ── implicit-layout flavour (Dragonfly-style: no per-page channel/z meta) ──

def test_implicit_layout_resolves_channels_via_params_and_xml(
    handler, make_stack_tiff_acq
):
    folder = make_stack_tiff_acq(
        wavelengths=("405", "730"), nz=3, ny=4, nx=5,
        per_page_meta=False,
    )
    # build is called via scan.ingest in real use, which passes
    # parse_acquisition_params(folder). Mirror that here.
    from gallery_view.sources._squid_common import parse_acquisition_params
    acq = handler.build(str(folder), parse_acquisition_params(str(folder)) or {})
    assert acq is not None
    assert [c.wavelength for c in acq.channels] == ["405", "730"]
    assert handler.read_shape(acq, acq.selected_fov) == (3, 4, 5)


def test_implicit_layout_pairs_data_with_correct_channel(
    handler, make_stack_tiff_acq
):
    """Even without per-page channel tags the handler must serve channel-c
    requests with channel-c data. Same +1000-per-slot offset trick as the
    per_page_meta variant."""
    folder = make_stack_tiff_acq(
        wavelengths=("405", "730"), nz=2, ny=3, nx=4,
        per_page_meta=False,
    )
    from gallery_view.sources._squid_common import parse_acquisition_params
    acq = handler.build(str(folder), parse_acquisition_params(str(folder)) or {})
    by_wl = {c.wavelength: c for c in acq.channels}
    s_a = next(handler.iter_z_slices(acq, acq.selected_fov, by_wl["405"]))
    s_b = next(handler.iter_z_slices(acq, acq.selected_fov, by_wl["730"]))
    # Slot 0 (XML order) → 405, gradient offset 0; slot 1 → 730, offset +1000.
    assert s_a.min() < 1000 <= s_b.min()


def test_implicit_layout_iter_full_channel_stacks_pairs_correctly(
    handler, make_stack_tiff_acq
):
    """The 3D viewer (napari) goes through ``iter_full_channel_stacks``.
    For the legacy implicit path the page→channel mapping is arithmetic
    (``page = z * nc + c``); regress the channel/data pairing the same
    way the per_page_meta variant tests it."""
    folder = make_stack_tiff_acq(
        wavelengths=("405", "730"), nz=2, ny=3, nx=4,
        per_page_meta=False,
    )
    from gallery_view.sources._squid_common import parse_acquisition_params
    acq = handler.build(str(folder), parse_acquisition_params(str(folder)) or {})
    pairs = list(handler.iter_full_channel_stacks(acq, acq.selected_fov))
    assert [c.wavelength for c, _ in pairs] == ["405", "730"]
    # Slot 0 → 405, gradient offset 0; slot 1 → 730, offset +1000.
    (_, s_405), (_, s_730) = pairs
    assert s_405.shape == (2, 3, 4)
    assert s_405.min() < 1000 <= s_730.min()


def test_per_page_meta_skips_pages_with_partial_metadata(
    handler, make_stack_tiff_acq, tmp_path
):
    """If a page is missing ``z_level`` or ``channel_index``, drop it
    rather than treating ``None`` as a valid index. Build still
    succeeds for the well-formed pages."""
    import json
    import tifffile
    folder = tmp_path / "_2026-05-08_partial.000000"
    folder.mkdir()
    (folder / "acquisition parameters.json").write_text(
        json.dumps({"sensor_pixel_size_um": 6.5, "dz(um)": 1.0})
    )
    t_dir = folder / "0"
    t_dir.mkdir()
    with tifffile.TiffWriter(t_dir / "current_0_stack.tiff") as tw:
        # Two well-formed pages (z=0,1 of channel 0) and one corrupt page
        # missing z_level — the handler should ignore the corrupt one.
        tw.write(np.zeros((3, 4), dtype=np.uint16), description=json.dumps({
            "z_level": 0, "channel": "Fluorescence 488 nm Ex",
            "channel_index": 0, "region_id": "current", "fov": 0,
        }), contiguous=False)
        tw.write(np.ones((3, 4), dtype=np.uint16), description=json.dumps({
            "z_level": 1, "channel": "Fluorescence 488 nm Ex",
            "channel_index": 0, "region_id": "current", "fov": 0,
        }), contiguous=False)
        tw.write(np.zeros((3, 4), dtype=np.uint16), description=json.dumps({
            "channel": "Fluorescence 488 nm Ex",  # no z_level
            "channel_index": 0, "region_id": "current", "fov": 0,
        }), contiguous=False)
    acq = handler.build(str(folder), {"dz(um)": 1.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert [c.wavelength for c in acq.channels] == ["488"]
    # Corrupt page dropped → 2 z-slices, not 3.
    assert handler.read_shape(acq, acq.selected_fov) == (2, 3, 4)


def test_implicit_layout_returns_none_when_configurations_xml_missing(
    handler, make_stack_tiff_acq, tmp_path
):
    """Implicit path needs configurations.xml to map slots to channels.
    Without it (or with no Selected="true" fluorescence modes) build must
    refuse rather than return generic channel labels."""
    folder = make_stack_tiff_acq(
        wavelengths=("405", "730"), nz=2, ny=3, nx=4,
        per_page_meta=False, write_configurations_xml=False,
    )
    from gallery_view.sources._squid_common import parse_acquisition_params
    acq = handler.build(str(folder), parse_acquisition_params(str(folder)) or {})
    assert acq is None


def test_implicit_layout_returns_none_when_xml_count_mismatches(
    handler, make_stack_tiff_acq
):
    """If configurations.xml claims a different fluorescence-mode count than
    ``total_pages / Nz`` derives, refuse to guess — better to fail loudly
    than to silently mislabel channels."""
    # Stack file has 2 channels (per the fixture's wavelengths). Misreport
    # Nz so total_pages / Nz != 2: e.g. params says Nz=4, real Nz=2 and
    # nc_real=2 → total=4, derived nc=1, but xml lists 2 fluorescence modes.
    folder = make_stack_tiff_acq(
        wavelengths=("405", "730"), nz=2, ny=3, nx=4,
        per_page_meta=False,
    )
    # Override params so Nz is wrong:
    import json as _json
    pp = folder / "acquisition parameters.json"
    p = _json.loads(pp.read_text())
    p["Nz"] = 4
    pp.write_text(_json.dumps(p))
    acq = handler.build(str(folder), p)
    assert acq is None
