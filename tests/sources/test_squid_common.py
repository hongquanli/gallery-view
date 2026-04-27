"""Direct unit tests for the squid metadata helpers."""

import json
from pathlib import Path

import yaml

from gallery_view.sources import _squid_common as common
from gallery_view.types import Channel


def _write(folder: Path, params=None, channels=None):
    folder.mkdir(parents=True, exist_ok=True)
    if params is not None:
        (folder / "acquisition parameters.json").write_text(json.dumps(params))
    if channels is not None:
        (folder / "acquisition_channels.yaml").write_text(
            yaml.safe_dump({"channels": channels})
        )


# ── parse_acquisition_params ──


def test_parse_params_returns_dict(tmp_path):
    _write(tmp_path, params={"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert common.parse_acquisition_params(str(tmp_path)) == {
        "dz(um)": 2.0,
        "sensor_pixel_size_um": 6.5,
    }


def test_parse_params_missing_returns_none(tmp_path):
    assert common.parse_acquisition_params(str(tmp_path)) is None


def test_parse_params_malformed_returns_none(tmp_path):
    (tmp_path / "acquisition parameters.json").write_text("{not json")
    assert common.parse_acquisition_params(str(tmp_path)) is None


# ── parse_acquisition_channels_yaml ──


def test_channels_yaml_extracts_wavelength_underscored(tmp_path):
    _write(tmp_path, channels=[
        {"name": "Fluorescence_488_nm_Ex", "enabled": True},
    ])
    channels = common.parse_acquisition_channels_yaml(str(tmp_path))
    assert len(channels) == 1
    assert channels[0].wavelength == "488"


def test_channels_yaml_extracts_wavelength_spaced(tmp_path):
    _write(tmp_path, channels=[
        {"name": "Fluorescence 488 nm Ex", "enabled": True},
    ])
    channels = common.parse_acquisition_channels_yaml(str(tmp_path))
    assert len(channels) == 1
    assert channels[0].wavelength == "488"


def test_channels_yaml_skips_disabled(tmp_path):
    _write(tmp_path, channels=[
        {"name": "Fluorescence_405_nm_Ex", "enabled": False},
        {"name": "Fluorescence_488_nm_Ex", "enabled": True},
    ])
    channels = common.parse_acquisition_channels_yaml(str(tmp_path))
    assert [c.wavelength for c in channels] == ["488"]


def test_channels_yaml_sorts_by_wavelength(tmp_path):
    _write(tmp_path, channels=[
        {"name": "Fluorescence_638_nm_Ex", "enabled": True},
        {"name": "Fluorescence_488_nm_Ex", "enabled": True},
        {"name": "Fluorescence_405_nm_Ex", "enabled": True},
    ])
    channels = common.parse_acquisition_channels_yaml(str(tmp_path))
    assert [c.wavelength for c in channels] == ["405", "488", "638"]


def test_channels_yaml_missing_returns_empty(tmp_path):
    assert common.parse_acquisition_channels_yaml(str(tmp_path)) == []


def test_channels_yaml_malformed_returns_empty(tmp_path):
    (tmp_path / "acquisition_channels.yaml").write_text(":::not yaml:::\nfoo: [")
    assert common.parse_acquisition_channels_yaml(str(tmp_path)) == []


# ── channel_extras_from_yaml ──


def test_channel_extras_returns_exposure_and_intensity(tmp_path):
    _write(tmp_path, channels=[
        {
            "name": "Fluorescence_488_nm_Ex",
            "enabled": True,
            "camera_settings": {"exposure_time_ms": 75.0},
            "illumination_settings": {"intensity": 30.0},
        },
    ])
    extras = common.channel_extras_from_yaml(
        str(tmp_path), Channel(name="Fluorescence_488_nm_Ex", wavelength="488")
    )
    assert extras == {"exposure_ms": 75.0, "intensity": 30.0}


def test_channel_extras_handles_space_for_underscore(tmp_path):
    """squid sometimes writes ``Fluorescence 488 nm Ex`` in YAML even when
    file paths use underscores; the lookup should match either way."""
    _write(tmp_path, channels=[
        {
            "name": "Fluorescence 488 nm Ex",
            "enabled": True,
            "camera_settings": {"exposure_time_ms": 50.0},
            "illumination_settings": {"intensity": 10.0},
        },
    ])
    extras = common.channel_extras_from_yaml(
        str(tmp_path), Channel(name="Fluorescence_488_nm_Ex", wavelength="488")
    )
    assert extras["exposure_ms"] == 50.0


def test_channel_extras_returns_empty_when_missing(tmp_path):
    assert (
        common.channel_extras_from_yaml(
            str(tmp_path),
            Channel(name="Fluorescence_488_nm_Ex", wavelength="488"),
        )
        == {}
    )


# ── display_name_for + parse_timestamp + parse_mag ──


def test_display_name_strips_timestamp():
    assert (
        common.display_name_for("25x_A1_2026-04-26_12-00-00.000000")
        == "25x_A1"
    )


def test_display_name_passthrough_when_no_timestamp():
    assert common.display_name_for("not_a_timestamp_folder") == "not_a_timestamp_folder"


def test_parse_timestamp_returns_mmdd_hhmm():
    assert (
        common.parse_timestamp("25x_A1_2026-04-26_12-30-00.000000")
        == ("04-26", "12:30")
    )


def test_parse_timestamp_none_when_missing():
    assert common.parse_timestamp("no timestamp here") is None


def test_parse_mag_extracts_int():
    assert common.parse_mag("25x_A1_2026-04-26_12-00-00.000000") == 25


def test_parse_mag_none_when_no_prefix():
    assert common.parse_mag("A1_no_mag") is None


# ── parse_mag_well_wl ──


def test_parse_mag_well_wl_happy_path():
    assert common.parse_mag_well_wl(
        "25x_C3_488_LR1000_2026-04-26_12-00-00.000000"
    ) == (25, "C3", "488")


def test_parse_mag_well_wl_rejects_unknown_wavelength():
    """Wavelengths outside KNOWN_WAVELENGTHS are filtered out — they likely
    aren't actually channels (could be FOV indices, lookup IDs, etc.)."""
    assert (
        common.parse_mag_well_wl(
            "25x_C3_999_LR1000_2026-04-26_12-00-00.000000"
        )
        is None
    )


def test_parse_mag_well_wl_rejects_missing_well():
    assert (
        common.parse_mag_well_wl(
            "25x__488_LR1000_2026-04-26_12-00-00.000000"
        )
        is None
    )


def test_parse_mag_well_wl_rejects_missing_timestamp():
    assert common.parse_mag_well_wl("25x_C3_488_LR1000") is None
