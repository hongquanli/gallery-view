"""Shared parsers for squid metadata files. SQUID_TESTED_AGAINST: master @ 2026-04-26."""

import json
import os
import re

import yaml

from ..types import Channel

KNOWN_WAVELENGTHS = frozenset({"405", "488", "561", "638", "730"})


def parse_acquisition_params(folder: str) -> dict | None:
    """Read ``<folder>/acquisition parameters.json`` or return None."""
    p = os.path.join(folder, "acquisition parameters.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def parse_acquisition_channels_yaml(folder: str) -> list[Channel]:
    """Read ``<folder>/acquisition_channels.yaml`` -> list of Channel.

    Returns ``[]`` if the file is absent or unparseable. Channels marked
    ``enabled: false`` are excluded. Output is sorted by wavelength.
    """
    yaml_path = os.path.join(folder, "acquisition_channels.yaml")
    if not os.path.exists(yaml_path):
        return []
    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return []
    channels: list[Channel] = []
    for ch in config.get("channels", []):
        if not ch.get("enabled", True):
            continue
        name = ch.get("name", "")
        if not name:
            continue
        wl_match = re.search(r"(\d+)\s*nm", name)
        wl = wl_match.group(1) if wl_match else "unknown"
        channels.append(Channel(name=name, wavelength=wl))
    channels.sort(
        key=lambda c: int(c.wavelength) if c.wavelength.isdigit() else 999
    )
    return channels


def channel_extras_from_yaml(
    folder: str, channel: Channel
) -> dict:
    """Look up exposure_ms / intensity for one channel in
    ``acquisition_channels.yaml``. Returns ``{}`` if not found."""
    yaml_path = os.path.join(folder, "acquisition_channels.yaml")
    if not os.path.exists(yaml_path):
        return {}
    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    for ch in config.get("channels", []):
        if ch.get("name") == channel.name or ch.get("name") == channel.name.replace("_", " "):
            return {
                "exposure_ms": ch.get("camera_settings", {}).get("exposure_time_ms"),
                "intensity": ch.get("illumination_settings", {}).get("intensity"),
            }
    return {}


def display_name_for(folder_name: str) -> str:
    """A short human label for a squid acquisition folder.

    Default: the folder name with the trailing timestamp stripped.
    """
    m = re.match(
        r"(.+?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+)$", folder_name
    )
    return m.group(1) if m else folder_name


def parse_timestamp(folder_name: str) -> tuple[str, str] | None:
    """Return ``("MM-DD", "HH:MM")`` parsed from the folder-name suffix or None."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})", folder_name)
    if not m:
        return None
    return f"{m.group(2)}-{m.group(3)}", f"{m.group(4)}:{m.group(5)}"


def parse_mag(folder_name: str) -> int | None:
    """Pull the leading magnification (e.g. ``25`` from ``25x_…``) or None."""
    m = re.match(r"(\d+)x_", folder_name)
    return int(m.group(1)) if m else None
