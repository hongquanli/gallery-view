"""On-disk MIP cache and per-channel LUT-override sidecar.

Keys are MD5 of ``"<src_path>|<channel_id>"`` truncated to 16 hex chars; the
FOV is encoded into ``channel_id`` by each handler when relevant. The cache
module itself is FOV-agnostic.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import platformdirs

from .types import AxisMip, ChannelMips, ShapeZYX

CACHE_VERSION = 2  # v2: OmeTiffHandler honors Plane tags (fixes XLight V3 channels)
AXES = ("z", "y", "x")
CACHE_DIR = str(Path(platformdirs.user_cache_dir("gallery-view")) / "mips")


def _cache_path(src_path: str, channel_id: str) -> Path:
    key = f"{src_path}|{channel_id}"
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return Path(CACHE_DIR) / f"{h}.npz"


def _lut_override_path(src_path: str, channel_id: str) -> Path:
    return _cache_path(src_path, channel_id).with_suffix(".lut.json")


def _load_lut_override(src_path: str, channel_id: str) -> dict | None:
    p = _lut_override_path(src_path, channel_id)
    if not p.exists():
        return None
    try:
        with p.open() as f:
            data = json.load(f)
        return {ax: (float(v["p1"]), float(v["p999"])) for ax, v in data.items()}
    except (OSError, ValueError, KeyError):
        return None


def load(
    src_path: str, channel_id: str
) -> tuple[ChannelMips | None, ShapeZYX | None]:
    """Read the cached MIPs (with any LUT override applied) for one channel.

    Returns ``(None, None)`` if the file is missing or the version is stale.
    """
    path = _cache_path(src_path, channel_id)
    if not path.exists():
        return None, None
    try:
        data = np.load(path)
        if "version" not in data.files or int(data["version"]) != CACHE_VERSION:
            return None, None
        out: ChannelMips = {}
        for ax in AXES:
            if f"mip_{ax}" not in data.files:
                return None, None
            out[ax] = AxisMip(
                mip=np.asarray(data[f"mip_{ax}"]),
                p1=float(data[f"p1_{ax}"]),
                p999=float(data[f"p999_{ax}"]),
            )
        shape: ShapeZYX | None = None
        if "nz_orig" in data.files:
            shape = (
                int(data["nz_orig"]),
                int(data["ny_orig"]),
                int(data["nx_orig"]),
            )
    except (OSError, ValueError, KeyError):
        return None, None

    overrides = _load_lut_override(src_path, channel_id)
    if overrides:
        for ax, (p1, p999) in overrides.items():
            if ax in out:
                out[ax] = AxisMip(mip=out[ax].mip, p1=p1, p999=p999)
    return out, shape


def save(
    src_path: str,
    channel_id: str,
    axis_data: ChannelMips,
    shape_zyx: ShapeZYX | None = None,
) -> None:
    """Full save: arrays + auto-percentiles + shape.

    Preserves any pre-existing LUT-override sidecar — the override is a
    user preference (intensity bounds) that's independent of the MIP
    arrays it applies to, so re-computing MIPs (e.g. on a cache version
    bump) shouldn't nuke saved contrast settings.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(src_path, channel_id)
    payload: dict = {"version": np.int32(CACHE_VERSION)}
    for ax, ax_mip in axis_data.items():
        payload[f"mip_{ax}"] = ax_mip.mip.astype(np.float32)
        payload[f"p1_{ax}"] = np.float32(ax_mip.p1)
        payload[f"p999_{ax}"] = np.float32(ax_mip.p999)
    if shape_zyx is not None:
        payload["nz_orig"] = np.int32(shape_zyx[0])
        payload["ny_orig"] = np.int32(shape_zyx[1])
        payload["nx_orig"] = np.int32(shape_zyx[2])
    np.savez_compressed(path, **payload)


def save_lut_only(
    src_path: str,
    channel_id: str,
    axis_data: dict,
) -> None:
    """Quick LUT-only save. Writes a tiny JSON next to the .npz cache.

    ``axis_data`` is ``{axis: (mip_unused, p1, p999)}`` — same shape the LUT
    dialog hands us; only the floats are written.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    payload = {
        ax: {"p1": float(p1), "p999": float(p999)}
        for ax, (_mip, p1, p999) in axis_data.items()
    }
    with _lut_override_path(src_path, channel_id).open("w") as f:
        json.dump(payload, f)


def clear_all() -> None:
    """Delete the cache directory if it exists. Used by 'Clear MIP cache…'."""
    import shutil

    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
