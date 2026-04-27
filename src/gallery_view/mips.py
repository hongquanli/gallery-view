"""Pure-function MIP math (no Qt, no I/O).

Lifted from aion-explorer/explorer_ovelle.py with no behavioral changes.
"""

from typing import Iterable

import numpy as np

from .types import AxisMip, ChannelMips


def new_axis_state() -> dict:
    """Fresh accumulator for streaming MIPs along Z/Y/X.

    Holds a running ``z`` MIP (2-D) plus per-Z 1-D ``y_strips`` and
    ``x_strips``; ``finalize`` stacks the strips into 2-D Y and X MIPs.
    """
    return {"z": None, "y_strips": [], "x_strips": []}


def accumulate_axes(img: np.ndarray, state: dict) -> None:
    """Update ``state`` with one Z-slice (YX, float32)."""
    if state["z"] is None:
        state["z"] = img.copy()
    else:
        np.maximum(state["z"], img, out=state["z"])
    state["y_strips"].append(img.max(axis=0))  # 1-D along X
    state["x_strips"].append(img.max(axis=1))  # 1-D along Y


def finalize(state: dict) -> dict | None:
    """Turn accumulator state into ``{axis: 2D mip}`` or ``None`` if empty."""
    if state["z"] is None:
        return None
    return {
        "z": state["z"],
        "y": np.stack(state["y_strips"]) if state["y_strips"] else state["z"][:1],
        "x": np.stack(state["x_strips"]).T
        if state["x_strips"]
        else state["z"][:, :1].T,
    }


def axis_data_with_percentiles(axis_mips: dict) -> ChannelMips:
    """Wrap each MIP into an ``AxisMip`` with auto-contrast percentiles."""
    out: ChannelMips = {}
    for ax, mip in axis_mips.items():
        p1 = float(np.percentile(mip, 0.5))
        p999 = float(np.percentile(mip, 99.5))
        out[ax] = AxisMip(mip=mip.astype(np.float32), p1=p1, p999=p999)
    return out


def mip_to_rgba(
    mip: np.ndarray,
    p1: float,
    p999: float,
    color_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Normalize ``mip`` to [0, 1] using ``[p1, p999]`` and false-color it.

    Returns an HxWx4 uint8 array with alpha=255.
    """
    if p999 > p1:
        norm = np.clip((mip - p1) / (p999 - p1), 0.0, 1.0)
    else:
        norm = np.zeros_like(mip)
    h, w = norm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = (norm * color_rgb[0]).astype(np.uint8)
    rgba[:, :, 1] = (norm * color_rgb[1]).astype(np.uint8)
    rgba[:, :, 2] = (norm * color_rgb[2]).astype(np.uint8)
    rgba[:, :, 3] = 255
    return rgba


def stream_mips(slices: Iterable[np.ndarray]) -> tuple[dict | None, int]:
    """Convenience: drain an iterable of YX float32 slices, return finalized
    MIPs and the number of slices seen. Returns ``(None, 0)`` if empty."""
    state = new_axis_state()
    n = 0
    for img in slices:
        accumulate_axes(img, state)
        n += 1
    return finalize(state), n
