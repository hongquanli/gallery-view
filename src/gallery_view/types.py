"""Shared dataclasses and type aliases."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from .sources.base import FormatHandler

Axis = Literal["z", "y", "x"]
ShapeZYX = tuple[int, int, int]


@dataclass(frozen=True)
class Channel:
    """A single fluorescence channel.

    ``name`` is the squid channel name (e.g. ``"Fluorescence_488_nm_Ex"``).
    ``wavelength`` is the digit string parsed from it, or ``"unknown"``.
    """

    name: str
    wavelength: str


@dataclass(frozen=True)
class AxisMip:
    """One axis's MIP plus its auto-contrast percentiles."""

    mip: np.ndarray  # 2-D float32, full resolution
    p1: float  # 0.5 percentile of mip values
    p999: float  # 99.5 percentile of mip values


# {axis: AxisMip} — one channel's MIPs along Z, Y, X
ChannelMips = dict[Axis, AxisMip]


@dataclass
class Acquisition:
    """One z-stack acquisition discovered by a FormatHandler.

    Fields are populated by ``handler.build()``; ``selected_fov`` is mutable
    and tracks the FOV picker state in the UI. ``extra`` is a per-handler
    escape hatch for format-private fields (no one outside the owning
    handler reads it).
    """

    handler: "FormatHandler"
    path: str
    folder_name: str
    display_name: str
    params: dict
    channels: list[Channel]
    fovs: list[str]
    shape_zyx: ShapeZYX | None = None
    selected_fov: str = "0"
    timepoints: list[str] = field(default_factory=lambda: ["0"])
    selected_timepoint: str = "0"
    extra: dict = field(default_factory=dict)
