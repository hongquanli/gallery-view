"""Coordinate-based region stitcher: places per-FOV Z-MIPs by stage position
and mean-blends overlaps. Pure functions, no Qt, no thread state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FovCoord:
    """Stage position of one FOV center (in millimeters)."""
    fov: str   # composite '<region>_<fov>' matching acq.fovs
    x_mm: float
    y_mm: float
