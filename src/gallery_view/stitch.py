"""Coordinate-based region stitcher: places per-FOV Z-MIPs by stage position
and mean-blends overlaps. Pure functions, no Qt, no thread state."""

import math
from dataclasses import dataclass

import numpy as np

from .types import AxisMip


@dataclass(frozen=True)
class FovCoord:
    """Stage position of one FOV center (in millimeters)."""
    fov: str   # composite '<region>_<fov>' matching acq.fovs
    x_mm: float
    y_mm: float


def stitch_region(
    fov_mips: dict[str, np.ndarray],
    coords: list[FovCoord],
    pixel_um: float,
    target_longest_px: int = 1024,
    flip_y: bool = False,
) -> AxisMip | None:
    """Place each FOV's Z-MIP at its stage coordinate, downsample to fit
    ``target_longest_px`` on the longest axis, mean-blend overlaps.

    Returns ``None`` when ``fov_mips`` is empty, when FOV shapes differ, or
    when no FovCoord lines up with a key in ``fov_mips``. Auto-contrast
    percentiles are computed over the covered region only (so black gaps
    don't drag ``p1`` toward zero).
    """
    if not fov_mips:
        return None

    if pixel_um <= 0:
        return None

    # Shape uniformity check — region view doesn't support heterogeneous FOVs.
    shapes = {arr.shape for arr in fov_mips.values()}
    if len(shapes) != 1:
        return None
    ny, nx = next(iter(shapes))

    # Bail if no coord lines up with any tile we have.
    if not any(c.fov in fov_mips for c in coords):
        return None
    # We size the canvas from *all* coords (so a missing FOV still leaves a
    # black gap at the right spot), but only paint tiles we have.

    pixel_mm = pixel_um / 1000.0

    # FOV top-left in stage *pixels* (CSV holds FOV centers). Rounding here
    # — instead of subtracting then ceiling later — keeps tiny float jitter
    # near large stage offsets from inflating the canvas by an extra pixel.
    def corner_px(c: FovCoord) -> tuple[int, int]:
        col = int(round(c.x_mm / pixel_mm - nx / 2.0))
        row = int(round(c.y_mm / pixel_mm - ny / 2.0))
        return col, row

    px_corners = [corner_px(c) for c in coords]
    min_col = min(col for col, _ in px_corners)
    max_col = max(col for col, _ in px_corners) + nx
    min_row = min(row for _, row in px_corners)
    max_row = max(row for _, row in px_corners) + ny

    # Full-res canvas in pixels.
    W_full = max(nx, max_col - min_col)
    H_full = max(ny, max_row - min_row)

    # Integer downsample factor so the longest axis fits target_longest_px.
    factor = max(1, math.ceil(max(H_full, W_full) / max(target_longest_px, 1)))
    H = max(1, H_full // factor)
    W = max(1, W_full // factor)

    accum = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for c, (col_px, row_px) in zip(coords, px_corners):
        if c.fov not in fov_mips:
            continue
        tile = fov_mips[c.fov]
        # Block-mean downsample, truncating to multiples of factor. When
        # factor > ny or factor > nx (very aggressive downsample relative
        # to FOV size), fall back to averaging the whole tile down to 1 px
        # along that axis — keeps placement correct without crashing in
        # reshape.
        ny_use = (ny // factor) * factor
        nx_use = (nx // factor) * factor
        if ny_use == 0 or nx_use == 0:
            tile_ds = np.array([[tile.mean()]], dtype=np.float32)
            tile_ny_ds, tile_nx_ds = 1, 1
        else:
            tile_ny_ds = ny_use // factor
            tile_nx_ds = nx_use // factor
            tile_ds = (
                tile[:ny_use, :nx_use]
                .reshape(tile_ny_ds, factor, tile_nx_ds, factor)
                .mean(axis=(1, 3))
                .astype(np.float32)
            )

        col = (col_px - min_col) // factor
        if flip_y:
            # Flip stage Y so increasing y_mm moves *up* on the canvas
            # (origin at top): y=max ends up at row 0.
            row = (max_row - (row_px + ny)) // factor
        else:
            row = (row_px - min_row) // factor

        row = max(0, min(H - tile_ny_ds, row))
        col = max(0, min(W - tile_nx_ds, col))

        accum[row:row + tile_ny_ds, col:col + tile_nx_ds] += tile_ds
        weight[row:row + tile_ny_ds, col:col + tile_nx_ds] += 1.0

    mosaic = accum / np.maximum(weight, 1.0)
    covered = weight > 0

    if not covered.any():
        return None

    p1 = float(np.percentile(mosaic[covered], 0.5))
    p999 = float(np.percentile(mosaic[covered], 99.5))
    return AxisMip(mip=mosaic, p1=p1, p999=p999)
