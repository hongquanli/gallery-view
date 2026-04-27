"""Open the napari 3D viewer for a single acquisition (one FOV).

Loads each channel's full ZYX stack via the handler, scales axes to µm,
adds a 100 µm-tick bounding box overlay, and reuses LUT limits from the
gallery's currently displayed projection axis (with sensible fallbacks)."""

from typing import Callable

import numpy as np

from ..sources._squid_common import parse_mag, parse_timestamp
from ..types import Acquisition
from .colors import napari_cmap_for


def open_napari(
    acq: Acquisition,
    fov: str,
    lut_lookup: Callable[[int, str], tuple[float, float] | None],
) -> None:
    """``lut_lookup(ch_idx, axis)`` returns the gallery's contrast limits if
    they're known, else None. We try the current axis, then Z, then fall
    back to a fresh percentile pass on the loaded stack."""

    import napari

    sensor_pixel_um = acq.params.get("sensor_pixel_size_um", 6.5)
    mag = parse_mag(acq.folder_name) or 1
    pixel_um = sensor_pixel_um / mag if mag else sensor_pixel_um
    dz_um = acq.params.get("dz(um)", pixel_um)
    scale = (dz_um, pixel_um, pixel_um)

    ts = parse_timestamp(acq.folder_name)
    datetime_str = f"{ts[0]} {ts[1]}" if ts else ""
    title = f"{acq.display_name} | FOV {fov}"
    if datetime_str:
        title += f" | {datetime_str}"

    viewer = napari.Viewer(ndisplay=3, title=title)

    first_shape: tuple[int, int, int] | None = None
    # ``iter_full_channel_stacks`` lets the OME-TIFF handler load the file
    # ONCE and yield per-channel views that share the underlying buffer,
    # rather than load_full_stack reloading the whole multi-GB file once
    # per channel (~4× memory blowup for a 4-channel acquisition).
    channel_index = {c.name: i for i, c in enumerate(acq.channels)}
    try:
        stack_iter = acq.handler.iter_full_channel_stacks(acq, fov)
    except Exception:
        return
    for channel, stack in stack_iter:
        ch_idx = channel_index.get(channel.name, 0)
        clim = lut_lookup(ch_idx, "current") or lut_lookup(ch_idx, "z")
        if clim is None:
            clim = (
                float(np.percentile(stack, 1)),
                float(np.percentile(stack, 99.9)),
            )
        viewer.add_image(
            stack,
            scale=scale,
            name=f"{channel.wavelength}nm",
            colormap=napari_cmap_for(channel.wavelength),
            blending="additive",
            contrast_limits=clim,
        )
        if first_shape is None:
            first_shape = stack.shape

    if first_shape is not None:
        _add_bounding_box(viewer, scale, first_shape)

    viewer.text_overlay.visible = True
    viewer.text_overlay.text = title
    viewer.text_overlay.font_size = 12
    viewer.text_overlay.color = "white"
    viewer.text_overlay.position = "top_center"
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"


def _add_bounding_box(viewer, scale, shape_zyx) -> None:
    nz, ny, nx = shape_zyx
    z_max = nz * scale[0]
    y_max = ny * scale[1]
    x_max = nx * scale[2]

    edges = [
        [[0, 0, 0], [0, 0, x_max]],
        [[0, 0, x_max], [0, y_max, x_max]],
        [[0, y_max, x_max], [0, y_max, 0]],
        [[0, y_max, 0], [0, 0, 0]],
        [[z_max, 0, 0], [z_max, 0, x_max]],
        [[z_max, 0, x_max], [z_max, y_max, x_max]],
        [[z_max, y_max, x_max], [z_max, y_max, 0]],
        [[z_max, y_max, 0], [z_max, 0, 0]],
        [[0, 0, 0], [z_max, 0, 0]],
        [[0, 0, x_max], [z_max, 0, x_max]],
        [[0, y_max, x_max], [z_max, y_max, x_max]],
        [[0, y_max, 0], [z_max, y_max, 0]],
    ]
    tick_len = min(z_max, y_max, x_max) * 0.02
    ticks: list[list[list[float]]] = []
    for x in np.arange(100, x_max, 100):
        ticks += [[[0, 0, x], [0, tick_len, x]], [[0, 0, x], [tick_len, 0, x]]]
    for y in np.arange(100, y_max, 100):
        ticks += [[[0, y, 0], [0, y, tick_len]], [[0, y, 0], [tick_len, y, 0]]]
    for z in np.arange(100, z_max, 100):
        ticks += [[[z, 0, 0], [z, tick_len, 0]], [[z, 0, 0], [z, 0, tick_len]]]
    viewer.add_shapes(
        [np.array(line) for line in edges + ticks],
        shape_type="line",
        edge_color="white",
        edge_width=2,
        name="Bounding Box (100µm ticks)",
    )
