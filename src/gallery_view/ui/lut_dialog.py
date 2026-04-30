"""Per-channel LUT dialog with min/max sliders, dataset-wide actions, and PNG
export. Operates on the current projection axis; saves apply across all three
axes via the `.lut.json` sidecar."""

import os
import re
from typing import Callable

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from .. import cache
from ..mips import mip_to_rgba
from ..sources._squid_common import display_fov, resolve_mag
from ..types import Acquisition, AxisMip
from .colors import rgb_for
from .zoomable_view import ZoomableImageView

AXES = ("z", "y", "x")
PREVIEW_SIZE = 400
EXPORT_DPI = 600


def show_lut_dialog(
    parent,
    acq: Acquisition,
    fov: str,
    timepoint: str,
    axis: str,
    mip_data: dict,
    refresh_thumb: Callable[[int, str, str, int, AxisMip], None],
    acq_id: int,
) -> None:
    """Open the LUT dialog. ``mip_data`` is the gallery window's
    ``{(acq_id, timepoint, fov, ch_idx, axis): AxisMip}`` map; we mutate
    it in place. ``refresh_thumb(acq_id, timepoint, fov, ch_idx, ax_mip)``
    is called whenever a channel's LUT changes so the gallery thumbnail
    re-renders."""

    ch_keys = sorted(
        ci for (a, t, f, ci, ax) in mip_data
        if a == acq_id and t == timepoint and f == fov and ax == axis
    )
    if not ch_keys:
        return

    axis_label = {"z": "Z (XY)", "y": "Y (XZ)", "x": "X (YZ)"}[axis]
    snapshot: dict[tuple[int, str, str, int, str], tuple[float, float]] = {}
    for ci in ch_keys:
        for ax in AXES:
            entry = mip_data.get((acq_id, timepoint, fov, ci, ax))
            if entry is not None:
                snapshot[(acq_id, timepoint, fov, ci, ax)] = (entry.p1, entry.p999)

    dlg = QDialog(parent)
    mag = resolve_mag(acq.folder_name, acq.params) or "?"
    dlg.setWindowTitle(
        f"LUT — {acq.display_name} | {mag}x | FOV {display_fov(fov)} | {axis_label}"
    )
    dlg.setStyleSheet("background-color: #1e1e1e; color: white;")

    def revert_unsaved() -> None:
        for k, (p1, p999) in snapshot.items():
            entry = mip_data.get(k)
            if entry is None:
                continue
            mip_data[k] = AxisMip(mip=entry.mip, p1=p1, p999=p999)
            _, _, _, ch_idx, ax = k
            if ax == axis:
                refresh_thumb(acq_id, timepoint, fov, ch_idx, mip_data[k])

    dlg.rejected.connect(revert_unsaved)

    outer = QVBoxLayout(dlg)
    outer.setSpacing(8)

    channels_row = QHBoxLayout()
    channels_row.setSpacing(12)
    outer.addLayout(channels_row)

    sensor_pixel_um = acq.params.get("sensor_pixel_size_um", 6.5)
    pixel_um = sensor_pixel_um / mag if isinstance(mag, int) else sensor_pixel_um
    dz_um = acq.params.get("dz(um)", pixel_um)
    y_scale = 1.0 if axis == "z" else (dz_um / pixel_um if pixel_um > 0 else 1.0)

    min_slider_targets: list[tuple[QSlider, int]] = []

    for ch_idx in ch_keys:
        ax_mip = mip_data[(acq_id, timepoint, fov, ch_idx, axis)]
        wl = acq.channels[ch_idx].wavelength
        color = rgb_for(wl)
        data_min, data_max = float(ax_mip.mip.min()), float(ax_mip.mip.max())

        col = QVBoxLayout()
        col.setSpacing(4)

        ch_lbl = QLabel(f"{wl} nm")
        ch_lbl.setAlignment(Qt.AlignCenter)
        ch_lbl.setStyleSheet(
            f"color: rgb({color[0]},{color[1]},{color[2]}); font-size: 12px; font-weight: bold;"
        )
        col.addWidget(ch_lbl)

        preview = ZoomableImageView(PREVIEW_SIZE)
        col.addWidget(preview)

        def make_render(prev_view, m, c, ys):
            def render(lo: float, hi: float) -> None:
                rgba = mip_to_rgba(m, lo, hi, c)
                h, w = rgba.shape[:2]
                qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
                prev_view.set_pixmap(QPixmap.fromImage(qimg), ys)
            return render

        render_fn = make_render(preview, ax_mip.mip, color, y_scale)
        render_fn(ax_mip.p1, ax_mip.p999)

        slider_lo = min(0, int(data_min))
        slider_hi = int(data_max)

        min_header = QHBoxLayout()
        min_header.setContentsMargins(0, 0, 0, 0)
        min_header.addWidget(QLabel("Min"))
        min_header.addStretch()
        btn_min_zero = QPushButton("→ 0")
        btn_min_zero.setFixedSize(38, 18)
        btn_min_zero.setStyleSheet(
            "QPushButton { background-color: #444; color: #ddd; border: 1px solid #555;"
            " border-radius: 3px; font-size: 9px; }"
            "QPushButton:hover { background-color: #666; }"
        )
        min_header.addWidget(btn_min_zero)
        col.addLayout(min_header)

        sl_min = QSlider(Qt.Horizontal)
        sl_min.setRange(slider_lo, slider_hi)
        sl_min.setValue(int(ax_mip.p1))
        col.addWidget(sl_min)
        btn_min_zero.clicked.connect(lambda _, s=sl_min: s.setValue(0))
        min_slider_targets.append((sl_min, int(data_min)))

        col.addWidget(QLabel("Max"))
        sl_max = QSlider(Qt.Horizontal)
        sl_max.setRange(slider_lo, slider_hi)
        sl_max.setValue(int(ax_mip.p999))
        col.addWidget(sl_max)

        val_lbl = QLabel(f"{int(ax_mip.p1)} — {int(ax_mip.p999)}")
        val_lbl.setAlignment(Qt.AlignCenter)
        val_lbl.setStyleSheet("color: #888; font-size: 10px;")
        col.addWidget(val_lbl)

        def make_handler(s_min, s_max, v_lbl, rfn, ci):
            def on_change() -> None:
                lo, hi = s_min.value(), s_max.value()
                if hi <= lo:
                    hi = lo + 1
                v_lbl.setText(f"{lo} — {hi}")
                rfn(float(lo), float(hi))
                # Apply to all three axes (synchronized)
                for ax in AXES:
                    key = (acq_id, timepoint, fov, ci, ax)
                    entry = mip_data.get(key)
                    if entry is None:
                        continue
                    mip_data[key] = AxisMip(mip=entry.mip, p1=float(lo), p999=float(hi))
                refresh_thumb(
                    acq_id, timepoint, fov, ci,
                    mip_data[(acq_id, timepoint, fov, ci, axis)],
                )
            return on_change

        handler = make_handler(sl_min, sl_max, val_lbl, render_fn, ch_idx)
        sl_min.valueChanged.connect(handler)
        sl_max.valueChanged.connect(handler)

        channels_row.addLayout(col)

    # Bottom bar
    bottom = QHBoxLayout()
    bottom.setSpacing(8)
    btn_min_reset = QPushButton("Min → Data Min (all channels)")
    btn_min_reset.setFixedHeight(30)
    btn_min_reset.setStyleSheet(
        "QPushButton { background-color: #555; color: white; border-radius: 4px;"
        " font-size: 11px; padding: 0 12px; }"
        "QPushButton:hover { background-color: #777; }"
    )
    btn_min_reset.clicked.connect(
        lambda: [s.setValue(d) for s, d in min_slider_targets]
    )
    bottom.addWidget(btn_min_reset)

    btn_export = QPushButton("Export PNG…")
    btn_export.setFixedHeight(30)
    btn_export.setStyleSheet(btn_min_reset.styleSheet())
    btn_export.clicked.connect(
        lambda: _export_png(
            dlg, acq, fov, timepoint, axis, axis_label, ch_keys,
            mip_data, acq_id, dz_um, pixel_um,
        )
    )
    bottom.addWidget(btn_export)
    bottom.addStretch()

    btn_save = QPushButton("Save")
    btn_save.setFixedSize(80, 30)
    btn_save.setStyleSheet(
        "QPushButton { background-color: #2d5aa0; color: white; border-radius: 4px;"
        " font-size: 12px; font-weight: bold; }"
        "QPushButton:hover { background-color: #3a6fc0; }"
    )

    def save_all() -> None:
        for ci in ch_keys:
            channel = acq.channels[ci]
            src, ch_id = acq.handler.cache_key(
                acq, fov, channel, timepoint=timepoint
            )
            axis_data = {}
            for ax in AXES:
                entry = mip_data.get((acq_id, timepoint, fov, ci, ax))
                if entry is None:
                    continue
                axis_data[ax] = (entry.mip, entry.p1, entry.p999)
            if axis_data:
                cache.save_lut_only(src, ch_id, axis_data)
        dlg.accept()

    btn_save.clicked.connect(save_all)
    bottom.addWidget(btn_save)
    outer.addLayout(bottom)

    dlg.adjustSize()
    if hasattr(dlg, "exec"):
        dlg.exec()
    else:
        dlg.exec_()


def _export_png(
    dlg, acq, fov, timepoint, axis, axis_label, ch_keys, mip_data, acq_id,
    dz_um, pixel_um,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        QMessageBox.warning(
            dlg, "Missing dependency",
            "matplotlib is required to export. `pip install matplotlib`.",
        )
        return

    ts = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})", acq.folder_name)
    datetime_str = f"{ts.group(1)} {ts.group(2)}:{ts.group(3)}" if ts else ""
    ts_part = f"_{ts.group(1)}_{ts.group(2)}{ts.group(3)}" if ts else ""
    safe_name = acq.display_name.replace(" ", "_").replace("/", "_")
    default_name = f"{safe_name}_{axis}_fov{display_fov(fov)}{ts_part}.png"
    path, _ = QFileDialog.getSaveFileName(dlg, "Export view", default_name, "PNG (*.png)")
    if not path:
        return

    n_channels = len(ch_keys)
    fig, axes_grid = plt.subplots(
        2, n_channels,
        figsize=(5.0 * n_channels, 8.5),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="black",
    )
    if n_channels == 1:
        axes_grid = axes_grid.reshape(2, 1)

    imshow_aspect = 1.0 if axis == "z" else (
        dz_um / pixel_um if pixel_um > 0 else 1.0
    )
    text_pad_pt = 60 * 72 / EXPORT_DPI

    for col_i, ci in enumerate(ch_keys):
        entry = mip_data.get((acq_id, timepoint, fov, ci, axis))
        if entry is None:
            continue
        wl = acq.channels[ci].wavelength
        color = rgb_for(wl)
        rgba = mip_to_rgba(entry.mip, entry.p1, entry.p999, color)
        color_norm = tuple(c / 255 for c in color)

        ax_img = axes_grid[0, col_i]
        ax_img.set_facecolor("black")
        ax_img.imshow(rgba, aspect=imshow_aspect, interpolation="none")
        ax_img.set_title(f"{wl} nm", color=color_norm, fontsize=11, pad=text_pad_pt)
        extras = acq.handler.channel_yaml_extras(acq, acq.channels[ci])
        parts = []
        if extras.get("exposure_ms") is not None:
            parts.append(f"{extras['exposure_ms']:.0f} ms")
        if extras.get("intensity") is not None:
            parts.append(f"{extras['intensity']:.0f}% laser")
        if parts:
            ax_img.set_xlabel(" · ".join(parts), color="#888",
                              fontsize=10, labelpad=text_pad_pt)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_color("#888")
            spine.set_linewidth(6 * 72 / EXPORT_DPI)

        ax_hist = axes_grid[1, col_i]
        ax_hist.set_facecolor("black")
        ax_hist.hist(entry.mip.ravel(), bins=256, color=color_norm, log=True)
        ax_hist.axvline(entry.p1, color="white", linestyle="--", linewidth=1,
                        label=f"min={entry.p1:.0f}")
        ax_hist.axvline(entry.p999, color="orange", linestyle="--", linewidth=1,
                        label=f"max={entry.p999:.0f}")
        leg = ax_hist.legend(fontsize=8, loc="upper right",
                             facecolor="black", labelcolor="white")
        leg.get_frame().set_edgecolor("#444")
        ax_hist.set_xlabel("intensity", color="white")
        ax_hist.set_ylabel("count (log)", color="white")
        ax_hist.tick_params(colors="white")
        for spine in ax_hist.spines.values():
            spine.set_color("#666")

    title_main = f"{acq.display_name} | FOV {display_fov(fov)} | {axis_label}"
    fig.text(0.5, 0.985, title_main, ha="center", va="top", fontsize=13, color="white")
    if datetime_str:
        fig.text(0.5, 0.96, datetime_str, ha="center", va="top", fontsize=11, color="#888")
        top_rect = 0.945
    else:
        top_rect = 0.97
    fig.tight_layout(rect=[0, 0, 1, top_rect], h_pad=0.6, w_pad=0.4)
    try:
        fig.savefig(path, dpi=EXPORT_DPI, facecolor="black",
                    bbox_inches="tight", pad_inches=0.15)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.warning(dlg, "Export failed", str(exc))
    finally:
        plt.close(fig)
