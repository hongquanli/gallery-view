"""Gallery window: drag-drop ingestion, sources strip, scroll area, loader."""

import os
from dataclasses import dataclass

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import scan
from ..loader import Job, MipLoader, RegionStitchJob
from ..mips import mip_to_rgba
from ..sources._squid_common import display_fov, parse_timestamp, resolve_mag
from ..types import Acquisition, AxisMip
from .colors import CHANNEL_ORDER, rgb_for

THIN_Z_THRESHOLD = 5
THUMB_SIZE_PRESETS = [("Small", 80), ("Medium", 160), ("Large", 320)]
DEFAULT_THUMB_SIZE = 160

# QComboBox popups are top-level Qt widgets, so the QMainWindow stylesheet
# doesn't cascade in. Style each combo explicitly — both the closed widget
# and its QAbstractItemView popup — so light-on-light-grey doesn't make
# the dropdown unreadable.
_COMBO_STYLE = (
    "QComboBox { background-color: #333; color: white; border: 1px solid #555;"
    " border-radius: 3px; padding: 2px 8px; font-size: 11px; }"
    "QComboBox::drop-down { border: none; }"
    "QComboBox QAbstractItemView { background-color: #2a2a2a; color: white;"
    " selection-background-color: #2d5aa0; selection-color: white;"
    " border: 1px solid #555; }"
)
# Pixels added to the longest item's text width to leave room for the
# dropdown arrow, the border, and a touch of breathing room. Tuned for
# the 11px font in ``_COMBO_STYLE`` at default DPI; revisit if the theme
# font size or a non-default DPI throws off the visible padding.
_COMBO_PADDING_PX = 40


def _style_combo(combo: "QComboBox") -> None:
    """Apply the dark-theme stylesheet to a QComboBox.

    Call ``_size_combo_to_contents(combo)`` AFTER ``addItem`` to size the
    closed widget and its popup view to the longest item.
    """
    combo.setStyleSheet(_COMBO_STYLE)


def _size_combo_to_contents(combo: "QComboBox") -> None:
    """Set the combo's minimum width — and its popup view's minimum width
    — so the longest item shows without clipping. ``AdjustToContents``
    alone isn't enough: with the dark stylesheet applied, macOS Qt
    renders short items like ``t=12`` clipped to ``t=`` in the popup.
    Must be called after items have been added.
    """
    metrics = combo.fontMetrics()
    longest = max(
        (metrics.horizontalAdvance(combo.itemText(i)) for i in range(combo.count())),
        default=0,
    )
    width = longest + _COMBO_PADDING_PX
    combo.setMinimumWidth(width)
    combo.view().setMinimumWidth(width)


@dataclass
class Source:
    path: str
    acq_ids: list[int]


@dataclass
class RowKey:
    """A row in the gallery: ``(acq_id, timepoint, unit)`` — ``unit`` is a
    composite ``<region>_<fov>`` in FOV view and a region id in region view."""
    acq_id: int
    timepoint: str
    fov: str


@dataclass
class RowWidgets:
    container: QWidget
    mag_lbl: "QLabel"
    time_lbl: "QLabel"
    name_lbl: "QLabel"
    thumb_labels: "dict[int, QLabel]"   # ch_idx -> data thumb
    thumb_columns: "dict[str, QLabel]"  # wavelength -> the cell currently rendered (data or placeholder)
    unit_combo: "QComboBox | None"
    time_combo: "QComboBox | None"


class GalleryWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("gallery-view")
        self.setMinimumSize(1100, 700)
        self.setAcceptDrops(True)
        # Bare (un-scoped) selectors so the dark theme cascades to every
        # descendant widget — same approach as explorer_ovelle.py. Scoping
        # the rule to ``QMainWindow`` only paints the frame, leaving the
        # scroll area, content widget, and group margins platform-default
        # (light gray on macOS).
        self.setStyleSheet(
            "background-color: #1a1a1a; color: white;"
            "QGroupBox { font-size: 13px; font-weight: bold; border: 1px solid #333;"
            " border-radius: 6px; margin-top: 8px; padding: 6px 4px 4px 4px; color: #ddd; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )

        self.acquisitions: list[Acquisition | None] = []
        self.sources: list[Source] = []
        self.row_keys: list[RowKey] = []
        self.row_widgets: dict[tuple[int, str, str], RowWidgets] = {}
        self.source_groups: list[QGroupBox] = []
        self.expanded_fov_mode: bool = False
        self.square_footprint: bool = False
        # (acq_id, timepoint, fov, ch_idx, axis) -> AxisMip
        self.mip_data: dict[tuple[int, str, str, int, str], "AxisMip"] = {}
        self.view_mode: str = "fov"  # "fov" | "region"
        self.expanded_region_mode: bool = False
        # (acq_id, timepoint, region, ch_idx) -> AxisMip for stitched mosaics
        self.region_mip_data: dict[tuple[int, str, str, int], AxisMip] = {}
        # (acq_id, timepoint, region) -> set of (ch_idx, fov_id) we've seen,
        # used to detect "all FOV MIPs ready, time to stitch".
        self._region_fov_readiness: dict[
            tuple[int, str, str], set[tuple[int, str]]
        ] = {}

        # Loader thread (long-lived)
        self.loader = MipLoader()
        self.loader.mip_ready.connect(self._on_mip_ready)
        self.loader.progress.connect(self._on_progress)
        self.loader.idle.connect(self._on_idle)
        self.loader.region_mip_ready.connect(self._on_region_mip_ready)
        self.loader.start()

        self._build_ui()

    # ── ui scaffolding ──

    def _build_ui(self) -> None:
        from .sources_panel import SourcesPanel

        self._build_menus()
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        self.sources_panel = SourcesPanel()
        self.sources_panel.add_requested.connect(self._add_source)
        self.sources_panel.remove_requested.connect(self._remove_source)
        self.sources_panel.hide()  # appears after first source
        layout.addWidget(self.sources_panel)

        self._build_filter_row(layout)
        self._build_display_row(layout)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll_layout.setSpacing(4)
        self.scroll_layout.addStretch()
        self.scroll.setWidget(self.scroll_content)
        layout.addWidget(self.scroll, stretch=1)

        self.empty_overlay = QLabel("Drop folders here to begin")
        self.empty_overlay.setAlignment(Qt.AlignCenter)
        self.empty_overlay.setStyleSheet(
            "QLabel { color: #666; font-size: 18px;"
            " border: 2px dashed #444; border-radius: 12px; padding: 60px; }"
        )
        self.scroll_layout.insertWidget(0, self.empty_overlay)

    def _build_filter_row(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(10)
        mag_label = QLabel("Magnification:")
        mag_label.setStyleSheet("color: #888; font-size: 11px;")
        row.addWidget(mag_label)
        self.mag_row_layout = QHBoxLayout()
        self.mag_row_layout.setSpacing(6)
        row.addLayout(self.mag_row_layout)
        row.addSpacing(16)

        self.hide_thin_btn = QPushButton(f"Hide thin (Z<{THIN_Z_THRESHOLD})")
        self.hide_thin_btn.setCheckable(True)
        self.hide_thin_btn.setChecked(True)
        self.hide_thin_btn.setCursor(Qt.PointingHandCursor)
        self.hide_thin_btn.setStyleSheet(self._toggle_style())
        self.hide_thin_btn.toggled.connect(self._refresh_visibility)
        row.addWidget(self.hide_thin_btn)

        row.addStretch()

        self.status = QLabel("Drop folders to begin.")
        self.status.setStyleSheet("color: #666; font-size: 11px; padding: 2px 4px;")
        row.addWidget(self.status)
        layout.addLayout(row)

        self.mag_checkboxes: dict[int, QCheckBox] = {}

    def _build_display_row(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(10)
        proj_lbl = QLabel("Project:")
        proj_lbl.setStyleSheet("color: #888; font-size: 11px;")
        row.addWidget(proj_lbl)

        self.axis_btn_group = QButtonGroup(self)
        self.axis_btn_group.setExclusive(True)
        self.axis_buttons: dict[str, QPushButton] = {}
        for ax, label in [("z", "XY"), ("y", "XZ"), ("x", "YZ")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedSize(36, 22)
            btn.setStyleSheet(self._toggle_style())
            btn.setCursor(Qt.PointingHandCursor)
            btn.setChecked(ax == "z")
            btn.clicked.connect(lambda _, a=ax: self._set_axis(a))
            self.axis_btn_group.addButton(btn)
            self.axis_buttons[ax] = btn
            row.addWidget(btn)
        self.view_axis: str = "z"

        row.addSpacing(16)
        view_lbl = QLabel("View:")
        view_lbl.setStyleSheet("color: #888; font-size: 11px;")
        row.addWidget(view_lbl)

        self.view_btn_group = QButtonGroup(self)
        self.view_btn_group.setExclusive(True)
        self.view_buttons: dict[str, QPushButton] = {}
        for mode, label in [("fov", "FOV"), ("region", "Region")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedSize(54, 22)
            btn.setStyleSheet(self._toggle_style())
            btn.setCursor(Qt.PointingHandCursor)
            btn.setChecked(mode == "fov")
            btn.clicked.connect(lambda _, m=mode: self._set_view_mode(m))
            self.view_btn_group.addButton(btn)
            self.view_buttons[mode] = btn
            row.addWidget(btn)
        # Region button starts disabled; _refresh_region_button_enabled enables
        # it once a source with multi-region data is added.
        self.view_buttons["region"].setEnabled(False)
        self.view_buttons["region"].setToolTip(
            "No source supports region view"
        )

        row.addSpacing(16)
        size_lbl = QLabel("Thumbnail size:")
        size_lbl.setStyleSheet("color: #888; font-size: 11px;")
        row.addWidget(size_lbl)
        self.size_combo = QComboBox()
        _style_combo(self.size_combo)
        for label, size in THUMB_SIZE_PRESETS:
            self.size_combo.addItem(label, size)
        _size_combo_to_contents(self.size_combo)
        self.size_combo.setCurrentIndex(
            next(i for i, (_, s) in enumerate(THUMB_SIZE_PRESETS) if s == DEFAULT_THUMB_SIZE)
        )
        self.thumb_size: int = DEFAULT_THUMB_SIZE
        self.size_combo.currentIndexChanged.connect(
            lambda i: self._set_thumb_size(self.size_combo.itemData(i))
        )
        row.addWidget(self.size_combo)
        row.addStretch()
        layout.addLayout(row)

    def _build_menus(self) -> None:
        from qtpy.QtGui import QAction

        menubar = self.menuBar()
        menubar.setStyleSheet(
            "QMenuBar { background-color: #2a2a2a; color: #ccc; }"
            "QMenuBar::item:selected { background-color: #444; }"
            "QMenu { background-color: #2a2a2a; color: #ccc; border: 1px solid #444; }"
            "QMenu::item:selected { background-color: #2d5aa0; color: white; }"
        )

        file_menu = menubar.addMenu("File")
        add_action = QAction("Add Folder…", self)
        add_action.setShortcut("Ctrl+O")
        add_action.triggered.connect(self._on_add_folder)
        file_menu.addAction(add_action)
        refresh_action = QAction("Refresh sources", self)
        refresh_action.triggered.connect(self._on_refresh_sources)
        file_menu.addAction(refresh_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        settings_menu = menubar.addMenu("Settings")

        self.square_action = QAction("Square footprint for XZ/YZ", self)
        self.square_action.setCheckable(True)
        self.square_action.toggled.connect(self._set_square_footprint)
        settings_menu.addAction(self.square_action)

        self.expand_action = QAction("Expand all FOVs as separate rows", self)
        self.expand_action.setCheckable(True)
        self.expand_action.toggled.connect(self._set_expanded_fov_mode)
        settings_menu.addAction(self.expand_action)

        self.expand_region_action = QAction(
            "Expand all regions as separate rows", self
        )
        self.expand_region_action.setCheckable(True)
        self.expand_region_action.toggled.connect(self._set_expanded_region_mode)
        settings_menu.addAction(self.expand_region_action)

        settings_menu.addSeparator()
        clear_action = QAction("Clear MIP cache…", self)
        clear_action.triggered.connect(self._on_clear_cache)
        settings_menu.addAction(clear_action)

    def _on_add_folder(self) -> None:
        from qtpy.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Add folder")
        if path:
            self._add_source(path)

    def _on_refresh_sources(self) -> None:
        roots = [s.path for s in self.sources]
        # Remove + re-add each root: keeps API simple, may re-enqueue cached jobs
        # which the loader skips quickly via cache.load.
        for path in roots:
            self._remove_source(path)
        for path in roots:
            self._add_source(path)

    def _on_clear_cache(self) -> None:
        from qtpy.QtWidgets import QMessageBox
        from .. import cache

        reply = QMessageBox.question(
            self, "Clear MIP cache",
            f"Delete the MIP cache at {cache.CACHE_DIR}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            cache.clear_all()
            self.status.setText("MIP cache cleared.")

    def _set_square_footprint(self, checked: bool) -> None:
        self.square_footprint = checked
        self._apply_label_sizes()
        for (acq_id, t, fov, ch_idx, axis), ax_mip in self.mip_data.items():
            if axis != self.view_axis:
                continue
            self._render_thumb(acq_id, t, fov, ch_idx, ax_mip)

    def _set_expanded_fov_mode(self, checked: bool) -> None:
        self.expanded_fov_mode = checked
        self._rebuild_rows()

    def _set_expanded_region_mode(self, checked: bool) -> None:
        self.expanded_region_mode = checked
        if self.view_mode == "region":
            self._rebuild_rows()
            # Eagerly enqueue prereqs for any region that's now visible.
            for acq_id, acq in enumerate(self.acquisitions):
                if acq is None or len(acq.regions) <= 1:
                    continue
                regions_now_visible = (
                    acq.regions if self.expanded_region_mode else [acq.selected_region]
                )
                for region in regions_now_visible:
                    self._enqueue_region_prereqs(
                        acq_id, acq, acq.selected_timepoint, region
                    )

    @staticmethod
    def _toggle_style() -> str:
        return (
            "QPushButton { background-color: #333; color: #ccc; border: 1px solid #444;"
            " border-radius: 3px; font-size: 10px; padding: 2px 8px; }"
            "QPushButton:hover { background-color: #444; }"
            "QPushButton:checked { background-color: #2d5aa0; color: white; border-color: #3a6fc0; }"
        )

    # ── drag-drop ──

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local and os.path.isdir(local):
                self._add_source(local)
        event.acceptProposedAction()

    # ── source management ──

    def _add_source(self, path: str) -> None:
        real = os.path.realpath(path)
        if any(os.path.realpath(s.path) == real for s in self.sources):
            return  # already loaded
        new_acqs = scan.ingest(path)
        if not new_acqs:
            self.status.setText(f"No acquisitions found in {path}")
            return
        first_id = len(self.acquisitions)
        for acq in new_acqs:
            self.acquisitions.append(acq)
        ids = list(range(first_id, first_id + len(new_acqs)))
        self.sources.append(Source(path=path, acq_ids=ids))
        self.empty_overlay.hide()
        self.sources_panel.show()
        self._sync_sources_panel()
        self._rebuild_mag_filter()
        self._refresh_region_button_enabled()
        self._rebuild_rows()
        for acq_id, acq in zip(ids, new_acqs):
            self._enqueue_jobs_for_acq(
                acq_id, acq, acq.selected_timepoint, acq.selected_fov
            )

    def _remove_source(self, path: str) -> None:
        target = next((s for s in self.sources if s.path == path), None)
        if target is None:
            return
        for acq_id in target.acq_ids:
            self.loader.cancel(acq_id)
        for acq_id in target.acq_ids:
            self.acquisitions[acq_id] = None  # type: ignore[assignment]
        self.sources.remove(target)
        removed_ids = set(target.acq_ids)
        self.region_mip_data = {
            k: v for k, v in self.region_mip_data.items()
            if k[0] not in removed_ids
        }
        self._region_fov_readiness = {
            k: v for k, v in self._region_fov_readiness.items()
            if k[0] not in removed_ids
        }
        self._sync_sources_panel()
        self._rebuild_mag_filter()
        self._refresh_region_button_enabled()
        self._rebuild_rows()
        if not self.sources:
            self.empty_overlay.show()
            self.sources_panel.hide()

    def _sync_sources_panel(self) -> None:
        self.sources_panel.set_sources([(s.path, len(s.acq_ids)) for s in self.sources])

    def _enqueue_jobs_for_acq(
        self, acq_id: int, acq: Acquisition, timepoint: str, fov: str,
    ) -> None:
        for ch_idx, channel in enumerate(acq.channels):
            self.loader.enqueue(
                Job(
                    acq_id=acq_id, acq=acq, fov=fov, channel=channel,
                    ch_idx=ch_idx, timepoint=timepoint,
                )
            )

    # ── stub hooks filled in by later tasks ──

    def _rebuild_mag_filter(self) -> None:
        # Collect distinct mags from current acquisitions
        mags = set()
        for acq in self.acquisitions:
            if acq is None:
                continue
            mag = resolve_mag(acq.folder_name, acq.params)
            if mag is not None:
                mags.add(mag)

        # Tear down old checkboxes
        for cb in list(self.mag_checkboxes.values()):
            self.mag_row_layout.removeWidget(cb)
            cb.deleteLater()
        self.mag_checkboxes.clear()

        if len(mags) <= 1:
            return  # hide the filter when only one mag is present

        for mag in sorted(mags):
            cb = QCheckBox(f"{mag}x")
            cb.setChecked(True)
            cb.setStyleSheet(
                "QCheckBox { color: #ccc; font-size: 11px; spacing: 6px; padding-right: 10px; }"
                "QCheckBox::indicator { width: 14px; height: 14px; }"
            )
            cb.stateChanged.connect(self._refresh_visibility)
            self.mag_row_layout.addWidget(cb)
            self.mag_checkboxes[mag] = cb

    def _refresh_visibility(self) -> None:
        active_mags = {m for m, cb in self.mag_checkboxes.items() if cb.isChecked()}
        hide_thin = self.hide_thin_btn.isChecked()

        # Pass 1: compute visibility per row (don't trust ``isVisible()``,
        # which depends on the parent chain that we're about to update).
        row_visible: dict[tuple[int, str, str], bool] = {}
        for (acq_id, t, fov), rw in self.row_widgets.items():
            acq = self.acquisitions[acq_id]
            visible = acq is not None
            if visible and self.mag_checkboxes:
                mag = resolve_mag(acq.folder_name, acq.params)
                visible = mag in active_mags if mag is not None else True
            if visible and hide_thin and acq.shape_zyx is not None:
                visible = acq.shape_zyx[0] >= THIN_Z_THRESHOLD
            row_visible[(acq_id, t, fov)] = visible
            rw.container.setVisible(visible)

        # Pass 2: hide source groups whose rows are all hidden.
        total_visible = sum(1 for v in row_visible.values() if v)
        for grp_idx, src in enumerate(self.sources):
            if grp_idx >= len(self.source_groups):
                continue
            any_visible = any(
                row_visible.get((acq_id, self.acquisitions[acq_id].selected_timepoint, unit), False)
                for acq_id in src.acq_ids
                if self.acquisitions[acq_id] is not None
                for unit in self._row_units_for(acq_id)
            )
            self.source_groups[grp_idx].setVisible(any_visible)

        # If a load has settled and the filters wipe everything, tell the
        # user — otherwise the window looks broken. When filters loosen and
        # rows reappear, replace the "all hidden" message rather than leave
        # it stale on the status bar.
        total_rows = len(self.row_widgets)
        unit_label = "regions" if self.view_mode == "region" else (
            "FOVs" if self.expanded_fov_mode else "acquisitions"
        )
        if total_rows and total_visible == 0:
            reasons = []
            if hide_thin:
                reasons.append('"Hide thin (Z<5)" is on')
            if self.mag_checkboxes and not active_mags:
                reasons.append("no magnification selected")
            why = " and ".join(reasons) if reasons else "filters"
            self.status.setText(
                f"All {total_rows} {unit_label} hidden by {why}."
            )
        elif total_rows:
            self.status.setText(
                f"{total_visible}/{total_rows} {unit_label} visible"
            )

    def _row_units_for(self, acq_id: int) -> list[str]:
        """Return the row-unit ids for this acquisition under the current view.

        FOV view: one or all FOVs (composite ``<region>_<fov>``).
        Region view: one or all regions.
        """
        acq = self.acquisitions[acq_id]
        if acq is None:
            return []
        if self.view_mode == "region":
            return acq.regions if self.expanded_region_mode else [acq.selected_region]
        return acq.fovs if self.expanded_fov_mode else [acq.selected_fov]

    # ── row rendering ──

    def _rebuild_rows(self) -> None:
        # Clear existing rows AND existing source groups
        for rw in self.row_widgets.values():
            rw.container.deleteLater()
        self.row_widgets.clear()
        self.row_keys = []
        for grp in self.source_groups:
            self.scroll_layout.removeWidget(grp)
            grp.deleteLater()
        self.source_groups = []

        # Compute the active wavelength column set (drop any column no row uses)
        active_wls: list[str] = self._active_wavelengths()

        # One QGroupBox per source root, in drop order. Acquisitions in a
        # group are listed in the order they were ingested.
        insert_at = self.scroll_layout.count() - 1  # before trailing stretch
        for src_idx, src in enumerate(self.sources):
            group = self._make_source_group(src, src_idx, active_wls)
            self.source_groups.append(group)
            self.scroll_layout.insertWidget(insert_at, group)
            insert_at += 1

        # Re-render any thumbs we already have data for
        for (acq_id, t, fov, ch_idx, axis), ax_mip in list(self.mip_data.items()):
            if axis != self.view_axis:
                continue
            self._render_thumb(acq_id, t, fov, ch_idx, ax_mip)

        self._refresh_visibility()
        self._apply_label_sizes()

    def _make_source_group(self, src: "Source", src_idx: int, active_wls: list[str]) -> QGroupBox:
        bg = "#2a2a2a" if src_idx % 2 == 0 else "#1a1a1a"
        title = os.path.basename(os.path.normpath(src.path)) or src.path
        group = QGroupBox(f"{title}  ({len(src.acq_ids)})")
        group.setToolTip(src.path)
        # Only style the group itself + its title — do NOT add a
        # ``QGroupBox QWidget { background-color: transparent }`` cascade.
        # On macOS Qt that cascade can blank out QLabel pixmaps and other
        # widgets that rely on opaque painting; we explicitly transparent
        # the row container instead (see _make_row_widget).
        group.setStyleSheet(
            f"QGroupBox {{ font-size: 13px; font-weight: bold; border: 1px solid #333;"
            f" border-radius: 6px; margin-top: 8px; padding: 6px 4px 4px 4px;"
            f" color: #ddd; background-color: {bg}; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;"
            f" background-color: {bg}; color: #ddd; }}"
        )
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)
        group_layout.setSpacing(2)

        for acq_id in src.acq_ids:
            acq = self.acquisitions[acq_id]
            if acq is None:
                continue
            t = acq.selected_timepoint
            for unit in self._row_units_for(acq_id):
                key = RowKey(acq_id, t, unit)
                self.row_keys.append(key)
                row = self._make_row_widget(key, acq, active_wls)
                self.row_widgets[(acq_id, t, unit)] = row
                group_layout.addWidget(row.container)

        return group

    def _active_wavelengths(self) -> list[str]:
        seen: set[str] = set()
        for acq in self.acquisitions:
            if acq is None:
                continue
            for ch in acq.channels:
                seen.add(ch.wavelength)
        ordered = [wl for wl in CHANNEL_ORDER if wl in seen]
        extras = sorted(
            [wl for wl in seen if wl not in CHANNEL_ORDER],
            key=lambda w: int(w) if w.isdigit() else 999,
        )
        return ordered + extras

    def _make_row_widget(self, key, acq, active_wls):
        container = QWidget()
        # Row container is transparent so the parent group's bg colour shows
        # through. We set this directly instead of via a ``QGroupBox QWidget``
        # cascade to avoid blanking QLabel pixmaps on macOS Qt.
        container.setStyleSheet("background-color: transparent;")
        h = QHBoxLayout(container)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(4)

        unit = key.fov  # in region view, this is the region id
        is_region_view = self.view_mode == "region"

        mag = resolve_mag(acq.folder_name, acq.params) or "?"
        mag_lbl = QLabel(f"{mag}x" if isinstance(mag, int) else str(mag))
        mag_lbl.setFixedWidth(30)
        mag_lbl.setStyleSheet("color: #ccc; font-size: 11px; font-weight: bold;")
        h.addWidget(mag_lbl)

        ts = parse_timestamp(acq.folder_name)
        time_lbl = QLabel(f"{ts[0]}\n{ts[1]}" if ts else "")
        time_lbl.setFixedWidth(40)
        time_lbl.setAlignment(Qt.AlignCenter)
        time_lbl.setStyleSheet("color: #888; font-size: 9px;")
        h.addWidget(time_lbl)

        # Per-row unit label (visible in expanded mode for either view).
        expanded = self.expanded_region_mode if is_region_view else self.expanded_fov_mode
        nunits = len(acq.regions if is_region_view else acq.fovs)
        if expanded and nunits > 1:
            label = f"Region {unit}" if is_region_view else f"FOV {display_fov(unit)}"
            unit_lbl = QLabel(label)
            unit_lbl.setFixedWidth(60)
            unit_lbl.setStyleSheet("color: #888; font-size: 10px;")
            h.addWidget(unit_lbl)

        # Compact name column — analogous to the explorer's "device" column.
        # Multi-line wrap so the layout stays tight; full path in tooltip.
        name_lbl = QLabel(acq.display_name)
        name_lbl.setFixedWidth(80)
        name_lbl.setWordWrap(True)
        name_lbl.setToolTip(acq.path)
        name_lbl.setStyleSheet("color: #ccc; font-size: 9px; font-weight: bold;")
        h.addWidget(name_lbl)

        # One column per active wavelength
        thumb_labels: dict[int, QLabel] = {}
        thumb_columns: dict[str, QLabel] = {}
        ch_by_wl = {ch.wavelength: (i, ch) for i, ch in enumerate(acq.channels)}
        for wl in active_wls:
            col = QVBoxLayout()
            col.setSpacing(1)
            color = rgb_for(wl)
            ch_lbl = QLabel(wl if wl in ch_by_wl else "")
            ch_lbl.setAlignment(Qt.AlignCenter)
            ch_lbl.setFixedHeight(14)
            if wl in ch_by_wl:
                ch_lbl.setStyleSheet(
                    f"color: rgb({color[0]},{color[1]},{color[2]}); font-size: 9px;"
                )
            else:
                ch_lbl.setStyleSheet("color: transparent; font-size: 9px;")
            col.addWidget(ch_lbl)

            thumb = QLabel()
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setFixedSize(self.thumb_size, self.thumb_size)
            if wl in ch_by_wl:
                thumb.setStyleSheet(
                    "background-color: #222; border: 1px solid #2a2a2a; border-radius: 3px;"
                )
                thumb_labels[ch_by_wl[wl][0]] = thumb
            else:
                thumb.setStyleSheet("background-color: transparent; border: none;")
            thumb_columns[wl] = thumb
            col.addWidget(thumb)
            h.addLayout(col)

        h.addStretch()

        # Time picker (only for multi-timepoint acquisitions; both views use it).
        time_combo: QComboBox | None = None
        if len(acq.timepoints) > 1:
            time_combo = QComboBox()
            _style_combo(time_combo)
            for t in acq.timepoints:
                time_combo.addItem(f"t={t}", t)
            _size_combo_to_contents(time_combo)
            time_combo.setCurrentIndex(acq.timepoints.index(acq.selected_timepoint))
            time_combo.currentIndexChanged.connect(
                lambda i, k=key, c=time_combo: self._on_timepoint_changed(k, c.itemData(i))
            )
            h.addWidget(time_combo)

        # Unit picker (FOV combo in FOV view, Region combo in region view).
        unit_combo: QComboBox | None = None
        if not expanded:
            if is_region_view and len(acq.regions) > 1:
                unit_combo = QComboBox()
                _style_combo(unit_combo)
                for r in acq.regions:
                    unit_combo.addItem(f"Region {r}", r)
                _size_combo_to_contents(unit_combo)
                unit_combo.setCurrentIndex(acq.regions.index(acq.selected_region))
                unit_combo.currentIndexChanged.connect(
                    lambda i, k=key, c=unit_combo: self._on_region_changed(
                        k, c.itemData(i)
                    )
                )
                h.addWidget(unit_combo)
            elif (not is_region_view) and len(acq.fovs) > 1:
                unit_combo = QComboBox()
                _style_combo(unit_combo)
                for fov in acq.fovs:
                    unit_combo.addItem(f"FOV {display_fov(fov)}", fov)
                _size_combo_to_contents(unit_combo)
                unit_combo.setCurrentIndex(acq.fovs.index(acq.selected_fov))
                unit_combo.currentIndexChanged.connect(
                    lambda i, k=key, c=unit_combo: self._on_fov_changed(
                        k, c.itemData(i)
                    )
                )
                h.addWidget(unit_combo)

        # Action buttons — vertical stack to match explorer layout.
        # 'Open 3D View' is suppressed in region view (stitched mosaics are 2D).
        btn_col = QVBoxLayout()
        btn_col.setSpacing(2)
        if not is_region_view:
            btn_3d = QPushButton("Open 3D View")
            btn_3d.setFixedSize(120, 30)
            btn_3d.setCursor(Qt.PointingHandCursor)
            btn_3d.setStyleSheet(
                "QPushButton { background-color: #2d5aa0; color: white; border-radius: 4px;"
                " font-size: 11px; font-weight: bold; }"
                "QPushButton:hover { background-color: #3a6fc0; }"
            )
            btn_3d.clicked.connect(lambda _, k=key: self._open_napari(k))
            btn_col.addWidget(btn_3d)

        btn_lut = QPushButton("Adjust Contrast")
        btn_lut.setFixedSize(120, 30)
        btn_lut.setCursor(Qt.PointingHandCursor)
        btn_lut.setStyleSheet(
            "QPushButton { background-color: #555; color: white; border-radius: 4px;"
            " font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background-color: #777; }"
        )
        btn_lut.clicked.connect(lambda _, k=key: self._adjust_lut(k))
        btn_col.addWidget(btn_lut)

        h.addLayout(btn_col)

        return RowWidgets(
            container=container,
            mag_lbl=mag_lbl,
            time_lbl=time_lbl,
            name_lbl=name_lbl,
            thumb_labels=thumb_labels,
            thumb_columns=thumb_columns,
            unit_combo=unit_combo,
            time_combo=time_combo,
        )

    def _on_fov_changed(self, key, new_fov: str) -> None:
        acq = self.acquisitions[key.acq_id]
        acq.selected_fov = new_fov
        # Re-key the row widget; the easiest is to rebuild rows since rows are cheap
        self._rebuild_rows()
        # Enqueue jobs for the new FOV's channels (cache-aware; loader will skip cached)
        self._enqueue_jobs_for_acq(
            key.acq_id, acq, acq.selected_timepoint, new_fov
        )

    def _on_region_changed(self, key, new_region: str) -> None:
        acq = self.acquisitions[key.acq_id]
        acq.selected_region = new_region
        self._rebuild_rows()
        # Region view eagerly enqueues per-FOV MIPs for all FOVs in the
        # selected region; readiness is tracked in _region_fov_readiness.
        if self.view_mode == "region":
            self._enqueue_region_prereqs(
                key.acq_id, acq, acq.selected_timepoint, new_region
            )

    def _enqueue_region_prereqs(
        self, acq_id: int, acq, timepoint: str, region: str
    ) -> None:
        """For region view: enqueue per-FOV Z-MIP jobs for every FOV in
        ``region``, and register a readiness set we tick off as
        ``_on_mip_ready`` lands MIPs. When the set is complete we enqueue
        a ``RegionStitchJob`` per channel.

        Cache hits short-circuit through the loader as usual; this method is
        cheap to call repeatedly.
        """
        # FOVs in this region (composite '<region>_<fov>' filter on acq.fovs).
        fovs_in_region = [f for f in acq.fovs if f.split("_", 1)[0] == region]
        if not fovs_in_region:
            return
        key = (acq_id, timepoint, region)
        needed: set[tuple[int, str]] = set()
        for ch_idx in range(len(acq.channels)):
            for fov in fovs_in_region:
                needed.add((ch_idx, fov))
        self._region_fov_readiness[key] = needed.copy()
        # Track what's already in mip_data so the stitch trigger doesn't wait
        # for jobs that already completed before we switched modes.
        already = {
            (ci, f)
            for (a, t, f, ci, ax) in self.mip_data
            if a == acq_id and t == timepoint and ax == "z" and (
                ci, f
            ) in needed
        }
        self._region_fov_readiness[key] -= already

        # Enqueue any missing FOV jobs (loader skips cached ones in O(1)).
        for ch_idx, channel in enumerate(acq.channels):
            for fov in fovs_in_region:
                if (ch_idx, fov) in already:
                    continue
                self.loader.enqueue(Job(
                    acq_id=acq_id, acq=acq, fov=fov, channel=channel,
                    ch_idx=ch_idx, timepoint=timepoint,
                ))

        # If all FOV MIPs were already in memory, fire stitches immediately.
        if not self._region_fov_readiness[key]:
            for ch_idx, channel in enumerate(acq.channels):
                self._dispatch_region_stitch(
                    acq_id, acq, timepoint, region, ch_idx, channel
                )

    def _dispatch_region_stitch(
        self, acq_id: int, acq, timepoint: str, region: str,
        ch_idx: int, channel,
    ) -> None:
        # Don't support region view for handlers that can't supply coords or
        # a region cache key.
        coords_map = (
            acq.handler._load_coords(acq)
            if hasattr(acq.handler, "_load_coords")
            else None
        )
        if not coords_map or region not in coords_map:
            self.status.setText(
                f"{acq.display_name}: region view needs coordinates.csv"
            )
            return
        coords = coords_map[region]

        fov_mips = {
            f: self.mip_data[(a, t, f, ci, ax)].mip
            for (a, t, f, ci, ax) in list(self.mip_data)
            if a == acq_id and t == timepoint and ci == ch_idx and ax == "z"
            and f.split("_", 1)[0] == region
        }
        if not fov_mips:
            return
        self.loader.enqueue_region(RegionStitchJob(
            acq_id=acq_id, acq=acq, region=region,
            channel=channel, ch_idx=ch_idx, timepoint=timepoint,
            fov_mips=fov_mips, coords=coords,
        ))

    def _on_region_mip_ready(
        self, acq_id, timepoint, region, ch_idx, wavelength, ax_mip,
    ) -> None:
        if acq_id >= len(self.acquisitions) or self.acquisitions[acq_id] is None:
            return
        self.region_mip_data[(acq_id, timepoint, region, ch_idx)] = ax_mip
        if self.view_mode == "region":
            self._render_region_thumb(acq_id, timepoint, region, ch_idx, ax_mip)

    def _on_timepoint_changed(self, key, new_timepoint: str) -> None:
        acq = self.acquisitions[key.acq_id]
        acq.selected_timepoint = new_timepoint
        self._rebuild_rows()
        if self.view_mode == "region":
            regions_to_load = (
                acq.regions if self.expanded_region_mode
                else [acq.selected_region]
            )
            for region in regions_to_load:
                self._enqueue_region_prereqs(
                    key.acq_id, acq, new_timepoint, region
                )
        else:
            self._enqueue_jobs_for_acq(
                key.acq_id, acq, new_timepoint, acq.selected_fov
            )

    # ── physical aspect and label sizing ──

    def _phys_aspect(self, acq_id, fov, axis: str) -> float:
        """Return physical_height / physical_width for the given axis."""
        acq = self.acquisitions[acq_id]
        if acq is None:
            return 1.0
        sensor_pixel_um = acq.params.get("sensor_pixel_size_um", 6.5)
        mag = resolve_mag(acq.folder_name, acq.params) or 1
        pixel_um = sensor_pixel_um / mag if mag else sensor_pixel_um
        dz_um = acq.params.get("dz(um)", pixel_um)
        shape = acq.shape_zyx
        if shape is None:
            return 1.0
        nz, ny, nx = shape
        if axis == "z":
            return (ny * pixel_um) / max(nx * pixel_um, 1e-9)
        if axis == "y":
            return (nz * dz_um) / max(nx * pixel_um, 1e-9)
        if axis == "x":
            return (nz * dz_um) / max(ny * pixel_um, 1e-9)
        return 1.0

    def _row_label_size(self, acq_id, fov) -> tuple[int, int]:
        if self.square_footprint:
            return self.thumb_size, self.thumb_size
        aspect = self._phys_aspect(acq_id, fov, self.view_axis)
        return self.thumb_size, max(20, int(round(self.thumb_size * aspect)))

    def _image_render_size(
        self, acq_id, fov, cell_w: int, cell_h: int
    ) -> tuple[int, int]:
        """Pixmap size that fits in the cell while preserving physical aspect.

        In non-square mode the cell already matches physical aspect so this
        returns ``(cell_w, cell_h)``. In square_footprint mode it returns a
        letterboxed size; the QLabel's center alignment fills the rest.
        """
        aspect = self._phys_aspect(acq_id, fov, self.view_axis)
        cell_aspect = cell_h / max(cell_w, 1)
        if cell_aspect >= aspect:
            return cell_w, max(1, int(round(cell_w * aspect)))
        return max(1, int(round(cell_h / max(aspect, 1e-9)))), cell_h

    def _apply_label_sizes(self) -> None:
        for (acq_id, timepoint, fov), rw in self.row_widgets.items():
            self._apply_label_sizes_for(acq_id, timepoint, fov, rw)

    def _apply_label_sizes_for(self, acq_id, timepoint, fov, rw=None) -> None:
        """Resize thumb cells in one row only — used when shape_zyx becomes
        known and only that row's aspect changes."""
        if rw is None:
            rw = self.row_widgets.get((acq_id, timepoint, fov))
            if rw is None:
                return
        w, h = self._row_label_size(acq_id, fov)
        for thumb in rw.thumb_columns.values():
            thumb.setFixedSize(w, h)

    def _render_thumb(self, acq_id, timepoint, fov, ch_idx, ax_mip) -> None:
        rw = self.row_widgets.get((acq_id, timepoint, fov))
        if rw is None or ch_idx not in rw.thumb_labels:
            return
        acq = self.acquisitions[acq_id]
        wl = acq.channels[ch_idx].wavelength
        rgba = mip_to_rgba(ax_mip.mip, ax_mip.p1, ax_mip.p999, rgb_for(wl))
        h, w = rgba.shape[:2]
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        cell_w, cell_h = self._row_label_size(acq_id, fov)
        img_w, img_h = self._image_render_size(acq_id, fov, cell_w, cell_h)
        pixmap = QPixmap.fromImage(qimg).scaled(
            img_w, img_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        rw.thumb_labels[ch_idx].setPixmap(pixmap)
        rw.thumb_labels[ch_idx].setStyleSheet(
            "background-color: #111; border: 1px solid #2a2a2a; border-radius: 3px;"
        )

    def _render_region_thumb(
        self, acq_id, timepoint, region, ch_idx, ax_mip
    ) -> None:
        rw = self.row_widgets.get((acq_id, timepoint, region))
        if rw is None or ch_idx not in rw.thumb_labels:
            return
        acq = self.acquisitions[acq_id]
        wl = acq.channels[ch_idx].wavelength
        rgba = mip_to_rgba(ax_mip.mip, ax_mip.p1, ax_mip.p999, rgb_for(wl))
        h, w = rgba.shape[:2]
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        # Region thumbs scale to fit the cell, preserving the mosaic aspect.
        cell = self.thumb_size
        cell_aspect = 1.0
        aspect = h / max(w, 1)
        if aspect >= cell_aspect:
            img_w, img_h = max(1, int(round(cell / max(aspect, 1e-9)))), cell
        else:
            img_w, img_h = cell, max(1, int(round(cell * aspect)))
        pixmap = QPixmap.fromImage(qimg).scaled(
            img_w, img_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        rw.thumb_labels[ch_idx].setPixmap(pixmap)
        rw.thumb_labels[ch_idx].setStyleSheet(
            "background-color: #111; border: 1px solid #2a2a2a; border-radius: 3px;"
        )

    # ── loader callbacks ──

    def _on_mip_ready(
        self, acq_id, timepoint, fov, ch_idx, wavelength, channel_mips, shape,
    ) -> None:
        if acq_id >= len(self.acquisitions) or self.acquisitions[acq_id] is None:
            return
        acq = self.acquisitions[acq_id]
        shape_was_unknown = acq.shape_zyx is None
        if shape is not None and shape_was_unknown:
            acq.shape_zyx = shape
        for axis, ax_mip in channel_mips.items():
            self.mip_data[(acq_id, timepoint, fov, ch_idx, axis)] = ax_mip
        if self.view_axis in channel_mips and self.view_mode == "fov":
            self._render_thumb(
                acq_id, timepoint, fov, ch_idx, channel_mips[self.view_axis]
            )
        # Aspect may have become known for this row; resize only this row.
        if shape_was_unknown and acq.shape_zyx is not None:
            self._apply_label_sizes_for(acq_id, timepoint, fov)

        # Region-view readiness: tick this (ch_idx, fov) off the pending set;
        # if the set is now empty, enqueue a stitch.
        if self.view_mode == "region" and "z" in channel_mips:
            region = fov.split("_", 1)[0] if "_" in fov else "0"
            key = (acq_id, timepoint, region)
            pending = self._region_fov_readiness.get(key)
            if pending is not None:
                pending.discard((ch_idx, fov))
                if not pending:
                    # All FOV MIPs for this region are in — stitch every channel.
                    for ci, channel in enumerate(acq.channels):
                        self._dispatch_region_stitch(
                            acq_id, acq, timepoint, region, ci, channel
                        )
                    # Drop the readiness entry so we don't re-stitch.
                    self._region_fov_readiness.pop(key, None)

    def _on_progress(self, done, queued, message) -> None:
        self.status.setText(f"{done}/{queued} channels — {message}")

    def _on_idle(self) -> None:
        loaded = sum(1 for a in self.acquisitions if a is not None)
        self.status.setText(f"{loaded} acquisitions loaded")

    # ── display controls ──

    def _set_axis(self, axis: str) -> None:
        if axis == self.view_axis:
            return
        self.view_axis = axis
        self._apply_label_sizes()
        for (acq_id, t, fov, ch_idx, ax), ax_mip in self.mip_data.items():
            if ax != axis:
                continue
            self._render_thumb(acq_id, t, fov, ch_idx, ax_mip)

    def _set_view_mode(self, mode: str) -> None:
        if mode == self.view_mode:
            return
        self.view_mode = mode
        # XZ/YZ stitched MIPs aren't supported; force XY in region view.
        if mode == "region":
            if self.view_axis != "z":
                self.view_axis = "z"
                self.axis_buttons["z"].setChecked(True)
            self.axis_buttons["y"].setEnabled(False)
            self.axis_buttons["x"].setEnabled(False)
        else:
            self.axis_buttons["y"].setEnabled(True)
            self.axis_buttons["x"].setEnabled(True)
        self._rebuild_rows()
        if mode == "region":
            for acq_id, acq in enumerate(self.acquisitions):
                if acq is None:
                    continue
                for region in (
                    acq.regions if self.expanded_region_mode else [acq.selected_region]
                ):
                    self._enqueue_region_prereqs(
                        acq_id, acq, acq.selected_timepoint, region
                    )
            # Render any cached region thumbs we already have data for.
            for (acq_id, t, region, ch_idx), ax_mip in self.region_mip_data.items():
                self._render_region_thumb(acq_id, t, region, ch_idx, ax_mip)

    def _refresh_region_button_enabled(self) -> None:
        """Region button is enabled iff at least one loaded acquisition has
        more than one region."""
        any_multi_region = any(
            acq is not None and len(acq.regions) > 1
            for acq in self.acquisitions
        )
        self.view_buttons["region"].setEnabled(any_multi_region)
        if any_multi_region:
            self.view_buttons["region"].setToolTip("")
        else:
            self.view_buttons["region"].setToolTip(
                "No source supports region view"
            )
            if self.view_mode == "region":
                # Source removed; fall back to FOV view.
                self.view_buttons["fov"].setChecked(True)
                self._set_view_mode("fov")

    def _set_thumb_size(self, size: int) -> None:
        self.thumb_size = size
        self._apply_label_sizes()
        for (acq_id, t, fov, ch_idx, axis), ax_mip in self.mip_data.items():
            if axis != self.view_axis:
                continue
            self._render_thumb(acq_id, t, fov, ch_idx, ax_mip)

    def _open_napari(self, key) -> None:
        from .viewer3d import open_napari

        acq = self.acquisitions[key.acq_id]

        def lut_lookup(ch_idx, axis):
            ax = self.view_axis if axis == "current" else axis
            entry = self.mip_data.get(
                (key.acq_id, key.timepoint, key.fov, ch_idx, ax)
            )
            if entry is None:
                return None
            return float(entry.p1), float(entry.p999)

        open_napari(acq, key.fov, lut_lookup)

    def _adjust_lut(self, key) -> None:
        from .lut_dialog import show_lut_dialog

        acq = self.acquisitions[key.acq_id]
        is_region = self.view_mode == "region"

        if is_region:
            mip_data = {
                (a, t, u, ci, "z"): ax_mip
                for (a, t, u, ci), ax_mip in self.region_mip_data.items()
            }
            key_fn = lambda a, u, c, t: a.handler.cache_key_region(
                a, u, c, timepoint=t
            )
            unit_label = "Region"
            refresh = self._render_region_thumb
        else:
            mip_data = self.mip_data
            key_fn = None
            unit_label = "FOV"
            refresh = self._render_thumb

        show_lut_dialog(
            parent=self,
            acq=acq,
            unit=key.fov,
            timepoint=key.timepoint,
            axis=self.view_axis,
            mip_data=mip_data,
            refresh_thumb=refresh,
            acq_id=key.acq_id,
            key_fn=key_fn,
            unit_label=unit_label,
        )

        # In region view, mip_data was a local copy; push p1/p999 changes
        # back to self.region_mip_data so the next thumb render picks them up.
        if is_region:
            for (a, t, u, ci, _), ax_mip in mip_data.items():
                self.region_mip_data[(a, t, u, ci)] = ax_mip

    # ── shutdown ──

    def closeEvent(self, event) -> None:  # noqa: N802
        self.loader.stop()
        self.loader.wait(2000)
        super().closeEvent(event)
