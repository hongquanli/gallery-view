"""Gallery window: drag-drop ingestion, sources strip, scroll area, loader."""

import os
from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import scan
from ..loader import Job, MipLoader
from ..types import Acquisition

THIN_Z_THRESHOLD = 5
THUMB_SIZE_PRESETS = [("Small", 80), ("Medium", 160), ("Large", 320)]
DEFAULT_THUMB_SIZE = 160


@dataclass
class Source:
    path: str
    acq_ids: list[int]


class GalleryWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("gallery-view")
        self.setMinimumSize(1100, 700)
        self.setAcceptDrops(True)
        self.setStyleSheet(
            "QMainWindow { background-color: #1a1a1a; color: white; }"
            "QLabel { color: white; }"
        )

        self.acquisitions: list[Acquisition] = []
        self.sources: list[Source] = []

        # Loader thread (long-lived)
        self.loader = MipLoader()
        self.loader.mip_ready.connect(self._on_mip_ready)
        self.loader.progress.connect(self._on_progress)
        self.loader.idle.connect(self._on_idle)
        self.loader.start()

        self._build_ui()

    # ── ui scaffolding ──

    def _build_ui(self) -> None:
        from .sources_panel import SourcesPanel

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

        self.status = QLabel("Drop folders to begin.")
        self.status.setStyleSheet("color: #888; font-size: 11px; padding: 2px 4px;")
        layout.addWidget(self.status)

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
        size_lbl = QLabel("Thumbnail size:")
        size_lbl.setStyleSheet("color: #888; font-size: 11px;")
        row.addWidget(size_lbl)
        self.size_combo = QComboBox()
        for label, size in THUMB_SIZE_PRESETS:
            self.size_combo.addItem(label, size)
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
        self._rebuild_rows()
        for acq_id, acq in zip(ids, new_acqs):
            self._enqueue_jobs_for_acq(acq_id, acq, acq.selected_fov)

    def _remove_source(self, path: str) -> None:
        target = next((s for s in self.sources if s.path == path), None)
        if target is None:
            return
        for acq_id in target.acq_ids:
            self.loader.cancel(acq_id)
        for acq_id in target.acq_ids:
            self.acquisitions[acq_id] = None  # type: ignore[assignment]
        self.sources.remove(target)
        self._sync_sources_panel()
        self._rebuild_mag_filter()
        self._rebuild_rows()
        if not self.sources:
            self.empty_overlay.show()
            self.sources_panel.hide()

    def _sync_sources_panel(self) -> None:
        self.sources_panel.set_sources([(s.path, len(s.acq_ids)) for s in self.sources])

    def _enqueue_jobs_for_acq(self, acq_id: int, acq: Acquisition, fov: str) -> None:
        for ch_idx, channel in enumerate(acq.channels):
            self.loader.enqueue(
                Job(acq_id=acq_id, acq=acq, fov=fov, channel=channel, ch_idx=ch_idx)
            )

    # ── stub hooks filled in by later tasks ──

    def _rebuild_mag_filter(self) -> None:
        # Filled in by Task 20.
        pass

    def _rebuild_rows(self) -> None:
        # Filled in by Task 19.
        pass

    def _refresh_visibility(self) -> None:
        # Filled in by Task 20.
        pass

    def _set_axis(self, axis: str) -> None:
        # Filled in by Task 19.
        self.view_axis = axis

    def _set_thumb_size(self, size: int) -> None:
        # Filled in by Task 19.
        self.thumb_size = size

    # ── loader callbacks (stubs filled in by Task 19) ──

    def _on_mip_ready(self, acq_id, fov, ch_idx, wavelength, channel_mips, shape) -> None:
        pass

    def _on_progress(self, done, queued, message) -> None:
        self.status.setText(f"{done}/{queued} channels — {message}")

    def _on_idle(self) -> None:
        loaded = sum(1 for a in self.acquisitions if a is not None)
        self.status.setText(f"{loaded} acquisitions loaded")

    # ── shutdown ──

    def closeEvent(self, event) -> None:  # noqa: N802
        self.loader.stop()
        self.loader.wait(2000)
        super().closeEvent(event)
