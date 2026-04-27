"""Top-of-window strip showing dropped source roots.

Each source is a chip with the basename, the acq count, and an × to remove.
The right-most ``+`` chip opens a folder picker (handled by the gallery
window via the ``add_requested`` signal)."""

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
)


class SourcesPanel(QScrollArea):
    remove_requested = Signal(str)  # source path
    add_requested = Signal(str)     # source path picked from dialog

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFixedHeight(38)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea { border: none; background: #1a1a1a; }")

        self._content = QWidget()
        self._row = QHBoxLayout(self._content)
        self._row.setContentsMargins(6, 4, 6, 4)
        self._row.setSpacing(6)
        self.setWidget(self._content)

        self._add_btn = QPushButton("+ Add folder…")
        self._add_btn.setCursor(Qt.PointingHandCursor)
        self._add_btn.setStyleSheet(
            "QPushButton { background-color: #333; color: #ccc; border: 1px dashed #555;"
            " border-radius: 12px; padding: 2px 10px; font-size: 11px; }"
            "QPushButton:hover { background-color: #444; }"
        )
        self._add_btn.clicked.connect(self._on_add_clicked)
        self._row.addWidget(self._add_btn)
        self._row.addStretch()

        self._chips: dict[str, QWidget] = {}

    def set_sources(self, sources: list[tuple[str, int]]) -> None:
        """Replace the chip set. ``sources`` is a list of ``(path, acq_count)``."""
        for chip in list(self._chips.values()):
            self._row.removeWidget(chip)
            chip.deleteLater()
        self._chips.clear()
        # Insert new chips before the trailing add button + stretch
        for i, (path, count) in enumerate(sources):
            chip = self._make_chip(path, count)
            self._row.insertWidget(i, chip)
            self._chips[path] = chip

    def _make_chip(self, path: str, count: int) -> QWidget:
        import os

        chip = QWidget()
        h = QHBoxLayout(chip)
        h.setContentsMargins(8, 2, 4, 2)
        h.setSpacing(4)
        chip.setStyleSheet(
            "QWidget { background-color: #2d5aa0; color: white; border-radius: 12px; }"
            "QLabel { color: white; font-size: 11px; }"
        )
        chip.setToolTip(path)

        label = QLabel(f"{os.path.basename(path) or path}  ({count})")
        h.addWidget(label)

        rm = QPushButton("×")
        rm.setFixedSize(16, 16)
        rm.setCursor(Qt.PointingHandCursor)
        rm.setStyleSheet(
            "QPushButton { color: white; background: transparent; border: none;"
            " font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { color: #ffcccc; }"
        )
        rm.clicked.connect(lambda _, p=path: self.remove_requested.emit(p))
        h.addWidget(rm)
        return chip

    def _on_add_clicked(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Add folder")
        if path:
            self.add_requested.emit(path)
