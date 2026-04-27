"""Zoomable / pannable image view used by the LUT dialog."""

from qtpy.QtCore import Qt
from qtpy.QtGui import QPainter, QPixmap, QTransform
from qtpy.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QPushButton,
)


class ZoomableImageView(QGraphicsView):
    """Fixed-size view with mouse-wheel zoom (anchored under cursor),
    click-drag pan, and an overlay 'home' button that fits to view."""

    ZOOM_STEP = 1.25

    def __init__(self, size: int, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setFixedSize(size, size)
        self.setStyleSheet(
            "QGraphicsView { background-color: #111; border: 1px solid #333; border-radius: 4px; }"
        )

        self._home_btn = QPushButton("⌂", self)
        self._home_btn.setFixedSize(26, 26)
        self._home_btn.setCursor(Qt.PointingHandCursor)
        self._home_btn.setToolTip("Fit to view")
        self._home_btn.setStyleSheet(
            "QPushButton { background-color: rgba(0, 0, 0, 160); color: white;"
            " border: 1px solid rgba(255,255,255,80); border-radius: 13px;"
            " font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background-color: rgba(60, 60, 60, 200); }"
        )
        self._home_btn.clicked.connect(self.fit)
        self._home_btn.hide()
        self._home_btn.move(size - 32, 6)

    def set_pixmap(self, pixmap: QPixmap, y_scale: float = 1.0) -> None:
        was_empty = self._pixmap_item.pixmap().isNull()
        self._pixmap_item.setPixmap(pixmap)
        transform = QTransform()
        transform.scale(1.0, max(y_scale, 1e-6))
        self._pixmap_item.setTransform(transform)
        self._scene.setSceneRect(self._pixmap_item.sceneBoundingRect())
        if was_empty:
            self.fit()
        self._update_home_visibility()

    def fit(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._update_home_visibility()

    def _fit_scale(self) -> float:
        if self._pixmap_item.pixmap().isNull():
            return 1.0
        rect = self._pixmap_item.sceneBoundingRect()
        if rect.width() == 0 or rect.height() == 0:
            return 1.0
        vw, vh = self.viewport().width(), self.viewport().height()
        return min(vw / rect.width(), vh / rect.height())

    def _update_home_visibility(self) -> None:
        zoomed_in = self.transform().m11() > self._fit_scale() * 1.001
        self._home_btn.setVisible(zoomed_in)
        if zoomed_in:
            self._home_btn.raise_()

    def wheelEvent(self, event) -> None:
        zooming_in = event.angleDelta().y() > 0
        factor = self.ZOOM_STEP if zooming_in else 1 / self.ZOOM_STEP
        if not zooming_in:
            current = self.transform().m11()
            if current * factor <= self._fit_scale():
                self.fit()
                return
        self.scale(factor, factor)
        self._update_home_visibility()
