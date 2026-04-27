"""``python -m gallery_view`` entry point."""

import sys

from qtpy.QtWidgets import QApplication

from .cli import parse_args
from .ui.gallery_window import GalleryWindow


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    window = GalleryWindow()
    window.show()
    for path in args.source:
        window._add_source(path)
    return app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    sys.exit(main())
