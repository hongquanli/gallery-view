"""Headless smoke test for GalleryWindow.

Catches the class of regression where ``row_widgets`` / ``mip_data`` key
shapes change but a single iteration site is missed and crashes the app
on first folder drop. Runs Qt with the offscreen platform plugin so the
test is CI-safe.

The test exercises the ``_add_source → _rebuild_rows → _refresh_visibility``
chain, which is where the last such bug surfaced (commit ``e846464``):
the row-widgets key grew from 2-tuple to 3-tuple but the visibility
unpacking site lagged, raising ``ValueError: too many values to unpack``.
"""

import os

# MUST be set before any Qt import so the platform plugin is chosen
# without trying to connect to a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication; Qt complains if multiple instances are made."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_gallery_window_handles_legacy_folder_drop(
    qapp, make_single_tiff_acq,
):
    """Drop a legacy single_tiff folder and assert the gallery wires up
    without raising. Specifically catches the ``_refresh_visibility``
    unpacking-shape regression."""
    from gallery_view.ui.gallery_window import GalleryWindow

    folder = make_single_tiff_acq()
    win = GalleryWindow()
    try:
        win._add_source(str(folder))
        # _add_source -> _rebuild_rows -> _refresh_visibility runs synchronously
        # on the GUI thread. If any key-shape bug is present, those calls
        # raise before returning here.
        assert len(win.acquisitions) == 1
        assert win.acquisitions[0] is not None
        # row_widgets keys must match the (acq_id, timepoint, fov) shape
        for key in win.row_widgets.keys():
            assert len(key) == 3, (
                f"row_widgets key {key!r} doesn't match (acq_id, timepoint, "
                f"fov) shape — likely a missed update to a key-shape change"
            )
    finally:
        # Stop the loader thread so the test doesn't leave a daemon QThread
        # hanging across tests.
        win.loader.stop()
        win.loader.wait(2000)
        win.close()


def test_gallery_window_handles_squid_folder_drop(
    qapp, make_squid_single_tiff_acq,
):
    """Same drop test, but with a multi-timepoint squid acquisition so the
    Time combo path is exercised."""
    from gallery_view.ui.gallery_window import GalleryWindow

    folder = make_squid_single_tiff_acq(nt=2, regions=2, fovs_per_region=2)
    win = GalleryWindow()
    try:
        win._add_source(str(folder))
        assert len(win.acquisitions) == 1
        acq = win.acquisitions[0]
        assert acq.timepoints == ["0", "1"]
        assert acq.fovs == ["0_0", "0_1", "1_0", "1_1"]
        # The default mode hides expanded FOVs, so only one row per
        # acquisition; key includes the selected timepoint and FOV.
        assert (0, "0", "0_0") in win.row_widgets
    finally:
        win.loader.stop()
        win.loader.wait(2000)
        win.close()
