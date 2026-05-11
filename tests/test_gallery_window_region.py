"""Region-view UI: toolbar wiring, button enable/disable rules, mode flag."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_view_toolbar_present(qapp):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        assert "fov" in win.view_buttons
        assert "region" in win.view_buttons
        assert win.view_mode == "fov"
        assert win.view_buttons["fov"].isChecked()
        assert not win.view_buttons["region"].isChecked()
    finally:
        win.close()


def test_region_button_disabled_when_no_multi_region_source(qapp, make_single_tiff_acq):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        win._add_source(str(make_single_tiff_acq()))
        # Legacy folder -> regions == ["0"] -> Region button disabled.
        assert not win.view_buttons["region"].isEnabled()
    finally:
        win.close()


def test_region_button_enabled_with_multi_region_source(qapp, make_squid_single_tiff_acq):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        folder = make_squid_single_tiff_acq(regions=3, fovs_per_region=1)
        win._add_source(str(folder))
        assert win.view_buttons["region"].isEnabled()
    finally:
        win.close()


def test_switching_to_region_disables_axis_xz_yz(qapp, make_squid_single_tiff_acq):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        folder = make_squid_single_tiff_acq(regions=2, fovs_per_region=1)
        win._add_source(str(folder))
        win._set_view_mode("region")
        assert win.view_mode == "region"
        assert win.view_axis == "z"
        assert not win.axis_buttons["y"].isEnabled()
        assert not win.axis_buttons["x"].isEnabled()
        win._set_view_mode("fov")
        assert win.axis_buttons["y"].isEnabled()
        assert win.axis_buttons["x"].isEnabled()
    finally:
        win.close()


def test_settings_has_expand_all_regions(qapp):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        assert win.expand_region_action.isCheckable()
        assert not win.expand_region_action.isChecked()
    finally:
        win.close()
