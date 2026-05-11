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


def test_region_view_compact_renders_one_row_per_acq_with_region_combo(
    qapp, make_squid_single_tiff_acq
):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        folder = make_squid_single_tiff_acq(regions=3, fovs_per_region=2)
        win._add_source(str(folder))
        win._set_view_mode("region")
        # One row per acquisition (compact mode).
        assert len(win.row_widgets) == 1
        key = next(iter(win.row_widgets))
        acq_id, t, unit = key
        rw = win.row_widgets[key]
        # Region combo is shown when multi-region; FOV combo is hidden.
        assert rw.unit_combo is not None  # repurposed for region in region view
        # Open 3D View is not rendered in region rows.
        from qtpy.QtWidgets import QPushButton
        buttons = rw.container.findChildren(QPushButton)
        assert not any(b.text() == "Open 3D View" for b in buttons)
    finally:
        win.close()


def test_region_view_expanded_renders_one_row_per_region(
    qapp, make_squid_single_tiff_acq
):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        folder = make_squid_single_tiff_acq(regions=3, fovs_per_region=2)
        win._add_source(str(folder))
        win._set_view_mode("region")
        win.expand_region_action.setChecked(True)  # triggers _set_expanded_region_mode
        # 3 regions = 3 rows.
        assert len(win.row_widgets) == 3
    finally:
        win.close()


def test_switching_back_to_fov_restores_fov_rows(qapp, make_squid_single_tiff_acq):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        folder = make_squid_single_tiff_acq(regions=2, fovs_per_region=3)
        win._add_source(str(folder))
        win._set_view_mode("region")
        assert len(win.row_widgets) == 1  # compact region view
        win._set_view_mode("fov")
        # FOV view compact: one row per acquisition.
        assert len(win.row_widgets) == 1
        # In FOV view, the row's unit slot is a composite fov, not a region.
        key = next(iter(win.row_widgets))
        _, _, unit = key
        assert "_" in unit  # composite "<region>_<fov>"
    finally:
        win.close()
