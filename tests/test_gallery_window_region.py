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


def test_region_view_enqueues_fov_prereqs_then_stitch(
    qapp, make_squid_single_tiff_acq, tmp_path, monkeypatch,
):
    """Switching to region view should enqueue per-FOV MIP jobs for the
    selected region, and once all of them land we get a region stitch job.
    Verify by waiting for region_mip_ready to fire."""
    import time

    from gallery_view import cache
    from gallery_view.ui.gallery_window import GalleryWindow

    # Isolate the on-disk cache so test artefacts don't pollute the user's
    # real ~/.cache/gallery-view/mips dir.
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))

    folder = make_squid_single_tiff_acq(
        regions=2, fovs_per_region=2, nz=2, ny=4, nx=4, write_coords=True,
    )
    win = GalleryWindow()
    received: list = []
    win.loader.region_mip_ready.connect(lambda *args: received.append(args))
    try:
        win._add_source(str(folder))
        win._set_view_mode("region")
        # Switching modes triggers _rebuild_rows which (in region view)
        # enqueues prereqs and waits for them; the loader emits region_mip_ready
        # after stitch completes.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            QApplication.processEvents()
            if received:
                break
            time.sleep(0.05)
        assert received, "region_mip_ready never fired"
    finally:
        win.close()


def test_region_lut_sidecar_uses_region_cache_key(qapp, make_squid_single_tiff_acq, tmp_path, monkeypatch):
    """Saving a LUT override from a region row writes to the region-keyed
    sidecar path, not the FOV-keyed one."""
    from gallery_view import cache
    from gallery_view.types import AxisMip
    from gallery_view.ui.gallery_window import GalleryWindow

    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))

    folder = make_squid_single_tiff_acq(
        regions=1, fovs_per_region=1, write_coords=True,
    )
    win = GalleryWindow()
    try:
        win._add_source(str(folder))
        win._set_view_mode("region")
        # Synthesize a region mosaic in memory so the LUT dialog has something
        # to show without waiting for the loader thread.
        import numpy as np
        ax_mip = AxisMip(
            mip=np.zeros((4, 4), dtype=np.float32), p1=0.0, p999=1.0,
        )
        win.region_mip_data[(0, "0", "0", 0)] = ax_mip

        # The LUT-only save path: directly call save_lut_only with the key
        # the window's _adjust_lut should produce.
        handler = win.acquisitions[0].handler
        src, ch_id = handler.cache_key_region(
            win.acquisitions[0], "0", win.acquisitions[0].channels[0]
        )
        cache.save_lut_only(src, ch_id, {"z": (ax_mip.mip, 0.1, 0.9)})

        sidecar = cache._lut_override_path(src, ch_id)
        assert sidecar.exists()
        # The sidecar path encodes 'region:' — distinct from the FOV path.
        fov_src, fov_id = handler.cache_key(
            win.acquisitions[0], win.acquisitions[0].fovs[0],
            win.acquisitions[0].channels[0],
        )
        assert cache._lut_override_path(fov_src, fov_id) != sidecar
    finally:
        win.close()


def test_remove_source_prunes_region_state(qapp, make_squid_single_tiff_acq):
    """Removing a source should clear its entries from region_mip_data
    and _region_fov_readiness so they don't leak across source churn."""
    from gallery_view.types import AxisMip
    import numpy as np
    from gallery_view.ui.gallery_window import GalleryWindow

    folder = make_squid_single_tiff_acq(regions=2, fovs_per_region=1)
    win = GalleryWindow()
    try:
        win._add_source(str(folder))
        # Seed both region dicts with synthetic entries.
        win.region_mip_data[(0, "0", "0", 0)] = AxisMip(
            mip=np.zeros((2, 2), dtype=np.float32), p1=0.0, p999=1.0,
        )
        win._region_fov_readiness[(0, "0", "0")] = {(0, "0_0")}

        win._remove_source(str(folder))
        assert win.region_mip_data == {}
        assert win._region_fov_readiness == {}
    finally:
        win.close()


def test_adjust_lut_in_region_view_uses_cache_key_region(qapp, make_squid_single_tiff_acq, monkeypatch):
    """Clicking 'Adjust Contrast' on a region row passes a key_fn that routes
    to cache_key_region, not cache_key."""
    from gallery_view.types import AxisMip
    import numpy as np
    from gallery_view.ui import gallery_window as gw_mod
    from gallery_view.ui.gallery_window import GalleryWindow, RowKey

    folder = make_squid_single_tiff_acq(regions=2, fovs_per_region=1, write_coords=True)
    win = GalleryWindow()
    captured: dict = {}

    def fake_show_lut_dialog(**kwargs):
        captured.update(kwargs)
        # Don't actually open a dialog.

    monkeypatch.setattr(
        "gallery_view.ui.lut_dialog.show_lut_dialog", fake_show_lut_dialog
    )

    try:
        win._add_source(str(folder))
        win._set_view_mode("region")
        win.region_mip_data[(0, "0", "0", 0)] = AxisMip(
            mip=np.zeros((2, 2), dtype=np.float32), p1=0.0, p999=1.0,
        )
        key = RowKey(acq_id=0, timepoint="0", fov="0")  # region "0"
        win._adjust_lut(key)
        assert captured.get("unit_label") == "Region"
        # The key_fn should route to cache_key_region.
        acq = win.acquisitions[0]
        src, ch_id = captured["key_fn"](acq, "0", acq.channels[0], "0")
        assert ch_id.startswith("region:0/")
    finally:
        win.close()


def test_status_text_reports_regions_in_region_mode(qapp, make_squid_single_tiff_acq):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        folder = make_squid_single_tiff_acq(regions=2, fovs_per_region=2)
        win._add_source(str(folder))
        win._set_view_mode("region")
        win.expand_region_action.setChecked(True)  # 2 region rows
        win._refresh_visibility()
        text = win.status.text()
        assert "region" in text.lower(), f"status was {text!r}"
    finally:
        win.close()
