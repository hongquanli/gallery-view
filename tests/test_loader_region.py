"""MipLoader dispatches RegionStitchJob via stitch_region, caches the
result, and emits region_mip_ready. Failure path emits progress."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from gallery_view import cache
from gallery_view.loader import MipLoader, RegionStitchJob
from gallery_view.sources.single_tiff import SingleTiffHandler
from gallery_view.stitch import FovCoord
from gallery_view.types import AxisMip


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))
    return tmp_path / "mips"


def _run_loader_until(loader: MipLoader, predicate, timeout=2.0):
    """Spin the loader thread until ``predicate()`` is True or we time out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _make_acq(make_squid_single_tiff_acq, regions=1, fovs_per_region=2):
    # Folder name without ``<mag>x_`` prefix + sensor=1 yields pixel_um=1
    # so the FovCoord values in this file (in mm) map 1:1 to pixels.
    folder = make_squid_single_tiff_acq(
        regions=regions, fovs_per_region=fovs_per_region,
        nz=2, ny=4, nx=4, write_coords=True,
        folder_name="A1_2026-04-26_12-00-00.000000",
        sensor_pixel_size_um=1.0,
    )
    handler = SingleTiffHandler()
    return handler, handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 1.0}
    )


def test_region_stitch_job_emits_ready_on_cache_miss(qapp, make_squid_single_tiff_acq):
    handler, acq = _make_acq(make_squid_single_tiff_acq, regions=1, fovs_per_region=2)
    fov_mips = {
        "0_0": np.ones((4, 4), dtype=np.float32),
        "0_1": np.full((4, 4), 2.0, dtype=np.float32),
    }
    coords = [FovCoord("0_0", 0.0, 0.0), FovCoord("0_1", 4e-3, 0.0)]

    received: list = []
    loader = MipLoader()
    loader.region_mip_ready.connect(
        lambda *args: received.append(args)
    )
    loader.start()
    try:
        loader.enqueue_region(RegionStitchJob(
            acq_id=0, acq=acq, region="0",
            channel=acq.channels[0], ch_idx=0,
            timepoint="0", fov_mips=fov_mips, coords=coords,
        ))
        assert _run_loader_until(loader, lambda: bool(received))
        acq_id, t, region, ch_idx, wl, ax_mip = received[0]
        assert acq_id == 0 and t == "0" and region == "0" and ch_idx == 0
        assert isinstance(ax_mip, AxisMip)
        assert ax_mip.mip.shape == (4, 8)
    finally:
        loader.stop()
        loader.wait(2000)


def test_region_stitch_job_short_circuits_on_cache_hit(qapp, make_squid_single_tiff_acq):
    handler, acq = _make_acq(make_squid_single_tiff_acq)
    # Pre-populate cache.
    src, ch_id = handler.cache_key_region(acq, "0", acq.channels[0], timepoint="0")
    cached_mip = np.arange(16, dtype=np.float32).reshape(4, 4)
    cache.save(src, ch_id, {"z": AxisMip(mip=cached_mip, p1=0.0, p999=15.0)})

    received: list = []
    loader = MipLoader()
    loader.region_mip_ready.connect(lambda *args: received.append(args))
    loader.start()
    try:
        # Pass empty fov_mips — a cache miss would fail, but the cache hit
        # short-circuits before stitch_region is called.
        loader.enqueue_region(RegionStitchJob(
            acq_id=0, acq=acq, region="0",
            channel=acq.channels[0], ch_idx=0,
            timepoint="0", fov_mips={}, coords=[],
        ))
        assert _run_loader_until(loader, lambda: bool(received))
        ax_mip = received[0][-1]
        np.testing.assert_array_equal(ax_mip.mip, cached_mip)
    finally:
        loader.stop()
        loader.wait(2000)


def test_region_stitch_job_failure_emits_progress(qapp, make_squid_single_tiff_acq):
    handler, acq = _make_acq(make_squid_single_tiff_acq)

    progress_messages: list[str] = []
    loader = MipLoader()
    loader.progress.connect(lambda d, q, msg: progress_messages.append(msg))
    loader.start()
    try:
        # stitch_region returns None for empty inputs; loader treats None
        # as a failure and emits a progress message.
        loader.enqueue_region(RegionStitchJob(
            acq_id=0, acq=acq, region="0",
            channel=acq.channels[0], ch_idx=0,
            timepoint="0", fov_mips={}, coords=[],
        ))
        assert _run_loader_until(
            loader,
            lambda: any("region" in m and "failed" in m for m in progress_messages),
        )
    finally:
        loader.stop()
        loader.wait(2000)
