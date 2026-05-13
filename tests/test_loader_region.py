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


def test_region_stitch_job_jumps_ahead_of_pending_fov_jobs(qapp, make_squid_single_tiff_acq):
    """RegionStitchJob enqueued after several FOV Jobs should still be
    processed first."""
    handler, acq = _make_acq(make_squid_single_tiff_acq, regions=1, fovs_per_region=2)
    # Pre-populate cache so all FOV jobs are no-ops (cache hits) and the
    # region stitch job can run without waiting on FOV TIFFs.
    import numpy as np
    from gallery_view import cache as cache_mod
    from gallery_view.types import AxisMip
    for fov in acq.fovs:
        for channel in acq.channels:
            src, ch_id = handler.cache_key(acq, fov, channel, timepoint="0")
            cache_mod.save(src, ch_id, {
                "z": AxisMip(mip=np.zeros((4, 4), dtype=np.float32), p1=0.0, p999=1.0),
                "y": AxisMip(mip=np.zeros((2, 4), dtype=np.float32), p1=0.0, p999=1.0),
                "x": AxisMip(mip=np.zeros((2, 4), dtype=np.float32), p1=0.0, p999=1.0),
            }, (2, 4, 4))
    region_src, region_ch_id = handler.cache_key_region(acq, "0", acq.channels[0], timepoint="0")
    cache_mod.save(region_src, region_ch_id, {
        "z": AxisMip(mip=np.ones((4, 8), dtype=np.float32), p1=0.0, p999=1.0),
    })

    fov_progress: list[str] = []
    region_received: list = []
    loader = MipLoader()
    loader.progress.connect(lambda d, q, msg: fov_progress.append(msg))
    loader.region_mip_ready.connect(lambda *args: region_received.append(args))
    loader.start()
    try:
        # Enqueue 10 FOV jobs first.
        from gallery_view.loader import Job
        for fov in acq.fovs * 5:
            for channel in acq.channels:
                loader.enqueue(Job(
                    acq_id=0, acq=acq, fov=fov, channel=channel,
                    ch_idx=0, timepoint="0",
                ))
        # Then enqueue one region stitch.
        loader.enqueue_region(RegionStitchJob(
            acq_id=0, acq=acq, region="0",
            channel=acq.channels[0], ch_idx=0,
            timepoint="0", fov_mips={}, coords=[],
        ))
        assert _run_loader_until(loader, lambda: bool(region_received), timeout=3.0)
        # The region message should appear before all FOV jobs finish — find the
        # first "region" mention in progress and verify some FOV jobs are still
        # un-processed at that point.
        first_region_idx = next(
            (i for i, m in enumerate(fov_progress) if "region" in m), -1
        )
        assert first_region_idx >= 0, "no region progress message"
        # In FIFO order the region message would appear at index ~30 (after all
        # FOV cache-hit messages); with priority it should appear much earlier.
        assert first_region_idx < 15, (
            f"region job was index {first_region_idx} of {len(fov_progress)} — "
            f"priority queue not jumping ahead"
        )
    finally:
        loader.stop()
        loader.wait(2000)
