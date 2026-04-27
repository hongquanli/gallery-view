"""Long-lived MIP loader thread.

Job queue holds ``(acq_id, acq, fov, channel)`` tuples. ``cancel(acq_id)``
prunes pending jobs for that acq from the queue; in-flight jobs run to
completion. Emits ``mip_ready`` per finished channel and ``progress`` after
each step.
"""

import queue
from dataclasses import dataclass

from qtpy.QtCore import QThread, Signal

from . import cache, mips
from .types import Acquisition, AxisMip, Channel, ChannelMips, ShapeZYX


@dataclass
class Job:
    acq_id: int
    acq: Acquisition
    fov: str
    channel: Channel
    ch_idx: int
    timepoint: str = "0"


class MipLoader(QThread):
    # acq_id, fov, ch_idx, wavelength, channel_mips, shape_zyx
    mip_ready = Signal(int, str, int, str, object, object)
    # done_total, queued_total, message
    progress = Signal(int, int, str)
    idle = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._done = 0
        self._enqueued = 0
        self._cancelled_acqs: set[int] = set()
        self._stop = False
        self._idle_emitted = False

    # ── public API (call from GUI thread) ──

    def enqueue(self, job: Job) -> None:
        self._enqueued += 1
        self._idle_emitted = False
        self._queue.put(job)

    def cancel(self, acq_id: int) -> None:
        """Drop pending jobs for this acq. In-flight job runs to completion."""
        self._cancelled_acqs.add(acq_id)

    def stop(self) -> None:
        self._stop = True
        self._queue.put(None)  # wake the worker

    # ── thread loop ──

    def run(self) -> None:
        while not self._stop:
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                if (
                    self._enqueued
                    and self._done >= self._enqueued
                    and not self._idle_emitted
                ):
                    self._idle_emitted = True
                    self.idle.emit()
                continue
            if job is None:
                break
            if job.acq_id in self._cancelled_acqs:
                self._done += 1
                self._emit_progress(f"skipped (cancelled)")
                continue
            try:
                self._process(job)
            except Exception as exc:  # noqa: BLE001
                self._done += 1
                self._emit_progress(
                    f"failed {job.channel.wavelength}nm — {job.acq.display_name}: {exc}"
                )

    def _process(self, job: Job) -> None:
        src, ch_id = job.acq.handler.cache_key(
            job.acq, job.fov, job.channel, timepoint=job.timepoint
        )
        cached, shape = cache.load(src, ch_id)
        if cached is not None:
            self._emit_ready(job, cached, shape)
            self._done += 1
            self._emit_progress(
                f"{job.channel.wavelength}nm cached — {job.acq.display_name}"
            )
            return
        self._emit_progress(
            f"computing {job.channel.wavelength}nm MIP — {job.acq.display_name}"
        )
        state = mips.new_axis_state()
        n = 0
        ny = nx = 0
        for slice_yx in job.acq.handler.iter_z_slices(
            job.acq, job.fov, job.channel, timepoint=job.timepoint,
        ):
            if n == 0:
                ny, nx = slice_yx.shape
            elif slice_yx.shape != (ny, nx):
                # squid never produces heterogeneous slice shapes; bail out
                # rather than computing a meaningless MIP.
                raise ValueError(
                    f"Slice shape changed mid-stack: expected {(ny, nx)}, got {slice_yx.shape}"
                )
            mips.accumulate_axes(slice_yx, state)
            n += 1
        finalized = mips.finalize(state)
        if finalized is None:
            self._done += 1
            self._emit_progress(
                f"empty {job.channel.wavelength}nm — {job.acq.display_name}"
            )
            return
        channel_mips = mips.axis_data_with_percentiles(finalized)
        shape_zyx: ShapeZYX = (n, ny, nx)
        cache.save(src, ch_id, channel_mips, shape_zyx)
        # Re-apply any saved LUT override on top of the fresh percentiles —
        # cache.save preserves the sidecar, but the in-memory channel_mips
        # we hand to the gallery still has the auto-computed bounds.
        overrides = cache._load_lut_override(src, ch_id)
        if overrides:
            for ax, (p1, p999) in overrides.items():
                if ax in channel_mips:
                    channel_mips[ax] = AxisMip(
                        mip=channel_mips[ax].mip, p1=p1, p999=p999
                    )
        self._emit_ready(job, channel_mips, shape_zyx)
        self._done += 1
        self._emit_progress(
            f"{job.channel.wavelength}nm computed — {job.acq.display_name}"
        )

    def _emit_ready(
        self, job: Job, channel_mips: ChannelMips, shape: ShapeZYX | None
    ) -> None:
        self.mip_ready.emit(
            job.acq_id, job.fov, job.ch_idx,
            job.channel.wavelength, channel_mips, shape,
        )

    def _emit_progress(self, message: str) -> None:
        self.progress.emit(self._done, self._enqueued, message)
