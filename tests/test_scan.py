"""Folder ingestion: walks, dedupe, hidden-skip."""

import os
from pathlib import Path

import pytest

from gallery_view import scan


def test_ingest_single_acquisition(make_ome_tiff_acq):
    folder = make_ome_tiff_acq()
    acqs = scan.ingest(str(folder))
    assert len(acqs) == 1
    assert acqs[0].handler.name == "ome_tiff"
    assert acqs[0].path == str(folder)


def test_ingest_walks_a_parent_dir(tmp_path, make_ome_tiff_acq):
    # The fixture writes into its own ``tmp_path`` (shared with this test)
    folder_a = make_ome_tiff_acq(folder_name="25x_A1_2026-04-26_12-00-00.000000")
    folder_b = make_ome_tiff_acq(folder_name="25x_A2_2026-04-26_12-30-00.000000")
    acqs = scan.ingest(str(tmp_path))
    assert len(acqs) == 2
    assert {os.path.basename(a.path) for a in acqs} == {folder_a.name, folder_b.name}


def test_ingest_skips_hidden_dirs(tmp_path):
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    assert scan.ingest(str(tmp_path)) == []


def test_ingest_dedupes_via_realpath(tmp_path, make_ome_tiff_acq):
    folder = make_ome_tiff_acq()
    seen: set[str] = set()
    a1 = scan.ingest(str(folder), _seen=seen)
    a2 = scan.ingest(str(folder), _seen=seen)
    assert len(a1) == 1
    assert len(a2) == 0


def test_ingest_respects_max_depth(tmp_path, make_ome_tiff_acq):
    # Build the acquisition normally inside tmp_path …
    acq_folder = make_ome_tiff_acq(folder_name="25x_A1_2026-04-26_12-00-00.000000")
    # … then move it to a path 5 levels below tmp_path (exceeds MAX_DEPTH=3).
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    moved = deep / acq_folder.name
    acq_folder.rename(moved)
    assert scan.ingest(str(tmp_path)) == []


def test_ingest_skips_unreadable_subdir(tmp_path, make_ome_tiff_acq, monkeypatch):
    """If ``os.listdir`` raises (permissions, transient FS), the walker
    silently skips the offending dir rather than crashing."""
    make_ome_tiff_acq(folder_name="25x_A1_2026-04-26_12-00-00.000000")

    real_listdir = os.listdir

    def selectively_raise(p):
        if os.path.realpath(p) == os.path.realpath(str(tmp_path)):
            raise PermissionError("simulated unreadable dir")
        return real_listdir(p)

    monkeypatch.setattr(os, "listdir", selectively_raise)
    assert scan.ingest(str(tmp_path)) == []


def test_ingest_can_be_re_called_after_completion(make_ome_tiff_acq):
    """A fresh ``ingest`` call with a fresh ``_seen`` re-discovers the same
    acquisition — important for the gallery's "remove then re-add" flow,
    which doesn't share ``_seen`` across drops."""
    folder = make_ome_tiff_acq()
    a1 = scan.ingest(str(folder))
    a2 = scan.ingest(str(folder))
    assert len(a1) == 1
    assert len(a2) == 1
