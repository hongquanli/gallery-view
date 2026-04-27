"""Handler registry: detect() returns the right handler per format."""

from gallery_view import sources


def test_detect_returns_ome_tiff_handler(make_ome_tiff_acq):
    h = sources.detect(str(make_ome_tiff_acq()))
    assert h is not None
    assert h.name == "ome_tiff"


def test_detect_returns_single_tiff_handler(make_single_tiff_acq):
    h = sources.detect(str(make_single_tiff_acq()))
    assert h is not None
    assert h.name == "single_tiff"


def test_detect_returns_none_for_unrelated_folder(tmp_path):
    (tmp_path / "random").mkdir()
    assert sources.detect(str(tmp_path / "random")) is None
