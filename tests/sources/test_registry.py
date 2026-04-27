"""Handler registry: detect() returns the right handler per format."""

from gallery_view import sources


def test_detect_returns_ome_tiff_handler(make_ome_tiff_acq):
    h = sources.detect(str(make_ome_tiff_acq()))
    assert h is not None
    assert h.name == "ome_tiff"


def test_detect_returns_multi_channel_handler(make_multi_channel_tiff_acq):
    h = sources.detect(str(make_multi_channel_tiff_acq()))
    assert h is not None
    assert h.name == "multi_channel_tiff"


def test_detect_returns_none_for_unrelated_folder(tmp_path):
    (tmp_path / "random").mkdir()
    assert sources.detect(str(tmp_path / "random")) is None
