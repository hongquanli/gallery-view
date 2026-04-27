"""Synthetic squid-format acquisition fixtures.

Each generator writes a real on-disk acquisition into ``tmp_path`` so handler
tests exercise the actual file-layout detection and reading code paths.
The synthetic data is a deterministic ``z*100 + y*10 + x`` gradient so MIP
results are predictable.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml


def _gradient_3d(nz: int, ny: int, nx: int, channel_offset: int = 0) -> np.ndarray:
    z, y, x = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    return (z * 100 + y * 10 + x + channel_offset * 1000).astype(np.uint16)


def _write_params(folder: Path, params: dict) -> None:
    (folder / "acquisition parameters.json").write_text(json.dumps(params))


def _write_channels_yaml(folder: Path, channels: list[dict]) -> None:
    (folder / "acquisition_channels.yaml").write_text(
        yaml.safe_dump({"channels": channels})
    )


@pytest.fixture
def make_ome_tiff_acq(tmp_path):
    """Build an OME-TIFF acquisition with axes ZCYX."""

    def _build(
        wavelengths=("488", "561"),
        nz=4,
        ny=8,
        nx=10,
        folder_name="25x_A1_2026-04-26_12-00-00.000000",
        sensor_pixel_size_um=6.5,
        dz_um=2.0,
        mag=25,
    ) -> Path:
        folder = tmp_path / folder_name
        folder.mkdir()
        _write_params(folder, {
            "sensor_pixel_size_um": sensor_pixel_size_um,
            "dz(um)": dz_um,
        })
        _write_channels_yaml(folder, [
            {
                "name": f"Fluorescence_{wl}_nm_Ex",
                "enabled": True,
                "camera_settings": {"exposure_time_ms": 100.0 * (i + 1)},
                "illumination_settings": {"intensity": 25.0 * (i + 1)},
            }
            for i, wl in enumerate(wavelengths)
        ])
        ome_dir = folder / "ome_tiff"
        ome_dir.mkdir()
        nc = len(wavelengths)
        # Build a (nz, nc, ny, nx) ZCYX volume from per-channel 3-D gradients.
        zcyx = np.stack([_gradient_3d(nz, ny, nx, c) for c in range(nc)], axis=1)
        assert zcyx.shape == (nz, nc, ny, nx), zcyx.shape
        tifffile.imwrite(
            ome_dir / "current_0.ome.tiff",
            zcyx,
            metadata={"axes": "ZCYX"},
        )
        return folder

    return _build


@pytest.fixture
def make_single_tiff_acq(tmp_path):
    """Build a per-Z-per-channel TIFF acquisition (``./0/current_0_<z>_<chname>.tiff``)."""

    def _build(
        wavelengths=("488", "561"),
        nz=4,
        ny=8,
        nx=10,
        folder_name="25x_B2_2026-04-26_12-30-00.000000",
        sensor_pixel_size_um=6.5,
        dz_um=2.0,
        mag=25,
    ) -> Path:
        folder = tmp_path / folder_name
        folder.mkdir()
        _write_params(folder, {
            "sensor_pixel_size_um": sensor_pixel_size_um,
            "dz(um)": dz_um,
        })
        _write_channels_yaml(folder, [
            {
                "name": f"Fluorescence_{wl}_nm_Ex",
                "enabled": True,
                "camera_settings": {"exposure_time_ms": 100.0 * (i + 1)},
                "illumination_settings": {"intensity": 25.0 * (i + 1)},
            }
            for i, wl in enumerate(wavelengths)
        ])
        fov_dir = folder / "0"
        fov_dir.mkdir()
        for c, wl in enumerate(wavelengths):
            stack = _gradient_3d(nz, ny, nx, c)
            for z in range(nz):
                tifffile.imwrite(
                    fov_dir / f"current_0_{z}_Fluorescence_{wl}_nm_Ex.tiff",
                    stack[z],
                )
        return folder

    return _build
