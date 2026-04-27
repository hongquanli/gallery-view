"""Wavelength → display color maps (lifted from explorer_ovelle.py)."""

CHANNEL_ORDER: list[str] = ["405", "488", "561", "638", "730"]

CHANNEL_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "405": (80, 120, 255),
    "488": (0, 255, 80),
    "561": (255, 255, 0),
    "638": (255, 50, 50),
    "730": (255, 0, 255),
}

NAPARI_COLORMAPS: dict[str, str] = {
    "405": "blue",
    "488": "green",
    "561": "yellow",
    "638": "red",
    "730": "magenta",
}

DEFAULT_RGB: tuple[int, int, int] = (200, 200, 200)
DEFAULT_NAPARI_CMAP: str = "gray"


def rgb_for(wavelength: str) -> tuple[int, int, int]:
    return CHANNEL_COLORS_RGB.get(wavelength, DEFAULT_RGB)


def napari_cmap_for(wavelength: str) -> str:
    return NAPARI_COLORMAPS.get(wavelength, DEFAULT_NAPARI_CMAP)
