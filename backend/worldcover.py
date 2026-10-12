"""ESA WorldCover v200 overlay rendering.

The optional `worldcover/` folder at the project root holds one raster per
image (same basename, same pixel grid, uint8 WorldCover class values with
nodata 0 -- see the dataset build notes in implementation_log.txt). When
present, the active image's overlay is colourised with the official ESA legend
and travels in the image payload as an RGBA PNG, so the frontend can stack it
over the imagery and the labels with zero per-pixel work of its own.
"""

import os

import numpy as np
import rasterio

from data import array_to_png_b64

WORLDCOVER_DIR = "worldcover"

# Official ESA WorldCover 2021 (v200) legend: class value -> RGB.
# 0 is nodata and stays transparent; any value outside this table (a corrupt
# clip, a future class) is transparent too rather than a guess.
WORLDCOVER_COLORS = {
    10: (0, 100, 0),        # Tree cover
    20: (255, 187, 34),     # Shrubland
    30: (255, 255, 76),     # Grassland
    40: (240, 150, 255),    # Cropland
    50: (250, 0, 0),        # Built-up
    60: (180, 180, 180),    # Bare / sparse vegetation
    70: (240, 240, 240),    # Snow and ice
    80: (0, 100, 200),      # Permanent water bodies
    90: (0, 150, 160),      # Herbaceous wetland
    95: (0, 207, 117),      # Mangroves
    100: (250, 230, 160),   # Moss and lichen
}

# Alpha baked into the overlay PNG. The WorldCover layer sits on TOP of the
# painted labels, so full opacity would hide them; 140/255 keeps both the land
# cover colours and the label colours readable at once.
WORLDCOVER_ALPHA = 140


def worldcover_path(root, image_name):
    """The overlay raster for one image, whether or not it exists."""
    return os.path.join(root, WORLDCOVER_DIR, image_name)


def read_worldcover(path):
    """(H, W) uint8 of WorldCover class values, or None if missing/unreadable.

    A missing or corrupt overlay is not an error anywhere upstream: the image
    simply has no worldcover key in its payload and the frontend's checkbox
    disables itself.
    """
    if not os.path.exists(path):
        return None
    try:
        with rasterio.open(path) as src:
            if src.count != 1:
                return None
            return src.read(1).astype(np.uint8)
    except Exception:
        return None


def worldcover_overlay_b64(path):
    """RGBA PNG (base64) colourising the overlay raster, or None."""
    data = read_worldcover(path)
    if data is None:
        return None
    rgba = np.zeros(data.shape + (4,), dtype=np.uint8)
    for value, (r, g, b) in WORLDCOVER_COLORS.items():
        hit = data == value
        if hit.any():
            rgba[hit] = (r, g, b, WORLDCOVER_ALPHA)
    return array_to_png_b64(rgba)
