"""GeoTIFF I/O, normalization and label-mask storage for multi-band stacks.

Band count is project-configurable (project_config.json data_profile); the
default matches the original 10-band Sentinel-2 setup.
"""

import base64
import io
import os

import numpy as np
import rasterio
from PIL import Image

UNLABELED = 255  # label value meaning "no user annotation"

# How label masks are stored on disk. Projects override this via the config's
# `mask_format` when their masks follow the GIS convention of a negative nodata
# (e.g. int32 / -9999) instead of the app's internal uint8 / 255.
MASK_DTYPES = ("uint8", "int16", "int32")
DEFAULT_MASK_FORMAT = {"dtype": "uint8", "nodata": UNLABELED}

# Band order of the original 10-band Sentinel-2 stacks (kept for reference).
BAND_NAMES = ["Red", "Green", "Blue", "NIR1", "RE1", "RE2", "RE3", "NIR2", "SWIR1", "SWIR2"]


class SentinelImage:
    """A multi-band image stack with cached robust normalization stats.

    Keeps the raw bands, the [0,1]-normalized stack, the validity mask and the
    georeferencing needed to write masks back out.

    The full app also carried a patch grid here, and refused any image smaller
    than the model's patch size. Neither applies to labeling: there is no
    minimum size an image has to reach to be painted on.
    """

    def __init__(self, path, expected_bands=None):
        self.path = path
        with rasterio.open(path) as src:
            self.raw = src.read().astype(np.float32)  # (C, H, W)
            self.crs = src.crs
            self.transform = src.transform
            self.nodata = src.nodata if src.nodata is not None else -9999.0
        self.bands, self.height, self.width = self.raw.shape
        if expected_bands is not None and self.bands != expected_bands:
            raise ValueError(f"Expected a {expected_bands}-band stack (project "
                             f"data_profile), got {self.bands} bands")

        self.valid_mask = np.all(self.raw != self.nodata, axis=0)  # (H, W) bool
        self.lo, self.hi = self._percentiles()
        self.normalized = self._normalize()

    def _percentiles(self):
        """Per-band 2nd/98th percentiles over valid pixels only."""
        lo = np.zeros(self.bands, dtype=np.float32)
        hi = np.ones(self.bands, dtype=np.float32)
        for b in range(self.bands):
            band = self.raw[b][self.valid_mask]
            if band.size:
                lo[b], hi[b] = np.percentile(band, (2, 98))
                if hi[b] <= lo[b]:
                    hi[b] = lo[b] + 1.0
        return lo, hi

    def _normalize(self):
        """Robust per-band scaling to [0, 1]; nodata pixels become 0."""
        lo = self.lo[:, None, None]
        hi = self.hi[:, None, None]
        norm = np.clip((self.raw - lo) / (hi - lo), 0.0, 1.0)
        norm[:, ~self.valid_mask] = 0.0
        return norm.astype(np.float32)

    def rgb_composite(self, bands=None):
        """8-bit RGB preview.

        `bands` is a 3-list of 0-based band indices mapped to R, G, B; when
        omitted it uses the first three bands (last repeated if fewer). Indices
        are clamped to the available bands.
        """
        if bands is None:
            n = min(3, self.bands)
            idx = list(range(n)) + [n - 1] * (3 - n)
        else:
            idx = [min(max(int(b), 0), self.bands - 1) for b in bands]
        rgb = (self.normalized[idx] * 255).astype(np.uint8)  # (3, H, W)
        rgb[:, ~self.valid_mask] = 0
        return np.transpose(rgb, (1, 2, 0))  # (H, W, 3)

def array_to_png_b64(arr):
    """Encode a (H, W) uint8 / (H, W, 3) RGB / (H, W, 4) RGBA array as base64 PNG."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def png_b64_to_array(b64):
    """Decode a base64 PNG into a numpy array."""
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    return np.array(img)


def read_label_mask(path):
    """A saved mask as (H, W) uint8 in label space, or None if missing/unreadable.

    Reads at the file's native dtype and maps everything that is not a class id
    -- the declared nodata, and any value outside [0, 254] -- to UNLABELED. That
    mapping has to be explicit: the GIS convention of a large negative nodata
    and this app's uint8 label space only agree once it is. A bare
    `.astype(np.uint8)` wraps -9999 to 49, inventing ground truth for a class
    that does not exist.

    Corrupt masks read as None so callers fall back to unlabeled rather than
    failing an image switch or a training run.
    """
    if not os.path.exists(path):
        return None
    try:
        with rasterio.open(path) as src:
            raw = src.read(1)
            nodata = src.nodata
    except Exception:
        return None
    is_class = (raw >= 0) & (raw <= 254)
    if nodata is not None and not np.isnan(nodata):
        is_class &= raw != nodata
    return np.where(is_class, raw, UNLABELED).astype(np.uint8)


def write_label_mask(path, mask, crs, transform, fmt=None):
    """Write (H, W) uint8 label-space data as a 1-band GeoTIFF.

    `fmt` is the project's mask_format ({"dtype", "nodata"}): UNLABELED is
    written back as that nodata value and the array cast to that dtype, so
    edited masks keep the on-disk format the surrounding GIS pipeline expects
    rather than being silently rewritten as uint8. Defaults to uint8/255.
    """
    fmt = fmt or DEFAULT_MASK_FORMAT
    dtype = fmt.get("dtype", DEFAULT_MASK_FORMAT["dtype"])
    nodata = fmt.get("nodata", DEFAULT_MASK_FORMAT["nodata"])
    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        # uint8 mask + a negative nodata upcasts through int64 before the cast.
        dst.write(np.where(mask == UNLABELED, nodata, mask).astype(dtype), 1)


def tile_grid(total, tile, stride):
    """Start offsets covering `total` px with `tile`-wide steps of `stride`.

    The last offset is snapped back to the edge so the grid always covers the
    full extent and every tile is exactly `tile` px — the same convention as
    the same convention throughout. Tiles are uniform on purpose: a short edge
    tile would misalign with the rest of the grid.
    """
    if tile >= total:
        return [0]
    offsets = list(range(0, total - tile + 1, stride))
    if offsets[-1] != total - tile:
        offsets.append(total - tile)
    return offsets


def tile_geotiff(src_path, out_dir, tile_width, tile_height, overlap=0,
                 skip_empty=True, progress=None):
    """Cut one GeoTIFF into georeferenced tiles under `out_dir`.

    Each tile keeps the source's CRS, dtype, band count and nodata, with its own
    affine transform, so the tiles stay georeferenced and line up with the
    original. `overlap` is in pixels and applies to both axes (stride = tile -
    overlap). Tiles are read window-by-window, so this does not load the whole
    raster into memory.

    With `skip_empty`, tiles that are entirely nodata are not written — a large
    scene is often mostly padding, and empty tiles are dead weight in a project.
    They are only detectable when the source declares a nodata value.

    `progress` is an optional callback(done, total). Returns a summary dict.
    """
    if tile_width < 1 or tile_height < 1:
        raise ValueError("Tile size must be at least 1 px")
    if overlap < 0:
        raise ValueError("Overlap cannot be negative")
    if overlap >= min(tile_width, tile_height):
        raise ValueError(f"Overlap ({overlap} px) must be smaller than the tile "
                         f"({tile_width}x{tile_height} px)")

    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(src_path) as src:
        # Clamp to the source: a tile larger than the raster would otherwise be
        # padded out, and the padding is not real data.
        tile_w = min(tile_width, src.width)
        tile_h = min(tile_height, src.height)
        xs = tile_grid(src.width, tile_w, tile_w - overlap)
        ys = tile_grid(src.height, tile_h, tile_h - overlap)
        total = len(xs) * len(ys)
        profile = src.profile.copy()
        profile.update(driver="GTiff", width=tile_w, height=tile_h,
                       compress="deflate")
        profile.pop("blockxsize", None)  # source blocking may exceed the tile
        profile.pop("blockysize", None)
        profile.pop("tiled", None)

        written, skipped, done = [], 0, 0
        digits = max(3, len(str(max(len(xs), len(ys)))))
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                window = rasterio.windows.Window(x, y, tile_w, tile_h)
                data = src.read(window=window)
                done += 1
                if skip_empty and src.nodata is not None and np.all(data == src.nodata):
                    skipped += 1
                    if progress:
                        progress(done, total)
                    continue
                name = f"tile_{row:0{digits}d}_{col:0{digits}d}.tif"
                profile["transform"] = rasterio.windows.transform(window, src.transform)
                with rasterio.open(os.path.join(out_dir, name), "w", **profile) as dst:
                    dst.write(data)
                written.append(name)
                if progress:
                    progress(done, total)

    return {"tiles": len(written), "skipped": skipped, "total": total,
            "rows": len(ys), "cols": len(xs),
            "tile_width": tile_w, "tile_height": tile_h}
