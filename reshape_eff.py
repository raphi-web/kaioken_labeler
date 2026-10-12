import glob
import os
from typing import Iterator, NamedTuple

import numpy as np
import rasterio
from numpy.lib.format import open_memmap
from tqdm import tqdm

TILE_RADIUS = 3  # 7x7 tiles
TILE_SIDE = 2 * TILE_RADIUS + 1
STRIDES = (1, 2, 4)
MAX_RADIUS = TILE_RADIUS * max(STRIDES)  # 12 px: the stride-4 tile is the limiting one
STEP = 2  # sample every 2nd row / col (row % 2 == 0, col % 2 == 0)
LABEL_MIN, LABEL_MAX = 0, 12  # keep labels where LABEL_MIN < label < LABEL_MAX


class Sample(NamedTuple):
    tile: np.ndarray
    tile_dilated_2: np.ndarray
    tile_dilated_4: np.ndarray
    worldcover: np.ndarray
    label: np.integer


def get_tile_stride(row: int, col: int, arr: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    Return a 7x7 view centered at (row, col), sampling every `stride` pixels.
    No bounds check (callers only pass valid centers) and no copy: the data is
    copied once, directly into the output memmap.
    """
    r = TILE_RADIUS * stride
    return arr[:, row - r : row + r + 1 : stride, col - r : col + r + 1 : stride]


def _valid_mask(label_raster: np.ndarray):
    """Candidate center rows/cols plus a mask of which (row, col) pairs to keep."""
    n_rows, n_cols = label_raster.shape
    start = -(-MAX_RADIUS // STEP) * STEP  # first multiple of STEP >= MAX_RADIUS
    rows = np.arange(start, n_rows - MAX_RADIUS, STEP)
    cols = np.arange(start, n_cols - MAX_RADIUS, STEP)
    sub = label_raster[np.ix_(rows, cols)]
    mask = (sub > LABEL_MIN) & (sub < LABEL_MAX)
    return rows, cols, mask


def candidate_positions(label_raster: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield every (row, col) whose tiles fit in the image and whose label is valid."""
    rows, cols, mask = _valid_mask(label_raster)
    ri, ci = np.nonzero(mask)
    for r, c in zip(rows[ri], cols[ci]):
        yield int(r), int(c)


def _paths(image: str, labels_dir: str, worldcover_dir: str) -> tuple[str, str, str]:
    base = os.path.basename(image)
    return image, os.path.join(labels_dir, base), os.path.join(worldcover_dir, base)


def iter_samples(
    image_paths: list[str], labels_dir: str, worldcover_dir: str
) -> Iterator[Sample]:
    """Lazily yield one Sample at a time; only one image is held in memory."""
    for image in image_paths:
        image_path, label_path, wc_path = _paths(image, labels_dir, worldcover_dir)
        with rasterio.open(image_path) as src, rasterio.open(
            label_path
        ) as label_src, rasterio.open(wc_path) as wc_src:
            s2_raster = src.read()
            label_raster = label_src.read(1)
            wc_raster = wc_src.read()

        for row, col in candidate_positions(label_raster):
            yield Sample(
                tile=get_tile_stride(row, col, s2_raster, 1),
                tile_dilated_2=get_tile_stride(row, col, s2_raster, 2),
                tile_dilated_4=get_tile_stride(row, col, s2_raster, 4),
                worldcover=get_tile_stride(row, col, wc_raster, 1),
                label=label_raster[row, col],
            )


def count_samples(image_paths: list[str], labels_dir: str, worldcover_dir: str) -> int:
    """Cheap first pass: reads only the label rasters to size the output arrays."""
    total = 0
    for image in tqdm(image_paths, desc="Counting"):
        _, label_path, _ = _paths(image, labels_dir, worldcover_dir)
        with rasterio.open(label_path) as label_src:
            total += int(_valid_mask(label_src.read(1))[2].sum())
    return total


def build_dataset(
    image_paths: list[str],
    labels_dir: str,
    worldcover_dir: str,
    out_dir: str = ".",
) -> int:
    n = count_samples(image_paths, labels_dir, worldcover_dir)
    if n == 0:
        raise ValueError("No valid samples found.")

    # Band counts and dtypes come from file metadata (no pixel data is read).
    image_path, label_path, wc_path = _paths(image_paths[0], labels_dir, worldcover_dir)
    with rasterio.open(image_path) as src, rasterio.open(
        label_path
    ) as label_src, rasterio.open(wc_path) as wc_src:
        s2_bands, s2_dtype = src.count, src.dtypes[0]
        wc_bands, wc_dtype = wc_src.count, wc_src.dtypes[0]
        label_dtype = label_src.dtypes[0]

    def mm(name, dtype, shape):
        return open_memmap(
            os.path.join(out_dir, name), mode="w+", dtype=dtype, shape=shape
        )

    # Output file names kept the same as the original script.
    tiles = mm("tiles.npy", s2_dtype, (n, s2_bands, TILE_SIDE, TILE_SIDE))
    tiles_d2 = mm("tiles_dialated_2.npy", s2_dtype, (n, s2_bands, TILE_SIDE, TILE_SIDE))
    tiles_d4 = mm("tiles_dialated_4.npy", s2_dtype, (n, s2_bands, TILE_SIDE, TILE_SIDE))
    worldcover = mm("worldcover.npy", wc_dtype, (n, wc_bands, TILE_SIDE, TILE_SIDE))
    labels = mm("labels.npy", label_dtype, (n,))

    samples = iter_samples(image_paths, labels_dir, worldcover_dir)
    for i, s in enumerate(tqdm(samples, total=n, desc="Writing")):
        tiles[i] = s.tile
        tiles_d2[i] = s.tile_dilated_2
        tiles_d4[i] = s.tile_dilated_4
        worldcover[i] = s.worldcover
        labels[i] = s.label

    for arr in (tiles, tiles_d2, tiles_d4, worldcover, labels):
        arr.flush()
    del tiles, tiles_d2, tiles_d4, worldcover, labels
    return n


if __name__ == "__main__":
    images = sorted(glob.glob("input_images/*.tif"))
    n = build_dataset(images, labels_dir="label_raster", worldcover_dir="worldcover")
    print(f"Wrote {n} samples.")
