import cv2
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image, reshape_as_raster

file = "input_images/070.tif"
SP, SR = 10, 20  # spatial window radius, color window radius
MAX_ITER_CAP = 1000

with rasterio.open(file) as src:
    raster = src.read()
    profile = src.profile
    image = reshape_as_image(raster)  # (H, W, 10)

# pyrMeanShiftFiltering only accepts 8-bit, 3-channel input:
# project the 10 bands onto their first 3 principal components.
h, w, c = image.shape
pixels = image.reshape(-1, c).astype(np.float64)
pixels -= pixels.mean(axis=0)
_, _, vt = np.linalg.svd(pixels, full_matrices=False)
pcs = pixels @ vt[:3].T  # (H*W, 3)

# Stretch each component to 0..255 with a 2-98% percentile clip.
lo, hi = np.percentile(pcs, [2, 98], axis=0)
pcs = np.clip((pcs - lo) / (hi - lo), 0, 1) * 255
img8 = pcs.reshape(h, w, 3).astype(np.uint8)


def mean_shift(img, max_iter):
    # Stops per pixel after max_iter iterations or once the shift is < 1.
    crit = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, max_iter, 1)
    return cv2.pyrMeanShiftFiltering(img, SP, SR, termcrit=crit)


# Converged reference: iteration cap high enough that every pixel hits eps.
full = mean_shift(img8, MAX_ITER_CAP)

# Smallest iteration count that reproduces the converged result exactly.
n_full = next(
    n for n in range(1, MAX_ITER_CAP + 1)
    if np.array_equal(mean_shift(img8, n), full)
)
n_half = max(1, n_full // 2)
n_quarter = max(1, n_full // 4)
half = mean_shift(img8, n_half)
quarter = mean_shift(img8, n_quarter)
print(f"converged after {n_full} iterations, early-stop versions use "
      f"{n_half} (50%) and {n_quarter} (25%)")


def write_geotiff(path, img):
    out_profile = profile.copy()
    out_profile.update(count=img.shape[2], dtype="uint8", nodata=None)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(reshape_as_raster(img))


write_geotiff("mean_shift_input.tif", img8)
write_geotiff(f"mean_shift_full_{n_full}it.tif", full)
write_geotiff(f"mean_shift_half_{n_half}it.tif", half)
write_geotiff(f"mean_shift_quarter_{n_quarter}it.tif", quarter)
