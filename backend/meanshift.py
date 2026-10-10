"""PCA -> uint8 -> cv2.pyrMeanShiftFiltering: the mean-shift wand's field image.

The wand's fourth distance source ("meanshift") measures a plain RMS colour
difference -- the same measure as the spectral source -- but on a MEAN-SHIFT
FILTERED image instead of the raw bands. pyrMeanShiftFiltering walks every
pixel toward the mode of its spatio-colour neighbourhood, so same-material
pixels converge onto near-identical colours and boundaries sharpen: measured on
this project's labels, the intra/inter class distance ratio drops from 0.64
(raw spectral) to 0.27 (filtered), which is what turns the tolerance from a
muddy knob into a cliff you can aim at.

Why PCA first, and to exactly three components:

  * The filter needs a colour image; OpenCV's meanshift path takes 8-bit 1..4
    channels, and 3 is what the algorithm was designed around. Ten bands of
    Sentinel-2 do not fit, and picking the display triple would throw away the
    infrared the way the SAM source does.
  * PCA keeps the three directions of most spectral variance over the tile's
    own pixels, so nothing the scene actually varies in is discarded -- the
    multiband analogue of "use the bands that matter here". Components are
    computed per image, not per dataset: a tile is what the user wands at, and
    its own statistics are the honest reference.

The uint8 round trip costs quantisation (1/255 of each component's range);
that is finer than the filter's own convergence tolerance and the labels it
feeds, so it is noise, not signal loss.

Cost: one filter pass per image (~0.25 s for 512px at the defaults below,
measured), cached by the wand per image like every other stack. The defaults
are OpenCV's own documented starting point (sp=7, sr=25) with maxLevel=1: the
pyramid level 0 keeps every shift on the full-resolution grid, so region
boundaries stay pixel-sharp -- which is the entire point of this source.
"""

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a project dependency
    cv2 = None

N_COMPONENTS = 3   # PCA components fed to the filter
SP = 7             # spatial radius, px
SR = 25            # colour radius, in uint8 units
ITERATIONS = 10    # max mean-shift steps per pixel (the requested default)
_MAX_LEVEL = 1     # no pyramid: boundaries stay pixel-sharp


def meanshift_available():
    """Whether the mean-shift source can run here."""
    return cv2 is not None


def _pca_uint8(stack, valid, n=N_COMPONENTS):
    """(C, H, W) float in [0,1] -> (H, W, n) uint8 of its top-n PCA scores.

    Eigenvectors come from the covariance over VALID pixels only, so a nodata
    border (zeros in the normalized stack) cannot tilt the axes. The scores are
    min-max scaled to 0..255 over the whole image -- each component separately,
    since PCA gives them different variances and a shared scale would waste
    most of the uint8 range on the minor ones.
    """
    channels = stack.shape[0]
    n = min(n, channels)
    pixels = stack[:, valid].T if valid.any() else stack.reshape(channels, -1).T
    mean = pixels.mean(axis=0)
    centred = pixels - mean
    cov = np.cov(centred.T)
    cov = np.atleast_2d(cov)
    # eigh: symmetric matrix, ascending eigenvalues -- take the top n.
    _, vectors = np.linalg.eigh(cov)
    axes = vectors[:, ::-1][:, :n]                       # (C, n)
    scores = (stack.reshape(channels, -1).T - mean) @ axes   # (H*W, n)
    scores = scores.reshape(stack.shape[1], stack.shape[2], n)
    lo = scores.reshape(-1, n).min(axis=0)
    hi = scores.reshape(-1, n).max(axis=0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    scaled = (scores - lo) / span * 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def meanshift_image(stack, valid, sp=SP, sr=SR, iterations=ITERATIONS):
    """Mean-shift-filtered image as (n, H, W) float32 in [0, 1].

    `stack` is SentinelImage.normalized (C, H, W) in [0, 1] with nodata zeroed;
    `valid` is its (H, W) mask, used for the PCA statistics. The filter itself
    sees the whole rectangle -- nodata pixels are filtered like any other and
    the wand excludes them afterwards via the valid mask, the same division of
    labour as the spectral source.

    A degenerate `valid` (all False) falls back to statistics over every pixel
    rather than dividing by nothing; the caller selects nothing anyway.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not installed")
    u8 = _pca_uint8(np.asarray(stack, dtype=np.float32), valid, N_COMPONENTS)
    # cv2 insists on exactly 3 8-bit channels; a 1- or 2-band project gets the
    # last component repeated. (RMS over repeated channels weights them the
    # same as distinct ones would be weighted, so the field stays meaningful.)
    if u8.shape[2] < N_COMPONENTS:
        u8 = np.concatenate(
            [u8] + [u8[:, :, -1:]] * (N_COMPONENTS - u8.shape[2]), axis=2)
    crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, iterations, 1.0)
    filtered = cv2.pyrMeanShiftFiltering(u8, float(sp), float(sr),
                                         maxLevel=_MAX_LEVEL, termcrit=crit)
    return filtered.astype(np.float32).transpose(2, 0, 1) / 255.0
