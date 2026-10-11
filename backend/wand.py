"""The magic wand: click a pixel, get the region of similar material around it.

Lifted out of the pywebview bridge, which is why nothing here knows about JSON,
base64 or the Api class. The wand needs an image, its labels and somewhere to
cache -- and this module takes exactly those. `Api.wand_select` is now the
translation layer: it turns this module's return value into the wire format
and its exceptions into the {"ok": false, "error": ...} contract the frontend
expects.

Two things are worth knowing before reading:

  * Every knob (connectivity, the per-sample budget, the negatives, protection)
    consumes only a per-sample DISTANCE FIELD. Swapping how that field is
    measured -- `source="image"` for the raster's own bands, `source="meanshift"`
    for a mean-shift-filtered PCA image -- swaps the wand's whole notion of
    similarity and changes nothing else about how it behaves.

  * ADDING A POSITIVE CLICK NEVER REMOVES A PIXEL. Each sample is resolved
    completely independently and the results are unioned. Most of the design
    below serves that one property; see spec.txt for the measurements behind it.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from data import UNLABELED
from meanshift import meanshift_available, meanshift_image as _meanshift_image

# 4-connected structuring element for the wand's component labelling.
_WAND_CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

# Offsets of the 25 pixels of a 5x5 window, nearest-first. A wand sample can
# average over the first N of these, so "N pixels" always means the N closest
# to the click rather than an arbitrary N of the window.
_SAMPLE_OFFSETS = sorted(
    ((dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)),
    key=lambda o: (o[0] * o[0] + o[1] * o[1], o[0], o[1]),
)
MAX_SAMPLE_PIXELS = len(_SAMPLE_OFFSETS)

# What the wand measures distance in. "image" is the spectral wand: every band
# of the raster. "meanshift" is the spectral measure again -- an RMS colour
# distance -- but on a mean-shift-filtered PCA image rather than the raw bands
# (see Wand.meanshift_image): the filter collapses each material onto one
# near-constant colour and sharpens the boundary between materials, which is
# what turns the tolerance into a cliff instead of a slope.
#
# Two learned sources lived here and were removed after measurement: "sam"
# (cosine distance on SAM2 encoder features) and "embed" (an auto-encoder
# land-cover embedding). Mean-shift beat both on this dataset (mean IoU 0.641
# vs 0.478 / 0.476 over 385 regions) at a fraction of the cost, and neither
# earned its keep as a second opinion. SAM2 itself remains as the snap-to-edge
# tool, which consumes the decoder rather than a distance field. See
# implementation_log.txt.
WAND_SOURCES = ("image", "meanshift")


class WandError(Exception):
    """A request the wand refuses, carrying a message meant for the user.

    Raised rather than returned so the algorithm below never has to thread an
    error value through its own control flow; the bridge catches these and
    turns them into the frontend's {"ok": false, "error": ...}.
    """


@dataclass(frozen=True)
class WandOptions:
    """A parsed, validated selection request. Built only by parse_options.

    `everywhere` is the option the frontend calls `global`, renamed because the
    word is a Python keyword.
    """

    tolerance: float
    sample_size: int = 1
    budget: int | None = None
    negatives: tuple = ()
    everywhere: bool = False
    protect: bool = False
    source: str = "image"


def parse_options(options):
    """Validate a raw options dict into WandOptions, or raise WandError.

    Split from the selection itself because these are two different jobs: "is
    this request well formed" is a contract question with one answer per field,
    and it was previously interleaved with the algorithm, so reading either
    meant skipping over the other.

    The contract, all optional except `tolerance`:

      tolerance    distance threshold, in the units of the chosen source: an
                   RMS per-band difference for both "image" and "meanshift".
                   Required.
      sample_size  how many pixels around each click to average into that
                   sample's reference, 1..MAX_SAMPLE_PIXELS, nearest first.
                   Defaults to 1.
      max_pixels   budget on the selection size, PER SAMPLE. None (the default)
                   means unbounded. Per sample, not pooled: a shared budget
                   divides among them, so every new click would shrink what the
                   earlier ones had selected.
      negatives    [[x, y], ...] of clicks marking material to exclude. A pixel
                   stays only while it is closer to its own positive sample
                   than to every negative -- a nearest-neighbour rule, so one
                   right-click on the material a fill is bleeding into removes
                   it wherever it appears rather than only where it was
                   clicked. That reach is what global mode needs, where a bleed
                   can be hundreds of scattered fragments.
      global       when true, drop the connectivity bound: a sample's region
                   becomes every matching pixel in the scene rather than the
                   connected piece under it. These classes are split into
                   dozens of fragments (Forest in 00.tiff is 63), so one click
                   can take them all. The budget still applies, spending
                   nearest-first.
      protect      when true, drop pixels that already carry a label, so a fill
                   can only ever add to unlabeled ground. Applied to the
                   finished selection rather than at commit time so the mask,
                   the reported count and what actually gets painted are one
                   object instead of three that can disagree. The CALLER
                   decides when it applies -- notably it must NOT while
                   erasing, since the eraser only ever acts on labeled pixels.
      source       "image" (default) or "meanshift".
    """
    options = options or {}
    try:
        tolerance = float(options.get("tolerance"))
    except (TypeError, ValueError):
        raise WandError("Invalid tolerance")

    try:
        sample_size = int(options.get("sample_size", 1))
    except (TypeError, ValueError):
        raise WandError("Invalid sample size")
    sample_size = min(MAX_SAMPLE_PIXELS, max(1, sample_size))

    budget = None
    max_pixels = options.get("max_pixels")
    if max_pixels is not None:
        try:
            budget = int(max_pixels)
        except (TypeError, ValueError):
            raise WandError("Invalid pixel budget")
        if budget <= 0:
            raise WandError("Pixel budget must be positive")

    source = str(options.get("source") or "image")
    if source not in WAND_SOURCES:
        raise WandError(f"Unknown wand source: {source}")

    try:
        negatives = tuple((float(p[0]), float(p[1]))
                          for p in (options.get("negatives") or []))
    except (TypeError, ValueError, IndexError):
        raise WandError("Invalid negative sample")

    return WandOptions(tolerance=tolerance, sample_size=sample_size,
                       budget=budget, negatives=negatives,
                       everywhere=bool(options.get("global")),
                       protect=bool(options.get("protect")),
                       source=source)


@dataclass
class Selection:
    """What one wand_select produced. `mask` is a bool array, not bytes.

    Packing it for the wire is the bridge's job; keeping it an array here means
    the algorithm can be measured against ground truth without a base64
    round-trip in the way.
    """

    mask: np.ndarray
    count: int
    available: int   # what the selection would have been with no budget
    protected: int   # pixels withheld by `protect`
    capped: bool     # the budget trimmed at least one sample
    valid: int       # non-nodata pixels in the image
    width: int
    height: int


class Wand:
    """Selection over one image, with the caches that make dragging cheap.

    Holds no image of its own until set_image is called, and re-derives nothing
    while the key is unchanged -- so the caller can call set_image on every
    request without thinking about invalidation.
    """

    def __init__(self):
        self._image = None
        self._key = None
        # {(source, y, x, sample_size): distance field}
        self._fields = {}
        # {"meanshift": filtered stack}: one filter pass per image, reused by
        # every click on it. A dict rather than an attribute so set_image's
        # invalidation stays one clear() over everything derived.
        self._stacks = {}

    # ---------- the image ----------

    def set_image(self, image, key):
        """Point the wand at an image; drop every cache if it is a new one.

        `key` must capture everything that changes the pixels the wand reads
        (project, image name). Both sources consume every band of the raster,
        so -- unlike the SAM2 snap tool's cache -- the display-band mapping
        must NOT be part of this key: changing what the pane shows changes
        nothing the wand measures, and folding it in would throw the caches
        away for nothing.

        Keying rather than an explicit invalidate() is what makes this safe to
        call on every request: a stale field cannot survive an image switch,
        and there is no hook for a caller to forget.
        """
        if key is not None and key == self._key and image is self._image:
            return
        self._image = image
        self._key = key
        self._fields = {}
        self._stacks = {}

    # ---------- distance fields ----------

    def _sample_colour(self, y, x, sample_size, stack=None):
        """Mean spectrum of the `sample_size` pixels nearest (y, x).

        A one-pixel sample is a sample of size one: on textured ground the
        clicked pixel is easily an outlier that the rest of its own region
        fails to match. Averaging a small neighbourhood centres the reference
        on the material rather than on wherever the cursor landed.

        `stack` defaults to the raw normalized bands; the mean-shift source
        passes its filtered image so the same neighbourhood logic measures a
        reference colour in whatever space the field runs in.

        Only valid pixels contribute, so a click near a nodata edge averages
        the real ground beside it instead of pulling the reference toward the
        fill value. The clicked pixel itself is always included, so the sample
        still means what was clicked even at the border.
        """
        stack = self._image.normalized if stack is None else stack
        valid = self._image.valid_mask
        _, height, width = stack.shape
        taken = []
        for dy, dx in _SAMPLE_OFFSETS[:max(1, sample_size)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and valid[ny, nx]:
                taken.append(stack[:, ny, nx])
        if not taken:  # the click itself sits on nodata
            taken = [stack[:, y, x]]
        return np.mean(taken, axis=0)

    def _spectral_field(self, y, x, sample_size):
        """Per-pixel spectral distance (H, W) from the colour sampled at (y, x).

        Runs over every band of SentinelImage.normalized, not the three-band
        RGB composite the pane displays. That is the whole point of doing this
        here: water and terrain shadow are near-identical in true colour and
        separate cleanly in NIR/SWIR, and the composite has thrown those bands
        away before the frontend ever sees a pixel.

        The distance is the RMS difference per band rather than a plain
        Euclidean norm, so one tolerance value means the same thing whether a
        project has 4 bands or 12 -- otherwise the number would silently
        rescale with band count.
        """
        stack = self._image.normalized  # (C, H, W) float32 in [0, 1]
        ref = self._sample_colour(y, x, sample_size)
        diff = stack - ref[:, None, None]
        return np.sqrt(np.mean(diff * diff, axis=0))

    def meanshift_image(self):
        """The mean-shift-filtered PCA image (3, H, W) float32 in [0, 1].

        Cached in _stacks under "meanshift" like the encoder stacks: one filter
        pass per image (~0.25 s at 512px), free after the first click. Keyed on
        the image alone -- the filter consumes every band, so the display-band
        mapping cannot change its output.

        Not centred or unit-normalised like the embedding stacks: this field is
        the spectral RMS measure on a different image, and the filter's own
        [0,1] scale is what makes the tolerance numbers comparable to the
        spectral wand's.
        """
        if not meanshift_available():
            raise WandError("OpenCV (cv2) is not installed")
        stack = self._stacks.get("meanshift")
        if stack is None:
            stack = _meanshift_image(self._image.normalized,
                                     self._image.valid_mask)
            self._stacks["meanshift"] = stack
        return stack

    def _meanshift_field(self, y, x, sample_size):
        """Per-pixel RMS colour distance (H, W) on the mean-shift-filtered image.

        The same measure and the same neighbourhood sampling as _spectral_field
        -- the difference is entirely in what is being measured: mean-shift has
        already collapsed each material onto one near-constant colour and
        sharpened the boundary between materials, so the distance field is flat
        inside a region and jumps at its edge instead of drifting with the
        material's own texture. Measured on this project's labels, the
        intra/inter class distance ratio drops from 0.64 (raw bands) to 0.27
        (filtered), which is what a tolerance threshold wants to see.
        """
        stack = self.meanshift_image()
        ref = self._sample_colour(y, x, sample_size, stack)
        diff = stack - ref[:, None, None]
        return np.sqrt(np.mean(diff * diff, axis=0))

    def fields(self, samples, options):
        """One per-pixel distance field (H, W) per sample, from `options.source`.

        One field per sample, each from its own neighbourhood mean. Averaging
        *across* samples would collapse the clicks to a single centroid, so a
        new click would move the reference for everything and could drop pixels
        an earlier click had selected -- measured at 43 of 45 regions.
        Averaging *within* one click denoises it without any such reach: a
        sample's field depends on that sample alone, which is what keeps a
        later click provably additive.

        Cached per (source, point, sample_size), so adding a click costs one
        new field, dragging the tolerance slider costs none, and switching
        sources back and forth costs none after the first look.
        """
        height, width = self._image.height, self._image.width
        out = []
        for sx, sy in samples:
            x = min(width - 1, max(0, int(sx)))
            y = min(height - 1, max(0, int(sy)))
            key = (options.source, y, x, options.sample_size)
            field = self._fields.get(key)
            if field is None:
                if options.source == "meanshift":
                    field = self._meanshift_field(y, x, options.sample_size)
                else:
                    field = self._spectral_field(y, x, options.sample_size)
                self._fields[key] = field
            out.append(((y, x), field))
        return out

    # ---------- the selection ----------

    def select(self, samples, options, labels=None):
        """Connected selection around `samples`; returns a Selection.

        `samples` is [[x, y], ...] in image pixel coordinates, `options` a
        WandOptions from parse_options, `labels` the (H, W) uint8 label buffer
        that `protect` consults (None means nothing is labeled yet).

        Raises WandError for anything the caller can fix: no image, no samples,
        or a filter dependency that cannot run.
        """
        if self._image is None:
            raise WandError("No image loaded")
        points = [(float(p[0]), float(p[1])) for p in (samples or [])]
        if not points:
            raise WandError("No sample points")

        if options.source == "meanshift":
            # Filter up front rather than inside the sample loop: it is the one
            # slow, failure-prone step here, and a dependency error must come
            # back as a message the pane can show instead of a traceback.
            self.meanshift_image()

        valid = self._image.valid_mask
        height, width = valid.shape
        mask = np.zeros((height, width), dtype=bool)
        uncapped = np.zeros((height, width), dtype=bool)
        capped = False

        # Distance to the nearest excluded colour. A pixel survives only while
        # its own sample is a better match than any of these.
        against = None
        for _, neg_field in self.fields(options.negatives, options):
            against = neg_field if against is None else np.minimum(against, neg_field)

        # Each sample is resolved on its own: its own match set, its own
        # connected region, its own budget. Nothing about one sample can reach
        # another, which is what makes a later click purely additive.
        for (y, x), field in self.fields(points, options):
            within = (field <= options.tolerance) & valid
            if against is not None:
                within &= field < against
            # The sample joins its own region whether or not it passes its own
            # threshold, so a click is never a no-op.
            within[y, x] = True

            if options.everywhere:
                region = within
            else:
                # 4-connected: with 8, regions touching at a single diagonal
                # pixel merge and the selection leaks through gaps too narrow
                # to see.
                labelled, _ = ndimage.label(within, structure=_WAND_CONNECTIVITY)
                region = labelled == labelled[y, x]
            uncapped |= region

            if options.budget is not None and int(region.sum()) > options.budget:
                if options.everywhere:
                    region = _nearest_within_budget(region, (y, x), options.budget)
                else:
                    region, _ = _grow_within_budget(region, [(y, x)], options.budget)
                capped = True
            mask |= region

        available = int(uncapped.sum())

        # Protection is applied to the finished selection rather than to the
        # match set: a labeled pixel should still conduct the fill through
        # itself, it just does not get painted. Masking earlier would sever
        # regions at the first labeled pixel they touch.
        protected = 0
        if options.protect and labels is not None:
            keep = labels == UNLABELED
            protected = int((mask & ~keep).sum())
            mask = mask & keep

        return Selection(mask=mask, count=int(mask.sum()), available=available,
                         protected=protected, capped=capped,
                         valid=int(valid.sum()), width=width, height=height)


# ---------- pure helpers ----------
#
# Free functions rather than methods: none of them touch the image or the
# caches, which is also what makes them straightforward to reason about in
# isolation.


def _grow_within_budget(within, seeds, budget):
    """Union of a breadth-first growth from each seed, `budget` px apiece.

    One BFS per seed rather than one shared queue over all of them. A
    shared queue looks equivalent and is not: it hands pixels out in
    proportion to each region's frontier, so a fast-spreading region eats
    the allowance of a slower one and adding a sample visibly shrinks what
    the earlier samples had selected. Independent budgets make that
    impossible -- an existing selection cannot lose pixels because a new
    click appeared somewhere else.

    Breadth-first is what makes a budget meaningful: pixels arrive in order
    of their distance *through* the region, so spending it yields the
    innermost `budget` pixels around each sample. Taking the nearest N by
    straight-line distance would happily return two halves of a horseshoe
    with the middle missing.

    Cost is O(seeds * budget), and it only runs when the region actually
    exceeds the cap.
    """
    height, width = within.shape
    passable = within.reshape(-1)
    mask = np.zeros(height * width, dtype=bool)

    for sy, sx in seeds:
        start = sy * width + sx
        seen = {start}
        queue = deque((start,))
        count = 1
        mask[start] = True
        while queue and count < budget:
            idx = queue.popleft()
            y, x = divmod(idx, width)
            neighbours = ()
            if x > 0:
                neighbours += (idx - 1,)
            if x < width - 1:
                neighbours += (idx + 1,)
            if y > 0:
                neighbours += (idx - width,)
            if y < height - 1:
                neighbours += (idx + width,)
            for n in neighbours:
                # `seen` is per-seed: a pixel another seed already took
                # must still be walked through, or this seed's growth would
                # stop dead at the boundary of an earlier one's territory.
                if n in seen or not passable[n]:
                    continue
                seen.add(n)
                mask[n] = True
                queue.append(n)
                count += 1
                if count >= budget:
                    break

    return mask.reshape(height, width), int(mask.sum())


def _nearest_within_budget(region, seed, budget):
    """The `budget` pixels of `region` closest to `seed`, straight-line.

    Global mode's counterpart to the breadth-first walk. With connectivity
    dropped there is no path through the region to measure along, so
    distance falls back to the plain radial one -- which is also what
    "nearest the click" means to someone looking at the screen.
    """
    ys, xs = np.nonzero(region)
    if len(ys) <= budget:
        return region
    sy, sx = seed
    d2 = (ys - sy) ** 2 + (xs - sx) ** 2
    keep = np.argpartition(d2, budget)[:budget]
    out = np.zeros_like(region)
    out[ys[keep], xs[keep]] = True
    return out
