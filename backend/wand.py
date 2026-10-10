"""The magic wand: click a pixel, get the region of similar material around it.

Lifted out of the pywebview bridge, which is why nothing here knows about JSON,
base64 or the Api class. The wand needs four things -- an image, its labels, a
SAM2 service for the embedding source, and somewhere to cache -- and this module
takes exactly those. `Api.wand_select` is now the translation layer: it turns
this module's return value into the wire format and its exceptions into the
{"ok": false, "error": ...} contract the frontend expects.

Two things are worth knowing before reading:

  * Every knob (connectivity, the per-sample budget, the negatives, protection)
    consumes only a per-sample DISTANCE FIELD. Swapping how that field is
    measured -- `source="image"` for the raster's own bands, `source="sam"` for
    SAM2 encoder features -- swaps the wand's whole notion of similarity and
    changes nothing else about how it behaves.

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
from sam_service import FEATURE_LEVELS, resize_bilinear

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
# of the raster. "sam" runs the identical machinery over a SAM2 encoder feature
# map instead -- see Wand._embedding_field for what that buys and what it costs.
# "embed" is a per-pixel land-cover embedding from the auto-encoder graph in
# auto_encoder/ (see Wand.embed_features), pixel resolution rather than a
# coarse feature grid, and trained on exactly these classes rather than on
# photographs. "meanshift" is the spectral measure again -- an RMS colour
# distance -- but on a mean-shift-filtered PCA image rather than the raw bands
# (see Wand.meanshift_image): the filter collapses each material onto one
# near-constant colour and sharpens the boundary between materials, which is
# what turns the tolerance into a cliff instead of a slope.
WAND_SOURCES = ("image", "sam", "embed", "meanshift")


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
    level: str = "fine"


def parse_options(options):
    """Validate a raw options dict into WandOptions, or raise WandError.

    Split from the selection itself because these are two different jobs: "is
    this request well formed" is a contract question with one answer per field,
    and it was previously interleaved with the algorithm, so reading either
    meant skipping over the other.

    The contract, all optional except `tolerance`:

      tolerance    distance threshold, in the units of the chosen source: an
                   RMS per-band difference for "image" and "meanshift", a
                   centred cosine distance for "sam" and "embed". Required.
      sample_size  how many pixels around each click to average into that
                   sample's reference, 1..MAX_SAMPLE_PIXELS, nearest first.
                   Counts feature cells, not pixels, under source="sam".
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
      source       "image" (default), "sam", "embed" or "meanshift".
      level        which SAM2 feature map, for source="sam": "fine", "mid" or
                   "deep" (see sam_service.FEATURE_LEVELS). Ignored otherwise.
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
    level = str(options.get("level") or "fine")
    if source == "sam" and level not in FEATURE_LEVELS:
        raise WandError(f"Unknown feature level: {level}")

    try:
        negatives = tuple((float(p[0]), float(p[1]))
                          for p in (options.get("negatives") or []))
    except (TypeError, ValueError, IndexError):
        raise WandError("Invalid negative sample")

    return WandOptions(tolerance=tolerance, sample_size=sample_size,
                       budget=budget, negatives=negatives,
                       everywhere=bool(options.get("global")),
                       protect=bool(options.get("protect")),
                       source=source, level=level)


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

    def __init__(self, sam=None, embed=None):
        self._sam = sam
        self._embed = embed
        self._image = None
        self._key = None
        self._display_bands = None
        # {(source, level, y, x, sample_size): distance field}
        self._fields = {}
        # {level: centred unit-length SAM2 features}; "embed" keys the
        # auto-encoder stack in the same cache, so set_image invalidates both.
        self._stacks = {}

    # ---------- the image ----------

    def set_image(self, image, key, display_bands=None):
        """Point the wand at an image; drop every cache if it is a new one.

        `key` must capture everything that changes the pixels (project, image
        name, display bands) -- the same key SamService is given, so the two
        agree about what "the same image" means and neither re-encodes behind
        the other's back.

        Keying rather than an explicit invalidate() is what makes this safe to
        call on every request: a stale field cannot survive an image switch,
        and there is no hook for a caller to forget.

        Note the spectral fields do not actually depend on the display bands --
        only the embedding ones do -- but both live in one cache, and dropping
        the lot costs one recomputed field.
        """
        if key is not None and key == self._key and image is self._image:
            return
        self._image = image
        self._key = key
        self._display_bands = display_bands
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

    def features(self, level):
        """Centred, unit-length SAM2 encoder features (C, h, w) for the image.

        Public because the caller forces it before selecting: encoding is the
        one slow, failure-prone step here, and doing it up front means a model
        error surfaces as a message rather than from inside the sample loop.

        Encoding is shared with the SAM2 snap tool through SamService's own
        cache, so the first embedding click on an image pays for it and the
        SAM2 tool then gets it free, or the other way round.

        Two preparations, both load-bearing:

        **Centred on the image's own mean feature.** Raw SAM2 features share a
        large common component, so everything is far from everything: measured
        on a 512px tile, the cosine distance from a click to the whole rest of
        the scene ran 0.4 to 0.8 with the tolerance that separates a region
        from its surroundings sitting somewhere inside the last tenth of that
        -- between 0.5 and 0.7 one level went from selecting 1.9% of the image
        to 91.6%. Subtracting the mean removes the shared part and leaves the
        differences, which is what the tolerance is trying to threshold; the
        same field then starts at 0 at the click and spreads to ~1.5.

        The mean is per image, so a tolerance is relative to the variety in
        the scene in front of you rather than to SAM2's feature space at
        large. That is the right reference for a tool you aim by eye at one
        tile, and it does mean the same number can behave differently on a
        uniform tile and a busy one.

        **Unit length**, which makes each subsequent sample cheap: with both
        sides unit vectors, a cosine distance field is one dot product against
        the stack rather than a normalize per pixel per click.
        """
        if self._sam is None:
            raise WandError("SAM2 is not available")
        stack = self._stacks.get(level)
        if stack is None:
            rgb = self._image.rgb_composite(self._display_bands)
            self._sam.set_image(rgb, self._key)
            feats = np.asarray(self._sam.feature_stack(level), dtype=np.float32)
            channels = feats.shape[0]
            feats = feats - feats.reshape(channels, -1).mean(axis=1)[:, None, None]
            norm = np.sqrt((feats * feats).sum(axis=0))
            # A cell sitting exactly at the image's mean feature has no
            # direction left after centring. Clamping the divisor leaves it at
            # the origin, where it is equidistant from everything, instead of
            # producing NaNs that would poison the comparison against the
            # tolerance.
            stack = feats / np.maximum(norm, 1e-12)
            self._stacks[level] = stack
        return stack

    def _cosine_field(self, stack, y, x, sample_size):
        """Cosine distance field (H, W) from a unit-vector stack at pixel (y, x).

        Shared by the two embedding sources, which differ only in how `stack`
        was produced: map the pixel onto the feature grid (an identity when
        the grid is already pixel-resolution, as the embed stack is), sample
        a reference direction, one tensordot for the whole field, and upsample
        only if the grid was coarser than the image.

        Both sides are unit vectors, so the dot product is the cosine itself.
        """
        _, grid_h, grid_w = stack.shape
        height, width = self._image.height, self._image.width
        # The SAM encoder resizes the whole image to a square 1024x1024
        # regardless of its aspect (see SamService._preprocess), so a pixel
        # lands on the grid by plain proportional scaling in each axis
        # independently. The embed grid needs no mapping at all.
        fy = min(grid_h - 1, int((y + 0.5) * grid_h / height))
        fx = min(grid_w - 1, int((x + 0.5) * grid_w / width))
        ref = _feature_sample(stack, fy, fx, sample_size)
        field = 1.0 - np.tensordot(ref, stack, axes=(0, 0))
        if (grid_h, grid_w) != (height, width):
            field = resize_bilinear(field, (height, width))
        return field

    def embed_features(self):
        """Centred, unit-length auto-encoder embeddings (8, H, W) for the image.

        The pixel-resolution counterpart to features(): the graph encodes
        128x128 patches of raw DN and EmbedService stitches them, so one cell
        is one pixel and the field needs no upsample -- which is what lets an
        embed selection sit exactly on the clicked pixel, something the SAM
        field's coarse grid blurs away.

        Centring and unit-norm are the same two preparations features() does,
        for the same reasons (see there for the measurements). On this
        project's own labels, centring lifts the intra/inter cosine-distance
        ratio from 0.043 to 0.018.

        Cached under the image's path rather than the SAM key: the encoder
        consumes every band, so the display-band mapping the SAM key carries
        cannot change its output, and folding it in would re-encode on a
        display change that changes nothing here.
        """
        if self._embed is None:
            raise WandError("the embedding encoder is not available")
        stack = self._stacks.get("embed")
        if stack is None:
            key = getattr(self._image, "path", None) or self._key
            self._embed.set_image(self._image.raw, key)
            feats = np.asarray(self._embed.feature_stack(), dtype=np.float32)
            channels = feats.shape[0]
            feats = feats - feats.reshape(channels, -1).mean(axis=1)[:, None, None]
            norm = np.sqrt((feats * feats).sum(axis=0))
            stack = feats / np.maximum(norm, 1e-12)
            self._stacks["embed"] = stack
        return stack

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

    def _embedding_field(self, y, x, sample_size, level):
        """Per-pixel cosine distance (H, W) from the SAM2 features at (y, x).

        The experimental counterpart to _spectral_field: instead of asking
        "which pixels are this colour", it asks "which pixels does SAM2's
        encoder describe the way it describes this one" -- so it groups by
        texture and learned appearance rather than by reflectance.

        The trade is real and worth stating: the encoder only ever sees
        rgb_composite(), so this source throws away exactly the extra bands
        that _spectral_field exists to use. It is a different question about
        the same click, not a better answer to the same one.

        Cosine rather than a Euclidean distance, because feature magnitude
        carries no meaning here and varies wildly between the three levels;
        the angle does, and it lands in [0, 2] whatever the channel count --
        the same reason the spectral field uses an RMS.

        The distance is computed on the feature grid and only the finished
        one-channel field is upsampled. Upsampling the deep level's 256
        channels to a 512px image would cost 268 MB; its field costs 1 MB, and
        nothing downstream can tell the difference since the flood fill, the
        valid mask and label protection all still run at pixel resolution.
        """
        return self._cosine_field(self.features(level), y, x, sample_size)

    def fields(self, samples, options):
        """One per-pixel distance field (H, W) per sample, from `options.source`.

        One field per sample, each from its own neighbourhood mean. Averaging
        *across* samples would collapse the clicks to a single centroid, so a
        new click would move the reference for everything and could drop pixels
        an earlier click had selected -- measured at 43 of 45 regions.
        Averaging *within* one click denoises it without any such reach: a
        sample's field depends on that sample alone, which is what keeps a
        later click provably additive.

        Cached per (source, level, point, sample_size), so adding a click costs
        one new field, dragging the tolerance slider costs none, and switching
        sources back and forth costs none after the first look.
        """
        height, width = self._image.height, self._image.width
        out = []
        for sx, sy in samples:
            x = min(width - 1, max(0, int(sx)))
            y = min(height - 1, max(0, int(sy)))
            key = (options.source, options.level, y, x, options.sample_size)
            field = self._fields.get(key)
            if field is None:
                if options.source == "sam":
                    field = self._embedding_field(y, x, options.sample_size,
                                                  options.level)
                elif options.source == "embed":
                    field = self._cosine_field(self.embed_features(), y, x,
                                               options.sample_size)
                elif options.source == "meanshift":
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
        SAM2 missing or failing to encode.
        """
        if self._image is None:
            raise WandError("No image loaded")
        points = [(float(p[0]), float(p[1])) for p in (samples or [])]
        if not points:
            raise WandError("No sample points")

        if options.source == "sam":
            # Encode up front rather than inside the sample loop: it is the one
            # slow, failure-prone step here, and a model error must come back
            # as a message the pane can show instead of a traceback.
            self.features(options.level)
        elif options.source == "embed":
            self.embed_features()  # same reason as above
        elif options.source == "meanshift":
            self.meanshift_image()  # same reason as above

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
# Free functions rather than methods: none of them touch the image, the caches
# or the SAM service, which is also what makes them straightforward to reason
# about in isolation.


def _feature_sample(stack, fy, fx, sample_size):
    """Mean direction of the `sample_size` feature cells nearest (fy, fx).

    The same nearest-first neighbourhood the spectral wand averages over,
    counted in feature cells rather than pixels -- at the fine level on a
    256px tile the two coincide, and at the deep level one cell already
    covers several pixels, so widening further would reach across whole
    objects.

    Renormalized after averaging: the mean of unit vectors is shorter than
    one, and leaving it short would shrink every distance in the field and
    silently rescale the tolerance.
    """
    _, grid_h, grid_w = stack.shape
    taken = []
    for dy, dx in _SAMPLE_OFFSETS[:max(1, sample_size)]:
        ny, nx = fy + dy, fx + dx
        if 0 <= ny < grid_h and 0 <= nx < grid_w:
            taken.append(stack[:, ny, nx])
    ref = np.mean(taken, axis=0) if taken else stack[:, fy, fx]
    return ref / max(float(np.sqrt((ref * ref).sum())), 1e-12)


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
