"""pywebview bridge: the Api class is exposed to JS as window.pywebview.api.

Every method is called by pywebview on a worker thread and returns
JSON-serializable data, so each JS call resolves as a promise.

This is the labeling-only build: no model, no training, no inference and no
accuracy report, so nothing here imports torch. Masks are painted, autosaved
and read back; SAM2 click-assist runs on onnxruntime.
"""

import base64
import os
import threading
import traceback
from collections import deque

import numpy as np
import rasterio
import webview
from scipy import ndimage

from data import (UNLABELED, SentinelImage, array_to_png_b64, read_label_mask,
                  tile_geotiff, write_label_mask)
from project import (Project, config_from_settings, config_path,
                     default_config, probe_band_count, save_config)
from sam_service import SamService, mask_to_polygons

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


class Api:
    def __init__(self, image_path=None, project_root=None):
        self._project = None
        self._image = None
        self._image_name = None
        self._labels = None
        self._dirty_labels = {}  # name -> label buffer, edited this session and not yet saved
        self._dirty_geo = {}     # name -> (crs, transform) for each dirty entry above
        self._lock = threading.Lock()
        self._sam = SamService()  # Efficient-SAM2 click assist (onnxruntime)
        # Background GeoTIFF tiling (see create_project_from_geotiff)
        self._tiling = {"running": False, "done": 0, "total": 0,
                        "result": None, "error": None}
        # name -> labeled pixel count, so the thumbnail strip can show which
        # images still need work. Filled by a background scan when a project
        # opens (one mask read each) and kept current from memory afterwards,
        # so painting never re-reads the folder.
        self._label_counts = {}
        self._label_scan = {"running": False, "done": 0, "total": 0}
        self._label_thread = None
        # (image name, {sample point: distance field}) for the wand; see _wand_fields.
        self._wand_cache = None
        if project_root:
            result = self._init_project(project_root)
            if not result["ok"]:
                raise ValueError(result["error"])
        elif image_path:
            self._set_image(SentinelImage(image_path),
                            os.path.basename(image_path))

    def _set_image(self, image, name, labels=None):
        """Swap the active image and its label buffer."""
        self._wand_cache = None  # keyed on the image; a swap makes it stale
        self._image = image
        self._image_name = name
        if image is not None and labels is None:
            labels = np.full((image.height, image.width), UNLABELED, dtype=np.uint8)
        self._labels = labels

    def _display_bands(self):
        """The project's configured [R, G, B] display band indices, or None
        (which makes rgb_composite fall back to the first three bands)."""
        if self._project is not None:
            return self._project.config["data_profile"].get("display_bands")
        return None

    def _image_payload(self):
        """Everything the frontend needs to display the active image, or None."""
        if self._image is None:
            return None
        return {
            "name": self._image_name,
            "png": array_to_png_b64(self._image.rgb_composite(self._display_bands())),
            "width": self._image.width,
            "height": self._image.height,
            "labels": base64.b64encode(self._labels.tobytes()).decode("ascii"),
            "labeled_pixels": int((self._labels != UNLABELED).sum()),
        }

    # ---------- image / overlay ----------

    def get_image(self):
        return self._image_payload()

    # ---------- labels ----------

    def set_labels(self, b64):
        """Receive the raw label buffer (H*W uint8: class ids, 255 unlabeled)."""
        if self._image is None:
            return {"labeled_pixels": 0}
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        self._labels = buf.reshape(self._image.height, self._image.width).copy()
        self._dirty_labels[self._image_name] = self._labels
        self._dirty_geo[self._image_name] = (self._image.crs, self._image.transform)
        self._note_label_count()
        return {"labeled_pixels": int((self._labels != UNLABELED).sum())}

    # ---------- magic wand ----------

    def _sample_colour(self, y, x, sample_size):
        """Mean spectrum of the `sample_size` pixels nearest (y, x).

        A one-pixel sample is a sample of size one: on textured ground the
        clicked pixel is easily an outlier that the rest of its own region
        fails to match. Averaging a small neighbourhood centres the reference
        on the material rather than on wherever the cursor landed.

        Only valid pixels contribute, so a click near a nodata edge averages
        the real ground beside it instead of pulling the reference toward the
        fill value. The clicked pixel itself is always included, so the sample
        still means what was clicked even at the border.
        """
        stack = self._image.normalized
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

    def _wand_fields(self, samples, sample_size):
        """One per-pixel spectral distance field (H, W) per sample.

        Runs over every band of SentinelImage.normalized, not the three-band
        RGB composite the pane displays. That is the whole point of doing this
        here: water and terrain shadow are near-identical in true colour and
        separate cleanly in NIR/SWIR, and the composite has thrown those bands
        away before the frontend ever sees a pixel.

        One field per sample, each from its own neighbourhood mean. Averaging
        *across* samples would collapse the clicks to a single centroid, so a
        new click would move the reference for everything and could drop pixels
        an earlier click had selected -- measured at 43 of 45 regions.
        Averaging *within* one click denoises it without any such reach: a
        sample's field depends on that sample alone, which is what keeps a
        later click provably additive.

        The distance is the RMS difference per band rather than a plain
        Euclidean norm, so one tolerance value means the same thing whether a
        project has 4 bands or 12 -- otherwise the number would silently
        rescale with band count.

        Cached per (point, sample_size), so adding a click costs one new field
        and dragging the tolerance slider costs none.
        """
        if self._wand_cache is None or self._wand_cache[0] != self._image_name:
            self._wand_cache = (self._image_name, {})
        cache = self._wand_cache[1]

        stack = self._image.normalized  # (C, H, W) float32 in [0, 1]
        _, height, width = stack.shape
        fields = []
        for sx, sy in samples:
            x = min(width - 1, max(0, int(sx)))
            y = min(height - 1, max(0, int(sy)))
            key = (y, x, sample_size)
            field = cache.get(key)
            if field is None:
                ref = self._sample_colour(y, x, sample_size)
                diff = stack - ref[:, None, None]
                field = np.sqrt(np.mean(diff * diff, axis=0))
                cache[key] = field
            fields.append(((y, x), field))
        return fields

    @staticmethod
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

    @staticmethod
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

    def wand_select(self, samples, options=None):
        """Connected selection around `samples`, over all bands.

        samples: [[x, y], ...] in image pixel coordinates.

        options is one object rather than a growing positional list, so the JS
        side stays readable and a new knob is one key instead of another
        argument nobody can remember the order of:

          tolerance    spectral distance, in the same units as the normalized
                       bands (an RMS per-band difference). Required.
          max_pixels   budget on the selection size, PER SAMPLE. None (the
                       default) means unbounded. When a sample's region is
                       larger, it grows breadth-first from that sample and
                       stops once its budget is spent, keeping the innermost
                       pixels around it. Per sample, not pooled: a shared
                       budget divides among them, so every new click would
                       shrink what the earlier ones had selected.
          sample_size  how many pixels around each click to average into that
                       sample's reference colour, 1..MAX_SAMPLE_PIXELS,
                       nearest first. Defaults to 1.
          negatives    [[x, y], ...] of clicks marking material to exclude. A
                       pixel stays only while it is closer to its own positive
                       sample than to every negative -- a nearest-neighbour
                       rule, so one right-click on the material a fill is
                       bleeding into removes it wherever it appears rather
                       than only where it was clicked. That reach is what
                       global mode needs, where a bleed can be hundreds of
                       scattered fragments.
          global       when true, drop the connectivity bound: a sample's
                       region becomes every matching pixel in the scene rather
                       than the connected piece under it. These classes are
                       split into dozens of fragments (Forest in 00.tiff is
                       63), so one click can take them all. The budget still
                       applies, spending nearest-first.
          protect      when true, drop pixels that already carry a label, so a
                       fill can only ever add to unlabeled ground. Applied here
                       rather than at commit time so the mask, the reported
                       count and what actually gets painted are one object
                       instead of three that can disagree. The caller decides
                       when it applies -- notably it must NOT while erasing,
                       since the eraser only ever acts on labeled pixels.

        Returns the mask bit-packed and base64'd -- 32 KB for a 512x512 image
        rather than the 256 KB a byte-per-pixel buffer would cost on every
        slider step. `available` reports what the selection would have been
        without the budget, so the UI can say how hard the cap is biting.
        """
        if self._image is None:
            return {"ok": False, "error": "No image loaded"}
        options = options or {}
        points = [(float(p[0]), float(p[1])) for p in (samples or [])]
        if not points:
            return {"ok": False, "error": "No sample points"}
        try:
            tol = float(options.get("tolerance"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "Invalid tolerance"}
        try:
            size = int(options.get("sample_size", 1))
        except (TypeError, ValueError):
            return {"ok": False, "error": "Invalid sample size"}
        size = min(MAX_SAMPLE_PIXELS, max(1, size))
        budget = None
        max_pixels = options.get("max_pixels")
        if max_pixels is not None:
            try:
                budget = int(max_pixels)
            except (TypeError, ValueError):
                return {"ok": False, "error": "Invalid pixel budget"}
            if budget <= 0:
                return {"ok": False, "error": "Pixel budget must be positive"}

        valid = self._image.valid_mask
        height, width = valid.shape
        mask = np.zeros((height, width), dtype=bool)
        uncapped = np.zeros((height, width), dtype=bool)
        capped = False

        # Each sample is resolved on its own: its own match set, its own
        # connected region, its own budget. Nothing about one sample can reach
        # another, which is what makes a later click purely additive.
        everywhere = bool(options.get("global"))

        # Distance to the nearest excluded colour. A pixel survives only while
        # its own sample is a better match than any of these.
        negatives = [(float(p[0]), float(p[1]))
                     for p in (options.get("negatives") or [])]
        against = None
        for _, neg_field in self._wand_fields(negatives, size):
            against = neg_field if against is None else np.minimum(against, neg_field)

        for (y, x), field in self._wand_fields(points, size):
            within = (field <= tol) & valid
            if against is not None:
                within &= field < against
            # The sample joins its own region whether or not it passes its own
            # threshold, so a click is never a no-op.
            within[y, x] = True

            if everywhere:
                region = within
            else:
                # 4-connected: with 8, regions touching at a single diagonal
                # pixel merge and the selection leaks through gaps too narrow
                # to see.
                labelled, _ = ndimage.label(within, structure=_WAND_CONNECTIVITY)
                region = labelled == labelled[y, x]
            uncapped |= region

            if budget is not None and int(region.sum()) > budget:
                if everywhere:
                    region = self._nearest_within_budget(region, (y, x), budget)
                else:
                    region, _ = self._grow_within_budget(region, [(y, x)], budget)
                capped = True
            mask |= region

        available = int(uncapped.sum())

        # Protection is applied to the finished selection rather than to the
        # match set: a labeled pixel should still conduct the fill through
        # itself, it just does not get painted. Masking earlier would sever
        # regions at the first labeled pixel they touch.
        protected = 0
        if options.get("protect") and self._labels is not None:
            keep = self._labels == UNLABELED
            protected = int((mask & ~keep).sum())
            mask = mask & keep

        packed = np.packbits(mask.reshape(-1))
        return {"ok": True,
                "mask": base64.b64encode(packed.tobytes()).decode("ascii"),
                "count": int(mask.sum()),
                "available": available,
                "protected": protected,
                "capped": capped,
                "valid": int(self._image.valid_mask.sum()),
                "width": width, "height": height}

    # ---------- SAM2 snap-to-edge assist ----------

    def sam_available(self):
        """Whether the Efficient-SAM2 click assist can run (files present)."""
        reason = self._sam.unavailable_reason()
        return {"available": reason is None, "reason": reason}

    def _sam_image_key(self):
        """Cache key for the SAM embeddings: anything that changes the RGB pixels."""
        root = self._project.root if self._project is not None else None
        bands = self._display_bands()
        return (root, self._image_name, tuple(bands) if bands else ())

    def sam_snap(self, points):
        """Segment from a list of clicks and return editable polygon vertices.
        """
        if self._image is None:
            return {"ok": False, "error": "No image loaded"}
        reason = self._sam.unavailable_reason()
        if reason is not None:
            return {"ok": False, "error": reason}
        clicks = [(float(p[0]), float(p[1]), int(p[2]) if len(p) > 2 else 1)
                  for p in (points or [])]
        if not clicks:
            return {"ok": False, "error": "No points given"}
        if all(lab == 0 for *_, lab in clicks):
            # SAM2 has nothing to grow from; the caller should keep the previous
            # mask rather than showing an empty one.
            return {"ok": False, "error": "Add an include point first"}
        try:
            rgb = self._image.rgb_composite(self._display_bands())  # (H, W, 3) uint8
            self._sam.set_image(rgb, self._sam_image_key())
            mask, score = self._sam.predict_points(clicks)
            polygons = mask_to_polygons(mask)
        except Exception as e:  # model/runtime errors must not crash the bridge
            traceback.print_exc()
            return {"ok": False, "error": f"SAM2 failed: {e}"}
        if not polygons:
            return {"ok": False, "error": "SAM2 found no object at those points"}
        return {"ok": True, "polygons": polygons, "score": score}


    # ---------- project ----------

    def new_project(self):
        """Pick a directory; open its existing project or propose setup defaults.

        If the folder has no project_config.json yet, nothing is written:
        the frontend shows the setup dialog with the returned defaults and
        calls create_project() once the user confirms.
        """
        root = self._folder_dialog()
        if not root:
            return {"ok": False, "error": None}  # user cancelled
        if os.path.exists(config_path(root)):
            return self._init_project(root)
        channels = probe_band_count(root)
        config = default_config(os.path.basename(os.path.normpath(root)) or "project",
                                input_channels=channels or 10)
        return {"ok": True, "needs_setup": True, "root": root,
                "defaults": {
                    "project_name": config["project_name"],
                    "input_channels": config["data_profile"]["input_channels"],
                    "input_patch_size": config["data_profile"]["input_patch_size"][0],
                    "band_names": config["data_profile"]["band_names"],
                    "display_bands": config["data_profile"]["display_bands"],
                    "images_folder": config["paths"]["images_folder"],
                    "masks_user": config["paths"]["masks_user"],
                    "bands_detected": channels is not None,
                }}

    def new_project_from_geotiff(self):
        """Pick one GeoTIFF and propose a project built by tiling it.

        Nothing is written here: the band profile is read off the raster and
        returned as setup defaults (alongside tile size / overlap), and
        `create_project_from_geotiff` does the work once the user confirms.
        """
        source = self._file_dialog()
        if not source:
            return {"ok": False, "error": None}  # user cancelled
        try:
            with rasterio.open(source) as src:
                channels, width, height = src.count, src.width, src.height
                descriptions = [d for d in (src.descriptions or []) if d]
                has_crs = src.crs is not None
        except Exception as e:
            return {"ok": False, "error": f"Could not read that GeoTIFF: {e}"}

        stem = os.path.splitext(os.path.basename(source))[0]
        config = default_config(stem or "project", input_channels=channels)
        patch = config["data_profile"]["input_patch_size"][0]
        # Default tiles comfortably larger than the model patch (a tile smaller
        # than the patch is unusable), clamped to the raster itself.
        tile = max(patch, min(256, width, height))
        band_names = (list(descriptions) if len(descriptions) == channels
                      else config["data_profile"]["band_names"])
        return {"ok": True, "needs_setup": True, "mode": "geotiff",
                "source": source,
                "defaults": {
                    "project_name": config["project_name"],
                    "input_channels": channels,
                    "input_patch_size": patch,
                    "band_names": band_names,
                    "display_bands": config["data_profile"]["display_bands"],
                    "images_folder": "images",
                    "masks_user": config["paths"]["masks_user"],
                    "bands_detected": True,
                    "output_root": self._suggest_project_dir(source, stem),
                    "tile_width": tile,
                    "tile_height": tile,
                    "overlap": 0,
                    "source_name": os.path.basename(source),
                    "source_width": width,
                    "source_height": height,
                    "source_georeferenced": has_crs,
                }}

    @staticmethod
    def _suggest_project_dir(source, stem):
        """A not-yet-existing sibling folder to tile into."""
        parent = os.path.dirname(os.path.abspath(source))
        base = os.path.join(parent, f"{stem}_project")
        candidate, n = base, 2
        while os.path.exists(candidate):
            candidate = f"{base}_{n}"
            n += 1
        return candidate

    def create_project_from_geotiff(self, source, settings):
        """Tile `source` into a new folder and open it as a project."""
        root = str(settings.get("output_root", "")).strip()
        if not root:
            return {"ok": False, "error": "Choose an output folder"}
        root = os.path.abspath(os.path.expanduser(root))
        # Never tile into a folder that already holds something: this writes
        # many files and would be tedious to disentangle from existing data.
        if os.path.exists(config_path(root)):
            return {"ok": False, "error": "That folder already contains a project"}
        if os.path.isdir(root) and os.listdir(root):
            return {"ok": False, "error": "That folder is not empty; pick a new one"}
        if os.path.isfile(root):
            return {"ok": False, "error": "That path is a file, not a folder"}

        try:
            tile_w = int(settings["tile_width"])
            tile_h = int(settings["tile_height"])
            overlap = int(settings.get("overlap") or 0)
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "Tile size and overlap must be whole numbers"}
        if min(tile_w, tile_h) < 1:
            return {"ok": False, "error": "Tile size must be at least 1 px"}
        # The full app rejects tiles smaller than the model's patch size, since
        # such an image cannot be trained on. There is no model here, so any
        # tile you can see is a tile you can label.

        if self._tiling["running"]:
            return {"ok": False, "error": "Already tiling"}
        images_folder = str(settings.get("images_folder", "")).strip() or "images"

        # Everything above is validated synchronously so mistakes come back
        # immediately. The tiling itself can run for minutes on a big raster,
        # so it moves to a worker and the dialog polls tiling_progress().
        self._tiling = {"running": True, "done": 0, "total": 0,
                        "result": None, "error": None}
        threading.Thread(
            target=self._tile_worker,
            args=(source, root, images_folder, tile_w, tile_h, overlap, settings),
            daemon=True).start()
        return {"ok": True, "started": True}

    def _tile_worker(self, source, root, images_folder, tile_w, tile_h, overlap, settings):
        """Tile, write the config and open the project; report via self._tiling."""
        try:
            def on_progress(done, total):
                self._tiling["done"] = done
                self._tiling["total"] = total

            summary = tile_geotiff(source, os.path.join(root, images_folder),
                                   tile_w, tile_h, overlap, progress=on_progress)
            if not summary["tiles"]:
                raise ValueError("Tiling produced no images (every tile was empty)")
            config = config_from_settings(settings)
            save_config(root, config)
            result = self._init_project(root)
            if result.get("ok"):
                result["created"] = True
                result["tiling"] = summary
            self._tiling["result"] = result
        except Exception as e:  # a worker crash must surface in the dialog
            traceback.print_exc()
            self._tiling["error"] = str(e)
        finally:
            self._tiling["running"] = False

    def tiling_progress(self):
        """Poll the background tiling: counts while running, payload when done."""
        t = self._tiling
        return {"running": t["running"], "done": t["done"], "total": t["total"],
                "result": t["result"], "error": t["error"]}

    def pick_folder(self):
        """Folder chooser for the setup dialog's output path."""
        return {"path": self._folder_dialog()}

    def create_project(self, root, settings):
        """Write the config assembled in the setup dialog and open the project."""
        if os.path.exists(config_path(root)):
            return {"ok": False, "error": "This folder already contains a project"}
        try:
            config = config_from_settings(settings)
            save_config(root, config)
        except (OSError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        result = self._init_project(root)
        if result.get("ok"):
            result["created"] = True
        return result

    def probe_bands(self, root, images_folder="."):
        """Band count of the images in root/images_folder (None if unreadable)."""
        folder = os.path.normpath(os.path.join(root, images_folder))
        return {"bands": probe_band_count(folder)}

    def _init_project(self, root):
        """Open (or create) the project at root, kick off thumbnails, load image 1."""
        try:
            if os.path.exists(config_path(root)):
                project = Project.open(root)
                created = False
            else:
                project = Project.create(root)
                save_config(root, project.config)
                created = True
        except (OSError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        with self._lock:
            self._autosave_user_mask()  # labels of the previously open project
            self._set_image(None, None)
            self._dirty_labels = {}
            self._dirty_geo = {}
        self._project = project
        project.start_thumbnail_generation()
        self._start_label_scan()
        images = project.list_images()
        image_error = None
        if images:
            result = self.load_image(images[0])
            image_error = result.get("error")
        return {"ok": True, "created": created, "project": self.get_project(),
                "image": self._image_payload(), "image_error": image_error}

    def save_project(self):
        """Write/update project_config.json in the project root."""
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        try:
            path = save_config(self._project.root, self._project.config)
        except (OSError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "path": path}

    def update_settings(self, settings):
        """Apply edited settings to the open project's config and persist them.

        Only the project name, band names, class colors, display bands and
        folder paths are editable here. Changing a folder path reopens the
        project so the new folders take effect (mask dirs are created and
        images reload).

        This build shares project_config.json with the full training app, and
        config_from_settings rebuilds the dict from scratch -- so anything not
        carried over is dropped. Keys this build has no concept of (the
        training split, the patch size, the PointRend flag) are preserved
        verbatim rather than reset to a default, because saving settings in the
        labeler must never quietly reconfigure someone's training run.
        """
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        old = self._project.config
        profile = old["data_profile"]
        merged = dict(settings)
        merged["input_channels"] = profile["input_channels"]
        merged["input_patch_size"] = profile["input_patch_size"][0]
        try:
            config = config_from_settings(merged)
            # config_from_settings resets the palette to the defaults; keep the
            # project's ids/names and only accept edited colors from the dialog
            # (validated below by save_config -> validate_config).
            new_classes = settings.get("classes")
            by_id = {c["id"]: c for c in old["classes"]}
            merged_classes = None
            if isinstance(new_classes, list) and len(new_classes) == len(old["classes"]):
                merged_classes = []
                for nc in new_classes:
                    if not isinstance(nc, dict) or nc.get("id") not in by_id:
                        merged_classes = None
                        break
                    cls = dict(by_id[nc["id"]])
                    if isinstance(nc.get("color"), str):
                        cls["color"] = nc["color"]
                    merged_classes.append(cls)
            config["classes"] = merged_classes or [dict(c) for c in old["classes"]]
            # Carry over every top-level key this build does not manage (the
            # training split, mask_format, anything a sibling app added). A
            # generic copy rather than a named list, so a key introduced
            # elsewhere survives a save here instead of vanishing.
            for key, value in old.items():
                if key not in config:
                    config[key] = value
            # Same for the model-shaped parts of data_profile, which
            # config_from_settings would otherwise reset to its own defaults.
            for key in ("use_pointrend", "input_patch_size"):
                if key in profile:
                    config["data_profile"][key] = profile[key]
            save_config(self._project.root, config)
        except (OSError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        if config["paths"] != old["paths"]:
            result = self._init_project(self._project.root)
            result["reopened"] = True
            return result
        self._project.config = config
        result = {"ok": True, "project": self.get_project()}
        if config["data_profile"]["display_bands"] != profile.get("display_bands"):
            # New RGB mapping: re-render the open image (labels/geometry unchanged,
            # so the frontend swaps only the base png) and rebuild every cached
            # thumbnail, overwriting the old ones.
            if self._image is not None:
                result["image"] = self._image_payload()
            self._project.start_thumbnail_generation(force=True)
        return result

    def get_project(self):
        """Current project state for the frontend, or None."""
        if self._project is None:
            return None
        return {
            "root": self._project.root,
            "name": self._project.config["project_name"],
            "classes": self._project.config["classes"],
            "data_profile": self._project.config["data_profile"],
            "paths": self._project.config["paths"],
            "images": self._project.list_images(),
            "active": self._image_name,
        }

    def delete_image(self, name):
        """Permanently remove an image and its user/AI masks and thumbnail.

        If the deleted image is the active one, its in-memory buffer is
        dropped without autosaving (there is nothing left on disk to save it
        against) and the next remaining image is loaded in its place, or the
        panes clear if none are left.
        """
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        if name not in self._project.list_images():
            return {"ok": False, "error": f"Unknown image: {name}"}

        was_active = name == self._image_name
        user_path, ai_path = self._project.mask_paths(name)
        for path in (self._project.image_path(name), user_path, ai_path,
                     self._project.thumb_path(name)):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as e:
                return {"ok": False, "error": f"Could not delete {name}: {e}"}

        self._dirty_labels.pop(name, None)
        self._dirty_geo.pop(name, None)
        self._label_counts.pop(name, None)

        # A split override in a config shared with the training app would now
        # point at a file that no longer exists. This build does not manage the
        # split, but it does own the deletion, so it prunes just that entry and
        # leaves the rest of the split untouched.
        config = self._project.config
        split = config.get("split")
        if isinstance(split, dict) and isinstance(split.get("overrides"), dict) \
                and name in split["overrides"]:
            split = dict(split)
            overrides = dict(split["overrides"])
            overrides.pop(name)
            split["overrides"] = overrides
            config["split"] = split
            try:
                save_config(self._project.root, config)
            except (OSError, ValueError) as e:
                return {"ok": False, "error": str(e)}

        result = {"ok": True}
        if was_active:
            with self._lock:
                self._set_image(None, None)  # the buffer's file is already gone
            remaining = self._project.list_images()
            if remaining:
                loaded = self.load_image(remaining[0])
                result["image"] = loaded.get("image")
                if loaded.get("error"):
                    result["image_error"] = loaded["error"]
            else:
                result["image"] = None
        result["project"] = self.get_project()
        return result

    def load_image(self, name):
        """Make an image of the project active in both panes.

        The current user labels are autosaved to masks_user/ first, and the
        new image's saved user mask (if any) is restored as its labels.
        """
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        if name not in self._project.list_images():
            return {"ok": False, "error": f"Unknown image: {name}"}
        if name == self._image_name:
            return {"ok": True, "image": self._image_payload()}
        try:
            image = SentinelImage(self._project.image_path(name),
                                  expected_bands=self._project.input_channels)
        except Exception as e:
            return {"ok": False, "error": f"Could not load {name}: {e}"}
        with self._lock:
            self._autosave_user_mask()
            self._set_image(image, name, labels=self._read_user_mask(image, name))
        return {"ok": True, "image": self._image_payload()}

    def _autosave_user_mask(self):
        """Persist current labels before switching images so no work is lost."""
        if self._project is None or self._image is None:
            return
        user_path, _ = self._project.mask_paths(self._image_name)
        if (self._labels != UNLABELED).any() or os.path.exists(user_path):
            write_label_mask(user_path, self._labels, self._image.crs,
                             self._image.transform, self._project.mask_format)

    def _read_user_mask(self, image, name):
        """Labels for `name`: this session's edits if any, else the saved mask_user/ file.

        Edits made earlier in the session (possibly not yet written to disk by
        an explicit Save) take priority, so switching back to an image always
        shows its latest state instead of only what's on disk.
        """
        cached = self._dirty_labels.get(name)
        if cached is not None and cached.shape == (image.height, image.width):
            return cached.copy()
        user_path, _ = self._project.mask_paths(name)
        mask = read_label_mask(user_path)
        if mask is not None and mask.shape == (image.height, image.width):
            return mask
        return None

    # ---------- masks ----------

    def autosave_user_mask(self):
        """Persist the current image's labels to masks_user/ after an edit.

        Writes only the active image (cheap enough to run per stroke) and,
        unlike save_user_mask, leaves the session dirty-tracking untouched so
        the manual "Save Masks" still writes every edited image at once.
        """
        if self._project is None or self._image is None:
            return {"ok": False}
        with self._lock:
            self._autosave_user_mask()
        return {"ok": True}

    def save_user_mask(self):
        """Save every user-labeled image edited this session to masks_user/ as GeoTIFF.

        Tracks per-image label buffers in self._dirty_labels as the user edits
        (set_labels/transfer_prediction), so switching between images during a
        session and then saving once still writes all of them, not just the
        currently active image.
        """
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        if self._image_name and self._image is not None:
            self._dirty_labels[self._image_name] = self._labels
            self._dirty_geo[self._image_name] = (self._image.crs, self._image.transform)
        if not self._dirty_labels:
            return {"ok": False, "error": "No edited masks to save"}
        with self._lock:
            paths = []
            for name, labels in self._dirty_labels.items():
                user_path, _ = self._project.mask_paths(name)
                crs, transform = self._dirty_geo[name]
                write_label_mask(user_path, labels, crs, transform,
                                 self._project.mask_format)
                paths.append(user_path)
            self._dirty_labels.clear()
            self._dirty_geo.clear()
        return {"ok": True, "path": paths[0] if len(paths) == 1 else f"{len(paths)} masks"}

    # ---------- thumbnails ----------

    def generate_thumbnails(self):
        """(Re)start async generation of missing 128px thumbnails in .thumbnails/."""
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        self._project.start_thumbnail_generation()
        return {"ok": True}

    def get_thumbnails(self):
        """Thumbnail PNGs (base64, None while pending) plus generation progress."""
        if self._project is None:
            return {"running": False, "done": 0, "total": 0, "thumbs": {}}
        thumbs = {}
        for name in self._project.list_images():
            path = self._project.thumb_path(name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    thumbs[name] = base64.b64encode(f.read()).decode("ascii")
            else:
                thumbs[name] = None
        return {**self._project.thumb_progress, "thumbs": thumbs}

    # ---------- labeling progress ----------

    def _start_label_scan(self):
        """Count labeled pixels per image on a background thread.

        One mask read per image, so it runs for a second or two on a large
        project -- the same reason thumbnail generation is backgrounded. The
        counts are only a progress readout, so the strip shows what it has and
        fills in the rest as the scan lands.
        """
        if self._label_thread and self._label_thread.is_alive():
            return
        names = self._project.list_images() if self._project else []
        self._label_counts = {}
        self._label_scan = {"running": True, "done": 0, "total": len(names)}
        self._label_thread = threading.Thread(
            target=self._label_scan_worker, args=(names,), daemon=True)
        self._label_thread.start()

    @staticmethod
    def _mask_stats(mask):
        """{labeled, total} for one mask; coverage is derived in the frontend.

        Both numbers travel because coverage is the useful readout here: this
        dataset arrives with every image partly labeled, so "has any labels" is
        true everywhere and says nothing about what is left to do.
        """
        if mask is None:
            return {"labeled": 0, "total": 0}
        return {"labeled": int((mask != UNLABELED).sum()), "total": int(mask.size)}

    def _label_scan_worker(self, names):
        project = self._project
        for i, name in enumerate(names):
            if project is not self._project:
                return  # a different project opened; this scan is stale
            # The active image's labels may hold unsaved edits, so prefer the
            # in-memory buffer over the file for it.
            if name == self._image_name and self._labels is not None:
                mask = self._labels
            else:
                mask = read_label_mask(project.mask_paths(name)[0])
            self._label_counts[name] = self._mask_stats(mask)
            self._label_scan["done"] = i + 1
        self._label_scan["running"] = False

    def _note_label_count(self, name=None):
        """Refresh one image's stats from the in-memory buffer (no disk read)."""
        name = name or self._image_name
        if name is not None and self._labels is not None:
            self._label_counts[name] = self._mask_stats(self._labels)

    def get_label_stats(self):
        """{name: {labeled, total}} plus scan progress, for the thumbnail strip."""
        return {**self._label_scan, "counts": dict(self._label_counts)}

    @staticmethod
    def _file_dialog():
        """Pick a single existing GeoTIFF."""
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return None
        result = window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=False,
            file_types=("GeoTIFF (*.tif;*.tiff)", "All files (*.*)"))
        if isinstance(result, (tuple, list)):
            return result[0] if result else None
        return result

    @staticmethod
    def _folder_dialog():
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return None
        result = window.create_file_dialog(webview.FileDialog.FOLDER)
        if isinstance(result, (tuple, list)):
            return result[0] if result else None
        return result
