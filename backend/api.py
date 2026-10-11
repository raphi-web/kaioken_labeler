"""pywebview bridge: the Api class is exposed to JS as window.pywebview.api.

Every method is called by pywebview on a worker thread and returns
JSON-serializable data, so each JS call resolves as a promise.

This is the labeling-only build: no model, no training, no inference and no
accuracy report, so nothing here imports torch. Masks are painted, autosaved
and read back; SAM2 click-assist runs on onnxruntime.
"""

import base64
import os
import shutil
import threading
import traceback

import numpy as np
import rasterio
import webview

from data import (UNLABELED, SentinelImage, array_to_png_b64, read_label_mask,
                  tile_geotiff, write_label_mask)
from project import (CLASS_PRESETS, DEFAULT_PRESET, Project,
                     config_from_settings, config_path, default_config,
                     probe_band_count, save_config)
from sam_service import SamService, mask_to_polygons
from wand import Wand, WandError, parse_options


def _class_defaults(config):
    """The palette parts of a setup dialog's `defaults`.

    The presets travel with them rather than being restated in the frontend:
    the taxonomy is one thing, and a copy of it in JS would drift from the one
    that actually gets written into project_config.json.
    """
    return {"classes": [dict(c) for c in config["classes"]],
            "class_preset": DEFAULT_PRESET,
            "class_presets": {name: [dict(c) for c in classes]
                              for name, classes in CLASS_PRESETS.items()}}


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
        # The wand keeps its own caches, keyed on the image (see Wand.set_image).
        # It no longer shares anything with the snap tool: both of its sources
        # (spectral, mean-shift) read the raster's own bands.
        self._wand = Wand()
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
        if project_root:
            result = self._init_project(project_root)
            if not result["ok"]:
                raise ValueError(result["error"])
        elif image_path:
            self._set_image(SentinelImage(image_path),
                            os.path.basename(image_path))

    def _set_image(self, image, name, labels=None):
        """Swap the active image and its label buffer."""
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

    def wand_select(self, samples, options=None):
        """Selection around `samples`, bit-packed for the wire.

        samples: [[x, y], ...] in image pixel coordinates. `options` is one
        object rather than a growing positional list, so the JS side stays
        readable and a new knob is one key instead of another argument nobody
        can remember the order of; wand.parse_options documents every key.

        The algorithm lives in wand.py. What happens here is the translation
        this side of the bridge owns: the image and the label buffer go in, a
        WandError comes back as a message the pane can show, and the mask goes
        out bit-packed -- 32 KB for a 512x512 image rather than the 256 KB a
        byte-per-pixel buffer would cost on every slider step.

        `available` reports what the selection would have been without the
        budget, so the UI can say how hard the cap is biting.
        """
        # Ahead of parse_options, so a click with no image open says so rather
        # than reporting whichever option it also dislikes. Wand.select guards
        # this too, for callers that reach it directly.
        if self._image is None:
            return {"ok": False, "error": "No image loaded"}
        try:
            options = parse_options(options)
            # Cheap when the key is unchanged, which is what lets this be
            # unconditional: there is no invalidation hook here to forget.
            # The wand's key omits the display bands (both sources read every
            # band); the snap tool's key below keeps them.
            self._wand.set_image(self._image, self._wand_image_key())
            result = self._wand.select(samples, options, self._labels)
        except WandError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:  # a model/runtime error must not crash the bridge
            traceback.print_exc()
            return {"ok": False, "error": f"Wand failed: {e}"}
        packed = np.packbits(result.mask.reshape(-1))
        return {"ok": True,
                "mask": base64.b64encode(packed.tobytes()).decode("ascii"),
                "count": result.count,
                "available": result.available,
                "protected": result.protected,
                "capped": result.capped,
                "valid": result.valid,
                "width": result.width, "height": result.height}

    # ---------- SAM2 snap-to-edge assist ----------

    def sam_available(self):
        """Whether the Efficient-SAM2 click assist can run (files present)."""
        reason = self._sam.unavailable_reason()
        return {"available": reason is None, "reason": reason}

    def _wand_image_key(self):
        """Cache key for the wand: anything that changes the raster's bands.

        Deliberately NOT the display bands -- both wand sources read every
        band, so re-keying on a display change would throw away valid caches.
        """
        root = self._project.root if self._project is not None else None
        return (root, self._image_name)

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


    # ---------- classes ----------

    def _iter_label_masks(self):
        """(name, labels, on_disk) for every image of the project.

        The freshest copy of each: this session's edits if there are any, the
        saved mask otherwise. Which matters for both callers below -- a class
        the user painted two minutes ago and has not saved is still in use, and
        deleting it has to clear those pixels too, not just the ones on disk.
        """
        if self._project is None:
            return
        for name in self._project.list_images():
            labels = self._dirty_labels.get(name)
            if labels is None and name == self._image_name:
                labels = self._labels
            if labels is not None:
                yield name, labels, False
                continue
            saved = read_label_mask(self._project.mask_paths(name)[0])
            if saved is not None:
                yield name, saved, True

    def class_usage(self, ids):
        """{id: {pixels, images}} for class ids across the whole project.

        Read-only, and the reason the settings dialog can warn before it
        removes anything: "this deletes 12,345 px across 7 images" is a
        different decision from "this deletes a row from a list".

        One mask read per image, so it runs for a second or two on a large
        project -- called when a row is removed, not on every keystroke.
        """
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        try:
            wanted = [int(i) for i in (ids or [])]
        except (TypeError, ValueError):
            return {"ok": False, "error": "Invalid class ids"}
        usage = {i: {"pixels": 0, "images": 0} for i in wanted}
        if wanted:
            for _, labels, _ in self._iter_label_masks():
                for i in wanted:
                    hit = int((labels == i).sum())
                    if hit:
                        usage[i]["pixels"] += hit
                        usage[i]["images"] += 1
        # JSON object keys are strings on the way through the bridge; make that
        # explicit here rather than leaving the frontend to guess.
        return {"ok": True, "usage": {str(i): v for i, v in usage.items()}}

    def _mask_geometry(self, name):
        """(crs, transform) to write `name`'s mask back with.

        Preferring the mask already on disk over the image keeps a rewrite
        byte-identical in georeferencing to what was there, even if the two
        ever disagree.
        """
        if name == self._image_name and self._image is not None:
            return self._image.crs, self._image.transform
        if name in self._dirty_geo:
            return self._dirty_geo[name]
        user_path, _ = self._project.mask_paths(name)
        for path in (user_path, self._project.image_path(name)):
            try:
                with rasterio.open(path) as src:
                    return src.crs, src.transform
            except Exception:
                continue
        return None, None

    def _unlabel_classes(self, ids):
        """Reset every pixel of `ids` to UNLABELED across the user masks.

        Deleting a class from the config without this would leave masks storing
        an id nothing explains: it would read back as a class that is not in
        the palette, paint as an unknown colour, and quietly train as a
        category the model was never told about.

        In-memory buffers are edited in place *and* written out, so the copy on
        disk and the one being labeled cannot disagree about a class that no
        longer exists. masks_ai is deliberately untouched -- this build never
        reads it and the training app owns it.
        """
        removed = np.array(sorted(set(int(i) for i in ids)), dtype=np.uint8)
        pixels = images = 0
        for name, labels, _ in list(self._iter_label_masks()):
            hit = np.isin(labels, removed)
            count = int(hit.sum())
            if not count:
                continue
            labels[hit] = UNLABELED  # in place: this may be the live buffer
            crs, transform = self._mask_geometry(name)
            write_label_mask(self._project.mask_paths(name)[0], labels, crs,
                             transform, self._project.mask_format)
            pixels += count
            images += 1
            if name == self._image_name:
                self._note_label_count(name)
            else:
                self._label_counts[name] = self._mask_stats(labels)
        return {"pixels": pixels, "images": images}

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
                    **_class_defaults(config),
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
                    **_class_defaults(config),
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

        The project name, the class palette, band names, display bands and
        folder paths are editable here. Changing a folder path reopens the
        project so the new folders take effect (mask dirs are created and
        images reload).

        Removing a class rewrites the saved masks: every pixel of a removed id
        becomes unlabeled, project-wide, before the new config is written. That
        is destructive and irreversible, which is why class_usage() exists for
        the dialog to warn with first -- but the alternative is masks storing
        ids the palette cannot explain, and a config that lies about its own
        label space is worse than a deletion the user asked for and saw
        measured.

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
            # The open project's label space goes in, so ids survive the
            # rebuild and new rows take ids that have never been used.
            config = config_from_settings(merged, existing_classes=old["classes"],
                                          next_class_id=old.get("next_class_id"))
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
            # Masks first, config second. Both orders can fail badly and this
            # one fails better: if the rewrite throws, the config still
            # describes the palette the masks were written against, and the
            # save can be retried. The reverse would leave a config whose label
            # space the masks on disk contradict.
            kept = {c["id"] for c in config["classes"]}
            gone = [c["id"] for c in old["classes"] if c["id"] not in kept]
            cleared = None
            if gone:
                with self._lock:
                    cleared = self._unlabel_classes(gone)
            save_config(self._project.root, config)
        except (OSError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        if config["paths"] != old["paths"]:
            result = self._init_project(self._project.root)
            result["reopened"] = True
            result["cleared"] = cleared
            return result
        self._project.config = config
        result = {"ok": True, "project": self.get_project(), "cleared": cleared}
        if cleared and cleared["pixels"] and self._image is not None:
            # The frontend is holding the labels this just edited. Without the
            # refreshed buffer it would push its stale copy back on the next
            # stroke and resurrect the pixels of a class that no longer exists.
            result["image"] = self._image_payload()
        if config["data_profile"]["display_bands"] != profile.get("display_bands"):
            # New RGB mapping: re-render the open image (labels/geometry unchanged,
            # so the frontend swaps only the base png) and rebuild every cached
            # thumbnail, overwriting the old ones.
            #
            # The wand needs no hand-invalidation here: both of its sources read
            # every band of the raster, not the display composite, so a display
            # change leaves its cached fields correct. Its key (_wand_image_key)
            # deliberately omits the display bands for exactly that reason.
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

    def clear_image_labels(self, name):
        """Reset every labeled pixel of one image to UNLABELED.

        Per-image counterpart to the toolbar's Clear Labels: same action,
        on a different image. The user_mask is rewritten with everything
        != UNLABELED turned to UNLABELED; the in-memory buffer is reset if
        this is the active image so a subsequent stroke does not bring the
        cleared pixels back.

        Returns the cleared pixel count and a flag for whether the active
        image was affected, so the frontend can reload its label buffer.
        """
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        if name not in self._project.list_images():
            return {"ok": False, "error": f"Unknown image: {name}"}

        is_active = name == self._image_name
        pixels = 0
        with self._lock:
            if is_active and self._labels is not None:
                # Active-image branch: the buffer is in memory; mutate in
                # place and write through. Mirrors how handleClearLabels
                # in App.jsx mutates before pushLabels().
                labels = self._labels
                hit = labels != UNLABELED
                pixels = int(hit.sum())
                if pixels:
                    labels[hit] = UNLABELED
                    user_path, _ = self._project.mask_paths(name)
                    write_label_mask(user_path, labels, self._image.crs,
                                     self._image.transform,
                                     self._project.mask_format)
                self._note_label_count(name)
            else:
                # Non-active branch: read the saved mask, modify, write.
                user_path, _ = self._project.mask_paths(name)
                existing = read_label_mask(user_path)
                if existing is None:
                    self._label_counts[name] = self._mask_stats(None)
                    return {"ok": True, "pixels": 0, "active": is_active}
                hit = existing != UNLABELED
                pixels = int(hit.sum())
                if pixels:
                    existing[hit] = UNLABELED
                    # Geometry for non-active images: prefer the dirty-geo
                    # cache (covers edits not yet autosaved), then the saved
                    # mask's georef, then the image file's. Last resort is
                    # None/None; rasterio handles that on write.
                    if name in self._dirty_geo:
                        crs, transform = self._dirty_geo[name]
                    else:
                        crs, transform = None, None
                        try:
                            with rasterio.open(user_path) as src:
                                crs, transform = src.crs, src.transform
                        except Exception:
                            try:
                                with rasterio.open(self._project.image_path(name)) as src:
                                    crs, transform = src.crs, src.transform
                            except Exception:
                                pass
                    write_label_mask(user_path, existing, crs, transform,
                                     self._project.mask_format)
                self._label_counts[name] = self._mask_stats(existing)
        return {"ok": True, "pixels": pixels, "active": is_active}

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

    def add_image(self, source_path, force=False):
        """Copy a user-chosen GeoTIFF into the project's images folder.

        The destination filename is the source's basename -- the project already
        lists images by name, so two files of the same name in the same folder
        are at best a hidden conflict and at worst a silent overwrite. A band
        mismatch with the project's `data_profile.input_channels` is rejected
        outright: the model the masks feed into is fixed at construction time,
        and adding a 3-band image to a 10-band project would turn every label
        we paint into a different material.

        A SIZE mismatch is the only soft check, because users sometimes add
        a differently-sized tile to a project that was tiled from a larger
        raster (the on-disk image has its own grid). That is reported back with
        `requires_confirm=True` so the frontend can ask for an explicit yes
        before the copy is committed.

        A copy, not a move: the user's source folder is rarely the project
        folder, and silently relocating well-named files is the kind of
        "I did not mean to do that" moment the test suite is supposed to catch.

        The new image is added to the project list but NOT loaded as the
        active image. The user picked the file from a menu, not from the
        thumbnail strip, so stealing the current image would be a surprising
        side effect.
        """
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        if not source_path or not os.path.isfile(source_path):
            return {"ok": False, "error": "Pick a file to add"}
        name = os.path.basename(source_path)
        if not name:
            return {"ok": False, "error": "Pick a file to add"}
        try:
            with rasterio.open(source_path) as src:
                bands = src.count
                width, height = src.width, src.height
        except Exception as e:
            return {"ok": False, "error": f"Could not read that file: {e}"}
        if bands != self._project.input_channels:
            return {"ok": False, "error":
                    f"Expected a {self._project.input_channels}-band stack "
                    f"(project data_profile), got {bands} bands"}
        existing = self._project.list_images()
        if name in existing:
            return {"ok": False, "error":
                    f"An image named '{name}' already exists in this project"}
        # The size check is the only thing that asks before committing. The
        # "no images yet" case is the same as "matching size" -- there is
        # nothing to compare against, so the dialog has nothing to say.
        if not force and existing:
            with rasterio.open(self._project.image_path(existing[0])) as src:
                ref = {"width": src.width, "height": src.height,
                       "name": existing[0]}
            if (width, height) != (ref["width"], ref["height"]):
                return {"ok": False, "requires_confirm": True,
                        "existing": ref,
                        "candidate": {"width": width, "height": height,
                                      "name": name, "bands": bands}}
        try:
            # shutil.copy2 keeps the source's mtime/atime so the on-disk
            # project ordering matches the file's history rather than the
            # moment of import (small, but the kind of thing someone notices).
            shutil.copy2(source_path, self._project.image_path(name))
        except OSError as e:
            return {"ok": False, "error": f"Could not copy {name}: {e}"}
        # The label scan and the thumbnail scan are both keyed on the on-disk
        # list, so the new file lands in their next pass without an explicit
        # kick. start_thumbnail_generation() is the cheap one; it picks up
        # the new file on restart when needed, and `force=True` only matters
        # when an existing thumbnail has gone stale for another reason.
        self._start_label_scan()
        return {"ok": True, "name": name, "width": width, "height": height,
                "bands": bands, "project": self.get_project()}

    def pick_file(self, title="Pick a file"):
        """Open the native file picker and return its first selection, or null.

        The frontend's Add Image path is the only caller: it needs the
        absolute path of the GeoTIFF the user wants to import, and the
        pywebview dialog is the only thing that has access to that. A null
        return means the user cancelled, which is not an error.
        """
        path = self._file_dialog()
        return {"path": path}


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
