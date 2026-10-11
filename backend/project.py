"""Project configuration (project_config.json) and the .thumbnails cache.

A project is a folder of GeoTIFF images plus a config file describing the
data profile, the class palette and where masks are stored. Paths in the
config are relative to the project root.

project_config.json is shared with the full training app, so this build
validates and preserves the whole schema even where it has no use for part of
it -- the training split, the patch size and the PointRend flag are read,
checked and written back untouched rather than dropped.
"""

import json
import os
import re
import threading

import numpy as np
import rasterio
from data import (DEFAULT_MASK_FORMAT, MASK_DTYPES,
                  display_band_indices, stretch_to_unit)
from PIL import Image
from rasterio.enums import Resampling

CONFIG_NAME = "project_config.json"
THUMB_DIR = ".thumbnails"
THUMB_SIZE = 128
IMAGE_EXTENSIONS = (".tif", ".tiff")

# Label space everywhere (UI, masks, model classes): one id per entry in a
# project's `classes`, plus 255 unlabeled (data.UNLABELED, not a class).
#
# Starting palettes offered by the setup dialog. A project's list is editable
# from there afterwards, so these only decide what a new project opens with --
# which matters because retyping ten names and colours to start work is the
# kind of friction that gets a tool abandoned.
#
# "binary" is what this app shipped with when it did one-target segmentation.
# It is kept because a two-class project is still a perfectly good project, not
# because anything here assumes two.
CLASS_PRESETS = {
    "sentinel2": [
        {"id": 0, "name": "Urban", "color": "#e05a4f"},
        {"id": 1, "name": "Agriculture-Green", "color": "#3ddc6b"},
        {"id": 2, "name": "Raw-Earth/Agriculture", "color": "#c98f3c"},
        {"id": 3, "name": "Meadow", "color": "#d4e34a"},
        {"id": 4, "name": "Forest", "color": "#14562b"},
        {"id": 5, "name": "Shrub", "color": "#9ac48d"},
        {"id": 6, "name": "Water", "color": "#2f7ce0"},
        {"id": 7, "name": "Snow", "color": "#eef2f7"},
        {"id": 8, "name": "Wetland", "color": "#9b5fd0"},
    ],
    "binary": [
        {"id": 0, "name": "Target", "color": "#ff5028"},
        {"id": 1, "name": "Background", "color": "#3c8cff"},
    ],
}
DEFAULT_PRESET = "sentinel2"


def preset_classes(name=None):
    """A fresh copy of a starting palette, by name."""
    return [dict(c) for c in CLASS_PRESETS.get(name or DEFAULT_PRESET,
                                               CLASS_PRESETS[DEFAULT_PRESET])]


def classes_from_settings(raw, existing=None, next_id=None):
    """(classes, next_class_id) from the setup dialog's class rows.

    Ids are the label space: they are what every saved mask stores, so this is
    the one part of the config that cannot be rebuilt from the form each time.
    A row that carries an id the project already knows keeps it; a row without
    one is new and takes the next id from a high-water counter.

    That counter, not `max(id) + 1`, is what makes an id mean one thing
    forever. Deleting the last class and adding another would otherwise hand
    the newcomer the departed class's id, and every pixel of it still sitting
    in a backup, in masks_ai, or in a mask the training app wrote would come
    back to life as the wrong material. Ids are cheap; there are 255 of them.
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError("Invalid project config: 'classes' must be a non-empty list")
    known = {c["id"] for c in (existing or [])}
    counter = max([next_id or 0] + [i + 1 for i in known] + [0])
    classes, seen = [], set()
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid project config: classes[{i}] must be an object")
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError(f"Invalid project config: classes[{i}] needs a name")
        cid = row.get("id")
        if isinstance(cid, bool) or not isinstance(cid, int) or cid not in known:
            cid = counter
            counter += 1
        if cid in seen:
            raise ValueError(f"Invalid project config: duplicate class id {cid}")
        if cid > 254:
            raise ValueError("Invalid project config: out of class ids "
                             "(0-254 have all been used)")
        seen.add(cid)
        classes.append({"id": cid, "name": name,
                        "color": str(row.get("color", "")).strip()})
    return classes, counter


# Band names of the original 10-band Sentinel-2 stacks, used as the default
# naming when a project has exactly 10 bands.
SENTINEL2_BAND_NAMES = ["Red", "Green", "Blue", "NIR1", "RE1", "RE2", "RE3",
                        "NIR2", "SWIR1", "SWIR2"]

# The training split is not used by this build, but project_config.json is
# shared with the full app, so the schema is still validated and the values are
# still round-tripped untouched.
DEFAULT_VALIDATION_RATIO = 0.2
MAX_VALIDATION_RATIO = 0.9
SPLIT_ROLES = ("training", "validation")


def default_band_names(input_channels):
    if input_channels == len(SENTINEL2_BAND_NAMES):
        return list(SENTINEL2_BAND_NAMES)
    return [f"Band {i + 1}" for i in range(input_channels)]


def default_display_bands(input_channels):
    """0-based [R, G, B] band indices used for the on-screen RGB composite:
    the first three bands, with the last repeated when fewer than three exist."""
    return display_band_indices(None, input_channels)


def default_config(project_name, input_channels=10, patch_size=96, preset=None):
    classes = preset_classes(preset)
    return {
        "project_name": project_name,
        "data_profile": {"input_channels": input_channels,
                         "input_patch_size": [patch_size, patch_size],
                         "band_names": default_band_names(input_channels),
                         "display_bands": default_display_bands(input_channels),
                         "use_pointrend": False},
        "classes": classes,
        "next_class_id": max(c["id"] for c in classes) + 1,
        "split": {"validation_ratio": DEFAULT_VALIDATION_RATIO,
                  "overrides": {}},
        "paths": {"images_folder": ".", "masks_user": "masks_user", "masks_ai": "masks_ai"},
    }


def config_from_settings(settings, existing_classes=None, next_class_id=None):
    """Config dict from the frontend's project-setup form (raises ValueError).

    `existing_classes` and `next_class_id` carry the open project's label space
    in when this is an edit, so ids survive the rebuild; both are None when
    creating, and the palette then comes from `settings["classes"]` or, failing
    that, from the named preset.
    """
    try:
        channels = int(settings["input_channels"])
        size = int(settings["input_patch_size"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid project config: bands and patch size must be integers")
    names = settings.get("band_names") or default_band_names(channels)
    raw_display = settings.get("display_bands")
    try:
        display = display_band_indices(raw_display, channels)
    except (TypeError, ValueError):
        raise ValueError("Invalid project config: display_bands must be integers")
    raw_ratio = settings.get("validation_ratio", DEFAULT_VALIDATION_RATIO)
    try:
        ratio = round(float(DEFAULT_VALIDATION_RATIO if raw_ratio is None else raw_ratio), 3)
    except (TypeError, ValueError):
        raise ValueError("Invalid project config: validation_ratio must be a number")
    ratio = min(max(ratio, 0.0), MAX_VALIDATION_RATIO)
    # The palette is the dialog's to define. It used to be overwritten with the
    # two-class default here no matter what came in, which is what made the
    # class list uneditable everywhere upstream: the create dialogs had nothing
    # to gain by offering it, and editing one meant merging it back in after
    # this function had thrown it away.
    # `is not None`, not truthiness: an empty list is a broken submission, not
    # an absent one, and answering it with the default palette would hand back
    # ten classes to a caller that asked for none.
    raw_classes = settings.get("classes")
    if raw_classes is not None:
        classes, next_id = classes_from_settings(raw_classes, existing_classes,
                                                 next_class_id)
    else:
        classes = preset_classes(settings.get("class_preset"))
        next_id = max(c["id"] for c in classes) + 1
    config = {
        "project_name": str(settings.get("project_name", "")).strip() or "project",
        "data_profile": {
            "input_channels": channels,
            "input_patch_size": [size, size],
            "band_names": [str(n).strip() for n in names],
            "display_bands": display,
            "use_pointrend": bool(settings.get("use_pointrend", False)),
        },
        "classes": classes,
        "next_class_id": next_id,
        # Overrides are per-image and only exist once the user moves one, so a
        # freshly built config always starts from the plain ratio.
        "split": {"validation_ratio": ratio, "overrides": {}},
        "paths": {
            "images_folder": str(settings.get("images_folder", "")).strip() or ".",
            "masks_user": str(settings.get("masks_user", "")).strip() or "masks_user",
            "masks_ai": str(settings.get("masks_ai", "")).strip() or "masks_ai",
        },
    }
    return validate_config(config)


def probe_band_count(folder):
    """Band count of the first readable GeoTIFF in folder, or None."""
    try:
        names = sorted(n for n in os.listdir(folder) if n.lower().endswith(IMAGE_EXTENSIONS))
    except FileNotFoundError:
        return None
    for name in names:
        try:
            with rasterio.open(os.path.join(folder, name)) as src:
                return src.count
        except Exception:
            continue
    return None


def _expect(condition, message):
    """Shared by every section check below; the message is the whole point."""
    if not condition:
        raise ValueError(f"Invalid project config: {message}")


def _check_data_profile(profile):
    """The raster shape the project expects: bands, patch size, band names.

    The patch size and the PointRend flag belong to the training app rather
    than to this build, but the schema is shared -- so they are validated here
    and round-tripped untouched rather than dropped.
    """
    _expect(isinstance(profile, dict), "'data_profile' must be an object")
    _expect(isinstance(profile.get("input_channels"), int) and profile["input_channels"] > 0,
            "'data_profile.input_channels' must be a positive integer")
    patch = profile.get("input_patch_size")
    _expect(isinstance(patch, list) and len(patch) == 2
            and all(isinstance(v, int) and v > 0 for v in patch),
            "'data_profile.input_patch_size' must be [int, int]")
    _expect(patch[0] == patch[1],
            "'data_profile.input_patch_size' must be square, e.g. [96, 96]")
    _expect(patch[0] % 32 == 0,
            "'data_profile.input_patch_size' must be a multiple of 32 "
            "(the encoder downsamples 32x)")

    band_names = profile.get("band_names")
    if band_names is not None:  # optional: older configs don't have it
        _expect(isinstance(band_names, list)
                and all(isinstance(n, str) and n.strip() for n in band_names),
                "'data_profile.band_names' must be a list of non-empty strings")
        _expect(len(band_names) == profile["input_channels"],
                "'data_profile.band_names' must have one name per input channel")

    use_pointrend = profile.get("use_pointrend")
    if use_pointrend is not None:  # optional: older configs default to off
        _expect(isinstance(use_pointrend, bool),
                "'data_profile.use_pointrend' must be a boolean")

    display_bands = profile.get("display_bands")
    if display_bands is not None:  # optional: older configs default to first three
        _expect(isinstance(display_bands, list) and len(display_bands) == 3
                and all(isinstance(b, int) and 0 <= b < profile["input_channels"]
                        for b in display_bands),
                "'data_profile.display_bands' must be three band indices in "
                "[0, input_channels)")


def _check_classes(config):
    """The label space. Every saved mask stores these ids, so this is the
    section where a mistake changes what existing ground truth means."""
    classes = config.get("classes")
    _expect(isinstance(classes, list) and classes, "'classes' must be a non-empty list")
    seen_ids = set()
    for i, cls in enumerate(classes):
        _expect(isinstance(cls, dict), f"classes[{i}] must be an object")
        _expect(isinstance(cls.get("id"), int) and 0 <= cls["id"] <= 254,
                f"classes[{i}].id must be an integer in [0, 254]")
        _expect(cls["id"] not in seen_ids, f"duplicate class id {cls['id']}")
        seen_ids.add(cls["id"])
        _expect(isinstance(cls.get("name"), str) and cls["name"].strip(),
                f"classes[{i}].name must be a non-empty string")
        _expect(isinstance(cls.get("color"), str)
                and re.fullmatch(r"#[0-9a-fA-F]{6}", cls["color"]),
                f"classes[{i}].color must be a '#rrggbb' string")

    # High-water mark for class ids (see classes_from_settings). Optional:
    # configs written before the class editor existed have none, and a project
    # opened from one starts its counter above whatever ids it already carries.
    next_class_id = config.get("next_class_id")
    if next_class_id is not None:
        _expect(isinstance(next_class_id, int) and not isinstance(next_class_id, bool)
                and 0 <= next_class_id <= 255,
                "'next_class_id' must be an integer in [0, 255]")
        _expect(next_class_id > max(c["id"] for c in classes),
                "'next_class_id' must be above every id in 'classes', or a new "
                "class would take an id that is already in use")


def _check_split(split):
    """Unused by this build; validated and round-tripped for the training app."""
    if split is None:
        return
    _expect(isinstance(split, dict), "'split' must be an object")
    ratio = split.get("validation_ratio")
    _expect(isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
            and 0.0 <= ratio <= MAX_VALIDATION_RATIO,
            f"'split.validation_ratio' must be a number in "
            f"[0, {MAX_VALIDATION_RATIO}]")
    overrides = split.get("overrides")
    if overrides is not None:
        _expect(isinstance(overrides, dict), "'split.overrides' must be an object")
        for name, role in overrides.items():
            _expect(isinstance(name, str) and name.strip(),
                    "'split.overrides' keys must be non-empty image names")
            _expect(role in SPLIT_ROLES,
                    f"split.overrides['{name}'] must be one of {SPLIT_ROLES}")


def _check_mask_format(mask_format):
    """How labels are stored on disk. Optional: older configs mean uint8/255."""
    if mask_format is None:
        return
    _expect(isinstance(mask_format, dict), "'mask_format' must be an object")
    _expect(mask_format.get("dtype") in MASK_DTYPES,
            f"'mask_format.dtype' must be one of {MASK_DTYPES}")
    nodata = mask_format.get("nodata")
    _expect(isinstance(nodata, int) and not isinstance(nodata, bool),
            "'mask_format.nodata' must be an integer")
    info = np.iinfo(mask_format["dtype"])
    _expect(info.min <= nodata <= info.max,
            f"'mask_format.nodata' ({nodata}) is not representable in "
            f"{mask_format['dtype']}")
    # A nodata inside the class range would make unlabeled pixels
    # indistinguishable from that class on the next read.
    _expect(not (0 <= nodata <= 254),
            "'mask_format.nodata' must fall outside the class id range "
            "[0, 254], or unlabeled pixels would read back as that class")


def _check_paths(paths):
    _expect(isinstance(paths, dict), "'paths' must be an object")
    for key in ("images_folder", "masks_user", "masks_ai"):
        value = paths.get(key)
        _expect(isinstance(value, str) and value.strip(),
                f"'paths.{key}' must be a non-empty string")
        _expect(not os.path.isabs(value),
                f"'paths.{key}' must be relative to the project root")


def validate_config(config):
    """Return the config if it matches the schema, else raise ValueError.

    One section per helper, in the order they appear in the file. Each is
    independently readable, which is what this was not when it was 112 lines of
    one function and you had to find the section you cared about by counting
    blank lines.
    """
    _expect(isinstance(config, dict), "root must be a JSON object")
    _expect(isinstance(config.get("project_name"), str) and config["project_name"].strip(),
            "'project_name' must be a non-empty string")
    _check_data_profile(config.get("data_profile"))
    _check_classes(config)
    _check_split(config.get("split"))
    _check_mask_format(config.get("mask_format"))
    _check_paths(config.get("paths"))
    return config


def config_path(root):
    return os.path.join(root, CONFIG_NAME)


def load_config(root):
    with open(config_path(root), encoding="utf-8") as f:
        return validate_config(json.load(f))


def save_config(root, config):
    """Atomically write the validated config; returns its path."""
    validate_config(config)
    path = config_path(root)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path


class Project:
    """An open project: config, resolved folders and thumbnail cache state."""

    def __init__(self, root, config):
        self.root = os.path.abspath(root)
        self.config = config
        self.thumb_progress = {"running": False, "done": 0, "total": 0}
        self._thumb_thread = None
        for folder in (self.masks_user_dir, self.masks_ai_dir, self.thumb_dir):
            os.makedirs(folder, exist_ok=True)

    @classmethod
    def create(cls, root):
        """New project rooted at an image directory, with default config.

        input_channels is inferred from the first image found so the model
        matches the project's data (e.g. 3-band RGB or 4-band RGBN rasters).
        """
        name = os.path.basename(os.path.normpath(root)) or "project"
        channels = probe_band_count(root) or 10
        return cls(root, default_config(name, input_channels=channels))

    @classmethod
    def open(cls, root):
        return cls(root, load_config(root))

    def _resolve(self, relative):
        return os.path.normpath(os.path.join(self.root, relative))

    @property
    def images_dir(self):
        return self._resolve(self.config["paths"]["images_folder"])

    @property
    def masks_user_dir(self):
        return self._resolve(self.config["paths"]["masks_user"])

    @property
    def masks_ai_dir(self):
        return self._resolve(self.config["paths"]["masks_ai"])

    @property
    def thumb_dir(self):
        return os.path.join(self.root, THUMB_DIR)

    @property
    def input_channels(self):
        return self.config["data_profile"]["input_channels"]

    @property
    def display_bands(self):
        """0-based [R, G, B] band indices for previews, or None (first three)."""
        return self.config["data_profile"].get("display_bands")

    @property
    def mask_format(self):
        """On-disk {dtype, nodata} for label masks (see data.write_label_mask)."""
        return self.config.get("mask_format") or dict(DEFAULT_MASK_FORMAT)

    def list_images(self):
        try:
            return sorted(name for name in os.listdir(self.images_dir)
                          if name.lower().endswith(IMAGE_EXTENSIONS))
        except FileNotFoundError:
            return []

    def image_path(self, name):
        return os.path.join(self.images_dir, name)

    def mask_paths(self, image_name):
        """(user_mask_path, ai_mask_path) for an image, always .tif."""
        base = os.path.splitext(image_name)[0] + ".tif"
        return os.path.join(self.masks_user_dir, base), os.path.join(self.masks_ai_dir, base)

    def thumb_path(self, image_name):
        return os.path.join(self.thumb_dir, os.path.splitext(image_name)[0] + ".png")

    def start_thumbnail_generation(self, force=False):
        """Generate thumbnails on a background thread.

        Existing files are kept unless force=True, which regenerates (overwrites)
        every thumbnail with the project's current display bands — used when the
        RGB display mapping changes.
        """
        if self._thumb_thread and self._thumb_thread.is_alive():
            return
        names = self.list_images()
        self.thumb_progress = {"running": True, "done": 0, "total": len(names)}
        self._thumb_thread = threading.Thread(target=self._thumb_worker,
                                              args=(names, force), daemon=True)
        self._thumb_thread.start()

    def _thumb_worker(self, names, force=False):
        bands = self.display_bands
        for i, name in enumerate(names):
            path = self.thumb_path(name)
            if force or not os.path.exists(path):
                try:
                    generate_thumbnail(self.image_path(name), path, bands=bands)
                except Exception:
                    pass  # unreadable image: skip, leave no thumbnail
            self.thumb_progress["done"] = i + 1
        self.thumb_progress["running"] = False


def generate_thumbnail(image_path, thumb_path, size=THUMB_SIZE, bands=None):
    """Write an RGB PNG preview (bounded by size x size).

    `bands` is a 3-list of 0-based band indices mapped to R, G, B (the project's
    display_bands); when omitted it uses the first three bands (last repeated if
    fewer). Decimated reading keeps this cheap for large rasters; each channel is
    percentile-stretched like the main viewer so thumbnails match the panes.
    """
    with rasterio.open(image_path) as src:
        scale = size / max(src.width, src.height)
        out_h = max(1, round(src.height * scale))
        out_w = max(1, round(src.width * scale))
        idx = display_band_indices(bands, src.count)
        data = src.read([i + 1 for i in idx], out_shape=(3, out_h, out_w),
                        resampling=Resampling.average).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0

    # The same stretch the labeling pane applies, from the same function --
    # which is what makes "thumbnails match the panes" a property rather than a
    # coincidence. Computed on the decimated read, so the limits come from the
    # pixels actually being drawn.
    valid = np.all(data != nodata, axis=0)
    rgb = np.transpose((stretch_to_unit(data, valid) * 255).astype(np.uint8),
                       (1, 2, 0))

    tmp = thumb_path + ".tmp"
    Image.fromarray(rgb).save(tmp, format="PNG")
    os.replace(tmp, thumb_path)  # atomic: readers never see a half-written file
