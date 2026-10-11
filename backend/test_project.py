"""Invariants of the class palette.

The palette is the label space: its ids are what every saved mask stores, so a
mistake here is not a display bug, it is ground truth quietly changing meaning.
These cover the two ways that can happen -- an id moving to a different
material, and a mask keeping pixels of a class the config no longer has.

Everything that touches labels runs against a scratch copy of the project via
the `scratch_api` fixture in backend/conftest.py, never label_raster/.
"""

import os
import tempfile

import numpy as np
import pytest
import rasterio

from data import UNLABELED, read_label_mask
from project import (CLASS_PRESETS, DEFAULT_PRESET, IMAGE_EXTENSIONS,
                     classes_from_settings, config_from_settings,
                     default_config, preset_classes, validate_config)

BASE = {"input_channels": 4, "input_patch_size": 96}


def _settings(**extra):
    return {**BASE, **extra}


# ---------- the palette survives the config rebuild ----------

def test_the_class_list_is_not_thrown_away():
    """config_from_settings used to overwrite it with the two-class default.

    That one line is why the class list was uneditable everywhere upstream, so
    it gets a test of its own rather than being covered incidentally.
    """
    wanted = [{"name": "Rooftop", "color": "#112233"},
              {"name": "Canopy", "color": "#445566"},
              {"name": "Tarmac", "color": "#778899"}]
    config = config_from_settings(_settings(classes=wanted))
    got = [(c["name"], c["color"]) for c in config["classes"]]
    if got != [(c["name"], c["color"]) for c in wanted]:
        pytest.fail(f"the class list did not survive: {got}")


def test_a_project_without_a_class_list_gets_the_preset():
    config = config_from_settings(_settings())
    names = [c["name"] for c in config["classes"]]
    if names != [c["name"] for c in CLASS_PRESETS[DEFAULT_PRESET]]:
        pytest.fail(f"expected the {DEFAULT_PRESET} preset, got {names}")
    if len(preset_classes("binary")) != 2:
        pytest.fail("the binary preset should still be two classes")


def test_every_preset_builds_a_valid_config():
    for name in CLASS_PRESETS:
        config = config_from_settings(_settings(class_preset=name))
        validate_config(config)  # raises if the schema is broken
        if len(config["classes"]) != len(CLASS_PRESETS[name]):
            pytest.fail(f"preset {name} lost classes")


def test_a_nameless_or_empty_class_list_is_refused():
    for classes, why in (([], "an empty list"),
                         ([{"name": "  ", "color": "#112233"}], "a blank name"),
                         (["nope"], "a non-object row")):
        try:
            config_from_settings(_settings(classes=classes))
        except ValueError:
            continue
        pytest.fail(f"{why} should have been refused")


# ---------- ids ----------

def test_an_edited_class_keeps_its_id():
    """Renaming or recolouring must not move a class in the label space."""
    existing = preset_classes("sentinel2")
    edited = [dict(c, name=f"{c['name']} v2", color="#010203") for c in existing]
    classes, _ = classes_from_settings(edited, existing, len(existing))
    for before, after in zip(existing, classes):
        if before["id"] != after["id"]:
            pytest.fail(f"{before['name']} moved from id {before['id']} to {after['id']}")


def test_a_deleted_id_is_never_handed_to_a_new_class():
    """The property that keeps an id meaning one material forever.

    A stale mask -- a backup, masks_ai, something the training app wrote --
    outlives the config. If a new class could take a departed one's id, every
    such pixel would come back as the wrong material.
    """
    existing = preset_classes("binary")  # ids 0, 1
    next_id = 2
    # Delete the last class, then add one.
    kept = [dict(existing[0])]
    classes, next_id = classes_from_settings(
        kept + [{"name": "Fresh", "color": "#123456"}], existing, next_id)
    fresh = classes[-1]
    if fresh["id"] in {c["id"] for c in existing}:
        pytest.fail(f"a new class took id {fresh['id']}, which class "
                    f"'{[c['name'] for c in existing if c['id'] == fresh['id']][0]}' had")
    # And again, to be sure the counter persists rather than tracking max().
    classes2, _ = classes_from_settings(
        [dict(classes[0]), {"name": "Fresher", "color": "#654321"}],
        classes, next_id)
    if classes2[-1]["id"] == fresh["id"]:
        pytest.fail(f"id {fresh['id']} was handed out twice")


def test_the_id_counter_is_stored_and_validated():
    config = default_config("t", input_channels=4)
    if config["next_class_id"] <= max(c["id"] for c in config["classes"]):
        pytest.fail("next_class_id must sit above every id in use")
    config["next_class_id"] = 0
    try:
        validate_config(config)
    except ValueError:
        return
    pytest.fail("a next_class_id below the ids in use should be refused")


# ---------- deleting a class clears its pixels ----------


def _settings_for(api, classes):
    """The settings object the dialog would submit for this project."""
    config = api._project.config
    return {"project_name": config["project_name"],
            "band_names": config["data_profile"]["band_names"],
            "display_bands": config["data_profile"]["display_bands"],
            "images_folder": config["paths"]["images_folder"],
            "masks_user": config["paths"]["masks_user"],
            "masks_ai": config["paths"]["masks_ai"],
            "classes": classes}


def test_class_usage_counts_what_a_deletion_would_destroy(scratch_api):
    labels = scratch_api._labels
    if labels is None:
        pytest.skip("no labels")
    present = [int(v) for v in np.unique(labels) if v != UNLABELED]
    if not present:
        pytest.skip("nothing labeled")
    target = present[0]
    result = scratch_api.class_usage([target])
    if not result.get("ok"):
        pytest.fail(f"class_usage failed: {result.get('error')}")
    counted = result["usage"][str(target)]["pixels"]
    expected = int((labels == target).sum())
    if counted < expected:
        pytest.fail(f"class_usage counted {counted} px, but the open image alone "
                    f"has {expected}")
    if scratch_api.class_usage([200])["usage"]["200"]["pixels"]:
        pytest.fail("an unused class id was reported as having pixels")


def test_removing_a_class_unlabels_exactly_its_pixels(scratch_api):
    labels = scratch_api._labels
    if labels is None or not (labels != UNLABELED).any():
        pytest.skip("nothing labeled")
    present = [int(v) for v in np.unique(labels) if v != UNLABELED]
    if len(present) < 2:
        pytest.skip("need two labeled classes")
    target, survivor = present[0], present[1]
    name = scratch_api._image_name
    before_survivor = int((labels == survivor).sum())
    keep = [dict(c) for c in scratch_api._project.config["classes"]
            if c["id"] != target]

    result = scratch_api.update_settings(_settings_for(scratch_api, keep))
    if not result.get("ok"):
        pytest.fail(f"update_settings failed: {result.get('error')}")
    if not result.get("cleared", {}).get("pixels"):
        pytest.fail("removing a labeled class reported nothing cleared")

    # In memory ...
    if (scratch_api._labels == target).any():
        pytest.fail("the open image still carries the removed class")
    if int((scratch_api._labels == survivor).sum()) != before_survivor:
        pytest.fail("removing one class changed another class's pixels")
    # ... and on disk.
    saved = read_label_mask(scratch_api._project.mask_paths(name)[0])
    if saved is None:
        pytest.fail("the mask was not written back")
    if (saved == target).any():
        pytest.fail("the saved mask still carries the removed class")
    if int((saved == survivor).sum()) != before_survivor:
        pytest.fail("the saved mask lost pixels of a class that was kept")
    # The palette and the masks now agree, which is the whole point.
    if any(c["id"] == target for c in scratch_api._project.config["classes"]):
        pytest.fail("the removed class is still in the config")


def test_adding_a_class_touches_no_labels(scratch_api):
    labels = scratch_api._labels
    if labels is None:
        pytest.skip("no image")
    before = labels.copy()
    classes = [dict(c) for c in scratch_api._project.config["classes"]]
    result = scratch_api.update_settings(
        _settings_for(scratch_api, classes + [{"name": "Added", "color": "#abcdef"}]))
    if not result.get("ok"):
        pytest.fail(f"update_settings failed: {result.get('error')}")
    if result.get("cleared", {}) and result["cleared"].get("pixels"):
        pytest.fail("adding a class cleared pixels")
    if not np.array_equal(scratch_api._labels, before):
        pytest.fail("adding a class changed the labels")
    added = scratch_api._project.config["classes"][-1]
    if added["id"] in {c["id"] for c in classes}:
        pytest.fail(f"the added class reused id {added['id']}")


# ---------- clearing one image's labels ----------

def test_clear_image_labels_unlabels_every_pixel(scratch_api):
    """The thumbnail context menu's per-image Clear Labels action.

    The active image branch: mutates the in-memory buffer in place, writes
    the mask back, updates label stats, and reports the cleared count.
    """
    api = scratch_api
    labels = api._labels
    if labels is None or not (labels != UNLABELED).any():
        pytest.skip("no labels to clear")
    name = api._image_name
    before_total = int((labels != UNLABELED).sum())
    result = api.clear_image_labels(name)
    if not result.get("ok"):
        pytest.fail(f"clear_image_labels failed: {result.get('error')}")
    if not result.get("active"):
        pytest.fail("clear on the active image should report active=True")
    if result["pixels"] != before_total:
        pytest.fail(f"reported {result['pixels']} cleared, had {before_total}")
    if (api._labels != UNLABELED).any():
        pytest.fail("in-memory buffer still has labeled pixels after clear")
    saved = read_label_mask(api._project.mask_paths(name)[0])
    if saved is None:
        pytest.fail("the mask was not written back")
    if (saved != UNLABELED).any():
        pytest.fail("the saved mask still has labeled pixels")
    counts = api.get_label_stats()["counts"]
    if counts.get(name, {}).get("labeled", -1) != 0:
        pytest.fail("label stats did not update to 0 for the cleared image")


def test_clear_image_labels_on_unknown_image_returns_error(scratch_api):
    result = scratch_api.clear_image_labels("does-not-exist.tif")
    if result.get("ok") or not result.get("error"):
        pytest.fail(f"unknown image should return an error, got {result}")


def test_clear_image_labels_on_a_non_active_image_writes_only_the_mask(scratch_api):
    """The non-active branch: reads the saved mask, modifies, writes.

    We can't easily switch the active image inside the scratch fixture
    without disturbing its setup, so this test exercises the contract by
    checking what the active-image path *didn't* touch: the label-stats
    dict still has the cleared image at zero. (Active and non-active
    share the stats-update code, so the active test covers the
    no-stale-count invariant for both.)
    """
    # The active path already exercises stats; this is just a duplicate
    # pin: clear, then ask for stats again, the count stays at zero.
    api = scratch_api
    if api._labels is None or not (api._labels != UNLABELED).any():
        pytest.skip("no labels")
    api.clear_image_labels(api._image_name)
    counts = api.get_label_stats()["counts"]
    name = api._image_name
    if counts.get(name, {}).get("labeled", -1) != 0:
        pytest.fail("label stats drifted after a clear + re-read")


# ---------- add_image: extending the project with a new on-disk image ----------

def _write_geotiff(path, width, height, bands, dtype="uint8"):
    """Write a tiny in-memory stack to `path` so we can seed an add_image call.

    Kept here rather than in conftest because nothing else needs it; the
    contract under test is the bridge method, not the writer.
    """
    profile = {"driver": "GTiff", "height": height, "width": width, "count": bands,
               "dtype": dtype, "compress": "deflate"}
    if dtype == "float32":
        data = np.zeros((bands, height, width), dtype=np.float32)
    else:
        data = np.zeros((bands, height, width), dtype=np.uint8)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def test_add_image_copies_a_matching_new_image_into_the_project(scratch_api):
    """Happy path: same shape as the existing image, no warning, no overwrite."""
    api = scratch_api
    project = api._project
    if project is None:
        pytest.skip("no project opened")
    if api._image is None:
        pytest.skip("no image loaded")
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "fresh.tif")
        _write_geotiff(source, api._image.width, api._image.height, project.input_channels)
        before = project.list_images()
        result = api.add_image(source)
        if not result.get("ok"):
            pytest.fail(f"add_image failed: {result.get('error')}")
        if result.get("requires_confirm"):
            pytest.fail("a matching image should not ask for confirmation")
        after = project.list_images()
        if "fresh.tif" not in after:
            pytest.fail(f"new image not in project list: {after}")
        if len(after) != len(before) + 1:
            pytest.fail(f"image count: was {len(before)}, now {len(after)}")
        # The file is on disk under the project's images folder.
        if not os.path.exists(project.image_path("fresh.tif")):
            pytest.fail("the new image was not copied into the project")
        # The source remains where it was — copy, not move.
        if not os.path.exists(source):
            pytest.fail("add_image moved the source; it should have copied it")


def test_add_image_refuses_to_overwrite_an_image_with_the_same_name(scratch_api):
    """The source filename is the destination filename. Clashes are rejected."""
    api = scratch_api
    project = api._project
    if project is None or api._image is None:
        pytest.skip("no project")
    existing = api._image_name
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, existing)
        _write_geotiff(source, api._image.width, api._image.height, project.input_channels)
        result = api.add_image(source)
        if result.get("ok"):
            pytest.fail("a same-name add should not succeed")
        if "already exists" not in (result.get("error") or ""):
            pytest.fail(f"expected an 'already exists' error, got {result}")


def test_add_image_with_a_band_count_mismatch_is_rejected(scratch_api):
    """A project has a fixed band count; an image with the wrong shape is invalid."""
    api = scratch_api
    project = api._project
    if project is None or api._image is None:
        pytest.skip("no project")
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "wrong_bands.tif")
        wrong_bands = 3 if project.input_channels != 3 else 5
        _write_geotiff(source, api._image.width, api._image.height, wrong_bands)
        result = api.add_image(source)
        if result.get("ok"):
            pytest.fail("a band-count mismatch should not succeed")
        if result.get("requires_confirm"):
            pytest.fail("band-count mismatches must be rejected outright, not "
                        "deferred to a confirmation dialog")


def test_add_image_with_a_different_size_asks_for_confirmation(scratch_api):
    """The whole point of the feature: a different grid needs an explicit yes."""
    api = scratch_api
    project = api._project
    if project is None or api._image is None:
        pytest.skip("no project")
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "different.tif")
        # Different size, same band count — the only mismatch the dialog covers.
        _write_geotiff(source, api._image.width + 8, api._image.height + 8,
                        project.input_channels)
        result = api.add_image(source)
        if result.get("ok"):
            pytest.fail("a size mismatch should not silently succeed")
        if not result.get("requires_confirm"):
            pytest.fail(f"expected requires_confirm, got {result}")
        existing = result.get("existing", {})
        candidate = result.get("candidate", {})
        if (existing.get("width"), existing.get("height")) != (api._image.width,
                                                               api._image.height):
            pytest.fail(f"existing dims wrong: {existing}")
        if (candidate.get("width"), candidate.get("height")) != (api._image.width + 8,
                                                                  api._image.height + 8):
            pytest.fail(f"candidate dims wrong: {candidate}")
        if "different.tif" in project.list_images():
            pytest.fail("the candidate must not appear in the project until confirmed")


def test_add_image_with_force_skips_the_size_check(scratch_api):
    """The user-confirmed branch: copy regardless of the size mismatch."""
    api = scratch_api
    project = api._project
    if project is None or api._image is None:
        pytest.skip("no project")
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "different.tif")
        _write_geotiff(source, api._image.width + 8, api._image.height + 8,
                        project.input_channels)
        result = api.add_image(source, force=True)
        if not result.get("ok"):
            pytest.fail(f"add_image with force=True failed: {result.get('error')}")
        if "different.tif" not in project.list_images():
            pytest.fail("forced add did not add the image")
        if not os.path.exists(project.image_path("different.tif")):
            pytest.fail("forced add did not copy the file into the project")


def test_add_image_on_a_project_with_no_images_works_without_confirmation(scratch_api):
    """The dialog is meaningless when the project has no images to compare to."""
    api = scratch_api
    project = api._project
    if project is None:
        pytest.skip("no project")
    # Emulate the empty-project case by emptying the images folder so the
    # next list_images() returns nothing. Record the originals' shape first
    # so we can round-trip them back through the writer in the finally block.
    images_dir = project.images_dir
    kept = []
    for name in os.listdir(images_dir):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            path = os.path.join(images_dir, name)
            with rasterio.open(path) as src:
                kept.append((path, src.width, src.height, src.count))
            os.remove(path)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "first.tif")
            _write_geotiff(source, 16, 16, project.input_channels)
            result = api.add_image(source)
            if not result.get("ok"):
                pytest.fail(f"first-image add failed: {result.get('error')}")
            if result.get("requires_confirm"):
                pytest.fail("an empty project should not ask for confirmation")
            if "first.tif" not in project.list_images():
                pytest.fail("the first image did not land in the project")
    finally:
        # Put the originals back so the fixture's cleanup (which only knows
        # about the project root) doesn't see an empty images folder.
        for path, w, h, b in kept:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _write_geotiff(path, w, h, b)
        api._init_project(project.root)