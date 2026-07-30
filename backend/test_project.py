"""Invariants of the class palette.

The palette is the label space: its ids are what every saved mask stores, so a
mistake here is not a display bug, it is ground truth quietly changing meaning.
These cover the two ways that can happen -- an id moving to a different
material, and a mask keeping pixels of a class the config no longer has.

Everything that touches labels runs against a scratch copy of the project via
the `scratch_api` fixture in backend/conftest.py, never label_raster/.
"""

import numpy as np
import pytest

from data import UNLABELED, read_label_mask
from project import (CLASS_PRESETS, DEFAULT_PRESET, classes_from_settings,
                     config_from_settings, default_config, preset_classes,
                     validate_config)

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