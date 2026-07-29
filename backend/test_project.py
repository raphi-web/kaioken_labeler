"""Invariants of the class palette. Run: python3 backend/test_project.py

The palette is the label space: its ids are what every saved mask stores, so a
mistake here is not a display bug, it is ground truth quietly changing meaning.
These cover the two ways that can happen -- an id moving to a different
material, and a mask keeping pixels of a class the config no longer has.

Everything runs against a scratch copy of the project, never label_raster/ --
see test_no_torch._scratch_project for why.
"""

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import UNLABELED, read_label_mask  # noqa: E402
from project import (CLASS_PRESETS, DEFAULT_PRESET, classes_from_settings,  # noqa: E402
                     config_from_settings, default_config, preset_classes,
                     validate_config)
from test_no_torch import _scratch_project  # noqa: E402

BASE = {"input_channels": 4, "input_patch_size": 96}


def _fail(message):
    raise AssertionError(message)


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
        _fail(f"the class list did not survive: {got}")


def test_a_project_without_a_class_list_gets_the_preset():
    config = config_from_settings(_settings())
    names = [c["name"] for c in config["classes"]]
    if names != [c["name"] for c in CLASS_PRESETS[DEFAULT_PRESET]]:
        _fail(f"expected the {DEFAULT_PRESET} preset, got {names}")
    if len(preset_classes("binary")) != 2:
        _fail("the binary preset should still be two classes")


def test_every_preset_builds_a_valid_config():
    for name in CLASS_PRESETS:
        config = config_from_settings(_settings(class_preset=name))
        validate_config(config)  # raises if the schema is broken
        if len(config["classes"]) != len(CLASS_PRESETS[name]):
            _fail(f"preset {name} lost classes")


def test_a_nameless_or_empty_class_list_is_refused():
    for classes, why in (([], "an empty list"),
                         ([{"name": "  ", "color": "#112233"}], "a blank name"),
                         (["nope"], "a non-object row")):
        try:
            config_from_settings(_settings(classes=classes))
        except ValueError:
            continue
        _fail(f"{why} should have been refused")


# ---------- ids ----------

def test_an_edited_class_keeps_its_id():
    """Renaming or recolouring must not move a class in the label space."""
    existing = preset_classes("sentinel2")
    edited = [dict(c, name=f"{c['name']} v2", color="#010203") for c in existing]
    classes, _ = classes_from_settings(edited, existing, len(existing))
    for before, after in zip(existing, classes):
        if before["id"] != after["id"]:
            _fail(f"{before['name']} moved from id {before['id']} to {after['id']}")


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
        _fail(f"a new class took id {fresh['id']}, which class "
              f"'{[c['name'] for c in existing if c['id'] == fresh['id']][0]}' had")
    # And again, to be sure the counter persists rather than tracking max().
    classes2, _ = classes_from_settings(
        [dict(classes[0]), {"name": "Fresher", "color": "#654321"}],
        classes, next_id)
    if classes2[-1]["id"] == fresh["id"]:
        _fail(f"id {fresh['id']} was handed out twice")


def test_the_id_counter_is_stored_and_validated():
    config = default_config("t", input_channels=4)
    if config["next_class_id"] <= max(c["id"] for c in config["classes"]):
        _fail("next_class_id must sit above every id in use")
    config["next_class_id"] = 0
    try:
        validate_config(config)
    except ValueError:
        return
    _fail("a next_class_id below the ids in use should be refused")


# ---------- deleting a class clears its pixels ----------

def _run(check):
    api, tmp = None, None
    try:
        from api import Api
        tmp, _ = _scratch_project(None)
        if tmp is None:
            print("     (no project to copy — skipping)")
            return
        api = Api(project_root=tmp)
        check(api, tmp)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


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


def test_class_usage_counts_what_a_deletion_would_destroy():
    def check(api, _tmp):
        labels = api._labels
        if labels is None:
            print("     (no labels — skipping)")
            return
        present = [int(v) for v in np.unique(labels) if v != UNLABELED]
        if not present:
            print("     (nothing labeled — skipping)")
            return
        target = present[0]
        result = api.class_usage([target])
        if not result.get("ok"):
            _fail(f"class_usage failed: {result.get('error')}")
        counted = result["usage"][str(target)]["pixels"]
        expected = int((labels == target).sum())
        if counted < expected:
            _fail(f"class_usage counted {counted} px, but the open image alone "
                  f"has {expected}")
        if api.class_usage([200])["usage"]["200"]["pixels"]:
            _fail("an unused class id was reported as having pixels")
    _run(check)


def test_removing_a_class_unlabels_exactly_its_pixels():
    def check(api, _tmp):
        labels = api._labels
        if labels is None or not (labels != UNLABELED).any():
            print("     (nothing labeled — skipping)")
            return
        present = [int(v) for v in np.unique(labels) if v != UNLABELED]
        if len(present) < 2:
            print("     (need two labeled classes — skipping)")
            return
        target, survivor = present[0], present[1]
        name = api._image_name
        before_survivor = int((labels == survivor).sum())
        keep = [dict(c) for c in api._project.config["classes"]
                if c["id"] != target]

        result = api.update_settings(_settings_for(api, keep))
        if not result.get("ok"):
            _fail(f"update_settings failed: {result.get('error')}")
        if not result.get("cleared", {}).get("pixels"):
            _fail("removing a labeled class reported nothing cleared")

        # In memory ...
        if (api._labels == target).any():
            _fail("the open image still carries the removed class")
        if int((api._labels == survivor).sum()) != before_survivor:
            _fail("removing one class changed another class's pixels")
        # ... and on disk.
        saved = read_label_mask(api._project.mask_paths(name)[0])
        if saved is None:
            _fail("the mask was not written back")
        if (saved == target).any():
            _fail("the saved mask still carries the removed class")
        if int((saved == survivor).sum()) != before_survivor:
            _fail("the saved mask lost pixels of a class that was kept")
        # The palette and the masks now agree, which is the whole point.
        if any(c["id"] == target for c in api._project.config["classes"]):
            _fail("the removed class is still in the config")
    _run(check)


def test_adding_a_class_touches_no_labels():
    def check(api, _tmp):
        labels = api._labels
        if labels is None:
            print("     (no image — skipping)")
            return
        before = labels.copy()
        classes = [dict(c) for c in api._project.config["classes"]]
        result = api.update_settings(
            _settings_for(api, classes + [{"name": "Added", "color": "#abcdef"}]))
        if not result.get("ok"):
            _fail(f"update_settings failed: {result.get('error')}")
        if result.get("cleared", {}) and result["cleared"].get("pixels"):
            _fail("adding a class cleared pixels")
        if not np.array_equal(api._labels, before):
            _fail("adding a class changed the labels")
        added = api._project.config["classes"][-1]
        if added["id"] in {c["id"] for c in classes}:
            _fail(f"the added class reused id {added['id']}")
    _run(check)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {test.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - surface the real error
            failures += 1
            print(f"ERROR {test.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
