"""The little bit of machinery every test file here shares.

These are plain scripts rather than a framework: `python3 backend/test_x.py`
runs them, and there is nothing to install. That is worth keeping, but it had
grown five copies of the same runner loop and five copies of `fail`, so a fix
to one of them reached one file.

A test file needs:

    from testlib import fail, main            # or run_tests
    ...
    if __name__ == "__main__":
        sys.exit(main(globals()))
"""

import shutil


def fail(message):
    """Fail the current check. A plain assertion carries no message worth reading."""
    raise AssertionError(message)


def scratch_api(prefer=None):
    """(Api on a throwaway copy of the project, its path), or (None, None).

    Every check that touches labels goes through one of these rather than the
    real project: the app edits masks in place, so a test writing to
    label_raster/ would mutate the ground truth it exists to protect and race
    the running app's autosave besides.
    """
    from api import Api
    from test_no_torch import _scratch_project
    tmp, _ = _scratch_project(prefer)
    if tmp is None:
        return None, None
    return Api(project_root=tmp), tmp


def with_scratch_api(check, prefer=None):
    """Give `check` an Api on a scratch project, and clean up after it."""
    api, tmp = scratch_api(prefer)
    if api is None:
        print("     (no project to copy — skipping)")
        return
    try:
        check(api)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(namespace):
    """Run every test_* in `namespace`, in name order. Returns an exit code.

    Sorted rather than definition order so a run is reproducible, and each test
    is isolated to its own try block so one failure does not hide the rest.
    """
    tests = [v for k, v in sorted(namespace.items())
             if k.startswith("test_") and callable(v)]
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
