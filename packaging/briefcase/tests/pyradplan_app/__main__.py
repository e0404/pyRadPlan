"""Smoke test executed inside the built bundle via ``briefcase run --test``.

Briefcase starts this module instead of the app and reads the result from the
``>>>>>>>>>> EXIT <code> <<<<<<<<<<`` line, so no test framework is needed in
the bundle. The test exercises the pieces most likely to break when pyRadPlan
runs from an installed, read-only bundle: importing the package, loading the
bundled TG119 phantom, a small photon dose calculation, a short optimization and
constructing the main window offscreen.
"""

# Imports belong inside the checks so failures reach main() and Briefcase's exit marker.
# ruff: noqa: PLC0415

import os
import sys
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Briefcase streams the output through its own console; the Unicode progress
# bars are noise there and crash it on a cp1252 Windows console.
os.environ.setdefault("TQDM_DISABLE", "1")


def _check_import():
    import pyRadPlan

    print(f"pyRadPlan {pyRadPlan.__version__} from {pyRadPlan.__file__}")
    from pyRadPlan.core import xp_utils

    xp = xp_utils.choose_array_api_namespace()
    assert "numpy" in xp.__name__, xp.__name__
    print(f"array backend: {xp.__name__}")


def _check_dose_and_optimization():
    import numpy as np
    from pyRadPlan import (
        PhotonPlan,
        calc_dose_influence,
        fluence_optimization,
        generate_stf,
        load_tg119,
    )

    ct, cst = load_tg119()
    # Coarse grid and a single beam keep this well under a minute on a laptop CPU.
    pln = PhotonPlan(
        gantry_angles=[0.0],
        couch_angles=[0.0],
        prop_dose_calc={"dose_grid": {"resolution": {"x": 8.0, "y": 8.0, "z": 8.0}}},
        prop_stf={"bixel_width": 10.0},
        prop_opt={"solver": "scipy", "max_iter": 5, "display": False},
    )
    stf = generate_stf(ct, cst, pln)
    dij = calc_dose_influence(ct, cst, stf, pln)
    assert dij.physical_dose is not None
    print(f"dij: {dij.num_of_voxels} voxels x {dij.total_num_of_bixels} bixels")

    opt_info = {}
    weights = fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info)
    dose = np.asarray(dij.compute_result_ct_grid(weights)["physical_dose"])
    assert np.isfinite(dose).all() and dose.max() > 0
    print(f"optimized {opt_info.get('num_iter')} iterations: max dose {dose.max():.3g} Gy")


def _check_main_window():
    from PySide6.QtWidgets import QApplication

    from pyRadPlan.gui.windows._main_win import MainWindow
    from pyRadPlan.gui.workspace import WorkspaceManager

    app = QApplication.instance() or QApplication(sys.argv)
    workspace = WorkspaceManager.instance()
    workspace.clear()
    win = MainWindow(workspace)
    win.show()
    app.processEvents()
    assert win.isVisible()

    phantoms = {a.text(): a for a in win._file_menu._phantom_menu.actions()}
    assert "TG119" in phantoms, list(phantoms)
    phantoms["TG119"].trigger()
    deadline = time.monotonic() + 120
    while not workspace.has("ct", "cst") and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert workspace.has("ct", "cst"), "TG119 did not load from the File menu"

    win.close()
    app.processEvents()
    print("main window built offscreen, TG119 opened from the File menu")


def main() -> int:
    """Run each bundled check and return a failure status if any check raises."""
    checks = [_check_import, _check_dose_and_optimization, _check_main_window]
    for check in checks:
        name = check.__name__.lstrip("_")
        print(f"--- {name}", flush=True)
        try:
            check()
        except Exception:
            traceback.print_exc()
            print(f"FAILED: {name}", flush=True)
            return 1
        print(f"ok: {name}", flush=True)
    return 0


if __name__ == "__main__":
    returncode = main()
    print(f">>>>>>>>>> EXIT {returncode} <<<<<<<<<<", flush=True)
    sys.exit(returncode)
