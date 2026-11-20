import numpy as np
import pytest
import SimpleITK as sitk
from pyRadPlan.result import PlanResult, create_result

from pyRadPlan.core import ProgressLevel, ProgressReport, StatusReport
from pyRadPlan.gui.widgets.workflow import WorkflowWidget
from pyRadPlan.gui.workspace import WorkspaceManager

# Widget methods are patched on the class: undoing a patch on an instance leaves
# the bound method behind as an attribute, a reference cycle whose garbage
# collection segfaults PySide (Linux, Python 3.14).


def test_workflow_widget_init(qapp):
    w = WorkflowWidget(WorkspaceManager())
    assert w is not None
    assert not w._progress.isVisibleTo(w)  # busy bar hidden until a run starts


def test_busy_toggles_progress_and_buttons(qapp):
    w = WorkflowWidget(WorkspaceManager())
    events = []
    w.busy_changed.connect(events.append)

    w._set_busy(True)
    assert events == [True]
    assert w._progress.isVisibleTo(w)
    assert all(not btn.isEnabled() for btn in w._action_buttons)

    w._set_busy(False)
    assert events == [True, False]
    assert not w._progress.isVisibleTo(w)


def test_busy_text_shown_in_status(qapp):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    # A no-op long task: capture that busy_text lands in the status label.
    def _noop():
        return None

    w._run_in_thread(_noop, busy_text="Working…")
    assert w._lbl_status.text() == "Working…"
    # Let the worker thread finish and clean up.
    if w._thread is not None:
        w._thread.wait(2000)
    qapp.processEvents()


def test_progress_report_drives_combined_nested_bar(qapp):
    w = WorkflowWidget(WorkspaceManager())

    report = ProgressReport(
        levels=(
            ProgressLevel("Beam", 0, 2),
            ProgressLevel("Ray", 50, 100),
        )
    )
    w._on_compute_report(report)

    # Bar shows combined nested progress: 0/2 + (1/2)*(50/100) = 0.25.
    assert w._progress.maximum() == w._PROGRESS_STEPS
    assert w._progress.value() == round(0.25 * w._PROGRESS_STEPS)
    assert w._lbl_status.text() == "Beam 0/2 · Ray 50/100"


def test_nested_fraction_advances_with_outer_level(qapp):
    w = WorkflowWidget(WorkspaceManager())
    levels = (ProgressLevel("Beam", 1, 2), ProgressLevel("Ray", 50, 100))
    assert w._nested_fraction(levels) == pytest.approx(0.75)
    # Outermost indeterminate -> None (bar pulses).
    assert w._nested_fraction((ProgressLevel("Setup", 0, None),)) is None


def test_status_report_drives_busy_bar_and_status(qapp):
    w = WorkflowWidget(WorkspaceManager())
    w._progress.setRange(0, 10)
    w._progress.setValue(3)

    w._on_compute_report(StatusReport(message="iter 1", data={"iteration": 1, "objective": 1.2}))

    # An optimization status report drives the indeterminate (pulsing) busy bar...
    assert w._progress.minimum() == 0 and w._progress.maximum() == 0
    # ...and surfaces the per-iteration summary in the status line.
    assert "Optimizing" in w._lbl_status.text()
    assert "iter 1" in w._lbl_status.text()


def test_indeterminate_progress_sets_busy_range(qapp):
    w = WorkflowWidget(WorkspaceManager())
    w._on_compute_report(ProgressReport(levels=(ProgressLevel("Setup", 0, None),)))
    assert w._progress.minimum() == 0 and w._progress.maximum() == 0  # pulsing


def test_plan_change_marks_downstream_stale(qapp):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    def _result():
        return create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})

    # Loading everything at once leaves the products current, not stale.
    ws.set_many(
        ct=object(), cst=object(), pln=object(), stf=object(), dij=object(), result=_result()
    )
    assert not w._dij_stale and not w._result_stale

    # Changing the plan invalidates the dose influence and the result.
    ws.pln = object()
    assert w._dij_stale and w._result_stale
    assert w._indicators["dij"].toolTip().startswith("Outdated")

    # Recomputing the dose influence clears its flag (and the result stays stale).
    ws.set_many(stf=object(), dij=object())
    assert not w._dij_stale and w._result_stale

    # Re-optimizing clears the result flag.
    ws.result = _result()
    assert not w._result_stale

    # Moved beams (stf without dij) outdate the dose influence as well.
    ws.set_many(stf=object(), dij=object())
    ws.stf = object()
    assert w._dij_stale and w._result_stale


def test_tissue_parameter_change_marks_dij_stale(qapp):
    """Tissue parameters feed the biological model, so the dose influence goes stale."""
    from pyRadPlan.cst import create_voi
    from pyRadPlan.io import load_tg119
    from pyRadPlan.biology.tissues import MKMTissueProperties

    ct, cst = load_tg119()
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    ws.set_many(ct=ct, cst=cst, pln=object(), stf=object(), dij=object(), result=_result())
    assert not w._dij_stale and not w._result_stale

    # Other structure set edits leave the dose influence valid, only the result stale
    cst.vois[0].objectives = []
    ws.cst = cst
    assert not w._dij_stale
    assert w._result_stale

    # Changing a structure's tissue properties invalidates the dose influence as well
    voi = cst.vois[0]
    tissue = MKMTissueProperties(alpha_x=0.2, beta_x=0.04, r_nucleus=3.9, r_domain=0.32)
    data = dict(voi)
    data.update(tissue_properties=tissue, alpha_x=tissue.alpha_x, beta_x=tissue.beta_x)
    cst.vois = [create_voi(data), *cst.vois[1:]]
    ws.cst = cst

    assert w._dij_stale
    assert w._indicators["dij"].toolTip().startswith("Outdated")
    assert "recalculate" in w._lbl_status.text().lower()


def test_cst_export_options_resolve_against_registry(qapp):
    """Every offered CST export format has a registered exporter.

    Guards the GUI's "Save CST" against a stale format key (e.g. the MetaImage
    exporter registers as ``"meta"``, not ``"metaimage"``).
    """
    from pyRadPlan.io import get_available_formats

    available = get_available_formats()
    options = WorkflowWidget._cst_export_options()
    keys = [fmt for _label, fmt, *_ in options]

    assert all(fmt in available for fmt in keys), keys
    # The container formats that preserve objectives, plus the image and both
    # DICOM structure representations, are all offered.
    assert {"mat", "pickle", "npz", "nifti", "nrrd", "meta", "dcm"} <= set(keys)
    dicom_structs = {struct for _l, fmt, struct, *_ in options if fmt == "dcm"}
    assert dicom_structs == {"rtstruct", "seg"}
    # Only mat/pickle advertise objective preservation.
    keeps = {fmt for _l, fmt, _s, _d, keep in options if keep}
    assert keeps == {"mat", "pickle"}


def test_cst_export_all_formats_write_loadable_masks(qapp, tmp_path):
    """The backends behind "Save CST" write every format and reload the masks."""
    import os

    from pyRadPlan.io import (
        load_tg119,
        save_data,
        MatlabHandler,
        PickleHandler,
        NpzHandler,
        NiftiHandler,
        NrrdHandler,
        MetaImageHandler,
        DicomHandler,
    )
    from pyRadPlan.io.dicom import DicomExporter

    ct, cst = load_tg119()
    n = len(cst.vois)

    # Container single-file formats (cst-only; mat needs the ct to reconstruct masks).
    for fmt, ext, handler in [
        ("mat", ".mat", MatlabHandler),
        ("pickle", ".pkl", PickleHandler),
        ("npz", ".npz", NpzHandler),
    ]:
        path = str(tmp_path / f"cst{ext}")
        save_data(file_name=path, format=fmt, cst=cst)
        assert os.path.exists(path)
        assert len(handler(path).load_cst(ct).vois) == n

    # Directory label-map formats.
    for fmt, handler in [
        ("nifti", NiftiHandler),
        ("nrrd", NrrdHandler),
        ("meta", MetaImageHandler),
    ]:
        folder = str(tmp_path / fmt)
        save_data(file_name=folder, format=fmt, cst=cst)
        assert len(handler(folder).load_cst().vois) == n

    # DICOM RTSTRUCT and SEG.
    rt = str(tmp_path / "dcm_rt")
    save_data(file_name=rt, format="dcm", cst=cst)
    assert len(DicomHandler(rt).load_cst().vois) == n

    seg = str(tmp_path / "dcm_seg")
    DicomExporter(seg, structure_format="seg").save(cst=cst)
    assert len(DicomHandler(seg).load_cst().vois) == n


def _result():
    return create_result({"physical_dose": _dose_image()})


def _dose_image(value=1.0):
    image = sitk.Image([4, 4, 2], sitk.sitkFloat32)
    image += value
    return image


def test_workspace_validates_a_result_dict(qapp):
    """A dict assigned to ws.result becomes a PlanResult for every consumer."""
    ws = WorkspaceManager()

    ws.result = {"physical_dose": _dose_image()}
    assert isinstance(ws.result, PlanResult)
    assert "physical_dose" in ws.result

    ws.set_many(result={"physical_dose": _dose_image()})
    assert isinstance(ws.result, PlanResult)

    ws.result = None
    assert ws.result is None


def test_snapshot_is_a_result_of_its_own(qapp, monkeypatch):
    """Snapshot Result keeps a copy of the active result under the given name."""
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    ws.result = create_result(
        {"physical_dose": _dose_image(), "physical_dose_beam": [_dose_image()]},
        fluence=np.ones(2),
    )
    original = ws.result

    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QInputDialog.getText",
        lambda *a, **k: ("run 1", True),
    )
    w._on_save_result()

    assert ws.result is original
    assert list(ws.results) == ["current", "run_1"]
    snapshot = ws.results["run_1"]
    assert snapshot is not original
    assert snapshot["physical_dose"] is original["physical_dose"]
    assert snapshot.stored_beam_quantities == ["physical_dose"]
    assert snapshot.fluence is original.fluence

    # The snapshot survives a re-optimization untouched.
    ws.result = create_result({"physical_dose": _dose_image(2.0)})
    assert ws.results["run_1"] is snapshot
    assert ws.result is not original


def test_imported_dose_is_a_result_on_its_own_grid(qapp, monkeypatch, tmp_path):
    """An imported dose keeps its grid and does not join the active result."""
    from types import SimpleNamespace

    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    ws.set_many(
        ct=SimpleNamespace(cube_hu=sitk.Image([4, 4, 2], sitk.sitkFloat32)),
        result={"physical_dose": _dose_image()},
    )
    active = ws.result
    measured = sitk.Image([8, 8, 4], sitk.sitkFloat32)
    measured.SetSpacing((0.5, 0.5, 0.5))
    path = tmp_path / "measured.nrrd"
    sitk.WriteImage(measured, str(path))

    from PySide6.QtWidgets import QDialog

    dialog = SimpleNamespace(
        exec=lambda: QDialog.DialogCode.Accepted,
        selection=lambda: {"mode": "dose", "name": "film scan"},
    )
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._image_import_dialog.ImageImportDialog",
        lambda *a, **k: dialog,
    )
    monkeypatch.setattr(
        WorkflowWidget,
        "_run_in_thread",
        staticmethod(lambda fn, on_success, **kw: on_success(fn())),
    )
    w._open_image_import_dialog(str(path))
    w._open_image_import_dialog(str(path))

    assert ws.result is active
    assert list(ws.results) == ["current", "film_scan", "film_scan_2"]
    imported = ws.results["film_scan"]["physical_dose"].values
    assert imported.GetSize() == (8, 8, 4)
    assert imported.GetSpacing() == (0.5, 0.5, 0.5)
    # Importing does not mark an outdated result as up to date
    assert not w._result_stale
    ws.pln = object()
    assert w._result_stale
    w._open_image_import_dialog(str(path))
    assert w._result_stale


def test_session_of_version_0_9_is_split_into_results(qapp):
    """Snapshots and imported doses of older sessions were mangled into one result."""
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    dose, snapshot, imported = _dose_image(1.0), _dose_image(2.0), _dose_image(3.0)

    w._merge_loaded_data(
        {
            "result": {
                "physical_dose": dose,
                "physical_dose_beam": [dose],
                "w": np.ones(2),
                "physical_dose_run1": snapshot,
                "physical_dose_beam_run1": [snapshot],
                "import_measured": imported,
            },
            "saved_result_tags": ["run1"],
        }
    )

    assert list(ws.results) == ["current", "run1", "measured"]
    assert ws.result.stored_quantities == ["physical_dose"]
    assert ws.result.stored_beam_quantities == ["physical_dose"]
    np.testing.assert_array_equal(ws.result.fluence, np.ones(2))
    assert ws.results["run1"].stored_quantities == ["physical_dose"]
    assert ws.results["run1"].stored_beam_quantities == ["physical_dose"]
    assert ws.results["run1"]["physical_dose"].values is snapshot
    assert ws.results["measured"]["physical_dose"].values is imported


@pytest.mark.parametrize("same_grid", [False, True])
def test_new_patient_clears_missing_objects_atomically(qapp, same_grid):
    import SimpleITK as sitk
    from pyRadPlan.ct import validate_ct

    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    old_ct = validate_ct(cube_hu=sitk.Image([5, 5, 5], sitk.sitkFloat32))
    new_ct = validate_ct(cube_hu=sitk.Image([5 if same_grid else 7, 5, 5], sitk.sitkFloat32))
    ws.set_many(
        ct=old_ct, cst=object(), pln=object(), stf=object(), dij=object(), result=_result()
    )
    ws.add_result("old", _result())
    states = []
    ws.workspace_changed.connect(lambda keys: states.append({k: getattr(ws, k) for k in ws.keys}))

    w._merge_loaded_data({"ct": new_ct})

    assert len(states) == 1
    assert states[0]["ct"] is new_ct
    assert all(states[0][key] is None for key in ws.keys if key != "ct")
    assert not w._btn_optimize.isEnabled()
    assert ws.results == {}


def test_results_loaded_without_ct_join_the_workspace(qapp):
    """Only a new patient replaces the results; a loaded plan result the active one."""
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    ws.set_many(ct=object(), result=_result())
    ws.add_result("saved", _result())
    ws.add_result("imported", _result())
    kept = ws.results
    events = []
    ws.workspace_changed.connect(events.append)

    w._merge_loaded_data({"dose": _dose_image(2.0)})
    assert list(ws.results) == ["current", "saved", "imported", "imported_2"]
    assert all(ws.results[name] is kept[name] for name in kept)
    assert events == [["result"]]

    w._merge_loaded_data({"result": {"physical_dose": _dose_image(3.0)}})
    assert list(ws.results) == ["current", "saved", "imported", "imported_2"]
    assert ws.result is not kept["current"]
    assert ws.results["saved"] is kept["saved"]


def test_import_without_ct_keeps_current_patient(qapp):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    ct, cst, new_plan = object(), object(), object()
    ws.set_many(ct=ct, cst=cst)
    w._merge_loaded_data({"pln": new_plan})
    assert ws.ct is ct and ws.cst is cst and ws.pln is new_plan


def _recalc_setup(monkeypatch, prop_stf, beam_isos, forward):
    """A workflow widget with a result to recalculate, run synchronously."""
    import numpy as np
    from types import SimpleNamespace

    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    ws.set_many(
        ct=object(),
        cst=SimpleNamespace(target_center_of_mass=lambda: np.array([1.0, 2.0, 3.0])),
        pln=SimpleNamespace(prop_stf=prop_stf),
        stf=SimpleNamespace(
            beams=[SimpleNamespace(iso_center=np.asarray(iso, dtype=float)) for iso in beam_isos]
        ),
        dij=object(),
        result={"w": np.array([2.0, 3.0]), "physical_dose": _dose_image()},
    )
    ws.add_result("saved", {"physical_dose": _dose_image()})
    monkeypatch.setattr("pyRadPlan.calc_dose_forward", forward)
    monkeypatch.setattr(
        WorkflowWidget,
        "_run_in_thread",
        staticmethod(lambda fn, on_success, **kw: on_success(fn())),
    )
    return ws, w


@pytest.mark.parametrize(
    ("prop_stf", "isos"),
    [
        ({"iso_center": [4.0, 5.0, 6.0]}, [[4.0, 5.0, 6.0]] * 2),
        ({"iso_center": [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]}, [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        ({}, [[1.0, 2.0, 3.0]] * 2),
        ({"iso_center": []}, [[1.0, 2.0, 3.0]] * 2),
    ],
    ids=["plan-isocenter", "per-beam", "automatic", "empty"],
)
def test_recalculate_uses_forward_result_and_keeps_weights(qapp, monkeypatch, prop_stf, isos):
    import numpy as np
    from unittest.mock import Mock

    fresh = create_result({"physical_dose": _dose_image(2.0)})
    forward = Mock(return_value=fresh)
    ws, w = _recalc_setup(monkeypatch, prop_stf, [np.zeros(3)] * 2, forward)
    old_stf, weights = ws.stf, ws.result.fluence
    snapshot = ws.results["saved"]

    w._on_recalc_dose()

    # The beams moved on a copy, which is stored with the result and outdates the dij.
    forward.assert_called_once_with(ws.ct, ws.cst, ws.stf, ws.pln, weights)
    assert ws.stf is not old_stf
    assert np.allclose([beam.iso_center for beam in ws.stf.beams], isos)
    assert np.allclose([beam.iso_center for beam in old_stf.beams], 0.0)
    assert w._dij_stale and not w._result_stale
    assert ws.result["physical_dose"] is fresh["physical_dose"]
    assert ws.result.fluence is weights
    assert ws.results["saved"] is snapshot


def test_recalculate_keeps_the_stf_of_unmoved_beams(qapp, monkeypatch):
    from unittest.mock import Mock

    forward = Mock(return_value=_result())
    ws, w = _recalc_setup(monkeypatch, {"iso_center": [4.0, 5.0, 6.0]}, [[4.0, 5.0, 6.0]], forward)
    stf = ws.stf

    w._on_recalc_dose()

    forward.assert_called_once_with(ws.ct, ws.cst, stf, ws.pln, ws.result.fluence)
    assert ws.stf is stf
    assert not w._dij_stale
    assert w._lbl_status.text().endswith("1 further result")


def test_failed_recalculation_leaves_the_stf_untouched(qapp, monkeypatch):
    import numpy as np
    from unittest.mock import Mock

    forward = Mock(side_effect=RuntimeError("engine failed"))
    ws, w = _recalc_setup(monkeypatch, {"iso_center": [4.0, 5.0, 6.0]}, [np.zeros(3)], forward)
    stf, result = ws.stf, ws.result

    with pytest.raises(RuntimeError):
        w._on_recalc_dose()

    assert ws.stf is stf and ws.result is result
    assert np.allclose(stf.beams[0].iso_center, 0.0)


@pytest.mark.parametrize(
    "angles",
    [
        {"gantry_angles": [90, 180], "couch_angles": [0, 0]},
        {"gantry_angles": [0, 180], "couch_angles": [0, 45]},
        {"gantry_angles": [0], "couch_angles": [0]},
        {"gantry_angles": [0, 180, 270], "couch_angles": [0]},
        {"gantry_angles": []},
        {"gantry_angles": [float("nan"), 180]},
    ],
    ids=["gantry", "couch", "removed-beam", "added-beam", "empty", "nonfinite"],
)
def test_recalculate_rejects_changed_beam_geometry(qapp, monkeypatch, angles):
    from unittest.mock import Mock

    forward = Mock()
    ws, w = _recalc_setup(monkeypatch, {}, [[1, 2, 3]] * 2, forward)
    for beam, gantry in zip(ws.stf.beams, [0.0, 180.0]):
        beam.gantry_angle, beam.couch_angle = gantry, 0.0
    ws.pln.prop_stf.update(angles)
    ws.pln = ws.pln  # Applying a geometry edit marks the old result stale.
    original_stf, original_result = ws.stf, ws.result
    warning = Mock()
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QMessageBox.warning", warning
    )

    w._on_recalc_dose()

    forward.assert_not_called()
    warning.assert_called_once()
    assert "Calculate Dose Influence" in warning.call_args.args[2]
    assert ws.stf is original_stf and ws.result is original_result
    assert w._result_stale and w._dij_stale


@pytest.mark.parametrize("couch", [None, [], [0], [0, 360]])
def test_recalculate_accepts_equivalent_angles(qapp, monkeypatch, couch):
    from unittest.mock import Mock

    forward = Mock(return_value=_result())
    angles = {"gantry_angles": [360, -180], "couch_angles": couch}
    ws, w = _recalc_setup(monkeypatch, angles, [[1, 2, 3]] * 2, forward)
    for beam, gantry in zip(ws.stf.beams, [0.0, 180.0]):
        beam.gantry_angle, beam.couch_angle = gantry, 0.0
    original_stf = ws.stf

    w._on_recalc_dose()

    forward.assert_called_once_with(ws.ct, ws.cst, original_stf, ws.pln, ws.result.fluence)
    assert ws.stf is original_stf


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ""])
def test_save_workspace_preserves_all_results(qapp, tmp_path, monkeypatch, suffix):
    import numpy as np
    import SimpleITK as sitk
    from pyRadPlan.ct import validate_ct
    from pyRadPlan.plan import PhotonPlan
    from pyRadPlan.io import load_data

    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    ct = validate_ct(cube_hu=sitk.Image([5, 4, 3], sitk.sitkFloat32))
    ct.cube_hu.SetOrigin((3.0, 4.0, 5.0))
    dose = sitk.Image(ct.cube_hu) + 2.0
    weights = np.array([2.0, 3.0])
    ws.set_many(
        ct=ct,
        pln=PhotonPlan(prop_opt={"solver": "scipy"}),
        stf={"test": 1},
        dij={"test": 2},
        result={"physical_dose": dose, "w": weights},
    )
    ws.add_result("saved", ws.result.model_copy())
    ws.add_result("measured", {"physical_dose": dose * 2.0})
    path = tmp_path / ("workspace" + suffix)
    filters = []

    def choose_file(*args):
        filters.append(args[-1])
        return str(path), ""

    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QFileDialog.getSaveFileName", choose_file
    )
    monkeypatch.setattr(WorkflowWidget, "_run_save", staticmethod(lambda fn, *args: fn()))

    w._on_save_workspace()
    data = load_data(path if suffix else path.with_suffix(".pkl"))
    restored_ws = WorkspaceManager()
    restored = WorkflowWidget(restored_ws)
    restored._merge_loaded_data(data)

    assert "*.pkl" in filters[0] and "*.mat" not in filters[0] and "*.npz" not in filters[0]
    assert restored_ws.pln.prop_opt == {"solver": "scipy"}
    assert restored_ws.stf == ws.stf and restored_ws.dij == ws.dij
    assert restored_ws.result.stored_quantities == ws.result.stored_quantities
    np.testing.assert_array_equal(restored_ws.result.fluence, weights)
    restored_dose = restored_ws.result["physical_dose"].values
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(restored_dose), sitk.GetArrayFromImage(dose)
    )
    assert restored_dose.GetOrigin() == dose.GetOrigin()
    assert list(restored_ws.results) == ["current", "saved", "measured"]
    assert restored_ws.active_result == "current"
    # A snapshot still shares its images with the result it was taken from
    assert restored_ws.results["saved"]["physical_dose"].values is restored_dose


@pytest.mark.parametrize("suffix", [".mat", ".npz"])
def test_save_workspace_rejects_lossy_extension(qapp, tmp_path, monkeypatch, suffix):
    from unittest.mock import Mock

    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    ws.ct = object()
    path = tmp_path / ("workspace" + suffix)
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QFileDialog.getSaveFileName",
        lambda *args: (str(path), ""),
    )
    warning = Mock()
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QMessageBox.warning", warning
    )
    save = Mock()
    monkeypatch.setattr(WorkflowWidget, "_run_save", save)
    w._on_save_workspace()
    warning.assert_called_once()
    save.assert_not_called()
    assert not path.exists()


def test_save_workspace_without_ct_stops_before_file_dialog(qapp, monkeypatch):
    from unittest.mock import Mock

    w = WorkflowWidget(WorkspaceManager())
    dialog, save, warning = Mock(), Mock(), Mock()
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QFileDialog.getSaveFileName", dialog
    )
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QMessageBox.warning", warning
    )
    monkeypatch.setattr(WorkflowWidget, "_run_save", save)

    w._on_save_workspace()

    warning.assert_called_once()
    assert "Load a CT" in warning.call_args.args[2]
    dialog.assert_not_called()
    save.assert_not_called()


def test_computation_warnings_shown_once_after_run(qapp, monkeypatch):
    import time
    from unittest.mock import Mock

    from pyRadPlan.core import ProgressReporter

    w = WorkflowWidget(WorkspaceManager())
    warning = Mock()
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QMessageBox.warning", warning
    )

    def _compute():
        reporter = ProgressReporter()
        reporter.report_warning("objectives converted")
        reporter.report_warning("objectives converted")
        reporter.report_warning("objectives kept")
        return 1

    done = []
    w._run_in_thread(_compute, on_success=done.append)
    deadline = time.monotonic() + 5.0
    while not done and time.monotonic() < deadline:
        qapp.processEvents()

    assert done == [1]
    warning.assert_called_once()
    assert warning.call_args.args[2] == "objectives converted\n\nobjectives kept"
    assert w._pending_warnings == []


def test_stop_button_cancels_running_computation(qapp, monkeypatch):
    import time
    from unittest.mock import Mock

    from pyRadPlan.core import ProgressReporter

    w = WorkflowWidget(WorkspaceManager())
    critical = Mock()
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QMessageBox.critical", critical
    )
    assert not w._btn_stop.isEnabled()

    def _compute():
        reporter = ProgressReporter()
        reporter.console_progress = False
        for _ in reporter.track(range(10_000), name="Ray"):
            time.sleep(0.001)
        return "finished"

    done = []
    w._run_in_thread(_compute, on_success=done.append)
    assert w._btn_stop.isEnabled()

    w._btn_stop.click()
    assert not w._btn_stop.isEnabled()
    assert w._lbl_status.text() == "Stopping…"

    deadline = time.monotonic() + 5.0
    while w.is_busy and time.monotonic() < deadline:
        qapp.processEvents()
    qapp.processEvents()

    assert done == []
    critical.assert_not_called()
    assert w._lbl_status.text() == "Stopped"
    assert not w._btn_stop.isEnabled()


class _Editor:
    """An editor with pending changes, applying successfully unless told otherwise."""

    def __init__(self, applies=True):
        self.pending, self.applies, self.applied = True, applies, 0

    def has_pending(self):
        return self.pending

    def apply(self):
        self.applied += 1
        self.pending = not self.applies
        return self.applies


def _workflow_with_editor(monkeypatch, choice, applies=True):
    w = WorkflowWidget(WorkspaceManager())
    editor = _Editor(applies)
    w.add_pending_changes_source("plan", editor.has_pending, editor.apply)
    asked = []

    def ask(step, names, outdates_input):
        asked.append((step, names, outdates_input))
        return choice

    monkeypatch.setattr(WorkflowWidget, "_ask_pending_changes", staticmethod(ask))
    runs = []
    monkeypatch.setattr(
        WorkflowWidget,
        "_run_in_thread",
        staticmethod(lambda fn, **kw: runs.append(kw["busy_text"])),
    )
    return w, editor, asked, runs


@pytest.mark.parametrize(
    ("choice", "applied", "runs"),
    [("apply", 1, 1), ("continue", 0, 1), ("cancel", 0, 0)],
)
def test_pending_plan_changes_are_offered_before_the_dose_influence(
    qapp, monkeypatch, choice, applied, runs
):
    w, editor, asked, started = _workflow_with_editor(monkeypatch, choice)

    w._on_calc_dose()

    assert asked == [("Calculate Dose Influence", ["plan"], False)]
    assert editor.applied == applied
    assert len(started) == runs


@pytest.mark.parametrize(
    ("handler", "step"), [("_on_optimize", "Optimize"), ("_on_recalc_dose", "Recalculate Dose")]
)
def test_applying_pending_changes_stops_steps_using_the_dose_influence(
    qapp, monkeypatch, handler, step
):
    w, editor, asked, started = _workflow_with_editor(monkeypatch, "apply")
    w._ws.set_many(result={"w": np.array([1.0])})

    getattr(w, handler)()

    assert asked == [(step, ["plan"], True)]
    assert editor.applied == 1
    assert started == []  # the dose influence is outdated after applying


def test_no_question_without_pending_changes_or_after_a_failed_apply(qapp, monkeypatch):
    w, editor, asked, started = _workflow_with_editor(monkeypatch, "apply", applies=False)

    w._on_calc_dose()
    assert editor.applied == 1 and started == []  # the plan did not validate

    editor.pending = False
    w._on_calc_dose()
    assert len(asked) == 1 and len(started) == 1


def test_renaming_a_structure_keeps_the_dose_influence(qapp):
    """A rename leaves every tissue untouched, so the dose influence stays valid."""
    from pyRadPlan.io import load_tg119

    ct, cst = load_tg119()
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    ws.set_many(ct=ct, cst=cst, pln=object(), stf=object(), dij=object(), result=_result())

    cst.vois = [cst.vois[0].model_copy(update={"name": "renamed"}), *cst.vois[1:]]
    ws.cst = cst

    assert not w._dij_stale


# ----------------------------------------------------------------------
# Biological model checks before a calculation
# ----------------------------------------------------------------------


def _answer_message_boxes(monkeypatch, button_text):
    """Answer every message box by clicking the button labelled `button_text`."""
    from PySide6.QtWidgets import QMessageBox

    shown = []

    def exec_(box):
        shown.append(f"{box.text()}\n{box.informativeText()}")
        next(b for b in box.buttons() if b.text().replace("&", "") == button_text).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    return shown


def _bio_workflow(monkeypatch, bio_model, cst=None, prop_dose_calc=None):
    """A workflow on TG119 with an oxygen plan, whose calculations are only recorded."""
    import numpy as np
    from types import SimpleNamespace

    from pyRadPlan.io import load_tg119
    from pyRadPlan.plan import IonPlan

    ct, tg119_cst = load_tg119()
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    pln = IonPlan(
        radiation_mode="oxygen",
        machine="Generic",
        bio_model=bio_model,
        prop_stf={"gantry_angles": [0.0], "couch_angles": [0.0]},
        prop_dose_calc=prop_dose_calc or {},
    )
    stf = SimpleNamespace(
        beams=[SimpleNamespace(gantry_angle=0.0, couch_angle=0.0, iso_center=np.zeros(3))]
    )
    ws.set_many(
        ct=ct, cst=cst or tg119_cst, pln=pln, stf=stf, result={"w": np.ones(2)}, dij=object()
    )
    runs = []
    monkeypatch.setattr(
        WorkflowWidget,
        "_run_in_thread",
        staticmethod(lambda fn, **kw: runs.append(kw["busy_text"])),
    )
    return ws, w, runs


@pytest.mark.parametrize("handler", ["_on_calc_dose", "_on_recalc_dose"])
def test_failing_tissues_offer_the_physical_dose_only(qapp, monkeypatch, handler):
    from pyRadPlan.gui.widgets.plan import PlanWidget

    ws, w, runs = _bio_workflow(monkeypatch, "zstar_kernels")
    plan_widget = PlanWidget(workspace=ws)
    assert plan_widget._cmb_bio_model.currentText() == "z* kernels · mMKM"
    shown = _answer_message_boxes(monkeypatch, "Calculate physical dose only")

    getattr(w, handler)()

    assert len(shown) == 1 and "No matching tissue class" in shown[0]
    assert ws.pln.bio_model.model == "none"
    assert ws.pln.radiation_mode == "oxygen" and ws.pln.prop_stf["gantry_angles"] == [0.0]
    assert len(runs) == 1
    # The plan widget follows the reconfigured plan
    assert plan_widget._cmb_bio_model.currentText() == "None (physical dose)"


@pytest.mark.parametrize("handler", ["_on_calc_dose", "_on_recalc_dose"])
def test_cancelling_the_tissue_check_runs_nothing(qapp, monkeypatch, handler):
    ws, w, runs = _bio_workflow(monkeypatch, "zstar_kernels")
    pln = ws.pln
    shown = _answer_message_boxes(monkeypatch, "Cancel")

    getattr(w, handler)()

    assert len(shown) == 1
    assert ws.pln is pln and ws.pln.bio_model.model == "zstar_kernels"
    assert runs == []


def test_matching_tissues_calculate_without_asking(qapp, monkeypatch):
    ws, w, runs = _bio_workflow(monkeypatch, "alpha_beta_kernels")
    pln = ws.pln
    shown = _answer_message_boxes(monkeypatch, "Cancel")

    w._on_calc_dose()

    assert shown == []
    assert ws.pln is pln
    assert runs == ["Calculating dose influence…"]


def test_the_check_uses_the_dose_engine_of_the_plan(qapp, monkeypatch):
    """TOPAS does not use the machine's α/β kernels: the same plan now needs a decision."""
    ws, w, runs = _bio_workflow(
        monkeypatch, "alpha_beta_kernels", prop_dose_calc={"engine": "TOPAS"}
    )
    shown = _answer_message_boxes(monkeypatch, "Calculate physical dose only")

    w._on_calc_dose()

    assert len(shown) == 1
    assert "TOPAS does not use the machine's α/β kernels" in shown[0]
    assert "dose engine" in shown[0]
    assert ws.pln.bio_model.model == "none"
    assert ws.pln.prop_dose_calc["engine"] == "TOPAS"
    assert runs == ["Calculating dose influence…"]


def test_invalid_reference_parameters_block_the_calculation(qapp, monkeypatch):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QMessageBox

    from pyRadPlan.biology.tissues import TissueProperties

    tissue = TissueProperties.model_construct(alpha_x=-0.1, beta_x=0.05)
    cst = SimpleNamespace(vois=[SimpleNamespace(name="PTV", tissue_properties=tissue)])
    ws, w, runs = _bio_workflow(monkeypatch, "alpha_beta_kernels", cst=cst)
    pln = ws.pln
    critical = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: critical.append(args[2]))
    shown = _answer_message_boxes(monkeypatch, "Calculate physical dose only")

    w._on_calc_dose()

    assert len(critical) == 1 and "PTV" in critical[0]
    assert shown == []  # no way around it
    assert ws.pln is pln
    assert runs == []


@pytest.mark.parametrize(
    ("quantity", "bio_model", "blocked"),
    [
        ("rbe_x_dose", None, True),
        ("physical_dose", None, False),
        ("rbe_x_dose", "constant_rbe", False),
    ],
)
def test_optimize_explains_biological_objectives_without_a_biological_dij(
    qapp, monkeypatch, quantity, bio_model, blocked
):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QMessageBox

    from pyRadPlan.biology.models import create_bio_model
    from pyRadPlan.plan import PhotonPlan

    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    objective = SimpleNamespace(quantity=quantity)
    ws.set_many(
        pln=PhotonPlan(),
        cst=SimpleNamespace(vois=[SimpleNamespace(name="PTV", objectives=[objective])]),
        dij=SimpleNamespace(
            bio_model=create_bio_model(bio_model) if bio_model else None, alpha_dose=None
        ),
    )
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    runs = []
    monkeypatch.setattr(
        WorkflowWidget,
        "_run_in_thread",
        staticmethod(lambda fn, **kw: runs.append(kw["busy_text"])),
    )
    monkeypatch.setattr(WorkflowWidget, "_open_optimization_status_window", lambda *a: None)

    w._on_optimize()

    if blocked:
        assert len(warnings) == 1
        assert "PTV: RBExDose" in warnings[0]
        assert "without a biological model" in warnings[0]
        assert runs == []
    else:
        assert warnings == []
        assert len(runs) == 1


def _mkm_cst(model="MKM"):
    """TG119 structures whose tissues have the geometry of the oxygen machine's MKM class."""
    from pyRadPlan.cst import StructureSet, create_voi
    from pyRadPlan.io import load_tg119
    from pyRadPlan.biology.tissues import MKMTissueProperties

    ct, cst = load_tg119()
    tissue = MKMTissueProperties(
        model=model,
        alpha_x=0.1,
        beta_x=0.05,
        r_nucleus=3.9,
        r_domain=0.32,
        alpha_0=0.1,
        beta_0=0.05,
    )
    vois = []
    for voi in cst.vois:
        data = dict(voi)
        data.update(tissue_properties=tissue, alpha_x=tissue.alpha_x, beta_x=tissue.beta_x)
        vois.append(create_voi(data))
    return StructureSet(ct_image=ct, vois=vois)


@pytest.mark.parametrize("handler", ["_on_calc_dose", "_on_recalc_dose"])
def test_tissue_warnings_calculate_with_the_model_without_asking(
    qapp, monkeypatch, caplog, handler
):
    from PySide6.QtWidgets import QMessageBox

    from pyRadPlan.gui.widgets.plan import PlanWidget

    ws, w, runs = _bio_workflow(monkeypatch, "zstar_kernels", cst=_mkm_cst())
    plan_widget = PlanWidget(workspace=ws)
    pln, cst = ws.pln, ws.cst
    assert plan_widget.tissue_marker_state == "warning"

    def no_box(*_args, **_kwargs):
        raise AssertionError("unexpected message box")

    monkeypatch.setattr(QMessageBox, "exec", no_box)
    monkeypatch.setattr(QMessageBox, "warning", no_box)
    monkeypatch.setattr(QMessageBox, "critical", no_box)

    with caplog.at_level("WARNING"):
        getattr(w, handler)()

    # The plan keeps its biological model and the structures their labels
    assert ws.pln is pln and ws.pln.bio_model.model == "zstar_kernels"
    assert ws.cst is cst
    assert all(voi.tissue_properties.model == "MKM" for voi in ws.cst.vois)
    assert len(runs) == 1
    # Shown without blocking: logged, and in the status line
    assert any("is labelled MKM" in record.getMessage() for record in caplog.records)
    assert "tissue warning" in w._lbl_status.text()


def test_tissue_check_without_a_quick_fix_offers_none(qapp, monkeypatch):
    ws, w, runs = _bio_workflow(monkeypatch, "zstar_kernels")
    shown = []

    def exec_(box):
        shown.append([b.text() for b in box.buttons()])
        next(b for b in box.buttons() if b.text().replace("&", "") == "Cancel").click()
        return 0

    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "exec", exec_)

    w._on_calc_dose()

    assert len(shown) == 1
    assert not any(text.startswith("Use the base data's model") for text in shown[0])
    assert runs == []


def _optimize_setup(monkeypatch, scenarios, calls, objectives=()):
    """A workflow widget whose optimization runs synchronously on a stand-in dij."""
    from types import SimpleNamespace

    from pyRadPlan.plan import PhotonPlan

    optimized = create_result(
        {
            "physical_dose": _dose_image(),
            "physical_dose_scenario": [_dose_image(), _dose_image()],
            "scenario_labels": ("a", "b"),
        },
        fluence=np.ones(2),
        info={"status": "done"},
    )

    def fluence_optimization(*args, **kwargs):
        calls.append(kwargs)
        return optimized

    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    voi = SimpleNamespace(name="t", objectives=list(objectives))
    ws.set_many(
        pln=PhotonPlan(),
        cst=SimpleNamespace(vois=[voi]),
        dij=SimpleNamespace(
            bio_model=object(),
            alpha_dose=None,
            scenarios=None if scenarios is None else list(range(scenarios)),
        ),
    )
    monkeypatch.setattr("pyRadPlan.fluence_optimization", fluence_optimization)
    monkeypatch.setattr(
        WorkflowWidget,
        "_run_in_thread",
        staticmethod(lambda fn, on_success, **kw: on_success(fn())),
    )
    monkeypatch.setattr(WorkflowWidget, "_open_optimization_status_window", lambda *a: None)
    return ws, w, optimized


def test_optimize_requests_results_per_scenario(qapp, monkeypatch):
    calls = []
    ws, w, optimized = _optimize_setup(monkeypatch, 3, calls)

    w._on_optimize()

    # The optimization reports the scenarios itself; the result is not computed again
    assert calls == [{"scenarios": True}]
    assert ws.result is optimized


def test_recalculate_replaces_the_active_result_only(qapp, monkeypatch):
    from unittest.mock import Mock

    forward = Mock(return_value=_result())
    ws, w = _recalc_setup(monkeypatch, {"iso_center": [4.0, 5.0, 6.0]}, [[4.0, 5.0, 6.0]], forward)
    ws.result = {
        "w": np.array([2.0, 3.0]),
        "physical_dose": _dose_image(),
        "let": _dose_image(),
        "physical_dose_scenario": [_dose_image(), _dose_image()],
        "scenario_labels": ("a", "b"),
    }
    ws.add_result("saved", ws.result.model_copy())
    snapshot = ws.results["saved"]

    w._on_recalc_dose()

    # Nothing of the previous doses is left behind in the recalculated result
    assert ws.result.stored_quantities == forward.return_value.stored_quantities
    assert ws.result.scenario_quantities == {}
    assert ws.result.scenario_labels is None
    assert ws.results["saved"] is snapshot
    assert snapshot.scenario_labels == ("a", "b")


def test_snapshots_keep_results_per_scenario(qapp, monkeypatch):
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    ws.result = {
        "physical_dose": _dose_image(),
        "physical_dose_scenario": [_dose_image(), None],
        "scenario_labels": ("nominal", "shifted"),
    }
    scenario_doses = ws.result.scenario_quantities["physical_dose"]
    names = iter(["current", "scenario"])
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kw: (next(names), True))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))

    w._on_save_result()  # the name of the result being worked on
    assert len(warnings) == 1
    assert list(ws.results) == ["current"]

    w._on_save_result()  # no longer clashes with the per-scenario entries
    assert len(warnings) == 1
    assert ws.results["scenario"].scenario_quantities["physical_dose"] == scenario_doses

    # The snapshot keeps its scenarios and their labels through a re-optimization
    ws.result = create_result({"physical_dose": _dose_image(2.0)})
    assert ws.results["scenario"].stored_scenario_quantities == ["physical_dose"]
    assert ws.results["scenario"].scenario_labels == ("nominal", "shifted")


def test_optimize_keeps_every_result_per_scenario_with_their_labels(qapp, monkeypatch):
    from types import SimpleNamespace

    calls = []
    objectives = [SimpleNamespace(quantity="physical_dose")]
    ws, w, _ = _optimize_setup(monkeypatch, 2, calls, objectives)

    w._on_optimize()

    # Also the RBE dose the viewer shows first, though the objectives are on physical dose
    assert calls == [{"scenarios": True}]
    assert ws.result.scenario_labels == ("a", "b")


def test_recalculate_keeps_the_scenarios_of_every_quantity(qapp, monkeypatch):
    from unittest.mock import Mock

    dose = _dose_image()
    forward = Mock(
        return_value=create_result(
            {
                "physical_dose": dose,
                "rbe_x_dose": dose,
                "physical_dose_scenario": [dose, dose],
                "rbe_x_dose_scenario": [dose, dose],
                "let_scenario": [dose, dose],
                "scenario_labels": ("from plan 0", "from plan 1"),
            }
        )
    )
    ws, w = _recalc_setup(monkeypatch, {"iso_center": [4.0, 5.0, 6.0]}, [[4.0, 5.0, 6.0]], forward)
    ws.result = ws.result.model_copy(update={"scenario_labels": ("stale",)})

    w._on_recalc_dose()

    assert ws.result.scenario_labels == ("from plan 0", "from plan 1")
    # Not only those the objectives evaluate: the viewer shows a proton plan's RBE dose first
    for key in ("physical_dose", "rbe_x_dose", "let"):
        assert [q.values for q in ws.result.scenario_quantities[key]] == [dose, dose]
