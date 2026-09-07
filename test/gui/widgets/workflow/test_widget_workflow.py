import pytest

from pyRadPlan.core import ProgressLevel, ProgressReport, StatusReport
from pyRadPlan.gui.widgets.workflow import WorkflowWidget
from pyRadPlan.gui.workspace import WorkspaceManager


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

    # Loading everything at once leaves the products current, not stale.
    ws.set_many(ct=object(), cst=object(), pln=object(), stf=object(), dij=object(), result={})
    assert not w._dij_stale and not w._result_stale

    # Changing the plan invalidates the dose influence and the result.
    ws.pln = object()
    assert w._dij_stale and w._result_stale
    assert w._indicators["dij"].toolTip().startswith("Outdated")

    # Recomputing the dose influence clears its flag (and the result stays stale).
    ws.set_many(stf=object(), dij=object())
    assert not w._dij_stale and w._result_stale

    # Re-optimizing clears the result flag.
    ws.result = {"w": 1}
    assert not w._result_stale


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


@pytest.mark.parametrize("same_grid", [False, True])
def test_new_patient_clears_missing_objects_atomically(qapp, same_grid):
    import SimpleITK as sitk
    from pyRadPlan.ct import validate_ct

    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    old_ct = validate_ct(cube_hu=sitk.Image([5, 5, 5], sitk.sitkFloat32))
    new_ct = validate_ct(cube_hu=sitk.Image([5 if same_grid else 7, 5, 5], sitk.sitkFloat32))
    ws.set_many(ct=old_ct, cst=object(), pln=object(), stf=object(), dij=object(), result={})
    w._saved_tags = ["old"]
    states = []
    ws.workspace_changed.connect(lambda keys: states.append({k: getattr(ws, k) for k in ws.keys}))

    w._merge_loaded_data({"ct": new_ct})

    assert len(states) == 1
    assert states[0]["ct"] is new_ct
    assert all(states[0][key] is None for key in ws.keys if key != "ct")
    assert not w._btn_optimize.isEnabled()
    assert w._saved_tags == []


def test_import_without_ct_keeps_current_patient(qapp):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    ct, cst, new_plan = object(), object(), object()
    ws.set_many(ct=ct, cst=cst)
    w._merge_loaded_data({"pln": new_plan})
    assert ws.ct is ct and ws.cst is cst and ws.pln is new_plan


def test_recalculate_uses_forward_result_and_keeps_weights(qapp, monkeypatch):
    import numpy as np
    from unittest.mock import Mock

    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    weights = np.array([2.0, 3.0])
    snapshot, fresh_dose = object(), object()
    ws.set_many(
        ct=object(),
        cst=object(),
        pln=object(),
        stf=object(),
        result={"w": weights, "physical_dose": object(), "physical_dose_saved": snapshot},
    )
    forward = Mock(return_value={"physical_dose": fresh_dose})
    monkeypatch.setattr("pyRadPlan.calc_dose_forward", forward)
    monkeypatch.setattr(w, "_run_in_thread", lambda fn, on_success, **kw: on_success(fn()))

    w._on_recalc_dose()

    forward.assert_called_once_with(ws.ct, ws.cst, ws.stf, ws.pln, weights)
    assert ws.result["physical_dose"] is fresh_dose
    assert ws.result["w"] is weights
    assert ws.result["physical_dose_saved"] is snapshot


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ""])
def test_save_workspace_preserves_results_and_snapshot_tags(qapp, tmp_path, monkeypatch, suffix):
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
        result={
            "physical_dose": dose,
            "w": weights,
            "physical_dose_saved": dose,
            "w_saved": weights.copy(),
        },
    )
    w._saved_tags = ["saved"]
    path = tmp_path / ("workspace" + suffix)
    filters = []

    def choose_file(*args):
        filters.append(args[-1])
        return str(path), ""

    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QFileDialog.getSaveFileName", choose_file
    )
    monkeypatch.setattr(w, "_run_save", lambda fn, *args: fn())

    w._on_save_workspace()
    data = load_data(path if suffix else path.with_suffix(".pkl"))
    restored_ws = WorkspaceManager()
    restored = WorkflowWidget(restored_ws)
    restored._merge_loaded_data(data)

    assert "*.pkl" in filters[0] and "*.mat" not in filters[0] and "*.npz" not in filters[0]
    assert restored_ws.pln.prop_opt == {"solver": "scipy"}
    assert restored_ws.stf == ws.stf and restored_ws.dij == ws.dij
    assert set(restored_ws.result) == set(ws.result)
    np.testing.assert_array_equal(restored_ws.result["w"], weights)
    np.testing.assert_array_equal(restored_ws.result["w_saved"], weights)
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(restored_ws.result["physical_dose"]), sitk.GetArrayFromImage(dose)
    )
    assert restored_ws.result["physical_dose"].GetOrigin() == dose.GetOrigin()
    assert restored._is_tagged("physical_dose_saved")


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
    monkeypatch.setattr(w, "_run_save", save)
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
    monkeypatch.setattr(w, "_run_save", save)

    w._on_save_workspace()

    warning.assert_called_once()
    assert "Load a CT" in warning.call_args.args[2]
    dialog.assert_not_called()
    save.assert_not_called()
