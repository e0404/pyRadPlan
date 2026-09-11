import pytest
import SimpleITK as sitk
from pyRadPlan.plan import IonPlan, PhotonPlan
from pyRadPlan.result import PlanResult, create_result

from pyRadPlan.core import ProgressLevel, ProgressReport, StatusReport
from pyRadPlan.gui.widgets.workflow import WorkflowWidget
from pyRadPlan.gui.widgets.workflow._workflow_widget import IMPORT_PREFIX, _result_with
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


def test_sequence_button_gated_on_result_stf_and_photon_mode(qapp):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    assert not w._btn_sequence.isEnabled()

    pln = PhotonPlan(prop_seq={"run_sequencing": True, "sequencer": "xia"})
    result = create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})
    ws.set_many(pln=pln, stf=object(), dij=object(), result=result)
    assert w._btn_sequence.isEnabled()

    ws.pln = IonPlan()
    assert not w._btn_sequence.isEnabled()


def test_sequence_button_disabled_without_sequencing_configured(qapp):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    result = create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})
    ws.set_many(pln=PhotonPlan(), stf=object(), dij=object(), result=result)

    assert not w._btn_sequence.isEnabled()
    assert "Plan panel" in w._btn_sequence.toolTip()


def test_sequence_button_disabled_without_dij(qapp):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    pln = PhotonPlan(prop_seq={"run_sequencing": True, "sequencer": "xia"})
    result = create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})
    ws.set_many(pln=pln, stf=object(), result=result)

    assert not w._btn_sequence.isEnabled()
    assert "dose influence" in w._btn_sequence.toolTip()


def test_on_sequence_warns_without_dij(qapp, monkeypatch):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    pln = PhotonPlan(prop_seq={"run_sequencing": True, "sequencer": "xia"})
    result = create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})
    ws.set_many(pln=pln, stf=object(), result=result)

    warned = []
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QMessageBox.warning",
        lambda *a, **k: warned.append(a),
    )

    w._on_sequence()

    assert len(warned) == 1
    assert w._thread is None


def test_on_sequence_runs_in_thread_and_updates_result(qapp, monkeypatch):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    pln = PhotonPlan(prop_seq={"run_sequencing": True, "sequencer": "xia"})
    result = create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})
    ws.set_many(pln=pln, stf=object(), dij=object(), result=result)

    sequenced = create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})
    calls = []

    def _fake_run_sequencing(passed_result, pln, stf, dij=None):
        calls.append((passed_result, pln, stf, dij))
        return sequenced

    monkeypatch.setattr("pyRadPlan.run_sequencing", _fake_run_sequencing)

    w._on_sequence()
    if w._thread is not None:
        w._thread.wait(2000)
    qapp.processEvents()

    assert len(calls) == 1
    assert ws.result is sequenced


def test_sequence_button_disabled_when_result_stale_after_radiation_mode_switch(qapp):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    result = create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})
    ws.set_many(pln=IonPlan(), stf=object(), dij=object(), result=result)
    ws.pln = PhotonPlan(prop_seq={"run_sequencing": True, "sequencer": "xia"})

    assert not w._btn_sequence.isEnabled()
    assert "outdated" in w._btn_sequence.toolTip()


def test_on_sequence_warns_when_result_stale(qapp, monkeypatch):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    result = create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})
    ws.set_many(pln=IonPlan(), stf=object(), dij=object(), result=result)
    ws.pln = PhotonPlan(prop_seq={"run_sequencing": True, "sequencer": "xia"})

    warned = []
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QMessageBox.warning",
        lambda *a, **k: warned.append(a),
    )

    w._on_sequence()

    assert len(warned) == 1
    assert w._thread is None


def test_on_sequence_warns_without_result(qapp, monkeypatch):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    warned = []
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QMessageBox.warning",
        lambda *a, **k: warned.append(a),
    )

    w._on_sequence()

    assert len(warned) == 1
    assert w._thread is None


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
    pln = PhotonPlan()

    def _result():
        return create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})

    # Loading everything at once leaves the products current, not stale.
    ws.set_many(ct=object(), cst=object(), pln=pln, stf=object(), dij=object(), result=_result())
    assert not w._dij_stale and not w._result_stale

    # Changing a dose-influence-relevant field (dose grid) invalidates both.
    ws.pln = pln.model_copy(update={"prop_dose_calc": {"dose_grid": {"resolution": [1, 1, 1]}}})
    assert w._dij_stale and w._result_stale
    assert w._indicators["dij"].toolTip().startswith("Outdated")

    # Recomputing the dose influence clears its flag (and the result stays stale).
    ws.set_many(stf=object(), dij=object())
    assert not w._dij_stale and w._result_stale


def test_sequencing_only_plan_change_does_not_mark_stale(qapp):
    """Editing prop_seq (sequencing config) doesn't require redoing dij/optimization."""
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    pln = PhotonPlan()

    def _result():
        return create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})

    ws.set_many(ct=object(), cst=object(), pln=pln, stf=object(), dij=object(), result=_result())
    assert not w._dij_stale and not w._result_stale

    ws.pln = pln.model_copy(
        update={"prop_seq": {"run_sequencing": True, "sequencer": "xia", "num_of_levels": 9}}
    )

    assert not w._dij_stale
    assert not w._result_stale
    assert w._btn_sequence.isEnabled()

    # Re-optimizing clears the result flag.
    ws.result = _result()
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


def test_save_result_snapshots_quantities(qapp, monkeypatch):
    """Save/Keep Result tags a copy of the current quantities onto the result."""
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    ws.result = create_result(
        {"physical_dose": _dose_image(), "physical_dose_beam": [_dose_image()]}
    )
    original = ws.result

    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.workflow._workflow_widget.QInputDialog.getText",
        lambda *a, **k: ("run1", True),
    )
    w._on_save_result()

    assert isinstance(ws.result, PlanResult)
    # The snapshot is added; a new object is stored so the viewer refreshes.
    assert ws.result is not original
    assert sorted(ws.result.stored_quantities) == ["physical_dose", "physical_dose_run1"]
    assert sorted(ws.result.stored_beam_quantities) == ["physical_dose", "physical_dose_run1"]
    assert w._is_tagged("physical_dose_run1")

    # Snapshots survive a re-optimization, the fresh quantities take precedence.
    fresh = create_result({"physical_dose": _dose_image(2.0)})
    w._carry_forward_extras(fresh, ws.result)
    assert sorted(fresh.stored_quantities) == ["physical_dose", "physical_dose_run1"]


def test_imported_dose_is_carried_forward(qapp):
    """An imported quantity is prefixed so it survives the next optimization."""
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    imported = _result_with(None, quantities={f"{IMPORT_PREFIX}ct_dose": _dose_image()})
    fresh = create_result({"physical_dose": _dose_image(2.0)})
    w._carry_forward_extras(fresh, imported)

    assert f"{IMPORT_PREFIX}ct_dose" in fresh


def test_result_to_dose_image_reads_the_model(qapp):
    """Saving a workspace picks the physical dose out of the PlanResult."""
    assert WorkflowWidget._result_to_dose_image(None) is None
    assert WorkflowWidget._result_to_dose_image(create_result({"let": _dose_image()})) is None

    result = create_result({"physical_dose": _dose_image()})
    assert WorkflowWidget._result_to_dose_image(result) is result["physical_dose"].values


def test_sequencing_finished_is_emitted_after_the_result_lands(qapp, monkeypatch):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)

    pln = PhotonPlan(prop_seq={"run_sequencing": True, "sequencer": "xia"})
    result = create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})
    ws.set_many(pln=pln, stf=object(), dij=object(), result=result)

    sequenced = create_result({"physical_dose": sitk.Image([2, 2, 2], sitk.sitkFloat32)})
    monkeypatch.setattr("pyRadPlan.run_sequencing", lambda *a, **k: sequenced)

    seen = []
    w.sequencing_finished.connect(lambda: seen.append(ws.result))

    w._on_sequence()
    if w._thread is not None:
        w._thread.wait(2000)
    qapp.processEvents()

    assert seen == [sequenced]
