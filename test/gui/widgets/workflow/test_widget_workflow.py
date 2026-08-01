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
