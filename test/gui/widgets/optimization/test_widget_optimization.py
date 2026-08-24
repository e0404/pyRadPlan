"""Tests for the optimization objectives editor widget."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox

from pyRadPlan.io import load_tg119
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets.optimization import OptimizationWidget
from pyRadPlan.optimization.objectives import Objective, get_objective


@pytest.fixture
def tg119():
    ct, cst = load_tg119()
    return ct, cst


def _objectives(voi):
    return [get_objective(o) for o in voi.objectives if o is not None]


def _total_objectives(cst):
    return sum(len(_objectives(v)) for v in cst.vois)


def test_constructs_with_empty_workspace(qapp):
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    assert widget is not None
    assert widget._table.rowCount() == 0


def test_builds_rows_for_vois_with_objectives(qapp, tg119):
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)

    ws.set_many(ct=ct, cst=cst)

    # All objectives are listed together, one row each, grouped by VOI.
    expected = _total_objectives(cst)
    assert expected > 0
    assert widget._table.rowCount() == expected


def test_add_objective_updates_cst_and_adds_row(qapp, tg119):
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst)

    before_rows = widget._table.rowCount()
    before_total = _total_objectives(ws.cst)

    widget._cmb_voi.setCurrentIndex(0)
    widget._on_add_objective()

    assert _total_objectives(ws.cst) == before_total + 1
    assert widget._table.rowCount() == before_rows + 1


def test_remove_objective_updates_cst(qapp, tg119):
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst)

    # find a VOI that has at least one objective
    voi_idx = next(i for i, v in enumerate(ws.cst.vois) if _objectives(v))
    before_total = _total_objectives(ws.cst)
    before_rows = widget._table.rowCount()

    widget._on_remove_objective(voi_idx, 0)

    assert _total_objectives(ws.cst) == before_total - 1
    assert widget._table.rowCount() == before_rows - 1


def test_change_penalty_writes_through(qapp, tg119):
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst)

    voi_idx = next(i for i, v in enumerate(ws.cst.vois) if _objectives(v))
    widget._on_penalty_changed(voi_idx, 0, 123.0)

    assert _objectives(ws.cst.vois[voi_idx])[0].priority == pytest.approx(123.0)


def test_change_param_writes_through(qapp, tg119):
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst)

    voi_idx = next(i for i, v in enumerate(ws.cst.vois) if _objectives(v))
    obj = _objectives(ws.cst.vois[voi_idx])[0]
    param = obj.parameter_names[0]

    widget._on_param_changed(voi_idx, 0, param, 42.0)

    updated = _objectives(ws.cst.vois[voi_idx])[0]
    assert getattr(updated, param) == pytest.approx(42.0)


def test_change_objective_type_preserves_penalty(qapp, tg119):
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst)

    voi_idx = next(i for i, v in enumerate(ws.cst.vois) if _objectives(v))
    obj = _objectives(ws.cst.vois[voi_idx])[0]
    obj.priority = 77.0
    cst2 = ws.cst
    cst2.vois[voi_idx].objectives = [obj]
    with widget.hold_updates():
        ws.cst = cst2

    new_name = next(n for n in widget._available if n != obj.name)
    widget._on_objective_changed(voi_idx, 0, new_name)

    result = _objectives(ws.cst.vois[voi_idx])[0]
    assert result.name == new_name
    assert result.priority == pytest.approx(77.0)


def test_hold_updates_prevents_recursion(qapp, tg119):
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst)

    calls = []
    original = widget._do_update
    widget._do_update = lambda keys: (calls.append(keys), original(keys))[1]

    voi_idx = next(i for i, v in enumerate(ws.cst.vois) if _objectives(v))
    widget._on_penalty_changed(voi_idx, 0, 5.0)

    # the write happened inside hold_updates, so _do_update must not have fired
    assert calls == []


def test_invalid_param_is_ignored(qapp, tg119):
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst)

    voi_idx = next(i for i, v in enumerate(ws.cst.vois) if _objectives(v))
    obj = _objectives(ws.cst.vois[voi_idx])[0]
    param = obj.parameter_names[0]
    before = getattr(obj, param)

    # negative value violates the ge=0.0 constraint on reference parameters
    widget._on_param_changed(voi_idx, 0, param, -100.0)

    after = getattr(_objectives(ws.cst.vois[voi_idx])[0], param)
    assert after == pytest.approx(before)


def test_ai_button_requires_cst_and_pln(qapp, tg119, monkeypatch):
    # Pretend AI is usable so the cst/pln gating is exercised even on machines
    # without pydantic-ai or provider API keys.
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.optimization._optimization_widget.ai_disabled_reason",
        lambda: None,
    )
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)

    # cst but no pln -> still disabled
    ws.set_many(ct=ct, cst=cst)
    assert not widget._btn_ai.isEnabled()

    from pyRadPlan.plan import PhotonPlan

    ws.pln = PhotonPlan()
    assert widget._btn_ai.isEnabled()


def test_ai_button_disabled_with_reason(qapp, tg119, monkeypatch):
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.optimization._optimization_widget.ai_disabled_reason",
        lambda: "no model",
    )
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)

    from pyRadPlan.plan import PhotonPlan

    ws.set_many(ct=ct, cst=cst)
    ws.pln = PhotonPlan()
    assert not widget._btn_ai.isEnabled()
    assert widget._btn_ai.toolTip() == "no model"


def test_change_quantity_writes_through(qapp, tg119):
    from pyRadPlan.quantities import get_available_quantities

    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst)

    voi_idx = next(i for i, v in enumerate(ws.cst.vois) if _objectives(v))
    obj = _objectives(ws.cst.vois[voi_idx])[0]
    new_quantity = next(q for q in get_available_quantities() if q != obj.quantity)

    row = next(
        r
        for r in range(widget._table.rowCount())
        if widget._table.cellWidget(r, OptimizationWidget._COL_QUANTITY) is not None
    )
    cmb = widget._table.cellWidget(row, OptimizationWidget._COL_QUANTITY)
    assert cmb.currentText() == obj.quantity

    widget._on_quantity_changed(voi_idx, 0, new_quantity)
    assert _objectives(ws.cst.vois[voi_idx])[0].quantity == new_quantity


def _image_reference_combos(widget):
    """All image-reference dropdowns in the parameters column."""
    combos = []
    for row in range(widget._table.rowCount()):
        cell = widget._table.cellWidget(row, OptimizationWidget._COL_PARAMS)
        if cell is not None:
            combos.extend(cell.findChildren(QComboBox))
    return combos


@pytest.fixture
def mimicking_workspace(qapp, tg119):
    """Workspace with a Squared Mimicking objective and one result dose."""
    import SimpleITK as sitk

    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)

    voi_idx = next(i for i, v in enumerate(cst.vois) if _objectives(v))
    cst.vois[voi_idx].objectives = list(cst.vois[voi_idx].objectives) + [
        get_objective("Squared Mimicking")
    ]

    ref = sitk.Image([4, 4, 4], sitk.sitkFloat32)
    ws.set_many(ct=ct, cst=cst, result={"physical_dose": ref})
    return ws, widget, voi_idx, ref


def test_image_reference_param_shows_dropdown(mimicking_workspace):
    ws, widget, voi_idx, ref = mimicking_workspace

    combos = _image_reference_combos(widget)
    assert len(combos) == 1
    labels = [combos[0].itemText(i) for i in range(combos[0].count())]
    # default d_ref is not a workspace entry, so it is kept as a "Custom" line
    assert labels[0].startswith("Custom image (1×1×1")
    assert "physical_dose" in labels


def test_image_reference_selection_writes_through(mimicking_workspace):
    ws, widget, voi_idx, ref = mimicking_workspace

    cmb = _image_reference_combos(widget)[0]
    cmb.setCurrentIndex(cmb.findText("physical_dose"))

    assert _objectives(ws.cst.vois[voi_idx])[-1].d_ref is ref

    # after a rebuild the selection is re-identified from the workspace
    widget._rebuild_table(ws.cst)
    cmb = _image_reference_combos(widget)[0]
    assert cmb.currentText() == "physical_dose"
    labels = [cmb.itemText(i) for i in range(cmb.count())]
    assert not any(label.startswith("Custom") for label in labels)


def test_image_reference_accepts_imported_numpy_dose(mimicking_workspace):
    import numpy as np

    ws, widget, voi_idx, ref = mimicking_workspace
    import SimpleITK as sitk

    cube = sitk.GetArrayFromImage(ws.ct.cube_hu).astype(float)
    ws.result = {**ws.result, "import_measured": cube}

    cmb = _image_reference_combos(widget)[0]
    cmb.setCurrentIndex(cmb.findText("import_measured"))

    d_ref = _objectives(ws.cst.vois[voi_idx])[-1].d_ref
    assert isinstance(d_ref, tuple)
    assert d_ref[0] is cube


def _tg119_dose_image(ct):
    import numpy as np
    import SimpleITK as sitk

    dose_array = np.zeros(ct.size[::-1])
    dose_array[60:80, 80:120, 80:120] = 1.0
    dose = sitk.GetImageFromArray(dose_array)
    dose.SetSpacing((ct.resolution["x"], ct.resolution["y"], ct.resolution["z"]))
    dose.SetOrigin(tuple(ct.origin))
    dose.SetDirection(tuple(ct.direction[i] for i in range(9)))
    return dose


def test_prompt_qi_adaptation_skipped_without_result(qapp, tg119):
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst)

    qis, accepted = widget._prompt_qi_adaptation(ws.cst)
    assert qis is None
    assert accepted


def test_prompt_qi_adaptation_computes_collection(qapp, tg119, monkeypatch):
    from pyRadPlan.analysis import QICollection
    from pyRadPlan.gui.widgets.optimization import _optimization_widget as mod

    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst, result={"physical_dose": _tg119_dose_image(ct)})

    monkeypatch.setattr(
        mod.QInputDialog,
        "getItem",
        staticmethod(lambda *a, **k: ("Adapt using QIs from 'physical_dose'", True)),
    )

    qis, accepted = widget._prompt_qi_adaptation(ws.cst)
    assert accepted
    assert isinstance(qis, QICollection)
    assert all(voi.name in qis for voi in ws.cst.vois)


def test_prompt_qi_adaptation_cancel_and_decline(qapp, tg119, monkeypatch):
    from pyRadPlan.gui.widgets.optimization import _optimization_widget as mod

    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst, result={"physical_dose": _tg119_dose_image(ct)})

    monkeypatch.setattr(mod.QInputDialog, "getItem", staticmethod(lambda *a, **k: ("", False)))
    qis, accepted = widget._prompt_qi_adaptation(ws.cst)
    assert qis is None
    assert not accepted

    monkeypatch.setattr(
        mod.QInputDialog,
        "getItem",
        staticmethod(lambda *a, **k: ("No — suggest new objectives", True)),
    )
    qis, accepted = widget._prompt_qi_adaptation(ws.cst)
    assert qis is None
    assert accepted


def test_objectives_normalized_to_instances(qapp, tg119):
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = OptimizationWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst)

    voi_idx = next(i for i, v in enumerate(ws.cst.vois) if _objectives(v))
    widget._on_penalty_changed(voi_idx, 0, 9.0)

    for obj in ws.cst.vois[voi_idx].objectives:
        assert isinstance(obj, Objective)
