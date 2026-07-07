"""Optimization objectives editor widget for the pyRadPlan GUI.

Python translation of matRad's ``matRad_OptimizationWidget``: a per-VOI editor of
the optimization objectives stored on the structure set (``cst``).  Each row binds
one objective on one VOI; the user can change the objective type, its penalty
(priority) and its numeric parameters, and add or remove objectives.  Constraints
are intentionally out of scope because pyRadPlan currently ships objectives only.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import SimpleITK as sitk

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.cst import validate_cst
from pyRadPlan.optimization.objectives import (
    Objective,
    get_available_objectives,
    get_objective,
)
from pyRadPlan.quantities import get_available_quantities
from pyRadPlan.gui.workspace import WorkspaceManager
from .._base import AdaptiveDoubleSpinBox, WorkspaceWidget
from ..ai import AI_MISSING_TIP, ai_available


#: Default objective name picked for a freshly added objective, per VOI type.
_DEFAULT_OBJECTIVE_BY_TYPE = {
    "TARGET": "Squared Deviation",
    "OAR": "Squared Overdosing",
    "EXTERNAL": "Squared Overdosing",
    "HELPER": "Squared Overdosing",
}
_FALLBACK_OBJECTIVE = "Squared Overdosing"


class OptimizationWidget(WorkspaceWidget):
    """Editor for the optimization objectives held on the workspace ``cst``.

    Binds to a :class:`~pyRadPlan.gui.workspace.WorkspaceManager` and rebuilds a
    table (one row per VOI/objective pair) whenever ``ct`` or ``cst`` change.
    Edits are written straight back to ``workspace.cst`` inside
    :meth:`~WorkspaceWidget.hold_updates` so other widgets are notified while this
    widget does not re-enter its own update.

    Parameters
    ----------
    workspace:
        Shared :class:`WorkspaceManager`.  Falls back to the process-wide
        singleton when *None*.
    parent:
        Optional Qt parent widget.
    """

    _watched_keys = ("ct", "cst", "pln", "result")

    # All objectives are listed together, sorted/grouped by VOI; the VOI label is
    # shown once on the first row of each group.
    _COL_VOI = 0
    _COL_OBJECTIVE = 1
    _COL_PENALTY = 2
    _COL_QUANTITY = 3
    _COL_PARAMS = 4
    _COL_REMOVE = 5
    _COLUMNS = ("VOI", "Objective", "Penalty", "Quantity", "Parameters", "")

    def __init__(
        self,
        workspace: Optional[WorkspaceManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(workspace, parent)
        self._available = list(get_available_objectives().keys())
        self._ai_available = ai_available()
        self._setup_ui()
        self.initialize()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        controls = QHBoxLayout()
        self._cmb_voi = QComboBox()
        self._cmb_voi.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        # The selector chooses which VOI a newly added objective belongs to.
        self._btn_add = QPushButton("+ Add objective")
        self._btn_add.clicked.connect(self._on_add_objective)
        self._btn_ai = QPushButton("✨ AI")
        self._btn_ai.clicked.connect(self._on_ai_objectives)
        controls.addWidget(QLabel("VOI:"))
        controls.addWidget(self._cmb_voi)
        controls.addWidget(self._btn_add)
        controls.addWidget(self._btn_ai)
        controls.addStretch()
        # Plan-wide objective count.
        self._lbl_count = QLabel("0 objectives")
        controls.addWidget(self._lbl_count)
        root.addLayout(controls)

        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        # Size every column to its content and let the table scroll horizontally
        # rather than squeezing the spinbox cells until their arrows overrun the
        # value. The last section is not stretched for the same reason.
        for col in range(len(self._COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        # Smooth (per-pixel) scrolling in both directions; per-item scrolling
        # jumps by whole rows, which looks erratic with the tall widget cells.
        self._table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        # Every cell holds an editor widget, so a cell selection highlight only
        # draws a stray square around it. Disable selection and the focus frame.
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setFocusPolicy(Qt.NoFocus)
        self._table.setShowGrid(False)
        root.addWidget(self._table)

        self._lbl_status = QLabel("")
        root.addWidget(self._lbl_status)

    # ------------------------------------------------------------------
    # Workspace -> UI synchronisation
    # ------------------------------------------------------------------

    def _do_update(self, _changed_keys: list) -> None:
        cst = self._ws.cst
        self._rebuild_voi_combo(cst)
        self._rebuild_table(cst)
        self._update_ai_button()

    def _update_ai_button(self) -> None:
        ready = self._ai_available and self._ws.cst is not None and self._ws.pln is not None
        self._btn_ai.setEnabled(ready)
        if not self._ai_available:
            self._btn_ai.setToolTip(AI_MISSING_TIP)
        elif self._ws.cst is None:
            self._btn_ai.setToolTip("Load a structure set first")
        elif self._ws.pln is None:
            self._btn_ai.setToolTip("Configure a plan first")
        else:
            self._btn_ai.setToolTip("Suggest objectives for all VOIs using an LLM")

    def _rebuild_voi_combo(self, cst) -> None:
        previous = self._cmb_voi.currentData()
        self._cmb_voi.blockSignals(True)
        self._cmb_voi.clear()
        if cst is not None:
            for idx, voi in enumerate(cst.vois):
                self._cmb_voi.addItem(f"{voi.name} ({voi.voi_type})", idx)
            if previous is not None:
                restored = self._cmb_voi.findData(previous)
                if restored >= 0:
                    self._cmb_voi.setCurrentIndex(restored)
        self._cmb_voi.blockSignals(False)
        self._btn_add.setEnabled(cst is not None and self._cmb_voi.count() > 0)

    def _rebuild_table(self, cst) -> None:
        self._table.setRowCount(0)
        if cst is None:
            self._set_status("No structure set loaded")
            self._update_count(cst)
            return

        for voi_idx, voi in enumerate(cst.vois):
            objectives = self._iter_objectives(voi)
            for obj_idx, obj in enumerate(objectives):
                # Show the VOI label only on the first row of each group.
                label = f"{voi.name} ({voi.voi_type})" if obj_idx == 0 else ""
                self._insert_row(voi_idx, obj_idx, obj, label)
        self._set_status("")
        self._update_count(cst)

    def _update_count(self, cst) -> None:
        total = 0
        if cst is not None:
            total = sum(len(self._iter_objectives(voi)) for voi in cst.vois)
        self._lbl_count.setText(f"{total} objective{'s' if total != 1 else ''}")

    @staticmethod
    def _iter_objectives(voi) -> list[Objective]:
        """Return the VOI's objectives normalized to :class:`Objective` instances."""
        objectives = []
        for raw in voi.objectives or []:
            if raw is None or (isinstance(raw, (list, tuple)) and len(raw) == 0):
                continue
            try:
                objectives.append(get_objective(raw))
            except (KeyError, ValueError, TypeError):
                continue
        return objectives

    def _insert_row(self, voi_idx: int, obj_idx: int, obj: Objective, voi_label: str) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        voi_item = QTableWidgetItem(voi_label)
        voi_item.setFlags(Qt.ItemIsEnabled)
        if voi_label:
            font = voi_item.font()
            font.setBold(True)
            voi_item.setFont(font)
        self._table.setItem(row, self._COL_VOI, voi_item)

        cmb = QComboBox()
        cmb.addItems(self._available)
        cmb.setCurrentText(obj.name)
        cmb.currentTextChanged.connect(
            lambda text, v=voi_idx, o=obj_idx: self._on_objective_changed(v, o, text)
        )
        self._table.setCellWidget(row, self._COL_OBJECTIVE, cmb)

        penalty = AdaptiveDoubleSpinBox()
        penalty.setRange(0.0, 1.0e9)
        penalty.setValue(float(obj.priority))
        # The size hint of the spinbox spans its full 1e9 range, which blows up
        # the ResizeToContents column; typical penalties are much shorter.
        penalty.setMaximumWidth(100)
        penalty.valueChanged.connect(
            lambda value, v=voi_idx, o=obj_idx: self._on_penalty_changed(v, o, value)
        )
        self._table.setCellWidget(row, self._COL_PENALTY, penalty)

        cmb_quantity = QComboBox()
        cmb_quantity.addItems(list(get_available_quantities().keys()))
        cmb_quantity.setCurrentText(obj.quantity)
        cmb_quantity.currentTextChanged.connect(
            lambda text, v=voi_idx, o=obj_idx: self._on_quantity_changed(v, o, text)
        )
        self._table.setCellWidget(row, self._COL_QUANTITY, cmb_quantity)

        self._table.setCellWidget(
            row, self._COL_PARAMS, self._build_param_editor(voi_idx, obj_idx, obj)
        )

        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(
            lambda _=False, v=voi_idx, o=obj_idx: self._on_remove_objective(v, o)
        )
        self._table.setCellWidget(row, self._COL_REMOVE, btn_remove)

    def _build_param_editor(self, voi_idx: int, obj_idx: int, obj: Objective) -> QWidget:
        container = QWidget()
        lay = QHBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        names = obj.parameter_names
        if not names:
            lay.addWidget(QLabel("—"))
            lay.addStretch()
            return container

        for name, kind in zip(names, obj.parameter_types):
            value = getattr(obj, name)
            lay.addWidget(QLabel(f"{name}:"))
            if kind == "image_reference":
                lay.addWidget(self._build_image_reference_combo(voi_idx, obj_idx, name, value))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                spin = AdaptiveDoubleSpinBox()
                spin.setRange(-1.0e9, 1.0e9)
                spin.setValue(float(value))
                spin.valueChanged.connect(
                    lambda val, v=voi_idx, o=obj_idx, n=name: self._on_param_changed(v, o, n, val)
                )
                lay.addWidget(spin)
            else:
                label = QLabel(str(value))
                lay.addWidget(label)
        lay.addStretch()
        return container

    def _build_image_reference_combo(
        self, voi_idx: int, obj_idx: int, name: str, value: Any
    ) -> QComboBox:
        """Dropdown of workspace result doses usable as an image-reference parameter.

        The current value is matched by identity against the workspace entries;
        a value not present there (set programmatically, or the objective's
        default placeholder) is kept as a summarized "Custom" entry so it stays
        selectable.
        """
        cmb = QComboBox()
        cmb.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        current = -1
        for label, candidate in self._reference_dose_options():
            if self._image_ref_matches(value, candidate):
                current = cmb.count()
            cmb.addItem(label, candidate)
        if current < 0:
            cmb.insertItem(0, self._summarize_image_reference(value), value)
            current = 0
        cmb.setCurrentIndex(current)
        if cmb.count() == 1 and current == 0 and self._ws.result is None:
            cmb.setToolTip("No result doses available yet — compute or import a dose first")
        else:
            cmb.setToolTip("Reference dose image (picked from the workspace results)")
        cmb.currentIndexChanged.connect(
            lambda idx, c=cmb, v=voi_idx, o=obj_idx, n=name: self._on_image_ref_changed(
                v, o, n, c.itemData(idx)
            )
        )
        return cmb

    def _reference_dose_options(self) -> list[tuple[str, Any]]:
        """Workspace result entries convertible to an image-reference value.

        Result images come out of ``Dij.compute_result_ct_grid`` as
        ``sitk.Image`` and are used directly; imported cubes are numpy arrays
        on the CT grid and are paired with the CT grid.  Weight vectors and
        per-beam lists are skipped.
        """
        options: list[tuple[str, Any]] = []
        ct = self._ws.ct
        for key, val in (self._ws.result or {}).items():
            if isinstance(val, sitk.Image):
                options.append((key, val))
            elif isinstance(val, np.ndarray) and val.ndim == 3 and ct is not None:
                options.append((key, (val, ct.grid)))
        return options

    @staticmethod
    def _image_ref_matches(current: Any, candidate: Any) -> bool:
        if current is candidate:
            return True
        if isinstance(current, tuple) and isinstance(candidate, tuple):
            return current[0] is candidate[0]
        return False

    @staticmethod
    def _summarize_image_reference(value: Any) -> str:
        """One-line description of an image-reference value not held in the workspace."""
        if isinstance(value, sitk.Image):
            size = "×".join(str(s) for s in value.GetSize())
            if value.GetNumberOfPixels() == 1:
                return f"Custom image ({size}, value {value.GetPixel(0, 0, 0):g})"
            return f"Custom image ({size})"
        if isinstance(value, tuple) and len(value) == 2:
            size = "×".join(str(s) for s in getattr(value[0], "shape", ()))
            return f"Custom array ({size})"
        return f"Custom ({type(value).__name__})"

    # ------------------------------------------------------------------
    # Edit callbacks (write back through the workspace)
    # ------------------------------------------------------------------

    def _on_add_objective(self) -> None:
        cst = self._ws.cst
        if cst is None or self._cmb_voi.count() == 0:
            return
        voi_idx = self._cmb_voi.currentData()
        voi = cst.vois[voi_idx]
        name = _DEFAULT_OBJECTIVE_BY_TYPE.get(voi.voi_type, _FALLBACK_OBJECTIVE)
        if name not in self._available:
            name = self._available[0]
        new_obj = get_objective(name)

        objectives = self._iter_objectives(voi)
        objectives.append(new_obj)
        self._write_objectives(cst, voi_idx, objectives)

    def _on_ai_objectives(self) -> None:
        cst = self._ws.cst
        pln = self._ws.pln
        if cst is None or pln is None:
            return

        # Deferred: keeps the optional ai_agents stack out of widget construction.
        from pyRadPlan.ai_agents import (  # noqa: PLC0415
            OBJECTIVES_SYSTEM_PROMPT,
            available_models,
            cst_context_summary,
            generate_voi_objectives,
        )
        from pyRadPlan.gui.widgets.ai import AiTask, AiTaskDialog  # noqa: PLC0415

        def _run(model: str, site: str, context: str):
            return generate_voi_objectives(
                pln,
                cst,
                treatment_site=site or "unspecified",
                additional_context=context or None,
                model=model,
            )

        def _apply(new_cst) -> None:
            self._ws.cst = new_cst

        def _summarize(new_cst) -> str:
            total = sum(len(self._iter_objectives(voi)) for voi in new_cst.vois)
            return f"Applied — {total} objective(s) suggested."

        task = AiTask(
            title="Suggest objectives (AI)",
            system_prompt=OBJECTIVES_SYSTEM_PROMPT,
            context_text=json.dumps(cst_context_summary(pln, cst), indent=2, default=str),
            run=_run,
            apply=_apply,
            summarize=_summarize,
        )
        AiTaskDialog(task, available_models(), parent=self).exec()

    def _on_remove_objective(self, voi_idx: int, obj_idx: int) -> None:
        cst = self._ws.cst
        if cst is None:
            return
        objectives = self._iter_objectives(cst.vois[voi_idx])
        if 0 <= obj_idx < len(objectives):
            del objectives[obj_idx]
            self._write_objectives(cst, voi_idx, objectives)

    def _on_objective_changed(self, voi_idx: int, obj_idx: int, name: str) -> None:
        cst = self._ws.cst
        if cst is None:
            return
        objectives = self._iter_objectives(cst.vois[voi_idx])
        if not (0 <= obj_idx < len(objectives)):
            return
        if objectives[obj_idx].name == name:
            return
        priority = objectives[obj_idx].priority
        new_obj = get_objective(name)
        new_obj.priority = priority
        objectives[obj_idx] = new_obj
        self._write_objectives(cst, voi_idx, objectives)

    def _on_penalty_changed(self, voi_idx: int, obj_idx: int, value: float) -> None:
        cst = self._ws.cst
        if cst is None:
            return
        objectives = self._iter_objectives(cst.vois[voi_idx])
        if not (0 <= obj_idx < len(objectives)):
            return
        try:
            objectives[obj_idx].priority = float(value)
        except (ValueError, TypeError):
            self._set_status(f"Ignored invalid penalty {value!r}")
            return
        self._write_objectives(cst, voi_idx, objectives, rebuild=False)

    def _on_quantity_changed(self, voi_idx: int, obj_idx: int, quantity: str) -> None:
        cst = self._ws.cst
        if cst is None:
            return
        objectives = self._iter_objectives(cst.vois[voi_idx])
        if not (0 <= obj_idx < len(objectives)):
            return
        if objectives[obj_idx].quantity == quantity:
            return
        try:
            objectives[obj_idx].quantity = quantity
        except (ValueError, TypeError) as exc:
            self._set_status(f"Rejected quantity {quantity!r}: {exc}")
            return
        self._write_objectives(cst, voi_idx, objectives, rebuild=False)

    def _on_param_changed(self, voi_idx: int, obj_idx: int, param: str, value: float) -> None:
        cst = self._ws.cst
        if cst is None:
            return
        objectives = self._iter_objectives(cst.vois[voi_idx])
        if not (0 <= obj_idx < len(objectives)):
            return
        try:
            setattr(objectives[obj_idx], param, float(value))
        except (ValueError, TypeError):
            self._set_status(f"Ignored invalid value for {param}: {value!r}")
            return
        self._write_objectives(cst, voi_idx, objectives, rebuild=False)

    def _on_image_ref_changed(self, voi_idx: int, obj_idx: int, param: str, value: Any) -> None:
        cst = self._ws.cst
        if cst is None or value is None:
            return
        objectives = self._iter_objectives(cst.vois[voi_idx])
        if not (0 <= obj_idx < len(objectives)):
            return
        try:
            setattr(objectives[obj_idx], param, value)
        except (ValueError, TypeError) as exc:
            self._set_status(f"Rejected reference image for {param}: {exc}")
            return
        self._write_objectives(cst, voi_idx, objectives, rebuild=False)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _write_objectives(
        self, cst, voi_idx: int, objectives: list[Objective], rebuild: bool = True
    ) -> None:
        """Assign *objectives* to the VOI and push the ``cst`` back to the workspace.

        The structure set is mutated in place (VOIs are mutable pydantic models)
        and re-validated to keep it consistent, then re-assigned to the workspace
        inside :meth:`hold_updates` so peer widgets refresh while this widget does
        not react to its own write.
        """
        try:
            cst.vois[voi_idx].objectives = list(objectives)
            cst = validate_cst(cst)
        except (ValueError, TypeError) as exc:
            self._set_status(f"Rejected change: {exc}")
            return

        with self.hold_updates():
            self._ws.cst = cst

        if rebuild:
            self._rebuild_table(cst)

    def _set_status(self, text: str) -> None:
        self._lbl_status.setText(text)
