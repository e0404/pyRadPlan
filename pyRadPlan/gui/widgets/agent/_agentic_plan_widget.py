"""Settings of a planning agent run: model, task, clinical goals and tools."""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.gui.workspace import WorkspaceManager

from .._base import CompactComboBox, WorkspaceWidget
from ..ai import AI_MISSING_TIP, ai_available

#: Tools the agent cannot plan without; always enabled.
REQUIRED_TOOLS = ("set_objectives", "optimize")

DEFAULT_TASK = (
    "Create a clinically acceptable plan. Explore different trade-offs between target "
    "coverage and organ-at-risk sparing."
)

#: Initial values of the run settings, also restored by :meth:`AgenticPlanWidget.reset`.
_DEFAULTS = {
    "plans": 2,
    "optimizations": 8,
    "requests": 40,
    "compact": 0,
    "dij_calcs": 3,
    "min_beams": 1,
    "max_beams": 9,
    "max_iter": 5000,
    "review": 0,
}

_COLUMNS = ("Structure", "Metric", "", "Value", "Hard")
_STRUCTURE, _METRIC, _OP, _VALUE, _HARD = range(len(_COLUMNS))
_GOAL_FILES = "Goal files (*.json);;All files (*)"


class AgenticPlanWidget(WorkspaceWidget):
    """Settings of a planning agent run.

    The goals form the protocol the plans are judged on. Without goals the agent picks
    them from the bundled constraint tables. Unchecking ``set_beams`` keeps the beam
    angles of the *Plan* tab.

    Parameters
    ----------
    workspace:
        Shared workspace; the plan's dose convention sets the unit of the goal doses.
    tools:
        Name and description of every planning tool (see
        :func:`pyRadPlan.ai.agents.planning_tools`).
    models:
        Suggested model names; any other name can be typed.
    """

    _watched_keys = ("pln", "cst")

    def __init__(
        self,
        workspace: Optional[WorkspaceManager] = None,
        tools: Optional[dict[str, str]] = None,
        models: Optional[list[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(workspace, parent)
        # Dose convention the goal values are in; they follow the plan's when it changes
        self._goal_convention: Optional[str] = None
        self._ai_available = ai_available()
        self._setup_ui(tools or {}, models or [])
        self.initialize()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self, tools: dict[str, str], models: list[str]) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        grid = QGridLayout()
        grid.setSpacing(4)
        self._cmb_model = QComboBox()
        self._cmb_model.setEditable(True)
        self._cmb_model.addItems(models)
        self._cmb_model.setToolTip("pydantic-ai model name, e.g. anthropic:claude-sonnet-4-6")
        grid.addWidget(QLabel("Model:"), 0, 0)
        grid.addWidget(self._cmb_model, 0, 1, 1, 7)
        self._spn_plans = self._spin(1, 5, "Number of plans the agent's results are reduced to")
        self._spn_opt = self._spin(1, 50, "Maximum number of optimizations")
        self._spn_requests = self._spin(5, 500, "Maximum number of model requests (cost)")
        self._spn_compact = self._spin(
            0,
            50,
            "Compact the chat history every this many optimizations: older tool results are\n"
            "replaced by short notes (optimization results keep score and goal values), so\n"
            "later requests send fewer tokens.\n\n"
            "Compacting rewrites the history: a provider's prompt cache (e.g. Anthropic's)\n"
            "is rebuilt from the first shortened message on, once per compaction.",
        )
        self._spn_compact.setSpecialValueText("off")
        for col, (label, spin) in enumerate(
            (
                ("Plans:", self._spn_plans),
                ("Optimizations:", self._spn_opt),
                ("Requests:", self._spn_requests),
                ("Compact after:", self._spn_compact),
            )
        ):
            grid.addWidget(QLabel(label), 1, 2 * col)
            grid.addWidget(spin, 1, 2 * col + 1)

        self._spn_dij = self._spin(
            1,
            20,
            "Maximum number of dose influence calculations. Every beam change needs one;\n"
            "with the budget used up the agent can no longer change the beams.",
        )
        self._spn_min_beams = self._spin(1, 30, "Fewest beams the agent may use")
        self._spn_max_beams = self._spin(1, 30, "Most beams the agent may use")
        self._spn_min_beams.valueChanged.connect(self._spn_max_beams.setMinimum)
        self._spn_max_beams.valueChanged.connect(self._spn_min_beams.setMaximum)
        self._spn_min_beams.setMaximum(self._spn_max_beams.value())
        beams = QHBoxLayout()
        beams.setSpacing(2)
        beams.addWidget(self._spn_min_beams)
        beams.addWidget(QLabel("–"))
        beams.addWidget(self._spn_max_beams)
        beams.addSpacing(6)
        self._spn_iter = self._spin(
            50, 5000, "Most optimizer iterations per optimization (computation time)"
        )
        self._spn_iter.setSingleStep(250)
        self._chk_couch = QCheckBox("Non-coplanar")
        self._chk_couch.setChecked(True)
        self._chk_couch.setToolTip("Allow couch angles other than 0")
        beams.addWidget(self._chk_couch)
        self._spn_review = self._spin(
            0,
            50,
            "Pause for your review every this many optimizations, also when all hard goals\n"
            "pass the first time and when the optimization budget is used up. Then you can\n"
            "continue, give feedback, change goals and budgets, or end the run.",
        )
        self._spn_review.setSpecialValueText("off")
        self._apply_defaults()
        grid.addWidget(QLabel("Dose calcs:"), 2, 0)
        grid.addWidget(self._spn_dij, 2, 1)
        grid.addWidget(QLabel("Beams:"), 2, 2)
        grid.addLayout(beams, 2, 3)
        grid.addWidget(QLabel("Max iter.:"), 2, 4)
        grid.addWidget(self._spn_iter, 2, 5)
        grid.addWidget(QLabel("Review every:"), 2, 6)
        grid.addWidget(self._spn_review, 2, 7)
        root.addLayout(grid)

        root.addWidget(QLabel("Task:"))
        self._txt_task = QPlainTextEdit(DEFAULT_TASK)
        self._txt_task.setMaximumHeight(60)
        root.addWidget(self._txt_task)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_goals_box())
        split.addWidget(self._build_tools_box(tools))
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

    @staticmethod
    def _spin(lo: int, hi: int, tip: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setToolTip(tip)
        return spin

    def _apply_defaults(self) -> None:
        for spin, key in (
            (self._spn_plans, "plans"),
            (self._spn_opt, "optimizations"),
            (self._spn_requests, "requests"),
            (self._spn_compact, "compact"),
            (self._spn_dij, "dij_calcs"),
            (self._spn_min_beams, "min_beams"),  # first: the maximum's lower bound follows it
            (self._spn_max_beams, "max_beams"),
            (self._spn_iter, "max_iter"),
            (self._spn_review, "review"),
        ):
            spin.setValue(_DEFAULTS[key])
        self._chk_couch.setChecked(True)

    def _build_goals_box(self) -> QGroupBox:
        box = QGroupBox("Goals")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        self._tbl_goals = QTableWidget(0, len(_COLUMNS))
        self._tbl_goals.setHorizontalHeaderLabels(_COLUMNS)
        self._tbl_goals.verticalHeader().setVisible(False)
        self._tbl_goals.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self._tbl_goals.horizontalHeader()
        header.setSectionResizeMode(_STRUCTURE, QHeaderView.ResizeMode.Stretch)
        for col in (_METRIC, _OP, _VALUE, _HARD):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._tbl_goals.setToolTip(
            "Metric: mean, max, min, D<vol%> (e.g. D95) or V<dose>Gy (e.g. V20Gy).\n"
            "Structure names may be clinical names; the agent maps them to the structures."
        )
        lay.addWidget(self._tbl_goals, 1)

        self._lbl_unit = QLabel()
        self._lbl_unit.setWordWrap(True)
        lay.addWidget(self._lbl_unit)
        self._tbl_goals.model().rowsInserted.connect(lambda *_: self._do_update([]))
        self._tbl_goals.model().rowsRemoved.connect(lambda *_: self._do_update([]))

        row = QHBoxLayout()
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(lambda _checked=False: self._add_goal_row())
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self._remove_selected_goals)
        self._btn_load = QToolButton()
        self._btn_load.setText("From table")
        self._btn_load.setToolTip("Replace the goals by a site of a bundled constraint table")
        self._btn_load.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._btn_load.setMenu(self._build_table_menu())
        self._btn_import = QPushButton("Import…")
        self._btn_import.setToolTip(
            "Replace the goals by a goal file (JSON); its doses are converted to the plan's "
            "dose convention"
        )
        self._btn_import.clicked.connect(lambda _checked=False: self.import_goals())
        self._btn_export = QPushButton("Export…")
        self._btn_export.setToolTip("Save the goals to a goal file (JSON)")
        self._btn_export.clicked.connect(lambda _checked=False: self.export_goals())
        for w in (btn_add, btn_remove, self._btn_load, self._btn_import, self._btn_export):
            row.addWidget(w)
            if not self._ai_available:  # goals are models of the planning agent
                w.setEnabled(False)
                w.setToolTip(AI_MISSING_TIP)
        row.addStretch()
        lay.addLayout(row)
        return box

    def _build_table_menu(self) -> QMenu:
        menu = QMenu(self)
        if not self._ai_available:
            return menu
        from pyRadPlan.ai.agents import (  # noqa: PLC0415
            available_constraint_tables,
            load_constraint_table,
        )

        for table_id, name in available_constraint_tables().items():
            sub = menu.addMenu(name)
            for site in load_constraint_table(table_id).sites:
                action = sub.addAction(site)
                action.triggered.connect(
                    lambda _checked=False, t=table_id, s=site: self._load_site_goals(t, s)
                )
        return menu

    def _build_tools_box(self, tools: dict[str, str]) -> QGroupBox:
        box = QGroupBox("Tools")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 4, 4, 4)
        self._lst_tools = QListWidget()
        for name, description in tools.items():
            item = QListWidgetItem(name)
            item.setToolTip(description)
            item.setCheckState(Qt.CheckState.Checked)
            if name in REQUIRED_TOOLS:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                item.setToolTip(f"Always available.\n\n{description}")
            elif name == "set_beams":
                item.setToolTip(
                    f"Beam angle selection. Unchecked, the angles of the Plan tab are used."
                    f"\n\n{description}"
                )
            self._lst_tools.addItem(item)
        lay.addWidget(self._lst_tools)
        return box

    # ------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------

    def _add_goal_row(
        self,
        voi: str = "",
        metric: str = "max",
        op: str = "<=",
        value: float = 0.0,
        hard: bool = True,
        source: Optional[str] = None,
    ) -> None:
        tbl = self._tbl_goals
        row = tbl.rowCount()
        tbl.insertRow(row)
        structure = QTableWidgetItem(voi)
        structure.setData(Qt.ItemDataRole.UserRole, source)
        tbl.setItem(row, _STRUCTURE, structure)
        tbl.setItem(row, _METRIC, QTableWidgetItem(metric))
        cmb_op = CompactComboBox(2)
        cmb_op.addItems(["<=", ">="])
        cmb_op.setCurrentText(op)
        tbl.setCellWidget(row, _OP, cmb_op)
        tbl.setItem(row, _VALUE, QTableWidgetItem(f"{value:g}"))
        item_hard = QTableWidgetItem()
        item_hard.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item_hard.setCheckState(Qt.CheckState.Checked if hard else Qt.CheckState.Unchecked)
        tbl.setItem(row, _HARD, item_hard)

    def _remove_selected_goals(self) -> None:
        for row in sorted({i.row() for i in self._tbl_goals.selectedIndexes()}, reverse=True):
            self._tbl_goals.removeRow(row)

    def _load_site_goals(self, table_id: str, site: str) -> None:
        from pyRadPlan.ai.agents import load_constraint_table, site_goals  # noqa: PLC0415

        pln = self._ws.pln
        if pln is None:
            QMessageBox.warning(self, "Goals", "Configure a plan first: goals depend on it.")
            return
        goals, notes = site_goals(
            load_constraint_table(table_id),
            site,
            pln.num_of_fractions,
            getattr(pln, "prescribed_dose", None),
            per_fraction=pln.dose_convention == "per_fraction",
        )
        self.set_goals(goals)
        if notes:  # the table's disclaimer and conversion warnings
            QMessageBox.information(self, "Goals", "\n\n".join(notes))

    def import_goals(self, path: Optional[str] = None) -> None:
        """Replace the goals by those of a goal file (asks for *path* if not given)."""
        from pyRadPlan.ai.agents import load_goals  # noqa: PLC0415

        pln = self._ws.pln
        if pln is None:
            QMessageBox.warning(self, "Goals", "Configure a plan first: goals depend on it.")
            return
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Import goals", "", _GOAL_FILES)
            if not path:
                return
        try:
            goals, notes = load_goals(path, pln.dose_convention, pln.num_of_fractions)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Goals", f"Cannot read the goal file:\n{exc}")
            return
        self.set_goals(goals)
        if notes:
            QMessageBox.information(self, "Goals", "\n".join(notes))

    def export_goals(self, path: Optional[str] = None) -> None:
        """Save the goals to a goal file (asks for *path* if not given)."""
        from pyRadPlan.ai.agents import save_goals  # noqa: PLC0415

        pln = self._ws.pln
        if pln is None:
            QMessageBox.warning(self, "Goals", "Configure a plan first: goals depend on it.")
            return
        try:
            goals = self.goals()
        except ValueError as exc:
            QMessageBox.warning(self, "Goals", str(exc))
            return
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export goals", "goals.json", _GOAL_FILES)
            if not path:
                return
        save_goals(path, goals, pln.dose_convention, pln.num_of_fractions)

    def set_goals(self, goals: list) -> None:
        """Replace the goals table by *goals* (:class:`pyRadPlan.ai.agents.Goal`).

        Their doses are taken to be in the plan's dose convention.
        """
        self._tbl_goals.setRowCount(0)
        for g in goals:
            self._add_goal_row(g.voi, g.metric, g.op, g.value, g.hard, g.source)
        if self._ws.pln is not None:
            self._goal_convention = self._ws.pln.dose_convention

    def _row_goal(self, row: int) -> Any:
        from pyRadPlan.ai.agents import Goal  # noqa: PLC0415

        tbl = self._tbl_goals
        return Goal(
            voi=tbl.item(row, _STRUCTURE).text().strip(),
            metric=tbl.item(row, _METRIC).text().strip(),
            op=tbl.cellWidget(row, _OP).currentText(),
            value=float(tbl.item(row, _VALUE).text().replace(",", ".")),
            hard=tbl.item(row, _HARD).checkState() == Qt.CheckState.Checked,
            source=tbl.item(row, _STRUCTURE).data(Qt.ItemDataRole.UserRole),
        )

    def goals(self) -> list:
        """Return the goals of the table; a ``ValueError`` names the first invalid row."""
        goals = []
        for row in range(self._tbl_goals.rowCount()):
            try:
                goals.append(self._row_goal(row))
            except Exception as exc:  # noqa: BLE001 - reported with the row number
                raise ValueError(f"Goal in row {row + 1}: {exc}") from exc
        return goals

    def _follow_dose_convention(self, pln: Any) -> None:
        """Convert the goal doses when the plan's dose convention changed.

        Converted with the plan's number of fractions; a change of the number of
        fractions alone keeps the values. Invalid rows stay as they are.
        """
        before, self._goal_convention = self._goal_convention, pln.dose_convention
        if before is None or before == pln.dose_convention or not self._tbl_goals.rowCount():
            return
        from pyRadPlan.ai.agents._planning._constraints import _scale_goal  # noqa: PLC0415

        factor = (
            1.0 / pln.num_of_fractions
            if pln.dose_convention == "per_fraction"
            else float(pln.num_of_fractions)
        )
        tbl = self._tbl_goals
        for row in range(tbl.rowCount()):
            try:
                goal = _scale_goal(self._row_goal(row), factor)
            except Exception:  # noqa: BLE001 - an invalid row is reported on use
                continue
            tbl.item(row, _METRIC).setText(goal.metric)
            tbl.item(row, _VALUE).setText(f"{goal.value:g}")

    def protocol(self) -> Any:
        """Return the goals as a locked protocol (an empty, unlocked one without goals)."""
        from pyRadPlan.ai.agents import Protocol  # noqa: PLC0415

        goals = self.goals()
        if goals:
            return Protocol(name="GUI goals", goals=goals, locked=True)
        return Protocol(goals=[], locked=False)

    # ------------------------------------------------------------------
    # Run settings
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """The model name."""
        return self._cmb_model.currentText().strip()

    @property
    def task(self) -> str:
        """The task (user prompt)."""
        return self._txt_task.toPlainText().strip()

    @property
    def num_plans(self) -> int:
        """How many plans to keep."""
        return self._spn_plans.value()

    @property
    def max_optimizations(self) -> int:
        """The optimization budget."""
        return self._spn_opt.value()

    @property
    def request_limit(self) -> int:
        """The model request budget."""
        return self._spn_requests.value()

    @property
    def compact_after(self) -> int:
        """Compact the history every this many optimizations (0: never)."""
        return self._spn_compact.value()

    @property
    def max_dij_calcs(self) -> int:
        """The dose influence calculation budget."""
        return self._spn_dij.value()

    @property
    def beam_range(self) -> tuple[int, int]:
        """Fewest and most beams the agent may use."""
        return self._spn_min_beams.value(), self._spn_max_beams.value()

    @property
    def allow_couch(self) -> bool:
        """Whether the agent may use non-coplanar beams."""
        return self._chk_couch.isChecked()

    @property
    def review_every(self) -> int:
        """Pause for a review every this many optimizations (0: never)."""
        return self._spn_review.value()

    @property
    def max_iter(self) -> int:
        """The most optimizer iterations per optimization."""
        return self._spn_iter.value()

    def tools(self) -> list[str]:
        """Return the names of the checked tools."""
        items = (self._lst_tools.item(i) for i in range(self._lst_tools.count()))
        return [i.text() for i in items if i.checkState() == Qt.CheckState.Checked]

    def reset(self) -> None:
        """Reset the goals and run settings to their defaults."""
        self._tbl_goals.setRowCount(0)
        self._txt_task.setPlainText(DEFAULT_TASK)
        if self._cmb_model.count():
            self._cmb_model.setCurrentIndex(0)
        self._apply_defaults()
        for i in range(self._lst_tools.count()):
            self._lst_tools.item(i).setCheckState(Qt.CheckState.Checked)
        self._do_update([])

    def set_running(self, running: bool) -> None:
        """Lock the settings while the agent runs."""
        for w in (
            self._cmb_model,
            self._spn_plans,
            self._spn_opt,
            self._spn_requests,
            self._spn_compact,
            self._spn_dij,
            self._spn_min_beams,
            self._spn_max_beams,
            self._spn_iter,
            self._chk_couch,
            self._spn_review,
            self._tbl_goals,
            self._lst_tools,
            self._btn_load,
            self._btn_import,
            self._btn_export,
        ):
            w.setEnabled(not running)
        self._txt_task.setReadOnly(running)

    def _do_update(self, _changed_keys: list) -> None:
        pln = self._ws.pln
        if pln is not None:
            self._follow_dose_convention(pln)
        if pln is None:
            unit = "Doses in Gy, in the plan's dose convention."
        elif pln.dose_convention == "per_fraction":
            unit = "Doses in Gy per fraction, volumes in %."
        else:
            unit = "Doses in Gy for the whole course, volumes in %."
        empty = (
            "" if self._tbl_goals.rowCount() else " No goals: the agent picks a constraint table."
        )
        self._lbl_unit.setText(unit + empty)
