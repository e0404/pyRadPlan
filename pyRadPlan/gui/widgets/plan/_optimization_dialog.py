"""Configuration popup for the plan's optimization: planning problem and solver."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.core import field_constraints
from pyRadPlan.optimization.problems import PlanningProblem
from pyRadPlan.optimization.solvers import (
    IPOPT_DISABLED_REASON,
    SolverBase,
    get_available_solvers,
)
from .._config_form import ConfigFormWidget

logger = logging.getLogger(__name__)

#: Problem settings the problem form leaves out. The solver has its own selector; the
#: legacy conversion of physical-dose objectives is switched off by the GUI workflow, whose
#: objectives table sets every quantity, unless a plan explicitly asks for it.
_HIDDEN_PROBLEM_FIELDS = frozenset({"solver", "convert_dose_objectives"})


def _unavailable_reason(name: str) -> str:
    if name == "ipopt" and IPOPT_DISABLED_REASON:
        return IPOPT_DISABLED_REASON
    return "it is not installed in this environment"


def _configurable_fields(model: type) -> list[str]:
    names = []
    for name, info in model.model_fields.items():
        meta = field_constraints(info).get("param_meta")
        if meta is None or meta.configurable:
            names.append(name)
    return names


def _solver_defaults(solver_cls: type[SolverBase]) -> dict[str, Any]:
    """Effective defaults of a solver, which most solvers only set in ``__init__``."""
    try:
        solver = solver_cls()
    except Exception as exc:  # noqa: BLE001 - the form then shows the model defaults
        logger.debug("Could not instantiate solver %s for its defaults: %s", solver_cls, exc)
        return {}
    return {
        name: getattr(solver, name)
        for name in _configurable_fields(solver_cls.config_model())
        if hasattr(solver, name)
    }


def _split_solver_spec(spec: Any) -> tuple[Optional[str], dict[str, Any]]:
    """Solver name and configuration of a ``prop_opt["solver"]`` entry."""
    if spec is None:
        return None, {}
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, SolverBase):
        # Hidden parameters such as keyboard cancellation still belong to the
        # configuration and must survive replacing an instance with a dictionary.
        names = spec.config_model().model_fields
        return spec.short_name, {n: getattr(spec, n) for n in names if hasattr(spec, n)}
    if isinstance(spec, Mapping):
        values = dict(spec)
        return values.pop("name", None), values
    return None, {}


class OptimizationConfigDialog(QDialog):
    """Edit the ``pln.prop_opt`` settings of a planning problem, including its solver.

    The upper form edits the problem's own parameters; below, a dropdown selects the
    solver and a form generated from the solver's configuration model edits its settings,
    pre-filled with the solver's effective defaults.

    Parameters
    ----------
    problem_cls:
        The planning problem class being configured.
    initial:
        Current ``prop_opt`` entries, without the ``"problem"`` selector key. Entries the
        dialog does not show are kept as they are.
    parent:
        Optional Qt parent widget.

    Notes
    -----
    The generic solver knobs a problem forwards to its solver (``max_iter``, ``display``)
    are shown in the solver form when the solver has them, and written there; set on the
    problem they would override the solver's value anyway.
    """

    def __init__(
        self,
        problem_cls: type[PlanningProblem],
        initial: Optional[Mapping[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Configure {problem_cls.name}")
        self._problem_cls = problem_cls
        self._solvers = dict(get_available_solvers())

        initial = dict(initial or {})
        self._solver_given = "solver" in initial
        self._original_solver = initial.pop("solver", None)
        name, solver_values = _split_solver_spec(self._original_solver)
        if name is None:
            name = self._default_solver()
        #: Problem-level solver knobs, moved into the solver form where the solver has them.
        self._forwarded = {
            prop: initial.pop(prop)
            for prop in problem_cls.solver_properties
            if initial.get(prop) is not None
        }
        #: Problem entries as given; the form's values are laid over them, so entries it
        #: does not show survive.
        self._problem_initial = initial
        #: Solver settings per solver name, kept while switching between solvers.
        self._solver_values: dict[str, dict[str, Any]] = {name: solver_values}
        self._shown_solvers: set[str] = set()
        self._solver_name = name
        self._solver_form: Optional[ConfigFormWidget] = None

        root = QVBoxLayout(self)

        problem_box = QGroupBox(f"Planning problem: {problem_cls.name}")
        problem_layout = QVBoxLayout(problem_box)
        self._problem_form = ConfigFormWidget(
            problem_cls.config_model(),
            initial=initial,
            exclude=set(_HIDDEN_PROBLEM_FIELDS) | set(problem_cls.solver_properties),
        )
        problem_layout.addWidget(self._problem_form)
        root.addWidget(problem_box)

        solver_box = QGroupBox("Solver")
        solver_layout = QVBoxLayout(solver_box)
        self._cmb_solver = QComboBox()
        self._cmb_solver.setToolTip("Solver used to optimize the planning problem")
        for solver_name, solver_cls in self._solvers.items():
            self._cmb_solver.addItem(solver_name, solver_name)
            tooltip = solver_cls.name
            if getattr(solver_cls, "gpu_compatible", False) is True:
                tooltip += "\nRuns on the GPU backend."
            self._cmb_solver.setItemData(
                self._cmb_solver.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole
            )
        if name not in self._solvers:
            self._cmb_solver.addItem(f"{name} (not available)", name)
            index = self._cmb_solver.count() - 1
            self._cmb_solver.model().item(index).setEnabled(False)
            self._cmb_solver.setItemData(
                index,
                f"The plan's solver, which is unavailable: {_unavailable_reason(name)}.",
                Qt.ItemDataRole.ToolTipRole,
            )
        self._cmb_solver.setCurrentIndex(self._cmb_solver.findData(name))
        self._cmb_solver.currentIndexChanged.connect(self._on_solver_changed)
        solver_layout.addWidget(self._cmb_solver)

        self._lbl_solver = QLabel("")
        self._lbl_solver.setWordWrap(True)
        solver_layout.addWidget(self._lbl_solver)

        self._solver_scroll = QScrollArea()
        self._solver_scroll.setWidgetResizable(True)
        self._solver_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        solver_layout.addWidget(self._solver_scroll, 1)
        root.addWidget(solver_box, 1)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color: red;")
        root.addWidget(self._lbl_status)
        self._connect_form(self._problem_form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._show_solver(name)
        self.resize(480, 600)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def problem_form(self) -> ConfigFormWidget:
        """The form editing the planning problem's own parameters."""
        return self._problem_form

    @property
    def solver_form(self) -> Optional[ConfigFormWidget]:
        """The form editing the selected solver, None while it is unavailable."""
        return self._solver_form

    def solver_name(self) -> str:
        """Short name of the selected solver."""
        return self._solver_name

    def select_solver(self, name: str) -> None:
        """Select a solver by short name."""
        index = self._cmb_solver.findData(name)
        if index < 0:
            raise ValueError(f"Solver '{name}' is not listed")
        self._cmb_solver.setCurrentIndex(index)

    def values(self) -> dict[str, Any]:
        """Return the ``prop_opt`` entries (without ``"problem"``) the dialog describes."""
        values = dict(self._problem_initial)
        values.update(self._problem_form.values())

        name = self._solver_name
        solver_cls = self._solvers.get(name)
        solver_fields = solver_cls.config_model().model_fields if solver_cls else {}
        for prop, value in self._forwarded.items():
            if prop not in solver_fields:
                values[prop] = value

        if solver_cls is None:
            # Only the plan's own, unavailable solver can be selected without a class.
            values["solver"] = self._original_solver
            return values
        solver_values = self._solver_form.values() if self._solver_form is not None else {}
        if solver_values:
            values["solver"] = {"name": name, **solver_values}
        elif self._solver_given or name != self._default_solver():
            values["solver"] = name
        return values

    # ------------------------------------------------------------------
    # Solver selection
    # ------------------------------------------------------------------

    def _default_solver(self) -> str:
        """Return the solver the problem picks when none is configured."""
        default = self._problem_cls.config_model().model_fields["solver"].default
        if isinstance(default, str) and default in self._solvers:
            return default
        return next(iter(self._solvers), default)

    def _on_solver_changed(self, index: int) -> None:
        if self._solver_form is not None:
            self._solver_values[self._solver_name] = self._solver_form.values()
        self._show_solver(self._cmb_solver.itemData(index))

    def _show_solver(self, name: str) -> None:
        self._solver_name = name
        self._lbl_status.setText("")
        solver_cls = self._solvers.get(name)
        if solver_cls is None:
            self._solver_form = None
            self._lbl_solver.setText(
                f"Solver '{name}' is unavailable: {_unavailable_reason(name)}. "
                "Its configuration is kept; choose another solver to replace it."
            )
            self._solver_scroll.setWidget(QWidget())
            return

        self._lbl_solver.setText(solver_cls.name)
        model = solver_cls.config_model()
        initial = dict(self._solver_values.get(name, {}))
        if name not in self._shown_solvers:
            # Afterwards the stored values already hold them, possibly edited.
            self._shown_solvers.add(name)
            for prop, value in self._forwarded.items():
                if prop in model.model_fields:
                    initial[prop] = value
        self._solver_form = ConfigFormWidget(
            model, initial=initial, defaults=_solver_defaults(solver_cls)
        )
        self._connect_form(self._solver_form)
        self._solver_scroll.setWidget(self._solver_form)

    def _connect_form(self, form: ConfigFormWidget) -> None:
        form.validation_failed.connect(
            lambda name, msg: self._lbl_status.setText(f"{name}: {msg}")
        )
        form.value_changed.connect(lambda *_: self._lbl_status.setText(""))
