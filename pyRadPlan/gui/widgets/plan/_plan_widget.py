"""Plan configuration widget for the pyRadPlan GUI."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.biology.models import (
    BiologicalModel,
    BioMachineSummary,
    BioModelOption,
    available_bio_models,
)
from pyRadPlan.dose.engines import get_available_engines
from pyRadPlan._settings import get_settings
from pyRadPlan.gui.widgets._ui_mode import UIMode
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.machines import Machine, get_available_machines
from pyRadPlan.optimization.problems import get_available_problems
from pyRadPlan.optimization.solvers import SolverBase
from pyRadPlan.plan import IonPlan, PhotonPlan, Plan, default_bio_models
from pyRadPlan.scenarios import available_scenario_models
from pyRadPlan.stf import get_beam_angle_presets
from pyRadPlan.biology.tissues import provenance_label
from .._base import WorkspaceWidget, format_number_list, parse_number_list
from .._config_form import ConfigFormDialog
from .._display_names import (
    add_identified_item,
    bio_model_display_name,
    show_identifiers_when_open,
)
from .._tissue_params import (
    SELECTION_PARAMETERS,
    bio_model_quietly,
    bio_options_quietly,
    dose_engine_quietly,
    load_machine_quietly,
    machine_summary,
    matching_bio_option,
    selection_parameters,
    tissue_errors,
    tissue_issues_quietly,
    tissue_issues_text,
)
from ._optimization_dialog import OptimizationConfigDialog
from ._tissue_dialog import TissueParametersDialog
from ..ai import ai_disabled_reason

logger = logging.getLogger(__name__)

_ION_MODES = list(IonPlan.available_radiation_modes)
_RADIATION_MODES = ["photons", *_ION_MODES]

#: Orange border marking a field whose value differs from the applied plan.
_MODIFIED_STYLE = "border: 1px solid #e67e22;"
#: Persistent status note shown while the form has unapplied edits.
_MODIFIED_NOTE = "Modified — not applied"
#: Suffix of the machine dropdown entry for the base data a plan carries itself.
_FROM_PLAN_SUFFIX = "(from plan)"
#: Suffix of the model dropdown entry keeping a plan's model the machine cannot run.
_UNSUPPORTED_SUFFIX = "(not supported by this machine)"
#: Tissue marker while the structures' tissues fail the model's check.
_TISSUE_ERROR_MARK = "⚠"
_TISSUE_ERROR_STYLE = "color: #e67e22; font-weight: bold;"
#: Tissue marker while the tissues resolve with warnings only (a differing model label).
_TISSUE_WARNING_MARK = "ℹ"
_TISSUE_WARNING_STYLE = "color: #b7950b; font-weight: bold;"
#: Tooltip of the beam-count spin box (also restored after an invalid gantry entry).
_NUM_BEAMS_TOOLTIP = (
    "Number of beams.\n"
    "Setting it replaces the gantry angles by that many equally spaced\n"
    "angles starting at 0 deg."
)


def _plan_class(radiation_mode: str) -> type[Plan]:
    return PhotonPlan if radiation_mode == "photons" else IonPlan


class PlanWidget(WorkspaceWidget):
    """Configure the treatment :class:`~pyRadPlan.plan.Plan` for the session.

    Binds to a :class:`~pyRadPlan.gui.workspace.WorkspaceManager` and exposes a
    form to edit the core plan parameters (radiation mode, machine, number of
    fractions, beam geometry and dose-grid resolution).  Pressing *Apply* builds
    a validated plan of the correct subclass (:class:`PhotonPlan` for photons,
    :class:`IonPlan` for ion modalities) and writes it to the workspace.

    Parameters
    ----------
    workspace:
        Shared :class:`WorkspaceManager` instance.  Falls back to the
        process-wide singleton when *None*.
    parent:
        Optional Qt parent widget.
    ui_mode:
        Initial presentation mode, see :class:`UIMode`; *None* follows the
        configured ``pyRadPlan.settings.gui.ui_mode``.

    Notes
    -----
    :meth:`set_ui_mode` / :meth:`ui_mode` switch the form between the full
    editor and a simplified presentation for non-expert users that only offers
    the basic operations (radiation mode, beam angles, isocenter, bixel width,
    biological model and dose-grid resolution); the main window drives them
    from the GUI settings.  The advanced controls are registered by the
    form-building methods when the form is built and stay visible but locked;
    locked fields keep their values and keep taking part in workspace
    synchronisation, dirty tracking and plan construction, so switching modes
    never changes the plan that *Apply* builds.
    """

    _watched_keys = ("ct", "cst", "pln")

    #: Emitted after the presentation mode changed, with the new :class:`UIMode`.
    mode_changed = Signal(object)
    #: Emitted when the form's beam angles change, applied or not: ``(gantry, couch)``
    #: lists with one couch angle per beam, or ``None`` while they do not parse.
    beam_angles_edited = Signal(object)

    def __init__(
        self,
        workspace: Optional[WorkspaceManager] = None,
        parent: Optional[QWidget] = None,
        ui_mode: UIMode | str | None = None,
    ) -> None:
        super().__init__(workspace, parent)
        #: Tooltip explaining why AI features are disabled, or None if usable.
        self._ai_disabled_reason = ai_disabled_reason()
        #: Dose engine configuration values per engine short name, edited via
        #: the [...] popup and written into ``pln.prop_dose_calc`` on Apply.
        self._engine_props: dict[str, dict] = {}
        #: ``pln.prop_opt`` entries per planning problem short name (without the
        #: ``"problem"`` key), edited via the [...] popup.
        self._problem_props: dict[str, dict] = {}
        self._problems: dict[str, type] = {}
        #: Whether the applied plan names its planning problem; Apply then keeps naming it,
        #: while a plan relying on the default problem stays without the key.
        self._plan_names_problem = False
        #: Parameter overrides per biological model name (edited via the "…" button).
        self._bio_params: dict[str, dict] = {}
        self._bio_model_classes: dict[str, type[BiologicalModel]] = {}
        #: Option of each model dropdown entry; None for the entry keeping a plan's model
        #: the machine cannot run (``_unsupported_bio_model``).
        self._bio_options: list[Optional[BioModelOption]] = []
        self._unsupported_bio_model: Optional[BiologicalModel] = None
        #: Machine (base data) the applied plan carries instead of a name, and its summary;
        #: offered as the "<name> (from plan)" machine entry and kept on Apply while selected.
        self._plan_machine: Optional[Machine] = None
        self._plan_machine_summary: Optional[BioMachineSummary] = None
        #: Radiation mode and machine name the model dropdown was built for.
        self._bio_models_key: tuple[str, str] = ("", "")
        self._bio_models_mode: Optional[str] = None
        #: Short name of the dose engine the model dropdown was built for.
        self._bio_models_engine: str = ""
        self._engines: dict[str, type] = {}
        #: True while syncing the form *from* the workspace, so programmatic
        #: widget changes don't register as user edits.
        self._syncing = False
        #: The plan's per-beam iso centers, which the iso field cannot show; kept
        #: on Apply until the iso setting is changed.
        self._per_beam_iso_center: Optional[list] = None
        #: UI field values at the last sync point; the form is "modified" when
        #: the live values differ from this snapshot.
        self._clean_snapshot: dict[str, Any] = {}
        self._modified = False
        #: Whether the status label currently shows the modified note (so real
        #: messages aren't mistaken for it and vice versa).
        self._status_is_modified_note = False
        self._status_is_error = False
        #: Active presentation mode; affects visibility only, never plan data.
        self._ui_mode = UIMode(get_settings().gui.ui_mode if ui_mode is None else ui_mode)
        #: Containers of the controls locked (shown greyed out) in educational
        #: mode, registered by the form-building methods through _add_advanced.
        self._advanced_items: list[QWidget] = []
        self._setup_ui()
        #: Maps each tracked field key to the editor widget to border.
        self._field_widgets: dict[str, QWidget] = {
            "radiation": self._cmb_radiation,
            "machine": self._cmb_machine,
            "engine": self._cmb_engine,
            "engine_props": self._btn_engine_config,
            "problem": self._cmb_problem,
            "problem_props": self._btn_problem_config,
            "bio_model": self._cmb_bio_model,
            "bio_params": self._btn_bio_config,
            "fractions": self._spn_fractions,
            "dose_convention": self._cmb_dose_convention,
            "gantry": self._txt_gantry,
            "couch": self._txt_couch,
            "bixel": self._spn_bixel,
            "iso_auto": self._txt_iso,
            "iso_per_beam": self._txt_iso,
            "iso": self._txt_iso,
            "scenario": self._cmb_scenario,
            "res_x": self._spn_res_x,
            "res_y": self._spn_res_y,
            "res_z": self._spn_res_z,
        }
        self._connect_dirty_signals()
        self._apply_ui_mode()
        self.initialize()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self._build_fields(grid)
        root.addLayout(grid)

        bottom = QHBoxLayout()
        self._lbl_status = QLabel("")
        bottom.addWidget(self._lbl_status)
        bottom.addStretch()
        self._btn_discard = QPushButton("Discard")
        self._btn_discard.setToolTip("Reset the form to the applied plan")
        self._btn_discard.setEnabled(False)
        self._btn_discard.clicked.connect(self._on_discard)
        bottom.addWidget(self._btn_discard)
        self._btn_apply = QPushButton("Apply")
        self._btn_apply.clicked.connect(self._on_apply)
        bottom.addWidget(self._btn_apply)
        root.addLayout(bottom)

        root.addStretch()

        self._refresh_machines(self._cmb_radiation.currentText())
        self._refresh_engines(self._cmb_radiation.currentText())
        self._refresh_problems(self._cmb_radiation.currentText())
        self._refresh_bio_models(self._cmb_radiation.currentText())
        self._refresh_angle_presets(self._cmb_radiation.currentText())
        self._update_beam_count()

    def _build_fields(self, grid: QGridLayout) -> None:
        # Row layout mirrors matRad's PlanWidget, with modality and machine as
        # the first options to set.
        self._build_setup_rows(grid)
        self._build_geometry_rows(grid)
        self._build_model_rows(grid)
        self._build_optimization_row(grid)
        self._build_sequencing_row(grid)
        self._build_dose_grid_row(grid)

    def _build_setup_rows(self, grid: QGridLayout) -> None:
        self._cmb_radiation = QComboBox()
        self._cmb_radiation.addItems(_RADIATION_MODES)
        self._cmb_radiation.currentTextChanged.connect(self._on_radiation_changed)

        self._cmb_machine = QComboBox()
        self._cmb_machine.setEditable(True)
        # Reading a machine's base data takes a while, so the models follow a finished
        # edit or a picked entry rather than every keystroke.
        self._cmb_machine.activated.connect(self._on_machine_changed)
        self._cmb_machine.lineEdit().editingFinished.connect(self._on_machine_changed)

        self._engine_row = self._build_engine_selector()

        self._spn_fractions = QSpinBox()
        self._spn_fractions.setRange(1, 1000)
        self._spn_fractions.setValue(30)

        self._cmb_dose_convention = QComboBox()
        self._cmb_dose_convention.addItem("per fraction", "per_fraction")
        self._cmb_dose_convention.addItem("total", "total")
        # Total-course doses keep objectives meaningful for fraction-size dependent
        # biological models (ions); per fraction is the plan model's default.
        self._cmb_dose_convention.setCurrentIndex(self._cmb_dose_convention.findData("total"))
        self._cmb_dose_convention.setToolTip(
            "Convention for dose values in objectives and results:\n"
            "per fraction - objective doses and reported doses refer to one fraction;\n"
            "total - objective doses are total-course doses (scaled by 1/fractions for "
            "optimization) and reported doses are multiplied by the number of fractions."
        )
        fractions_row = QHBoxLayout()
        fractions_row.setContentsMargins(0, 0, 0, 0)
        fractions_row.addWidget(self._spn_fractions, 1)
        fractions_row.addWidget(QLabel("Doses:"))
        fractions_row.addWidget(self._cmb_dose_convention, 1)

        grid.addWidget(QLabel("Radiation mode:"), 0, 0)
        grid.addWidget(self._cmb_radiation, 0, 1)
        self._add_advanced(grid, QLabel("Machine:"), 0, 2)
        self._add_advanced(grid, self._cmb_machine, 0, 3)

        self._add_advanced(grid, QLabel("Dose engine:"), 1, 0)
        self._add_advanced(grid, self._engine_row, 1, 1)
        self._add_advanced(grid, QLabel("Fractions:"), 1, 2)
        self._add_advanced(grid, fractions_row, 1, 3)

    def _build_geometry_rows(self, grid: QGridLayout) -> None:
        self._txt_gantry = QLineEdit("0")
        self._txt_gantry.setToolTip(
            "Gantry angles in the matRad coordinate system.\n"
            "Every gantry angle defines a beam; separate angles by blanks."
        )
        self._txt_gantry.textChanged.connect(self._update_beam_count)
        self._txt_gantry.editingFinished.connect(self._sync_couch_angles)
        self._txt_couch = QLineEdit("0")
        self._txt_couch.setToolTip(
            "Couch angles in the matRad coordinate system.\n"
            "Every couch angle belongs to a gantry angle; separate angles by blanks.\n"
            "The field follows the gantry angles: a single couch angle is repeated for\n"
            "added beams, otherwise an added beam gets a couch angle of 0."
        )
        self._spn_num_beams = QSpinBox()
        self._spn_num_beams.setRange(1, 360)
        self._spn_num_beams.setSuffix(" beams")
        self._spn_num_beams.setToolTip(_NUM_BEAMS_TOOLTIP)
        # Commit on focus-out/Enter only, so typing "16" does not first generate
        # a single beam for the intermediate "1".
        self._spn_num_beams.setKeyboardTracking(False)
        self._spn_num_beams.valueChanged.connect(self._on_num_beams_changed)

        self._btn_angle_preset = QToolButton()
        self._btn_angle_preset.setText("Choose preset…")
        self._btn_angle_preset.setFocusPolicy(Qt.StrongFocus)
        self._btn_angle_preset.setPopupMode(QToolButton.InstantPopup)
        self._btn_angle_preset.setToolTip(
            "Set gantry and couch angles from a preset;\n"
            "presets typical for the radiation mode are listed first."
        )
        preset_menu = QMenu(self._btn_angle_preset)
        preset_menu.setToolTipsVisible(True)
        preset_menu.triggered.connect(self._on_angle_preset_activated)
        self._btn_angle_preset.setMenu(preset_menu)

        self._btn_ai_beams = QPushButton("✨ AI")
        self._btn_ai_beams.clicked.connect(self._on_ai_beam_angles)
        self._btn_ai_beams.setEnabled(self._ai_disabled_reason is None)
        self._btn_ai_beams.setToolTip(
            self._ai_disabled_reason or "Suggest gantry/couch angles using an LLM"
        )

        # Both angle fields share the full row width, with the beam count and the
        # AI button pushed to the right so neither list of angles is squeezed.
        beam_row = QHBoxLayout()
        beam_row.setContentsMargins(0, 0, 0, 0)
        beam_row.setSpacing(4)
        beam_row.addWidget(self._txt_gantry, 1)
        beam_row.addWidget(QLabel("Couch angles [deg]:"))
        beam_row.addWidget(self._txt_couch, 1)
        beam_row.addWidget(self._spn_num_beams)
        beam_row.addWidget(self._btn_angle_preset)
        beam_row.addWidget(self._btn_ai_beams)

        self._spn_bixel = QDoubleSpinBox()
        self._spn_bixel.setRange(0.1, 100.0)
        self._spn_bixel.setDecimals(2)
        self._spn_bixel.setValue(5.0)
        self._spn_bixel.setToolTip(
            "Photons: width (and height) of quadratic photon bixel\n"
            "Particles: lateral spot distance"
        )
        self._lbl_bixel = QLabel("Bixel width / spot spacing [mm]:")

        self._txt_iso = QLineEdit("0 0 0")
        self._txt_iso.setEnabled(False)
        self._chk_iso_auto = QCheckBox("Auto.")
        self._chk_iso_auto.setChecked(True)
        self._chk_iso_auto.setToolTip(
            "If checked, the isocenter is computed automatically from the target structures"
        )
        self._chk_iso_auto.toggled.connect(self._on_iso_auto_toggled)

        iso_row = QHBoxLayout()
        iso_row.setContentsMargins(0, 0, 0, 0)
        iso_row.setSpacing(4)
        iso_row.addWidget(self._txt_iso, 1)
        iso_row.addWidget(self._chk_iso_auto)

        grid.addWidget(QLabel("Gantry angles [deg]:"), 2, 0)
        grid.addLayout(beam_row, 2, 1, 1, 3)

        grid.addWidget(self._lbl_bixel, 3, 0)
        grid.addWidget(self._spn_bixel, 3, 1)
        grid.addWidget(QLabel("Iso center [mm]:"), 3, 2)
        grid.addLayout(iso_row, 3, 3)

    def _build_model_rows(self, grid: QGridLayout) -> None:
        self._cmb_bio_model = QComboBox()
        self._cmb_bio_model.setToolTip(
            "Biological model used in dose calculation and optimization, listed as "
            "formalism · base data model.\n"
            "The entries follow the radiation mode and the base data of the machine; "
            "those the machine cannot run are shown disabled, their tooltip says why."
        )
        self._btn_bio_config = QPushButton("…")
        self._btn_bio_config.setFixedWidth(28)
        self._btn_bio_config.clicked.connect(self._on_bio_config)
        show_identifiers_when_open(self._cmb_bio_model)
        # Connected before the dirty tracking, which reads the parameters set here.
        self._cmb_bio_model.currentIndexChanged.connect(self._on_bio_option_changed)
        self._cmb_bio_model.currentIndexChanged.connect(self._update_bio_config_button)
        self._cmb_bio_model.currentIndexChanged.connect(self._update_tissue_indicator)
        bio_row = QHBoxLayout()
        bio_row.setContentsMargins(0, 0, 0, 0)
        bio_row.setSpacing(4)
        bio_row.addWidget(self._cmb_bio_model, 1)
        bio_row.addWidget(self._btn_bio_config)

        self._cmb_scenario = QComboBox()
        self._cmb_scenario.addItems(available_scenario_models())
        for model in ("wcScen", "impScen", "rndScen"):
            self._add_disabled_item(self._cmb_scenario, model)
        self._cmb_scenario.setToolTip("Scenario sampling model for uncertainty handling")

        # Plan-level target/reporting quantity, as opposed to the quantities the
        # individual objectives are evaluated on.
        self._cmb_quantity = QComboBox()
        self._cmb_quantity.addItem("auto")
        self._set_not_implemented(
            self._cmb_quantity,
            "Choosing a global optimized quantity",
            hint="The optimizer derives the quantity from the objectives table, "
            'where "Change all quantities…" sets it on all objectives at once.',
        )

        self._btn_tissue = QPushButton("Set tissue parameters")
        self._btn_tissue.setToolTip(
            "Edit the tissue properties of the loaded structures: the reference photon "
            "alpha_x / beta_x and, per RBE model, its own parameters"
        )
        self._btn_tissue.setEnabled(False)
        self._btn_tissue.clicked.connect(self._on_tissue_parameters)
        #: Shown while the structures' tissues do not resolve for the selected model (⚠),
        #: or resolve with warnings only (ℹ), see :attr:`tissue_marker_state`.
        self._lbl_tissue_warning = QLabel(_TISSUE_ERROR_MARK)
        self._lbl_tissue_warning.setStyleSheet(_TISSUE_ERROR_STYLE)
        self._lbl_tissue_warning.setVisible(False)
        self.tissue_marker_state = ""
        tissue_row = QHBoxLayout()
        tissue_row.setContentsMargins(0, 0, 0, 0)
        tissue_row.setSpacing(4)
        tissue_row.addWidget(self._lbl_tissue_warning)
        tissue_row.addWidget(self._btn_tissue, 1)

        grid.addWidget(QLabel("Biological model:"), 4, 0)
        grid.addLayout(bio_row, 4, 1)
        self._add_advanced(grid, QLabel("Scenario model:"), 4, 2)
        self._add_advanced(grid, self._cmb_scenario, 4, 3)

        self._add_advanced(grid, QLabel("Optimized quantity:"), 5, 0)
        self._add_advanced(grid, self._cmb_quantity, 5, 1)
        grid.addLayout(tissue_row, 5, 3)

    def _build_optimization_row(self, grid: QGridLayout) -> None:
        self._cmb_problem = QComboBox()
        self._cmb_problem.setToolTip("Planning problem formulation used by the optimization")
        self._btn_problem_config = QPushButton("…")
        self._btn_problem_config.setFixedWidth(28)
        self._btn_problem_config.setToolTip(
            "Configure the selected planning problem and choose its solver"
        )
        self._btn_problem_config.clicked.connect(self._on_problem_config)

        problem_row = QHBoxLayout()
        problem_row.setContentsMargins(0, 0, 0, 0)
        problem_row.setSpacing(4)
        problem_row.addWidget(self._cmb_problem, 1)
        problem_row.addWidget(self._btn_problem_config)

        self._add_advanced(grid, QLabel("Optimization:"), 6, 0)
        self._add_advanced(grid, problem_row, 6, 1)

    def _build_sequencing_row(self, grid: QGridLayout) -> None:
        self._chk_sequencing = QCheckBox("Run sequencing")
        self._spn_seq_levels = QSpinBox()
        self._spn_seq_levels.setRange(1, 100)
        self._spn_seq_levels.setValue(7)
        self._chk_dao = QCheckBox("Run DAO")
        self._chk_conf3d = QCheckBox("3D conformal")
        for widget in (self._chk_sequencing, self._spn_seq_levels, self._chk_dao):
            self._set_not_implemented(widget, "Sequencing / aperture optimization")
        self._set_not_implemented(self._chk_conf3d, "3D conformal planning")

        seq_row = QHBoxLayout()
        seq_row.setContentsMargins(0, 0, 0, 0)
        seq_row.setSpacing(4)
        seq_row.addWidget(self._chk_sequencing)
        seq_row.addWidget(QLabel("Levels:"))
        seq_row.addWidget(self._spn_seq_levels)

        opt_row = QHBoxLayout()
        opt_row.setContentsMargins(0, 0, 0, 0)
        opt_row.setSpacing(4)
        opt_row.addWidget(self._chk_dao)
        opt_row.addWidget(self._chk_conf3d)

        self._add_advanced(grid, seq_row, 7, 0, 1, 2)
        self._add_advanced(grid, opt_row, 7, 2, 1, 2)

    def _build_dose_grid_row(self, grid: QGridLayout) -> None:
        self._spn_res_x = self._make_res_spin()
        self._spn_res_y = self._make_res_spin()
        self._spn_res_z = self._make_res_spin()
        self._btn_ct_grid = QPushButton("Use CT grid")
        self._btn_ct_grid.clicked.connect(self._on_use_ct_grid)

        res_row = QHBoxLayout()
        res_row.setContentsMargins(0, 0, 0, 0)
        res_row.setSpacing(4)
        for label, spin in (
            ("x:", self._spn_res_x),
            ("y:", self._spn_res_y),
            ("z:", self._spn_res_z),
        ):
            res_row.addWidget(QLabel(label))
            res_row.addWidget(spin)
        res_row.addWidget(self._btn_ct_grid)

        grid.addWidget(QLabel("Dose grid res. [mm]:"), 8, 0)
        grid.addLayout(res_row, 8, 1, 1, 3)

    def _add_advanced(
        self,
        grid: QGridLayout,
        item: QWidget | QLayout,
        row: int,
        col: int,
        row_span: int = 1,
        col_span: int = 1,
    ) -> None:
        """Place *item* in the form grid and mark it locked in educational mode.

        The item is wrapped in a container widget that is disabled as a whole,
        so the item's own enabled state (e.g. not-implemented controls) is
        preserved when the lock is lifted again.
        """
        container = QWidget()
        if isinstance(item, QLayout):
            container.setLayout(item)
        else:
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(item)
        grid.addWidget(container, row, col, row_span, col_span)
        self._advanced_items.append(container)

    @staticmethod
    def _set_not_implemented(widget: QWidget, feature: str, hint: str = "") -> None:
        widget.setEnabled(False)
        tooltip = f"{feature} not yet implemented."
        if hint:
            tooltip += f"\n{hint}"
        widget.setToolTip(tooltip)

    @staticmethod
    def _add_disabled_item(combo: QComboBox, text: str) -> None:
        combo.addItem(text)
        item = combo.model().item(combo.count() - 1)
        item.setEnabled(False)
        item.setToolTip("Not yet implemented")

    def _build_engine_selector(self) -> QHBoxLayout:
        self._cmb_engine = QComboBox()
        self._cmb_engine.setToolTip(
            "Dose engine. Which biological models can be computed depends on it: Monte "
            "Carlo engines do not use the machine's pencil-beam kernels."
        )
        self._cmb_engine.currentTextChanged.connect(self._on_engine_changed)
        self._btn_engine_config = QPushButton("…")
        self._btn_engine_config.setFixedWidth(28)
        self._btn_engine_config.setToolTip("Configure the selected dose engine")
        self._btn_engine_config.clicked.connect(self._on_engine_config)

        engine_row = QHBoxLayout()
        engine_row.setContentsMargins(0, 0, 0, 0)
        engine_row.setSpacing(4)
        engine_row.addWidget(self._cmb_engine, 1)
        engine_row.addWidget(self._btn_engine_config)
        return engine_row

    def _make_res_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.1, 50.0)
        spin.setDecimals(2)
        spin.setValue(3.0)
        return spin

    # ------------------------------------------------------------------
    # Presentation mode (default / educational)
    # ------------------------------------------------------------------

    def ui_mode(self) -> UIMode:
        """Return the currently active presentation mode."""
        return self._ui_mode

    def set_ui_mode(self, mode: UIMode | str) -> None:
        """Switch between the default and the educational presentation.

        Parameters
        ----------
        mode:
            :attr:`UIMode.DEFAULT` (``"default"``) shows every plan field;
            :attr:`UIMode.EDUCATIONAL` (``"educational"``) restricts editing
            to the basic operations and shows the advanced fields greyed out.
            Locked advanced fields keep their values, so switching modes never
            changes the plan that *Apply* builds.  Emits :attr:`mode_changed`
            when the mode actually changed; no-op otherwise.
        """
        mode = UIMode(mode)
        if mode is self._ui_mode:
            return
        self._ui_mode = mode
        self._apply_ui_mode()
        self.mode_changed.emit(mode)

    def _apply_ui_mode(self) -> None:
        """Apply the visibility policy of the active mode to the form.

        Pure presentation: locked fields keep their values and keep taking
        part in synchronisation, dirty tracking and plan construction.
        """
        educational = self._ui_mode is UIMode.EDUCATIONAL
        for container in self._advanced_items:
            container.setEnabled(not educational)

    # ------------------------------------------------------------------
    # Machine / mode handling
    # ------------------------------------------------------------------

    def refresh_base_data(self) -> None:
        """Re-list machines and biological models, e.g. after base data was added or removed."""
        radiation_mode = self._cmb_radiation.currentText()
        self._refresh_machines(radiation_mode)
        note = self._refresh_bio_models(radiation_mode)
        self._update_dirty_state()
        self._show_bio_model_note(note)

    def _refresh_machines(self, radiation_mode: str) -> None:
        """List the machine files found for ``radiation_mode``; keep or default the selection."""
        current = self._cmb_machine.currentText()
        names = get_available_machines(radiation_mode).get(radiation_mode, [])
        plan_label = self._plan_machine_label(radiation_mode)
        if plan_label is not None:
            names = [plan_label, *names]
        self._cmb_machine.blockSignals(True)
        self._cmb_machine.clear()
        self._cmb_machine.addItems(names)
        if plan_label is not None:
            self._cmb_machine.setItemData(
                0,
                "The base data the applied plan carries itself, used instead of a machine file.",
                Qt.ItemDataRole.ToolTipRole,
            )
        self._cmb_machine.blockSignals(False)

        # The box stays editable, so a machine outside the search paths can still be typed.
        if current in names:
            selected = current
        elif "Generic" in names:
            selected = "Generic"
        elif names:
            selected = names[0]
        else:
            selected = current or "Generic"
        self._cmb_machine.setEditText(selected)

    def _plan_machine_label(self, radiation_mode: Optional[str] = None) -> Optional[str]:
        """Machine dropdown entry of the plan's own base data, None without one for the mode."""
        machine = self._plan_machine
        mode = self._cmb_radiation.currentText() if radiation_mode is None else radiation_mode
        if machine is None or machine.radiation_mode != mode:
            return None
        return f"{machine.name} {_FROM_PLAN_SUFFIX}"

    def _form_machine(self) -> str | Machine:
        """Machine entered in the form: a name, or the plan's own base data if selected."""
        text = self._cmb_machine.currentText() or "Generic"
        if self._plan_machine is not None and text == self._plan_machine_label():
            return self._plan_machine
        return text

    def _form_machine_name(self) -> str:
        """Name of the machine entered in the form."""
        machine = self._form_machine()
        return machine if isinstance(machine, str) else machine.name

    def _on_radiation_changed(self, radiation_mode: str) -> None:
        self._refresh_machines(radiation_mode)
        self._refresh_engines(radiation_mode)
        self._refresh_problems(radiation_mode)
        note = self._refresh_bio_models(radiation_mode)
        self._refresh_angle_presets(radiation_mode)
        self._update_dirty_state()
        self._show_bio_model_note(note)

    # ------------------------------------------------------------------
    # Biological model handling
    # ------------------------------------------------------------------

    def _machine_key(self, radiation_mode: Optional[str] = None) -> tuple[str, str]:
        """Radiation mode and machine name entered in the form."""
        mode = self._cmb_radiation.currentText() if radiation_mode is None else radiation_mode
        return mode, self._cmb_machine.currentText() or "Generic"

    def _on_machine_changed(self, *_args) -> None:
        """List the biological models of a machine once its name is entered or picked."""
        if self._machine_key() == self._bio_models_key:
            return
        note = self._refresh_bio_models(self._cmb_radiation.currentText())
        self._update_dirty_state()
        self._show_bio_model_note(note)

    def _on_engine_changed(self, *_args) -> None:
        """List the biological models the newly selected dose engine can compute."""
        if self._syncing:
            return
        engine = self._current_engine()
        if (engine.short_name if engine is not None else "") != self._bio_models_engine:
            note = self._refresh_bio_models(self._cmb_radiation.currentText())
        else:
            note = None
        self._update_dirty_state()
        self._show_bio_model_note(note)

    def _show_bio_model_note(self, note: Optional[tuple[str, str]]) -> None:
        """Show why the selected biological model was replaced, if it was."""
        if note is None:
            return
        text, reason = note
        self._set_status(text)
        self._lbl_status.setToolTip(reason)

    def _refresh_bio_models(
        self, radiation_mode: str, restore: Any = None
    ) -> Optional[tuple[str, str]]:
        """
        List the biological models the selected machine and engine offer for ``radiation_mode``.

        One entry per option of :func:`~pyRadPlan.biology.models.bio_model_options`, the ones
        the machine or the dose engine cannot run disabled; a machine that cannot be loaded
        gets the plain formalism list (still judged by the engine).

        Parameters
        ----------
        radiation_mode : str
            The radiation mode to list the models of.
        restore : BiologicalModel, optional
            Model of a plan to select. Without an available entry matching it, an entry
            naming it as not supported by the machine (or the engine) is added and
            selected, so that Apply keeps it. Without ``restore``, a selection the new
            context can run is kept (or another entry of its formalism); otherwise, and for
            a mode's default followed into a new mode, the default of the mode is selected:
            its first available entry, else ``none``.

        Returns
        -------
        tuple[str, str] or None
            Status note and its detailed reason if a selection the new context cannot run
            was replaced, otherwise None.
        """
        had_entries = self._cmb_bio_model.count() > 0
        previous = self._selected_bio_model() if had_entries else None
        previous_name = self._bio_model_name()
        previous_label = self._cmb_bio_model.currentText().split(" (not supported")[0]
        previous_mode = self._bio_models_mode
        explicit = had_entries and previous_name not in (
            "",
            default_bio_models.get(previous_mode),
        )
        # A mode's default follows into the new mode's default; anything else is kept if the
        # new context can run it.
        keep = had_entries and (explicit or previous_mode == radiation_mode)
        self._bio_models_mode = radiation_mode
        self._bio_models_key = self._machine_key(radiation_mode)
        engine = self._current_engine()
        self._bio_models_engine = engine.short_name if engine is not None else ""
        machine = self._current_machine()
        options = bio_options_quietly(radiation_mode, machine, engine)
        self._bio_model_classes = {cls.model: cls for cls in available_bio_models(radiation_mode)}

        self._cmb_bio_model.blockSignals(True)
        self._cmb_bio_model.clear()
        self._bio_options = []
        self._unsupported_bio_model = None
        for option in options:
            self._add_bio_option(option)

        note = None
        if restore is not None:
            option = matching_bio_option(options, restore, machine)
            index = self._option_index(option) if option is not None and option.available else -1
            if index < 0:
                index = self._add_unsupported_bio_model(restore, machine, engine)
            self._cmb_bio_model.setCurrentIndex(index)
        else:
            index = -1
            if keep and previous_name in self._bio_model_classes:
                option = matching_bio_option(options, previous, machine)
                if option is not None and option.available:
                    index = self._option_index(option)
            if index >= 0:
                self._cmb_bio_model.setCurrentIndex(index)
            else:
                if keep:
                    index = self._first_option_index(previous_name, available_only=True)
                if index < 0:
                    default = default_bio_models.get(radiation_mode, "none")
                    index = self._first_option_index(default, available_only=True)
                if index < 0:
                    index = self._first_option_index("none")
                self._cmb_bio_model.setCurrentIndex(max(index, 0))
                self._apply_option_parameters(self._cmb_bio_model.currentIndex())
                if keep:
                    note = self._replacement_note(
                        previous, previous_label, radiation_mode, machine, engine
                    )
        self._cmb_bio_model.blockSignals(False)
        self._update_bio_config_button()
        self._update_tissue_indicator()
        return note

    def _replacement_note(
        self, previous: Any, previous_label: str, radiation_mode: str, machine: Any, engine: Any
    ) -> tuple[str, str]:
        """Status note and reason for a selection the new context cannot run."""
        current = self._cmb_bio_model.currentText()
        model = bio_model_quietly(previous, radiation_mode)
        reason = ""
        if model is None or model.model not in self._bio_model_classes:
            context = f"for {radiation_mode}"
        else:
            issues = [
                issue
                for issue in tissue_issues_quietly(machine, model, [], engine)
                if issue.structure is None
            ]
            reason = tissue_issues_text(issues)
            on_machine = not any(
                issue.structure is None for issue in tissue_issues_quietly(machine, model, [])
            )
            if engine is not None and on_machine:
                context = f"with the dose engine {engine.short_name}"
            else:
                context = f"on machine '{self._bio_models_key[1]}'"
        return (
            f"Biological model '{previous_label}' is not available {context}; "
            f"selected '{current}'.",
            reason,
        )

    def _add_bio_option(self, option: BioModelOption) -> None:
        """Append a model dropdown entry for one option, disabled if it is unavailable."""
        add_identified_item(self._cmb_bio_model, option.label, option.model, option.model)
        index = self._cmb_bio_model.count() - 1
        tooltip = option.description or option.label
        if not option.available:
            self._cmb_bio_model.model().item(index).setEnabled(False)
            tooltip += f"\nNot available: {option.reason or 'the machine cannot run it.'}"
        self._cmb_bio_model.setItemData(index, tooltip, Qt.ItemDataRole.ToolTipRole)
        self._bio_options.append(option)

    def _add_unsupported_bio_model(self, bio_model: Any, machine: Any, engine: Any = None) -> int:
        """
        Append a disabled entry keeping a plan's model the machine or engine cannot run.

        Only used when restoring a plan, so that Apply keeps its model. Returns its index,
        -1 if the model cannot be built at all.
        """
        model = bio_model_quietly(bio_model, self._bio_models_mode or "")
        if model is None:
            return -1
        suffix = []
        for key, value in selection_parameters(model.parameters).items():
            if key == "data_model":
                suffix.append(provenance_label(value))
            elif isinstance(value, dict):
                suffix.extend(str(variant) for variant in value.values())
            else:
                suffix.append(str(value))
        label = " · ".join([bio_model_display_name(model), *suffix])
        issues = [
            issue
            for issue in tissue_issues_quietly(machine, model, [], engine)
            if issue.structure is None
        ]
        on_machine = not any(
            issue.structure is None for issue in tissue_issues_quietly(machine, model, [])
        )
        if engine is not None and issues and on_machine:
            subject = f"the dose engine {engine.short_name}"
            unsupported = f"(not supported by {subject})"
        else:
            subject, unsupported = "the machine", _UNSUPPORTED_SUFFIX
        add_identified_item(
            self._cmb_bio_model, f"{label} {unsupported}", model.model, model.model
        )
        index = self._cmb_bio_model.count() - 1
        self._cmb_bio_model.model().item(index).setEnabled(False)
        reason = (
            tissue_issues_text(issues)
            if issues
            else "The machine offers no base data matching the configuration of this model."
        )
        self._cmb_bio_model.setItemData(
            index,
            f"Model of the plan, which {subject} cannot run:\n{reason}\n"
            "Apply keeps it; select another model to replace it.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._bio_options.append(None)
        self._unsupported_bio_model = model
        return index

    def _option_at(self, index: int) -> Optional[BioModelOption]:
        """Option of a model dropdown entry, None for none or the unsupported plan model."""
        return self._bio_options[index] if 0 <= index < len(self._bio_options) else None

    def _option_index(self, option: Optional[BioModelOption]) -> int:
        """Model dropdown entry of an option, -1 if it is not listed."""
        return next((i for i, o in enumerate(self._bio_options) if o is option), -1)

    def _first_option_index(self, name: str, available_only: bool = False) -> int:
        """
        First entry of a formalism (identifier or alias), available ones preferred.

        With ``available_only``, -1 if the formalism has no available entry.
        """
        entries = [i for i, o in enumerate(self._bio_options) if o is not None and o.model == name]
        if not entries:
            model = bio_model_quietly(name, self._bio_models_mode or "")
            if model is not None and model.model != name:
                return self._first_option_index(model.model, available_only)
            return -1
        fallback = -1 if available_only else entries[0]
        return next((i for i in entries if self._bio_options[i].available), fallback)

    def _apply_option_parameters(self, index: int) -> None:
        """Take over the base data parameters of the option of an entry into the model's."""
        option = self._option_at(index)
        if option is None:
            return
        params = {
            key: value
            for key, value in self._bio_params.get(option.model, {}).items()
            if key not in SELECTION_PARAMETERS
        }
        params.update(deepcopy(dict(option.parameters)))
        self._bio_params[option.model] = params

    def _on_bio_option_changed(self, index: int) -> None:
        if not self._syncing:
            self._apply_option_parameters(index)

    def _update_bio_config_button(self, *_args) -> None:
        """
        Enable the biological model's configure button only while it has something to edit.

        A model without constructor parameters (e.g. ``none``, the photon default) has
        nothing to configure. The button then says so rather than being greyed out
        without a reason, which reads as there being no such button.
        """
        cls = self._bio_model_classes.get(self._bio_model_name())
        configurable = cls is not None and bool(cls.config_model().model_fields)
        self._btn_bio_config.setEnabled(configurable)
        self._btn_bio_config.setToolTip(
            "Configure the selected biological model"
            if configurable
            else f"The biological model '{self._cmb_bio_model.currentText()}' "
            "has no parameters to configure"
        )

    def _bio_model_name(self) -> str:
        """Canonical ``model`` name of the selected biological model (``""`` if none)."""
        return self._cmb_bio_model.currentData() or ""

    def _select_bio_model(self, model: Any) -> bool:
        """
        Select a biological model; return whether the dropdown lists it.

        A canonical identifier (or alias) selects the first entry of that formalism the
        machine can run; a model instance or specification selects the entry reading the
        same base data.
        """
        if isinstance(model, str):
            index = self._first_option_index(model)
        else:
            options = [o for o in self._bio_options if o is not None]
            index = self._option_index(
                matching_bio_option(options, model, self._current_machine())
            )
        if index >= 0:
            self._cmb_bio_model.setCurrentIndex(index)
        return index >= 0

    def _on_bio_config(self) -> None:
        name = self._bio_model_name()
        cls = self._bio_model_classes.get(name)
        if cls is None:
            return
        dialog = ConfigFormDialog(
            cls.config_model(),
            initial=self._bio_params.get(name, {}),
            title=f"Configure {bio_model_display_name(cls)}",
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._bio_params[name] = dialog.values()
            self._select_matching_option()
            self._update_dirty_state()
            self._update_tissue_indicator()

    def _select_matching_option(self) -> None:
        """Select the entry reading the base data the configured parameters choose."""
        name = self._bio_model_name()
        spec = {"model": name, **self._bio_params.get(name, {})}
        options = [o for o in self._bio_options if o is not None]
        index = self._option_index(matching_bio_option(options, spec, self._current_machine()))
        if index >= 0:
            self._cmb_bio_model.blockSignals(True)
            self._cmb_bio_model.setCurrentIndex(index)
            self._cmb_bio_model.blockSignals(False)

    def _on_tissue_parameters(self) -> None:
        cst = self._ws.cst
        if cst is None:
            return
        dialog = TissueParametersDialog(
            cst,
            self._current_machine(),
            self._current_bio_model(),
            self,
            engine=self._current_engine(),
        )
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.changed:
            self._ws.cst = cst

    def _current_machine(self) -> Optional[BioMachineSummary]:
        """
        Summary of the machine the model dropdown lists, None if it cannot be loaded.

        Only the kernel-free summary is kept, see
        :func:`~pyRadPlan.gui.widgets._tissue_params.load_machine_quietly`; the base data
        itself is neither returned nor held on to. The base data a plan carries itself is
        summarized once, when the plan is shown.
        """
        radiation_mode, name = self._bio_models_key
        if self._plan_machine is not None and name == self._plan_machine_label(radiation_mode):
            return self._plan_machine_summary
        return load_machine_quietly(radiation_mode, name)

    def _current_engine(self) -> Optional[type]:
        """
        Dose engine class selected in the form, None if the mode has none.

        Resolved as the dose engine factory does: the selected engine if it is available
        for the radiation mode, otherwise the mode's default engine.
        """
        return dose_engine_quietly(
            self._cmb_radiation.currentText(), self._cmb_engine.currentText() or None
        )

    def _current_bio_model(self) -> Optional[BiologicalModel]:
        """Model instance of the biological model selected, None if it cannot be built."""
        return bio_model_quietly(self._selected_bio_model(), self._cmb_radiation.currentText())

    def _selected_bio_model(self) -> Any:
        """Model spec for the plan; the plan's own model while it is kept unsupported."""
        index = self._cmb_bio_model.currentIndex()
        if index >= 0 and self._option_at(index) is None and self._unsupported_bio_model:
            return self._unsupported_bio_model
        name = self._bio_model_name() or "none"
        params = self._bio_params.get(name)
        if params:
            return {"model": name, **params}
        return name

    def _update_tissue_indicator(self, *_args) -> None:
        """
        Show the tissue marker while the structures have issues for the model.

        Errors (the model cannot be evaluated) show ⚠; warnings only (e.g. a tissue
        labelled MKM matching mMKM base data, which is evaluated with the base data's
        model) show the softer ℹ. :attr:`tissue_marker_state` is ``"error"``,
        ``"warning"`` or empty.
        """
        engine = self._current_engine()
        issues = tissue_issues_quietly(
            self._current_machine(), self._selected_bio_model(), self._ws.cst, engine
        )
        errors = tissue_errors(issues)
        self.tissue_marker_state = "error" if errors else "warning" if issues else ""
        self._lbl_tissue_warning.setVisible(bool(issues))
        with_engine = f" with the dose engine {engine.short_name}" if engine else ""
        context = (
            f"the structures on machine '{self._bio_models_key[1]}'{with_engine}" if issues else ""
        )
        model = self._cmb_bio_model.currentText()
        if errors:
            self._lbl_tissue_warning.setText(_TISSUE_ERROR_MARK)
            self._lbl_tissue_warning.setStyleSheet(_TISSUE_ERROR_STYLE)
            self._lbl_tissue_warning.setToolTip(
                f"The biological model '{model}' cannot be evaluated for {context}:\n"
                f"{tissue_issues_text(issues)}\n"
                "Set the tissue parameters or choose another model; otherwise dose "
                "calculation offers the physical dose only."
            )
        elif issues:
            self._lbl_tissue_warning.setText(_TISSUE_WARNING_MARK)
            self._lbl_tissue_warning.setStyleSheet(_TISSUE_WARNING_STYLE)
            self._lbl_tissue_warning.setToolTip(
                f"The biological model '{model}' is evaluated for {context}, with "
                f"warnings:\n{tissue_issues_text(issues)}\n"
                "Dose calculation uses the base data's model; relabelling the tissues in "
                "the tissue parameters is optional."
            )
        else:
            self._lbl_tissue_warning.setToolTip("")

    # ------------------------------------------------------------------
    # Dose engine handling
    # ------------------------------------------------------------------

    def _refresh_engines(self, radiation_mode: str) -> None:
        self._engines = get_available_engines(radiation_mode)
        current = self._cmb_engine.currentText()
        self._cmb_engine.blockSignals(True)
        self._cmb_engine.clear()
        self._cmb_engine.addItems(list(self._engines))
        if current in self._engines:
            self._cmb_engine.setCurrentText(current)
        self._cmb_engine.blockSignals(False)
        self._btn_engine_config.setEnabled(bool(self._engines))

    def _on_engine_config(self) -> None:
        engine_name = self._cmb_engine.currentText()
        engine_cls = self._engines.get(engine_name)
        if engine_cls is None:
            return

        dialog = ConfigFormDialog(
            engine_cls.config_model(),
            initial=self._engine_props.get(engine_name, {}),
            title=f"Configure {engine_cls.name}",
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._engine_props[engine_name] = dialog.values()
            self._update_dirty_state()

    # ------------------------------------------------------------------
    # Optimization (planning problem / solver) handling
    # ------------------------------------------------------------------

    def _refresh_problems(self, radiation_mode: str) -> None:
        self._problems = {
            name: cls
            for name, cls in get_available_problems().items()
            if radiation_mode in cls.possible_radiation_modes
        }
        current = self._cmb_problem.currentText()
        self._cmb_problem.blockSignals(True)
        self._cmb_problem.clear()
        for name, cls in self._problems.items():
            self._cmb_problem.addItem(name)
            self._cmb_problem.setItemData(
                self._cmb_problem.count() - 1, cls.name, Qt.ItemDataRole.ToolTipRole
            )
        if current in self._problems:
            self._cmb_problem.setCurrentText(current)
        self._cmb_problem.blockSignals(False)
        self._btn_problem_config.setEnabled(bool(self._problems))

    def _on_problem_config(self) -> None:
        problem_name = self._cmb_problem.currentText()
        problem_cls = self._problems.get(problem_name)
        if problem_cls is None:
            return

        dialog = OptimizationConfigDialog(
            problem_cls, initial=self._problem_props.get(problem_name, {}), parent=self
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._problem_props[problem_name] = dialog.values()
            self._update_dirty_state()

    def _restore_problem_from_pln(self, pln: Plan) -> None:
        prop_opt = pln.prop_opt if isinstance(pln.prop_opt, dict) else {}
        requested = prop_opt.get("problem") or prop_opt.get("opti_prob")
        self._plan_names_problem = bool(requested)
        # An unavailable problem falls back to the default, as get_problem_from_pln does.
        default = next(iter(self._problems), None)
        problem_name = requested if requested in self._problems else default
        if problem_name is None:
            return
        self._cmb_problem.setCurrentText(problem_name)
        self._problem_props[problem_name] = self._copy_prop_opt(
            {k: v for k, v in prop_opt.items() if k not in ("problem", "opti_prob")}
        )

    @staticmethod
    def _copy_prop_opt(prop_opt: dict) -> dict:
        """Copy configuration values while retaining a live solver and its thread state."""
        solver = prop_opt.get("solver")
        memo = {id(solver): solver} if isinstance(solver, SolverBase) else {}
        return deepcopy(prop_opt, memo)

    def _form_prop_opt(self) -> dict:
        """``pln.prop_opt`` described by the optimization fields."""
        problem_name = self._cmb_problem.currentText()
        prop_opt = self._copy_prop_opt(self._problem_props.get(problem_name, {}))
        default = next(iter(self._problems), None)
        if problem_name and (self._plan_names_problem or problem_name != default):
            prop_opt = {"problem": problem_name, **prop_opt}
        return prop_opt

    # ------------------------------------------------------------------
    # Workspace → UI synchronisation
    # ------------------------------------------------------------------

    def _do_update(self, changed_keys: list) -> None:
        if not changed_keys or "pln" in changed_keys or self._ws.pln is None:
            self._syncing = True
            try:
                self._sync_from_workspace()
            finally:
                self._syncing = False
            self._mark_clean()
        else:
            # The plan is unchanged, so the form keeps its unapplied edits; only what
            # is derived from the patient data follows it.
            self._btn_ct_grid.setEnabled(self._ws.ct is not None)
            self._update_auto_iso_center()
        self._btn_tissue.setEnabled(self._ws.cst is not None)
        if self._ws.pln is None and self._ws.has("ct", "cst"):
            self._apply_default_plan()
        self._update_tissue_indicator()

    def _apply_default_plan(self) -> None:
        """Apply the form defaults as the initial plan once patient data is loaded.

        If the defaults do not validate, the form is flagged as modified instead
        so the user sees that the plan still needs to be applied.
        """
        try:
            pln = self._build_plan()
        except Exception as exc:  # noqa: BLE001 - surface, never crash
            self._set_status(f"Plan not applied — {type(exc).__name__}: {exc}", error=True)
            self._set_global_modified(True)
            return
        with self.hold_updates():
            self._ws.pln = pln
        self._set_status("Default plan applied.")

    def _sync_from_workspace(self) -> None:
        pln = self._ws.pln
        if pln is None:
            self._btn_ct_grid.setEnabled(self._ws.ct is not None)
            self._update_auto_iso_center()
            return

        self._cmb_radiation.blockSignals(True)
        self._cmb_radiation.setCurrentText(pln.radiation_mode)
        self._cmb_radiation.blockSignals(False)
        if not isinstance(pln.machine, Machine):
            self._plan_machine = self._plan_machine_summary = None
        elif pln.machine is not self._plan_machine:
            self._plan_machine = pln.machine
            self._plan_machine_summary = machine_summary(pln.machine)
        self._refresh_machines(pln.radiation_mode)
        self._refresh_engines(pln.radiation_mode)
        self._refresh_problems(pln.radiation_mode)
        self._restore_problem_from_pln(pln)
        if self._plan_machine is not None:
            machine = self._plan_machine_label(pln.radiation_mode) or self._plan_machine.name
        else:
            machine = pln.machine if isinstance(pln.machine, str) else "Generic"
        self._cmb_machine.setEditText(machine)
        # The engine first: it decides, with the machine, which models are offered.
        self._restore_engine_from_pln(pln)

        bio_model = pln.bio_model if isinstance(pln.bio_model, BiologicalModel) else None
        if bio_model is not None:
            self._bio_params[bio_model.model] = dict(bio_model.parameters)
        self._refresh_bio_models(pln.radiation_mode, restore=bio_model)
        self._refresh_angle_presets(pln.radiation_mode)

        self._spn_fractions.setValue(int(pln.num_of_fractions))
        idx = self._cmb_dose_convention.findData(pln.dose_convention)
        if idx >= 0:
            self._cmb_dose_convention.setCurrentIndex(idx)

        stf = pln.prop_stf or {}
        gantry = stf.get("gantry_angles")
        if gantry is not None:
            self._txt_gantry.setText(self._format_angles(gantry))
        couch = stf.get("couch_angles")
        if couch is not None:
            self._txt_couch.setText(self._format_angles(couch))

        self._restore_iso_center(stf.get("iso_center"))

        scenario = getattr(pln.mult_scen, "short_name", None)
        if scenario and self._cmb_scenario.findText(scenario) >= 0:
            self._cmb_scenario.setCurrentText(scenario)

        spacing = stf.get("bixel_width")
        if spacing is not None:
            self._spn_bixel.setValue(float(spacing))

        # dose_grid may be a dict, a Grid instance or None (see DoseEngineBase).
        dose_grid = (pln.prop_dose_calc or {}).get("dose_grid")
        if isinstance(dose_grid, dict):
            resolution = dose_grid.get("resolution")
        else:
            resolution = getattr(dose_grid, "resolution", None)
        if isinstance(resolution, dict):
            self._spn_res_x.setValue(float(resolution.get("x", self._spn_res_x.value())))
            self._spn_res_y.setValue(float(resolution.get("y", self._spn_res_y.value())))
            self._spn_res_z.setValue(float(resolution.get("z", self._spn_res_z.value())))

        self._btn_ct_grid.setEnabled(self._ws.ct is not None)
        self._update_beam_count()

    def _restore_engine_from_pln(self, pln: Plan) -> None:
        prop_dose_calc = pln.prop_dose_calc if isinstance(pln.prop_dose_calc, dict) else {}
        engine_name = prop_dose_calc.get("engine") or next(iter(self._engines), None)
        if engine_name not in self._engines:
            return
        self._cmb_engine.setCurrentText(engine_name)
        engine_props = {
            k: v for k, v in prop_dose_calc.items() if k not in ("engine", "dose_grid")
        }
        # An empty override mapping means defaults, not "keep the form's edits".
        self._engine_props[engine_name] = deepcopy(engine_props)

    def _restore_iso_center(self, iso_center) -> None:
        self._per_beam_iso_center = None
        if iso_center is None or np.size(iso_center) == 0:
            self._chk_iso_auto.setChecked(True)
            self._update_auto_iso_center()
            return
        iso = np.atleast_2d(np.asarray(iso_center, dtype=float))
        if len(np.unique(iso, axis=0)) == 1:
            self._txt_iso.setText(self._format_angles(iso[0]))
            self._chk_iso_auto.setChecked(False)
        else:
            # per-beam iso centers cannot be edited here; show auto, but keep them
            # (setChecked first, as the toggle drops them and auto-fills the field)
            self._chk_iso_auto.setChecked(True)
            self._per_beam_iso_center = iso.tolist()
            self._txt_iso.setText("multiple iso centers")

    @staticmethod
    def _format_angles(angles) -> str:
        values = np.atleast_1d(np.asarray(angles, dtype=float)).ravel()
        return format_number_list(values)

    # ------------------------------------------------------------------
    # Field callbacks
    # ------------------------------------------------------------------

    def _on_iso_auto_toggled(self, checked: bool) -> None:
        self._per_beam_iso_center = None
        self._txt_iso.setEnabled(not checked)
        if checked:
            self._update_auto_iso_center()

    def _update_auto_iso_center(self) -> None:
        """Show the automatic isocenter (target center of mass) in the iso field.

        Mirrors what the stf generators compute when no explicit iso center is
        set, so the disabled field previews the value that will actually be used.
        """
        if not self._chk_iso_auto.isChecked() or self._per_beam_iso_center is not None:
            return
        cst = self._ws.cst
        if cst is None:
            return
        try:
            iso = cst.target_center_of_mass()
        except Exception:  # noqa: BLE001 - e.g. no target structures defined
            return
        self._txt_iso.setText(self._format_angles(iso))

    def _refresh_angle_presets(self, radiation_mode: str) -> None:
        """List the angle presets, those typical for ``radiation_mode`` first."""
        presets = get_beam_angle_presets(radiation_mode)
        menu = self._btn_angle_preset.menu()
        menu.clear()
        for preset in presets:
            angles = self._format_angles(preset.gantry_angles)
            if any(preset.couch_angles):
                angles += f" / couch {self._format_angles(preset.couch_angles)}"
            action = menu.addAction(f"{preset.name} ({angles})")
            action.setData(preset)
            action.setToolTip(preset.description or preset.name)

    def _on_angle_preset_activated(self, action) -> None:
        """Overwrite the gantry/couch angles after a preset is explicitly activated."""
        preset = action.data()
        self._txt_gantry.setText(self._format_angles(preset.gantry_angles))
        self._txt_couch.setText(self._format_angles(preset.couch_angles))
        self._update_beam_count()

    def _update_beam_count(self) -> None:
        """Show the number of gantry angles in the beam-count spin box."""
        try:
            n = len(parse_number_list(self._txt_gantry.text()))
        except ValueError:
            self._spn_num_beams.setEnabled(False)
            self._spn_num_beams.setToolTip("Gantry angles are not a valid list of numbers.")
            return
        self._spn_num_beams.setEnabled(True)
        self._spn_num_beams.setToolTip(_NUM_BEAMS_TOOLTIP)
        if n and n != self._spn_num_beams.value():
            self._spn_num_beams.blockSignals(True)
            self._spn_num_beams.setValue(n)
            self._spn_num_beams.blockSignals(False)

    def _on_num_beams_changed(self, count: int) -> None:
        """Replace the gantry angles by *count* equally spaced angles."""
        if self._syncing:
            return
        self._txt_gantry.setText(
            self._format_angles(np.linspace(0.0, 360.0, count, endpoint=False))
        )
        self._sync_couch_angles()

    def _sync_couch_angles(self) -> None:
        """Keep exactly one couch angle per gantry angle.

        A single couch angle is repeated for every beam; otherwise an added beam
        gets a couch angle of 0 and a removed one drops its couch angle, so the
        two fields stay aligned while the angles already set for the remaining
        beams are preserved.  Runs once the gantry edit is
        finished rather than per keystroke, where the partial text would trim
        the couch angles on the way.
        """
        try:
            n = len(parse_number_list(self._txt_gantry.text()))
            couch = parse_number_list(self._txt_couch.text())
        except ValueError:
            return
        if n == 0 or len(couch) == n:
            return
        pad = couch[0] if len(couch) == 1 else 0.0
        self._txt_couch.setText(format_number_list((couch + [pad] * n)[:n]))

    def _on_use_ct_grid(self) -> None:
        ct = self._ws.ct
        if ct is None:
            self._set_status("No CT in workspace.", error=True)
            return
        resolution = ct.grid.resolution
        self._spn_res_x.setValue(float(resolution["x"]))
        self._spn_res_y.setValue(float(resolution["y"]))
        self._spn_res_z.setValue(float(resolution["z"]))
        self._set_status("Dose grid set from CT.")

    def _on_apply(self) -> None:
        try:
            pln = self._build_plan()
        except Exception as exc:  # noqa: BLE001 - surface, never crash
            self._set_status(f"{type(exc).__name__}: {exc}", error=True)
            self.update_failed.emit(f"{type(exc).__name__}: {exc}")
            return

        with self.hold_updates():
            self._ws.pln = pln
        if self._plan_machine is not None and pln.machine is not self._plan_machine:
            self._plan_machine = self._plan_machine_summary = None
            self._refresh_machines(pln.radiation_mode)
        self._mark_clean()
        self._set_status("Plan applied.")

    def has_pending_changes(self) -> bool:
        """Whether the form has edits that are not applied to the plan yet."""
        return self._modified

    def apply_changes(self) -> bool:
        """Apply the form to the plan, as *Apply* does; False if it does not validate."""
        self._on_apply()
        return not self._modified

    def _on_discard(self) -> None:
        """Reset the form to the plan in the workspace."""
        if self._ws.pln is None:
            return
        self._syncing = True
        try:
            # Imported plans can omit geometry fields (or individual grid axes).
            # Sync only writes their explicit values, so first restore the last
            # clean form values instead of letting discarded edits become defaults.
            state = self._clean_snapshot
            for key, edit in (("gantry", self._txt_gantry), ("couch", self._txt_couch)):
                value = state[key]
                edit.setText(value if isinstance(value, str) else self._format_angles(value))
            for key, spin in (
                ("bixel", self._spn_bixel),
                ("res_x", self._spn_res_x),
                ("res_y", self._spn_res_y),
                ("res_z", self._spn_res_z),
            ):
                spin.setValue(state[key])
            self._sync_from_workspace()
        finally:
            self._syncing = False
        self._mark_clean()
        self._set_status("Changes discarded.")

    def set_gantry_angle(self, index: int, angle: float) -> None:
        """Set the gantry angle of beam *index* in the form, without applying it.

        Used e.g. by the viewer when a beam is dragged.  Ignored while the gantry
        field does not parse or has no beam *index*.
        """
        try:
            gantry = parse_number_list(self._txt_gantry.text())
        except ValueError:
            return
        if not 0 <= index < len(gantry):
            return
        gantry[index] = float(angle)
        self._txt_gantry.setText(self._format_angles(gantry))

    def _form_beam_angles(self) -> tuple[list[float], list[float]]:
        """Gantry and couch angle of every beam in the form, one couch angle per beam.

        Raises ``ValueError`` when the fields do not describe a valid set of beams.
        """
        gantry = parse_number_list(self._txt_gantry.text())
        if not gantry:
            raise ValueError("at least one gantry angle is required")
        couch = parse_number_list(self._txt_couch.text())
        if len(couch) == 1:
            couch = couch * len(gantry)
        elif len(couch) not in (0, len(gantry)):
            raise ValueError("number of couch angles must be 1 or match gantry angles")
        if not couch:
            couch = [0.0] * len(gantry)
        return gantry, couch

    def edited_beam_angles(self) -> Optional[tuple[list[float], list[float]]]:
        """Return the form's ``(gantry, couch)`` angles, applied or not; ``None`` if invalid."""
        try:
            return self._form_beam_angles()
        except ValueError:
            return None

    def _emit_beam_angles(self, *_args) -> None:
        self.beam_angles_edited.emit(self.edited_beam_angles())

    def _on_ai_beam_angles(self) -> None:
        radiation_mode = self._cmb_radiation.currentText()
        cls = _plan_class(radiation_mode)
        try:
            base_pln = cls(
                radiation_mode=radiation_mode,
                machine=self._form_machine_name(),
                num_of_fractions=int(self._spn_fractions.value()),
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Cannot prepare plan for AI: {exc}", error=True)
            return

        from pyRadPlan.gui.widgets.ai import AiTask, AiTaskDialog, busy_cursor  # noqa: PLC0415

        # Deferred: keeps the optional ai.agents stack out of widget construction.
        with busy_cursor():
            from pyRadPlan.ai.agents import (  # noqa: PLC0415
                available_models,
                beam_angles_system_prompt,
                cst_geometry_summary,
                generate_beam_angles,
            )

        cst = self._ws.cst

        def _run(model: str, site: str, context: str):
            return generate_beam_angles(
                base_pln,
                treatment_site=site or "unspecified",
                additional_context=context or None,
                model=model,
                cst=cst,
            )

        def _apply(new_pln) -> None:
            # Populate the form fields; the user reviews and presses Apply.
            stf = new_pln.prop_stf or {}
            gantry = stf.get("gantry_angles")
            couch = stf.get("couch_angles")
            if gantry is not None:
                self._txt_gantry.setText(self._format_angles(gantry))
            if couch is not None:
                self._txt_couch.setText(self._format_angles(couch))
            self._sync_couch_angles()

        def _summarize(new_pln) -> str:
            n = len((new_pln.prop_stf or {}).get("gantry_angles", []))
            return f"Suggested {n} beam(s) — review and press Apply."

        context = {
            "radiation_mode": radiation_mode,
            "machine": self._form_machine_name(),
            "num_of_fractions": int(self._spn_fractions.value()),
        }
        if cst is not None:
            context["structure_set_geometry"] = cst_geometry_summary(cst)
        task = AiTask(
            title="Suggest beam angles (AI)",
            system_prompt=beam_angles_system_prompt(radiation_mode, with_geometry=cst is not None),
            context_text=json.dumps(context, indent=2, default=str),
            run=_run,
            apply=_apply,
            summarize=_summarize,
        )
        AiTaskDialog(task, available_models(), parent=self).exec()

    # ------------------------------------------------------------------
    # Plan construction
    # ------------------------------------------------------------------

    def _build_plan(self) -> Plan:
        radiation_mode = self._cmb_radiation.currentText()

        gantry, couch = self._form_beam_angles()

        current = self._ws.pln
        same_mode = current is not None and current.radiation_mode == radiation_mode
        prop_stf = deepcopy(current.prop_stf) if same_mode else {}
        prop_stf.update(
            gantry_angles=gantry,
            couch_angles=couch,
            bixel_width=float(self._spn_bixel.value()),
        )

        if not self._chk_iso_auto.isChecked():
            iso = parse_number_list(self._txt_iso.text())
            if len(iso) != 3:
                raise ValueError("iso center requires exactly three coordinates (x y z)")
            prop_stf["iso_center"] = iso
        elif self._per_beam_iso_center is not None:
            if len(self._per_beam_iso_center) != len(gantry):
                raise ValueError(
                    f"the plan's {len(self._per_beam_iso_center)} per-beam iso centers do not "
                    f"match {len(gantry)} beams; enter an iso center, or untick and re-tick "
                    "Auto to use the target center of mass"
                )
            prop_stf["iso_center"] = deepcopy(self._per_beam_iso_center)
        else:
            prop_stf.pop("iso_center", None)

        prop_dose_calc: dict = {}
        engine_name = self._cmb_engine.currentText()
        if engine_name:
            prop_dose_calc["engine"] = engine_name
            prop_dose_calc.update(self._engine_props.get(engine_name, {}))
        prop_dose_calc["dose_grid"] = {
            "resolution": {
                "x": float(self._spn_res_x.value()),
                "y": float(self._spn_res_y.value()),
                "z": float(self._spn_res_z.value()),
            }
        }

        # Keep settings this form does not edit (prescription, sequencing, etc.).
        # Read field values directly to retain arbitrary model types. The
        # machine is set below and never copied: base data runs to hundreds of megabytes.
        plan_data = (
            {
                name: deepcopy(getattr(current, name))
                for name in type(current).model_fields
                if name not in ("machine", "prop_opt")
            }
            if current is not None
            else {}
        )
        scenario = self._cmb_scenario.currentText()
        if current is not None and getattr(current.mult_scen, "short_name", None) == scenario:
            scenario = deepcopy(current.mult_scen)
        plan_data.update(
            {
                "radiation_mode": radiation_mode,
                "machine": self._form_machine(),
                "num_of_fractions": int(self._spn_fractions.value()),
                "dose_convention": self._cmb_dose_convention.currentData(),
                "bio_model": self._selected_bio_model(),
                "mult_scen": scenario,
                "prop_stf": prop_stf,
                "prop_seq": deepcopy(current.prop_seq) if same_mode else {},
                "prop_dose_calc": prop_dose_calc,
                "prop_opt": self._form_prop_opt(),
            }
        )

        # Constructing validates; validate_pln would deep-copy the plan's own base data.
        return _plan_class(radiation_mode)(**plan_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, message: str, error: bool = False) -> None:
        self._status_is_modified_note = False
        self._status_is_error = error
        self._lbl_status.setText(message)
        self._lbl_status.setToolTip("")
        self._lbl_status.setStyleSheet("color: red;" if error else "")

    # ------------------------------------------------------------------
    # Modified-state tracking
    # ------------------------------------------------------------------

    def _connect_dirty_signals(self) -> None:
        # ``_cmb_radiation`` and ``_cmb_engine`` are handled by ``_on_radiation_changed`` and
        # ``_on_engine_changed``.
        self._cmb_machine.editTextChanged.connect(self._on_field_edited)
        self._cmb_problem.currentTextChanged.connect(self._on_field_edited)
        self._cmb_bio_model.currentIndexChanged.connect(self._on_field_edited)
        self._spn_fractions.valueChanged.connect(self._on_field_edited)
        self._cmb_dose_convention.currentIndexChanged.connect(self._on_field_edited)
        self._txt_gantry.textChanged.connect(self._on_field_edited)
        self._txt_couch.textChanged.connect(self._on_field_edited)
        self._txt_gantry.textChanged.connect(self._emit_beam_angles)
        self._txt_couch.textChanged.connect(self._emit_beam_angles)
        self._spn_bixel.valueChanged.connect(self._on_field_edited)
        self._chk_iso_auto.toggled.connect(self._on_field_edited)
        self._txt_iso.textChanged.connect(self._on_field_edited)
        self._cmb_scenario.currentTextChanged.connect(self._on_field_edited)
        for spin in (self._spn_res_x, self._spn_res_y, self._spn_res_z):
            spin.valueChanged.connect(self._on_field_edited)

    def _on_field_edited(self, *_args) -> None:
        self._update_dirty_state()

    def _capture_state(self) -> dict[str, Any]:
        """Snapshot the tracked field values for modified-state comparison."""
        engine = self._cmb_engine.currentText()
        problem = self._cmb_problem.currentText()
        not_auto = not self._chk_iso_auto.isChecked()
        bio_params = self._bio_params.get(self._bio_model_name(), {})
        unsupported = self._option_at(self._cmb_bio_model.currentIndex()) is None
        return {
            "radiation": self._cmb_radiation.currentText(),
            "machine": self._cmb_machine.currentText(),
            "engine": engine,
            "engine_props": dict(self._engine_props.get(engine, {})),
            "problem": problem,
            "problem_props": dict(self._problem_props.get(problem, {})),
            # The base data the selected entry reads belongs to the model choice, not to
            # the parameters configured through the "…" button.
            "bio_model": json.dumps(
                [self._bio_model_name(), unsupported, selection_parameters(bio_params)],
                default=str,
                sort_keys=True,
            ),
            "bio_params": json.dumps(
                {k: v for k, v in bio_params.items() if k not in SELECTION_PARAMETERS},
                default=str,
                sort_keys=True,
            ),
            "fractions": self._spn_fractions.value(),
            "dose_convention": self._cmb_dose_convention.currentData(),
            "gantry": self._safe_parse(self._txt_gantry.text()),
            "couch": self._safe_parse(self._txt_couch.text()),
            "bixel": self._spn_bixel.value(),
            "iso_auto": self._chk_iso_auto.isChecked(),
            "iso_per_beam": self._per_beam_iso_center is not None,
            "iso": self._txt_iso.text() if not_auto else None,
            "scenario": self._cmb_scenario.currentText(),
            "res_x": self._spn_res_x.value(),
            "res_y": self._spn_res_y.value(),
            "res_z": self._spn_res_z.value(),
        }

    @staticmethod
    def _safe_parse(text: str):
        # Parse like the plan would; keep raw text on failure so invalid input
        # still registers as "modified" rather than silently matching.
        try:
            return parse_number_list(text)
        except ValueError:
            return text

    def _update_dirty_state(self) -> None:
        if self._syncing:
            return
        current = self._capture_state()
        # Several keys can share an editor (e.g. iso auto/value), so OR the
        # modified flag per widget before painting.
        per_widget: dict[int, tuple[QWidget, bool]] = {}
        any_modified = False
        for key, widget in self._field_widgets.items():
            modified = current.get(key) != self._clean_snapshot.get(key)
            any_modified = any_modified or modified
            prev = per_widget.get(id(widget))
            per_widget[id(widget)] = (widget, modified or (prev[1] if prev else False))
        for widget, modified in per_widget.values():
            self._set_field_modified(widget, modified)
        self._set_global_modified(any_modified)

    def _mark_clean(self) -> None:
        """Capture a fresh clean snapshot and clear all modified indicators."""
        self._clean_snapshot = self._capture_state()
        for widget in set(self._field_widgets.values()):
            self._set_field_modified(widget, False)
        self._set_global_modified(False)

    @staticmethod
    def _set_field_modified(widget: QWidget, modified: bool) -> None:
        widget.setStyleSheet(_MODIFIED_STYLE if modified else "")

    def _set_global_modified(self, modified: bool) -> None:
        self._modified = modified
        self._btn_apply.setText("Apply *" if modified else "Apply")
        self._btn_discard.setEnabled(modified and self._ws.pln is not None)
        self._btn_apply.setStyleSheet("font-weight: bold;" if modified else "")
        if modified:
            if not self._status_is_error:
                self._lbl_status.setText(_MODIFIED_NOTE)
                self._lbl_status.setStyleSheet("")
                self._status_is_modified_note = True
        elif self._status_is_modified_note:
            self._lbl_status.setText("")
            self._status_is_modified_note = False
