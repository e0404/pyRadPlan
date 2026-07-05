"""Plan configuration widget for the pyRadPlan GUI."""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.dose.engines import get_available_engines
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.plan import IonPlan, PhotonPlan, Plan, validate_pln
from pyRadPlan.quantities import get_available_quantities
from pyRadPlan.scenarios import available_scenario_models
from .._base import WorkspaceWidget, format_number_list, parse_number_list
from .._config_form import ConfigFormDialog
from ..ai import AI_MISSING_TIP, ai_available

_ION_MODES = list(IonPlan.available_radiation_modes)
_RADIATION_MODES = ["photons", *_ION_MODES]

#: Orange border marking a field whose value differs from the applied plan.
_MODIFIED_STYLE = "border: 1px solid #e67e22;"
#: Persistent status note shown while the form has unapplied edits.
_MODIFIED_NOTE = "Modified — not applied"


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
    """

    _watched_keys = ("ct", "cst", "pln")

    def __init__(
        self,
        workspace: Optional[WorkspaceManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(workspace, parent)
        self._ai_available = ai_available()
        #: Dose engine configuration values per engine short name, edited via
        #: the [...] popup and written into ``pln.prop_dose_calc`` on Apply.
        self._engine_props: dict[str, dict] = {}
        self._engines: dict[str, type] = {}
        #: True while syncing the form *from* the workspace, so programmatic
        #: widget changes don't register as user edits.
        self._syncing = False
        #: UI field values at the last sync point; the form is "modified" when
        #: the live values differ from this snapshot.
        self._clean_snapshot: dict[str, Any] = {}
        #: Whether the status label currently shows the modified note (so real
        #: messages aren't mistaken for it and vice versa).
        self._status_is_modified_note = False
        self._status_is_error = False
        self._setup_ui()
        #: Maps each tracked field key to the editor widget to border.
        self._field_widgets: dict[str, QWidget] = {
            "radiation": self._cmb_radiation,
            "machine": self._cmb_machine,
            "engine": self._cmb_engine,
            "engine_props": self._btn_engine_config,
            "fractions": self._spn_fractions,
            "gantry": self._txt_gantry,
            "couch": self._txt_couch,
            "bixel": self._spn_bixel,
            "iso_auto": self._txt_iso,
            "iso": self._txt_iso,
            "scenario": self._cmb_scenario,
            "res_x": self._spn_res_x,
            "res_y": self._spn_res_y,
            "res_z": self._spn_res_z,
        }
        self._connect_dirty_signals()
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
        self._btn_apply = QPushButton("Apply")
        self._btn_apply.clicked.connect(self._on_apply)
        bottom.addWidget(self._btn_apply)
        root.addLayout(bottom)

        root.addStretch()

        self._refresh_machines(self._cmb_radiation.currentText())
        self._refresh_engines(self._cmb_radiation.currentText())
        self._update_mode_dependent_fields()
        self._update_beam_count()

    def _build_fields(self, grid: QGridLayout) -> None:
        # Row layout mirrors matRad's PlanWidget, with modality and machine as
        # the first options to set.
        self._build_setup_rows(grid)
        self._build_geometry_rows(grid)
        self._build_model_rows(grid)
        self._build_sequencing_row(grid)
        self._build_dose_grid_row(grid)

    def _build_setup_rows(self, grid: QGridLayout) -> None:
        self._cmb_radiation = QComboBox()
        self._cmb_radiation.addItems(_RADIATION_MODES)
        self._cmb_radiation.currentTextChanged.connect(self._on_radiation_changed)

        self._cmb_machine = QComboBox()
        self._cmb_machine.setEditable(True)

        self._engine_row = self._build_engine_selector()

        self._spn_fractions = QSpinBox()
        self._spn_fractions.setRange(1, 1000)
        self._spn_fractions.setValue(30)

        grid.addWidget(QLabel("Radiation mode:"), 0, 0)
        grid.addWidget(self._cmb_radiation, 0, 1)
        grid.addWidget(QLabel("Machine:"), 0, 2)
        grid.addWidget(self._cmb_machine, 0, 3)

        grid.addWidget(QLabel("Dose engine:"), 1, 0)
        grid.addLayout(self._engine_row, 1, 1)
        grid.addWidget(QLabel("Fractions:"), 1, 2)
        grid.addWidget(self._spn_fractions, 1, 3)

    def _build_geometry_rows(self, grid: QGridLayout) -> None:
        self._txt_gantry = QLineEdit("0")
        self._txt_gantry.setToolTip(
            "Gantry angles in the matRad coordinate system.\n"
            "Every gantry angle defines a beam; separate angles by blanks."
        )
        self._txt_gantry.editingFinished.connect(self._update_beam_count)
        self._txt_couch = QLineEdit("0")
        self._txt_couch.setToolTip(
            "Couch angles in the matRad coordinate system.\n"
            "Every couch angle belongs to a gantry angle; separate angles by blanks."
        )
        self._lbl_beams = QLabel("1 beam")

        self._btn_ai_beams = QPushButton("✨ AI")
        self._btn_ai_beams.clicked.connect(self._on_ai_beam_angles)
        self._btn_ai_beams.setEnabled(self._ai_available)
        self._btn_ai_beams.setToolTip(
            "Suggest gantry/couch angles using an LLM" if self._ai_available else AI_MISSING_TIP
        )

        gantry_row = QHBoxLayout()
        gantry_row.setContentsMargins(0, 0, 0, 0)
        gantry_row.setSpacing(4)
        gantry_row.addWidget(self._txt_gantry, 1)
        gantry_row.addWidget(self._lbl_beams)
        gantry_row.addWidget(self._btn_ai_beams)

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
        self._chk_iso_auto.toggled.connect(lambda checked: self._txt_iso.setEnabled(not checked))

        iso_row = QHBoxLayout()
        iso_row.setContentsMargins(0, 0, 0, 0)
        iso_row.setSpacing(4)
        iso_row.addWidget(self._txt_iso, 1)
        iso_row.addWidget(self._chk_iso_auto)

        grid.addWidget(QLabel("Gantry angles [deg]:"), 2, 0)
        grid.addLayout(gantry_row, 2, 1)
        grid.addWidget(QLabel("Couch angles [deg]:"), 2, 2)
        grid.addWidget(self._txt_couch, 2, 3)

        grid.addWidget(self._lbl_bixel, 3, 0)
        grid.addWidget(self._spn_bixel, 3, 1)
        grid.addWidget(QLabel("Iso center [mm]:"), 3, 2)
        grid.addLayout(iso_row, 3, 3)

    def _build_model_rows(self, grid: QGridLayout) -> None:
        self._cmb_bio_model = QComboBox()
        self._cmb_bio_model.addItems(["none"])
        self._set_not_implemented(self._cmb_bio_model, "Biological models")

        self._cmb_scenario = QComboBox()
        self._cmb_scenario.addItems(available_scenario_models())
        for model in ("wcScen", "impScen", "rndScen"):
            self._add_disabled_item(self._cmb_scenario, model)
        self._cmb_scenario.setToolTip("Scenario sampling model for uncertainty handling")

        self._cmb_quantity = QComboBox()
        self._cmb_quantity.addItems(list(get_available_quantities()))
        self._set_not_implemented(
            self._cmb_quantity,
            "A global optimized quantity",
            hint="The quantity is currently set per objective in the objectives table.",
        )

        self._btn_tissue = QPushButton("Set tissue α/β")
        self._set_not_implemented(self._btn_tissue, "Tissue parameter configuration")

        grid.addWidget(QLabel("Biological model:"), 4, 0)
        grid.addWidget(self._cmb_bio_model, 4, 1)
        grid.addWidget(QLabel("Scenario model:"), 4, 2)
        grid.addWidget(self._cmb_scenario, 4, 3)

        grid.addWidget(QLabel("Optimized quantity:"), 5, 0)
        grid.addWidget(self._cmb_quantity, 5, 1)
        grid.addWidget(self._btn_tissue, 5, 3)

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

        grid.addLayout(seq_row, 6, 0, 1, 2)
        grid.addLayout(opt_row, 6, 2, 1, 2)

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

        grid.addWidget(QLabel("Dose grid res. [mm]:"), 7, 0)
        grid.addLayout(res_row, 7, 1, 1, 3)

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
    # Machine / mode handling
    # ------------------------------------------------------------------

    def _refresh_machines(self, radiation_mode: str) -> None:
        current = self._cmb_machine.currentText()
        self._cmb_machine.blockSignals(True)
        self._cmb_machine.clear()
        self._cmb_machine.addItem("Generic")
        self._cmb_machine.blockSignals(False)
        if current:
            self._cmb_machine.setEditText(current)
        else:
            self._cmb_machine.setEditText("Generic")

    def _update_mode_dependent_fields(self) -> None:
        is_ion = self._cmb_radiation.currentText() in _ION_MODES
        label = "Spot spacing [mm]:" if is_ion else "Bixel width [mm]:"
        self._lbl_bixel.setText(label)

    def _on_radiation_changed(self, radiation_mode: str) -> None:
        self._refresh_machines(radiation_mode)
        self._refresh_engines(radiation_mode)
        self._update_mode_dependent_fields()
        self._update_dirty_state()

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
    # Workspace → UI synchronisation
    # ------------------------------------------------------------------

    def _do_update(self, _changed_keys: list) -> None:
        self._syncing = True
        try:
            self._sync_from_workspace()
        finally:
            self._syncing = False
        self._mark_clean()

    def _sync_from_workspace(self) -> None:
        pln = self._ws.pln
        if pln is None:
            self._btn_ct_grid.setEnabled(self._ws.ct is not None)
            return

        self._cmb_radiation.blockSignals(True)
        self._cmb_radiation.setCurrentText(pln.radiation_mode)
        self._cmb_radiation.blockSignals(False)
        self._refresh_machines(pln.radiation_mode)
        self._refresh_engines(pln.radiation_mode)
        self._update_mode_dependent_fields()

        self._restore_engine_from_pln(pln)

        machine = pln.machine if isinstance(pln.machine, str) else "Generic"
        self._cmb_machine.setEditText(machine)
        self._spn_fractions.setValue(int(pln.num_of_fractions))

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

        is_ion = pln.radiation_mode in _ION_MODES
        spacing_key = "longitudinal_spot_spacing" if is_ion else "bixel_width"
        spacing = stf.get(spacing_key, stf.get("bixel_width"))
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
        engine_name = prop_dose_calc.get("engine")
        if engine_name not in self._engines:
            return
        self._cmb_engine.setCurrentText(engine_name)
        engine_props = {
            k: v for k, v in prop_dose_calc.items() if k not in ("engine", "dose_grid")
        }
        if engine_props:
            self._engine_props[engine_name] = engine_props

    def _restore_iso_center(self, iso_center) -> None:
        if iso_center is None:
            self._chk_iso_auto.setChecked(True)
            return
        iso = np.atleast_2d(np.asarray(iso_center, dtype=float))
        if len(np.unique(iso, axis=0)) == 1:
            self._txt_iso.setText(self._format_angles(iso[0]))
            self._chk_iso_auto.setChecked(False)
        else:
            # per-beam iso centers cannot be edited here; fall back to auto
            self._txt_iso.setText("multiple iso centers")
            self._chk_iso_auto.setChecked(True)

    @staticmethod
    def _format_angles(angles) -> str:
        values = np.atleast_1d(np.asarray(angles, dtype=float)).ravel()
        return format_number_list(values)

    # ------------------------------------------------------------------
    # Field callbacks
    # ------------------------------------------------------------------

    def _update_beam_count(self) -> None:
        try:
            n = len(parse_number_list(self._txt_gantry.text()))
        except ValueError:
            self._lbl_beams.setText("invalid gantry angles")
            return
        self._lbl_beams.setText(f"{n} beam{'s' if n != 1 else ''}")

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
        self._mark_clean()
        self._set_status("Plan applied.")

    def _on_ai_beam_angles(self) -> None:
        radiation_mode = self._cmb_radiation.currentText()
        cls = _plan_class(radiation_mode)
        try:
            base_pln = cls(
                radiation_mode=radiation_mode,
                machine=self._cmb_machine.currentText() or "Generic",
                num_of_fractions=int(self._spn_fractions.value()),
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Cannot prepare plan for AI: {exc}", error=True)
            return

        # Deferred: keeps the optional ai_agents stack out of widget construction.
        from pyRadPlan.ai_agents import (  # noqa: PLC0415
            available_models,
            beam_angles_system_prompt,
            generate_beam_angles,
        )
        from pyRadPlan.gui.widgets.ai import AiTask, AiTaskDialog  # noqa: PLC0415

        def _run(model: str, site: str, context: str):
            return generate_beam_angles(
                base_pln,
                treatment_site=site or "unspecified",
                additional_context=context or None,
                model=model,
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
            self._update_beam_count()

        def _summarize(new_pln) -> str:
            n = len((new_pln.prop_stf or {}).get("gantry_angles", []))
            return f"Suggested {n} beam(s) — review and press Apply."

        context = {
            "radiation_mode": radiation_mode,
            "machine": self._cmb_machine.currentText() or "Generic",
            "num_of_fractions": int(self._spn_fractions.value()),
        }
        task = AiTask(
            title="Suggest beam angles (AI)",
            system_prompt=beam_angles_system_prompt(radiation_mode),
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

        is_ion = radiation_mode in _ION_MODES
        spacing_key = "longitudinal_spot_spacing" if is_ion else "bixel_width"

        prop_stf = {
            "gantry_angles": gantry,
            "couch_angles": couch,
            "bixel_width": float(self._spn_bixel.value()),
        }
        if is_ion:
            prop_stf[spacing_key] = float(self._spn_bixel.value())

        if not self._chk_iso_auto.isChecked():
            iso = parse_number_list(self._txt_iso.text())
            if len(iso) != 3:
                raise ValueError("iso center requires exactly three coordinates (x y z)")
            prop_stf["iso_center"] = iso

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

        plan_data = {
            "radiation_mode": radiation_mode,
            "machine": self._cmb_machine.currentText() or "Generic",
            "num_of_fractions": int(self._spn_fractions.value()),
            "mult_scen": self._cmb_scenario.currentText(),
            "prop_stf": prop_stf,
            "prop_dose_calc": prop_dose_calc,
        }

        return validate_pln(_plan_class(radiation_mode)(**plan_data))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, message: str, error: bool = False) -> None:
        self._status_is_modified_note = False
        self._status_is_error = error
        self._lbl_status.setText(message)
        self._lbl_status.setStyleSheet("color: red;" if error else "")

    # ------------------------------------------------------------------
    # Modified-state tracking
    # ------------------------------------------------------------------

    def _connect_dirty_signals(self) -> None:
        # ``_cmb_radiation`` is handled by ``_on_radiation_changed``.
        self._cmb_machine.editTextChanged.connect(self._on_field_edited)
        self._cmb_engine.currentTextChanged.connect(self._on_field_edited)
        self._spn_fractions.valueChanged.connect(self._on_field_edited)
        self._txt_gantry.textChanged.connect(self._on_field_edited)
        self._txt_couch.textChanged.connect(self._on_field_edited)
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
        not_auto = not self._chk_iso_auto.isChecked()
        return {
            "radiation": self._cmb_radiation.currentText(),
            "machine": self._cmb_machine.currentText(),
            "engine": engine,
            "engine_props": dict(self._engine_props.get(engine, {})),
            "fractions": self._spn_fractions.value(),
            "gantry": self._safe_parse(self._txt_gantry.text()),
            "couch": self._safe_parse(self._txt_couch.text()),
            "bixel": self._spn_bixel.value(),
            "iso_auto": self._chk_iso_auto.isChecked(),
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
        self._btn_apply.setText("Apply *" if modified else "Apply")
        self._btn_apply.setStyleSheet("font-weight: bold;" if modified else "")
        if modified:
            if not self._status_is_error:
                self._lbl_status.setText(_MODIFIED_NOTE)
                self._lbl_status.setStyleSheet("")
                self._status_is_modified_note = True
        elif self._status_is_modified_note:
            self._lbl_status.setText("")
            self._status_is_modified_note = False
