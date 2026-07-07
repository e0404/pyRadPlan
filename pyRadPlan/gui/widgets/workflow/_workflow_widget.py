"""Workflow widget for the pyRadPlan GUI."""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.core import ComputeControl, ProgressReport, StatusReport
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets.optimization import OptimizationStatusWidget
from pyRadPlan.gui.widgets.optimization._status_window import DEFAULT_METRICS
from .._base import WorkspaceWidget, Worker


class WorkflowWidget(WorkspaceWidget):
    """Guides the user through the pyRadPlan treatment-planning workflow.

    Binds to a :class:`~pyRadPlan.gui.workspace.WorkspaceManager` and enables or
    disables each workflow step depending on which pipeline objects
    (ct, cst, pln, stf, dij, result) are currently available.  Long-running
    computations (dose influence, optimization, forward dose) run in a
    background :class:`~PySide6.QtCore.QThread` so the UI stays responsive.

    Parameters
    ----------
    workspace:
        Shared :class:`WorkspaceManager` instance.  Falls back to the
        process-wide singleton when *None*.
    parent:
        Optional Qt parent widget.
    """

    #: Resolution of the determinate progress bar (combined nested fraction).
    _PROGRESS_STEPS = 1000

    #: Emitted when a background computation starts (True) or ends (False).
    #: Hosts (e.g. the main window) can use it to lock other input widgets.
    busy_changed = Signal(bool)

    #: Internal: a compute report marshalled from the worker thread to the GUI
    #: thread.  Carries a :class:`~pyRadPlan.core.ComputeReport`.
    _report_received = Signal(object)

    def __init__(
        self,
        workspace: Optional[WorkspaceManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(workspace, parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[Worker] = None
        self._control: Optional[ComputeControl] = None
        self._pending_success: Optional[Callable] = None
        self._saved_tags: list[str] = []
        self._opt_status_win: Optional[OptimizationStatusWidget] = None
        # matRad-style staleness: a downstream product (dij/result) is "stale"
        # once an upstream object it was computed from changes.
        self._dij_stale = False
        self._result_stale = False
        self._indicators: dict[str, QLabel] = {}

        self._setup_ui()
        # Queued (cross-thread) delivery of compute reports to the GUI thread.
        self._report_received.connect(self._on_compute_report)
        self.initialize()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # Buttons disabled while a computation runs; filled by _add_button.
        self._action_buttons: list[QPushButton] = []

        # Status row
        row = QHBoxLayout()
        row.addWidget(QLabel("Status:"))
        self._lbl_status = QLabel("No data loaded")
        self._lbl_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row.addWidget(self._lbl_status)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._on_refresh)
        row.addWidget(btn_refresh)
        root.addLayout(row)

        # Indeterminate "busy" bar shown while a background computation runs.
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate / pulsing
        self._progress.setTextVisible(False)
        self._progress.setMaximumHeight(8)
        self._progress.hide()
        root.addWidget(self._progress)

        # Full pipeline buttons. Data I/O is also reachable from the File menu;
        # the duplication is intentional so the workflow reads top to bottom.
        self._btn_load_mat = self._add_button("Load .mat", self._on_load_mat)
        self._btn_load_dicom = self._add_button(
            "Load DICOM", self._on_load_dicom, implemented=False
        )
        self._btn_import_bin = self._add_button(
            "Import from Binary", self._on_import_binary, implemented=False
        )
        self._btn_calc_dose = self._add_button("Calc. Dose Influence", self._on_calc_dose)
        self._btn_import_dose = self._add_button("Import Dose", self._on_import_dose)
        self._btn_optimize = self._add_button("Optimize", self._on_optimize)
        self._btn_recalc = self._add_button("Recalculate Dose", self._on_recalc_dose)
        self._btn_save_result = self._add_button("Save / Keep Result", self._on_save_result)
        self._btn_export_bin = self._add_button(
            "Export Binary", self._on_export_binary, implemented=False
        )
        self._btn_export_dicom = self._add_button(
            "Export DICOM", self._on_export_dicom, implemented=False
        )

        # Each column is a workflow stage, progressing left to right (as in
        # matRad): loading -> dose influence -> result computation -> export.
        grid = QGridLayout()
        grid.setSpacing(4)

        # Status dots sit next to the column headers (one per pipeline stage).
        for col, (title, stage) in enumerate(
            (
                ("Loading", "load"),
                ("Dose Influence", "dij"),
                ("Result Computation", "result"),
                ("Export / Save", None),
            )
        ):
            grid.addWidget(self._header_cell(title, stage), 0, col)

        # Col 0: loading
        grid.addWidget(self._btn_load_mat, 1, 0)
        grid.addWidget(self._btn_load_dicom, 2, 0)
        grid.addWidget(self._btn_import_bin, 3, 0)
        # Col 1: dose influence
        grid.addWidget(self._btn_calc_dose, 1, 1)
        grid.addWidget(self._btn_import_dose, 2, 1)
        # Col 2: result computation (optimization, recalculation)
        grid.addWidget(self._btn_optimize, 1, 2)
        grid.addWidget(self._btn_recalc, 2, 2)
        # Col 3: export / keep / save
        grid.addWidget(self._btn_save_result, 1, 3)
        grid.addWidget(self._btn_export_bin, 2, 3)
        grid.addWidget(self._btn_export_dicom, 3, 3)
        root.addLayout(grid)

        root.addStretch()

    def _add_button(
        self,
        text: str,
        slot: Callable,
        *,
        implemented: bool = True,
        tooltip: Optional[str] = None,
    ) -> QPushButton:
        """Create a button wired to *slot*.

        Implemented buttons are registered as action buttons (toggled by the
        busy/enable cycle).  Buttons marked ``implemented=False`` are permanently
        disabled with an explanatory tooltip and deliberately *not* registered,
        so the busy cycle never re-enables them.
        """
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        if not implemented:
            btn.setEnabled(False)
            btn.setToolTip(tooltip or "Not yet implemented")
        else:
            self._action_buttons.append(btn)
            if tooltip:
                btn.setToolTip(tooltip)
        return btn

    @staticmethod
    def _make_dot() -> QLabel:
        dot = QLabel("○")
        dot.setFixedWidth(14)
        return dot

    def _header_cell(self, title: str, stage: Optional[str]) -> QWidget:
        """Build a bold column header, optionally with a status dot for *stage*."""
        cell = QWidget()
        lay = QHBoxLayout(cell)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        if stage is not None:
            dot = self._make_dot()
            self._indicators[stage] = dot
            lay.addWidget(dot)
        header = QLabel(title)
        header.setStyleSheet("font-weight: bold;")
        lay.addWidget(header)
        lay.addStretch()
        return cell

    # ------------------------------------------------------------------
    # Workspace → UI synchronisation
    # ------------------------------------------------------------------

    def _do_update(self, changed_keys: list) -> None:
        # This widget has no editable fields, only buttons/labels derived from
        # which pipeline objects exist.  It therefore *intentionally* reacts to
        # its own workspace writes (no ``hold_updates``) so that, e.g., the
        # "Save Result" button enables immediately after optimization finishes.
        self._prune_saved_tags()
        self._update_staleness(changed_keys)
        self._update_status()
        self._update_button_states()
        self._update_indicators()

    def _prune_saved_tags(self) -> None:
        """Drop snapshot tags whose keys no longer exist in the result.

        Keeps the tag list consistent when the result is replaced or cleared by
        any path (File menu, another widget, ``ws.clear()``), not just this
        widget's own load handler.
        """
        result = self._ws.result or {}
        self._saved_tags = [
            tag for tag in self._saved_tags if any(key.endswith(f"_{tag}") for key in result)
        ]

    def _is_tagged(self, key: str) -> bool:
        """Whether *key* is a snapshot key created by :meth:`_on_save_result`."""
        return any(key.endswith(f"_{tag}") for tag in self._saved_tags)

    def _update_staleness(self, changed_keys: list) -> None:
        """Flag downstream products as outdated when an upstream object changes.

        A product written in the same update as its upstream (e.g. loading a
        .mat that carries both ``pln`` and ``dij``) is fresh; only an
        upstream-only change marks the product outdated.
        """
        ws = self._ws
        keys = set(changed_keys)
        if not ws.has("dij"):
            self._dij_stale = False
        elif keys & {"stf", "dij"}:
            self._dij_stale = False
        elif "pln" in keys:
            self._dij_stale = True
        if not ws.has("result"):
            self._result_stale = False
        elif "result" in keys:
            self._result_stale = False
        elif keys & {"pln", "cst", "stf", "dij"}:
            self._result_stale = True

    def _update_indicators(self) -> None:
        ws = self._ws
        self._set_indicator("load", ws.has("ct", "cst"), stale=False)
        self._set_indicator("dij", ws.has("dij"), stale=self._dij_stale)
        self._set_indicator("result", ws.has("result"), stale=self._result_stale)

    def _set_indicator(self, stage: str, done: bool, stale: bool) -> None:
        dot = self._indicators.get(stage)
        if dot is None:
            return
        if not done:
            dot.setText("○")
            dot.setStyleSheet("color: #888888;")
            dot.setToolTip("Not yet computed")
        elif stale:
            dot.setText("●")
            dot.setStyleSheet("color: #e67e22;")
            dot.setToolTip("Outdated — plan parameters changed since this was computed")
        else:
            dot.setText("●")
            dot.setStyleSheet("color: #27ae60;")
            dot.setToolTip("Up to date")

    def _update_status(self) -> None:
        ws = self._ws
        if not ws.has("ct", "cst"):
            text = "No data loaded"
        elif not ws.has("pln"):
            text = "CT and structures loaded – configure a plan to continue"
        elif not ws.has("stf", "dij"):
            text = "Ready for dose influence calculation"
        elif self._dij_stale:
            text = "Dose influence outdated — recalculate"
        elif not ws.has("result"):
            text = "Ready for optimization"
        elif self._result_stale:
            text = "Result outdated — re-optimize"
        else:
            n = len(self._saved_tags)
            suffix = f" · {n} saved snapshot{'s' if n != 1 else ''}" if n else ""
            text = f"Optimization result available{suffix}"
        self._lbl_status.setText(text)

    def _update_button_states(self) -> None:
        ws = self._ws
        has_patient = ws.has("ct", "cst")
        has_pln = ws.has("pln")
        has_stf_dij = ws.has("stf", "dij")
        has_result = ws.has("result")

        self._btn_calc_dose.setEnabled(has_patient and has_pln)
        # Importing a dose only needs a CT (for the shape check); it creates the
        # result dict itself.  Mirrors the File menu's gating.
        self._btn_import_dose.setEnabled(ws.has("ct"))
        self._btn_optimize.setEnabled(has_stf_dij)
        self._btn_recalc.setEnabled(has_result and has_stf_dij)
        self._btn_save_result.setEnabled(has_result)

    # ------------------------------------------------------------------
    # Background-thread runner
    # ------------------------------------------------------------------

    def _run_in_thread(
        self,
        fn: Callable,
        *args: Any,
        on_success: Optional[Callable] = None,
        busy_text: Optional[str] = None,
        control: Optional[ComputeControl] = None,
        **kwargs: Any,
    ) -> None:
        """Execute *fn* in a QThread.

        Calls *on_success(result)* on the main thread when it finishes, or shows
        an error dialog on failure.  *busy_text* is shown in the status label
        while the computation runs.  *control* (if given) is installed so the
        computation can be cooperatively paused/stopped from the GUI thread.
        """
        if self.is_busy:
            QMessageBox.warning(self, "Busy", "A computation is already in progress.")
            return

        self._pending_success = on_success
        # Every task gets a control so shutdown() can always request a stop.
        self._control = control if control is not None else ComputeControl()
        self._worker = Worker(
            fn, *args, report_cb=self._report_received.emit, control=self._control, **kwargs
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)

        self._worker.finished.connect(self._on_thread_finished)
        self._worker.error.connect(self._on_thread_error)
        self._thread.started.connect(self._worker.run)

        if busy_text:
            self._lbl_status.setText(busy_text)
        self._set_busy(True)
        self._thread.start()

    @Slot(object)
    def _on_thread_finished(self, result: Any) -> None:
        self._cleanup_thread()
        callback, self._pending_success = self._pending_success, None
        if callback is not None:
            callback(result)

    @Slot(object)
    def _on_thread_error(self, exc: object) -> None:
        self._cleanup_thread()
        self._pending_success = None
        self._show_error(exc)

    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        self._thread = None
        self._control = None
        self._finalize_optimization_status()
        self._set_busy(False)

    @property
    def is_busy(self) -> bool:
        """Whether a background computation is currently running."""
        return self._thread is not None and self._thread.isRunning()

    def shutdown(self, timeout_ms: int = 5000) -> None:
        """Stop a running background computation before the widget is destroyed.

        Requests a cooperative stop and waits up to *timeout_ms*.  If the
        computation does not support cancellation (e.g. dose engines), the
        thread is terminated as a last resort — this is only acceptable because
        the application is exiting; a destroyed-while-running QThread would
        abort the whole process instead.
        """
        if not self.is_busy:
            return
        if self._control is not None:
            self._control.request_stop()
        # The worker's signals target this (dying) widget; results are moot now.
        self._worker.finished.disconnect(self._on_thread_finished)
        self._worker.error.disconnect(self._on_thread_error)
        self._thread.quit()
        if not self._thread.wait(timeout_ms):
            self._thread.terminate()
            self._thread.wait()

    def _set_busy(self, busy: bool) -> None:
        for btn in self._action_buttons:
            btn.setEnabled(not busy)
        if busy:
            self._progress.setRange(0, 0)  # indeterminate until first report
        self._progress.setVisible(busy)
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
            self._update_button_states()
        # Let the host (main window) lock/unlock other input widgets.
        self.busy_changed.emit(busy)

    @Slot(object)
    def _on_compute_report(self, report: object) -> None:
        """Update the progress bar/status from a compute report (GUI thread)."""
        if isinstance(report, StatusReport):
            self._on_status_report(report)
            return
        if not isinstance(report, ProgressReport) or not report.levels:
            return
        # Drive the bar from the *combined* nested progress so each inner step
        # advances the bar by its share of the enclosing level, rather than
        # resetting to the innermost loop on every outer iteration.
        fraction = self._nested_fraction(report.levels)
        if fraction is None:
            self._progress.setRange(0, 0)  # indeterminate / pulsing
        else:
            self._progress.setRange(0, self._PROGRESS_STEPS)
            self._progress.setValue(round(fraction * self._PROGRESS_STEPS))
        self._lbl_status.setText(
            " · ".join(
                f"{lvl.name} {lvl.current}/{lvl.total}"
                if lvl.total
                else f"{lvl.name} {lvl.current}"
                for lvl in report.levels
            )
        )

    @staticmethod
    def _nested_fraction(levels) -> Optional[float]:
        """Combine nested progress levels (outer→inner) into a single fraction.

        Each level contributes its progress weighted by the share of the
        enclosing level it represents, e.g. beam 1/2 with ray 50/100 gives
        ``0.5 + 0.5 * 0.5 = 0.75``.  Returns ``None`` when the outermost level is
        indeterminate (no total), so the bar pulses instead.
        """
        fraction = 0.0
        weight = 1.0
        seen = False
        for level in levels:
            if not level.total:
                break
            fraction += weight * level.fraction
            weight /= level.total
            seen = True
        return fraction if seen else None

    def _on_status_report(self, report: StatusReport) -> None:
        """Route a metric/status report to the status window and status line."""
        data = dict(report.data)
        if report.message:
            data.setdefault("message", report.message)
        summary = report.message
        if self._opt_status_win is not None:
            summary = self._opt_status_win.update_from_report(data)
        # Keep the busy bar pulsing; show the per-iteration summary in the status line.
        if self._progress.maximum() != 0:
            self._progress.setRange(0, 0)
        self._lbl_status.setText(f"Optimizing · {summary}" if summary else "Optimizing…")

    def _show_error(self, exc: object, title: str = "Error") -> None:
        QMessageBox.critical(self, title, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Data callbacks
    # ------------------------------------------------------------------

    # Public entry points used by the main window's toolbar and File menu.
    def open_file_dialog(self) -> None:
        """Public entry point for the main-window toolbar's "Open" action."""
        self._on_load_mat()

    def load_mat(self) -> None:
        """Prompt for and load a matRad ``*.mat`` patient file."""
        self._on_load_mat()

    def load_dicom(self) -> None:
        """Prompt for and import a DICOM data set (not yet implemented)."""
        self._on_load_dicom()

    def import_dose(self) -> None:
        """Prompt for and import dose cube(s) into the current result."""
        self._on_import_dose()

    def export_binary(self) -> None:
        """Export the current result to binary files (not yet implemented)."""
        self._on_export_binary()

    def export_dicom(self) -> None:
        """Export the current result as DICOM (not yet implemented)."""
        self._on_export_dicom()

    def _on_load_mat(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load patient data", "", "MATLAB files (*.mat)"
        )
        if not filepath:
            return

        def _load() -> dict:
            # Deferred: only needed inside the worker thread, not at widget construction.
            from pyRadPlan.io import matfile  # noqa: PLC0415
            from pyRadPlan.io._patient_loader import validate_matrad_patient  # noqa: PLC0415

            mdict = matfile.load(filepath)
            return validate_matrad_patient(mdict)

        def _on_success(data: dict) -> None:
            self._ws.clear()
            self._ws.set_many(**{k: v for k, v in data.items() if v is not None})

        self._run_in_thread(_load, on_success=_on_success, busy_text="Loading patient data…")

    def _on_load_dicom(self) -> None:
        QMessageBox.information(self, "DICOM Import", "DICOM import is not yet implemented.")

    def _on_import_binary(self) -> None:
        QMessageBox.information(self, "Binary Import", "Binary import is not yet implemented.")

    def _on_refresh(self) -> None:
        self._ws.refresh()

    # ------------------------------------------------------------------
    # Dose influence callbacks
    # ------------------------------------------------------------------

    def _on_calc_dose(self) -> None:
        ct, cst, pln = self._ws.ct, self._ws.cst, self._ws.pln

        def _compute():
            # Deferred: only needed inside the worker thread, not at widget construction.
            from pyRadPlan import calc_dose_influence, generate_stf  # noqa: PLC0415

            stf = generate_stf(ct, cst, pln)
            dij = calc_dose_influence(ct, cst, stf, pln)
            return stf, dij

        def _on_success(result: tuple) -> None:
            stf, dij = result
            self._ws.set_many(stf=stf, dij=dij)

        self._run_in_thread(
            _compute, on_success=_on_success, busy_text="Calculating dose influence…"
        )

    def _on_import_dose(self) -> None:
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Import dose cube(s)", "", "NRRD files (*.nrrd)"
        )
        if not filepaths:
            return

        ct = self._ws.ct
        result: dict[str, Any] = dict(self._ws.result or {})
        errors: list[str] = []

        try:
            # Deferred: SimpleITK is only needed for this dose-import path.
            import SimpleITK as sitk  # noqa: PLC0415

            ct_shape = sitk.GetArrayFromImage(ct.cube_hu).shape if ct is not None else None
        except Exception as exc:  # noqa: BLE001
            self._show_error(exc, "Import Dose")
            return

        for fp in filepaths:
            name = os.path.splitext(os.path.basename(fp))[0]
            try:
                cube = sitk.GetArrayFromImage(sitk.ReadImage(fp))
                if ct_shape is not None and cube.shape != ct_shape:
                    errors.append(f"{name}: shape {cube.shape} does not match CT {ct_shape}")
                    continue
                key = "import_" + re.sub(r"\W+", "_", name).strip("_")
                result[key] = cube
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")

        if errors:
            QMessageBox.warning(
                self,
                "Import Dose",
                "Some files could not be imported:\n" + "\n".join(errors),
            )

        self._ws.result = result

    # ------------------------------------------------------------------
    # Optimization callbacks
    # ------------------------------------------------------------------

    def _on_optimize(self) -> None:
        ct, cst, stf, dij, pln = (
            self._ws.ct,
            self._ws.cst,
            self._ws.stf,
            self._ws.dij,
            self._ws.pln,
        )
        prev_result: dict[str, Any] = dict(self._ws.result or {})

        def _compute() -> tuple:
            # Deferred: only needed inside the worker thread, not at widget construction.
            from pyRadPlan import fluence_optimization  # noqa: PLC0415

            weights = fluence_optimization(ct, cst, stf, dij, pln)
            result = dij.compute_result_ct_grid(weights)
            return weights, result

        def _on_success(data: tuple) -> None:
            weights, result = data
            result["w"] = weights
            # Carry forward named snapshots and imported doses from a previous
            # run.  Matching by saved tag (not key prefix) keeps untagged keys
            # of the fresh result (e.g. "physical_dose_beam") from being
            # overwritten with stale data.
            for key, val in prev_result.items():
                if self._is_tagged(key) or key.startswith("import_"):
                    result.setdefault(key, val)
            self._ws.result = result

        # Live status window + cooperative pause/stop control for this optimization.
        control = ComputeControl()
        self._open_optimization_status_window(control)

        self._run_in_thread(
            _compute,
            on_success=_on_success,
            busy_text="Optimizing fluence…",
            control=control,
        )

    def _open_optimization_status_window(self, control: ComputeControl) -> None:
        """Create (or reuse) the optimization status window and bind *control*."""
        if self._opt_status_win is None:
            self._opt_status_win = OptimizationStatusWidget(parent=self)
        else:
            # Reset the curves from a previous run before starting a new one.
            self._opt_status_win.configure_metrics(DEFAULT_METRICS)
        self._opt_status_win.bind_control(control)
        self._opt_status_win.show()
        self._opt_status_win.raise_()

    def _finalize_optimization_status(self) -> None:
        if self._opt_status_win is not None:
            self._opt_status_win.finalize()

    def _on_recalc_dose(self) -> None:
        ct, cst, stf, pln = self._ws.ct, self._ws.cst, self._ws.stf, self._ws.pln
        prev_result: dict[str, Any] = dict(self._ws.result or {})
        weights = prev_result.get("w")

        if weights is None:
            QMessageBox.warning(
                self,
                "Recalculate Dose",
                "No beamlet weights found in the current result.",
            )
            return

        def _compute() -> dict:
            # Deferred: only needed inside the worker thread, not at widget construction.
            from pyRadPlan import calc_dose_forward  # noqa: PLC0415

            new_dij = calc_dose_forward(ct, cst, stf, pln, weights)
            return new_dij.compute_result_ct_grid(weights)

        def _on_success(new_result: dict) -> None:
            result = dict(prev_result)
            result.update(new_result)
            self._ws.result = result

        self._run_in_thread(_compute, on_success=_on_success, busy_text="Recalculating dose…")

    # ------------------------------------------------------------------
    # Result callbacks
    # ------------------------------------------------------------------

    def _on_save_result(self) -> None:
        pln = self._ws.pln
        default = "result"
        if pln is not None:
            try:
                default = f"{pln.radiation_mode}_{pln.num_of_fractions}fx"
            except AttributeError:
                pass

        tag, ok = QInputDialog.getText(
            self, "Save Result", "Name for this result snapshot:", text=default
        )
        if not ok or not tag.strip():
            return

        tag = re.sub(r"\W+", "_", tag.strip()).strip("_")

        result: dict[str, Any] = dict(self._ws.result or {})
        # Snapshot the current active quantities: every key that is not itself
        # a snapshot from an earlier save (reusing a tag overwrites it).
        for key in list(result):
            if not self._is_tagged(key):
                result[f"{key}_{tag}"] = result[key]

        if tag not in self._saved_tags:
            self._saved_tags.append(tag)
        self._ws.result = result
        self._update_status()

    def _on_export_binary(self) -> None:
        QMessageBox.information(self, "Export Binary", "Binary export is not yet implemented.")

    def _on_export_dicom(self) -> None:
        QMessageBox.information(self, "Export DICOM", "DICOM export is not yet implemented.")
