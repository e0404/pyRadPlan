"""Workflow widget for the pyRadPlan GUI."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
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

from pyRadPlan.app import Operation, PreconditionError, Session
from pyRadPlan.app._io import (
    entry_for_filter,
    export_entries,
    export_filter,
    folder_kind,
    image_keys,
    is_image_file,
    load_filter,
    path_label,
    writes_folder,
)
from pyRadPlan.app._results import default_tag
from pyRadPlan.app._stages import available_actions, nested_fraction, progress_detail
from pyRadPlan.app._stages import status_text
from pyRadPlan.core import (
    ComputeCancelledError,
    ComputeControl,
    ProgressReport,
    StatusReport,
    WarningReport,
)
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets.optimization import OptimizationStatusWidget
from pyRadPlan.gui.widgets.optimization._status_window import DEFAULT_METRICS
from .._base import WorkspaceWidget, Worker
from .._display_names import bio_model_display_name, quantity_display_name
from pyRadPlan.biology._editing import tissue_issues_text

logger = logging.getLogger(__name__)

#: Status dot of each column header -> the session stage it shows.
_INDICATOR_STAGES = {"load": "data", "dij": "dij", "result": "optimization"}

#: Status line of a current result, followed by the number of further results.
_RESULT_AVAILABLE = "Optimization result available"


def _file_stem(key: str) -> str:
    """File name stem for the result key *key* (``"<result>/<quantity>"``)."""
    return re.sub(r"[^\w.-]+", "_", key)


class WorkflowWidget(WorkspaceWidget):
    """Guides the user through the pyRadPlan treatment-planning workflow.

    Binds to a :class:`~pyRadPlan.gui.workspace.WorkspaceManager` and enables or
    disables each workflow step depending on which pipeline objects
    (ct, cst, pln, stf, dij, result) are currently available.  The workflow rules
    (staleness, available steps, status text) and the steps themselves are those of the
    manager's :class:`pyRadPlan.app.Session`; this widget asks the user, shows progress
    and runs the long computations (dose influence, optimization, forward dose) in a
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
        self._pending_error: Optional[Callable[[BaseException], bool]] = None
        #: Warnings reported by the running computation, shown once it ends.
        self._pending_warnings: list[str] = []
        self._opt_status_win: Optional[OptimizationStatusWidget] = None
        self._indicators: dict[str, QLabel] = {}
        #: Editors whose unapplied changes are offered before a computation, as
        #: ``(name, has_pending, apply)``; see :meth:`add_pending_changes_source`.
        self._pending_sources: list[tuple[str, Callable[[], bool], Callable[[], bool]]] = []

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
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setToolTip(
            "Stop the running calculation.\n"
            "An optimization keeps its last iterate; other calculations are discarded.\n"
            "Takes effect at the next progress step (e.g. the next ray)."
        )
        self._btn_stop.clicked.connect(self._on_stop)
        row.addWidget(self._btn_stop)
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
        self._btn_load_file = self._add_button("Load File", self._on_load_file)
        self._btn_load_folder = self._add_button("Load Folder", self._on_load_folder)
        self._btn_import_bin = self._add_button(
            "Import from Binary", self._on_import_binary, implemented=False
        )
        self._btn_calc_dose = self._add_button("Calc. Dose Influence", self._on_calc_dose)
        self._btn_load_dij = self._add_button("Load Dij", self._on_load_dij)
        self._btn_optimize = self._add_button("Optimize", self._on_optimize)
        self._btn_recalc = self._add_button("Recalculate Dose", self._on_recalc_dose)
        self._btn_save_result = self._add_button(
            "Snapshot Result",
            self._on_save_result,
            tooltip="Keep a copy of the current result in the workspace. "
            "Use File > Save Result… to write it to disk.",
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
        grid.addWidget(self._btn_load_file, 1, 0)
        grid.addWidget(self._btn_load_folder, 2, 0)
        grid.addWidget(self._btn_import_bin, 3, 0)
        # Col 1: dose influence
        grid.addWidget(self._btn_calc_dose, 1, 1)
        grid.addWidget(self._btn_load_dij, 2, 1)
        # Col 2: result computation (optimization, recalculation)
        grid.addWidget(self._btn_optimize, 1, 2)
        grid.addWidget(self._btn_recalc, 2, 2)
        # Col 3: keep result in-memory (file saves live in the File menu)
        grid.addWidget(self._btn_save_result, 1, 3)
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

    @property
    def session(self) -> Session:
        """The workflow session of the workspace, whose rules and steps this widget uses."""
        return self._ws.session

    def _do_update(self, changed_keys: list) -> None:
        # This widget has no editable fields, only buttons/labels derived from
        # which pipeline objects exist.  It therefore *intentionally* reacts to
        # its own workspace writes (no ``hold_updates``) so that, e.g., the
        # "Save Result" button enables immediately after optimization finishes.
        # The session has seen the change already (it observes the store first).
        self._update_status()
        self._update_button_states()
        self._update_indicators()

    def _update_indicators(self) -> None:
        stages = self.session.stages()
        for indicator, stage in _INDICATOR_STAGES.items():
            self._set_indicator(indicator, stages[stage])

    def _set_indicator(self, stage: str, status: str) -> None:
        """Show *status* (``missing`` / ``done`` / ``stale``) on the dot of *stage*."""
        dot = self._indicators.get(stage)
        if dot is None:
            return
        if status == "missing":
            dot.setText("○")
            dot.setStyleSheet("color: #888888;")
            dot.setToolTip("Not yet computed")
        elif status == "stale":
            dot.setText("●")
            dot.setStyleSheet("color: #e67e22;")
            dot.setToolTip("Outdated — inputs changed since this was computed")
        else:
            dot.setText("●")
            dot.setStyleSheet("color: #27ae60;")
            dot.setToolTip("Up to date")

    def _update_status(self) -> None:
        text = status_text(self._ws.present(), self.session.tracker)
        n = len(self._ws.results) - 1
        if n > 0 and text == _RESULT_AVAILABLE:
            text += f" · {n} further result{'s' if n != 1 else ''}"
        self._lbl_status.setText(text)

    def _update_button_states(self) -> None:
        # Availability by the stored objects only: while a computation runs the busy
        # cycle disables the buttons, and they are updated when it ends.
        present = self._ws.present()
        actions = available_actions(present)
        self._btn_calc_dose.setEnabled(actions["calc_dose_influence"])
        # Loading a precomputed dij is always available; geometry compatibility is
        # not enforced for now.
        self._btn_load_dij.setEnabled(True)
        self._btn_optimize.setEnabled(actions["optimize"])
        self._btn_recalc.setEnabled(actions["recalc"])
        self._btn_save_result.setEnabled("result" in present)

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
        on_error: Optional[Callable[[BaseException], bool]] = None,
        **kwargs: Any,
    ) -> None:
        """Execute *fn* in a QThread.

        Calls *on_success(result)* on the main thread when it finishes, or shows
        an error dialog on failure.  *busy_text* is shown in the status label
        while the computation runs.  *control* (if given) is installed so the
        computation can be cooperatively paused/stopped from the GUI thread.
        *on_error(exc)* (if given) is called on the main thread when the computation
        fails or is stopped; returning True means it explained the error itself.
        """
        if self.is_busy:
            QMessageBox.warning(self, "Busy", "A computation is already in progress.")
            return

        self._pending_success = on_success
        self._pending_error = on_error
        self._pending_warnings = []
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
        self._pending_error = None
        callback, self._pending_success = self._pending_success, None
        if callback is not None:
            callback(result)
        self._show_pending_warnings()

    @Slot(object)
    def _on_thread_error(self, exc: object) -> None:
        self._cleanup_thread()
        self._pending_success = None
        handler, self._pending_error = self._pending_error, None
        handled = bool(handler(exc)) if handler is not None else False
        self._show_pending_warnings()
        if isinstance(exc, ComputeCancelledError):
            self._lbl_status.setText("Stopped")
            return
        if not handled:
            self._show_error(exc)

    def _run_operation(
        self,
        operation: Operation,
        on_done: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[BaseException], bool]] = None,
    ) -> None:
        """Run a prepared session operation: compute in the worker thread, commit here.

        The session marks itself busy (:meth:`~pyRadPlan.app.Session.begin`), computes in
        the worker thread with the operation's control (:meth:`~pyRadPlan.app.Session.run`)
        and writes the result to the workspace on the GUI thread
        (:meth:`~pyRadPlan.app.Session.finish`), then *on_done* receives what the commit
        returned. A failed or stopped run aborts the operation; *on_error* may explain the
        error instead of the default dialog by returning True.
        """
        if self.is_busy:
            QMessageBox.warning(self, "Busy", "A computation is already in progress.")
            return
        session = self.session
        session.begin(operation)

        def finished(value: Any) -> None:
            try:
                committed = session.finish(operation, value)
            except Exception as exc:  # noqa: BLE001 - the session has aborted the operation
                self._update_status()
                self._show_error(exc, operation.label)
                return
            # A step writing nothing (a save) leaves no workspace change to refresh it.
            self._update_status()
            if on_done is not None:
                on_done(committed)

        def failed(exc: BaseException) -> bool:
            session.abort(operation, exc)
            self._update_status()
            return on_error(exc) if on_error is not None else False

        try:
            self._run_in_thread(
                lambda: session.run(operation),
                on_success=finished,
                on_error=failed,
                control=operation.control,
                busy_text=f"{operation.label}…",
            )
        except BaseException as exc:
            session.abort(operation, exc)
            raise

    def _prepare(self, step: str, action: str, **params: Any) -> Optional[Operation]:
        """Prepare a session operation, answering unmet preconditions with dialogs.

        An unsupported biological model offers the physical dose only and prepares the
        step again with it; other preconditions are explained. Returns ``None`` if the
        step must not run.
        """
        try:
            return self.session.prepare(action, **params)
        except PreconditionError as exc:
            if exc.code != "bio_model_unsupported":
                self._explain_precondition(step, exc)
                return None
            name = bio_model_display_name(exc.data["model"])
            if self._ask_physical_dose_only(step, name, exc.data["issues"]) != "physical":
                return None
        except Exception as exc:  # noqa: BLE001 - surface, never crash
            self._show_error(exc, step)
            return None
        try:
            return self.session.prepare(action, physical_only=True, **params)
        except Exception as exc:  # noqa: BLE001 - surface, never crash
            self._show_error(exc, step)
            return None

    def _explain_precondition(self, step: str, exc: PreconditionError) -> None:
        """Show why *step* cannot run, in the words of the workflow's dialogs."""
        if exc.code == "busy":
            QMessageBox.warning(self, "Busy", "A computation is already in progress.")
        elif exc.code == "invalid_reference":
            invalid = [issue for issue in exc.data["issues"] if issue.kind == "invalid_reference"]
            QMessageBox.critical(
                self,
                step,
                "Invalid reference photon parameters α_x / β_x, which no biological "
                f"model can be evaluated with:\n\n{tissue_issues_text(invalid)}\n\n"
                "Correct the tissue parameters of these structures first.",
            )
        elif exc.code == "biological_objectives":
            objectives = dict.fromkeys(
                f"{name}: {quantity_display_name(identifier)}"
                for name, identifier in exc.data["objectives"]
            )
            QMessageBox.warning(
                self,
                step,
                "These objectives are evaluated on biological quantities:\n\n"
                + "\n".join(f"• {entry}" for entry in objectives)
                + "\n\nbut the dose influence was calculated without a biological model, so "
                "it holds the physical dose only. Select a biological model in the plan and "
                "calculate the dose influence again, or evaluate the objectives on the "
                "physical dose.",
            )
        elif exc.code == "no_weights":
            QMessageBox.warning(self, step, "No beamlet weights found in the current result.")
        elif exc.code == "beam_geometry_changed":
            QMessageBox.warning(
                self,
                step,
                "The plan's beam angles or beam count differ from the stored steering "
                "information. Calculate Dose Influence and optimize again before "
                "recalculating dose.",
            )
        else:
            QMessageBox.warning(self, step, str(exc))

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
        computation does not support cancellation (e.g. an external Monte Carlo run), the
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

    def _on_stop(self) -> None:
        """Request a cooperative stop of the running computation."""
        if self._control is None:
            return
        self._control.request_stop()
        self._btn_stop.setEnabled(False)
        self._lbl_status.setText("Stopping…")

    def _set_busy(self, busy: bool) -> None:
        for btn in self._action_buttons:
            btn.setEnabled(not busy)
        self._btn_stop.setEnabled(busy)
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
        if isinstance(report, WarningReport):
            if report.message not in self._pending_warnings:
                self._pending_warnings.append(report.message)
            return
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
        self._lbl_status.setText(progress_detail(report.levels))

    _nested_fraction = staticmethod(nested_fraction)

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

    def _show_pending_warnings(self) -> None:
        warnings, self._pending_warnings = self._pending_warnings, []
        if warnings:
            QMessageBox.warning(self, "Warning", "\n\n".join(warnings))

    def _show_error(self, exc: object, title: str = "Error") -> None:
        QMessageBox.critical(self, title, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Data callbacks
    # ------------------------------------------------------------------

    # Public entry points used by the main window's toolbar and File menu.
    def open_file_dialog(self) -> None:
        """Public entry point for the main-window toolbar's "Open" action."""
        self._on_load_file()

    def load_file(self) -> None:
        """Prompt for and load a patient data file (any supported format)."""
        self._on_load_file()

    def load_folder(self) -> None:
        """Prompt for and load a DICOM (or structured image) folder."""
        self._on_load_folder()

    def load_phantom(self, name: str) -> None:
        """Load the bundled phantom *name* (see :func:`pyRadPlan.io.available_phantoms`)."""
        operation = self._prepare("Load", "load_phantom", name=name)
        if operation is not None:
            self._run_operation(operation, on_done=self._show_load_warnings)

    def load_dij(self) -> None:
        """Prompt for and load a precomputed dose-influence matrix (.mat)."""
        self._on_load_dij()

    def save_workspace(self) -> None:
        """Save the complete workspace to a pickle file."""
        self._on_save_workspace()

    def save_plan(self) -> None:
        """Save the current plan (pln) to a .mat file."""
        self._on_save_plan()

    def save_dij(self) -> None:
        """Save the current dose-influence matrix (dij) to a .mat file."""
        self._on_save_dij()

    def save_cst(self) -> None:
        """Save the current structure set (cst, incl. objectives) to a .mat file."""
        self._on_save_cst()

    def save_result(self) -> None:
        """Export selected result quantities as image files."""
        self._on_save_result_to_disk()

    def _load_into_workspace(self, path: str, label: str) -> None:
        """Load everything available from *path* and merge it into the workspace.

        Runs :func:`pyRadPlan.io.load_data` in the worker thread; on success the
        recognized pipeline objects are merged into the workspace by the session (see
        :meth:`~pyRadPlan.app.Session.merge_loaded`). *label* is shown while it runs.
        """
        operation = self._prepare("Load", "load_path", path=path, label=label)
        if operation is not None:
            self._run_operation(operation, on_done=self._show_load_warnings)

    def _show_load_warnings(self, warnings: Optional[list[str]]) -> None:
        for warning in warnings or []:
            QMessageBox.warning(self, "Load", warning)

    def _on_load_file(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(self, "Load patient data", "", load_filter())
        if not filepath:
            return

        # A bare image file carries no semantics (CT? mask? dose?); ask the user.
        # Container formats (.mat/.npz/.pkl) and DICOM keep the direct load path.
        if is_image_file(filepath):
            self._open_image_import_dialog(filepath)
            return
        self._load_into_workspace(filepath, "Loading patient data")

    def _open_image_import_dialog(self, filepath: str) -> None:
        """Ask what a bare image file represents and import it accordingly."""
        from ._image_import_dialog import ImageImportDialog  # noqa: PLC0415

        ct = self._ws.ct
        dialog = ImageImportDialog(
            filepath,
            has_ct=ct is not None,
            ct_image=ct.cube_hu if ct is not None else None,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        sel = dialog.selection()
        operation = self._prepare(
            "Import Image", "import_image", path=filepath, mode=sel["mode"], name=sel.get("name")
        )
        if operation is not None:
            self._run_operation(operation)

    def _on_load_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Load patient folder")
        if not directory:
            return

        kind = folder_kind(directory)
        if kind == "dicom":
            self._open_dicom_import_dialog(directory)
        elif kind == "images":
            self._open_binary_import_dialog(directory)
        else:
            QMessageBox.warning(
                self,
                "Load Folder",
                "No DICOM series or supported image files were found in this folder.",
            )

    def _open_binary_import_dialog(self, directory: str) -> None:
        """Open the binary (CT + per-file masks) import dialog and load the choice."""
        from ._binary_import_dialog import BinaryImportDialog  # noqa: PLC0415

        dialog = BinaryImportDialog(directory, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        ct_file = dialog.ct_file()
        if not ct_file:
            QMessageBox.warning(self, "Import", "No CT file selected.")
            return
        operation = self._prepare(
            "Import", "import_binary", ct_file=ct_file, selections=dialog.selections()
        )
        if operation is not None:
            self._run_operation(operation, on_done=self._show_load_warnings)

    def _open_dicom_import_dialog(self, directory: str) -> None:
        """Scan a DICOM folder, ask what to import, and load the choice.

        Both phases run in the worker thread (the scan reads one header per file,
        the load reads the pixel data), so the progress bar and status line follow
        them instead of the window freezing.
        """
        operation = self._prepare("Import DICOM", "scan_dicom", directory=directory)
        if operation is not None:
            self._run_operation(
                operation,
                on_done=lambda scan: self._start_dicom_import(scan["importer"], scan["catalog"]),
            )

    def _start_dicom_import(self, importer: Any, catalog: dict) -> None:
        """Ask which series/structures/dose to import from a scanned folder, then load."""
        from ._dicom_import_dialog import DicomImportDialog  # noqa: PLC0415

        if not catalog["series"]:
            QMessageBox.warning(self, "Import DICOM", "No CT series found in this folder.")
            return

        dialog = DicomImportDialog(importer, self, catalog=catalog)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        operation = self._prepare(
            "Import DICOM", "import_dicom", importer=importer, selection=dialog.selection()
        )
        if operation is not None:
            self._run_operation(operation, on_done=self._show_load_warnings)

    def _on_load_dij(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load dose-influence matrix", "", "MATLAB files (*.mat)"
        )
        if not filepath:
            return

        def explain(exc: BaseException) -> bool:
            if not (isinstance(exc, PreconditionError) and exc.code == "no_dij"):
                return False
            QMessageBox.warning(self, "Load Dij", "No dose-influence matrix found in the file.")
            return True

        operation = self._prepare(
            "Load Dij", "load_dij", path=filepath, label="Loading dose influence"
        )
        if operation is not None:
            self._run_operation(operation, on_error=explain)

    def _on_import_binary(self) -> None:
        QMessageBox.information(self, "Binary Import", "Binary import is not yet implemented.")

    def _on_refresh(self) -> None:
        self._ws.refresh()

    def add_pending_changes_source(
        self, name: str, has_pending: Callable[[], bool], apply: Callable[[], bool]
    ) -> None:
        """Offer to apply an editor's pending changes before a step that uses the plan.

        Parameters
        ----------
        name:
            The editor as named to the user, e.g. ``"plan"``.
        has_pending:
            Whether the editor has changes that are not applied yet.
        apply:
            Applies them; returns False if that failed (the editor reports why).
        """
        self._pending_sources.append((name, has_pending, apply))

    def _ask_pending_changes(self, step: str, names: list[str], outdates_input: bool) -> str:
        """Ask how to handle unapplied changes: ``"apply"``, ``"continue"`` or ``"cancel"``."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(step)
        box.setText(f"The {' and '.join(names)} has changes that are not applied yet.")
        if outdates_input:
            box.setInformativeText(
                "Applying them outdates the dose influence, so it has to be "
                f'recalculated before "{step}" can run.'
            )
            apply_btn = box.addButton("Apply changes", QMessageBox.ButtonRole.AcceptRole)
        else:
            apply_btn = box.addButton("Apply and continue", QMessageBox.ButtonRole.AcceptRole)
        continue_btn = box.addButton("Continue without applying", QMessageBox.ButtonRole.NoRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(apply_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is apply_btn:
            return "apply"
        if clicked is continue_btn:
            return "continue"
        return "cancel"

    def _resolve_pending_changes(self, step: str, outdates_input: bool = False) -> bool:
        """Settle unapplied editor changes before *step*; False if it must not run.

        *outdates_input* marks steps working from the dose influence (e.g. the
        optimization): applying a plan change outdates it, so such a step stops
        after applying and the status asks for a recalculation.
        """
        pending = [source for source in self._pending_sources if source[1]()]
        if not pending:
            return True
        choice = self._ask_pending_changes(step, [name for name, *_ in pending], outdates_input)
        if choice == "continue":
            return True
        if choice == "cancel":
            return False
        if not all(apply() for _, _, apply in pending):
            return False
        return not outdates_input

    # ------------------------------------------------------------------
    # Dose influence callbacks
    # ------------------------------------------------------------------

    def _ask_physical_dose_only(self, step: str, model_name: str, issues: list) -> str:
        """
        Offer to calculate the physical dose only, for the errors among tissue issues.

        Returns ``"physical"`` if the user accepts the physical dose and ``"cancel"``
        otherwise.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(step)
        box.setText(
            f"The biological model '{model_name}' cannot be evaluated for the structures, "
            "machine and dose engine of the plan:"
        )
        box.setInformativeText(
            tissue_issues_text(issues) + "\n\nThe plan's biological model can be set to "
            "none to calculate the physical dose only, or the calculation cancelled to fix "
            "the tissue parameters or choose another model first."
        )
        physical_btn = box.addButton(
            "Calculate physical dose only", QMessageBox.ButtonRole.AcceptRole
        )
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_btn)
        box.exec()
        return "physical" if box.clickedButton() is physical_btn else "cancel"

    def _show_tissue_warnings(self, step: str, model_name: str, warnings: list) -> None:
        """
        Log tissue warnings and show them in the status line, without asking anything.

        The log reaches the log panel; the dose engine reports the warnings again once the
        calculation has run, like any engine warning.
        """
        if not warnings:
            return
        for issue in warnings:
            logger.warning("%s, biological model '%s': %s", step, model_name, issue)
        self._lbl_status.setText(
            f"{len(warnings)} tissue warning(s): evaluated with the base data's model"
        )

    def _run_calculation(self, step: str, action: str) -> None:
        """Prepare a step using the plan's biological model and run it.

        The session runs the check the plan's dose engine (its own, or the default engine
        of the radiation mode) runs, before *step* starts. Invalid reference α_x / β_x
        block the calculation; other errors offer to calculate the physical dose only,
        which sets the plan's model to ``none`` in the workspace. Warnings (e.g. MKM
        tissues matching mMKM kernels, which are evaluated with the base data's model)
        never ask: they are logged, shown in the status line, and the calculation runs with
        the biological model, whose dose engine reports them again.
        """
        operation = self._prepare(step, action)
        if operation is None:
            return
        if operation.warnings:
            name = bio_model_display_name(self._ws.pln.bio_model)
            self._show_tissue_warnings(step, name, operation.warnings)
        self._run_operation(operation)

    def _on_calc_dose(self) -> None:
        if not self._resolve_pending_changes("Calculate Dose Influence"):
            return
        self._run_calculation("Calculate Dose Influence", "calc_dose_influence")

    # ------------------------------------------------------------------
    # Optimization callbacks
    # ------------------------------------------------------------------

    def _on_optimize(self) -> None:
        if not self._resolve_pending_changes("Optimize", outdates_input=True):
            return
        # Biological objectives need a dose influence calculated with a biological model.
        operation = self._prepare("Optimize", "optimize")
        if operation is None:
            return
        # Live status window + cooperative pause/stop control for this optimization.
        self._open_optimization_status_window(operation.control)
        self._run_operation(operation)

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
        # The beams follow the plan's isocenter (moved on a copy, which outdates the dij);
        # only the active result is replaced, snapshots and imported doses are kept.
        if not self._resolve_pending_changes("Recalculate Dose", outdates_input=True):
            return
        self._run_calculation("Recalculate Dose", "recalc")

    # ------------------------------------------------------------------
    # Result callbacks
    # ------------------------------------------------------------------

    def _on_save_result(self) -> None:
        default = default_tag(self._ws.pln)
        tag, ok = QInputDialog.getText(
            self, "Snapshot Result", "Name for this result snapshot:", text=default
        )
        if not ok or not tag.strip():
            return
        # A result of its own that shares the images; reusing a name overwrites it.
        try:
            self.session.snapshot(tag)
        except (ValueError, PreconditionError) as exc:
            QMessageBox.warning(self, "Snapshot Result", str(exc))
            return
        self._update_status()

    # ------------------------------------------------------------------
    # Saving / export callbacks
    # ------------------------------------------------------------------

    def _save(self, title: str, obj: str, fmt: str, path: str, **params: Any) -> None:
        """Prepare the session's save of *obj* to *path* and run it; refusals are explained."""
        try:
            operation = self.session.prepare(
                "save_path", object=obj, format=fmt, path=path, **params
            )
        except (PreconditionError, ValueError) as exc:
            QMessageBox.warning(self, title, str(exc))
            return
        self._run_save(operation, title)

    def _run_save(self, operation: Operation, title: str) -> None:
        """Run a save operation and report the written path(s)."""

        def _on_done(written: list) -> None:
            QMessageBox.information(self, title, "Saved:\n" + "\n".join(str(p) for p in written))

        self._run_operation(operation, on_done=_on_done)

    def _on_save_workspace(self) -> None:
        if not self._ws.has("ct"):
            QMessageBox.warning(self, "Save Workspace", "Load a CT before saving a workspace.")
            return
        if not self._resolve_pending_changes("Save Workspace"):
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save workspace", "", export_filter(export_entries("workspace"))
        )
        if filepath:
            self._save("Save Workspace", "workspace", "pickle", filepath)

    def _save_single_object(self, title: str, obj: str) -> None:
        """Save a single workspace object (pln/dij) to a chosen ``.mat`` file."""
        if getattr(self._ws, obj) is None:
            QMessageBox.warning(self, title, f"No {obj} available to save.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, title, "", export_filter(export_entries(obj, ("mat",)))
        )
        if filepath:
            self._save(title, obj, "mat", filepath, label=title)

    def _on_save_plan(self) -> None:
        if not self._resolve_pending_changes("Save Plan"):
            return
        self._save_single_object("Save Plan", "pln")

    def _on_save_dij(self) -> None:
        self._save_single_object("Save Dij", "dij")

    def _on_save_cst(self) -> None:
        cst = self._ws.cst
        if cst is None:
            QMessageBox.warning(self, "Save CST", "No cst available to save.")
            return

        entries = export_entries("cst")
        labels = [path_label(entry) for entry in entries]
        choice, ok = QInputDialog.getItem(self, "Save CST", "Export format:", labels, 0, False)
        if not ok or not choice:
            return
        entry = entries[labels.index(choice)]

        # Warn before silently dropping objectives on an image/DICOM export.
        has_objectives = any(getattr(voi, "objectives", None) for voi in cst.vois)
        if has_objectives and not entry.keeps_objectives:
            if (
                QMessageBox.warning(
                    self,
                    "Save CST",
                    f"The {choice} format cannot store objectives; only the "
                    "structure masks will be exported. Continue?",
                    QMessageBox.Ok | QMessageBox.Cancel,
                )
                != QMessageBox.Ok
            ):
                return

        if writes_folder(entry):
            target = QFileDialog.getExistingDirectory(self, f"Save CST — {choice}")
        else:
            target, _ = QFileDialog.getSaveFileName(
                self, f"Save CST — {choice}", "", export_filter([entry])
            )
        if target:
            self._save(
                "Save CST",
                "cst",
                entry.format,
                target,
                structure_format=entry.structure_format,
                label="Saving CST",
            )

    def _on_save_result_to_disk(self) -> None:
        results = self._ws.result_entries()
        if not results:
            QMessageBox.warning(self, "Save Result", "No result available to save.")
            return

        # Offer only whole-volume quantities (skip beamlet weights and per-beam lists).
        keys = image_keys(results)
        if not keys:
            QMessageBox.warning(
                self, "Save Result", "The result has no exportable image quantities."
            )
            return

        # Deferred: keeps the Qt result-widget stack out of workflow construction.
        from ..result._save_result_dialog import SaveResultDialog  # noqa: PLC0415

        dialog = SaveResultDialog(keys, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_quantities()
        if not selected:
            QMessageBox.warning(self, "Save Result", "No quantities were selected.")
            return

        entries = export_entries("result", ("nifti", "nrrd", "meta"))
        if len(selected) == 1:
            filepath, chosen = QFileDialog.getSaveFileName(
                self, "Save quantity", _file_stem(selected[0]), export_filter(entries)
            )
            if filepath:
                entry = entry_for_filter(chosen, entries)
                self._save(
                    "Save Result",
                    "result",
                    entry.format,
                    filepath,
                    quantities=selected,
                    label="Saving quantity",
                )
            return

        # Multiple quantities: pick an output folder and an image format.
        directory = QFileDialog.getExistingDirectory(self, "Save quantities to folder")
        if not directory:
            return
        labels = [entry.label for entry in entries]
        choice, ok = QInputDialog.getItem(self, "Save Result", "Image format:", labels, 0, False)
        if not ok:
            return
        self._save(
            "Save Result",
            "result",
            entries[labels.index(choice)].format,
            directory,
            quantities=selected,
            label="Saving quantities",
        )
