"""Workflow widget for the pyRadPlan GUI."""

from __future__ import annotations

import os
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

from pyRadPlan.core import ComputeControl, ProgressReport, StatusReport
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets.optimization import OptimizationStatusWidget
from pyRadPlan.gui.widgets.optimization._status_window import DEFAULT_METRICS
from .._base import WorkspaceWidget, Worker


def _human_name(fmt: str, importer_cls) -> str:
    """Best-effort human-readable label for a format, falling back to the key."""
    return getattr(importer_cls, "name", None) or fmt.upper()


def _build_load_filter() -> str:
    """Build a QFileDialog filter string covering every registered importer."""
    from pyRadPlan.io import get_available_formats, get_importer  # noqa: PLC0415

    per_format: list[str] = []
    all_patterns: list[str] = []
    for fmt in sorted(get_available_formats()):
        try:
            importer_cls = get_importer(fmt)
        except ValueError:
            continue  # export-only format
        patterns = " ".join(f"*{ext}" for ext in importer_cls.extensions)
        per_format.append(f"{_human_name(fmt, importer_cls)} ({patterns})")
        all_patterns.extend(f"*{ext}" for ext in importer_cls.extensions)

    entries = [f"All supported ({' '.join(sorted(set(all_patterns)))})", *per_format]
    entries.append("All files (*)")
    return ";;".join(entries)


def _unique_name(name: str, taken: set) -> str:
    """Return *name*, or ``name_2``/``name_3``/... if it collides with *taken*."""
    if name not in taken:
        return name
    i = 2
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


def _folder_has_images(directory: str) -> bool:
    """Return True if the folder (or an immediate subfolder) holds SimpleITK images."""
    from pyRadPlan.io import list_image_files  # noqa: PLC0415

    if list_image_files(directory):
        return True
    for entry in sorted(os.listdir(directory)):
        full = os.path.join(directory, entry)
        if os.path.isdir(full) and list_image_files(full):
            return True
    return False


def _build_save_filter() -> str:
    """Build a QFileDialog filter string for single-file (container) exporters."""
    from pyRadPlan.io import get_available_formats, get_exporter  # noqa: PLC0415

    entries: list[str] = []
    for fmt in sorted(get_available_formats()):
        try:
            exporter_cls = get_exporter(fmt)
        except ValueError:
            continue  # import-only format
        if not exporter_cls.container:
            continue  # directory-based formats are handled by folder export
        patterns = " ".join(f"*{ext}" for ext in exporter_cls.extensions)
        entries.append(f"{getattr(exporter_cls, 'name', fmt.upper())} ({patterns})")
    return ";;".join(entries)


def _image_formats() -> list[tuple[str, str]]:
    """Return (format_key, default_extension) for single-image (SimpleITK) exporters.

    These write one 3-D image per file (nrrd/nifti/meta) and are the export targets
    for individual result quantities.
    """
    from pyRadPlan.io import get_available_formats, get_exporter  # noqa: PLC0415

    formats: list[tuple[str, str]] = []
    for fmt in sorted(get_available_formats()):
        try:
            exporter_cls = get_exporter(fmt)
        except ValueError:
            continue
        # Single-image formats are the non-container ones that write plain images;
        # DICOM is directory/modality based and excluded here.
        if exporter_cls.container or fmt == "dcm":
            continue
        formats.append((fmt, exporter_cls.extensions[0]))
    return formats


def _build_image_save_filter() -> str:
    """Build a QFileDialog filter string for the single-image (SimpleITK) formats."""
    from pyRadPlan.io import get_exporter  # noqa: PLC0415

    entries: list[str] = []
    for fmt, _ext in _image_formats():
        exporter_cls = get_exporter(fmt)
        patterns = " ".join(f"*{ext}" for ext in exporter_cls.extensions)
        entries.append(f"{getattr(exporter_cls, 'name', fmt.upper())} ({patterns})")
    return ";;".join(entries)


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
        self._btn_load_file = self._add_button("Load File", self._on_load_file)
        self._btn_load_folder = self._add_button("Load Folder", self._on_load_folder)
        self._btn_import_bin = self._add_button(
            "Import from Binary", self._on_import_binary, implemented=False
        )
        self._btn_calc_dose = self._add_button("Calc. Dose Influence", self._on_calc_dose)
        self._btn_load_dij = self._add_button("Load Dij", self._on_load_dij)
        self._btn_optimize = self._add_button("Optimize", self._on_optimize)
        self._btn_recalc = self._add_button("Recalculate Dose", self._on_recalc_dose)
        self._btn_save_result = self._add_button("Save / Keep Result", self._on_save_result)

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
        # Loading a precomputed dij is always available; geometry compatibility is
        # not enforced for now.
        self._btn_load_dij.setEnabled(True)
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
        self._on_load_file()

    def load_file(self) -> None:
        """Prompt for and load a patient data file (any supported format)."""
        self._on_load_file()

    def load_folder(self) -> None:
        """Prompt for and load a DICOM (or structured image) folder."""
        self._on_load_folder()

    def load_dij(self) -> None:
        """Prompt for and load a precomputed dose-influence matrix (.mat)."""
        self._on_load_dij()

    def save_workspace(self) -> None:
        """Save the current workspace to a container file (.mat/.npz/.pkl)."""
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

    def _load_into_workspace(self, path: str, busy_text: str) -> None:
        """Load everything available from *path* and merge it into the workspace.

        Runs :func:`pyRadPlan.io.load_data` in the worker thread; on success the
        recognized pipeline objects are merged into the workspace (existing objects
        are kept, so e.g. a plan can be loaded onto an already-loaded CT).  A loaded
        structure set requires a CT (in the file or already present); otherwise it
        is skipped with a warning.  A bare ``dose`` image (formats without a matRad
        ``result``) is wrapped into a result dict so the viewer can display it.
        """

        def _load() -> dict:
            # Deferred: only needed inside the worker thread, not at widget construction.
            from pyRadPlan.io import load_data  # noqa: PLC0415

            return load_data(path)

        self._run_in_thread(_load, on_success=self._merge_loaded_data, busy_text=busy_text)

    def _merge_loaded_data(self, data: dict) -> None:
        """Merge recognized pipeline objects from a load into the workspace.

        Existing objects are kept (so e.g. a plan can be loaded onto an already
        loaded CT). A structure set requires a CT (in the data or already present);
        otherwise it is skipped with a warning. A bare ``dose`` image (formats
        without a matRad ``result``) is wrapped into a result dict for the viewer.
        """
        payload = {k: data[k] for k in self._ws.keys if data.get(k) is not None}
        if "cst" in payload and "ct" not in payload and not self._ws.has("ct"):
            QMessageBox.warning(
                self,
                "Load",
                "Load a CT before loading a structure set; the structures were skipped.",
            )
            payload.pop("cst")
        if "result" not in payload and data.get("dose") is not None:
            # sitk.Image values are rendered directly by the result widget.
            payload["result"] = {"physical_dose": data["dose"]}
        if payload:
            self._ws.set_many(**payload)

    def _on_load_file(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load patient data", "", _build_load_filter()
        )
        if not filepath:
            return

        # A bare image file carries no semantics (CT? mask? dose?); ask the user.
        # Container formats (.mat/.npz/.pkl) and DICOM keep the direct load path.
        from pyRadPlan.io.sitk_based._binary_import import IMAGE_EXTENSIONS  # noqa: PLC0415

        if filepath.lower().endswith(IMAGE_EXTENSIONS):
            self._open_image_import_dialog(filepath)
            return
        self._load_into_workspace(filepath, busy_text="Loading patient data…")

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
        mode = sel["mode"]

        if mode in ("ct_new", "ct_replace"):

            def _load() -> Any:
                from pyRadPlan.io.sitk_based import read_ct_image  # noqa: PLC0415

                return read_ct_image(filepath)

            def _on_success(new_ct: Any) -> None:
                if mode == "ct_new":
                    self._ws.clear()
                elif not sel["grid_matches"]:
                    # Structures/dose influence/results were built on the old grid.
                    self._ws.clear(["cst", "dij", "result"])
                self._ws.ct = new_ct

            self._run_in_thread(_load, on_success=_on_success, busy_text="Importing CT…")

        elif mode == "structures":
            existing = list(self._ws.cst.vois) if self._ws.has("cst") else []

            def _load() -> Any:
                from pyRadPlan.cst import validate_cst  # noqa: PLC0415
                from pyRadPlan.io.sitk_based import image_file_to_vois  # noqa: PLC0415

                new_vois = image_file_to_vois(ct, filepath)
                taken = {v.name for v in existing}
                merged = list(existing)
                for voi in new_vois:
                    unique = _unique_name(voi.name, taken)
                    taken.add(unique)
                    merged.append(
                        voi if unique == voi.name else voi.model_copy(update={"name": unique})
                    )
                return validate_cst(merged, ct)

            def _on_success(cst: Any) -> None:
                self._ws.cst = cst

            self._run_in_thread(_load, on_success=_on_success, busy_text="Importing structures…")

        elif mode == "dose":

            def _load() -> Any:
                import SimpleITK as sitk  # noqa: PLC0415

                from pyRadPlan.core.resample import resample_image  # noqa: PLC0415

                image = sitk.ReadImage(filepath)
                reference = ct.cube_hu
                if (
                    image.GetSize() != reference.GetSize()
                    or image.GetSpacing() != reference.GetSpacing()
                    or image.GetOrigin() != reference.GetOrigin()
                    or image.GetDirection() != reference.GetDirection()
                ):
                    image = resample_image(
                        image,
                        interpolator=sitk.sitkLinear,
                        target_image=reference,
                        extrapolate=0,
                    )
                return image

            def _on_success(image: Any) -> None:
                result = dict(self._ws.result or {})
                key = _unique_name(sel["name"], set(result))
                result[key] = image
                self._ws.result = result

            self._run_in_thread(_load, on_success=_on_success, busy_text="Importing dose…")

    def _on_load_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Load patient folder")
        if not directory:
            return

        # Deferred: keep the DICOM stack out of widget construction.
        from pyRadPlan.io.dicom import DicomImporter  # noqa: PLC0415

        if DicomImporter.handles_directory(directory):
            self._open_dicom_import_dialog(directory)
        elif _folder_has_images(directory):
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
        selections = dialog.selections()

        def _load() -> dict:
            from pyRadPlan.io import load_binary_patient  # noqa: PLC0415
            from pyRadPlan.io.sitk_based import read_ct_image  # noqa: PLC0415

            active = [s for s in selections if str(s.get("voi_type", "")).upper() != "IGNORED"]
            if not active:
                # CT only; nothing to assemble into a structure set.
                return {"ct": read_ct_image(ct_file)}
            ct, cst = load_binary_patient(ct_file, [], selections=selections)
            return {"ct": ct, "cst": cst}

        self._run_in_thread(
            _load, on_success=self._merge_loaded_data, busy_text="Importing binary data…"
        )

    def _open_dicom_import_dialog(self, directory: str) -> None:
        """Scan a DICOM folder, ask what to import, and load the choice.

        Both phases run in the worker thread (the scan reads one header per file,
        the load reads the pixel data), so the progress bar and status line follow
        them instead of the window freezing.
        """
        from ._dicom_import_dialog import scan_folder  # noqa: PLC0415
        from pyRadPlan.io.dicom import DicomImporter  # noqa: PLC0415

        importer = DicomImporter(directory)

        self._run_in_thread(
            scan_folder,
            importer,
            on_success=lambda catalog: self._start_dicom_import(importer, catalog),
            busy_text="Scanning DICOM folder…",
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
        sel = dialog.selection()

        def _load() -> dict:
            # One step per object, so the bar reflects the whole import and each
            # loader's own progress advances it by that step's share.
            steps = 1 + bool(sel.get("struct_file")) + bool(sel.get("load_dose"))
            data: dict[str, Any] = {}
            with importer.progress("Importing DICOM", total=steps) as step:
                data["ct"] = importer.load_ct(series_uid=sel.get("series_uid"))
                step.advance()
                if sel.get("struct_file"):
                    cst = importer.load_cst(ct=data["ct"], struct_file=sel["struct_file"])
                    if cst is not None:
                        data["cst"] = cst
                    step.advance()
                if sel.get("load_dose"):
                    # dose_file None => importer auto-selects the plan physical dose.
                    dose = importer.load_dose(dose_file=sel.get("dose_file"))
                    if dose is not None:
                        data["dose"] = dose
                    step.advance()
            return data

        self._run_in_thread(
            _load, on_success=self._merge_loaded_data, busy_text="Importing DICOM data…"
        )

    def _on_load_dij(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load dose-influence matrix", "", "MATLAB files (*.mat)"
        )
        if not filepath:
            return

        def _load() -> dict:
            from pyRadPlan.io import load_data  # noqa: PLC0415

            return load_data(filepath)

        def _on_success(data: dict) -> None:
            payload = {k: data[k] for k in ("stf", "dij") if data.get(k) is not None}
            if "dij" not in payload:
                QMessageBox.warning(
                    self, "Load Dij", "No dose-influence matrix found in the file."
                )
                return
            self._ws.set_many(**payload)

        self._run_in_thread(_load, on_success=_on_success, busy_text="Loading dose influence…")

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

    # ------------------------------------------------------------------
    # Saving / export callbacks
    # ------------------------------------------------------------------

    def _result_to_dose_image(self, result: Optional[dict], ct: Any) -> Optional[Any]:
        """Extract a physical-dose ``sitk.Image`` from the current result, if any.

        Returns *None* (rather than raising) when no usable dose is present, so
        saving still succeeds with just the CT and structures.
        """
        if not isinstance(result, dict) or ct is None:
            return None
        dose = result.get("physical_dose")
        if dose is None:
            return None

        import numpy as np  # noqa: PLC0415
        import SimpleITK as sitk  # noqa: PLC0415

        if isinstance(dose, sitk.Image):
            return dose
        arr = np.asarray(dose)
        if arr.ndim != 3:
            return None
        # A raw 3-D array here comes from an imported matRad resultGUI, stored as
        # (y, x, z); SimpleITK expects (z, y, x) (matching the matlab importer).
        image = sitk.GetImageFromArray(np.transpose(arr, (2, 0, 1)))
        if image.GetSize() == ct.cube_hu.GetSize():
            image.CopyInformation(ct.cube_hu)
        return image

    def _quantity_to_image(self, value: Any, ct: Any) -> Optional[Any]:
        """Convert a result quantity into a CT-aligned ``sitk.Image`` (or *None*)."""
        import numpy as np  # noqa: PLC0415
        import SimpleITK as sitk  # noqa: PLC0415

        if isinstance(value, sitk.Image):
            return value
        arr = np.asarray(value)
        if arr.ndim != 3:
            return None
        # A raw 3-D array here comes from an imported matRad resultGUI, stored as
        # (y, x, z); SimpleITK expects (z, y, x) (matching the matlab importer).
        image = sitk.GetImageFromArray(np.transpose(arr, (2, 0, 1)))
        if ct is not None and image.GetSize() == ct.cube_hu.GetSize():
            image.CopyInformation(ct.cube_hu)
        return image

    @staticmethod
    def _image_format_from_filter(selected: str) -> tuple[str, str]:
        """Map a chosen file-dialog filter back to an (format_key, default_ext) pair.

        Falls back to the first single-image format if the filter cannot be matched.
        """
        from pyRadPlan.io import get_exporter  # noqa: PLC0415

        formats = _image_formats()
        for fmt, ext in formats:
            name = getattr(get_exporter(fmt), "name", fmt.upper())
            if selected and (name in selected or f"*{ext}" in selected):
                return fmt, ext
        return formats[0]

    def _run_save(self, save_fn: Callable, busy_text: str, title: str) -> None:
        """Run *save_fn* in the worker thread and report the written path(s)."""

        def _on_success(written: Any) -> None:
            paths = written if isinstance(written, list) else [written]
            QMessageBox.information(self, title, "Saved:\n" + "\n".join(str(p) for p in paths))

        self._run_in_thread(save_fn, on_success=_on_success, busy_text=busy_text)

    def _on_save_workspace(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(self, "Save workspace", "", _build_save_filter())
        if not filepath:
            return
        ws = self._ws
        objects: dict[str, Any] = {"ct": ws.ct, "cst": ws.cst, "pln": ws.pln, "stf": ws.stf}
        dij = ws.dij
        if dij is not None:
            objects["dij"] = dij
        dose = self._result_to_dose_image(ws.result, ws.ct)
        if dose is not None:
            objects["dose"] = dose
        objects = {k: v for k, v in objects.items() if v is not None}

        def _save() -> Any:
            from pyRadPlan.io import save_data  # noqa: PLC0415

            return save_data(file_name=filepath, **objects)

        self._run_save(_save, "Saving workspace…", "Save Workspace")

    def _save_single_object(self, title: str, kwarg: str, obj: Any) -> None:
        """Save a single workspace object (pln/dij/cst) to a chosen ``.mat`` file."""
        if obj is None:
            QMessageBox.warning(self, title, f"No {kwarg} available to save.")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, title, "", "MATLAB files (*.mat)")
        if not filepath:
            return

        def _save() -> Any:
            from pyRadPlan.io import save_data  # noqa: PLC0415

            return save_data(file_name=filepath, format="mat", **{kwarg: obj})

        self._run_save(_save, f"{title}…", title)

    def _on_save_plan(self) -> None:
        self._save_single_object("Save Plan", "pln", self._ws.pln)

    def _on_save_dij(self) -> None:
        self._save_single_object("Save Dij", "dij", self._ws.dij)

    @staticmethod
    def _cst_export_options() -> list[tuple[str, str, Optional[str], bool, bool]]:
        """Return the CST export targets: (label, format, dicom_structure, is_dir, keeps_objectives).

        Only formats with a registered exporter are offered. Container formats
        (mat/pickle/npz) write a single file; the image/DICOM backends write a
        folder (a label map for the SimpleITK formats, a CT + RTSTRUCT/SEG series
        for DICOM). Objectives only survive in mat/pickle.
        """
        from pyRadPlan.io import get_available_formats  # noqa: PLC0415

        available = get_available_formats()
        candidates = [
            ("MATLAB (*.mat)", "mat", None, False, True),
            ("Pickle (*.pkl)", "pickle", None, False, True),
            ("NumPy (*.npz)", "npz", None, False, False),
            ("NIfTI label map (folder)", "nifti", None, True, False),
            ("NRRD label map (folder)", "nrrd", None, True, False),
            ("MetaImage label map (folder)", "meta", None, True, False),
            ("DICOM RTSTRUCT (folder)", "dcm", "rtstruct", True, False),
            ("DICOM SEG (folder)", "dcm", "seg", True, False),
        ]
        return [c for c in candidates if c[1] in available]

    def _on_save_cst(self) -> None:
        cst = self._ws.cst
        if cst is None:
            QMessageBox.warning(self, "Save CST", "No cst available to save.")
            return

        options = self._cst_export_options()
        labels = [opt[0] for opt in options]
        choice, ok = QInputDialog.getItem(self, "Save CST", "Export format:", labels, 0, False)
        if not ok or not choice:
            return
        _label, fmt, structure_format, is_dir, keeps_objectives = next(
            opt for opt in options if opt[0] == choice
        )

        # Warn before silently dropping objectives on an image/DICOM export.
        has_objectives = any(getattr(voi, "objectives", None) for voi in cst.vois)
        if has_objectives and not keeps_objectives:
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

        if is_dir:
            target = QFileDialog.getExistingDirectory(self, f"Save CST — {choice}")
        else:
            from pyRadPlan.io import get_exporter  # noqa: PLC0415

            patterns = " ".join(f"*{ext}" for ext in get_exporter(fmt).extensions)
            target, _ = QFileDialog.getSaveFileName(
                self, f"Save CST — {choice}", "", f"{choice} ({patterns})"
            )
        if not target:
            return

        def _save() -> Any:
            from pyRadPlan.io import save_data  # noqa: PLC0415

            if structure_format == "seg":
                # save_data has no structure_format hook; use the exporter directly.
                from pyRadPlan.io.dicom import DicomExporter  # noqa: PLC0415

                DicomExporter(target, structure_format="seg").save(cst=cst)
                return target
            return save_data(file_name=target, format=fmt, cst=cst)

        self._run_save(_save, "Saving CST…", "Save CST")

    def _on_save_result_to_disk(self) -> None:  # noqa: PLR0911 - guard-heavy dialog flow
        import numpy as np  # noqa: PLC0415
        import SimpleITK as sitk  # noqa: PLC0415

        result = self._ws.result
        if not isinstance(result, dict) or not result:
            QMessageBox.warning(self, "Save Result", "No result available to save.")
            return

        # Offer only scalar image quantities (skip beamlet weights and per-beam lists).
        keys = [
            k
            for k, v in result.items()
            if isinstance(v, sitk.Image) or (isinstance(v, np.ndarray) and v.ndim == 3)
        ]
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

        ct = self._ws.ct
        images = {}
        for key in dialog.selected_quantities():
            image = self._quantity_to_image(result[key], ct)
            if image is not None:
                images[key] = image
        if not images:
            QMessageBox.warning(
                self, "Save Result", "Selected quantities could not be converted to images."
            )
            return

        if len(images) == 1:
            ((key, image),) = images.items()
            filepath, selected = QFileDialog.getSaveFileName(
                self, "Save quantity", key, _build_image_save_filter()
            )
            if not filepath:
                return

            # Honor the chosen image format explicitly: an extension-less name would
            # otherwise fall back to .mat (mislabeling the quantity as physical dose)
            # or make the sitk exporter write a folder instead of a single file.
            fmt, ext = self._image_format_from_filter(selected)
            if not filepath.lower().endswith(ext.lower()):
                filepath += ext

            def _save() -> Any:
                from pyRadPlan.io import save_data  # noqa: PLC0415

                return save_data(file_name=filepath, format=fmt, dose=image)

            self._run_save(_save, "Saving quantity…", "Save Result")
            return

        # Multiple quantities: pick an output folder and an image format.
        directory = QFileDialog.getExistingDirectory(self, "Save quantities to folder")
        if not directory:
            return
        formats = _image_formats()
        labels = [f"{fmt} (*{ext})" for fmt, ext in formats]
        choice, ok = QInputDialog.getItem(self, "Save Result", "Image format:", labels, 0, False)
        if not ok:
            return
        fmt, ext = formats[labels.index(choice)]

        def _save() -> list:
            from pyRadPlan.io import save_data  # noqa: PLC0415

            written = []
            for key, image in images.items():
                target = os.path.join(directory, f"{key}{ext}")
                written.append(save_data(format=fmt, file_name=target, dose=image))
            return written

        self._run_save(_save, "Saving quantities…", "Save Result")
