"""A small, reusable dialog for running a pydantic-ai task from the GUI.

The dialog is deliberately agnostic about *what* the task does: it is handed an
:class:`AiTask` describing the model choices, the (read-only) system prompt and
data context that will be sent, an editable user prompt, and callables that run
the task and apply its result.  Concrete tasks (suggesting objectives, beam
angles, …) are assembled by the widgets that launch the dialog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Slot
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.ai.agents import pop_last_run_usage

from .._base import Worker

#: Keeps detached (still-running) worker threads alive until they finish, so
#: closing the dialog mid-request neither blocks the GUI nor lets a running
#: QThread be garbage-collected (which would abort the process).
_DETACHED_THREADS: set = set()


@dataclass
class AiTask:
    """Description of an LLM task to be run from :class:`AiTaskDialog`.

    Parameters
    ----------
    title:
        Window title.
    system_prompt:
        The system prompt that will be sent (shown read-only).
    context_text:
        A human-readable summary of the data context that will be sent (shown
        read-only).  Must not contain numpy arrays or other large payloads.
    run:
        Callable ``(model, treatment_site, additional_context) -> result`` run in
        a background thread.
    apply:
        Callable ``(result) -> None`` invoked on the GUI thread to apply a
        successful result (e.g. write it back to the workspace).
    summarize:
        Callable ``(result) -> str`` producing a short success message.
    default_site, default_context:
        Initial values for the editable treatment-site / additional-context
        fields.
    """

    title: str
    system_prompt: str
    context_text: str
    run: Callable[[str, str, str], Any]
    apply: Callable[[Any], None]
    summarize: Callable[[Any], str] = field(default=lambda _result: "Done.")
    default_site: str = ""
    default_context: str = ""


class AiTaskDialog(QDialog):
    """Dialog to review, configure and run an :class:`AiTask`.

    Parameters
    ----------
    task:
        The task to run.
    models:
        Selectable model identifiers (e.g. from
        :func:`pyRadPlan.ai.agents.available_models`).  The combo is editable so
        any model string can be entered.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        task: AiTask,
        models: list[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._task = task
        self._thread: Optional[QThread] = None
        self._worker: Optional[Worker] = None
        self.setWindowTitle(task.title)
        self.resize(560, 600)
        self._setup_ui(models)

    def _setup_ui(self, models: list[str]) -> None:
        root = QVBoxLayout(self)

        form = QFormLayout()
        self._cmb_model = QComboBox()
        self._cmb_model.setEditable(True)
        self._cmb_model.addItems(models)
        if not models:
            self._cmb_model.setEditText("")
        form.addRow("Model:", self._cmb_model)

        self._txt_site = QLineEdit(self._task.default_site)
        self._txt_site.setPlaceholderText("e.g. prostate, head and neck")
        form.addRow("Treatment site:", self._txt_site)
        root.addLayout(form)

        root.addWidget(QLabel("User prompt (additional context, editable):"))
        self._txt_context = QPlainTextEdit(self._task.default_context)
        self._txt_context.setMaximumHeight(90)
        root.addWidget(self._txt_context)

        root.addWidget(QLabel("System prompt (read-only):"))
        self._txt_system = QPlainTextEdit(self._task.system_prompt.strip())
        self._txt_system.setReadOnly(True)
        self._txt_system.setMaximumHeight(140)
        root.addWidget(self._txt_system)

        root.addWidget(QLabel("Data context that will be sent (read-only):"))
        self._txt_data = QPlainTextEdit(self._task.context_text)
        self._txt_data.setReadOnly(True)
        root.addWidget(self._txt_data, 1)

        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        root.addWidget(self._lbl_status)

        self._buttons = QDialogButtonBox()
        self._btn_run = self._buttons.addButton("Run", QDialogButtonBox.AcceptRole)
        self._buttons.addButton(QDialogButtonBox.Close)
        self._btn_run.clicked.connect(self._on_run)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        model = self._cmb_model.currentText().strip()
        if not model:
            self._set_status("Select or enter a model first.", error=True)
            return
        site = self._txt_site.text().strip()
        context = self._txt_context.toPlainText().strip()

        self._set_running(True)
        self._set_status(f"Querying {model}…")
        pop_last_run_usage()  # discard usage left over from an earlier/detached run

        self._worker = Worker(self._task.run, model, site, context)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    @Slot(object)
    def _on_finished(self, result: Any) -> None:
        self._cleanup_thread()
        try:
            self._task.apply(result)
        except Exception as exc:  # noqa: BLE001
            self._set_running(False)
            self._set_status(f"Failed to apply result: {exc}", error=True)
            return
        self._set_running(False)
        status = self._task.summarize(result)
        usage = pop_last_run_usage()
        if usage:
            status = f"{status}\n{usage}"
        self._set_status(status)

    @Slot(object)
    def _on_error(self, exc: object) -> None:
        self._cleanup_thread()
        self._set_running(False)
        self._set_status(f"{type(exc).__name__}: {exc}", error=True)

    def _cleanup_thread(self) -> None:
        """Tear down an idle worker thread (the run has finished or errored)."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def _detach_running_thread(self) -> None:
        """Let a still-running request finish in the background.

        The worker is blocked inside a synchronous LLM call that cannot be
        interrupted; waiting on it here would freeze the whole GUI until the
        request (or its network timeout) completes.  Instead the thread is
        detached from the dialog, its result is discarded, and it disposes of
        itself once the call returns.
        """
        if self._thread is None or not self._thread.isRunning():
            return
        thread, worker = self._thread, self._worker
        self._thread = None
        self._worker = None
        worker.finished.disconnect(self._on_finished)
        worker.error.disconnect(self._on_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.setParent(None)
        entry = (thread, worker)
        _DETACHED_THREADS.add(entry)
        thread.finished.connect(lambda: _DETACHED_THREADS.discard(entry))

    def _set_running(self, running: bool) -> None:
        self._btn_run.setEnabled(not running)
        self._cmb_model.setEnabled(not running)
        self._txt_site.setReadOnly(running)
        self._txt_context.setReadOnly(running)

    def _set_status(self, text: str, error: bool = False) -> None:
        self._lbl_status.setText(text)
        self._lbl_status.setStyleSheet("color: red;" if error else "")

    def done(self, result: int) -> None:  # noqa: N802 (Qt override)
        # Single choke point for Close/X/accept on a (modal) dialog.
        self._detach_running_thread()
        self._cleanup_thread()
        super().done(result)
