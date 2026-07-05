"""In-GUI log console fed by Python's :mod:`logging` system.

Shown in the main window's bottom "Output" dock so warnings and progress
messages are visible even when the application was launched without a
terminal.  A handler on the root logger forwards formatted records through a
Qt signal, which marshals them to the GUI thread regardless of where they
were emitted (e.g. the workflow worker thread).
"""

from __future__ import annotations

import html
import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

#: (label, level) pairs offered by the minimum-level filter combo.
_LEVELS = (
    ("Debug", logging.DEBUG),
    ("Info", logging.INFO),
    ("Warning", logging.WARNING),
    ("Error", logging.ERROR),
)

_DEFAULT_LEVEL = logging.INFO

#: Maximum number of retained paragraphs before the oldest are dropped.
_MAX_BLOCKS = 2000


class _QtLogHandler(logging.Handler):
    """Forward formatted log records through a Qt signal.

    ``logging`` may call :meth:`emit` from any thread; the signal/slot
    connection delivers the record to the GUI thread, so the handler itself
    must not touch any widget.
    """

    class _Bridge(QObject):
        record_emitted = Signal(str, int)  # formatted message, levelno

    def __init__(self) -> None:
        super().__init__()
        self.bridge = self._Bridge()
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.bridge.record_emitted.emit(self.format(record), record.levelno)
        except Exception:  # noqa: BLE001 - logging must never raise into callers
            self.handleError(record)


class LogConsoleWidget(QWidget):
    """Read-only log view with a minimum-level filter and a Clear button.

    On construction the widget attaches a handler to the *root* logger, so it
    also catches direct ``logging.warning(...)`` calls and third-party
    warnings/errors.  Because the ``pyRadPlan`` package loggers otherwise
    inherit the root default of ``WARNING``, the level selected in the combo
    is additionally applied to the ``pyRadPlan`` logger — third-party loggers
    keep their own (usually ``WARNING``) threshold, so lowering the filter to
    Debug does not flood the view with library noise.

    Call :meth:`detach` before the application quits to remove the handler.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Level:"))
        self._level_combo = QComboBox()
        for label, level in _LEVELS:
            self._level_combo.addItem(label, level)
        self._level_combo.setCurrentIndex(
            next(i for i, (_, lvl) in enumerate(_LEVELS) if lvl == _DEFAULT_LEVEL)
        )
        self._level_combo.setToolTip(
            "Minimum level shown; also sets the pyRadPlan logger verbosity"
        )
        self._level_combo.currentIndexChanged.connect(self._on_level_changed)
        row.addWidget(self._level_combo)
        row.addStretch(1)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear)
        row.addWidget(clear_btn)
        root.addLayout(row)

        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(_MAX_BLOCKS)
        self._view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        root.addWidget(self._view, 1)

        self._handler = _QtLogHandler()
        self._handler.bridge.record_emitted.connect(self._append_record)
        self._apply_level(_DEFAULT_LEVEL)
        logging.getLogger().addHandler(self._handler)
        # Safety net for hosts that never call detach() (e.g. test windows):
        # capture the handler, not self, so the dead widget isn't kept alive.
        self.destroyed.connect(lambda *_, h=self._handler: logging.getLogger().removeHandler(h))

    def detach(self) -> None:
        """Remove the handler from the root logger (call once on shutdown)."""
        logging.getLogger().removeHandler(self._handler)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_level_changed(self, index: int) -> None:
        self._apply_level(self._level_combo.itemData(index))

    def _apply_level(self, level: int) -> None:
        self._handler.setLevel(level)
        # Package loggers use NOTSET and would inherit the root default of
        # WARNING; without this, Info/Debug records never reach the handler.
        logging.getLogger("pyRadPlan").setLevel(level)

    def _on_clear(self) -> None:
        self._view.clear()

    @Slot(str, int)
    def _append_record(self, message: str, levelno: int) -> None:
        if levelno >= logging.ERROR:
            color = "#e74c3c"
        elif levelno >= logging.WARNING:
            color = "#e67e22"
        elif levelno < logging.INFO:
            color = "#888888"
        else:
            color = None

        text = html.escape(message).replace("\n", "<br>")
        if color is not None:
            text = f'<span style="color:{color};">{text}</span>'
        self._view.appendHtml(text)
