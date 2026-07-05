"""Live optimization status window (metric plots + pause/stop controls)."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.core import ComputeControl

#: Default metrics plotted as ``(key, axis title, log_y)``.  The widget is agnostic to
#: which keys a report actually carries, so extra metrics (e.g. constraint violation,
#: step size) can be enabled by passing a longer spec list to :meth:`configure_metrics`.
#: The trailing ``log_y`` flag is optional (defaults to ``False``).
DEFAULT_METRICS: tuple[tuple, ...] = (("objective", "Objective value", True),)


class OptimizationStatusWidget(QWidget):
    """Top-level window tracking a running optimization.

    Shows a grid of live metric plots (objective value by default) fed by
    :class:`~pyRadPlan.core.StatusReport` data, a one-line summary, and Pause/Resume +
    Stop controls bound to a :class:`~pyRadPlan.core.ComputeControl`.

    The widget owns no knowledge of *how* the metrics are produced; it simply plots
    whatever configured keys appear in the report dicts it is handed, using an
    ``iteration`` value as the x coordinate when present (else a running index).
    """

    #: Emitted when the user toggles pause (True = paused) / requests a stop.
    pause_toggled = Signal(bool)
    stop_requested = Signal()

    def __init__(
        self,
        metrics: Optional[Sequence[tuple[str, str]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Optimization Status")
        self.setWindowFlag(Qt.Window, True)

        self._control: Optional[ComputeControl] = None
        # key -> {"curve", "xs", "ys"}
        self._metrics: dict[str, dict] = {}

        # Fast solvers report every iteration (hundreds per second); redrawing
        # the full series each time is O(n²) overall and floods the event loop.
        # Data points are appended immediately, redraws are rate-limited.
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(100)
        self._redraw_timer.timeout.connect(self._redraw)

        self._setup_ui()
        self.configure_metrics(metrics if metrics is not None else DEFAULT_METRICS)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)

        self._lbl_summary = QLabel("Waiting for optimization…")
        root.addWidget(self._lbl_summary)

        self._plot_layout = pg.GraphicsLayoutWidget()
        root.addWidget(self._plot_layout, stretch=1)

        controls = QHBoxLayout()
        self._btn_pause = QPushButton("Pause")
        self._btn_pause.setCheckable(True)
        self._btn_pause.toggled.connect(self._on_pause_toggled)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.clicked.connect(self._on_stop_clicked)
        controls.addWidget(self._btn_pause)
        controls.addWidget(self._btn_stop)
        controls.addStretch()
        root.addLayout(controls)

    def configure_metrics(self, specs: Sequence[tuple]) -> None:
        """(Re)build the plot grid from ``(metric_key, axis_title[, log_y])`` *specs*.

        Each metric is drawn as discrete marker points (thick crosses) rather than a
        connecting line; ``log_y`` (optional, default ``False``) puts that plot's y
        axis on a logarithmic scale.
        """
        self._redraw_timer.stop()
        self._plot_layout.clear()
        self._metrics.clear()

        ncols = max(1, math.ceil(math.sqrt(len(specs))))
        for i, spec in enumerate(specs):
            key, title = spec[0], spec[1]
            log_y = bool(spec[2]) if len(spec) > 2 else False
            row, col = divmod(i, ncols)
            plot = self._plot_layout.addPlot(row=row, col=col, title=title)
            plot.setLabel("bottom", "Iteration")
            plot.showGrid(x=True, y=True, alpha=0.3)
            if log_y:
                plot.setLogMode(x=False, y=True)
            # Discrete thick crosses, no connecting line.
            curve = plot.plot(
                [],
                [],
                pen=None,
                symbol="+",
                symbolSize=12,
                symbolPen=pg.mkPen(width=2.5),
            )
            self._metrics[key] = {"curve": curve, "xs": [], "ys": []}

    # ------------------------------------------------------------------
    # Control binding
    # ------------------------------------------------------------------

    def bind_control(self, control: ComputeControl) -> None:
        """Bind the Pause/Stop buttons to *control* and re-enable them."""
        self._control = control
        self._btn_pause.setEnabled(True)
        self._btn_pause.setChecked(False)
        self._btn_pause.setText("Pause")
        self._btn_stop.setEnabled(True)

    def finalize(self) -> None:
        """Disable the controls once the optimization has ended (keep the curves)."""
        self._control = None
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._redraw_timer.stop()
        self._redraw()

    def _on_pause_toggled(self, checked: bool) -> None:
        self._btn_pause.setText("Resume" if checked else "Pause")
        if self._control is not None:
            if checked:
                self._control.pause()
            else:
                self._control.resume()
        self.pause_toggled.emit(checked)

    def _on_stop_clicked(self) -> None:
        if self._control is not None:
            self._control.request_stop()
        self._btn_stop.setEnabled(False)
        self.stop_requested.emit()

    # ------------------------------------------------------------------
    # Data updates
    # ------------------------------------------------------------------

    def update_from_report(self, data: dict) -> str:
        """Append one report's metrics to the plots and return a summary string.

        The summary (also shown in the header) is suitable for the host's status line,
        e.g. ``iter 12 · f=1.23e+01 · Δf=-3.4e-03``.  ``Δf`` is derived here from the
        local objective series, so callers need not track it.
        """
        x = data.get("iteration")
        appended = False
        for key, series in self._metrics.items():
            if key not in data:
                continue
            xs, ys = series["xs"], series["ys"]
            xs.append(x if x is not None else len(xs))
            ys.append(float(data[key]))
            appended = True
        if appended and not self._redraw_timer.isActive():
            self._redraw_timer.start()

        summary = self._format_summary(data)
        self._lbl_summary.setText(summary)
        return summary

    def _redraw(self) -> None:
        for series in self._metrics.values():
            series["curve"].setData(series["xs"], series["ys"])

    def _format_summary(self, data: dict) -> str:
        parts: list[str] = []
        if "iteration" in data:
            parts.append(f"iter {int(data['iteration'])}")
        if "objective" in data:
            parts.append(f"f={float(data['objective']):.3e}")
        rel = self._relative_objective_change()
        if rel is not None:
            parts.append(f"Δf={rel:+.2e}")
        if not parts and data.get("message"):
            parts.append(str(data["message"]))
        return " · ".join(parts) if parts else "Optimizing…"

    def _relative_objective_change(self) -> Optional[float]:
        ys = self._metrics.get("objective", {}).get("ys", [])
        if len(ys) < 2 or ys[-2] == 0:
            return None
        return (ys[-1] - ys[-2]) / abs(ys[-2])
