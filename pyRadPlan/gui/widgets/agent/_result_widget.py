"""The *Agentic Result* tab: the agent's final report and every optimization it ran."""

from __future__ import annotations

import html
from typing import Any, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

_COLUMNS = ("", "Opt.", "Hard missed", "Soft missed", "Score", "Plan")
_LOAD, _INDEX, _HARD, _SOFT, _SCORE, _BRIEF = range(len(_COLUMNS))


def missed_goals(plan: Any) -> tuple[int, int]:
    """Return the number of failed hard and soft goals of *plan* (evaluated goals only)."""
    goals = plan.evaluation["goals"]
    hard = sum(not g["passed"] for g in goals if g["hard"])
    return hard, sum(not g["passed"] for g in goals if not g["hard"])


def plan_brief(plan: Any) -> str:
    """Summarize *plan* in a few characters: beams, objectives and helpers."""
    num_beams = len(plan.beams["gantry_angles"])
    objectives = plan.objectives
    num_obj = sum(len(objs) for objs in objectives.values())
    helpers = sum(1 for name, objs in objectives.items() if objs and name.startswith("opt_"))
    brief = f"{num_beams} beams · {num_obj} obj"
    return brief + (f" · {helpers} helper{'s' if helpers > 1 else ''}" if helpers else "")


def _plan_details(plan: Any) -> str:
    beams = plan.beams
    angles = ", ".join(
        f"{g:g}" if not c else f"{g:g}/{c:g}"
        for g, c in zip(beams["gantry_angles"], beams["couch_angles"])
    )
    kind = "Blend" if plan.blend else "Optimization"
    lines = [f"{kind} {plan.index}", f"Beams (gantry[/couch]): {angles or '–'}"]
    for voi, objs in plan.objectives.items():
        for o in objs:
            if "parameters" not in o:  # not an Objective (see _objective_summary)
                lines.append(f"{voi}: {o.get('definition', o)}")
                continue
            params = ", ".join(f"{k}={_num(v)}" for k, v in o["parameters"].items())
            lines.append(f"{voi}: {o['name']} (penalty {o['penalty']:g}; {params})")
    failed = [g["goal"] for g in plan.evaluation["goals"] if not g["passed"]]
    if failed:
        lines.append("Failed: " + "; ".join(failed))
    return "\n".join(lines)


class AgentResultWidget(QWidget):
    """The final report of a planning agent run above a table of all its optimizations.

    Each row's *Load plan* button emits :attr:`load_requested` with the optimization
    index; the buttons are enabled with :meth:`set_loading_enabled` (not during a run).
    """

    #: Emitted with the index of the optimization to open as a plan tab.
    load_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Vertical)

        self.report = QTextBrowser()
        self.report.setOpenLinks(False)
        split.addWidget(self.report)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        # Columns fit their contents; a narrow panel scrolls instead of eliding the summary
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setWordWrap(False)
        split.addWidget(self.table)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        lay.addWidget(split)

        self._loading_enabled = False
        self._header = ""
        self.clear()

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove the report and all optimizations."""
        self.table.setRowCount(0)
        self._header = ""
        self.report.setHtml("<i>No agent run yet.</i>")

    def start_run(self, header: Sequence[str]) -> None:
        """Clear the tab for a new run described by the *header* lines."""
        self.clear()
        self._header = "".join(f"{html.escape(line)}<br>" for line in header)
        self.report.setHtml(self._header + "<p><i>The agent is running…</i></p>")

    def add_plan(self, plan: Any) -> None:
        """Append a row for *plan* (one of ``PlanningState.plans``)."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        btn = QPushButton("Load plan")
        btn.setToolTip("Open this optimization as a plan tab")
        btn.setEnabled(self._loading_enabled)
        btn.clicked.connect(lambda _checked=False, i=plan.index: self.load_requested.emit(i))
        self.table.setCellWidget(row, _LOAD, btn)
        hard, soft = missed_goals(plan)
        values = {
            _INDEX: str(plan.index),
            _HARD: str(hard),
            _SOFT: str(soft),
            _SCORE: f"{plan.evaluation['score']:g}",
            _BRIEF: plan_brief(plan),
        }
        details = _plan_details(plan)
        for col, text in values.items():
            item = QTableWidgetItem(text)
            item.setToolTip(details)
            if col != _BRIEF:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col, item)

    def mark_picks(self, picks: Sequence[Any]) -> None:
        """Label the selected plans (Plan 1, 2, ...) in the table."""
        labels = {p.index: f"Plan {n}" for n, p in enumerate(picks, start=1)}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, _INDEX)
            index = int(item.text().split()[0])
            pick = labels.get(index)
            item.setText(f"{index} ({pick})" if pick else str(index))
            font = QFont(item.font())
            font.setBold(pick is not None)
            for col in range(_INDEX, len(_COLUMNS)):
                self.table.item(row, col).setFont(font)

    def set_report(
        self,
        report: Any = None,
        status: str = "",
        unresolved_goals: Sequence[str] = (),
        usage: Optional[str] = None,
        summary: str = "",
    ) -> None:
        """Show the agent's final *report* (a ``PlanningReport``, None if it stopped)."""
        parts = [self._header]
        if status:
            parts.append(f"<p><b>{html.escape(status)}</b></p>")
        if report is not None:
            met = "All hard goals met" if report.all_hard_goals_met else "Hard goals not met"
            parts.append(f"<p><b>{met}.</b> {html.escape(report.summary)}</p>")
            parts.append(_section("Trade-offs", report.trade_offs))
            parts.append(_section("Suggestions for the planner", report.suggestions_for_planner))
        parts.append(_section("Goals not evaluated (structure not found)", unresolved_goals))
        if summary:
            parts.append(f"<p>{html.escape(summary)}</p>")
        if usage:
            parts.append(f"<p><small>{html.escape(usage)}</small></p>")
        self.report.setHtml("".join(parts))

    def set_loading_enabled(self, enabled: bool) -> None:
        """Enable or disable the *Load plan* buttons."""
        self._loading_enabled = enabled
        for row in range(self.table.rowCount()):
            self.table.cellWidget(row, _LOAD).setEnabled(enabled)


def _num(value: Any) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _section(title: str, items: Sequence[str]) -> str:
    if not items:
        return ""
    lis = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f"<p><b>{html.escape(title)}</b></p><ul>{lis}</ul>"
