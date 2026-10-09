"""QI table widget."""

from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING

import numpy as np

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QGuiApplication, QPixmap, QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.analysis import format_metric_label

if TYPE_CHECKING:
    from pyRadPlan.analysis import QI, QICollection

logger = logging.getLogger(__name__)

_STRUCTURE_COL = "Structure"


def _swatch_icon(rgb: tuple[int, int, int]) -> QIcon:
    """Return a small solid-color icon used to mark a structure's row."""
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)


class QITableWidget(QWidget):
    """Widget displaying the quality indicators of one quantity as a table.

    Data is supplied via :meth:`set_qis`, one row per structure. Columns are
    derived from the metric ids actually present in the collection, so they
    follow whatever reference volumes and doses were used to compute them.

    Parameters
    ----------
    parent : QWidget or None, optional
        Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._headers: list[str] = []
        self._rows: list[list[str]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        if hasattr(header, "setSectionResizeMode"):
            # Metric columns are sized to their content rather than stretched:
            # the metric set grows with the reference doses, and stretching
            # would elide the column labels. Spare width goes to the structure
            # column instead (see _set_content), which holds the longest text.
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setStretchLastSection(False)
            header.setMinimumSectionSize(80)

        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addStretch(1)

        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setToolTip("Copy the table to the clipboard (tab-separated)")
        self.copy_btn.clicked.connect(self._on_copy_clicked)
        btn_row.addWidget(self.copy_btn)

        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.setToolTip("Save the table as a CSV file")
        self.export_btn.clicked.connect(self._on_export_clicked)
        btn_row.addWidget(self.export_btn)

        layout.addLayout(btn_row)

        self._sync_buttons()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_qis(
        self,
        qis: QICollection | None,
        structures: list[str] | None = None,
        voi_colors: dict[str, tuple[int, int, int]] | None = None,
        metrics: list[str] | None = None,
    ) -> None:
        """Populate the table from a QI collection.

        Parameters
        ----------
        qis : QICollection or None
            Quality indicators of one quantity, rendered as one row per
            structure. ``None`` or an empty collection clears the table.
        structures : list of str or None, optional
            Structure names to show, in display order. ``None`` shows every
            structure of the collection. Names missing from it are skipped.
        voi_colors : dict of str to tuple of int or None, optional
            RGB tuples (0-255) per structure name, used for the row swatches.
        metrics : list of str or None, optional
            Metric ids to show as columns, in the collection's own order.
            ``None`` shows every metric present.
        """
        if qis is None or len(qis) == 0:
            self._set_content([], [])
            return

        if structures is None:
            structures = list(qis.structures.keys())
        structures = [name for name in structures if name in qis]

        metric_keys = qis.metric_ids(structures)
        if metrics is not None:
            selected = set(metrics)
            metric_keys = [m for m in metric_keys if m in selected]

        if not structures or not metric_keys:
            self._set_content([], [])
            return

        headers = [_STRUCTURE_COL] + [
            format_metric_label(metric, self._first_qi(qis, structures, metric))
            for metric in metric_keys
        ]

        rows: list[list[str]] = []
        colors: list[tuple[int, int, int] | None] = []
        for name in structures:
            structure_qis = qis[name]
            rows.append(
                [name, *(self._format_value(structure_qis.metrics.get(m)) for m in metric_keys)]
            )
            colors.append((voi_colors or {}).get(name))

        self._set_content(headers, rows, colors)

    def to_rows(self) -> list[list[str]]:
        """Return the table as rows of strings, header first.

        Backs the clipboard and CSV export, and gives tests a way to read the
        rendered table without walking Qt items.
        """
        if not self._headers:
            return []
        return [list(self._headers), *(list(r) for r in self._rows)]

    @Slot()
    def _on_copy_clicked(self) -> None:
        """Button slot. Separate from the API method so Qt's ``checked`` bool is dropped."""
        self.copy_to_clipboard()

    @Slot()
    def _on_export_clicked(self) -> None:
        """Button slot.

        ``clicked`` emits a ``checked`` bool, which PySide6 would bind to
        :meth:`export_csv`'s optional *path* argument - ``open(False)`` then
        writes to file descriptor 0 instead of opening a save dialog.
        """
        self.export_csv()

    def copy_to_clipboard(self) -> None:
        """Copy the table to the clipboard as tab-separated text."""
        rows = self.to_rows()
        if not rows:
            return
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("\n".join("\t".join(cell for cell in row) for row in rows))

    def export_csv(self, path: str | None = None) -> str | None:
        """Write the table to a CSV file.

        Parameters
        ----------
        path:
            Target file. When ``None`` (the default, as used by the button) a
            save dialog is opened.

        Returns
        -------
        str or None
            The path written, or ``None`` if there was nothing to write or the
            dialog was cancelled.
        """
        rows = self.to_rows()
        if not rows:
            return None

        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export quality indicators", "quality_indicators.csv", "CSV files (*.csv)"
            )
            if not path:
                return None

        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
        except OSError:
            logger.exception("Could not write QI table to %s", path)
            return None
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_qi(qis: QICollection, structures: list[str], metric: str) -> QI | None:
        """Return any QI carrying *metric*, for its unit in the column label."""
        for name in structures:
            if name in qis and metric in qis[name]:
                return qis[name][metric]
        return None

    @staticmethod
    def _format_value(qi: QI | None) -> str:
        """Format a QI value, matching ``QICollection``'s table rendering."""
        if qi is None or np.isnan(qi.value):
            return "-"
        return f"{qi.value:.2f}"

    def _set_content(
        self,
        headers: list[str],
        rows: list[list[str]],
        colors: list[tuple[int, int, int] | None] | None = None,
    ) -> None:
        """Replace the whole table content."""
        self._headers = headers
        self._rows = rows

        self.table.clearContents()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))

        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                item = QTableWidgetItem(text)
                if c == 0:
                    rgb = (colors or [None] * len(rows))[r]
                    if rgb is not None:
                        item.setIcon(_swatch_icon(rgb))
                    item.setToolTip(text)
                else:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(r, c, item)

        # Spare width goes to the structure column; the metric columns keep the
        # width their labels need, and a scrollbar appears once they exceed it.
        header = self.table.horizontalHeader()
        if headers and hasattr(header, "setSectionResizeMode"):
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        self._sync_buttons()

    def _sync_buttons(self) -> None:
        """Enable the export actions only when there is something to export."""
        has_content = bool(self._rows)
        self.copy_btn.setEnabled(has_content)
        self.export_btn.setEnabled(has_content)
