"""Help menu for the pyRadPlan main window.

Links to the online documentation and the source repository, a browser of the
third-party licenses of the installed distributions (which is how a bundled
desktop application ships its license notices) and the *About* dialog carrying
the version, the license note and the non-clinical-use disclaimer (texts from
``pyRadPlan/app/about.json``, shared with the web GUI).
"""

from __future__ import annotations

import platform
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFontDatabase
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import pyRadPlan
from pyRadPlan.app._about import (
    DESCRIPTION,
    DISCLAIMER,
    DISCLAIMER_TITLE,
    IMPRINT_URL,
    LICENSE_NOTE,
    PRIVACY_URL,
)
from pyRadPlan.app._settings import (  # noqa: F401 - re-exported for the GUI
    DistributionInfo,
    installed_distributions,
    license_text,
)
from pyRadPlan.gui.assets import asset_path

DOCUMENTATION_URL = "https://pyradplan.readthedocs.io"
SOURCE_URL = "https://github.com/e0404/pyRadPlan"

NO_LICENSE_TEXT = "No license text found for this distribution."


class ThirdPartyLicensesDialog(QDialog):
    """Browser of the installed distributions and their license texts.

    Parameters
    ----------
    parent:
        Optional Qt parent widget.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Third-party Licenses")
        self._infos = installed_distributions()
        self._texts: dict[str, str] = {}

        root = QVBoxLayout(self)
        root.addWidget(
            QLabel(
                "pyRadPlan builds on the following open-source packages. "
                "Select one to read its license."
            )
        )
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter by name…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        root.addWidget(self._filter)

        self._table = QTableWidget(len(self._infos), 3)
        self._table.setHorizontalHeaderLabels(["Package", "Version", "License"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for row, info in enumerate(self._infos):
            for column, value in enumerate((info.name, info.version, info.license)):
                self._table.setItem(row, column, QTableWidgetItem(value))
        self._table.itemSelectionChanged.connect(self._show_selected)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._text.setPlaceholderText("Select a package to show its license.")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._table)
        splitter.addWidget(self._text)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.resize(900, 560)

    def distribution_names(self) -> list[str]:
        """Names of the listed distributions."""
        return [info.name for info in self._infos]

    def select(self, name: str) -> None:
        """Select the distribution *name* (case-insensitive) and show its license."""
        for row, info in enumerate(self._infos):
            if info.name.lower() == name.lower():
                self._table.selectRow(row)
                return
        raise KeyError(name)

    def current_license_text(self) -> str:
        """Return the license text currently shown."""
        return self._text.toPlainText()

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for row, info in enumerate(self._infos):
            self._table.setRowHidden(row, bool(needle) and needle not in info.name.lower())

    def _show_selected(self) -> None:
        rows = {item.row() for item in self._table.selectedItems()}
        if not rows:
            self._text.clear()
            return
        info = self._infos[min(rows)]
        if info.name not in self._texts:
            self._texts[info.name] = license_text(info.distribution) or NO_LICENSE_TEXT
        self._text.setPlainText(self._texts[info.name])


class AboutDialog(QDialog):
    """Minimal *About* dialog: logo, version, license note and disclaimer.

    Parameters
    ----------
    parent:
        Optional Qt parent widget.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About pyRadPlan")
        root = QVBoxLayout(self)
        root.setSpacing(12)

        logo = QSvgWidget(str(asset_path("logos", "pyradplan_logo_full_landscape.svg")))
        size = logo.renderer().defaultSize()
        height = 72
        width = round(height * size.width() / size.height()) if size.height() else height * 4
        logo.setFixedSize(width, height)
        root.addWidget(logo, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._lbl_title = QLabel(
            f"<b>pyRadPlan</b> {getattr(pyRadPlan, '__version__', '')}"
            f" (Python {platform.python_version()})"
        )
        self._lbl_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        root.addWidget(self._lbl_title)

        description = QLabel(DESCRIPTION)
        description.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        description.setWordWrap(True)
        root.addWidget(description)

        self._lbl_links = QLabel(
            f'<a href="{DOCUMENTATION_URL}">Documentation</a> &nbsp;|&nbsp; '
            f'<a href="{SOURCE_URL}">Source code</a> &nbsp;|&nbsp; '
            f'<a href="{IMPRINT_URL}">Imprint</a> &nbsp;|&nbsp; '
            f'<a href="{PRIVACY_URL}">Privacy policy</a>'
        )
        self._lbl_links.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._lbl_links.setOpenExternalLinks(True)
        root.addWidget(self._lbl_links)

        self._lbl_license = QLabel(LICENSE_NOTE)
        self._lbl_license.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._lbl_license.setWordWrap(True)
        root.addWidget(self._lbl_license)

        self._lbl_disclaimer = QLabel(f"<b>{DISCLAIMER_TITLE}</b><br>{DISCLAIMER}")
        self._lbl_disclaimer.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._lbl_disclaimer.setWordWrap(True)
        self._lbl_disclaimer.setStyleSheet(
            "color: #c0392b; border: 1px solid #c0392b; border-radius: 4px; padding: 8px;"
        )
        root.addWidget(self._lbl_disclaimer)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._btn_licenses = QPushButton("Third-party Licenses…")
        buttons.addButton(self._btn_licenses, QDialogButtonBox.ButtonRole.ActionRole)
        self._btn_licenses.clicked.connect(self._show_licenses)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.setFixedWidth(460)

    def text(self) -> str:
        """Plain text of the dialog's labels, for inspection."""
        parts = [
            self._lbl_title.text(),
            self._lbl_links.text(),
            self._lbl_license.text(),
            self._lbl_disclaimer.text(),
        ]
        return "\n".join(parts)

    def _show_licenses(self) -> None:
        ThirdPartyLicensesDialog(self).exec()


class HelpMenu(QMenu):
    """The main window's *Help* menu.

    Parameters
    ----------
    parent:
        Optional Qt parent widget.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("&Help", parent)
        self._act_docs = self.addAction("Documentation")
        self._act_docs.triggered.connect(lambda checked=False: self._open_url(DOCUMENTATION_URL))
        self._act_source = self.addAction("Source Code")
        self._act_source.triggered.connect(lambda checked=False: self._open_url(SOURCE_URL))
        self.addSeparator()
        self._act_licenses = self.addAction("Third-party Licenses…")
        self._act_licenses.triggered.connect(lambda checked=False: self._show_licenses())
        self._act_about = self.addAction("About pyRadPlan…")
        self._act_about.triggered.connect(lambda checked=False: self._show_about())

    @staticmethod
    def _open_url(url: str) -> None:
        QDesktopServices.openUrl(QUrl(url))

    def _show_licenses(self) -> None:
        ThirdPartyLicensesDialog(self.parentWidget()).exec()

    def _show_about(self) -> None:
        AboutDialog(self.parentWidget()).exec()
