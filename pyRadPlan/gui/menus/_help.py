"""Help menu for the pyRadPlan main window.

Links to the online documentation and the source repository, a browser of the
third-party licenses of the installed distributions (which is how a bundled
desktop application ships its license notices) and the *About* dialog carrying
the version, the license note and the non-clinical-use disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
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
from pyRadPlan.gui.assets import asset_path

DOCUMENTATION_URL = "https://pyradplan.readthedocs.io"
SOURCE_URL = "https://github.com/e0404/pyRadPlan"

#: Shown prominently in the About dialog.
DISCLAIMER = (
    "pyRadPlan is intended for research and education. It is not a medical device "
    "and must not be used for clinical treatment planning."
)
LICENSE_NOTE = "Released under the BSD 3-Clause License. Copyright (c) 2024, DKFZ."

#: Case-insensitive prefixes of files in a ``.dist-info`` directory that hold license texts.
_LICENSE_FILE_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "NOTICE")
#: Fallback candidates when a distribution ships no ``RECORD`` to enumerate its files.
_LICENSE_FILE_CANDIDATES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "LICENSE.rst",
    "LICENCE",
    "COPYING",
    "COPYING.txt",
)
NO_LICENSE_TEXT = "No license text found for this distribution."


@dataclass(frozen=True)
class DistributionInfo:
    """Name, version and license identifier of an installed distribution."""

    name: str
    version: str
    license: str
    distribution: metadata.Distribution


def _license_identifier(dist: metadata.Distribution) -> str:
    """Return a short license name from the metadata, or ``"unknown"``."""
    meta = dist.metadata
    expression = meta.get("License-Expression")
    if expression:
        return expression.strip()
    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            return classifier.split("::")[-1].strip()
    # Older packages put either a short name or the complete text into License.
    text = (meta.get("License") or "").strip()
    if text and "\n" not in text and len(text) <= 60:
        return text
    if text:
        return "see license text"
    return "unknown"


def _is_license_file(parts: tuple[str, ...]) -> bool:
    """Whether a path relative to the ``.dist-info`` directory is a license notice."""
    if parts[0] == "licenses":
        return len(parts) > 1
    return len(parts) == 1 and parts[0].upper().startswith(_LICENSE_FILE_PREFIXES)


def license_text(dist: metadata.Distribution) -> Optional[str]:
    """Return the license text(s) shipped with *dist*, or ``None`` if there is none.

    The ``LICENSE*`` / ``COPYING*`` / ``NOTICE*`` files and the ``licenses/``
    directory of the distribution's ``.dist-info`` are read through
    :meth:`importlib.metadata.Distribution.read_text`, which works from a
    read-only install and also covers wheels without a ``License`` metadata
    field. Several files are concatenated under their file names. As a last
    resort the ``License`` metadata field is returned when it holds a full text.
    """
    names: list[str] = []
    for path in dist.files or []:
        parts = path.parts
        if len(parts) >= 2 and parts[0].endswith((".dist-info", ".egg-info")):
            if _is_license_file(parts[1:]):
                names.append("/".join(parts[1:]))
    if not names:
        names = list(_LICENSE_FILE_CANDIDATES)

    texts: list[tuple[str, str]] = []
    for name in sorted(dict.fromkeys(names)):
        try:
            text = dist.read_text(name)
        except (OSError, UnicodeDecodeError, ValueError):
            text = None
        if text and text.strip():
            texts.append((name, text.strip()))

    if len(texts) == 1:
        return texts[0][1]
    if texts:
        return "\n\n\n".join(f"{name}\n{'=' * len(name)}\n\n{text}" for name, text in texts)

    field = (dist.metadata.get("License") or "").strip()
    if "\n" in field:
        return field
    return None


def installed_distributions() -> list[DistributionInfo]:
    """Return the installed distributions sorted by name, one entry per name."""
    found: dict[str, DistributionInfo] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name or name.lower() in found:
            continue
        found[name.lower()] = DistributionInfo(
            name=name,
            version=dist.version or "",
            license=_license_identifier(dist),
            distribution=dist,
        )
    return [found[key] for key in sorted(found)]


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

        self._lbl_title = QLabel(f"<b>pyRadPlan</b> {getattr(pyRadPlan, '__version__', '')}")
        self._lbl_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        root.addWidget(self._lbl_title)

        description = QLabel(
            "Python radiotherapy treatment planning toolkit, interoperable with matRad."
        )
        description.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        description.setWordWrap(True)
        root.addWidget(description)

        links = QLabel(
            f'<a href="{DOCUMENTATION_URL}">Documentation</a> &nbsp;|&nbsp; '
            f'<a href="{SOURCE_URL}">Source code</a>'
        )
        links.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        links.setOpenExternalLinks(True)
        root.addWidget(links)

        self._lbl_license = QLabel(LICENSE_NOTE)
        self._lbl_license.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._lbl_license.setWordWrap(True)
        root.addWidget(self._lbl_license)

        self._lbl_disclaimer = QLabel(f"<b>NOT FOR CLINICAL USE</b><br>{DISCLAIMER}")
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
        parts = [self._lbl_title.text(), self._lbl_license.text(), self._lbl_disclaimer.text()]
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
