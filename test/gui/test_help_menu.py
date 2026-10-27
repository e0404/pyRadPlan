"""Tests for the Help menu, the About dialog and the third-party licenses dialog."""

from importlib import metadata

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QDesktopServices

from pyRadPlan.gui.menus import HelpMenu
from pyRadPlan.gui.menus._help import (
    DOCUMENTATION_URL,
    NO_LICENSE_TEXT,
    SOURCE_URL,
    AboutDialog,
    ThirdPartyLicensesDialog,
    license_text,
)
from pyRadPlan.gui.windows._main_win import MainWindow
from pyRadPlan.gui.workspace import WorkspaceManager


def test_help_menu_is_last_in_menu_bar(qapp):
    win = MainWindow(WorkspaceManager())
    titles = [a.text() for a in win.menuBar().actions()]
    assert titles.index("&Help") == len(titles) - 1
    assert titles.index("&Help") > titles.index("&Settings")
    labels = [a.text() for a in win._help_menu.actions() if a.text()]
    assert labels == ["Documentation", "Source Code", "Third-party Licenses…", "About pyRadPlan…"]


def test_help_menu_actions_open_urls(qapp, monkeypatch):
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    menu = HelpMenu()
    menu._act_docs.trigger()
    menu._act_source.trigger()
    assert opened == [DOCUMENTATION_URL, SOURCE_URL]


def test_about_dialog_contains_version_and_disclaimer(qapp):
    import pyRadPlan

    dialog = AboutDialog()
    text = dialog.text()
    assert pyRadPlan.__version__ in text
    assert "BSD" in text
    assert "research and education" in text
    assert "not for clinical use" in text.lower()
    assert "must not be used for clinical treatment planning" in text


def test_licenses_dialog_lists_numpy_and_shows_its_license(qapp):
    dialog = ThirdPartyLicensesDialog()
    names = [n.lower() for n in dialog.distribution_names()]
    assert "numpy" in names
    # the gui extra installs pyside6-essentials, not the pyside6 meta-distribution
    assert any(n.replace("_", "-").startswith("pyside6") for n in names)

    dialog.select("numpy")
    text = dialog.current_license_text()
    assert "copyright" in text.lower()
    assert text != NO_LICENSE_TEXT


def test_licenses_dialog_filter_hides_rows(qapp):
    dialog = ThirdPartyLicensesDialog()
    dialog._filter.setText("numpy")
    visible = [
        dialog._table.item(row, 0).text()
        for row in range(dialog._table.rowCount())
        if not dialog._table.isRowHidden(row)
    ]
    assert visible and all("numpy" in name.lower() for name in visible)


def test_license_text_reads_dist_info_files(tmp_path):
    dist_info = tmp_path / "pkg-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: pkg\nVersion: 1.0\n", encoding="utf-8")
    (dist_info / "LICENSE").write_text("MIT License text", encoding="utf-8")
    (dist_info / "RECORD").write_text(
        "pkg-1.0.dist-info/METADATA,,\npkg-1.0.dist-info/LICENSE,,\n", encoding="utf-8"
    )
    dist = metadata.PathDistribution(dist_info)
    assert license_text(dist) == "MIT License text"


def test_license_text_without_record_tries_common_names(tmp_path):
    dist_info = tmp_path / "pkg-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: pkg\nVersion: 1.0\n", encoding="utf-8")
    (dist_info / "COPYING").write_text("GPL text", encoding="utf-8")
    assert license_text(metadata.PathDistribution(dist_info)) == "GPL text"


def test_license_text_falls_back_to_metadata_field(tmp_path):
    dist_info = tmp_path / "pkg-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Name: pkg\nVersion: 1.0\nLicense: Full license\n        second line\n", encoding="utf-8"
    )
    assert "second line" in license_text(metadata.PathDistribution(dist_info))


def test_license_text_missing_returns_none(tmp_path):
    dist_info = tmp_path / "pkg-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Name: pkg\nVersion: 1.0\nLicense: MIT\n", encoding="utf-8"
    )
    assert license_text(metadata.PathDistribution(dist_info)) is None
