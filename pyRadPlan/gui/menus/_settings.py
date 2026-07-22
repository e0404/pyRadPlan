"""Settings menu for the pyRadPlan main window.

Exposes the global pyRadPlan configuration (:class:`~pyRadPlan.PyRadPlanSettings`
including its sub-configurations, e.g. the AI agent settings) in an
auto-generated tabbed editor built from the pydantic model schema. Accepted
edits are applied to the runtime settings singleton and mirrored into the
process environment, so both the running session and freshly constructed
settings instances (e.g. by the AI agents) observe them.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMenu,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan._settings import PyRadPlanSettings, get_settings


def _nested_model_fields(model_cls: type[BaseModel]) -> dict[str, type[BaseModel]]:
    """Map field names of nested pydantic models (sub-configurations) to their classes."""
    nested: dict[str, type[BaseModel]] = {}
    for name, info in model_cls.model_fields.items():
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            nested[name] = annotation
    return nested


#: Display names for sub-configuration sections (fallback: `_tab_title`).
_SECTION_TITLES = {"xp": "XP (Backend)", "ai": "AI"}


def _tab_title(field_name: str) -> str:
    if field_name in _SECTION_TITLES:
        return _SECTION_TITLES[field_name]
    if len(field_name) <= 3:
        return field_name.upper()
    return field_name.replace("_", " ").title()


def _env_name(model_cls: type[BaseModel], field_name: str) -> str:
    prefix = model_cls.model_config.get("env_prefix", "")
    return f"{prefix}{field_name}".upper()


class SettingsDialog(QDialog):
    """Editor for the global pyRadPlan settings.

    Without a *section*, a tabbed editor for the full hierarchy is shown: a
    *General* tab holds the top-level fields (omitted when there are none) and
    each nested sub-configuration (e.g. ``xp``, ``ai``) gets its own tab. With
    a *section*, only that sub-configuration's form is shown.

    All forms reflect the runtime settings singleton; accepted edits are
    written back to it and mirrored into the process environment, so consumers
    that re-read settings from the environment per call (such as the AI
    agents) observe them as well.

    Parameters
    ----------
    settings:
        The settings instance to edit; defaults to the global singleton.
    section:
        Name of a nested sub-configuration field (e.g. ``"xp"``) to edit
        exclusively; ``None`` shows the full tabbed editor.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        settings: Optional[PyRadPlanSettings] = None,
        section: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        # Deferred: the widgets package is only needed once the dialog opens.
        from pyRadPlan.gui.widgets import ConfigFormWidget  # noqa: PLC0415

        self._settings = settings if settings is not None else get_settings()
        model_cls = type(self._settings)

        nested = _nested_model_fields(model_cls)
        if section is not None and section not in nested:
            raise ValueError(f"Unknown settings section {section!r}")

        root = QVBoxLayout(self)
        self._tabs: Optional[QTabWidget] = None
        self._general_form: Optional[ConfigFormWidget] = None
        self._sub_forms: dict[str, ConfigFormWidget] = {}

        def _sub_form(name: str) -> ConfigFormWidget:
            form = ConfigFormWidget(
                nested[name], initial=getattr(self._settings, name).model_dump()
            )
            self._sub_forms[name] = form
            return form

        if section is not None:
            self.setWindowTitle(f"{_tab_title(section)} Settings")
            root.addWidget(self._wrap(_sub_form(section)), 1)
        else:
            self.setWindowTitle("Preferences")
            self._tabs = QTabWidget()
            root.addWidget(self._tabs, 1)

            general_initial = {
                name: getattr(self._settings, name)
                for name in model_cls.model_fields
                if name not in nested
            }
            if general_initial:
                self._general_form = ConfigFormWidget(
                    model_cls, initial=general_initial, exclude=set(nested)
                )
                self._tabs.addTab(self._wrap(self._general_form), "General")

            for name in nested:
                self._tabs.addTab(self._wrap(_sub_form(name)), _tab_title(name))

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color: red;")
        root.addWidget(self._lbl_status)
        forms = [f for f in (self._general_form, *self._sub_forms.values()) if f is not None]
        for form in forms:
            form.validation_failed.connect(
                lambda name, msg: self._lbl_status.setText(f"{name}: {msg}")
            )
            form.value_changed.connect(lambda *_: self._lbl_status.setText(""))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.resize(460, 250 if section is not None else 340)

    @staticmethod
    def _wrap(form: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(form)
        return scroll

    def accept(self) -> None:  # noqa: D102 - Qt override
        self.apply()
        super().accept()

    def apply(self) -> None:
        """Write the form state to the settings singleton and the process environment."""
        if self._general_form is not None:
            self._apply_form(type(self._settings), self._settings, self._general_form.values())
        for name, form in self._sub_forms.items():
            self._apply_form(form.model_cls, getattr(self._settings, name), form.values())

    @staticmethod
    def _apply_form(model_cls: type[BaseModel], target: BaseModel, values: dict) -> None:
        for name, value in values.items():
            setattr(target, name, value)
            env_name = _env_name(model_cls, name)
            if value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = str(value)


class SettingsMenu(QMenu):
    """The main window's *Settings* menu.

    Offers a quick link per sub-configuration (e.g. *XP (Backend)*, *AI*)
    opening a single-section dialog, plus *Preferences* opening the full
    tabbed settings editor.

    Parameters
    ----------
    parent:
        Optional Qt parent widget.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("&Settings", parent)
        for name in _nested_model_fields(PyRadPlanSettings):
            action = self.addAction(f"{_tab_title(name)}…")
            action.triggered.connect(lambda checked=False, s=name: self._edit_settings(s))
        self.addSeparator()
        self._act_preferences = self.addAction("Preferences…")
        self._act_preferences.triggered.connect(lambda checked=False: self._edit_settings())

    def _edit_settings(self, section: Optional[str] = None) -> None:
        dialog = SettingsDialog(section=section, parent=self.parentWidget())
        dialog.exec()
