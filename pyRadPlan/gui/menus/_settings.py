"""Settings menu for the pyRadPlan main window.

Exposes the configuration that is normally instantiated from the environment /
``.env`` (currently the AI agent settings) in an auto-generated editor built from
the pydantic model schema.  Edits are applied to the process environment so they
take effect immediately for the running session.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtWidgets import QDialog, QMenu, QWidget

from pyRadPlan.gui.widgets.ai import AI_MISSING_TIP, ai_available


class SettingsMenu(QMenu):
    """The main window's *Settings* menu.

    Parameters
    ----------
    parent:
        Optional Qt parent widget.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("&Settings", parent)
        self._ai_available = ai_available()

        self._act_ai = self.addAction("AI Settings…")
        self._act_ai.triggered.connect(self._edit_ai_settings)
        if not self._ai_available:
            self._act_ai.setEnabled(False)
            self._act_ai.setToolTip(AI_MISSING_TIP)

    def _edit_ai_settings(self) -> None:
        # Deferred: keeps the optional ai_agents stack out of menu construction.
        from pyRadPlan.ai_agents import AiSettings, load_ai_env  # noqa: PLC0415
        from pyRadPlan.gui.widgets import ConfigFormDialog  # noqa: PLC0415

        # Reflect the current effective configuration (env + .env) in the form.
        load_ai_env()
        settings = AiSettings()
        dialog = ConfigFormDialog(
            AiSettings,
            initial=settings.model_dump(),
            title="AI Settings",
            parent=self.parentWidget(),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        prefix = AiSettings.model_config.get("env_prefix", "")
        for name, value in dialog.values().items():
            os.environ[f"{prefix}{name}".upper()] = str(value)
