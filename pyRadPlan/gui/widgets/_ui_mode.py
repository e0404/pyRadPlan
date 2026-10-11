"""Presentation modes of the GUI widgets."""

from __future__ import annotations

from enum import Enum


class UIMode(str, Enum):
    """Presentation mode of the GUI widgets.

    Mirrors the values of ``pyRadPlan.settings.gui.ui_mode`` (see
    :class:`pyRadPlan.GuiSettings`). The mode is a *presentation* concept
    only: it controls which controls a widget shows, never which values the
    data models carry or how they are processed.
    """

    #: Full editors — every field is shown.
    DEFAULT = "default"
    #: Simplified editors for non-expert users — only the basic operations
    #: are editable; advanced configuration stays functional, is shown
    #: greyed out and keeps its current values.
    EDUCATIONAL = "educational"
