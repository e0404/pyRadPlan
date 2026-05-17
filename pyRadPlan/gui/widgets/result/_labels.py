"""Shared widget helpers for the result viewer (truncated labels)."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QWidget


MAX_VOI_LABEL_CHARS = 12


def truncate_label(text: str, max_chars: int = MAX_VOI_LABEL_CHARS) -> str:
    """Truncate *text* to *max_chars* characters, appending an ellipsis if shortened."""
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)] + "\u2026"


class TruncatedCheckBox(QCheckBox):
    """A QCheckBox whose label is truncated to a fixed character length.

    The full text is preserved in the tooltip.
    """

    def __init__(
        self, text: str, parent: QWidget | None = None, max_chars: int = MAX_VOI_LABEL_CHARS
    ) -> None:
        super().__init__(truncate_label(text, max_chars), parent)
        self._full_text = text
        self._max_chars = max_chars
        self.setToolTip(text)
