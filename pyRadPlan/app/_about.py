"""Texts of the *About* dialog shared by the Qt and the web GUI.

The single source is ``about.json`` next to this module: the Qt dialog reads it through the
constants below, the web frontend (``web/vite.config.ts``) embeds it at build time, so the
research-use disclaimer reads the same in both and is available before the backend starts.
"""

from __future__ import annotations

import json
from importlib import resources

_TEXTS: dict[str, str] = json.loads(
    resources.files(__package__).joinpath("about.json").read_text(encoding="utf-8")
)

DESCRIPTION: str = _TEXTS["description"]
LICENSE_NOTE: str = _TEXTS["license_note"]
DISCLAIMER_TITLE: str = _TEXTS["disclaimer_title"]
DISCLAIMER: str = _TEXTS["disclaimer"]
IMPRINT_URL: str = _TEXTS["imprint_url"]
PRIVACY_URL: str = _TEXTS["privacy_url"]
