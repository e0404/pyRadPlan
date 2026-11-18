"""Session profiles: default plan and locked plan fields per deployment."""

from __future__ import annotations

import sys
from typing import Any, Mapping, Optional

from ._plan import DEFAULT_PLAN

#: Default plan of the browser frontend: coarser dose grid (owner decision) and the Qt
#: plan form's "total" dose convention.
WEB_DEFAULT_PLAN: dict[str, Any] = {
    **DEFAULT_PLAN,
    "dose_grid_resolution": [5.0, 5.0, 5.0],
    "dose_convention": "total",
}

PROFILES: dict[str, dict[str, Any]] = {
    "native": {"default_plan": dict(DEFAULT_PLAN), "locked_fields": []},
    "web": {"default_plan": dict(WEB_DEFAULT_PLAN), "locked_fields": []},
}


def running_in_pyodide() -> bool:
    """Whether this interpreter runs under Pyodide (Emscripten)."""
    return sys.platform == "emscripten"


def default_profile() -> str:
    """``"web"`` under Pyodide, else ``"native"``."""
    return "web" if running_in_pyodide() else "native"


def resolve_profile(
    profile: Optional[str], options: Optional[Mapping[str, Any]] = None
) -> tuple[str, dict[str, Any]]:
    """Return the profile name and the session options it gives, *options* laid over it."""
    name = profile or default_profile()
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; expected one of {sorted(PROFILES)}")
    resolved = {
        key: list(value) if isinstance(value, list) else dict(value)
        for key, value in PROFILES[name].items()
    }
    for key, value in dict(options or {}).items():
        if key == "default_plan":
            resolved["default_plan"] = {**resolved["default_plan"], **dict(value)}
        else:
            resolved[key] = value
    return name, resolved
