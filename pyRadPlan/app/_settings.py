"""App-level methods of protocol v1: capabilities, licenses and the settings forms.

``app.info`` v1 reports what this runtime can do (``capabilities``), detected rather than
assumed: image file IO from the functions the installed SimpleITK provides, optional
packages from their import specs, Pyodide from ``sys.platform``. ``app.licenses`` mirrors
the Qt Third-party Licenses dialog (``gui/menus/_help.py``), ``app.settings`` /
``app.settings_set`` the Qt settings dialog (``gui/menus/_settings.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, Mapping, Optional

from ._forms import build_form, jsonable, validate_values
from ._io import (
    export_formats,
    import_formats,
    module_missing,
    pickle_load_reason,
    sitk_missing,
)
from ._profiles import default_profile, running_in_pyodide
from ._session import PreconditionError

#: Shown when the optional AI dependencies are missing (the Qt tooltip ``AI_MISSING_TIP``).
AI_MISSING = "Install the optional 'ai' dependencies (pydantic-ai) to use this"
#: Shown when no AI provider key is configured (the Qt tooltip ``AI_NO_MODEL_TIP``).
AI_NO_MODEL = (
    "No AI model available: set a provider API key (e.g. ANTHROPIC_API_KEY,"
    " OPENAI_API_KEY) or OLLAMA_BASE_URL in the environment or a .env file"
)

_NO_HOST_FILES = "The browser runtime has no access to the files of this computer"


def runtime() -> str:
    """``"pyodide"`` under Pyodide (Emscripten), else ``"native"``."""
    return "pyodide" if running_in_pyodide() else "native"


def ai_disabled_reason() -> Optional[str]:
    """Why the AI features cannot run (Qt ``gui/widgets/ai.ai_disabled_reason``)."""
    if module_missing("pydantic_ai"):
        return AI_MISSING
    try:
        from pyRadPlan.ai.agents import available_models  # noqa: PLC0415

        if not available_models():
            return AI_NO_MODEL
    except Exception as exc:  # noqa: BLE001 - a broken optional stack disables the feature
        return f"The AI features cannot be loaded: {type(exc).__name__}: {exc}"
    return None


def _gpu_reason() -> Optional[str]:
    from pyRadPlan.core.xp_utils._backends import BACKENDS, backend_installed  # noqa: PLC0415

    if any(backend_installed(name) for name in BACKENDS):
        return None
    return f"No GPU array backend ({', '.join(BACKENDS.values())}) is installed"


def feature_reasons(allow_pickle: bool = True) -> dict[str, Optional[str]]:
    """Why each feature is unavailable in this runtime (``None``: available)."""
    from pyRadPlan.optimization.solvers import IPOPT_DISABLED_REASON  # noqa: PLC0415

    pyodide = running_in_pyodide()
    return {
        "image_file_io": sitk_missing("ReadImage", "WriteImage"),
        "dicom_import": sitk_missing("ImageSeriesReader", "ReadImage")
        or module_missing("pydicom"),
        "dicom_export": module_missing("pydicom", "contourpy"),
        "ai": ai_disabled_reason(),
        "ipopt": IPOPT_DISABLED_REASON,
        "gpu": _gpu_reason(),
        "numba": module_missing("numba"),
        "pickle_load": pickle_load_reason(allow_pickle),
        "monte_carlo": "Monte Carlo engines run external programs, which the browser "
        "runtime cannot start"
        if pyodide
        else None,
        "pause": "Pausing needs a second thread, which the browser runtime does not have; "
        "use Stop instead"
        if pyodide
        else None,
        "open_folder": _NO_HOST_FILES if pyodide else None,
        "host_paths": _NO_HOST_FILES if pyodide else None,
        "base_data_persistence": "Base data added in the browser lasts until the page is closed"
        if pyodide
        else None,
    }


def capabilities(allow_pickle: bool = True) -> dict[str, Any]:
    """Return the ``capabilities`` of ``app.info`` v1."""
    features = {
        name: {"available": reason is None, "reason": reason}
        for name, reason in feature_reasons(allow_pickle).items()
    }
    return {
        "features": features,
        "formats": {"import": import_formats(allow_pickle), "export": export_formats()},
    }


def quantity_list() -> list[dict[str, str]]:
    """List the quantities with their Qt selector name, colorbar label and unit."""
    from pyRadPlan.quantities import get_available_quantities  # noqa: PLC0415

    from ._results import quantity_meta  # noqa: PLC0415

    rows = []
    for identifier, quantity in get_available_quantities().items():
        meta_label, unit = quantity_meta(identifier)
        rows.append(
            {
                "identifier": identifier,
                "name": quantity.name,
                "meta_label": meta_label,
                "unit": unit,
            }
        )
    return rows


def runtime_info(allow_pickle: bool = True) -> dict[str, Any]:
    """Return the fields ``app.info`` v1 adds to the v0 ones."""
    return {
        "runtime": runtime(),
        "capabilities": capabilities(allow_pickle),
        "quantities": quantity_list(),
        "default_profile": default_profile(),
    }


# ----------------------------------------------------------------------
# Third-party licenses (Qt Help > Third-party Licenses)
# ----------------------------------------------------------------------

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


@dataclass(frozen=True)
class DistributionInfo:
    """Name, version and license identifier of an installed distribution."""

    name: str
    version: str
    license: str
    distribution: metadata.Distribution


def license_identifier(dist: metadata.Distribution) -> str:
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


_license_identifier = license_identifier


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
    return field if "\n" in field else None


_license_text = license_text


def _distributions() -> dict[str, metadata.Distribution]:
    found: dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name and name.lower() not in found:
            found[name.lower()] = dist
    return dict(sorted(found.items()))


def installed_distributions() -> list[DistributionInfo]:
    """Return the installed distributions sorted by name, one entry per name."""
    return [
        DistributionInfo(
            name=dist.metadata.get("Name"),
            version=dist.version or "",
            license=license_identifier(dist),
            distribution=dist,
        )
        for dist in _distributions().values()
    ]


def licenses(name: Optional[str] = None) -> Any:
    """Installed distributions, or one with its license text (``app.licenses``)."""
    if name is None:
        return [
            {"name": info.name, "version": info.version, "license": info.license}
            for info in installed_distributions()
        ]
    dist = _distributions().get(str(name).lower())
    if dist is None:
        raise KeyError(f"No installed distribution named {name!r}")
    return {
        "name": dist.metadata.get("Name"),
        "version": dist.version or "",
        "license": license_identifier(dist),
        "text": license_text(dist),
    }


# ----------------------------------------------------------------------
# Settings (Qt Settings menu)
# ----------------------------------------------------------------------

SECTION_TITLES = {"general": "General", "xp": "XP (Backend)", "ai": "AI", "gui": "GUI"}
_NESTED = ("xp", "ai", "gui")


def _section_model(section: str) -> tuple[type, Any]:
    from pyRadPlan._settings import PyRadPlanSettings, get_settings  # noqa: PLC0415

    settings = get_settings()
    if section == "general":
        return PyRadPlanSettings, settings
    if section in _NESTED:
        return type(getattr(settings, section)), getattr(settings, section)
    raise ValueError(f"Unknown settings section {section!r}; use one of {list(SECTION_TITLES)}")


def _locked_reason(section: str, name: str) -> Optional[str]:
    """Why a settings field cannot be changed in this runtime, ``None`` if it can."""
    if section == "general" and name == "base_data_path":
        return _NO_HOST_FILES if running_in_pyodide() else None
    if section == "xp" and _gpu_reason() and module_missing("numba"):
        return "This runtime computes with numpy only (no GPU array backend, no numba)"
    if section == "ai":
        return AI_MISSING if module_missing("pydantic_ai") else None
    return None


def _section_form(section: str) -> dict[str, Any]:
    model_cls, target = _section_model(section)
    exclude = _NESTED if section == "general" else ()
    names = [name for name in model_cls.model_fields if name not in exclude]
    form = build_form(
        model_cls,
        form_id=f"settings:{section}",
        title=f"{SECTION_TITLES[section]} Settings",
        values={name: getattr(target, name) for name in names},
        exclude=exclude,
    )
    for descriptor in form["fields"]:
        reason = _locked_reason(section, descriptor["name"])
        if reason is not None:
            descriptor["editable"] = False
            descriptor["reason"] = reason
    return form


def settings_sections() -> dict[str, Any]:
    """Return the settings forms, one per section (``app.settings``)."""
    return {
        "sections": [
            {"id": section, "title": title, "form": _section_form(section)}
            for section, title in SECTION_TITLES.items()
        ]
    }


def env_names(model_cls: type, field_name: str) -> list[str]:
    """Environment variable names a settings field is read from.

    The prefixed name comes first and is the one written back; any additional
    names declared through a ``validation_alias`` (e.g. a legacy variable) are
    returned after it so they can be cleared, which keeps a stale alias from
    overriding what the user just chose.
    """
    from pydantic import AliasChoices  # noqa: PLC0415

    prefix = model_cls.model_config.get("env_prefix", "")
    names = [f"{prefix}{field_name}".upper()]
    alias = model_cls.model_fields[field_name].validation_alias
    aliases = alias.choices if isinstance(alias, AliasChoices) else [alias]
    for candidate in aliases:
        if isinstance(candidate, str) and candidate.upper() not in names:
            names.append(candidate.upper())
    return names


_env_names = env_names


def apply_settings_values(
    model_cls: type,
    target: Any,
    values: Mapping[str, Any],
    render: Callable[[Any], str] = str,
) -> None:
    """Assign *values* to the settings object *target* and mirror them into the environment.

    The environment is written under the variable names *model_cls* reads the
    fields from (stale legacy aliases are cleared), so freshly constructed
    settings instances observe the change as well. ``None`` removes the variable;
    other values are written as ``render(value)``.
    """
    for name, value in values.items():
        setattr(target, name, value)
        env_name, *stale = env_names(model_cls, name)
        for other in stale:
            os.environ.pop(other, None)
        if value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = render(value)


def _render_json(value: Any) -> str:
    return str(jsonable(value))


def settings_set(section: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and apply settings values, mirrored into the ``PYRADPLAN_*`` variables.

    Nothing is written when a value is rejected (``FormValidationError`` with details) or
    a field cannot be changed in this runtime (``PreconditionError(code="locked")``).
    """
    if not isinstance(values, Mapping):
        raise TypeError("values must be an object")
    model_cls, target = _section_model(section)
    exclude = _NESTED if section == "general" else ()
    locked = {
        name: reason
        for name in values
        if name in model_cls.model_fields and (reason := _locked_reason(section, name)) is not None
    }
    if locked:
        text = "; ".join(f"{name}: {reason}" for name, reason in locked.items())
        raise PreconditionError(f"Cannot change {text}", "locked")
    validated = validate_values(model_cls, values, exclude=exclude)
    apply_settings_values(model_cls, target, validated, render=_render_json)
    return settings_sections()
