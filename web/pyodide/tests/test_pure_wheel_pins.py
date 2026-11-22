"""The vendored pure-Python wheels satisfy pyproject.toml and are hash-pinned."""

import importlib.util
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PYODIDE_DIR = Path(__file__).resolve().parents[1]
REPO = PYODIDE_DIR.parents[1]


def _build_module():
    spec = importlib.util.spec_from_file_location("pyodide_build", PYODIDE_DIR / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_pin_has_a_hash():
    build = _build_module()
    assert set(build.pure_wheel_pins()) == set(build.PURE_WHEEL_SHA256)


def test_pins_satisfy_pyproject_bounds():
    build = _build_module()
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    bounds = {}
    for text in pyproject["project"]["dependencies"]:
        req = Requirement(text)
        bounds[canonicalize_name(req.name)] = req.specifier
    checked = 0
    for pin in build.pure_wheel_pins():
        name, version = pin.split("==")
        specifier = bounds.get(canonicalize_name(name))
        if specifier is not None:
            assert specifier.contains(version), f"{pin} violates pyproject.toml ({specifier})"
            checked += 1
    assert checked >= 5
