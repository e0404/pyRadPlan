"""The desktop bundle must install the same GUI dependencies as the library's gui extra."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, where pytest brings tomli
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_bundle_requires_match_gui_extra():
    root = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    bundle = tomllib.loads((ROOT / "packaging/briefcase/pyproject.toml").read_text("utf-8"))
    requires = bundle["tool"]["briefcase"]["app"]["pyradplan_app"]["requires"]

    assert set(root["project"]["optional-dependencies"]["gui"]) <= set(requires)
