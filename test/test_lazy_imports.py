"""The top-level package imports its subpackages on first access."""

import ast
import importlib
import inspect
import json
import subprocess
import sys
from importlib.util import find_spec

import pytest

import pyRadPlan
import pyRadPlan.ai.agents

_HEAVY = ("torch", "cupy", "jax", "numba", "scipy", "pydantic", "pyRadPlan.dose")


def _assert_light(statement: str) -> None:
    # A fresh interpreter: the test session has long imported the numerical stack.
    check = (
        f"import sys\n{statement}\n"
        f"loaded = [m for m in {_HEAVY!r} if m in sys.modules]\n"
        "assert not loaded, f'loaded {loaded}'\n"
    )
    subprocess.run([sys.executable, "-c", check], check=True)


def test_import_pyradplan_is_light():
    _assert_light("import pyRadPlan")


@pytest.mark.skipif(find_spec("PySide6") is None, reason="needs the gui extra")
def test_import_gui_is_light():
    # The GUI shows its startup splash right after this import, before anything heavy.
    _assert_light("import pyRadPlan.gui")


def _type_checking_imports(package) -> dict[str, str]:
    """Map each name of the package's ``if TYPE_CHECKING:`` imports to its module."""
    tree = ast.parse(inspect.getsource(package))
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING"
    )
    return {
        alias.name: "." * node.level + node.module
        for node in block.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def _lazy_names(package_name: str) -> set[str]:
    """Names in ``__all__`` that a freshly imported package has not defined yet."""
    code = (
        f"import json, {package_name} as package; "
        "print(json.dumps(sorted(set(package.__all__) - set(vars(package)))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return set(json.loads(result.stdout.splitlines()[-1]))


@pytest.mark.parametrize("package", [pyRadPlan, pyRadPlan.ai.agents], ids=lambda p: p.__name__)
def test_lazy_names_match_type_checking_imports(package):
    """The names type checkers see are the ones resolved lazily, from the same modules."""
    imports = _type_checking_imports(package)
    assert set(imports) == _lazy_names(package.__name__)
    for name, module in imports.items():
        source = importlib.import_module(module, package.__name__)
        if hasattr(source, name):
            expected = getattr(source, name)
        else:
            expected = importlib.import_module(f"{source.__name__}.{name}")
        assert getattr(package, name) is expected


def test_all_names_resolve():
    for name in pyRadPlan.__all__:
        assert getattr(pyRadPlan, name) is not None
    assert set(pyRadPlan.__all__) <= set(dir(pyRadPlan))


def test_subpackages_resolve_without_explicit_import():
    check = (
        "import sys, pyRadPlan\n"
        "assert 'pyRadPlan.ct' not in sys.modules\n"
        "assert pyRadPlan.ct is sys.modules['pyRadPlan.ct']\n"
    )
    subprocess.run([sys.executable, "-c", check], check=True)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="no_such_name"):
        pyRadPlan.no_such_name  # noqa: B018
    assert not hasattr(pyRadPlan, "__wrapped__")
