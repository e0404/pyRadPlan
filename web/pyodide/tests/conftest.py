"""Load the SimpleITK shim under the name ``sitk_shim`` next to the real SimpleITK."""

import importlib.util
import sys
from pathlib import Path

import pytest

SHIM_DIR = Path(__file__).resolve().parents[1] / "shims" / "SimpleITK"


def load_shim():
    if "sitk_shim" in sys.modules:
        return sys.modules["sitk_shim"]
    spec = importlib.util.spec_from_file_location(
        "sitk_shim", SHIM_DIR / "__init__.py", submodule_search_locations=[str(SHIM_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["sitk_shim"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def shim():
    return load_shim()


@pytest.fixture(scope="session")
def sitk():
    return pytest.importorskip("SimpleITK")
