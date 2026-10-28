"""The special-cased array backends are imported only when pyRadPlan selects or probes them."""

import importlib
import subprocess
import sys
import threading
import time
from importlib.util import find_spec

import pytest

from pyRadPlan.core import xp_utils
from pyRadPlan.core.xp_utils import _backends

_GPU_BACKENDS = ("torch", "cupy", "jax")


def _run(code: str) -> None:
    # A fresh interpreter: the test session has long imported the backends.
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.fixture
def probe_module(tmp_path, monkeypatch):
    """Name of an importable module that has not been imported yet."""
    name = "_backend_probe_module"
    (tmp_path / f"{name}.py").write_text("VALUE = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    yield name
    sys.modules.pop(name, None)


@pytest.fixture
def slow_module(tmp_path, monkeypatch):
    """Name of a module whose import takes a while, so it can be queried meanwhile."""
    name = "_slow_backend_probe_module"
    (tmp_path / f"{name}.py").write_text("import time\ntime.sleep(0.5)\nVALUE = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    yield name
    sys.modules.pop(name, None)


def test_numerical_subpackages_load_no_gpu_backend():
    _run(
        "import sys\n"
        "import pyRadPlan.cst, pyRadPlan.plan, pyRadPlan.dose, pyRadPlan.optimization\n"
        "import pyRadPlan.analysis, pyRadPlan.io\n"
        f"loaded = [m for m in {_GPU_BACKENDS!r} if m in sys.modules]\n"
        "assert not loaded, f'loaded {loaded}'\n"
    )


def test_cpu_namespace_choice_probes_no_gpu_backend():
    _run(
        "import sys\n"
        "import pyRadPlan\n"
        "from pyRadPlan.core import xp_utils\n"
        "pyRadPlan.settings.xp.prefer_gpu = False\n"
        "assert xp_utils.choose_array_api_namespace().__name__ == 'array_api_compat.numpy'\n"
        f"loaded = [m for m in {_GPU_BACKENDS!r} if m in sys.modules]\n"
        "assert not loaded, f'loaded {loaded}'\n"
    )


@pytest.mark.skipif(find_spec("torch") is None, reason="needs torch")
def test_backend_imported_by_user_code_is_used():
    _run(
        "from pyRadPlan.core.xp_utils import to_numpy\n"
        "import torch\n"
        "assert to_numpy(torch.arange(3)).tolist() == [0, 1, 2]\n"
    )


def test_public_backend_attributes_import_on_access():
    assert xp_utils.torch is _backends.import_backend("torch")
    assert xp_utils.cp is _backends.import_backend("cupy")
    assert xp_utils.jax is _backends.import_backend("jax")


def test_loaded_backend_never_imports(probe_module):
    assert _backends.loaded_backend(probe_module) is None
    assert probe_module not in sys.modules

    module = importlib.import_module(probe_module)

    assert _backends.loaded_backend(probe_module) is module
    assert _backends.loaded_backend("_no_such_submodule", probe_module) is None


def test_blocked_backend_is_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "_blocked_backend", None)

    assert not xp_utils.backend_installed("_blocked_backend")
    assert xp_utils.import_backend("_blocked_backend") is None


@pytest.mark.parametrize("lookup", ["import_backend", "loaded_backend"])
def test_lookup_waits_for_import_in_another_thread(slow_module, lookup):
    importer = threading.Thread(target=importlib.import_module, args=(slow_module,))
    importer.start()
    deadline = time.monotonic() + 5.0
    while slow_module not in sys.modules and time.monotonic() < deadline:
        time.sleep(0.001)
    assert sys.modules[slow_module].__spec__._initializing

    module = getattr(_backends, lookup)(slow_module)

    importer.join()
    # Without waiting, this would be the half-initialized module, still without VALUE.
    assert module.VALUE == 42


def test_import_backend_tells_observers_before_importing(probe_module):
    seen = []
    with xp_utils.observe_backend_imports(lambda name: seen.append((name, name in sys.modules))):
        assert xp_utils.import_backend(probe_module) is sys.modules[probe_module]
        assert xp_utils.import_backend(probe_module) is sys.modules[probe_module]
        assert xp_utils.import_backend("_no_such_backend") is None

    assert seen == [(probe_module, False)]


def test_unknown_namespace_is_chosen_by_module_name(probe_module):
    """Array API namespaces unknown to xp_utils are imported by their module name."""
    seen = []
    with xp_utils.observe_backend_imports(seen.append):
        namespace = xp_utils.choose_array_api_namespace(probe_module)

    assert namespace is sys.modules[probe_module]
    assert seen == [probe_module]
