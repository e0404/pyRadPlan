"""Tests for the OpenMP runtime clash detection in pyRadPlan.util.openmp."""

import os
import sys

import pytest

from pyRadPlan.util import openmp

#: Captured before the autouse fixture below monkeypatches the module attribute, so
#: the two tests exercising the real .env lookup can still call the genuine function.
_real_kmp_env_from_dotenv = openmp._kmp_env_from_dotenv


@pytest.fixture(autouse=True)
def _clear_scan_cache():
    """The on-disk scan is cached per package; keep tests independent of each other."""
    openmp.runtimes_shipped_by.cache_clear()
    yield
    openmp.runtimes_shipped_by.cache_clear()


@pytest.fixture(autouse=True)
def _no_dotenv_fallback(monkeypatch):
    """Isolate tests from whatever real .env file happens to exist on this machine.

    A developer's local .env may itself set KMP_DUPLICATE_LIB_OK (see
    test_duplicate_runtimes_allowed_from_dotenv below); tests that want that
    behaviour re-patch _kmp_env_from_dotenv themselves.
    """
    monkeypatch.setattr(openmp, "_kmp_env_from_dotenv", lambda: None)


# --------------------------------------------------------------------------
# Name normalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("libiomp5md.dll", "libiomp5md"),
        # delvewheel/auditwheel append a content hash when they vendor a library.
        ("libiomp5md-bbfd5d1c3843841454f68a54ee61f1f8.dll", "libiomp5md"),
        (r"C:\site-packages\torch\lib\libiomp5md.dll", "libiomp5md"),
        ("libiomp5.so", "libiomp5"),
        ("libomp.so.5", "libomp"),
        ("libomp.dylib", "libomp"),
        ("LibIOMP5MD.DLL", "libiomp5md"),
        # Runtimes that tolerate duplicates are not tracked.
        ("libgomp.so.1", None),
        ("vcomp140.dll", None),
        ("libiomp5mdstubs.dll", None),
        ("numpy.core._multiarray_umath.pyd", None),
    ],
)
def test_runtime_key(path, expected):
    assert openmp.runtime_key(path) == expected


# --------------------------------------------------------------------------
# KMP_DUPLICATE_LIB_OK
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["TRUE", "true", "1", "yes", "on", " True "])
def test_duplicate_runtimes_allowed(monkeypatch, value):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", value)
    assert openmp.duplicate_runtimes_allowed()


@pytest.mark.parametrize("value", ["", "FALSE", "0", "no"])
def test_duplicate_runtimes_not_allowed(monkeypatch, value):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", value)
    assert not openmp.duplicate_runtimes_allowed()


def test_duplicate_runtimes_not_allowed_when_unset(monkeypatch):
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    assert not openmp.duplicate_runtimes_allowed()


def test_duplicate_runtimes_allowed_from_dotenv(monkeypatch):
    """A value only present in .env (pydantic-settings never puts it in os.environ) is honoured.

    KMP_DUPLICATE_LIB_OK is read by the *native* OpenMP runtime via C getenv(), not
    anything Python-side, so a .env-only value must be copied into os.environ for
    it to actually take effect there -- not just satisfy this Python-side check.
    """
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.setattr(openmp, "_kmp_env_from_dotenv", lambda: "TRUE")

    assert openmp.duplicate_runtimes_allowed()
    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"


def test_dotenv_fallback_not_consulted_when_env_var_is_set(monkeypatch):
    """An explicitly set (even falsy) env var wins over .env, without reading it."""
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "FALSE")
    monkeypatch.setattr(
        openmp,
        "_kmp_env_from_dotenv",
        lambda: pytest.fail(".env should not be consulted when the env var is set"),
    )

    assert not openmp.duplicate_runtimes_allowed()


def test_dotenv_fallback_missing_dotenv_or_key(monkeypatch):
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.setattr(openmp, "_kmp_env_from_dotenv", lambda: None)

    assert not openmp.duplicate_runtimes_allowed()
    assert "KMP_DUPLICATE_LIB_OK" not in os.environ


def test_kmp_env_from_dotenv_reads_only_that_key(tmp_path, monkeypatch):
    """The real .env lookup pulls in just this one key, not the whole file.

    Uses a made-up sentinel name rather than a real-looking one (e.g.
    ANTHROPIC_API_KEY): something else in a full suite run may legitimately have
    that in os.environ already (pyRadPlan.ai.agents.load_ai_env reading the
    project's own .env), which would make that name a false negative here.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(["_TEST_OPENMP_SENTINEL=should-not-be-touched", "KMP_DUPLICATE_LIB_OK=TRUE", ""])
    )
    monkeypatch.chdir(tmp_path)
    before = set(os.environ)

    assert _real_kmp_env_from_dotenv() == "TRUE"
    assert set(os.environ) == before  # dotenv_values() reads the file; it never mutates os.environ


def test_kmp_env_from_dotenv_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _real_kmp_env_from_dotenv() is None


# --------------------------------------------------------------------------
# Loaded / shipped runtimes
# --------------------------------------------------------------------------


def test_loaded_modules_finds_the_interpreter():
    """The enumeration works on the platforms we support it on."""
    modules = openmp._loaded_modules()
    if sys.platform not in ("win32",) and not sys.platform.startswith("linux"):
        pytest.skip("module enumeration is not implemented on this platform")
    assert modules
    assert any(os.path.basename(m).lower().startswith("python") for m in modules)


def test_loaded_runtimes_keys_are_normalized(monkeypatch):
    monkeypatch.setattr(
        openmp,
        "_loaded_modules",
        lambda: [
            "/opt/torch/lib/libiomp5.so",
            "/opt/other/libiomp5-0123456789ab.so",
            "/opt/numpy/libgomp.so.1",
        ],
    )
    loaded = openmp.loaded_runtimes()
    assert set(loaded) == {"libiomp5"}  # libgomp tolerates duplicates
    assert len(loaded["libiomp5"]) == 2
    assert openmp.duplicate_loaded_runtimes() == loaded


def test_no_duplicates_when_one_runtime(monkeypatch):
    monkeypatch.setattr(openmp, "_loaded_modules", lambda: ["/opt/torch/lib/libiomp5.so"])
    assert openmp.duplicate_loaded_runtimes() == {}


def test_runtimes_shipped_by_scans_vendor_directory(tmp_path, monkeypatch):
    """A wheel's vendored runtime is found without importing the package."""
    package = tmp_path / "fakeopt"
    package.mkdir()
    (package / "__init__.py").write_text("raise AssertionError('must not be imported')")
    vendor = tmp_path / "fakeopt.libs"
    vendor.mkdir()
    (vendor / "libiomp5md-0123456789abcdef0123456789abcdef.dll").write_bytes(b"")
    (vendor / "unrelated.dll").write_bytes(b"")

    monkeypatch.syspath_prepend(str(tmp_path))
    shipped = openmp.runtimes_shipped_by("fakeopt")

    assert set(shipped) == {"libiomp5md"}
    assert "fakeopt" not in sys.modules  # find_spec must not execute the package


def test_runtimes_shipped_by_unknown_package():
    assert openmp.runtimes_shipped_by("a_package_that_does_not_exist") == {}


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------


def _fake_clash(monkeypatch):
    monkeypatch.setattr(
        openmp,
        "runtimes_shipped_by",
        lambda pkg: {"libiomp5md": frozenset({os.path.normcase("/site/pkg.libs/libiomp5md.dll")})},
    )
    monkeypatch.setattr(openmp, "_loaded_modules", lambda: ["/site/torch/lib/libiomp5md.dll"])


def test_blocked_when_shipped_runtime_already_loaded(monkeypatch):
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    _fake_clash(monkeypatch)

    reason = openmp.blocked_by_openmp("pkg")
    assert reason is not None
    assert "libiomp5md" in reason


def test_not_blocked_when_env_var_allows_duplicates(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    _fake_clash(monkeypatch)

    assert openmp.blocked_by_openmp("pkg") is None


def test_not_blocked_when_same_file_is_the_loaded_one(monkeypatch):
    """The package's own runtime being loaded is not a clash -- it is the only copy."""
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.setattr(
        openmp,
        "runtimes_shipped_by",
        lambda pkg: {"libiomp5md": frozenset({os.path.normcase("/site/pkg.libs/libiomp5md.dll")})},
    )
    monkeypatch.setattr(openmp, "_loaded_modules", lambda: ["/site/pkg.libs/libiomp5md.dll"])

    assert openmp.blocked_by_openmp("pkg") is None


def test_blocked_when_runtime_already_loaded_twice(monkeypatch):
    """Catches the clash even for a package that ships no runtime of its own."""
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.setattr(openmp, "runtimes_shipped_by", lambda pkg: {})
    monkeypatch.setattr(
        openmp,
        "_loaded_modules",
        lambda: ["/site/torch/lib/libiomp5md.dll", "/site/other.libs/libiomp5md-abc12345.dll"],
    )

    reason = openmp.blocked_by_openmp("pkg")
    assert reason is not None and "twice" in reason


def test_not_blocked_in_a_clean_process(monkeypatch):
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.setattr(openmp, "runtimes_shipped_by", lambda pkg: {})
    monkeypatch.setattr(openmp, "_loaded_modules", lambda: ["/usr/lib/libc.so.6"])

    assert openmp.blocked_by_openmp("pkg") is None


# --------------------------------------------------------------------------
# Solver registration
# --------------------------------------------------------------------------


def test_ipopt_registration_matches_the_openmp_verdict():
    """IPOPT is registered exactly when no clash was detected at import time."""
    from pyRadPlan.optimization.solvers import (
        IPOPT_DISABLED_REASON,
        OptimizerIpopt,
        get_available_solvers,
    )

    if IPOPT_DISABLED_REASON is None:
        assert OptimizerIpopt is not None
        assert "ipopt" in get_available_solvers()
    else:
        assert OptimizerIpopt is None
        assert "ipopt" not in get_available_solvers()
