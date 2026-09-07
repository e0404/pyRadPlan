"""Tests for the OpenBLAS threaded-GEMM race warning."""

import warnings

import numpy as np
import pytest

from pyRadPlan.core.xp_utils import openblas_has_gemm_race, warn_on_unreliable_openblas


def _set_config(monkeypatch, blas: dict):
    monkeypatch.setattr(np.__config__, "CONFIG", {"Build Dependencies": {"blas": blas}})


@pytest.mark.parametrize(
    "configuration",
    [
        "OpenBLAS 0.3.27  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Zen MAX_THREADS=24",
        "OpenBLAS 0.2.20",
    ],
)
def test_warns_on_affected_openblas(monkeypatch, configuration):
    _set_config(monkeypatch, {"openblas configuration": configuration})
    with pytest.warns(RuntimeWarning, match="multithreaded GEMM"):
        warn_on_unreliable_openblas()


@pytest.mark.parametrize(
    "configuration",
    [
        "OpenBLAS 0.3.28",
        "OpenBLAS 0.3.31.188.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=24",
    ],
)
def test_no_warning_on_fixed_openblas(monkeypatch, configuration):
    _set_config(monkeypatch, {"openblas configuration": configuration})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_unreliable_openblas()


def test_no_warning_without_openblas(monkeypatch):
    # e.g. MKL-backed builds expose no openblas configuration
    _set_config(monkeypatch, {"name": "mkl"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_unreliable_openblas()


def test_no_warning_on_unexpected_config(monkeypatch):
    monkeypatch.setattr(np.__config__, "CONFIG", {})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_unreliable_openblas()


@pytest.mark.parametrize(
    ("configuration", "expected"),
    [
        ("OpenBLAS 0.3.27 DYNAMIC_ARCH Zen", True),
        ("OpenBLAS 0.3.28", False),
        ("OpenBLAS 0.3.31.188.0 SkylakeX", False),
        (None, False),
    ],
)
def test_openblas_has_gemm_race(monkeypatch, configuration, expected):
    blas = {} if configuration is None else {"openblas configuration": configuration}
    _set_config(monkeypatch, blas)
    assert openblas_has_gemm_race() is expected
