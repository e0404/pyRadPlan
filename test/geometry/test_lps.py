import math

import array_api_strict as xp
import pytest
from pyRadPlan.geometry.lps import (
    get_beam_rotation_matrix,
    get_gantry_rotation_matrix,
    get_couch_rotation_matrix,
)
from pyRadPlan.core.xp_utils import cupy_available

ATOL = 1e-6


def test_get_gantry_rotation_matrix_90():
    gantry_angle = xp.asarray(90.0)
    expected = xp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=xp.float64)
    result = get_gantry_rotation_matrix(gantry_angle)
    assert xp.all(xp.abs(result - expected) < ATOL)


def test_get_gantry_rotation_matrix_0():
    gantry_angle = xp.asarray(0.0)
    expected = xp.eye(3)
    result = get_gantry_rotation_matrix(gantry_angle)
    assert xp.all(result == expected)


def test_get_gantry_rotation_matrix_180():
    gantry_angle = xp.asarray(180.0)
    expected = xp.asarray([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=xp.float64)
    result = get_gantry_rotation_matrix(gantry_angle)
    assert xp.all(xp.abs(result - expected) < ATOL)


def test_get_gantry_rotation_matrix_360():
    gantry_angle = xp.asarray(360.0)
    expected = xp.eye(3)
    result = get_gantry_rotation_matrix(gantry_angle)
    assert xp.all(xp.abs(result - expected) < ATOL)


def test_get_couch_rotation_matrix_90():
    couch_angle = xp.asarray(90.0)
    expected = xp.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=xp.float64)
    result = get_couch_rotation_matrix(couch_angle)
    assert xp.all(xp.abs(result - expected) < ATOL)


def test_get_couch_rotation_matrix_0():
    couch_angle = xp.asarray(0.0)
    expected = xp.eye(3)
    result = get_couch_rotation_matrix(couch_angle)
    assert xp.all(xp.abs(result - expected) < ATOL)


def test_get_couch_rotation_matrix_180():
    couch_angle = xp.asarray(180.0)
    expected = xp.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=xp.float64)
    result = get_couch_rotation_matrix(couch_angle)
    assert xp.all(xp.abs(result - expected) < ATOL)


def test_get_couch_rotation_matrix_360():
    couch_angle = xp.asarray(360.0)
    expected = xp.eye(3)
    result = get_couch_rotation_matrix(couch_angle)
    assert xp.all(xp.abs(result - expected) < ATOL)


def test_get_beam_rotation_matrix_90_45():
    gantry_angle = xp.asarray(90.0)
    couch_angle = xp.asarray(45.0)
    a45 = math.sin(math.radians(45))
    expected = xp.asarray([[0.0, -a45, a45], [1.0, 0.0, 0.0], [0.0, a45, a45]])
    result = get_beam_rotation_matrix(gantry_angle, couch_angle)
    assert xp.all(xp.abs(result - expected) < ATOL)


def test_get_beam_rotation_matrix_0_0():
    gantry_angle = xp.asarray(0.0)
    couch_angle = xp.asarray(0.0)
    expected = xp.eye(3)
    result = get_beam_rotation_matrix(gantry_angle, couch_angle)
    assert xp.all(xp.abs(result - expected) < ATOL)


def test_get_beam_rotation_matrix_45_90():
    gantry_angle = xp.asarray(45.0)
    couch_angle = xp.asarray(90.0)
    a45 = math.sin(math.radians(45))
    expected = xp.asarray([[0.0, 0, 1.0], [a45, a45, 0.0], [-a45, a45, 0.0]])
    result = get_beam_rotation_matrix(gantry_angle, couch_angle)
    assert xp.all(xp.abs(result - expected) < ATOL)


def test_get_beam_rotation_matrix_360_360():
    gantry_angle = xp.asarray(360.0)
    couch_angle = xp.asarray(360.0)
    expected = xp.eye(3)
    result = get_beam_rotation_matrix(gantry_angle, couch_angle)
    assert xp.all(xp.abs(result - expected) < ATOL)


# Before changes, lps.py might encountered cp.asarray([0, cupy_array])
# which won't work. These tests make sure that it won't be changed back in the future.
@pytest.mark.skipif(not cupy_available(), reason="CuPy is not available")
def test_get_gantry_rotation_matrix_cupy():
    import cupy as cp

    gantry_angle = cp.asarray(90.0)
    result = get_gantry_rotation_matrix(gantry_angle)
    expected = cp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=cp.float64)
    assert cp.all(cp.abs(result - expected) < ATOL)


@pytest.mark.skipif(not cupy_available(), reason="CuPy is not available")
def test_get_couch_rotation_matrix_cupy():
    import cupy as cp

    couch_angle = cp.asarray(90.0)
    result = get_couch_rotation_matrix(couch_angle)
    expected = cp.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=cp.float64)
    assert cp.all(cp.abs(result - expected) < ATOL)
