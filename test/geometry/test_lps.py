import math

import array_api_strict as xp
from pyRadPlan.geometry.lps import (
    get_beam_rotation_matrix,
    get_gantry_rotation_matrix,
    get_couch_rotation_matrix,
)

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
