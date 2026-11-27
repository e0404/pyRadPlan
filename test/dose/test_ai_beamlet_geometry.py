"""Unit tests of the AI beamlet engine's beamlet rotation."""

import array_api_strict
import numpy as np
import pytest

from pyRadPlan.dose.engines._base_ai_beamlet import _rotation_onto

_TOLERANCE = {np.float32: 1e-6, np.float64: 1e-14}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("angle", [0.0, 1e-9, 1e-6, 1e-3, 0.3])
def test_rotation_maps_the_central_ray_onto_a_beamlet(angle, dtype):
    """The central ray, at 10 m from the source, maps onto a beamlet at a small angle.

    ``angle == 0`` is the central ray itself.
    """
    a = np.array([0.0, 1e4, 0.0])
    b = 2e4 * np.array([np.sin(angle) * 0.6, np.cos(angle), np.sin(angle) * 0.8])

    rot = _rotation_onto(a.astype(dtype), b.astype(dtype))

    tol = _TOLERANCE[dtype]
    assert rot.dtype == dtype
    assert np.allclose(rot.T @ rot, np.eye(3), rtol=0, atol=tol)
    assert np.isclose(np.linalg.det(rot.astype(np.float64)), 1.0, rtol=0, atol=tol)
    assert np.allclose(rot @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), rtol=0, atol=tol)


def test_rotation_of_the_central_ray_is_the_identity():
    """A beamlet along the central ray needs no special case."""
    a = np.array([1.0, -2.0, 0.5])
    assert np.array_equal(_rotation_onto(a, 3 * a), np.eye(3))


def test_rotation_follows_the_array_namespace():
    """The rotation runs on any Array API namespace."""
    a = array_api_strict.asarray([0.0, 1.0, 0.0])
    b = array_api_strict.asarray([0.1, 1.0, -0.2])

    rot = _rotation_onto(a, b)

    assert isinstance(rot, type(a))
    expected = _rotation_onto(np.asarray([0.0, 1.0, 0.0]), np.asarray([0.1, 1.0, -0.2]))
    assert np.allclose(np.asarray(rot), expected, rtol=0, atol=1e-15)
