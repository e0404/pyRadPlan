"""Geometry functions for LPS system."""

import numpy as np
import array_api_compat

from ..core.xp_utils.typing import Array


def get_gantry_rotation_matrix(gantry_angle: Array) -> Array:
    """
    Calculate the rotation matrix for gantry.

    Represents an active, counter-clockwise rotation Matrix for Gantry around z
    with pre-multiplication of the matrix (R*x).

    Parameters
    ----------
    gantry_angle : Array
        The angle of gantry rotation in degrees.

    Returns
    -------
    Array
        The rotation matrix.

    Notes
    -----
    Gantry rotation is physically an active rotation of a beam
    vector around the target / isocenterin the patient coordinate
    system

    The LPS system is right-handed, where the gantry rotates counter-clockwise around the z-axis
    and the couch rotates counter-clockwise around the y-axis.
    The gantry rotation is an active rotation of a beam vector around the target/isocenter
    in the patient coordinate system.
    """
    xp = array_api_compat.array_namespace(gantry_angle)

    gantry_angle = gantry_angle / 180 * xp.pi

    c = xp.cos(gantry_angle)
    s = xp.sin(gantry_angle)
    dev = array_api_compat.device(gantry_angle)
    zero = xp.zeros((), dtype=c.dtype, device=dev)
    one = xp.ones((), dtype=c.dtype, device=dev)

    r_gantry = xp.stack(
        [
            xp.stack([c, -s, zero]),
            xp.stack([s, c, zero]),
            xp.stack([zero, zero, one]),
        ]
    )

    return r_gantry


def get_couch_rotation_matrix(couch_angle: Array) -> Array:
    """
    Calculate the rotation matrix for the couch.

    Parameters
    ----------
    couch_angle : Array
        The angle of couch rotation in degrees.

    Returns
    -------
    Array
        The rotation matrix.

    Notes
    -----
    The LPS system is right-handed, where the gantry rotates counter-clockwise around the z-axis
    and the couch rotates counter-clockwise around the y-axis.
    The gantry rotation is an active rotation of a beam vector around the target/isocenter
    in the patient coordinate system.
    """

    xp = array_api_compat.array_namespace(couch_angle)
    couch_angle = couch_angle / 180 * xp.pi

    c = xp.cos(couch_angle)
    s = xp.sin(couch_angle)
    dev = array_api_compat.device(couch_angle)
    zero = xp.zeros((), dtype=c.dtype, device=dev)
    one = xp.ones((), dtype=c.dtype, device=dev)

    r_gantry = xp.stack(
        [
            xp.stack([c, zero, s]),
            xp.stack([zero, one, zero]),
            xp.stack([-s, zero, c]),
        ]
    )

    return r_gantry


def get_beam_rotation_matrix(gantry_angle: Array, couch_angle: Array) -> Array:
    """
    Calculate the rotation matrix for gantry and couch angles.

    Represents active, counter-clockwise rotation for couch around y
    with pre-multiplication of the matrix (R*x)

    Parameters
    ----------
    gantry_angle : Array
        The angle of gantry rotation in degrees.
    couch_angle : Array
        The angle of couch rotation in degrees.

    Returns
    -------
    Array
        The 3x3 rotation matrix.

    Notes
    -----
    Couch rotation is physically a passive rotation of the
    patient system around the beam target point / isocenter

    The LPS system is right-handed, where the gantry rotates counter-clockwise around the z-axis
    and the couch rotates counter-clockwise around the y-axis.
    The gantry rotation is an active rotation of a beam vector around the target/isocenter
    in the patient coordinate system.

    Examples
    --------
    >>> get_beam_rotation_matrix(90, 45)
    array([[ 0.        , -0.70710678,  0.70710678],
           [ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.70710678,  0.70710678]])
    """
    # Array API compatibility requires angles to be arrays.
    # Force to numpy arrays. Legacy code would have done so either way.
    if isinstance(gantry_angle, (int, float)):
        gantry_angle = np.asarray(gantry_angle)

    if isinstance(couch_angle, (int, float)):
        couch_angle = np.asarray(couch_angle)

    r_gantry = get_gantry_rotation_matrix(gantry_angle)
    r_couch = get_couch_rotation_matrix(couch_angle)

    rot_mat = r_couch @ r_gantry

    return rot_mat
