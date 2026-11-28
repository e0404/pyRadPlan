"""Collapsed cone kernel: file data, refit into polar bins and energy check."""

import numpy as np
import pytest
from pydantic import ValidationError

from pyRadPlan.machines import PhotonCollapsedConeKernel
from pyRadPlan.machines.photons import load_collapsed_cone_kernel
from pyRadPlan.machines.photons._collapsed_cone_fit import fit_collapsed_cone_kernel, load_kernel


def _synthetic_edk():
    """Angle and radius edges of a synthetic conical kernel and its inner and outer radii."""
    radius_edges = 10 * np.array(
        [
            0,
            0.05,
            0.1,
            0.15,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.8,
            1,
            1.5,
            2,
            3,
            4,
            5,
            6,
            8,
            10,
            15,
            20,
            30,
            40,
            50,
            60,
        ]
    )
    return np.arange(0, 180.01, 3.75), radius_edges, radius_edges[:-1], radius_edges[1:]


def test_fit_kernel_synthetic():
    """The fit recovers the attenuation coefficients and energies of a synthetic kernel."""
    angle_edges, radius_edges, r1, r2 = _synthetic_edk()
    theta = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    cone_a = 0.02 * np.cos(np.deg2rad(theta / 2)) ** 4
    cone_b = 0.01 * np.ones_like(theta)
    edk = {
        "angleEdges": angle_edges,
        "radiusEdges": radius_edges,
        "primary": np.outer(cone_a, np.exp(-0.3 * r1) - np.exp(-0.3 * r2)),
        "scatter": np.outer(cone_b, np.exp(-0.02 * r1) - np.exp(-0.02 * r2)),
    }
    kernel = fit_collapsed_cone_kernel(edk, 12)
    assert kernel.num_polar_angles == 12
    np.testing.assert_allclose(kernel.polar_edges, np.arange(0, 181, 15))
    np.testing.assert_allclose(kernel.primary_weights.sum(), edk["primary"].sum(), rtol=1e-12)
    np.testing.assert_allclose(kernel.scatter_weights.sum(), edk["scatter"].sum(), rtol=1e-12)
    np.testing.assert_allclose(kernel.primary_attenuations, 0.3, rtol=1e-4)
    np.testing.assert_allclose(kernel.scatter_attenuations, 0.02, rtol=1e-3)
    assert np.all(kernel.residual["primaryCum"] < 1e-4)

    kernel = fit_collapsed_cone_kernel(edk, 12, 1.5, 1, 200)
    assert kernel.min_radius == 1.5
    assert kernel.scatter_max_radius == 200
    np.testing.assert_allclose(kernel.scatter_attenuations, 0.02, rtol=1e-3)


def test_fit_kernel_two_scatter_terms():
    """The two-term fit recovers the faster scatter coefficient."""
    angle_edges, radius_edges, r1, r2 = _synthetic_edk()

    def shell(a):
        return np.exp(-a * r1) - np.exp(-a * r2)

    edk = {
        "angleEdges": angle_edges,
        "radiusEdges": radius_edges,
        "primary": 0.02 * np.outer(np.ones(48), shell(0.3)),
        "scatter": np.outer(np.ones(48), 0.006 * shell(0.1) + 0.004 * shell(0.02)),
    }
    kernel = fit_collapsed_cone_kernel(edk, 12, 0, 2)
    assert kernel.scatter_weights.shape == (2, 12)
    np.testing.assert_allclose(kernel.scatter_weights.sum(), edk["scatter"].sum(), rtol=1e-10)
    np.testing.assert_allclose(kernel.scatter_attenuations[0], 0.1, rtol=0.02)


def test_default_kernel_data():
    """The 6 MV kernel is valid and its energy does not change with refitting."""
    kernel = load_kernel(None, 12)
    assert kernel.primary_weights.shape == (12,)
    assert np.all(
        (kernel.primary_weights >= 0)
        & (kernel.scatter_weights >= 0)
        & (kernel.primary_attenuations > 0)
        & (kernel.scatter_attenuations > 0)
    )
    total = kernel.primary_weights.sum() + kernel.scatter_weights.sum()
    assert 0.9 < total < 1
    assert kernel.nominal_energy == 6
    kernel6 = load_kernel(None, 6)
    np.testing.assert_allclose(
        kernel6.primary_weights.sum() + kernel6.scatter_weights.sum(), total, rtol=1e-10
    )
    kernel2 = load_kernel(None, 12, 2)
    assert kernel2.scatter_weights.shape == (2, 12)
    np.testing.assert_allclose(
        kernel2.primary_weights.sum() + kernel2.scatter_weights.sum(), total, rtol=1e-10
    )


def test_check_kernel_energy():
    """A kernel of another nominal energy is refused, an unknown energy only warned about."""
    kernel = load_collapsed_cone_kernel()
    assert kernel.check_energy(6.0) == (True, "")
    ok, msg = kernel.check_energy(18.0)
    assert not ok and "does not match" in msg
    ok, msg = kernel.model_copy(update={"nominal_energy": None}).check_energy(6.0)
    assert ok and "unknown" in msg


def test_kernel_model_round_trip():
    """The kernel validates its own dump with matRad's field names (A, a, B, b)."""
    kernel = load_kernel(None, 6, 2)
    data = kernel.model_dump(by_alias=True)
    assert {"A", "a", "B", "b", "polarAngles", "polarEdges"} <= set(data)
    copy = PhotonCollapsedConeKernel.model_validate(data)
    np.testing.assert_array_equal(copy.weights, kernel.weights)
    np.testing.assert_array_equal(copy.attenuations, kernel.attenuations)
    assert copy.num_polar_angles == 6 and copy.num_scatter_exponentials == 2


def test_kernel_model_rejects_inconsistent_shapes():
    """Bins, edges and components have to match."""
    kernel = load_collapsed_cone_kernel()
    data = kernel.model_dump(by_alias=True)
    with pytest.raises(ValidationError, match="one entry more"):
        PhotonCollapsedConeKernel.model_validate({**data, "polarEdges": data["polarEdges"][:-1]})
    with pytest.raises(ValidationError, match="one entry per polar bin"):
        PhotonCollapsedConeKernel.model_validate({**data, "A": data["A"][:-1]})
    with pytest.raises(ValidationError, match="do not match"):
        PhotonCollapsedConeKernel.model_validate({**data, "b": np.vstack([data["b"], data["b"]])})


def test_missing_kernel_file():
    """A missing or foreign file is reported."""
    with pytest.raises(FileNotFoundError):
        load_collapsed_cone_kernel("does_not_exist.mat")
