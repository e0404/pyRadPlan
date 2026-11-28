"""Collapsed cone kernel: file data, refit into polar bins and energy check."""

import numpy as np

from pyRadPlan.dose.engines import _collapsed_cone as cc


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
    kernel = cc.fit_collapsed_cone_kernel(edk, 12)
    assert kernel.num_polar_angles == 12
    np.testing.assert_allclose(kernel.polar_edges, np.arange(0, 181, 15))
    np.testing.assert_allclose(kernel.A.sum(), edk["primary"].sum(), rtol=1e-12)
    np.testing.assert_allclose(kernel.B.sum(), edk["scatter"].sum(), rtol=1e-12)
    np.testing.assert_allclose(kernel.a, 0.3, rtol=1e-4)
    np.testing.assert_allclose(kernel.b, 0.02, rtol=1e-3)
    assert np.all(kernel.residual["primaryCum"] < 1e-4)

    kernel = cc.fit_collapsed_cone_kernel(edk, 12, 1.5, 1, 200)
    assert kernel.min_radius == 1.5
    assert kernel.scatter_max_radius == 200
    np.testing.assert_allclose(kernel.b, 0.02, rtol=1e-3)


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
    kernel = cc.fit_collapsed_cone_kernel(edk, 12, 0, 2)
    assert kernel.B.shape == (2, 12)
    np.testing.assert_allclose(kernel.B.sum(), edk["scatter"].sum(), rtol=1e-10)
    np.testing.assert_allclose(kernel.b[0], 0.1, rtol=0.02)


def test_default_kernel_data():
    """The 6 MV kernel is valid and its energy does not change with refitting."""
    kernel, meta = cc.load_kernel(None, 12)
    assert kernel.A.shape == (1, 12)
    assert np.all((kernel.A >= 0) & (kernel.B >= 0) & (kernel.a > 0) & (kernel.b > 0))
    total = kernel.A.sum() + kernel.B.sum()
    assert 0.9 < total < 1
    assert meta["nominalEnergy"] == 6
    kernel6, _ = cc.load_kernel(None, 6)
    np.testing.assert_allclose(kernel6.A.sum() + kernel6.B.sum(), total, rtol=1e-10)
    kernel2, _ = cc.load_kernel(None, 12, 2)
    assert kernel2.B.shape == (2, 12)
    np.testing.assert_allclose(kernel2.A.sum() + kernel2.B.sum(), total, rtol=1e-10)


def test_check_kernel_energy():
    """Matching, mismatching and unknown kernel energies."""
    assert cc.check_kernel_energy({"nominalEnergy": 6.0}, 6.0) == (True, "")
    ok, msg = cc.check_kernel_energy({"nominalEnergy": 6.0}, 18.0)
    assert not ok and "18" in msg
    ok, msg = cc.check_kernel_energy({}, 6.0)
    assert ok and msg
