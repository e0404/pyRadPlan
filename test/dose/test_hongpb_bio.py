"""Biological output of the Hong pencil-beam engine.

The matRad reference data carries no biological quantities, so the alpha_dose /
sqrt_beta_dose influence matrices are verified analytically per bixel against the
physical-dose and LET-dose matrices produced in the same run.
"""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

import pyRadPlan
from pyRadPlan import calc_dose_influence
from pyRadPlan.machines import load_machine_from_mat, validate_machine


def _per_entry(dij):
    """Physical dose, alpha, sqrt(beta) and LET per non-zero (voxel, bixel) entry."""
    dose = dij.physical_dose.flat[0].tocoo()
    rows, cols, d = dose.row, dose.col, dose.data
    keep = d > 0
    rows, cols, d = rows[keep], cols[keep], d[keep]

    def pick(container):
        return np.asarray(container.flat[0].tocsr()[rows, cols]).ravel()

    return {
        "rows": rows,
        "dose": d,
        "alpha": pick(dij.alpha_dose) / d,
        "sqrt_beta": pick(dij.sqrt_beta_dose) / d,
        "let": pick(dij.let_dose) / d if dij.let_dose is not None else None,
        "alpha_x": dij.alphax[rows, 0],
        "beta_x": dij.betax[rows, 0],
    }


def _assert_rbe_min_max_model(dij, rbe_min_max):
    e = _per_entry(dij)
    assert e["alpha"].size > 0
    assert np.all(e["beta_x"] > 0), "dose deposited outside any structure"
    abr = e["alpha_x"] / e["beta_x"]
    rbe_min, rbe_max = rbe_min_max(e["let"], e["alpha_x"], abr)
    assert np.allclose(e["alpha"], rbe_max * e["alpha_x"], rtol=1e-5)
    assert np.allclose(e["sqrt_beta"], np.sqrt(rbe_min**2 * e["beta_x"]), rtol=1e-5)


def test_protons_WED_alpha_beta_matrices(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "WED"
    dij_py = calc_dose_influence(ct, cst, stf, pln)

    def wedenberg(let, alpha_x, abr):
        return 1.0, 1.0 + 0.434 * let / abr

    _assert_rbe_min_max_model(dij_py, wedenberg)


def test_protons_MCN_alpha_beta_matrices(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "MCN"
    dij_py = calc_dose_influence(ct, cst, stf, pln)

    def mcnamara(let, alpha_x, abr):
        rbe_max = 0.999064 + 0.35605 * let / abr
        rbe_min = 1.1012 - 0.0038703 * np.sqrt(abr) * let
        return rbe_min, rbe_max

    _assert_rbe_min_max_model(dij_py, mcnamara)


def test_helium_HEL_alpha_beta_matrices(test_data_helium):
    pln, ct, cst, stf, dij, result = test_data_helium
    pln.bio_model = "HEL"
    dij_py = calc_dose_influence(ct, cst, stf, pln)

    def mairani(let, alpha_x, abr):
        f_qe = 9.73154e-3 * let**2 * np.exp(-1.51998e-2 * let)
        return 1.0, 1.0 + (1.36938e-1 + 1.0 / abr) * f_qe

    _assert_rbe_min_max_model(dij_py, mairani)


def test_protons_constant_rbe(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "constant_rbe"
    dij_py = calc_dose_influence(ct, cst, stf, pln)

    assert dij_py.rbe == pytest.approx(1.1)
    assert dij_py.alpha_dose is None
    assert dij_py.sqrt_beta_dose is None

    res = dij_py.compute_result_ct_grid(np.asarray(result["w"]))
    phys = sitk.GetArrayFromImage(res["physical_dose"])
    rbe_x = sitk.GetArrayFromImage(res["rbe_x_dose"])
    assert np.allclose(rbe_x, 1.1 * phys)


def test_protons_bio_model_none_has_no_bio_matrices(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "none"
    dij_py = calc_dose_influence(ct, cst, stf, pln)
    assert dij_py.alpha_dose is None
    assert dij_py.sqrt_beta_dose is None
    assert dij_py.rbe is None


def test_carbon_kernel_based_lq_alpha_beta_matrices(test_data_carbon):
    pln, ct, cst, stf, dij, result = test_data_carbon
    pln.bio_model = "kernel_based_lq"
    dij_py = calc_dose_influence(ct, cst, stf, pln)

    e = _per_entry(dij_py)
    assert e["alpha"].size > 0
    assert np.all(np.isfinite(e["alpha"])) and np.all(np.isfinite(e["sqrt_beta"]))

    # every voxel's alpha/beta must be an interpolated value of the kernel of its tissue class
    machine_file = Path(pyRadPlan.__file__).parent / "data" / "machines" / "carbon_Generic.mat"
    machine = validate_machine(load_machine_from_mat(machine_file))
    kernel = machine.pb_kernels[machine.energies[0]]
    tissue = np.flatnonzero(
        (kernel.alpha_x == np.unique(e["alpha_x"])) & (kernel.beta_x == np.unique(e["beta_x"]))
    )
    assert tissue.size == 1, "test data is expected to contain a single tissue class"
    alphas = np.concatenate([k.alpha[tissue[0]] for k in machine.pb_kernels.values()])
    betas = np.concatenate([k.beta[tissue[0]] for k in machine.pb_kernels.values()])
    assert np.all(e["alpha"] >= alphas.min() - 1e-9) and np.all(e["alpha"] <= alphas.max() + 1e-9)
    assert np.all(e["sqrt_beta"] ** 2 >= betas.min() - 1e-9)
    assert np.all(e["sqrt_beta"] ** 2 <= betas.max() + 1e-9)

    # RBE-weighted dose from the result must be consistent with the LQ inversion of the effect
    w = np.asarray(result["w"])
    res = dij_py.compute_result_dose_grid(w)
    effect = dij_py.alpha_dose.flat[0] @ w + (dij_py.sqrt_beta_dose.flat[0] @ w) ** 2
    ax, bx = dij_py.alphax[:, 0], dij_py.betax[:, 0]
    valid = (bx > 0) & (effect > 0)
    expected = np.zeros_like(effect)
    expected[valid] = (np.sqrt(ax[valid] ** 2 + 4 * bx[valid] * effect[valid]) - ax[valid]) / (
        2 * bx[valid]
    )
    assert np.allclose(np.asarray(res["rbe_x_dose"]).ravel(), expected, rtol=1e-5, atol=1e-8)
