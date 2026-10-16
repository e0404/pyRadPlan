"""Physical- and biological-dose workflows with the default biological model of each ion.

The plan defaults to a biological model per ion (constant RBE for protons, HEL for helium,
LEM kernels for carbon / oxygen). A physical-dose workflow has to keep working when that
model cannot be applied to the patient's tissues; only an explicit request for biological
dose (``calc_bio_dose=True``) turns that into an error.
"""

import logging

import numpy as np
import pytest

from pyRadPlan import calc_dose_influence, fluence_optimization
from pyRadPlan.biology.models import BioModelUnavailableError, create_bio_model
from pyRadPlan.plan import default_bio_models
from pyRadPlan.biology.tissues import (
    LEMTissueProperties,
    MKMTissueProperties,
    TissueProperties,
)

ION_MODES = ("protons", "helium", "carbon", "oxygen")
KERNEL_MODES = ("carbon", "oxygen")

# Matches no tissue class of the generic LEM base data (alpha_x = 0.1 / 0.5).
UNMATCHED_TISSUE = TissueProperties(alpha_x=0.2, beta_x=0.05)


def _case(request, mode):
    pln, ct, cst, stf, _dij, _result = request.getfixturevalue(f"test_data_{mode}")
    pln.bio_model = default_bio_models[mode]
    pln.prop_opt = {"solver": "scipy", "max_iter": 5}
    return pln, ct, cst, stf


def _set_tissue(cst, tissue):
    for i, voi in enumerate(cst.vois):
        cst.vois[i] = voi.model_copy(
            update={
                "tissue_properties": tissue,
                "alpha_x": tissue.alpha_x,
                "beta_x": tissue.beta_x,
            }
        )


def _optimized_result(ct, cst, stf, dij, pln):
    w = fluence_optimization(ct, cst, stf, dij, pln)
    assert np.all(np.isfinite(w))
    return dij.get_result_arrays_from_intensity(w)


@pytest.mark.parametrize("mode", ION_MODES)
def test_default_bio_model_with_matching_tissue(request, mode, caplog):
    """Physical-dose objectives are optimized on RBExDose, which the dij provides."""
    pln, ct, cst, stf = _case(request, mode)

    dij = calc_dose_influence(ct, cst, stf, pln)
    assert dij.bio_model == create_bio_model(default_bio_models[mode])

    with caplog.at_level(logging.WARNING):
        result = _optimized_result(ct, cst, stf, dij, pln)

    assert "converted to 'rbe_x_dose'" in caplog.text
    assert np.all(np.isfinite(result["rbe_x_dose"]))
    assert np.max(result["rbe_x_dose"]) > 0
    if mode == "protons":
        assert dij.rbe == pytest.approx(1.1)
        assert np.allclose(result["rbe_x_dose"], 1.1 * result["physical_dose"])
    else:
        assert np.all(np.isfinite(result["effect"]))


@pytest.mark.parametrize("mode", ION_MODES)
def test_default_bio_model_with_unmatched_tissue_keeps_physical_workflow(request, mode, caplog):
    """Tissue without base data only drops the kernel-based biology, with a warning."""
    pln, ct, cst, stf = _case(request, mode)
    _set_tissue(cst, UNMATCHED_TISSUE)

    with caplog.at_level(logging.WARNING):
        dij = calc_dose_influence(ct, cst, stf, pln)
        result = _optimized_result(ct, cst, stf, dij, pln)

    assert np.max(result["physical_dose"]) > 0
    if mode in KERNEL_MODES:
        assert "No matching tissue class" in caplog.text
        assert "RBExDose" in caplog.text
        assert dij.bio_model is None
        assert dij.alpha_dose is None and dij.sqrt_beta_dose is None
        assert "rbe_x_dose" not in result
        # physical-dose objectives must not be converted to an unavailable quantity
        assert "converted to 'rbe_x_dose'" not in caplog.text
    else:
        # constant RBE / HEL do not depend on tissue classes of the base data
        assert dij.bio_model == create_bio_model(default_bio_models[mode])
        assert np.all(np.isfinite(result["rbe_x_dose"]))


@pytest.mark.parametrize("mode", KERNEL_MODES)
def test_unmatched_tissue_with_physical_dose_only(request, mode):
    """The physical-dose influence of an unmatched tissue works as before (calc_bio_dose=False)."""
    pln, ct, cst, stf = _case(request, mode)
    pln.prop_dose_calc["calc_bio_dose"] = False
    _set_tissue(cst, UNMATCHED_TISSUE)

    dij = calc_dose_influence(ct, cst, stf, pln)

    assert dij.bio_model is None
    assert dij.quantities == ["physical_dose"]


@pytest.mark.parametrize("mode", KERNEL_MODES)
def test_unmatched_tissue_with_requested_bio_dose_raises(request, mode):
    """Explicitly requested biological dose fails for a tissue without base data."""
    pln, ct, cst, stf = _case(request, mode)
    pln.prop_dose_calc["calc_bio_dose"] = True
    _set_tissue(cst, UNMATCHED_TISSUE)

    with pytest.raises(BioModelUnavailableError, match="No matching tissue class"):
        calc_dose_influence(ct, cst, stf, pln)


@pytest.mark.parametrize("alpha_x", [0.1, 0.5])
def test_oxygen_lem_dose_calculation(test_data_oxygen, alpha_x):
    """Both LEM classes of the generic oxygen machine yield finite biological results."""
    pln, ct, cst, stf, _dij, _result = test_data_oxygen
    pln.bio_model = "LEM"
    _set_tissue(cst, LEMTissueProperties(alpha_x=alpha_x, beta_x=0.05, d_t=30.0, r_nucleus=5.0))

    dij = calc_dose_influence(ct, cst, stf, pln)
    result = dij.get_result_arrays_from_intensity(np.ones(dij.total_num_of_bixels))

    for name in ("effect", "rbe_x_dose", "alpha", "beta"):
        assert np.all(np.isfinite(result[name])), name
    assert np.max(result["rbe_x_dose"]) > 0
    assert np.max(result["effect"]) > 0


def test_oxygen_mkm_dose_calculation(test_data_oxygen):
    """z* kernels: alpha/beta from the oxygen machine's mMKM z* and the tissue's alpha_0."""
    pln, ct, cst, stf, _dij, _result = test_data_oxygen
    pln.bio_model = "MKM"
    _set_tissue(
        cst,
        MKMTissueProperties(
            model="mMKM", r_nucleus=3.9, r_domain=0.32, alpha_0=0.172, beta_0=0.0615
        ),
    )

    dij = calc_dose_influence(ct, cst, stf, pln)
    assert dij.bio_model == create_bio_model("MKM")
    assert dij.alpha_dose is not None and dij.sqrt_beta_dose is not None

    result = dij.get_result_arrays_from_intensity(np.ones(dij.total_num_of_bixels))
    for name in ("effect", "rbe_x_dose", "alpha", "beta"):
        assert np.all(np.isfinite(result[name])), name
    assert np.max(result["rbe_x_dose"]) > 0
    # beta = beta_0 wherever dose is deposited
    dosed = result["physical_dose"] > 0
    assert np.allclose(result["beta"][dosed], 0.0615)


@pytest.mark.parametrize("calc_bio_dose", [True, "auto"])
def test_oxygen_mkm_rejects_unset_alpha_0(test_data_oxygen, calc_bio_dose):
    """A tissue without alpha_0 must not yield NaN biological quantities."""
    pln, ct, cst, stf, _dij, _result = test_data_oxygen
    pln.bio_model = "MKM"
    pln.prop_dose_calc["calc_bio_dose"] = calc_bio_dose
    _set_tissue(
        cst, MKMTissueProperties(model="mMKM", r_nucleus=3.9, r_domain=0.32, beta_0=0.0615)
    )

    if calc_bio_dose is True:
        with pytest.raises(BioModelUnavailableError, match="lacks alpha_0"):
            calc_dose_influence(ct, cst, stf, pln)
        return

    dij = calc_dose_influence(ct, cst, stf, pln)
    assert dij.bio_model is None
    assert dij.alpha_dose is None
