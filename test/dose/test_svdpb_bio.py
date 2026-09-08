"""Biological model propagation of the photon pencil-beam engine.

The photon engine computes physical dose only, but it must still record which biological
model produced the dij so that a constant RBE keeps working through the common result
assembly, and it must say so when a model declares outputs it cannot compute.
"""

import logging

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan import calc_dose_influence
from pyRadPlan.bio_models import BioModelEvaluator, ConstantRBEModel, EmptyModel, LQModel
from pyRadPlan.quantities._rbe_x_dose import RBExDose


def test_photons_constant_rbe_is_stored_and_applied(test_data_photons):
    """A constant-RBE photon plan yields dij.rbe and rbe * physical_dose."""
    pln, ct, cst, stf, _dij, result = test_data_photons
    pln.bio_model = {"model": "constant_rbe", "rbe": 1.5}

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    assert isinstance(dij_py.bio_model, ConstantRBEModel)
    assert dij_py.rbe == pytest.approx(1.5)
    assert dij_py.model_dump()["bio_model"] == {"model": "constant_rbe", "rbe": 1.5}

    res = dij_py.compute_result_ct_grid(np.asarray(result["w"]))
    phys = sitk.GetArrayFromImage(res["physical_dose"])
    rbe_x = sitk.GetArrayFromImage(res["rbe_x_dose"])
    assert phys.max() > 0.0
    assert np.allclose(rbe_x, 1.5 * phys)


def test_photons_default_model_is_recorded(test_data_photons):
    pln, ct, cst, stf, _dij, _result = test_data_photons

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    assert isinstance(dij_py.bio_model, EmptyModel)
    assert dij_py.rbe is None


class _PhotonLQModel(LQModel):
    """Photon-capable LQ model; no photon engine can compute its alpha/beta."""

    model = "photon_lq_capability_test"
    possible_radiation_modes = ("photons",)
    required_quantities = ("physical_dose",)

    def evaluator(self, machine, voxel_params):  # pragma: no cover - never evaluated
        return BioModelEvaluator(self)


def test_photon_engine_reports_model_outputs_it_cannot_compute(test_data_photons, caplog):
    """An unsupported model is not silently dropped: it is stored and reported."""
    pln, ct, cst, stf, _dij, _result = test_data_photons
    pln.bio_model = _PhotonLQModel()

    with caplog.at_level(logging.WARNING):
        dij_py = calc_dose_influence(ct, cst, stf, pln)

    assert "cannot compute the biological quantities" in caplog.text
    assert dij_py.bio_model == _PhotonLQModel()
    assert dij_py.alpha_dose is None and dij_py.sqrt_beta_dose is None

    with pytest.raises(ValueError, match="needs alpha_dose / sqrt_beta_dose"):
        RBExDose(dij_py)
