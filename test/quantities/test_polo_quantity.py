import pytest
import numpy as np
import array_api_strict as xp

import SimpleITK as sitk

from pyRadPlan.ct import create_ct
from pyRadPlan.cst import StructureSet, create_voi
from pyRadPlan.dij import Dij
from pyRadPlan.optimization.objectives import HellingerConvPOLO, LogSumExpConvPOLO
from pyRadPlan.quantities import POLO, LinearPOLO, QuantityResolver, total_ntcp


@pytest.fixture
def sample_base_dij_dict():
    dij_dict = {
        "ct_grid": {
            "resolution": {"x": 1.5, "y": 1.5, "z": 1.5},
            "dimensions": (10, 10, 10),
            "num_of_voxels": 1000,
        },
        "dose_grid": {
            "resolution": {"x": 3.0, "y": 3.0, "z": 3.0},
            "dimensions": (5, 5, 5),
            "num_of_voxels": 125,
        },
        "num_of_beams": 1,
        "total_num_of_bixels": 10,
        "physical_dose": np.empty((1, 1, 1), dtype=object),
        "let_dose": np.empty((1, 1, 1), dtype=object),
        "bixel_num": np.arange(10),
        "ray_num": np.arange(10),
        "beam_num": np.zeros((10,), dtype=np.int64),
    }
    return dij_dict


@pytest.fixture
def sample_dij_dense(sample_base_dij_dict):
    sample_base_dij_dict["physical_dose"].flat[0] = np.ones((125, 10), dtype=np.float32)
    sample_base_dij_dict["let_dose"].flat[0] = np.ones((125, 10), dtype=np.float32)
    dij = Dij.model_validate(sample_base_dij_dict)
    return dij


def test_polo_quantity_initialization_and_compute(sample_dij_dense):
    polo = POLO(sample_dij_dense, num_of_fractions=1, risk_voi_name=None)

    risk_mask = np.zeros(125, dtype=bool)
    risk_mask[:10] = True
    polo.binary_risk_mask = polo.array_backend.asarray(risk_mask, dtype=polo._dtype)

    assert polo.identifier == "polo"
    # Dose grid resolution (3, 3, 3) -> v_new = 27 mm^3. k = 27 / 1.08 = 25.0
    assert np.isclose(polo.k, 27.0 / 1.08)

    fluence = xp.arange(10, dtype=xp.float32)
    p_val = polo.compute(fluence)
    assert p_val.flat[0].shape[0] == 125


def test_linear_polo_quantity_compute(sample_dij_dense):
    lin_polo = LinearPOLO(sample_dij_dense, num_of_fractions=1, risk_voi_name=None)

    risk_mask = np.zeros(125, dtype=bool)
    risk_mask[:10] = True
    lin_polo.binary_risk_mask = lin_polo.array_backend.asarray(risk_mask, dtype=lin_polo._dtype)

    assert lin_polo.identifier == "linear_polo"

    fluence = xp.arange(10, dtype=xp.float32)
    eta = lin_polo.compute(fluence)
    assert eta.flat[0].shape[0] == 125


@pytest.fixture
def risk_cst():
    """A StructureSet on the dij's ct grid carrying a 'Ventricles_4mm' risk structure."""
    # Geometry must match the dij's ct_grid (10^3 at 1.5 mm, origin 0) so that the risk mask
    # still lands on the dose grid (5^3 at 3.0 mm, origin 0) after resampling.
    cube = sitk.GetImageFromArray(np.zeros((10, 10, 10), dtype=np.float32))
    cube.SetSpacing((1.5, 1.5, 1.5))
    cube.SetOrigin((0.0, 0.0, 0.0))
    ct = create_ct(cube_hu=cube)

    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[:4, :, :] = 1  # lower quarter along z, so it survives resampling to the 5^3 dose grid
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(cube)

    voi = create_voi(name="Ventricles_4mm", voi_type="OAR", mask=mask_image, ct_image=ct)
    return ct, StructureSet(ct_image=ct, vois=[voi])


def test_polo_quantity_cst_lookup(sample_dij_dense, risk_cst):
    _ct, cst = risk_cst
    polo = POLO(sample_dij_dense, num_of_fractions=1, cst=cst, risk_voi_name="Ventricles_4mm")

    assert polo.risk_voi_name == "Ventricles_4mm"

    risk = np.asarray(polo.binary_risk_mask)
    assert risk.shape == (125,)
    assert set(np.unique(risk)) <= {0.0, 1.0}
    # The structure covers part of the volume, so the resampled mask must be neither empty
    # nor everything -- an empty risk mask would silently drop the beta_3 term.
    assert 0 < risk.sum() < risk.size


def test_polo_quantity_missing_risk_voi_raises(sample_dij_dense, risk_cst):
    _ct, cst = risk_cst
    with pytest.raises(ValueError, match="Absent"):
        POLO(sample_dij_dense, num_of_fractions=1, cst=cst, risk_voi_name="Absent")


def test_polo_rejects_4d_masks(sample_dij_dense):
    # One risk mask serves all CT phases; a 4D mask cannot be resampled onto the 3D dose grid.
    phase = sitk.Image(10, 10, 10, sitk.sitkUInt8)
    phase.SetSpacing((1.5, 1.5, 1.5))
    with pytest.raises(ValueError, match="4D structure sets"):
        POLO(sample_dij_dense, num_of_fractions=1, binary_risk_mask=sitk.JoinSeries([phase] * 2))


def test_polo_dose_threshold_zeroes_below_domain(sample_dij_dense):
    # Bahn et al. fitted on voxels receiving at least 2 Gy; below that POLO is undefined.
    fluence = xp.ones(10, dtype=xp.float32)  # dij is all ones -> dose 10 Gy everywhere

    above = np.asarray(
        POLO(sample_dij_dense, num_of_fractions=1, risk_voi_name=None).compute(fluence).flat[0]
    )
    below = np.asarray(
        POLO(sample_dij_dense, num_of_fractions=1, risk_voi_name=None, dose_threshold=1e3)
        .compute(fluence)
        .flat[0]
    )

    assert np.all(above > 0.0)
    assert np.all(below == 0.0)


def test_linear_polo_marks_below_threshold_as_neg_inf(sample_dij_dense):
    # -inf is the sentinel the ConvPOLO objectives filter on; a finite 0 would mean p = 0.5.
    fluence = xp.ones(10, dtype=xp.float32)

    eta = np.asarray(
        LinearPOLO(sample_dij_dense, num_of_fractions=1, risk_voi_name=None, dose_threshold=1e3)
        .compute(fluence)
        .flat[0]
    )
    assert np.all(np.isneginf(eta))

    eta_in_domain = np.asarray(
        LinearPOLO(sample_dij_dense, num_of_fractions=1, risk_voi_name=None)
        .compute(fluence)
        .flat[0]
    )
    assert np.all(np.isfinite(eta_in_domain))


@pytest.mark.parametrize("objective_cls", [LogSumExpConvPOLO, HellingerConvPOLO])
def test_eta_objectives_are_continuous_across_2_gy(sample_dij_dense, objective_cls):
    # The objectives on eta drop the 2 Gy filter of Ortkamp et al. by default, because with it
    # a voxel jumps from "excluded" to eta of about -26 when its dose crosses the threshold.
    objective = objective_cls()

    def value(dose):
        quantity = LinearPOLO(
            sample_dij_dense,
            num_of_fractions=1,
            risk_voi_name=None,
            dose_threshold=objective.dose_threshold,
        )
        # dij is all ones over 10 bixels: a uniform fluence w gives 10 * w Gy everywhere.
        fluence = xp.full((10,), dose / 10.0, dtype=xp.float32)
        return float(objective.compute_objective(np.asarray(quantity.compute(fluence).flat[0])))

    # eta rises by 0.19 + 0.018 per Gy in each of the 125 voxels, so 2 mGy move the
    # log-sum-exp by 4e-4 and the sum by 0.05.
    assert abs(value(2.001) - value(1.999)) < 0.1


def test_total_ntcp_keeps_its_precision_in_float32():
    # 1 - exp(sum(log1p(-p))) cancels in float32: for an NTCP of 1e-6 it was 1.3 % off.
    p = np.full(1000, 1e-9, dtype=np.float32)
    exact = -np.expm1(np.sum(np.log1p(-p.astype(np.float64))))
    assert np.isclose(float(total_ntcp(p)), exact, rtol=1e-5)


def test_polo_derivative_without_forward_compute(sample_dij_dense):
    # The derivative path must not depend on a prior compute() having populated _w_cache.
    polo = POLO(sample_dij_dense, num_of_fractions=1, risk_voi_name=None)

    fluence = xp.ones(10, dtype=xp.float32)
    d_q = xp.ones((1, 125), dtype=xp.float32)
    deriv = np.asarray(polo.compute_chain_derivative(d_q, fluence).flat[0])

    assert np.all(np.isfinite(deriv))


def test_polo_derivative_uses_current_fluence(sample_dij_dense):
    # The chain derivative must be evaluated at _w_grad_cache, not at whatever fluence the
    # last forward pass happened to leave in _w_cache.
    polo = POLO(sample_dij_dense, num_of_fractions=1, risk_voi_name=None)
    d_q = xp.ones((1, 125), dtype=xp.float32)

    grad_fluence = xp.asarray(np.arange(1, 11, dtype=np.float32))
    expected = np.asarray(polo.compute_chain_derivative(d_q, grad_fluence).flat[0])

    stale = POLO(sample_dij_dense, num_of_fractions=1, risk_voi_name=None)
    stale.compute(xp.ones(10, dtype=xp.float32))  # different fluence, forward cache only
    actual = np.asarray(stale.compute_chain_derivative(d_q, grad_fluence).flat[0])

    assert np.allclose(actual, expected)


@pytest.mark.parametrize("cls", [POLO, LinearPOLO])
def test_polo_evaluates_course_dose(sample_dij_dense, cls):
    # The dij holds per-fraction dose and the model was fitted on course dose, so n fractions
    # at fluence w must equal one fraction at n * w, with the gradient scaled by n.
    n = 3
    fluence = xp.full(10, 4.0, dtype=xp.float32)
    d_q = xp.ones((1, 125), dtype=xp.float32)
    per_fx = cls(sample_dij_dense, num_of_fractions=n, risk_voi_name=None)
    course = cls(sample_dij_dense, num_of_fractions=1, risk_voi_name=None)

    assert np.allclose(
        np.asarray(per_fx.compute(fluence).flat[0]),
        np.asarray(course.compute(n * fluence).flat[0]),
    )
    assert np.allclose(
        np.asarray(per_fx.compute_chain_derivative(d_q, fluence).flat[0]),
        n * np.asarray(course.compute_chain_derivative(d_q, n * fluence).flat[0]),
    )


def test_resolver_injects_num_of_fractions(sample_dij_dense):
    resolver = QuantityResolver(
        sample_dij_dense,
        context={"num_of_fractions": 7},
        quantity_kwargs={"polo": {"risk_voi_name": None}},
    )
    assert resolver.get("polo").num_of_fractions == 7
