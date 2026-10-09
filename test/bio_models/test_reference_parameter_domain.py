"""Reference photon LQ parameters outside a model's domain must never reach a dij.

Voxels outside every structure carry ``alpha_x == beta_x == 0``; the LET-based models
evaluate to zero there. Reference parameters a model is not defined for are rejected
during dose calculation setup instead of producing NaN/Inf influence matrix entries.
"""

import array_api_strict as xps
import numpy as np
import pytest
from scipy import sparse

from pyRadPlan.bio_models import (
    BioEvaluationContext,
    BioModelEvaluator,
    BiologicalModel,
    Carabe,
    ConstantRBEModel,
    EmptyModel,
    HeliumMairani,
    LinearScaling,
    LQModel,
    MCNamara,
    Wedenberg,
    bio_influence_from_let,
)

ALL_MODELS = [Wedenberg, MCNamara, Carabe, HeliumMairani, LinearScaling]


def _influence(model, alpha_x, beta_x, dose, let, xp=np):
    evaluator = model.evaluator(machine=None, voxel_params={})
    return evaluator.evaluate_influence(
        BioEvaluationContext(
            {
                "alpha_x": xp.asarray(alpha_x),
                "beta_x": xp.asarray(beta_x),
                "physical_dose": xp.asarray(dose),
                "let": xp.asarray(let),
            }
        )
    )


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_outside_structure_voxels_evaluate_to_zero(model_cls):
    """alpha_x == beta_x == 0 means "no reference tissue", not a division by zero."""
    model = model_cls()
    alpha_x = [0.0, 0.1, 0.0]
    beta_x = [0.0, 0.05, 0.0]
    dose = [1.0, 1.0, 2.5]
    let = [3.0, 3.0, 12.0]

    evaluator = model.evaluator(machine=None, voxel_params={})
    result = evaluator.evaluate(
        BioEvaluationContext(
            {
                "alpha_x": np.asarray(alpha_x),
                "beta_x": np.asarray(beta_x),
                "physical_dose": np.asarray(dose),
                "let": np.asarray(let),
            }
        )
    )
    assert np.all(np.isfinite(np.asarray(result["alpha"])))
    assert np.all(np.isfinite(np.asarray(result["beta"])))
    assert np.asarray(result["alpha"])[[0, 2]].tolist() == [0.0, 0.0]
    assert np.asarray(result["beta"])[[0, 2]].tolist() == [0.0, 0.0]

    # masking must leave a voxel that does have reference parameters untouched
    plain_alpha, plain_beta = model.alpha_beta(
        np.asarray([0.1]), np.asarray([0.05]), {"let": np.asarray([3.0])}
    )
    assert np.asarray(result["alpha"])[1] == pytest.approx(float(np.asarray(plain_alpha)[0]))
    assert np.asarray(result["beta"])[1] == pytest.approx(float(np.asarray(plain_beta)[0]))

    influence = _influence(model, alpha_x, beta_x, dose, let)
    for name in ("alpha_dose", "sqrt_beta_dose"):
        values = np.asarray(influence[name])
        assert np.all(np.isfinite(values))
        assert values[[0, 2]].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_outside_structure_voxels_evaluate_to_zero_array_api(model_cls):
    """The masking uses only array-API operations, so it works on any backend."""
    influence = _influence(model_cls(), [0.0, 0.1], [0.0, 0.05], [1.0, 1.0], [3.0, 3.0], xp=xps)
    for name in ("alpha_dose", "sqrt_beta_dose"):
        values = np.asarray(influence[name])
        assert np.all(np.isfinite(values))
        assert values[0] == 0.0


def test_bio_influence_from_let_is_finite_for_zero_reference_parameters():
    """A voxel with dose but without reference parameters yields zero, not NaN."""
    dose = sparse.csc_array(np.asarray([[1.0, 0.0], [2.0, 3.0]]))
    let_dose = sparse.csc_array(np.asarray([[4.0, 0.0], [8.0, 15.0]]))
    alpha_x = np.asarray([0.0, 0.1])
    beta_x = np.asarray([0.0, 0.05])

    model = MCNamara()
    result = bio_influence_from_let(model.evaluator(None, {}), dose, let_dose, alpha_x, beta_x)
    for name in ("alpha_dose", "sqrt_beta_dose"):
        values = result[name].toarray()
        assert np.all(np.isfinite(values))
        assert values[0].tolist() == [0.0, 0.0]

    # the voxel with reference parameters keeps the independently computed model value
    let = np.asarray([8.0 / 2.0, 15.0 / 3.0])
    abr = 0.1 / 0.05
    rbe_max = 0.999064 + 0.35605 * let / abr
    rbe_min = 1.1012 - 0.0038703 * np.sqrt(abr) * let
    assert np.allclose(result["alpha_dose"].toarray()[1], np.asarray([2.0, 3.0]) * rbe_max * 0.1)
    assert np.allclose(
        result["sqrt_beta_dose"].toarray()[1],
        np.asarray([2.0, 3.0]) * np.sqrt(rbe_min**2 * 0.05),
    )


def test_bio_influence_from_let_rejects_non_finite_model_output():
    """A model producing nonfinite values fails loudly instead of poisoning the matrices."""

    class NonFiniteModel(LQModel):
        model = "non_finite_test_model"
        possible_radiation_modes = ("protons",)
        required_quantities = ("physical_dose", "let")

        def evaluator(self, machine, voxel_params):
            return NonFiniteEvaluator(self)

    class NonFiniteEvaluator(BioModelEvaluator):
        @property
        def influence_quantity_names(self):
            return ("alpha_dose", "sqrt_beta_dose")

        def _evaluate_influence(self, context):
            dose = context.require("physical_dose")
            return {
                "alpha_dose": dose * np.nan,
                "sqrt_beta_dose": dose,
            }

    dose = sparse.csc_array(np.asarray([[1.0], [2.0]]))
    with pytest.raises(ValueError, match="non-finite 'alpha_dose'"):
        bio_influence_from_let(
            NonFiniteModel().evaluator(None, {}),
            dose,
            dose,
            np.asarray([0.1, 0.1]),
            np.asarray([0.05, 0.05]),
        )


@pytest.mark.parametrize(
    ("model_cls", "declared"),
    [
        (Wedenberg, ("alpha_x",)),
        (Carabe, ("alpha_x",)),
        (HeliumMairani, ("alpha_x",)),
        (LinearScaling, ("alpha_x",)),
        (MCNamara, ("alpha_x", "beta_x")),
    ],
)
def test_models_declare_their_reference_parameter_domain(model_cls, declared):
    assert model_cls().requires_positive_reference == declared


@pytest.mark.parametrize(
    ("model_cls", "alpha_x", "beta_x", "message"),
    [
        (Wedenberg, [0.0, 0.1], [0.05, 0.05], "only defined for alpha_x > 0"),
        (MCNamara, [0.1, 0.1], [0.0, 0.05], "only defined for beta_x > 0"),
        (LinearScaling, [0.0, 0.1], [0.05, 0.05], "only defined for alpha_x > 0"),
    ],
)
def test_validate_reference_parameters_rejects_out_of_domain_voxels(
    model_cls, alpha_x, beta_x, message
):
    with pytest.raises(ValueError, match=message):
        model_cls().validate_reference_parameters(
            {"alpha_x": np.asarray(alpha_x), "beta_x": np.asarray(beta_x)}
        )


@pytest.mark.parametrize("model_cls", [*ALL_MODELS, EmptyModel, ConstantRBEModel])
@pytest.mark.parametrize(
    ("alpha_x", "beta_x", "name"),
    [
        ([0.1, 0.1], [0.05, -0.05], "beta_x"),
        ([0.1, 0.1], [0.05, np.nan], "beta_x"),
        ([0.1, np.inf], [0.05, 0.05], "alpha_x"),
        ([0.1, np.nan], [0.05, 0.05], "alpha_x"),
        ([0.1, -0.1], [0.05, 0.05], "alpha_x"),
        ([0.1, 0.1], [0.05, np.inf], "beta_x"),
    ],
)
def test_reference_coefficients_must_be_finite_and_non_negative(model_cls, alpha_x, beta_x, name):
    """A reference LQ rate is never negative, infinite or NaN, whatever the model."""
    with pytest.raises(ValueError, match=f"{name} must be finite and non-negative"):
        model_cls().validate_reference_parameters(
            {"alpha_x": np.asarray(alpha_x), "beta_x": np.asarray(beta_x)}
        )


@pytest.mark.parametrize("model_cls", [*ALL_MODELS, EmptyModel, ConstantRBEModel])
def test_valid_reference_coefficients_are_accepted(model_cls):
    """Zero (outside a structure) and a plain positive pair stay valid for every model."""
    model_cls().validate_reference_parameters(
        {"alpha_x": np.asarray([0.0, 0.1]), "beta_x": np.asarray([0.0, 0.05])}
    )


def test_wedenberg_still_rejects_a_non_finite_beta_it_does_not_declare():
    """Wedenberg does not declare beta_x positive, but it still must be a valid rate."""
    model = Wedenberg()
    assert "beta_x" not in model.requires_positive_reference

    with pytest.raises(ValueError, match="beta_x must be finite and non-negative"):
        model.validate_reference_parameters(
            {"alpha_x": np.asarray([0.1]), "beta_x": np.asarray([-0.05])}
        )


def test_infinite_reference_parameter_is_not_accepted_as_positive():
    """`> 0` alone would accept +inf, which evaluates to infinite alpha/beta."""
    with pytest.raises(ValueError, match="alpha_x must be finite and non-negative"):
        MCNamara().validate_reference_parameters(
            {"alpha_x": np.asarray([np.inf]), "beta_x": np.asarray([0.05])}
        )


def test_validate_reference_parameters_accepts_outside_structure_voxels():
    """Voxels without any reference parameters are not out-of-domain."""
    MCNamara().validate_reference_parameters(
        {"alpha_x": np.asarray([[0.0], [0.1]]), "beta_x": np.asarray([[0.0], [0.05]])}
    )


def test_validate_reference_parameters_reports_missing_inputs():
    with pytest.raises(ValueError, match="requires the reference photon parameters"):
        MCNamara().validate_reference_parameters({"alpha_x": np.asarray([0.1])})


def test_model_without_domain_declaration_accepts_anything():
    class NoDomainModel(BiologicalModel):
        model = "no_domain_test_model"
        possible_radiation_modes = ("protons",)

        def evaluator(self, machine, voxel_params):  # pragma: no cover - not used
            raise NotImplementedError

    NoDomainModel().validate_reference_parameters({})
