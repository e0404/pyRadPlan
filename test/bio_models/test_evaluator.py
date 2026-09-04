"""Tests for the generic biological evaluator contract."""

from collections.abc import Mapping

import array_api_strict as xp
import numpy as np
import pytest

from pyRadPlan.bio_models import (
    BioEvaluationContext,
    BioModelEvaluator,
    BioModelResult,
    ConstantRBEModel,
    Wedenberg,
)


def test_evaluation_context_is_a_read_only_named_mapping():
    """Context snapshots its source and only exposes read-only named inputs."""
    source = {"let": xp.asarray([1.0, 2.0])}
    context = BioEvaluationContext(source)
    source["late"] = "not copied"

    assert isinstance(context, Mapping)
    assert tuple(context) == ("let",)
    assert context.require("let") is context["let"]
    assert "late" not in context
    with pytest.raises(TypeError):
        context.inputs["other"] = 1
    with pytest.raises(ValueError, match="requires input 'physical_dose'.*let"):
        context.require("physical_dose")


def test_context_uses_the_documented_particle_input_vocabulary():
    """The public context uses canonical biological names, not engine-local names."""
    context = BioEvaluationContext(
        {
            "alpha_x": "alpha_x values",
            "beta_x": "beta_x values",
            "physical_dose": "calculated dose",
            "let": "LET values",
        }
    )

    assert context["alpha_x"] == "alpha_x values"
    assert context["beta_x"] == "beta_x values"
    assert context["let"] == "LET values"
    assert context["physical_dose"] == "calculated dose"
    assert "v_alpha_x" not in context
    assert "kernel" not in context


def test_model_result_is_a_read_only_extensible_mapping():
    """Results support arbitrary named quantities without permitting mutation."""
    source = {"rbe": xp.asarray([1.1, 1.2]), "survival": xp.asarray([0.8, 0.7])}
    result = BioModelResult(source)
    source["effect"] = "not copied"

    assert isinstance(result, Mapping)
    assert set(result) == {"rbe", "survival"}
    assert result.require("rbe") is result["rbe"]
    assert "effect" not in result
    with pytest.raises(TypeError):
        result.quantities["effect"] = 1
    with pytest.raises(ValueError, match="did not produce quantity 'alpha'.*rbe, survival"):
        result.require("alpha")


def test_parametric_evaluator_returns_named_result():
    """Parametric LQ evaluation produces alpha and beta as named quantities."""
    evaluator = Wedenberg().evaluator(machine=None, voxel_params={})
    assert evaluator.kernel_field_names == ()
    result = evaluator.evaluate(
        BioEvaluationContext(
            {
                "alpha_x": xp.asarray([0.1, 0.1]),
                "beta_x": xp.asarray([0.05, 0.05]),
                "physical_dose": xp.asarray([1.0, 1.0]),
                "let": xp.asarray([0.0, 2.0]),
            }
        )
    )

    assert isinstance(result, BioModelResult)
    assert set(result) == {"alpha", "beta"}
    assert np.allclose(np.asarray(result["alpha"]), [0.1, 0.1434])
    assert np.allclose(np.asarray(result["beta"]), [0.05, 0.05])


def test_generic_result_does_not_require_lq_outputs():
    """The general evaluator contract can represent non-LQ endpoints."""

    class DirectRBEModel(ConstantRBEModel):
        output_quantities = ("rbe",)

    class DirectRBEEvaluator(BioModelEvaluator):
        def evaluate(self, context):
            return BioModelResult({"rbe": context.require("rbe")})

    model = DirectRBEModel()
    evaluator = DirectRBEEvaluator(model)
    result = evaluator.evaluate(BioEvaluationContext({"rbe": xp.asarray([1.25])}))

    assert model.provides("rbe")
    assert not model.provides("alpha", "beta")
    with pytest.raises(TypeError):
        model.provides()
    assert set(result) == {"rbe"}
    assert np.allclose(np.asarray(result["rbe"]), [1.25])


def test_non_lq_parametric_evaluator_has_specific_failure_message():
    """The common unsupported alpha/beta path retains its specific diagnostic."""
    evaluator = ConstantRBEModel().evaluator(machine=None, voxel_params={})

    with pytest.raises(NotImplementedError, match="does not provide alpha/beta values"):
        evaluator.evaluate(BioEvaluationContext())
