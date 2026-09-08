import pytest

from pyRadPlan.bio_models import (
    BioEvaluationContext,
    BiologicalModel,
    EmptyModel,
    ParametricEvaluator,
)


def test_EmptyModel_constructor():
    empty_model = EmptyModel()
    assert isinstance(empty_model, BiologicalModel)
    assert empty_model.model == "none"
    assert empty_model.required_quantities == ()
    assert empty_model.possible_radiation_modes == (
        "photons",
        "protons",
        "helium",
        "carbon",
        "oxygen",
        "VHEE",
    )
    assert empty_model.output_quantities == ()
    assert not empty_model.provides("alpha", "beta")
    assert not empty_model.requires("let")
    assert empty_model.default_report_quantity == "physical_dose"


def test_EmptyModel_evaluator():
    evaluator = EmptyModel().evaluator(machine=None, voxel_params={})
    assert isinstance(evaluator, ParametricEvaluator)
    assert evaluator.kernel_quantities({"alpha": 1}) == {}
    with pytest.raises(NotImplementedError, match="does not provide alpha/beta values"):
        evaluator.evaluate(BioEvaluationContext())
