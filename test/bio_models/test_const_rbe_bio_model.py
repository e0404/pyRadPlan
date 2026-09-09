import pytest

from pyRadPlan.bio_models import (
    BioEvaluationContext,
    BiologicalModel,
    ConstantRBEModel,
    ParametricEvaluator,
)


def test_ConstantRBEModel_constructor():
    const_rbe_model = ConstantRBEModel()
    assert isinstance(const_rbe_model, BiologicalModel)
    assert const_rbe_model.model == "constant_rbe"
    assert const_rbe_model.required_quantities == ("physical_dose",)
    assert const_rbe_model.possible_radiation_modes == (
        "photons",
        "protons",
        "helium",
        "carbon",
        "oxygen",
        "VHEE",
    )
    assert const_rbe_model.output_quantities == ()
    assert not const_rbe_model.provides("alpha", "beta")
    assert not const_rbe_model.requires("let")
    assert const_rbe_model.rbe == pytest.approx(1.1)


def test_ConstantRBEModel_parameter():
    with pytest.raises(ValueError):
        ConstantRBEModel(rbe=0.0)


def test_ConstantRBEModel_evaluator():
    evaluator = ConstantRBEModel().evaluator(machine=None, voxel_params={})
    assert isinstance(evaluator, ParametricEvaluator)
    assert evaluator.kernel_quantities({}) == {}
    with pytest.raises(NotImplementedError, match="does not provide alpha/beta values"):
        evaluator.evaluate(BioEvaluationContext())
