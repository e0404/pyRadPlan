"""Public extension contract for externally defined biological models."""

from types import SimpleNamespace
from typing import ClassVar

import array_api_compat
import array_api_strict as xp
import numpy as np
import pytest

from pyRadPlan import calc_dose_influence
from pyRadPlan.bio_models import (
    BioEvaluationContext,
    BiologicalModel,
    BioModelEvaluator,
    BioModelResult,
    create_bio_model,
    get_available_models,
    register_model,
)
from pyRadPlan.bio_models._factory import BIO_MODELS
from pyRadPlan.cst import validate_cst
from pyRadPlan.ct import validate_ct
from pyRadPlan.dose.engines import DoseEngineBase
from pyRadPlan.plan import validate_pln
from pyRadPlan.stf import validate_stf


class _LinearLETEvaluator(BioModelEvaluator):
    @property
    def influence_quantity_names(self) -> tuple[str, ...]:
        return ("alpha_dose", "sqrt_beta_dose")

    def _evaluate(self, context: BioEvaluationContext) -> BioModelResult:
        alpha_x = context.require("alpha_x")
        beta_x = context.require("beta_x")
        let = context.require("let")
        return BioModelResult(
            {
                "alpha": alpha_x * (1.0 + self.model.slope * let),
                "beta": beta_x,
            }
        )

    def _evaluate_influence(self, context: BioEvaluationContext) -> dict[str, object]:
        result = self.evaluate(context)
        physical_dose = context.require("physical_dose")
        array_namespace = array_api_compat.array_namespace(physical_dose)
        return {
            "alpha_dose": physical_dose * result.require("alpha"),
            "sqrt_beta_dose": physical_dose * array_namespace.sqrt(result.require("beta")),
        }


class _LinearLETModel(BiologicalModel):
    model = "extension_test_linear_let"
    model_aliases = ("extensionTestLET",)
    required_quantities = ("physical_dose", "let")
    output_quantities = ("alpha", "beta")
    possible_radiation_modes = ("protons",)

    def __init__(self, slope: float = 0.02):
        self.slope = slope

    def evaluator(self, machine, voxel_params) -> BioModelEvaluator:
        return _LinearLETEvaluator(self)


@pytest.fixture
def registered_custom_model():
    """Register the external-style test model without leaking global registry state."""
    registered = register_model(_LinearLETModel)
    try:
        yield registered
    finally:
        for name in (_LinearLETModel.model, *_LinearLETModel.model_aliases):
            if BIO_MODELS.get(name) is _LinearLETModel:
                BIO_MODELS.pop(name)


def test_custom_model_registration_factory_and_availability(registered_custom_model):
    """Registration supports decorator use, aliases, parameters and machine filtering."""
    assert registered_custom_model is _LinearLETModel
    assert register_model(_LinearLETModel) is _LinearLETModel

    model = create_bio_model(
        {"model": "extensionTestLET", "slope": 0.05}, radiation_mode="protons"
    )

    assert isinstance(model, _LinearLETModel)
    assert model.to_dict() == {"model": "extension_test_linear_let", "slope": 0.05}
    assert "extension_test_linear_let" not in get_available_models("protons", ["physical_dose"])
    assert "extension_test_linear_let" in get_available_models("protons", ["physical_dose", "let"])


def test_register_model_is_a_class_decorator():
    """The returned class can be instantiated normally after decorator registration."""

    @register_model
    class DecoratedModel(_LinearLETModel):
        model = "extension_test_decorated"
        model_aliases = ()

    try:
        assert isinstance(create_bio_model("extension_test_decorated"), DecoratedModel)
    finally:
        if BIO_MODELS.get(DecoratedModel.model) is DecoratedModel:
            BIO_MODELS.pop(DecoratedModel.model)


def test_custom_evaluator_contract(registered_custom_model):
    """A custom evaluator consumes named inputs and returns its declared quantities."""
    model = create_bio_model({"model": registered_custom_model.model, "slope": 0.1})
    evaluator = model.evaluator(machine=None, voxel_params={})
    evaluator.validate_declarations()
    context = BioEvaluationContext(
        {
            "alpha_x": xp.asarray([0.1, 0.2]),
            "beta_x": xp.asarray([0.05, 0.04]),
            "physical_dose": xp.asarray([2.0, 3.0]),
            "let": xp.asarray([1.0, 2.0]),
        }
    )

    result = evaluator.evaluate(context)
    influence = evaluator.evaluate_influence(context)

    assert tuple(result) == model.output_quantities
    assert tuple(influence) == evaluator.influence_quantity_names
    assert np.allclose(np.asarray(result["alpha"]), [0.11, 0.24])
    assert np.allclose(np.asarray(influence["alpha_dose"]), [0.22, 0.72])


def test_custom_model_runs_in_particle_engine(registered_custom_model, test_data_protons_raw):
    """The public model/evaluator contract integrates with the particle dose engine."""
    pln = validate_pln(test_data_protons_raw["pln"])
    ct = validate_ct(test_data_protons_raw["ct"])
    cst = validate_cst(test_data_protons_raw["cst"], ct=ct)
    stf = validate_stf(test_data_protons_raw["stf"])
    pln.bio_model = {"model": registered_custom_model.model, "slope": 0.03}

    dij = calc_dose_influence(ct, cst, stf, pln)

    assert isinstance(dij.bio_model, _LinearLETModel)
    assert dij.bio_model.parameters == {"slope": 0.03}
    assert dij.alpha_dose is not None and dij.alpha_dose.flat[0].count_nonzero() > 0
    assert dij.sqrt_beta_dose is not None and dij.sqrt_beta_dose.flat[0].count_nonzero() > 0


def test_registration_does_not_partially_register_on_collision(registered_custom_model):
    """A colliding canonical name does not leave a free alias partially registered."""

    class ConflictingModel(BiologicalModel):
        model = registered_custom_model.model
        model_aliases = ("extension_test_unused_alias",)
        possible_radiation_modes = ("protons",)

        def evaluator(self, machine, voxel_params) -> BioModelEvaluator:
            return BioModelEvaluator(self)

    with pytest.raises(ValueError, match="already registered.*extension_test_linear_let"):
        register_model(ConflictingModel)

    assert "extension_test_unused_alias" not in BIO_MODELS


def test_direct_custom_instance_validates_model_declarations():
    """Unregistered instances receive the same declaration validation as registered models."""

    class InvalidModel(BiologicalModel):
        model = "extension_test_invalid"
        possible_radiation_modes: ClassVar[list[str]] = ["protons"]

        def evaluator(self, machine, voxel_params) -> BioModelEvaluator:
            return BioModelEvaluator(self)

    with pytest.raises(TypeError, match="possible_radiation_modes must be a tuple"):
        create_bio_model(InvalidModel())


def test_custom_evaluator_validates_tuple_declarations():
    """Evaluator declarations fail early instead of reaching a bixel calculation."""

    class InvalidEvaluator(_LinearLETEvaluator):
        @property
        def kernel_field_names(self):
            return ["depth_factor"]

    with pytest.raises(TypeError, match="kernel_field_names must be a tuple"):
        InvalidEvaluator(_LinearLETModel()).validate_declarations()


def test_dose_engine_rejects_invalid_custom_evaluator():
    """Dose-engine setup rejects a wrong return type or model binding immediately."""
    engine = SimpleNamespace(_machine=None)

    class WrongTypeModel(_LinearLETModel):
        model = "extension_test_wrong_type"

        def evaluator(self, machine, voxel_params):
            return object()

    with pytest.raises(TypeError, match="must return a BioModelEvaluator.*object"):
        DoseEngineBase._create_bio_evaluator(engine, WrongTypeModel(), {})

    class WrongBindingModel(_LinearLETModel):
        model = "extension_test_wrong_binding"

        def evaluator(self, machine, voxel_params):
            return _LinearLETEvaluator(_LinearLETModel())

    with pytest.raises(ValueError, match="bound to a different model instance"):
        DoseEngineBase._create_bio_evaluator(engine, WrongBindingModel(), {})
