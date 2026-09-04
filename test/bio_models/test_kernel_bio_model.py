import numpy as np
import pytest
import array_api_strict as xp

from pyRadPlan.bio_models import (
    BioEvaluationContext,
    BioModelResult,
    BiologicalModel,
    ExactClassLookup,
    KernelBasedEvaluator,
    KernelBasedLQModel,
)


class _Kernel:
    def __init__(self, alpha_x, beta_x):
        self.alpha_x = np.asarray(alpha_x)
        self.beta_x = np.asarray(beta_x)


class _Machine:
    """Minimal stand-in exposing the attributes the model reads."""

    def __init__(self, alpha_x, beta_x):
        self.energies = [100.0]
        self.pb_kernels = {100.0: _Kernel(alpha_x, beta_x)}


@pytest.fixture
def machine():
    return _Machine([0.1, 0.5], [0.05, 0.05])


@pytest.fixture
def voxel_params():
    return {
        "alpha_x": np.asarray([[0.1], [0.5], [0.0]]),
        "beta_x": np.asarray([[0.05], [0.05], [0.0]]),
    }


def test_KernelBasedLQModel_constructor():
    kernel_lq_model = KernelBasedLQModel()
    assert isinstance(kernel_lq_model, BiologicalModel)
    assert kernel_lq_model.model == "kernel_based_lq"
    assert kernel_lq_model.required_quantities == ["physical_dose", "alpha", "beta"]
    assert kernel_lq_model.possible_radiation_modes == ["protons", "helium", "carbon", "oxygen"]
    assert kernel_lq_model.kernel_quantities == ["alpha", "beta"]
    assert kernel_lq_model.provides_alpha_beta is True
    assert kernel_lq_model.requires_let is False


def test_KernelBasedLQModel_evaluator(machine, voxel_params):
    evaluator = KernelBasedLQModel().evaluator(machine, voxel_params)
    assert isinstance(evaluator, KernelBasedEvaluator)
    assert isinstance(evaluator.lookup, ExactClassLookup)

    kernel = {"alpha": "A", "beta": "B", "idd": "ignored"}
    assert evaluator.kernel_field_names == ("alpha", "beta")
    assert evaluator.kernel_quantities(kernel) == {"alpha": "A", "beta": "B"}


def test_KernelBasedLQModel_evaluator_rejects_unknown_tissue(machine):
    with pytest.raises(ValueError, match="No matching tissue class"):
        KernelBasedLQModel().evaluator(
            machine, {"alpha_x": np.asarray([[0.3]]), "beta_x": np.asarray([[0.05]])}
        )


def test_KernelBasedLQModel_evaluate_mixed_tissue_classes(machine, voxel_params):
    evaluator = KernelBasedLQModel().evaluator(machine, voxel_params)
    alpha_x = xp.asarray([0.1, 0.5, 0.1, 0.0])
    beta_x = xp.asarray([0.05, 0.05, 0.05, 0.0])
    # interpolated kernels: (n_tissue_classes, n_voxels)
    kernels = {
        "alpha": xp.asarray([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]),
        "beta": xp.asarray([[4.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 5.0]]),
    }
    result = evaluator.evaluate(
        BioEvaluationContext(
            {
                "alpha_x": alpha_x,
                "beta_x": beta_x,
                **kernels,
            }
        )
    )

    assert isinstance(result, BioModelResult)
    assert np.allclose(np.asarray(result["alpha"]), [1.0, 2.0, 1.0, 1.0])
    assert np.allclose(np.asarray(result["beta"]), [4.0, 5.0, 4.0, 4.0])


def test_KernelBasedLQModel_unknown_lookup(machine, voxel_params):
    with pytest.raises(ValueError, match="Unknown tissue lookup"):
        KernelBasedLQModel(tissue_lookup="nearest").evaluator(machine, voxel_params)
