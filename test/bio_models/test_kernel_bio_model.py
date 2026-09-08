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
    assert kernel_lq_model.required_quantities == ("physical_dose", "alpha", "beta")
    assert kernel_lq_model.possible_radiation_modes == ("protons", "helium", "carbon", "oxygen")
    assert kernel_lq_model.kernel_quantities == ("alpha", "beta")
    assert kernel_lq_model.output_quantities == ("alpha", "beta")
    assert kernel_lq_model.provides("alpha", "beta")
    assert not kernel_lq_model.requires("let")


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


class _MultiEnergyMachine:
    """Machine stand-in with one alpha/beta kernel per energy."""

    def __init__(self, classes_per_energy):
        self.energies = sorted(classes_per_energy)
        self.pb_kernels = {
            energy: _Kernel(*classes) for energy, classes in classes_per_energy.items()
        }


def test_reference_classes_require_the_same_order_at_every_energy():
    """Reordered tissue rows at another energy would silently select the wrong alpha/beta."""
    machine = _MultiEnergyMachine(
        {100.0: ([0.1, 0.5], [0.05, 0.05]), 200.0: ([0.5, 0.1], [0.05, 0.05])}
    )
    voxel_params = {
        "alpha_x": np.asarray([[0.1], [0.5]]),
        "beta_x": np.asarray([[0.05], [0.05]]),
    }

    with pytest.raises(ValueError, match="identical tissue classes in the same order"):
        KernelBasedLQModel().evaluator(machine, voxel_params)

    # why it must not be accepted: the class index is computed once from one ordering, so
    # index 1 selects (alpha_x=0.5) at 100 MeV but (alpha_x=0.1) at 200 MeV.
    lookup = ExactClassLookup(*[np.asarray(v) for v in ([0.1, 0.5], [0.05, 0.05])])
    class_ix = lookup.class_index(np.asarray([0.5]), np.asarray([0.05]))
    assert int(class_ix[0]) == 1
    assert machine.pb_kernels[200.0].alpha_x[int(class_ix[0])] == 0.1


def test_reference_classes_accept_identical_classes_at_every_energy(voxel_params):
    machine = _MultiEnergyMachine(
        {100.0: ([0.1, 0.5], [0.05, 0.05]), 200.0: ([0.1, 0.5], [0.05, 0.05])}
    )
    model = KernelBasedLQModel()
    class_alpha_x, class_beta_x = model.reference_classes(machine)

    assert class_alpha_x.tolist() == [0.1, 0.5]
    assert class_beta_x.tolist() == [0.05, 0.05]
    assert isinstance(model.evaluator(machine, voxel_params), KernelBasedEvaluator)


def test_reference_classes_reject_differing_class_values(voxel_params):
    machine = _MultiEnergyMachine(
        {100.0: ([0.1, 0.5], [0.05, 0.05]), 200.0: ([0.1, 0.4], [0.05, 0.05])}
    )
    with pytest.raises(ValueError, match="identical tissue classes in the same order"):
        KernelBasedLQModel().evaluator(machine, voxel_params)


def test_reference_classes_reject_missing_class_metadata(voxel_params):
    machine = _MultiEnergyMachine({100.0: ([0.1, 0.5], [0.05, 0.05])})
    machine.pb_kernels[200.0] = _Kernel([0.1, 0.5], [0.05, 0.05])
    machine.pb_kernels[200.0].beta_x = None

    with pytest.raises(ValueError, match="kernel of energy 200.0 declares none"):
        KernelBasedLQModel().evaluator(machine, voxel_params)


def test_reference_classes_reject_inconsistent_class_metadata(voxel_params):
    machine = _MultiEnergyMachine({100.0: ([0.1, 0.5], [0.05])})

    with pytest.raises(ValueError, match="2 reference alpha_x but 1 reference beta_x"):
        KernelBasedLQModel().evaluator(machine, voxel_params)


def test_reference_classes_reject_machine_without_kernels(voxel_params):
    machine = _MultiEnergyMachine({})

    with pytest.raises(ValueError, match="the machine carries none"):
        KernelBasedLQModel().evaluator(machine, voxel_params)
