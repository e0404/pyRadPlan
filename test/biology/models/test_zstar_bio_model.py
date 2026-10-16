"""z-Star based LQ models (MKM).

The zs kernels are tabulated per tissue class, while ``alpha_0`` / ``beta_0`` come from
the tissue itself, so the evaluator has to combine kernel rows with the parameters of the
class each voxel belongs to.
"""

import numpy as np
import pytest
import array_api_strict as xp

from pyRadPlan.biology.models import (
    MKM,
    BioEvaluationContext,
    BioModelResult,
    DerivedAlphaBetaEvaluator,
    ExactClassLookup,
    get_available_models,
)
from pyRadPlan.machines import load_from_name
from pyRadPlan.biology.tissues import MKMTissueProperties


@pytest.fixture
def tissue_classes():
    """Two classes of differing geometry, so each matches exactly one kernel row."""
    return [
        MKMTissueProperties(r_nucleus=3.9, r_domain=0.32, alpha_0=0.1, beta_0=0.05),
        MKMTissueProperties(r_nucleus=5.0, r_domain=0.40, alpha_0=0.2, beta_0=0.06),
    ]


@pytest.fixture
def machine(tissue_classes):
    class _Machine:
        tissue_properties = tissue_classes

    return _Machine()


@pytest.fixture
def voxel_params(tissue_classes):
    return {
        "alpha_x": np.asarray([[0.1], [0.2]]),
        "beta_x": np.asarray([[0.05], [0.06]]),
        "tissue_class": np.asarray(tissue_classes),
        "tissue_class_ix": np.asarray([[0], [1]]),
    }


def test_MKM_declarations():
    model = MKM()
    assert model.model == "zstar_kernels"
    assert model.model_aliases == ("MKM", "zs_based_lq")
    assert model.to_matrad() == "MKM"
    assert model.kernel_quantities == ("zs",)
    assert model.additional_fields == ["alpha_0", "beta_0"]
    assert model.kernel_set_kind == "zs"
    assert model.provides("alpha", "beta")


def test_MKM_evaluator(machine, voxel_params):
    evaluator = MKM().evaluator(machine, voxel_params)
    assert isinstance(evaluator, DerivedAlphaBetaEvaluator)
    assert isinstance(evaluator.lookup, ExactClassLookup)
    assert evaluator.kernel_field_names == ("zs",)
    assert evaluator.additional_fields == ["alpha_0", "beta_0"]


def test_MKM_evaluate_combines_kernel_rows_with_the_class_parameters(machine, voxel_params):
    """alpha = alpha_0 + beta_0 * zs and beta = beta_0, both of the voxel's own class."""
    evaluator = MKM().evaluator(machine, voxel_params)

    result = evaluator.evaluate(
        BioEvaluationContext(
            {
                "alpha_x": xp.asarray([0.1, 0.2, 0.2]),
                "beta_x": xp.asarray([0.05, 0.06, 0.06]),
                "tissue_class_ix": xp.asarray([0, 1, 1]),
                # interpolated zs kernels: (n_tissue_classes, n_voxels)
                "zs": xp.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            }
        )
    )

    assert isinstance(result, BioModelResult)
    expected_alpha = [0.1 + 0.05 * 1.0, 0.2 + 0.06 * 2.0, 0.2 + 0.06 * 2.0]
    assert np.allclose(np.asarray(result["alpha"]), expected_alpha)
    assert np.allclose(np.asarray(result["beta"]), [0.05, 0.06, 0.06])


def test_mkm_copies_no_host_data_per_bixel(machine, voxel_params, monkeypatch):
    """Evaluating further bixels only indexes arrays already on the device."""
    evaluator = MKM().evaluator(machine, voxel_params)
    rng = np.random.default_rng(1)
    bixels = [
        BioEvaluationContext(
            {
                "alpha_x": xp.asarray([0.1, 0.2, 0.2]),
                "beta_x": xp.asarray([0.05, 0.06, 0.06]),
                "tissue_class_ix": xp.asarray(rng.integers(0, 2, 3)),
                "physical_dose": xp.asarray(rng.random(3)),
                "zs": xp.asarray(rng.random((2, 3))),
            }
        )
        for _ in range(10)
    ]
    host_conversions = []
    asarray = xp.asarray

    def counting(obj, *args, **kwargs):
        if isinstance(obj, (list, tuple, np.ndarray)):
            host_conversions.append(obj)
        return asarray(obj, *args, **kwargs)

    monkeypatch.setattr(xp, "asarray", counting)
    evaluator.evaluate_influence(bixels[0])
    first = len(host_conversions)
    for context in bixels[1:]:
        evaluator.evaluate_influence(context)

    assert first > 0
    assert len(host_conversions) == first


def test_MKM_evaluate_influence_weights_the_class_parameters_with_the_dose(machine, voxel_params):
    evaluator = MKM().evaluator(machine, voxel_params)
    physical_dose = xp.asarray([2.0, 4.0])

    result = evaluator.evaluate_influence(
        BioEvaluationContext(
            {
                "alpha_x": xp.asarray([0.1, 0.2]),
                "beta_x": xp.asarray([0.05, 0.06]),
                "tissue_class_ix": xp.asarray([0, 1]),
                "physical_dose": physical_dose,
                "zs": xp.asarray([[1.0, 1.0], [2.0, 2.0]]),
            }
        )
    )

    alpha = np.asarray([0.1 + 0.05 * 1.0, 0.2 + 0.06 * 2.0])
    beta = np.asarray([0.05, 0.06])
    assert np.allclose(np.asarray(result["alpha_dose"]), np.asarray([2.0, 4.0]) * alpha)
    assert np.allclose(
        np.asarray(result["sqrt_beta_dose"]), np.asarray([2.0, 4.0]) * np.sqrt(beta)
    )


def test_MKM_rejects_a_machine_without_matching_tissue_classes(voxel_params):
    """Base data tabulated for another cell geometry cannot serve these structures."""

    class _Machine:
        tissue_properties = [MKMTissueProperties(r_nucleus=1.0, r_domain=1.0)]

    with pytest.raises(ValueError, match="No matching tissue class"):
        MKM().evaluator(_Machine(), voxel_params)


def test_MKM_requires_zs_kernels():
    assert MKM().required_quantities == ("physical_dose", "zs")
    assert not MKM().is_available("carbon", ["physical_dose", "alpha", "beta"])[0]
    assert MKM().is_available("oxygen", ["physical_dose", "zs"])[0]


@pytest.mark.parametrize(
    ("mode", "offered"),
    [("oxygen", True), ("carbon", False), ("helium", False), ("protons", False)],
)
def test_MKM_offered_only_for_machines_with_zs_kernels(mode, offered):
    machine = load_from_name(mode, "Generic")
    assert ("zs" in machine.provided_quantities()) is offered
    assert ("MKM" in get_available_models(mode, machine.provided_quantities())) is offered


@pytest.mark.parametrize("unset", ["alpha_0", "beta_0"])
def test_MKM_rejects_tissues_without_alpha_0_or_beta_0(voxel_params, unset):
    """An unset alpha_0 / beta_0 would evaluate to NaN in every voxel of the structure."""
    tissues = [
        MKMTissueProperties(r_nucleus=3.9, r_domain=0.32, alpha_0=0.1, beta_0=0.05),
        MKMTissueProperties(
            r_nucleus=5.0, r_domain=0.40, **{"alpha_0": 0.2, "beta_0": 0.06, unset: float("nan")}
        ),
    ]

    class _Machine:
        tissue_properties = tissues

    params = {**voxel_params, "tissue_class": np.asarray(tissues)}
    with pytest.raises(ValueError, match=f"lacks {unset}"):
        MKM().evaluator(_Machine(), params)


def test_MKM_declares_the_tissue_parameters_it_reads():
    model = MKM()
    assert model.required_tissue_parameters == ("alpha_0", "beta_0")
    assert model.missing_tissue_parameters(MKMTissueProperties(alpha_0=0.1)) == ("beta_0",)
    assert model.missing_tissue_parameters(MKMTissueProperties(alpha_0=0.1, beta_0=0.05)) == ()
