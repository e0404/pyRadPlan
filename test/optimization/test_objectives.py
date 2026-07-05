import pytest

from pyRadPlan.optimization.objectives import (
    DoseUniformity,
    SquaredDeviation,
    SquaredMimicking,
    SquaredOverdosing,
    SquaredUnderdosing,
    EUD,
    MeanDose,
    MinDVH,
    MaxDVH,
    get_available_objectives,
    get_objective,
)

import array_api_strict as xp
import numpy as np
import SimpleITK as sitk
from pyRadPlan.core import Grid, xp_utils


def test_objective_availability():
    available_objectives = get_available_objectives()
    assert "Dose Uniformity" in available_objectives
    assert "Squared Deviation" in available_objectives
    assert "Squared Overdosing" in available_objectives
    assert "Squared Underdosing" in available_objectives
    assert "EUD" in available_objectives
    assert "Mean Dose" in available_objectives
    assert "Min DVH" in available_objectives
    assert "Max DVH" in available_objectives


def test_get_objective_str():
    dose_uni = get_objective("Dose Uniformity")
    assert isinstance(dose_uni, DoseUniformity)


def test_get_objective_dict():
    dose_uni = get_objective({"name": "Dose Uniformity", "priority": 10.0})
    assert isinstance(dose_uni, DoseUniformity)
    assert dose_uni.priority == 10.0


def test_get_objective_instance():
    dose_uni = DoseUniformity(priority=10.0)
    dose_uni2 = get_objective(dose_uni)
    assert dose_uni == dose_uni2


def test_get_objective_from_matrad_tg119(tg119_raw):
    _, cst = tg119_raw

    obj_mat_1 = cst[0][5]
    obj = get_objective(obj_mat_1)
    obj_mat_2 = cst[1][5]
    obj2 = get_objective(obj_mat_2)

    assert isinstance(obj, SquaredOverdosing)
    assert obj.priority == obj_mat_1["penalty"]
    assert obj.d_max == obj_mat_1["parameters"]

    assert isinstance(obj2, SquaredDeviation)
    assert obj2.priority == obj_mat_2["penalty"]
    assert obj2.d_ref == obj_mat_2["parameters"]


def test_DoseUniformity_constructor():
    doseUni = DoseUniformity()
    assert doseUni.name == "Dose Uniformity"
    assert doseUni.parameters == []
    assert doseUni.priority == 1.0


def test_DoseUniformity_compute_objective():
    doseUni = DoseUniformity()

    dose = xp.asarray([1, 2, 3], dtype=xp.float32)
    assert xp.abs(doseUni.compute_objective(dose) - 1) <= xp.finfo(xp.float32).eps


def test_DoseUniformity_compute_gradient():
    doseUni = DoseUniformity()
    dose = xp.asarray([1, 2, 3], dtype=xp.float32)
    grad_expected = 1 / 2 * xp.asarray([-1, 0, 1], dtype=xp.float32)
    assert xp.all((doseUni.compute_gradient(dose) - grad_expected) <= xp.finfo(xp.float32).eps)


def test_SquaredDeviation_constructor():
    sq_dev = SquaredDeviation(d_ref=2, priority=100)
    assert sq_dev.parameters == [2.0]
    assert sq_dev.name == "Squared Deviation"
    assert sq_dev.d_ref == 2.0
    assert sq_dev.priority == 100.0


def test_SquaredDeviation_compute_objective():
    dose = xp.asarray([1, 2, 3], dtype=xp.float32)
    sq_dev = SquaredDeviation(d_ref=2.0)
    assert sq_dev.compute_objective(dose) == 2 / 3


def test_SquaredDeviation_compute_gradient():
    dose = xp.asarray([1, 2, 3], dtype=xp.float32)
    sq_dev = SquaredDeviation(d_ref=2.0)
    grad_expected = 2 / 3 * xp.asarray([-1, 0, 1], dtype=xp.float32)
    assert xp.all(sq_dev.compute_gradient(dose) == grad_expected)


def test_SquaredMimicking_image():
    image = sitk.GetImageFromArray(xp.full((5, 5, 5), 60.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetSpacing((1.0, 1.0, 1.0))
    sq_mimic = SquaredMimicking(d_ref=image)
    assert isinstance(sq_mimic, SquaredMimicking)
    assert isinstance(sq_mimic.d_ref, sitk.Image)


def test_SquaredMimicking_array():
    array = xp.full((5, 5, 5), 60.0)
    grid = Grid(
        resolution={"x": 3.0, "y": 3.0, "z": 3.0}, dimensions=(5, 5, 5), origin=[0.0, 0.0, 0.0]
    )
    sq_mimic = SquaredMimicking(d_ref=(array, grid))
    assert isinstance(sq_mimic, SquaredMimicking)
    assert isinstance(sq_mimic.d_ref, tuple)
    assert isinstance(sq_mimic.d_ref[0], xp._array_object.Array)
    assert isinstance(sq_mimic.d_ref[1], Grid)


def test_SquaredMimicking_compute_objective():
    sq_mimic = SquaredMimicking()
    sq_mimic._resampled_image_reference_cache["d_ref"] = xp.full(8, 60.0)
    dose = xp.full(8, 60.0)
    assert float(sq_mimic.compute_objective(dose)) == 0.0


def test_SquaredMimicking_cache():
    array = xp.full((2, 2, 2), 60.0)
    grid = Grid(
        resolution={"x": 3.0, "y": 3.0, "z": 3.0}, dimensions=(2, 2, 2), origin=[0.0, 0.0, 0.0]
    )
    sq_mimic = SquaredMimicking(d_ref=(array, grid))
    sq_mimic.preprocess_image_reference_parameters(target_grid=grid, index_list=xp.arange(8))
    assert sq_mimic._resampled_image_reference_cache["d_ref"].shape == (8,)
    assert np.allclose(
        xp_utils.to_numpy(sq_mimic._resampled_image_reference_cache["d_ref"]),
        np.full(8, 60.0),
        atol=1e-10,
    )


def _preprocessed_mimicking():
    """SquaredMimicking whose cache was built with numpy, as in the planning problem."""
    image = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    image += 2.0
    sq_mimic = SquaredMimicking(d_ref=image)
    sq_mimic.preprocess_image_reference_parameters(
        target_grid=Grid.from_sitk_image(image), index_list=np.arange(8)
    )
    assert isinstance(sq_mimic._resampled_image_reference_cache["d_ref"], np.ndarray)
    return sq_mimic


def test_SquaredMimicking_values_from_other_namespace():
    # the optimization backend (values) may differ from the numpy-built cache
    sq_mimic = _preprocessed_mimicking()
    dose = xp.full(8, 2.0, dtype=xp.float64)
    assert float(sq_mimic.compute_objective(dose)) == 0.0
    grad = sq_mimic.compute_gradient(dose)
    assert isinstance(grad, type(dose))


@pytest.mark.parametrize("backend", ["torch", "cupy"])
def test_SquaredMimicking_values_from_gpu_backends(backend):
    xpb = pytest.importorskip(backend)
    sq_mimic = _preprocessed_mimicking()
    dose = xpb.full((8,), 2.0)
    assert float(sq_mimic.compute_objective(dose)) == 0.0
    grad = sq_mimic.compute_gradient(dose)
    assert isinstance(grad, type(dose))


def test_SquaredOverdosing_constructor():
    sq_over = SquaredOverdosing(d_max=2, priority=100)
    assert sq_over.parameter_names == ["d_max"]
    # assert sq_over.parameter_types == ["dose"]
    assert sq_over.parameters == [2.0]
    assert sq_over.d_max == 2.0
    assert sq_over.priority == 100.0


def test_SquaredOverdosing_compute_objective():
    sq_over = SquaredOverdosing(d_max=2.0)
    dose = xp.asarray([1, 2, 3], dtype=xp.float32)
    assert sq_over.compute_objective(dose) == 1 / 3


def test_SquaredOverdosing_compute_gradient():
    dose = xp.asarray([1, 2, 3], dtype=xp.float32)
    sq_over = SquaredOverdosing(d_max=2)
    grad_expected = 2 / 3 * xp.asarray([0, 0, 1], dtype=xp.float32)
    assert xp.all(sq_over.compute_gradient(dose) == grad_expected)


def test_SquaredUnderdosing_constructor():
    sq_under = SquaredUnderdosing(d_min=2, priority=100)
    assert sq_under.name == "Squared Underdosing"
    assert sq_under.parameter_names == ["d_min"]
    # assert sq_under.parameter_types == ["dose"]
    assert sq_under.parameters == [2.0]
    assert sq_under.d_min == 2.0
    assert sq_under.priority == 100.0


def test_SquaredUnderdosing_compute_objective():
    sq_under = SquaredUnderdosing(d_min=2.0)
    dose = xp.asarray([1, 2, 3], dtype=xp.float32)
    assert sq_under.compute_objective(dose) == 1.0 / 3.0


def test_SquaredUnderdosing_compute_gradient():
    dose = xp.asarray([1, 2, 3], dtype=xp.float32)
    sq_under = SquaredUnderdosing(d_min=2.0)
    grad_expected = 2 / 3 * xp.asarray([-1, 0, 0], dtype=xp.float32)
    assert xp.all(sq_under.compute_gradient(dose) == grad_expected)


def test_EUD_constructor():
    eud = EUD(k=3, eud_ref=0.0, priority=100)
    assert eud.name == "EUD"
    assert eud.parameter_names == ["eud_ref", "k", "f_diff"]
    assert eud.parameter_types == ["reference", "numeric", ["linear", "quadratic"]]
    assert eud.parameters == [0.0, 3.0, "quadratic"]
    assert eud.priority == 100.0


def test_EUD_compute_objective():
    eud = EUD(k=3, EUD_ref=0.0)
    dose = xp.asarray([1, 2, 3], dtype=xp.float32)
    assert (eud.compute_objective(dose) - (1 / 3 * (1 + 2 ** (1 / 3) + 3 ** (1 / 3))) ** 6) < 1e-10

    eud.f_diff = "linear"
    assert (eud.compute_objective(dose) - (1 / 3 * (1 + 2 ** (1 / 3) + 3 ** (1 / 3))) ** 3) < 1e-10


def test_EUD_compute_gradient():
    eud_obj = EUD(k=3, EUD_ref=0.0)
    dose = xp.asarray([1, 2, 3], dtype=xp.float32)
    d_eud = (
        (1 + 2 ** (1 / 3) + 3 ** (1 / 3)) ** 2
        * xp.asarray([1, 2, 3], dtype=xp.float32) ** (-2 / 3)
        * 1
        / 3**3
    )
    eud = (1 / 3 * (1 + 2 ** (1 / 3) + 3 ** (1 / 3))) ** 3
    grad_expected = 2 * (eud - 0) * d_eud
    assert xp.all((eud_obj.compute_gradient(dose) - grad_expected) < 1e-10)

    eud_obj.f_diff = "linear"
    grad_expected = xp.sign(xp.asarray(eud - 0.0, dtype=xp.float32)) * d_eud
    assert xp.all((eud_obj.compute_gradient(dose) - grad_expected) < 1e-10)


def test_MeanDose_constructor():
    mean_dose = MeanDose(d_ref=2, priority=100)
    assert mean_dose.name == "Mean Dose"
    assert mean_dose.parameter_names == ["d_ref", "f_diff"]
    assert mean_dose.parameter_types == ["reference", ["linear", "quadratic"]]
    assert mean_dose.parameters == [2.0, "quadratic"]
    assert mean_dose.d_ref == 2.0
    assert mean_dose.f_diff == "quadratic"
    assert mean_dose.priority == 100.0


def test_MeanDose_compute_objective():
    mean_dose = MeanDose(d_ref=2.0)
    dose = xp.asarray([1, 2, 3], dtype=xp.float32)
    assert mean_dose.compute_objective(dose) == 0
    mean_dose.f_diff = "linear"
    mean_dose.d_ref = 1.0
    assert mean_dose.compute_objective(dose) == 1


def test_MeanDose_compute_gradient():
    mean_dose = MeanDose(d_ref=2.0)
    dose = xp.asarray([1, 2, 3], dtype=xp.float32)
    grad_expected = xp.zeros(3)
    assert xp.all(mean_dose.compute_gradient(dose) == grad_expected)

    mean_dose.f_diff = "linear"
    mean_dose.d_ref = 1.0
    grad_expected = xp.full(3, 1 / 3, dtype=xp.float32)
    assert xp.all(mean_dose.compute_gradient(dose) == grad_expected)


def test_MinDVH_constructor():
    min_dvh = MinDVH(d=2, v_min=3, priority=100)
    assert min_dvh.name == "Min DVH"
    assert min_dvh.parameter_names == ["d", "v_min"]
    assert min_dvh.parameter_types == ["reference", "relative_volume"]
    assert min_dvh.d == 2.0
    assert min_dvh.v_min == 3.0
    assert min_dvh.priority == 100.0
    assert min_dvh.parameters == [2.0, 3.0]


def test_MinDVH_compute_objective():
    min_dvh = MinDVH(d=30.0, v_min=95)
    dose = xp.ones(100)
    dose_2 = xp.ones(100) * 50
    assert min_dvh.compute_objective(dose) == 841
    assert min_dvh.compute_objective(dose_2) == 0


def test_MinDVH_compute_gradient():
    min_dvh = MinDVH(d=30.0, v_min=95)
    dose = xp.ones(100)
    dose_2 = xp.ones(100) * 50
    grad_expected = xp.ones(100) * -0.58
    grad_expected2 = xp.zeros(100)
    assert xp.all(min_dvh.compute_gradient(dose) == grad_expected)
    assert xp.all(min_dvh.compute_gradient(dose_2) == grad_expected2)


def test_MaxDVH_constructor():
    max_dvh = MaxDVH(d=30.0, v_max=50, priority=100)
    assert max_dvh.name == "Max DVH"
    assert max_dvh.parameter_names == ["d", "v_max"]
    assert max_dvh.parameter_types == ["reference", "relative_volume"]
    assert max_dvh.d == 30.0
    assert max_dvh.v_max == 50.0
    assert max_dvh.priority == 100.0
    assert max_dvh.parameters == [30.0, 50.0]


def test_MaxDVH_compute_objective():
    max_dvh = MaxDVH(d=30.0, v_max=50)
    dose = xp.ones(100)
    dose_2 = xp.ones(100) * 50
    assert max_dvh.compute_objective(dose) == 0
    assert max_dvh.compute_objective(dose_2) == 400


def test_MaxDVH_compute_gradient():
    max_dvh = MaxDVH(d=30.0, v_max=50)
    dose = xp.ones(100)
    dose_2 = xp.ones(100) * 50
    grad_expected = xp.zeros(100)
    grad_expected2 = xp.ones(100) * 0.4
    assert xp.all(max_dvh.compute_gradient(dose) == grad_expected)
    assert xp.all(max_dvh.compute_gradient(dose_2) == grad_expected2)
