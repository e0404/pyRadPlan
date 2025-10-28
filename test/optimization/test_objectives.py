# import numpy as np
from pyRadPlan.optimization.objectives import (
    DoseUniformity,
    SquaredDeviation,
    SquaredOverdosing,
    SquaredUnderdosing,
    EUD,
    MeanDose,
    MinDVH,
    MaxDVH,
    get_available_objectives,
    get_objective,
)

import array_api_strict as np


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

    dose = np.asarray([1, 2, 3], dtype=np.float32)
    assert np.abs(doseUni.compute_objective(dose) - 1) <= np.finfo(np.float32).eps


def test_DoseUniformity_compute_gradient():
    doseUni = DoseUniformity()
    dose = np.asarray([1, 2, 3], dtype=np.float32)
    grad_expected = 1 / 2 * np.asarray([-1, 0, 1], dtype=np.float32)
    assert np.all((doseUni.compute_gradient(dose) - grad_expected) <= np.finfo(np.float32).eps)


def test_SquaredDeviation_constructor():
    sq_dev = SquaredDeviation(d_ref=2, priority=100)
    assert sq_dev.parameters == [2.0]
    assert sq_dev.name == "Squared Deviation"
    assert sq_dev.d_ref == 2.0
    assert sq_dev.priority == 100.0


def test_SquaredDeviation_compute_objective():
    dose = np.asarray([1, 2, 3], dtype=np.float32)
    sq_dev = SquaredDeviation(d_ref=2.0)
    assert sq_dev.compute_objective(dose) == 2 / 3


def test_SquaredDeviation_compute_gradient():
    dose = np.asarray([1, 2, 3], dtype=np.float32)
    sq_dev = SquaredDeviation(d_ref=2.0)
    grad_expected = 2 / 3 * np.asarray([-1, 0, 1], dtype=np.float32)
    assert np.all(sq_dev.compute_gradient(dose) == grad_expected)


def test_SquaredOverdosing_constructor():
    sq_over = SquaredOverdosing(d_max=2, priority=100)
    assert sq_over.parameter_names == ["d_max"]
    # assert sq_over.parameter_types == ["dose"]
    assert sq_over.parameters == [2.0]
    assert sq_over.d_max == 2.0
    assert sq_over.priority == 100.0


def test_SquaredOverdosing_compute_objective():
    sq_over = SquaredOverdosing(d_max=2.0)
    dose = np.asarray([1, 2, 3], dtype=np.float32)
    assert sq_over.compute_objective(dose) == 1 / 3


def test_SquaredOverdosing_compute_gradient():
    dose = np.asarray([1, 2, 3], dtype=np.float32)
    sq_over = SquaredOverdosing(d_max=2)
    grad_expected = 2 / 3 * np.asarray([0, 0, 1], dtype=np.float32)
    assert np.all(sq_over.compute_gradient(dose) == grad_expected)


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
    dose = np.asarray([1, 2, 3], dtype=np.float32)
    assert sq_under.compute_objective(dose) == 1.0 / 3.0


def test_SquaredUnderdosing_compute_gradient():
    dose = np.asarray([1, 2, 3], dtype=np.float32)
    sq_under = SquaredUnderdosing(d_min=2.0)
    grad_expected = 2 / 3 * np.asarray([-1, 0, 0], dtype=np.float32)
    assert np.all(sq_under.compute_gradient(dose) == grad_expected)


def test_EUD_constructor():
    eud = EUD(k=3, eud_ref=0.0, priority=100)
    assert eud.name == "EUD"
    assert eud.parameter_names == ["eud_ref", "k", "f_diff"]
    assert eud.parameter_types == ["reference", "numeric", ["linear", "quadratic"]]
    assert eud.parameters == [0.0, 3.0, "quadratic"]
    assert eud.priority == 100.0


def test_EUD_compute_objective():
    eud = EUD(k=3, EUD_ref=0.0)
    dose = np.asarray([1, 2, 3], dtype=np.float32)
    assert (eud.compute_objective(dose) - (1 / 3 * (1 + 2 ** (1 / 3) + 3 ** (1 / 3))) ** 6) < 1e-10

    eud.f_diff = "linear"
    assert (eud.compute_objective(dose) - (1 / 3 * (1 + 2 ** (1 / 3) + 3 ** (1 / 3))) ** 3) < 1e-10


def test_EUD_compute_gradient():
    eud_obj = EUD(k=3, EUD_ref=0.0)
    dose = np.asarray([1, 2, 3], dtype=np.float32)
    d_eud = (
        (1 + 2 ** (1 / 3) + 3 ** (1 / 3)) ** 2
        * np.asarray([1, 2, 3], dtype=np.float32) ** (-2 / 3)
        * 1
        / 3**3
    )
    eud = (1 / 3 * (1 + 2 ** (1 / 3) + 3 ** (1 / 3))) ** 3
    grad_expected = 2 * (eud - 0) * d_eud
    assert np.all((eud_obj.compute_gradient(dose) - grad_expected) < 1e-10)

    eud_obj.f_diff = "linear"
    grad_expected = np.sign(np.asarray(eud - 0.0, dtype=np.float32)) * d_eud
    assert np.all((eud_obj.compute_gradient(dose) - grad_expected) < 1e-10)


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
    dose = np.asarray([1, 2, 3], dtype=np.float32)
    assert mean_dose.compute_objective(dose) == 0
    mean_dose.f_diff = "linear"
    mean_dose.d_ref = 1.0
    assert mean_dose.compute_objective(dose) == 1


def test_MeanDose_compute_gradient():
    mean_dose = MeanDose(d_ref=2.0)
    dose = np.asarray([1, 2, 3], dtype=np.float32)
    grad_expected = np.zeros(3)
    assert np.all(mean_dose.compute_gradient(dose) == grad_expected)

    mean_dose.f_diff = "linear"
    mean_dose.d_ref = 1.0
    grad_expected = np.full(3, 1 / 3, dtype=np.float32)
    assert np.all(mean_dose.compute_gradient(dose) == grad_expected)


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
    dose = np.ones(100)
    dose_2 = np.ones(100) * 50
    assert min_dvh.compute_objective(dose) == 841
    assert min_dvh.compute_objective(dose_2) == 0


def test_MinDVH_compute_gradient():
    min_dvh = MinDVH(d=30.0, v_min=95)
    dose = np.ones(100)
    dose_2 = np.ones(100) * 50
    grad_expected = np.ones(100) * -0.58
    grad_expected2 = np.zeros(100)
    assert np.all(min_dvh.compute_gradient(dose) == grad_expected)
    assert np.all(min_dvh.compute_gradient(dose_2) == grad_expected2)


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
    dose = np.ones(100)
    dose_2 = np.ones(100) * 50
    assert max_dvh.compute_objective(dose) == 0
    assert max_dvh.compute_objective(dose_2) == 400


def test_MaxDVH_compute_gradient():
    max_dvh = MaxDVH(d=30.0, v_max=50)
    dose = np.ones(100)
    dose_2 = np.ones(100) * 50
    grad_expected = np.zeros(100)
    grad_expected2 = np.ones(100) * 0.4
    assert np.all(max_dvh.compute_gradient(dose) == grad_expected)
    assert np.all(max_dvh.compute_gradient(dose_2) == grad_expected2)
