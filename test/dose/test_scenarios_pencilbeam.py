"""Scenario handling of the pencil-beam engines (ScenarioSet consumption, defects 1 and 2)."""

from typing import ClassVar, Literal

import numpy as np
import pytest
import SimpleITK as sitk
from scipy import sparse

from pyRadPlan.core.np2sitk import linear_indices_to_grid_coordinates
from pyRadPlan.ct import CT
from pyRadPlan.dose import calc_dose_forward, calc_dose_influence
from pyRadPlan.dose.engines import ParticleFredMCEngine, get_engine
from pyRadPlan.raytracer import RayTracerSiddon
from pyRadPlan.scenarios import (
    CTPhaseUncertainty,
    ExplicitScenarioModel,
    RangeUncertainty,
    ScenarioBasis,
    ScenarioSet,
    SetupUncertainty,
    UncertaintyDimension,
    register_dimension,
)


@register_dimension
class _ScenarioTestDimension(UncertaintyDimension):
    """A continuous dimension no dose engine knows."""

    name: ClassVar[str] = "scenario_test_dimension"
    components: ClassVar[tuple[str, ...]] = ("value",)
    kind: ClassVar[Literal["continuous", "discrete"]] = "continuous"
    units: ClassVar[tuple[str, ...]] = ("",)
    affects: ClassVar[frozenset[str]] = frozenset({"voxel_params"})

    def mean(self) -> np.ndarray:
        return np.zeros(1)

    def std(self) -> np.ndarray:
        return np.ones(1)


SHIFT_X = (3.0, 0.0, 0.0)
SHIFT_Y = (0.0, 3.0, 0.0)
RANGE_ABS = (2.0, 0.0)


@pytest.fixture(params=["protons", "photons"])
def case(request):
    pln, ct, cst, stf, _, _ = request.getfixturevalue(f"test_data_{request.param}")
    pln.prop_dose_calc["enable_dij_sampling"] = False
    return pln, ct, cst, stf


def _scenario_set(*, setup=None, range_=None, ct_phase=None, phases=(0,), bases=None):
    dims = (
        CTPhaseUncertainty(phases=list(phases), probabilities=[1.0 / len(phases)] * len(phases)),
        SetupUncertainty(),
        RangeUncertainty(),
    )
    points = {}
    if setup is not None:
        points["setup"] = np.asarray(setup, dtype=float)
    if range_ is not None:
        points["range"] = np.asarray(range_, dtype=float)
    if ct_phase is not None:
        points["ct_phase"] = np.asarray(ct_phase, dtype=float)
    return ScenarioSet.from_points(dims, points, bases=bases)


def _calc(pln, ct, cst, stf, scenarios=None, quantity="physical_dose"):
    engine = get_engine(pln)
    dij = engine.calc_dose_influence(ct, cst, stf, scenarios=scenarios)
    return engine, dij, list(getattr(dij, quantity))


def _shifted(stf, shift):
    shifted = stf.model_copy(deep=True)
    for beam in shifted.beams:
        beam.iso_center = beam.iso_center + np.asarray(shift)
    return shifted


def _assert_equal(a, b):
    a, b = sparse.csc_array(a), sparse.csc_array(b)
    assert a.shape == b.shape
    assert (a != b).nnz == 0


def _assert_differs(a, b):
    assert (sparse.csc_array(a) != sparse.csc_array(b)).nnz > 0


def test_nominal_set_has_one_flat_slot(case):
    pln, ct, cst, stf = case
    engine, dij, slots = _calc(pln, ct, cst, stf)

    assert dij.physical_dose.shape == (1,)
    assert dij.num_of_scenarios == 1
    assert dij.scenarios is engine._scenarios
    assert len(engine._scenarios) == 1
    assert slots[0] is not None


def test_scenarios_are_recorded_on_the_dij(case):
    pln, ct, cst, stf = case
    engine = get_engine(pln)
    dij = engine.calc_dose_influence(ct, cst, stf)
    recorded = getattr(dij, "scenarios", None)
    assert recorded is None or recorded is engine._scenarios


def test_setup_and_range_scenarios(case, monkeypatch):
    pln, ct, cst, stf = case
    scen_set = _scenario_set(
        setup=[(0.0, 0.0, 0.0), SHIFT_X, (0.0, 0.0, 0.0)],
        range_=[(0.0, 0.0), (0.0, 0.0), RANGE_ABS],
    )

    engine, dij, slots = _calc(pln, ct, cst, stf, scenarios=scen_set)
    assert engine._scenarios is scen_set
    assert dij.scenarios is scen_set
    assert dij.num_of_scenarios == 3
    assert engine.mult_scen is not scen_set

    _, _, nominal = _calc(pln, ct, cst, stf)
    _assert_equal(slots[0], nominal[0])

    _, _, shifted = _calc(pln, ct, cst, _shifted(stf, SHIFT_X))
    _assert_differs(slots[1], nominal[0])
    _assert_equal(slots[1], shifted[0])

    original_trace = RayTracerSiddon.trace_cubes

    def trace_with_offset(self, *args, **kwargs):
        return [cube + RANGE_ABS[0] for cube in original_trace(self, *args, **kwargs)]

    monkeypatch.setattr(RayTracerSiddon, "trace_cubes", trace_with_offset)
    _, _, deeper = _calc(pln, ct, cst, stf)

    _assert_differs(slots[2], nominal[0])
    range_slot = sparse.csc_array(slots[2]).toarray()
    reference = sparse.csc_array(deeper[0]).toarray()
    np.testing.assert_allclose(range_slot, reference, rtol=1e-4, atol=1e-6 * reference.max())


def test_proton_range_shift_pulls_the_dose_upstream(test_data_protons):
    """A larger radiological depth stops the protons earlier, closer to the source."""
    pln, ct, cst, stf, _, _ = test_data_protons
    scen_set = _scenario_set(range_=[(0.0, 0.0), (10.0, 0.0)])
    _, dij, slots = _calc(pln, ct, cst, stf, scenarios=scen_set)

    grid = dij.dose_grid
    coords = linear_indices_to_grid_coordinates(
        np.arange(grid.num_voxels), grid, index_type="numpy"
    )
    beam = stf.beams[0]
    distance = np.linalg.norm(coords - (beam.iso_center + beam.source_point), axis=1)
    columns = np.flatnonzero(dij.beam_num == 0)

    mean_distance = []
    for slot in slots:
        dose = np.asarray(sparse.csc_array(slot)[:, columns].sum(axis=1)).ravel()
        mean_distance.append(distance @ dose / dose.sum())

    assert mean_distance[1] < mean_distance[0] - 1.0


def test_shifts_do_not_accumulate(case):
    pln, ct, cst, stf = case
    iso_before = [beam.iso_center.copy() for beam in stf.beams]
    dump_before = stf.model_dump()

    scen_set = _scenario_set(setup=[SHIFT_X, SHIFT_Y, (0.0, 0.0, 0.0), SHIFT_X])
    _, _, slots = _calc(pln, ct, cst, stf, scenarios=scen_set)

    for beam, iso in zip(stf.beams, iso_before):
        np.testing.assert_array_equal(beam.iso_center, iso)
    after = stf.model_dump()
    for b_before, b_after in zip(dump_before["beams"], after["beams"]):
        np.testing.assert_array_equal(b_before["iso_center"], b_after["iso_center"])

    _assert_equal(slots[0], slots[3])

    _, _, nominal = _calc(pln, ct, cst, stf)
    _assert_equal(slots[2], nominal[0])

    _, _, shifted_y = _calc(pln, ct, cst, _shifted(stf, SHIFT_Y))
    _assert_equal(slots[1], shifted_y[0])


def _two_phase_ct(ct: CT, hu_offset: float = 300.0) -> tuple[CT, CT]:
    """CT whose phase 1 is denser inside the patient (outline unchanged), and phase 1 alone."""
    arr = sitk.GetArrayFromImage(ct.cube_hu)
    arr1 = np.where(arr > -500, arr + hu_offset, arr).astype(arr.dtype)
    phase1 = sitk.GetImageFromArray(arr1)
    phase1.CopyInformation(ct.cube_hu)
    return CT(cube_hu=sitk.JoinSeries([ct.cube_hu, phase1])), CT(cube_hu=phase1)


def test_ct_phase_selection(case):
    pln, ct, cst, stf = case
    ct2, ct_phase1 = _two_phase_ct(ct)
    assert ct2.num_of_ct_scen == 2

    scen_set = _scenario_set(ct_phase=[1, 0], phases=(0, 1))
    _, dij, slots = _calc(pln, ct2, cst, stf, scenarios=scen_set)
    assert dij.physical_dose.shape == (2,)

    _, _, nominal = _calc(pln, ct, cst, stf)
    _, _, nominal_phase1 = _calc(pln, ct_phase1, cst, stf)
    _assert_differs(nominal_phase1[0], nominal[0])

    _assert_equal(slots[0], nominal_phase1[0])
    _assert_equal(slots[1], nominal[0])


def test_ct_phase_selection_bio(test_data_protons):
    pln, ct, cst, stf, _, _ = test_data_protons
    pln.bio_model = "MCN"
    ct2, ct_phase1 = _two_phase_ct(ct)

    scen_set = _scenario_set(ct_phase=[0, 1], phases=(0, 1))
    _, dij, _ = _calc(pln, ct2, cst, stf, scenarios=scen_set)
    assert dij.alphax.shape[1] == 2

    _, ref, _ = _calc(pln, ct_phase1, cst, stf)
    for quantity in ("physical_dose", "alpha_dose", "sqrt_beta_dose"):
        _assert_equal(getattr(dij, quantity)[1], getattr(ref, quantity)[0])


def test_two_phase_nominal_model_validates(test_data_protons):
    pln, ct, cst, stf, _, _ = test_data_protons
    ct2, _ = _two_phase_ct(ct)
    pln.mult_scen = {"model": "nomScen", "ctScenProb": [(1, 0.5), (2, 0.5)]}

    engine = get_engine(pln)
    dij = engine.calc_dose_influence(ct2, cst, stf)

    assert engine._scenarios.ct_phases().tolist() == [0, 1]
    assert dij.physical_dose.shape == (2,)
    assert dij.alphax.shape[1] == 2
    _assert_differs(dij.physical_dose[0], dij.physical_dose[1])


def test_derived_scenarios_are_not_computed(case):
    pln, ct, cst, stf = case
    scen_set = _scenario_set(
        setup=[(0.0, 0.0, 0.0), SHIFT_X, (0.0, 0.0, 0.0)],
        bases=[None, None, ScenarioBasis(coefficients={0: 1.0})],
    )
    engine, dij, slots = _calc(pln, ct, cst, stf, scenarios=scen_set)

    assert dij.physical_dose.shape == (3,)
    assert slots[0] is not None and slots[1] is not None
    # the alias scenario is not computed: its slot shares the referenced matrix object
    assert slots[2] is slots[0]
    assert engine._scenarios.alias_of(2) == 0


def test_explicit_model_is_accepted(test_data_protons):
    pln, ct, cst, stf, _, _ = test_data_protons
    scen_set = _scenario_set(setup=[(0.0, 0.0, 0.0), SHIFT_X])
    pln.mult_scen = ExplicitScenarioModel.from_set(scen_set)

    engine = get_engine(pln)
    dij = engine.calc_dose_influence(ct, cst, stf)

    np.testing.assert_array_equal(engine._scenarios.values("setup"), scen_set.values("setup"))
    assert dij.physical_dose.shape == (2,)


def test_unsupported_dimension_is_refused(test_data_protons):
    pln, ct, cst, stf, _, _ = test_data_protons
    dims = (CTPhaseUncertainty(), SetupUncertainty(), RangeUncertainty(), _ScenarioTestDimension())

    perturbed = ScenarioSet.from_points(dims, {"scenario_test_dimension": [[0.0], [1.0]]})
    engine = get_engine(pln)
    with pytest.raises(ValueError, match="scenario_test_dimension"):
        engine.calc_dose_influence(ct, cst, stf, scenarios=perturbed)

    nominal_only = ScenarioSet.from_points(dims, {"scenario_test_dimension": [[0.0]]})
    engine.mult_scen = nominal_only
    assert engine._resolve_scenarios(ct) is nominal_only


def test_monte_carlo_engines_refuse_setup_and_range(test_data_protons):
    pln, ct, _, _, _, _ = test_data_protons
    assert ParticleFredMCEngine.supported_uncertainties == ("ct_phase",)

    engine = ParticleFredMCEngine(pln)
    engine.mult_scen = _scenario_set(setup=[(0.0, 0.0, 0.0), SHIFT_X])
    with pytest.raises(ValueError, match="setup"):
        engine._resolve_scenarios(ct)

    engine.mult_scen = _scenario_set(range_=[(0.0, 0.0), RANGE_ABS])
    with pytest.raises(ValueError, match="range"):
        engine._resolve_scenarios(ct)

    engine.mult_scen = "nomScen"
    assert len(engine._resolve_scenarios(ct)) == 1


def test_ct_phase_out_of_range_is_refused(test_data_protons):
    pln, ct, cst, stf, _, _ = test_data_protons
    scen_set = _scenario_set(ct_phase=[0, 1], phases=(0, 1))
    engine = get_engine(pln)
    with pytest.raises(ValueError, match="phase"):
        engine.calc_dose_influence(ct, cst, stf, scenarios=scen_set)


def test_public_entry_point_scenario_override(test_data_protons):
    pln, ct, cst, stf, _, _ = test_data_protons
    model_before = pln.mult_scen.model_dump()
    scen_set = _scenario_set(setup=[(0.0, 0.0, 0.0), SHIFT_X])

    dij = calc_dose_influence(ct, cst, stf, pln, scenarios=scen_set)

    assert dij.scenarios is scen_set
    assert dij.physical_dose.shape == (2,)
    assert pln.mult_scen.model_dump() == model_before


def test_forward_dose_refuses_several_scenarios(test_data_photons):
    """The forward result holds one dose; one of several scenarios would look nominal."""
    pln, ct, cst, stf, _, _ = test_data_photons
    scen_set = _scenario_set(setup=[(0.0, 0.0, 0.0), SHIFT_X])
    with pytest.raises(NotImplementedError, match="Forward dose calculation supports one"):
        calc_dose_forward(
            ct, cst, stf, pln, np.ones(stf.total_number_of_bixels), scenarios=scen_set
        )
    # a derived (alias) scenario does not count as a second computed one
    alias_set = _scenario_set(
        setup=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        bases=[None, ScenarioBasis(coefficients={0: 1.0})],
    )
    result = calc_dose_forward(
        ct, cst, stf, pln, np.ones(stf.total_number_of_bixels), scenarios=alias_set
    )
    assert "physical_dose" in result


BIO_SHIFTS = [(3.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, -3.0, 0.0), (2.0, 2.0, 0.0)]


def _bio_shift_dijs(test_data_protons):
    """MCN dij with four setup-shift scenarios and a nominal dij per shift on a shifted stf."""
    pln, ct, cst, stf, _, _ = test_data_protons
    pln.bio_model = "MCN"
    _, dij, _ = _calc(pln, ct, cst, stf, scenarios=_scenario_set(setup=BIO_SHIFTS))
    refs = [_calc(pln, ct, cst, _shifted(stf, shift))[1] for shift in BIO_SHIFTS]
    return dij, refs


def test_shift_scenarios_in_biological_results(test_data_protons):
    """Result assembly reads the reference alpha/beta of every shift scenario correctly."""
    dij, refs = _bio_shift_dijs(test_data_protons)
    w = np.ones(dij.total_num_of_bixels)
    for k, ref_dij in enumerate(refs):
        got = dij.get_result_arrays_from_intensity(w, scenario_index=k)
        ref = ref_dij.get_result_arrays_from_intensity(w)
        for name in ("physical_dose", "effect", "rbe_x_dose"):
            np.testing.assert_allclose(got[name], ref[name], rtol=1e-5, atol=1e-7)


def test_shift_scenarios_in_biological_quantities(test_data_protons):
    """RBExDose and the indirect AlphaDose / SqrtBetaDose paths per scenario."""
    from pyRadPlan.core.xp_utils import to_numpy
    from pyRadPlan.quantities import QuantityResolver

    dij, refs = _bio_shift_dijs(test_data_protons)
    w = np.ones(dij.total_num_of_bixels)

    def without_lq(d):
        return d.model_copy(update={"alpha_dose": None, "sqrt_beta_dose": None})

    for name, indirect in (("rbe_x_dose", False), ("alpha_dose", True), ("sqrt_beta_dose", True)):
        source = without_lq(dij) if indirect else dij
        quantity = QuantityResolver(source).get(name)
        xp = quantity.array_backend
        values = quantity.compute(xp.asarray(w, dtype=xp.float32))
        for k, ref_dij in enumerate(refs):
            ref_source = without_lq(ref_dij) if indirect else ref_dij
            ref_values = (
                QuantityResolver(ref_source).get(name).compute(xp.asarray(w, dtype=xp.float32))
            )
            np.testing.assert_allclose(
                to_numpy(values.flat[k]), to_numpy(ref_values.flat[0]), rtol=1e-4, atol=1e-6
            )
