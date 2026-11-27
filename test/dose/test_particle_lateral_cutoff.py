"""Particle pencil-beam engines: per-beam lateral cutoffs."""

from copy import deepcopy

import numpy as np
import pytest

from pyRadPlan.dose import calc_dose_influence
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine
from pyRadPlan.scenarios import (
    CTPhaseUncertainty,
    RangeUncertainty,
    ScenarioSet,
    SetupUncertainty,
)

SAMPLED = "relative_tail_sampling"


def _shifted_stf(stf, shift):
    shifted = deepcopy(stf)
    for beam in shifted.beams:
        beam.iso_center = np.asarray(beam.iso_center, dtype=float) + shift
    return shifted


def test_particle_beams_use_their_own_lateral_cutoffs(test_data_protons):
    # Lateral cutoffs are recomputed per beam from its SSDs; a later beam must not reuse
    # kernels cached with an earlier beam's cutoffs. Off-center, the beams' SSDs differ.
    pln, ct, cst, stf, _, _ = test_data_protons
    assert len(stf.beams) > 1
    stf = _shifted_stf(stf, np.array([0.0, -3.0, 0.0]))
    dij = calc_dose_influence(ct, cst, stf, pln)

    last = len(stf.beams) - 1
    alone = deepcopy(stf)
    alone.beams = [alone.beams[last]]
    dij_alone = calc_dose_influence(ct, cst, alone, deepcopy(pln))

    columns = np.asarray(dij.beam_num) == last
    got = dij.physical_dose.flat[0].toarray()[:, columns]
    np.testing.assert_array_equal(got, dij_alone.physical_dose.flat[0].toarray())


def _total_dose(ct, cst, stf, pln, scenarios=None, **settings):
    pln = deepcopy(pln)
    pln.prop_dose_calc.update(settings)
    dij = calc_dose_influence(ct, cst, stf, pln, scenarios=scenarios)
    return [np.asarray(matrix.sum(axis=1)).ravel() for matrix in dij.physical_dose.flat], dij


@pytest.mark.parametrize("model", ["single", "double"])
@pytest.mark.parametrize("method", ["integral", "relative"])
def test_tail_sampling_preserves_the_dose_without_compensation(test_data_protons, model, method):
    pln, ct, cst, stf, _, _ = test_data_protons
    settings = dict(lateral_model=model)
    (reference,), dij_reference = _total_dose(
        ct, cst, stf, pln, dosimetric_lateral_cutoff=0.9999, cut_off_method=method, **settings
    )
    (sampled,), dij = _total_dose(
        ct,
        cst,
        stf,
        pln,
        dosimetric_lateral_cutoff=0.9,
        cut_off_method=f"{method}_tail_sampling",
        **settings,
    )

    assert dij.physical_dose.flat[0].nnz < 0.7 * dij_reference.physical_dose.flat[0].nnz
    assert sampled.sum() == pytest.approx(reference.sum(), rel=5e-3)
    high_dose = reference > 0.5 * reference.max()
    assert sampled[high_dose].mean() == pytest.approx(reference[high_dose].mean(), rel=1e-2)


def test_tail_sampling_is_reproducible_through_its_seed(test_data_protons):
    pln, ct, cst, stf, _, _ = test_data_protons
    stf = deepcopy(stf)
    stf.beams = stf.beams[:1]
    doses = [
        _total_dose(ct, cst, stf, pln, cut_off_method=SAMPLED, tail_sampling_seed=seed)[0][0]
        for seed in (0, 0, 1)
    ]
    np.testing.assert_array_equal(doses[0], doses[1])
    assert np.any(doses[0] != doses[2])


def test_tail_sampling_covers_range_scenarios(test_data_protons):
    pln, ct, cst, stf, _, _ = test_data_protons
    stf = deepcopy(stf)
    stf.beams = stf.beams[:1]
    dims = (
        CTPhaseUncertainty(phases=[0], probabilities=[1.0]),
        SetupUncertainty(),
        RangeUncertainty(),
    )
    scenarios = ScenarioSet.from_points(
        dims, {"range": np.asarray([(0.0, 0.0), (8.0, 0.05), (-8.0, -0.05)])}
    )
    reference, _ = _total_dose(ct, cst, stf, pln, scenarios, dosimetric_lateral_cutoff=0.9999)
    sampled, _ = _total_dose(ct, cst, stf, pln, scenarios, cut_off_method=SAMPLED)
    for expected, actual in zip(reference, sampled):
        assert actual.sum() == pytest.approx(expected.sum(), rel=5e-3)
        high_dose = expected > 0.5 * expected.max()
        np.testing.assert_allclose(actual[high_dose].mean(), expected[high_dose].mean(), rtol=5e-3)


@pytest.mark.parametrize("case, bio_model", [("protons", "MCN"), ("carbon", "kernel_based_lq")])
def test_tail_sampling_weights_let_and_bio_quantities_like_the_dose(request, case, bio_model):
    pln, ct, cst, stf, _, _ = request.getfixturevalue(f"test_data_{case}")
    pln = deepcopy(pln)
    pln.bio_model = bio_model
    stf = deepcopy(stf)
    stf.beams = stf.beams[:1]

    def calc(**settings):
        plan = deepcopy(pln)
        plan.prop_dose_calc.update(settings)
        return calc_dose_influence(ct, cst, stf, plan)

    reference = calc(dosimetric_lateral_cutoff=0.9999)
    sampled = calc(dosimetric_lateral_cutoff=0.9, cut_off_method=SAMPLED)
    dose_reference = reference.physical_dose.flat[0].tocsc()
    dose = sampled.physical_dose.flat[0].tocsc()
    assert dose.nnz < dose_reference.nnz
    rows, columns = dose.nonzero()
    weights = (
        np.asarray(dose[rows, columns]).ravel() / np.asarray(dose_reference[rows, columns]).ravel()
    )
    assert weights.max() > 2

    quantities = ["alpha_dose", "sqrt_beta_dose"]
    if reference.let_dose is not None:
        quantities.append("let_dose")
    for name in quantities:
        expected = getattr(reference, name).flat[0].tocsc()
        actual = getattr(sampled, name).flat[0].tocsc()
        assert actual.nnz == dose.nnz
        # Every stored entry carries the weight of its dose entry
        np.testing.assert_allclose(
            np.asarray(actual[rows, columns]).ravel(),
            weights * np.asarray(expected[rows, columns]).ravel(),
            rtol=2e-4,
        )
        assert actual.sum() == pytest.approx(expected.sum(), rel=1e-2)


def test_tail_sampling_without_cut_off_computes_everything(test_data_protons):
    pln, ct, cst, stf, _, _ = test_data_protons
    stf = deepcopy(stf)
    stf.beams = stf.beams[:1]
    (full,), _ = _total_dose(ct, cst, stf, pln, dosimetric_lateral_cutoff=1.0)
    (sampled,), _ = _total_dose(
        ct, cst, stf, pln, dosimetric_lateral_cutoff=1.0, cut_off_method=SAMPLED
    )
    np.testing.assert_array_equal(sampled, full)


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, 0.995),
        ({"cut_off_method": "relative"}, 0.995),
        ({"cut_off_method": "integral_tail_sampling"}, 0.98),
        ({"cut_off_method": "relative_tail_sampling"}, 0.98),
        ({"cut_off_method": "relative_tail_sampling", "dosimetric_lateral_cutoff": 0.9}, 0.9),
        ({"dosimetric_lateral_cutoff": 0.99}, 0.99),
    ],
)
def test_default_cut_off_level_follows_the_method(test_data_protons, settings, expected):
    pln = deepcopy(test_data_protons[0])
    del pln.prop_dose_calc["dosimetric_lateral_cutoff"]
    pln.prop_dose_calc.update(settings)
    engine = ParticleHongPencilBeamEngine(pln)
    assert engine._lateral_cutoff_level == expected
    assert engine._samples_lateral_tail == ("tail_sampling" in engine.cut_off_method)
