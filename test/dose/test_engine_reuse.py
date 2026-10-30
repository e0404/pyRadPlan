"""Dose engines keep no state from one calculation to the next and leave their inputs alone."""

import numpy as np
import SimpleITK as sitk

from pyRadPlan.dose.engines import get_engine


def _dose(result):
    return sitk.GetArrayFromImage(result["physical_dose"])


def test_reused_engine_repeats_forward_dose(test_data_photons):
    pln, ct, cst, stf, _, _ = test_data_photons
    w = np.ones(stf.total_number_of_bixels)
    engine = get_engine(pln)

    first = _dose(engine.calc_dose_forward(ct, cst, stf, w))
    second = _dose(engine.calc_dose_forward(ct, cst, stf, w))
    np.testing.assert_array_equal(second, first)

    # An influence calculation after forward ones still works on the same engine
    dij = engine.calc_dose_influence(ct, cst, stf)
    np.testing.assert_allclose(_dose(dij.compute_result_ct_grid(w)), first, rtol=1e-5, atol=1e-7)


def test_reused_engine_uses_the_structures_of_each_call(test_data_photons):
    pln, ct, cst, stf, _, _ = test_data_photons
    w = np.ones(stf.total_number_of_bixels)
    targets = cst.model_copy(update={"vois": [v for v in cst.vois if v.voi_type == "TARGET"]})

    engine = get_engine(pln)
    engine.calc_dose_forward(ct, targets, stf, w)
    reused = _dose(engine.calc_dose_forward(ct, cst, stf, w))
    fresh = _dose(get_engine(pln).calc_dose_forward(ct, cst, stf, w))
    np.testing.assert_array_equal(reused, fresh)

    # The user's settings are left as they were
    assert engine.voxel_sub_ix is None


def test_forward_dose_leaves_the_stf_weights_alone(test_data_photons):
    pln, ct, cst, stf, _, _ = test_data_photons
    before = [b.weight for beam in stf.beams for ray in beam.rays for b in ray.beamlets]
    w = np.linspace(0.1, 1.0, stf.total_number_of_bixels)

    get_engine(pln).calc_dose_forward(ct, cst, stf, w)

    after = [b.weight for beam in stf.beams for ray in beam.rays for b in ray.beamlets]
    assert after == before


def test_keep_rad_depth_cubes(test_data_photons):
    pln, ct, cst, stf, _, _ = test_data_photons
    pln.prop_dose_calc["keep_rad_depth_cubes"] = True
    engine = get_engine(pln)

    for _ in range(2):
        dij = engine.calc_dose_influence(ct, cst, stf)
        assert len(dij.rad_depth_cubes) == len(stf.beams)
        grid = dij.dose_grid.dimensions
        for cube in dij.rad_depth_cubes:
            assert tuple(cube.shape) == (grid[2], grid[1], grid[0])


def test_reused_engine_resolves_the_bio_model_per_calculation(test_data_protons):
    pln, ct, cst, stf, _, _ = test_data_protons
    pln.bio_model = "MCN"
    no_beta = cst.model_copy(deep=True)
    for voi in no_beta.vois:
        voi.beta_x = 0.0

    # McNamara needs beta_x > 0, so the first calculation drops the model ...
    engine = get_engine(pln)
    assert engine.calc_dose_influence(ct, no_beta, stf).bio_model is None

    # ... which must not carry over to a calculation whose tissues fit
    reused = engine.calc_dose_influence(ct, cst, stf)
    fresh = get_engine(pln).calc_dose_influence(ct, cst, stf)
    assert type(reused.bio_model) is type(fresh.bio_model)
    assert reused.bio_model is not None


def test_reused_engine_resolves_the_lateral_model_per_calculation(test_data_protons):
    # "auto" resolves to the most accurate model the machine supports, for each calculation
    pln, ct, cst, stf, _, _ = test_data_protons
    pln.prop_dose_calc["lateral_model"] = "auto"
    engine = get_engine(pln)

    engine.calc_dose_influence(ct, cst, stf)
    assert engine.lateral_model == "auto"
