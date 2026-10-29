"""Particle pencil-beam engines: per-beam lateral cutoffs."""

from copy import deepcopy

import numpy as np

from pyRadPlan.dose import calc_dose_influence


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
