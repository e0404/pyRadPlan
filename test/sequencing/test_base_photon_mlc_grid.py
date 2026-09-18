"""Tests for physical-MLC-grid restrictions in photon sequencing."""

from copy import deepcopy

import numpy as np
import pytest

from pyRadPlan.machines import MLC, Jaw, load_from_name
from pyRadPlan.sequencing import run_sequencing
from pyRadPlan.sequencing.sequencer._xia_leaf_sequencing import XiaLeafSequencer


def _pln_with_mlc(pln, leaf_width: float, number_of_elements: int, device_type: str = "MLCX"):
    """Return a deep copy of pln whose machine carries the given MLC spec."""
    pln = deepcopy(pln)
    machine = load_from_name(pln.radiation_mode, pln.machine)
    machine.mlc = MLC(
        device_type=device_type, leaf_width=leaf_width, number_of_elements=number_of_elements
    )
    pln.machine = machine.model_dump(by_alias=True)
    return pln


def test_inititialize_beam_fluence_matrix_called_once_per_beam(test_data_xia, monkeypatch):
    """_compute_beam_aperture_info must reuse the geometry cached by the earlier,
    weighted call from sequence() rather than recomputing it from scratch.
    """
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia

    calls = []
    original = XiaLeafSequencer._inititialize_beam_fluence_matrix

    def _counting_wrapper(self, beam, w_of_curr_beams=None):
        calls.append(w_of_curr_beams is not None)
        return original(self, beam, w_of_curr_beams)

    monkeypatch.setattr(XiaLeafSequencer, "_inititialize_beam_fluence_matrix", _counting_wrapper)

    run_sequencing(deepcopy(result_mat), pln_mat, stf_mat, dij=dij_mat)

    # Exactly one call per beam, always with weights (the aperture-info pass
    # must not trigger a second, weight-less call for the same beam).
    assert len(calls) == len(stf_mat.beams)
    assert all(calls)


def test_leaf_width_matching_bixel_width_is_a_noop(test_data_xia):
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia

    baseline = run_sequencing(deepcopy(result_mat), pln_mat, stf_mat, dij=dij_mat)

    bixel_width = stf_mat.beams[0].bixel_width
    pln_mlc = _pln_with_mlc(pln_mat, leaf_width=bixel_width, number_of_elements=80)
    with_mlc = run_sequencing(deepcopy(result_mat), pln_mlc, stf_mat, dij=dij_mat)

    np.testing.assert_array_equal(baseline.sequence.fluence, with_mlc.sequence.fluence)
    np.testing.assert_array_equal(
        baseline.sequence.aperture_vector, with_mlc.sequence.aperture_vector
    )
    for b_base, b_mlc in zip(baseline.sequence.beam, with_mlc.sequence.beam):
        np.testing.assert_array_equal(b_base.shapes, b_mlc.shapes)
        np.testing.assert_array_equal(b_base.leaf_pair_pos, b_mlc.leaf_pair_pos)


def test_coarser_leaf_grid_resamples_and_broadcasts(test_data_xia):
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia

    bixel_width = stf_mat.beams[0].bixel_width
    leaf_width = 2.3 * bixel_width  # deliberately not an integer multiple
    pln_mlc = _pln_with_mlc(pln_mat, leaf_width=leaf_width, number_of_elements=40)

    baseline = run_sequencing(deepcopy(result_mat), pln_mat, stf_mat, dij=dij_mat)
    result_seq = run_sequencing(deepcopy(result_mat), pln_mlc, stf_mat, dij=dij_mat)

    first_beam = result_seq.sequence.beam[0]
    first_mlc = first_beam.mlc_sequence[0]
    assert np.isclose(first_mlc.leaf_width, leaf_width)
    assert first_mlc.number_of_elements == 40

    # The leaf grid is evenly spaced by leaf_width.
    np.testing.assert_allclose(np.diff(first_mlc.leaf_position_boundaries), leaf_width)

    # Interpolated fluence must stay finite and within the physical range of
    # the original (unsequenced) beam fluence -- no interpolation overshoot.
    fluence = result_seq.sequence.fluence
    assert np.all(np.isfinite(fluence))
    assert fluence.min() >= 0.0
    assert fluence.max() <= baseline.sequence.fluence.max() * 1.01


def test_full_leaf_positions_closed_elsewhere(test_data_xia):
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia

    bixel_width = stf_mat.beams[0].bixel_width
    pln_mlc = _pln_with_mlc(pln_mat, leaf_width=2 * bixel_width, number_of_elements=60)
    result_seq = run_sequencing(deepcopy(result_mat), pln_mlc, stf_mat, dij=dij_mat)

    beam = result_seq.sequence.beam[0]
    mlc = beam.mlc_sequence[0]

    full = mlc.full_leaf_positions(beam.central_leaf_pair)
    assert full.shape == (60, 2)

    is_active = mlc.is_active_leaf_pair(beam.central_leaf_pair).astype(bool)
    assert np.count_nonzero(is_active) == len(mlc.leaf_positions)
    np.testing.assert_array_equal(full[is_active], mlc.leaf_positions)
    np.testing.assert_array_equal(full[~is_active], np.zeros((np.count_nonzero(~is_active), 2)))


def test_extra_blds_jaw_pair_matches_field_window(test_data_xia):
    """Each shape gets a static X/Y jaw pair bounding the beam's field window."""
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia
    result_seq = run_sequencing(deepcopy(result_mat), pln_mat, stf_mat, dij=dij_mat)

    for beam in result_seq.sequence.beam:
        assert beam.extra_blds is not None
        assert len(beam.extra_blds) == beam.num_of_shapes

        travel_min, travel_max, stack_min, stack_max = beam.mlc_window
        jaw_by_orientation = {jaw.device_orientation: jaw for jaw in beam.extra_blds[0]}
        assert set(jaw_by_orientation) == {"X", "Y"}
        for jaw in jaw_by_orientation.values():
            assert isinstance(jaw, Jaw)
        # test_data_xia has no configured machine MLC, so it uses the
        # default MLCX orientation: travel axis == BEV "X".
        np.testing.assert_allclose(jaw_by_orientation["X"].positions, [travel_min, travel_max])
        np.testing.assert_allclose(jaw_by_orientation["Y"].positions, [stack_min, stack_max])

        # jaws are static: identical across every shape of the beam
        for shape_jaws in beam.extra_blds[1:]:
            for jaw in shape_jaws:
                reference = jaw_by_orientation[jaw.device_orientation]
                assert jaw.positions == reference.positions


def test_leaf_grid_centers_are_evenly_spaced_and_cover_the_range():
    """The MLC grid is a fixed, isocenter-centered lattice covering [min_stack, max_stack]."""
    sequencer = XiaLeafSequencer()
    leaf_width = 10.0

    # -22 is nearest to leaf center -20, 23 is nearest to leaf center 20.
    leaf_row_pos = sequencer._leaf_grid_centers(
        min_stack=-22.0, max_stack=23.0, leaf_width=leaf_width
    )

    np.testing.assert_allclose(leaf_row_pos, [-20.0, -10.0, 0.0, 10.0, 20.0])
    np.testing.assert_allclose(np.diff(leaf_row_pos), leaf_width)


def test_interp_rows_reproduces_a_linear_ramp_exactly():
    """Linear interpolation of a linear function must reproduce it exactly.

    This is the core operation _leaf_grid_centers's grid gets resampled with:
    downsampling native rows onto the (coarser) leaf grid, and projecting a
    sequenced quantity back onto the (finer) native grid, are the same
    _interp_rows call with src/dst swapped.
    """
    sequencer = XiaLeafSequencer()
    src_row_pos = np.arange(-40.0, 41.0, 5.0)
    dst_row_pos = np.array([-37.5, -12.3, 0.0, 8.1, 33.0])

    # Two columns: a linear ramp and a constant, exercising per-column interpolation.
    matrix = np.stack([src_row_pos, np.full_like(src_row_pos, 3.0)], axis=1)

    resampled = sequencer._interp_rows(matrix, src_row_pos, dst_row_pos)

    np.testing.assert_allclose(resampled[:, 0], dst_row_pos)
    np.testing.assert_allclose(resampled[:, 1], 3.0)


def test_interp_rows_round_trip_recovers_source_points():
    """Downsampling then re-upsampling at the original source points is a no-op
    for a linear signal (no interpolation error accumulates for straight lines).
    """
    sequencer = XiaLeafSequencer()
    native_row_z = np.arange(-40.0, 41.0, 5.0)
    leaf_row_pos = sequencer._leaf_grid_centers(
        min_stack=native_row_z.min(), max_stack=native_row_z.max(), leaf_width=12.0
    )
    matrix = native_row_z[:, None] * 2.0  # linear in z, one column

    downsampled = sequencer._interp_rows(matrix, native_row_z, leaf_row_pos)
    upsampled = sequencer._interp_rows(downsampled, leaf_row_pos, native_row_z)

    # Interior points (away from the edges, where clamping can bite) round-trip exactly.
    interior = (native_row_z > leaf_row_pos.min()) & (native_row_z < leaf_row_pos.max())
    np.testing.assert_allclose(upsampled[interior, 0], matrix[interior, 0])


def test_mlcy_orientation_swaps_travel_and_stack_axes(test_data_xia):
    """An MLCY reads leaf travel/stack from the opposite ray_pos_bev axes than an MLCX."""
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia
    bixel_width = stf_mat.beams[0].bixel_width

    pln_x = _pln_with_mlc(
        pln_mat, leaf_width=bixel_width, number_of_elements=80, device_type="MLCX"
    )
    pln_y = _pln_with_mlc(
        pln_mat, leaf_width=bixel_width, number_of_elements=80, device_type="MLCY"
    )

    seq_x = XiaLeafSequencer(pln_x)
    seq_y = XiaLeafSequencer(pln_y)
    assert seq_x.mlc_orientation == "X"
    assert seq_y.mlc_orientation == "Y"

    beam = stf_mat.beams[0]
    init_x = seq_x._inititialize_beam_fluence_matrix(beam)
    init_y = seq_y._inititialize_beam_fluence_matrix(beam)

    # Swapping orientation swaps which ray_pos_bev axis plays which role.
    assert init_x["dim_travel"] == init_y["dim_stack"]
    assert init_x["dim_stack"] == init_y["dim_travel"]
    np.testing.assert_allclose(init_x["min_travel"], init_y["min_stack"])
    np.testing.assert_allclose(init_x["min_stack"], init_y["min_travel"])

    result_seq_x = run_sequencing(deepcopy(result_mat), pln_x, stf_mat, dij=dij_mat)
    result_seq_y = run_sequencing(deepcopy(result_mat), pln_y, stf_mat, dij=dij_mat)

    mlc_y = result_seq_y.sequence.beam[0].mlc_sequence[0]
    assert mlc_y.device_type == "MLCY"
    assert mlc_y.device_orientation == "Y"

    # pos_of_corner_bixel places the travel/stack min at its true BEV index
    # (0 for X-orientation travel, 2 for Y-orientation travel).
    corner_x = result_seq_x.sequence.beam[0].pos_of_corner_bixel
    corner_y = result_seq_y.sequence.beam[0].pos_of_corner_bixel
    np.testing.assert_allclose(corner_x[0], corner_y[2])
    np.testing.assert_allclose(corner_x[2], corner_y[0])


def test_mlcy_resamples_the_stack_axis_not_the_travel_axis(test_data_xia):
    """With resampling actually active, an MLCY must coarsen ray_pos_bev[0]
    (its stack axis), not ray_pos_bev[2] (its travel axis, kept at native
    ray resolution) -- the opposite of what an MLCX coarsens.
    """
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia
    bixel_width = stf_mat.beams[0].bixel_width
    leaf_width = 2 * bixel_width
    beam = stf_mat.beams[0]

    # No MLC configured: pure native-resolution grid, for both axes.
    init_native = XiaLeafSequencer()._inititialize_beam_fluence_matrix(beam)

    pln_y = _pln_with_mlc(
        pln_mat, leaf_width=leaf_width, number_of_elements=40, device_type="MLCY"
    )
    seq_y = XiaLeafSequencer(pln_y)
    init_y = seq_y._inititialize_beam_fluence_matrix(beam)

    # Travel axis (columns) is ray_pos_bev[2] for an MLCY -- the same data as
    # the native grid's stack axis (rows) -- and must stay at native
    # resolution (unresampled), so the counts match exactly.
    assert init_y["dim_travel"] == init_native["dim_stack"]

    # Stack axis (rows) is ray_pos_bev[0] for an MLCY -- the same data as the
    # native grid's travel axis (columns) -- and must be coarsened onto the
    # (wider) leaf grid, so strictly fewer rows than the native count.
    assert init_y["dim_stack"] < init_native["dim_travel"]


def test_leaf_width_narrower_than_bixel_width_raises(test_data_xia):
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia

    bixel_width = stf_mat.beams[0].bixel_width
    pln_mlc = _pln_with_mlc(pln_mat, leaf_width=0.5 * bixel_width, number_of_elements=200)

    sequencer = XiaLeafSequencer(pln_mlc)
    with pytest.raises(ValueError):
        sequencer.sequence(deepcopy(result_mat), stf_mat, dij_mat)


def test_field_taller_than_mlc_leaf_bank_raises(test_data_xia):
    """A field whose height needs more leaf rows than the machine's MLC has must
    raise a clear error, rather than silently overflowing and crashing later
    (e.g. in MLC.full_leaf_positions/is_active_leaf_pair) with a shape-mismatch
    once the leaf-row count and the beam's central_leaf_pair disagree.
    """
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia

    bixel_width = stf_mat.beams[0].bixel_width
    # Deliberately too few leaf pairs (just one) to cover the beam's native
    # field height at this leaf width (leaf_width > bixel_width so
    # resampling -- and the check -- actually runs; leaf_width == bixel_width
    # is a no-resample no-op).
    pln_mlc = _pln_with_mlc(pln_mat, leaf_width=1.5 * bixel_width, number_of_elements=1)

    sequencer = XiaLeafSequencer(pln_mlc)
    with pytest.raises(ValueError, match="does not fit within the physical collimator"):
        sequencer.sequence(deepcopy(result_mat), stf_mat, dij_mat)
