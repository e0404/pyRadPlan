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
    # The object, not a dict: Plan validates a machine dict into the generic Machine
    pln.machine = machine
    return pln


def test_inititialize_beam_fluence_matrix_called_once_per_beam(test_data_xia, monkeypatch):
    """Building the delivery sequence must reuse the geometry cached by the weighted
    call from sequence() rather than recomputing it from scratch.
    """
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia

    calls = []
    original = XiaLeafSequencer._inititialize_beam_fluence_matrix

    def _counting_wrapper(self, beam, w_of_curr_beams=None):
        calls.append(w_of_curr_beams is not None)
        return original(self, beam, w_of_curr_beams)

    monkeypatch.setattr(XiaLeafSequencer, "_inititialize_beam_fluence_matrix", _counting_wrapper)

    run_sequencing(deepcopy(result_mat), pln_mat, stf_mat, dij=dij_mat)

    # Exactly one call per beam, always with weights (building the segments
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
    info_base = baseline.sequence.aperture_info(stf_mat)
    info_mlc = with_mlc.sequence.aperture_info(stf_mat)
    np.testing.assert_array_equal(info_base.to_vector(), info_mlc.to_vector())
    for bound_base, bound_mlc in zip(info_base.bounds(), info_mlc.bounds()):
        np.testing.assert_array_equal(bound_base, bound_mlc)
    for b_base, b_mlc in zip(baseline.sequence.beams, with_mlc.sequence.beams, strict=True):
        for s_base, s_mlc in zip(b_base.segments, b_mlc.segments, strict=True):
            np.testing.assert_array_equal(
                s_base.start.mlc.leaf_positions, s_mlc.start.mlc.leaf_positions
            )
            np.testing.assert_array_equal(
                s_base.start.mlc.leaf_position_boundaries,
                s_mlc.start.mlc.leaf_position_boundaries,
            )
            assert s_base.start.jaws == s_mlc.start.jaws


def test_coarser_leaf_grid_resamples_and_broadcasts(test_data_xia):
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia

    bixel_width = stf_mat.beams[0].bixel_width
    leaf_width = 2.3 * bixel_width  # deliberately not an integer multiple
    pln_mlc = _pln_with_mlc(pln_mat, leaf_width=leaf_width, number_of_elements=40)

    baseline = run_sequencing(deepcopy(result_mat), pln_mat, stf_mat, dij=dij_mat)
    result_seq = run_sequencing(deepcopy(result_mat), pln_mlc, stf_mat, dij=dij_mat)

    first_mlc = result_seq.sequence.beams[0].segments[0].start.mlc
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

    # Leaf rows wider than the bixels: the area overlap of the apertures with the bixels
    # distributes the weight differently than the row interpolation of the sequencer, but
    # delivers the same in total.
    info = result_seq.sequence.aperture_info(stf_mat)
    for grid in info.grids:
        assert np.isclose(grid.leaf_width, leaf_width)
    np.testing.assert_allclose(info.bixel_weights().sum(), fluence.sum())


def test_full_leaf_positions_closed_elsewhere(test_data_xia):
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia

    bixel_width = stf_mat.beams[0].bixel_width
    pln_mlc = _pln_with_mlc(pln_mat, leaf_width=2 * bixel_width, number_of_elements=60)
    result_seq = run_sequencing(deepcopy(result_mat), pln_mlc, stf_mat, dij=dij_mat)

    mlc = result_seq.sequence.beams[0].segments[0].start.mlc
    central_leaf_pair = XiaLeafSequencer(pln_mlc)._central_leaf_pair(bixel_width)
    assert central_leaf_pair == 30.5

    full = mlc.full_leaf_positions(central_leaf_pair)
    assert full.shape == (60, 2)

    is_active = mlc.is_active_leaf_pair(central_leaf_pair).astype(bool)
    assert np.count_nonzero(is_active) == len(mlc.leaf_positions)
    np.testing.assert_array_equal(full[is_active], mlc.leaf_positions)
    np.testing.assert_array_equal(full[~is_active], np.zeros((np.count_nonzero(~is_active), 2)))


def test_jaw_pair_matches_field_window(test_data_xia):
    """Each segment gets a static X/Y jaw pair bounding the beam's field window."""
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia
    result_seq = run_sequencing(deepcopy(result_mat), pln_mat, stf_mat, dij=dij_mat)

    for beam, stf_beam in zip(result_seq.sequence.beams, stf_mat.beams, strict=True):
        assert beam.num_of_segments > 0

        ray_pos = np.array([ray.ray_pos_bev for ray in stf_beam.rays]).reshape(-1, 3)
        half = 0.5 * stf_beam.bixel_width
        jaw_by_orientation = {jaw.device_orientation: jaw for jaw in beam.segments[0].start.jaws}
        assert set(jaw_by_orientation) == {"X", "Y"}
        for jaw in jaw_by_orientation.values():
            assert isinstance(jaw, Jaw)
        # test_data_xia has no configured machine MLC, so it uses the
        # default MLCX orientation: travel axis == BEV "X".
        np.testing.assert_allclose(
            jaw_by_orientation["X"].positions,
            [ray_pos[:, 0].min() - half, ray_pos[:, 0].max() + half],
        )
        np.testing.assert_allclose(
            jaw_by_orientation["Y"].positions,
            [ray_pos[:, 2].min() - half, ray_pos[:, 2].max() + half],
        )

        # jaws are static: identical across every segment of the beam
        for segment in beam.segments[1:]:
            assert len(segment.start.jaws) == len(jaw_by_orientation)
            for jaw in segment.start.jaws:
                reference = jaw_by_orientation[jaw.device_orientation]
                assert jaw.positions == reference.positions


@pytest.mark.parametrize("device_type", ["MLCX", "MLCY"])
def test_jaws_reach_over_leaf_rows_wider_than_the_field(test_data_xia, device_type):
    """Across the leaves the jaws open to the leaf rows in use, so they never clip a leaf."""
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia
    stf_beam = stf_mat.beams[0]
    bixel_width = stf_beam.bixel_width
    leaf_width = 2.3 * bixel_width
    pln_mlc = _pln_with_mlc(
        pln_mat, leaf_width=leaf_width, number_of_elements=40, device_type=device_type
    )
    result_seq = run_sequencing(deepcopy(result_mat), pln_mlc, stf_mat, dij=dij_mat)

    travel_axis, stack_axis = (0, 2) if device_type == "MLCX" else (2, 0)
    ray_pos = np.array([ray.ray_pos_bev for ray in stf_beam.rays]).reshape(-1, 3)
    half = 0.5 * bixel_width
    for segment in result_seq.sequence.beams[0].segments:
        mlc = segment.start.mlc
        jaws = {jaw.device_orientation: jaw for jaw in segment.start.jaws}
        travel_jaw, stack_jaw = (
            (jaws["X"], jaws["Y"]) if device_type == "MLCX" else (jaws["Y"], jaws["X"])
        )
        np.testing.assert_allclose(
            travel_jaw.positions,
            [ray_pos[:, travel_axis].min() - half, ray_pos[:, travel_axis].max() + half],
        )
        np.testing.assert_allclose(
            stack_jaw.positions,
            [mlc.leaf_position_boundaries[0], mlc.leaf_position_boundaries[-1] + leaf_width],
        )
        assert stack_jaw.positions[0] <= ray_pos[:, stack_axis].min() - half
        assert stack_jaw.positions[1] >= ray_pos[:, stack_axis].max() + half


def test_leaf_grid_centers_are_evenly_spaced_and_cover_the_range():
    """The MLC grid is a fixed lattice covering every part of (field_min, field_max)."""
    sequencer = XiaLeafSequencer()
    leaf_width = 10.0

    # Odd leaf count: a leaf center on isocenter.
    leaf_row_pos = sequencer._leaf_grid_centers(-22.0, 23.0, leaf_width, num_of_leaf_pairs=41)
    np.testing.assert_allclose(leaf_row_pos, [-20.0, -10.0, 0.0, 10.0, 20.0])

    # Even leaf count: a leaf boundary on isocenter.
    leaf_row_pos = sequencer._leaf_grid_centers(-22.0, 23.0, leaf_width, num_of_leaf_pairs=40)
    np.testing.assert_allclose(leaf_row_pos, [-25.0, -15.0, -5.0, 5.0, 15.0, 25.0])
    np.testing.assert_allclose(np.diff(leaf_row_pos), leaf_width)

    # A field edge exactly on a leaf boundary does not pull in the next row.
    leaf_row_pos = sequencer._leaf_grid_centers(-20.0, 20.0, leaf_width, num_of_leaf_pairs=40)
    np.testing.assert_allclose(leaf_row_pos, [-15.0, -5.0, 5.0, 15.0])


def test_symmetric_field_gets_symmetric_leaf_rows(test_data_xia):
    """A field symmetric about isocenter must stay symmetric on the leaf grid."""
    pln_mat, *_ = test_data_xia
    sequencer = XiaLeafSequencer(_pln_with_mlc(pln_mat, leaf_width=10.0, number_of_elements=40))

    leaf_row_pos = sequencer._leaf_grid_centers(-37.5, 37.5, 10.0, num_of_leaf_pairs=40)
    # leaf rows mirrored around 0 and span full field [-40, 40]
    np.testing.assert_allclose(leaf_row_pos, -leaf_row_pos[::-1])
    np.testing.assert_allclose([leaf_row_pos[0] - 5.0, leaf_row_pos[-1] + 5.0], [-40.0, 40.0])

    assert sequencer._central_leaf_pair(bixel_width=5.0) == 20.5
    mlc = MLC(
        device_type="MLCX",
        number_of_elements=40,
        leaf_width=10.0,
        leaf_position_boundaries=leaf_row_pos - 5.0,
        leaf_positions=np.tile([-5.0, 5.0], (len(leaf_row_pos), 1)),
    )
    # Rows straddle the boundary between pairs 20 and 21 (1-based), i.e. 0-based 16..23.
    np.testing.assert_array_equal(np.flatnonzero(mlc.is_active_leaf_pair(20.5)), np.arange(16, 24))


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
        native_row_z.min(), native_row_z.max(), leaf_width=12.0, num_of_leaf_pairs=40
    )
    matrix = native_row_z[:, None] * 2.0  # linear in z, one column

    downsampled = sequencer._interp_rows(matrix, native_row_z, leaf_row_pos)
    upsampled = sequencer._interp_rows(downsampled, leaf_row_pos, native_row_z)

    # Interior points (away from the edges, where clamping can bite) round-trip exactly.
    # The outermost leaf rows may reach past the native rows, so they are clamped too.
    interior = (native_row_z > leaf_row_pos[1]) & (native_row_z < leaf_row_pos[-2])
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

    mlc_y = result_seq_y.sequence.beams[0].segments[0].start.mlc
    assert mlc_y.device_type == "MLCY"
    assert mlc_y.device_orientation == "Y"

    # The beamlet grid follows the MLC: travel is BEV x for an MLCX and BEV z for an MLCY.
    grid_x = result_seq_x.sequence.aperture_info(stf_mat).grids[0]
    grid_y = result_seq_y.sequence.aperture_info(stf_mat).grids[0]
    assert (grid_x.orientation, grid_y.orientation) == ("X", "Y")
    ray_pos = np.array([ray.ray_pos_bev for ray in beam.rays]).reshape(-1, 3)
    np.testing.assert_allclose(grid_x.travel, ray_pos[:, 0])
    np.testing.assert_allclose(grid_x.stack, ray_pos[:, 2])
    np.testing.assert_allclose(grid_y.travel, ray_pos[:, 2])
    np.testing.assert_allclose(grid_y.stack, ray_pos[:, 0])
    _, travel_min_x, stack_min_x = grid_x.bixel_index_map()
    _, travel_min_y, stack_min_y = grid_y.bixel_index_map()
    np.testing.assert_allclose([travel_min_x, stack_min_x], [stack_min_y, travel_min_y])

    # The jaws stay on their BEV axes whichever of them the leaves travel along.
    for jaw_x, jaw_y in zip(
        result_seq_x.sequence.beams[0].segments[0].start.jaws,
        result_seq_y.sequence.beams[0].segments[0].start.jaws,
        strict=True,
    ):
        assert jaw_x.device_orientation == jaw_y.device_orientation
        np.testing.assert_allclose(jaw_x.positions, jaw_y.positions)


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


def test_sequence_reports_the_leaf_pairs_of_the_machine(test_data_xia):
    pln_mat, stf_mat, dij_mat, *_, result_mat, _ = test_data_xia
    bixel_width = stf_mat.beams[0].bixel_width
    pln_mlc = _pln_with_mlc(pln_mat, leaf_width=bixel_width, number_of_elements=40)

    result_seq = run_sequencing(deepcopy(result_mat), pln_mlc, stf_mat, dij=dij_mat)

    for beam in result_seq.sequence.beams:
        for segment in beam.segments:
            assert segment.start.mlc.number_of_elements == 40
