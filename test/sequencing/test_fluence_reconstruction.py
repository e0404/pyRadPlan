import numpy as np
import pytest

from pyRadPlan.plan import PhotonPlan
from pyRadPlan.result import PlanResult
from pyRadPlan.sequencing import run_sequencing
from pyRadPlan.sequencing.sequencer import get_available_sequencers
from pyRadPlan.stf import Beam, Beamlet, Ray, SteeringInformation

#: Intensity levels the fluence is stratified onto.
NUM_OF_LEVELS = 5

#: Peak of every input map, and the flat value of the constant one.
CONSTANT = 2.0

#: Side length (in rays) of the square field the constant and ramp maps use. Each
#: test is free to pick another shape -- the field is built to fit its input map.
FIELD_SIDE = 4

#: Ray spacing of that field (mm).
BIXEL_WIDTH = 10.0

#: More than one beam, so the per-beam bixel offset bookkeeping is exercised.
GANTRY_ANGLES = (0.0, 90.0)


def _ray_offsets(count):
    """``count`` ray positions (mm), spaced by ``BIXEL_WIDTH`` and centred on zero."""
    return (np.arange(count) - (count - 1) / 2.0) * BIXEL_WIDTH


def _filled_field_stf(n_stack, n_travel):
    """Steering information holding one filled ``n_stack`` x ``n_travel`` field per beam."""
    rays = [
        Ray(
            ray_pos_bev=np.array([x, 0.0, z]),
            ray_pos=np.array([x, 0.0, z]),
            target_point_bev=np.array([2 * x, 1000.0, 2 * z]),
            target_point=np.array([2 * x, 1000.0, 2 * z]),
            beamlets=[Beamlet(energy=6.0)],
        )
        for z in _ray_offsets(n_stack)
        for x in _ray_offsets(n_travel)
    ]
    beams = [
        Beam(
            gantry_angle=gantry_angle,
            couch_angle=0.0,
            bixel_width=BIXEL_WIDTH,
            radiation_mode="photons",
            machine="Generic",
            iso_center=np.zeros(3),
            rays=rays,
        )
        for gantry_angle in GANTRY_ANGLES
    ]
    return SteeringInformation(beams=beams)


def _stratify(fluence_map):
    """The sequencer's stratification of an input map, in level units."""
    return np.round(fluence_map / fluence_map.max() * NUM_OF_LEVELS)


def _weight_lower_bound(d_0):
    """Sliding-window (left to right sweep) lower bound on a beam's total weight,
    in level units. Per leaf row ``d[0] + sum(max(0, d[i] - d[i-1]))``.
    """
    d_0 = np.asarray(d_0, dtype=int)
    return max(row[0] + np.maximum(0, np.diff(row)).sum() for row in d_0)


def _sequence_filled_field(sequencer_name, input_fluence):
    """Sequence a filled field shaped like *input_fluence*, carried on every beam."""
    pln = PhotonPlan(
        machine="Generic",
        prop_seq={"sequencer": sequencer_name, "num_of_levels": NUM_OF_LEVELS},
    )
    # The field is built to fit the map, so the two cannot fall out of step.
    stf = _filled_field_stf(*input_fluence.shape)
    # Rays are laid out row-major (stack axis outer, travel axis inner).
    fluence = np.concatenate([input_fluence.ravel(order="C")] * len(stf.beams))

    # No dij: this checks the decomposition, not the dose it would deliver.
    sequenced = run_sequencing(PlanResult(fluence=fluence), pln, stf, dij=None)
    return sequenced, sequenced.sequence.aperture_info(stf)


def _beam_map(info, beam_ix, bixel_values, shape):
    """The values of one beam's bixels on its field (rays are laid out row-major)."""
    grid = info.grids[beam_ix]
    bixels = slice(grid.bixel_offset, grid.bixel_offset + grid.num_of_bixels)
    return np.asarray(bixel_values, dtype=float)[bixels].reshape(shape)


def _assert_reconstructs(where, info, beam_ix, input_fluence):
    """Assert what holds for any input map, and return what differs between them."""
    beam = info.sequence.beams[beam_ix]
    # The fluence is stratified against the beam's peak.
    level = input_fluence.max() / NUM_OF_LEVELS

    d_0 = _stratify(input_fluence)
    np.testing.assert_allclose(
        _beam_map(info, beam_ix, info.sequence.fluence, input_fluence.shape) / level,
        d_0,
        atol=1e-9,
        err_msg=f"{where}: input was not stratified to the expected level map",
    )

    # (dim_stack, dim_travel, n_shapes): the open fraction of every bixel per segment
    shapes = np.stack(
        [
            info.aperture_fluence(beam_ix, segment_ix).reshape(input_fluence.shape)
            for segment_ix in range(beam.num_of_segments)
        ],
        axis=-1,
    )
    assert set(np.unique(shapes)) <= {0, 1}, f"{where}: shapes are not binary"

    # Calculate the weight of each shape in the stratified level units
    weights = beam.weights
    level_weights = weights / level
    np.testing.assert_allclose(
        level_weights,
        np.rint(level_weights),
        atol=1e-9,
        err_msg=f"{where}: segment weights are not whole levels",
    )

    # Reconstruction: the stratified fluence (d_0) must be exactly the sum of the binary
    # shapes weighted by their level weights.
    d_0_reconstructed = (shapes.astype(int) * np.rint(level_weights).astype(int)).sum(axis=-1)
    np.testing.assert_array_equal(
        d_0_reconstructed,
        d_0.astype(int),
        err_msg=f"{where}: segments do not add back up to the stratified map",
    )

    # Stratification error: *physical* delivered fluence must be within half a level.
    delivered_fluence = (shapes * weights).sum(axis=-1)
    stratification_error = np.abs(delivered_fluence - input_fluence).max()
    assert stratification_error <= 0.5 * level, (
        f"{where}: delivery exceeds the stratification error bound"
    )

    return weights, stratification_error


@pytest.mark.parametrize("sequencer_name", sorted(get_available_sequencers("photons")))
def test_constant_fluence_reconstructs_from_weighted_shapes(sequencer_name):
    """A constant map must come back exactly: every bixel open at the top level."""
    input_fluence = np.full((FIELD_SIDE, FIELD_SIDE), CONSTANT)

    sequenced, info = _sequence_filled_field(sequencer_name, input_fluence)

    for beam_ix, beam in enumerate(sequenced.sequence.beams):
        where = f"{sequencer_name}: beam {beam_ix}"
        weights, stratification_error = _assert_reconstructs(where, info, beam_ix, input_fluence)

        # A flat map sits exactly on the top level, so stratifying it loses nothing.
        assert stratification_error == 0.0, f"{where}: a constant map must stratify exactly"

        # A constant map needs no more total weight than its own height.
        assert weights.sum() == pytest.approx(CONSTANT), f"{where}: delivers excess weight"


@pytest.mark.parametrize("sequencer_name", sorted(get_available_sequencers("photons")))
def test_ramp_fluence_reconstructs_from_weighted_shapes(sequencer_name):
    """A ramp rising along the leaf travel direction, one per leaf row."""
    input_fluence = np.tile(np.linspace(0.4, CONSTANT, FIELD_SIDE), (FIELD_SIDE, 1))

    scaled = input_fluence / input_fluence.max() * NUM_OF_LEVELS
    assert not np.array_equal(np.round(scaled), np.floor(scaled)), "ramp sits on the levels"

    sequenced, info = _sequence_filled_field(sequencer_name, input_fluence)

    for beam_ix, beam in enumerate(sequenced.sequence.beams):
        where = f"{sequencer_name}: beam {beam_ix}"
        weights, stratification_error = _assert_reconstructs(where, info, beam_ix, input_fluence)

        # Ramp does not sit on the levels, so the half-level bound is genuinely exercised
        # rather than trivially met.
        assert stratification_error > 0.0, f"{where}: ramp should not stratify exactly"

        # No delivery can beat the sliding-window bound; only Engel and Siochi meet it.
        level = input_fluence.max() / NUM_OF_LEVELS
        bound = _weight_lower_bound(_stratify(input_fluence)) * level
        assert weights.sum() >= bound - 1e-9, f"{where}: delivers less than the lower bound"


@pytest.mark.parametrize("sequencer_name", sorted(get_available_sequencers("photons")))
def test_pyramid_fluence_reconstructs_from_weighted_shapes(sequencer_name):
    """A pyramid with its peak at the centre of the field, one per leaf row."""
    # Fractions clear of .5 so the map does not sit on levels
    pyramid_levels = np.array([1.6, 3.4, 5.0, 3.4, 1.6])
    profile = pyramid_levels * CONSTANT / NUM_OF_LEVELS
    input_fluence = np.tile(profile, (FIELD_SIDE, 1))

    scaled = input_fluence / input_fluence.max() * NUM_OF_LEVELS
    assert not np.array_equal(np.round(scaled), np.floor(scaled)), "pyramid sits on the levels"

    sequenced, info = _sequence_filled_field(sequencer_name, input_fluence)

    for beam_ix, beam in enumerate(sequenced.sequence.beams):
        where = f"{sequencer_name}: beam {beam_ix}"
        weights, stratification_error = _assert_reconstructs(where, info, beam_ix, input_fluence)

        # Pyramid does not sit on the levels, so the half-level bound is genuinely exercised
        # rather than trivially met.
        assert stratification_error > 0.0, f"{where}: pyramid should not stratify exactly"

        # No delivery can beat the sliding-window bound; only Engel and Siochi meet it.
        level = input_fluence.max() / NUM_OF_LEVELS
        bound = _weight_lower_bound(_stratify(input_fluence)) * level
        assert weights.sum() >= bound - 1e-9, f"{where}: delivers less than the lower bound"


@pytest.mark.parametrize("sequencer_name", sorted(get_available_sequencers("photons")))
def test_single_spike_fluence_reconstructs_from_weighted_shapes(sequencer_name):
    """One open bixel in an otherwise closed field."""
    input_fluence = np.zeros((FIELD_SIDE, FIELD_SIDE))
    input_fluence[FIELD_SIDE // 2, FIELD_SIDE // 2] = CONSTANT

    sequenced, info = _sequence_filled_field(sequencer_name, input_fluence)

    for beam_ix, beam in enumerate(sequenced.sequence.beams):
        where = f"{sequencer_name}: beam {beam_ix}"
        weights, stratification_error = _assert_reconstructs(where, info, beam_ix, input_fluence)

        # Only zero and the peak occur, and both sit on a level.
        assert stratification_error == 0.0, f"{where}: a spike must stratify exactly"

        # One open bixel needs no more than its own height.
        assert weights.sum() == pytest.approx(CONSTANT), f"{where}: delivers excess weight"

        # Every leaf row but the spike's is closed.
        for shape_ix, segment in enumerate(beam.segments):
            left, right = np.asarray(segment.start.mlc.leaf_positions, dtype=float).T
            closed = np.isclose(left, right)
            assert closed.sum() == FIELD_SIDE - 1, (
                f"{where}, shape {shape_ix}: every row but the spike's should be closed"
            )


@pytest.mark.parametrize("sequencer_name", sorted(get_available_sequencers("photons")))
def test_checkerboard_fluence_reconstructs_from_weighted_shapes(sequencer_name):
    """Alternating open and closed bixels -- the worst case for a sliding window."""
    input_fluence = np.indices((FIELD_SIDE, FIELD_SIDE)).sum(axis=0) % 2 * CONSTANT

    sequenced, info = _sequence_filled_field(sequencer_name, input_fluence)

    for beam_ix, beam in enumerate(sequenced.sequence.beams):
        where = f"{sequencer_name}: beam {beam_ix}"
        weights, stratification_error = _assert_reconstructs(where, info, beam_ix, input_fluence)

        # Only zero and the peak occur, and both sit on a level.
        assert stratification_error == 0.0, f"{where}: a checkerboard must stratify exactly"

        # Each peak bixel must be paid for separately, so the total weight is above the peak.
        level = input_fluence.max() / NUM_OF_LEVELS
        bound = _weight_lower_bound(_stratify(input_fluence)) * level
        assert bound > input_fluence.max(), f"{where}: a checkerboard should cost above its peak"
        assert weights.sum() >= bound - 1e-9, f"{where}: delivers less than the lower bound"


@pytest.mark.parametrize("sequencer_name", sorted(get_available_sequencers("photons")))
def test_zero_fluence_delivers_nothing(sequencer_name):
    """A beam the optimization zeroed out sequences to nothing rather than raising."""
    input_fluence = np.zeros((FIELD_SIDE, FIELD_SIDE))

    sequenced, info = _sequence_filled_field(sequencer_name, input_fluence)

    # Not _assert_reconstructs: with no peak there is no level to express the weights in.
    for beam_ix, beam in enumerate(sequenced.sequence.beams):
        where = f"{sequencer_name}: beam {beam_ix}"
        assert beam.weights.sum() == pytest.approx(0.0), f"{where}: delivers weight for a zero map"
        for delivered in (info.bixel_weights(), sequenced.sequence.fluence):
            np.testing.assert_array_equal(
                _beam_map(info, beam_ix, delivered, input_fluence.shape),
                np.zeros_like(input_fluence),
                err_msg=f"{where}: delivers fluence for a zero map",
            )


@pytest.mark.parametrize("sequencer_name", sorted(get_available_sequencers("photons")))
def test_one_zeroed_beam_does_not_disturb_the_others(sequencer_name):
    """The realistic case: optimization drops one beam and keeps the rest."""
    stf = _filled_field_stf(FIELD_SIDE, FIELD_SIDE)
    per_beam = FIELD_SIDE * FIELD_SIDE
    fluence = np.concatenate([np.zeros(per_beam), np.full(per_beam, CONSTANT)])
    pln = PhotonPlan(
        machine="Generic",
        prop_seq={"sequencer": sequencer_name, "num_of_levels": NUM_OF_LEVELS},
    )

    sequenced = run_sequencing(PlanResult(fluence=fluence), pln, stf, dij=None)

    zeroed, kept = sequenced.sequence.beams
    assert zeroed.weights.sum() == pytest.approx(0.0)
    assert kept.weights.sum() == pytest.approx(CONSTANT)

    # The dropped beam must not leak NaN into the plan-wide structures it shares.
    info = sequenced.sequence.aperture_info(stf)
    assert np.isfinite(info.to_vector()).all(), f"{sequencer_name}: aperture vector holds NaN"
    lower, upper = info.bounds()
    leaves = slice(info.total_num_of_apertures, None)
    assert np.isfinite(lower).all(), f"{sequencer_name}: lower bounds hold NaN"
    assert np.isfinite(upper[leaves]).all(), f"{sequencer_name}: leaf limits hold NaN"

    delivered = np.asarray(sequenced.sequence.fluence, dtype=float)
    np.testing.assert_array_equal(delivered[:per_beam], np.zeros(per_beam))
    np.testing.assert_allclose(delivered[per_beam:], np.full(per_beam, CONSTANT))


def _bumpy_fluence():
    """A map with another profile in every leaf row, off the levels and with closed bixels."""
    rows = np.array(
        [
            [0.0, 0.7, 1.9, 0.3, 0.0],
            [1.1, 2.0, 0.4, 1.3, 0.9],
            [0.0, 0.0, 1.5, 1.5, 0.2],
            [0.6, 0.0, 0.0, 1.7, 1.0],
        ]
    )
    return rows * CONSTANT / rows.max()


@pytest.mark.parametrize("sequencer_name", sorted(get_available_sequencers("photons")))
def test_segments_carry_their_stf_beam_and_both_jaws(sequencer_name):
    """Every segment names the beam it is planned on and is bounded by an X and a Y jaw."""
    input_fluence = _bumpy_fluence()
    n_stack, n_travel = input_fluence.shape

    sequenced, _ = _sequence_filled_field(sequencer_name, input_fluence)

    assert len(sequenced.sequence.beams) == len(GANTRY_ANGLES)
    for beam_ix, beam in enumerate(sequenced.sequence.beams):
        assert beam.num_of_segments > 0, f"{sequencer_name}: beam {beam_ix} has no segments"
        assert beam.stf_beams == [beam_ix]
        for segment_ix, segment in enumerate(beam.segments):
            where = f"{sequencer_name}: beam {beam_ix}, segment {segment_ix}"
            assert segment.stf_beam == beam_ix, f"{where}: wrong stf_beam"
            assert segment.end is None, f"{where}: step and shoot does not move"
            assert segment.start.mlc is not None, f"{where}: no MLC"

            jaws = {jaw.device_orientation: jaw for jaw in segment.start.jaws}
            assert len(segment.start.jaws) == 2, f"{where}: expected one jaw per axis"
            assert set(jaws) == {"X", "Y"}, f"{where}: expected an X and a Y jaw"
            # The leaves travel along X: the X jaw bounds the columns, the Y jaw the rows.
            half_x, half_y = 0.5 * n_travel * BIXEL_WIDTH, 0.5 * n_stack * BIXEL_WIDTH
            np.testing.assert_allclose(jaws["X"].positions, [-half_x, half_x])
            np.testing.assert_allclose(jaws["Y"].positions, [-half_y, half_y])


@pytest.mark.parametrize("sequencer_name", sorted(get_available_sequencers("photons")))
def test_bixel_weights_of_the_apertures_reproduce_the_sequenced_fluence(sequencer_name):
    """On the ray grid the apertures open whole bixels, so their weights add up to the fluence."""
    input_fluence = _bumpy_fluence()

    sequenced, info = _sequence_filled_field(sequencer_name, input_fluence)

    for grid in info.grids:
        assert grid.leaf_width == grid.bixel_width == BIXEL_WIDTH
    bixel_weights = info.bixel_weights()
    assert bixel_weights.shape == sequenced.sequence.fluence.shape
    np.testing.assert_allclose(bixel_weights, sequenced.sequence.fluence, rtol=1e-12, atol=1e-12)
    # The fluence is the stratified input, so the comparison is not met by zeros.
    level = input_fluence.max() / NUM_OF_LEVELS
    for beam_ix in range(len(sequenced.sequence.beams)):
        np.testing.assert_allclose(
            _beam_map(info, beam_ix, bixel_weights, input_fluence.shape),
            _stratify(input_fluence) * level,
            atol=1e-9,
        )
    np.testing.assert_allclose(info.to_sequence().fluence, sequenced.sequence.fluence)
