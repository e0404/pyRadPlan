"""The delivery sequence model: segments, control points, and the bridge to beamlets."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyRadPlan.machines import MLC, Jaw
from pyRadPlan.result import (
    ApertureInfo,
    BeamSequence,
    MachineState,
    PlanResult,
    Segment,
    Sequence,
)


def _mlc(boundaries, positions, leaf_width=5.0):
    return MLC(
        device_orientation="X",
        number_of_elements=80,
        leaf_width=leaf_width,
        leaf_position_boundaries=boundaries,
        leaf_positions=positions,
    )


def _stf_beam(travel, stack, bixel_width=5.0):
    rays = [SimpleNamespace(ray_pos_bev=(x, 0.0, z)) for z in stack for x in travel]
    return SimpleNamespace(bixel_width=bixel_width, rays=rays)


@pytest.fixture
def stf():
    return SimpleNamespace(
        beams=[_stf_beam([-10, -5, 0, 5, 10], [-5, 0, 5]), _stf_beam([-5, 0, 5], [0, 5])]
    )


@pytest.fixture
def sequence():
    rows = [-7.5, -2.5, 2.5]
    jaws = [
        Jaw(device_orientation="X", positions=[-12.5, 12.5], field_width=50.0),
        Jaw(device_orientation="Y", positions=[-25.0, 20.0], field_width=50.0),
    ]
    first = BeamSequence.from_shapes(
        [
            _mlc(rows, [[-12.5, 2.5], [-7.5, 12.5], [-2.5, -2.5]]),
            _mlc(rows, [[-6.0, 1.0], [-7.5, 7.5], [0.0, 4.0]]),
        ],
        [1.0, 2.5],
        jaws=jaws,
        stf_beam=0,
    )
    second = BeamSequence.from_shapes(
        [_mlc([-2.5, 2.5], [[-7.5, 7.5], [-2.5, 2.5]])], [3.0], stf_beam=1
    )
    return Sequence(beams=[first, second], sequencer="xia", num_of_levels=5)


def test_from_shapes_builds_static_segments(sequence):
    beam = sequence.beams[0]

    assert beam.num_of_segments == 2
    np.testing.assert_array_equal(beam.weights, [1.0, 2.5])
    assert beam.total_weight == pytest.approx(3.5)
    assert not beam.is_dynamic
    assert beam.stf_beams == [0]
    assert [len(segment.start.jaws) for segment in beam.segments] == [2, 2]
    assert all(segment.end is None for segment in beam.segments)
    assert sequence.total_num_of_segments == 3


def test_from_shapes_takes_jaws_per_segment():
    mlcs = [_mlc([0.0], [[-5.0, 5.0]]), _mlc([0.0], [[-2.0, 2.0]])]
    wide = Jaw(device_orientation="X", positions=[-20.0, 20.0], field_width=40.0)
    narrow = Jaw(device_orientation="X", positions=[-10.0, 10.0], field_width=40.0)

    beam = BeamSequence.from_shapes(mlcs, [1.0, 1.0], jaws=[[wide], [narrow]])

    assert beam.segments[0].start.jaws == [wide]
    assert beam.segments[1].start.jaws == [narrow]


def test_from_shapes_rejects_mismatching_counts():
    with pytest.raises(ValueError, match="1 MLC shapes for 2 weights"):
        BeamSequence.from_shapes([_mlc([0.0], [[-5.0, 5.0]])], [1.0, 2.0])


def test_jaw_only_field():
    jaw = Jaw(device_orientation="X", positions=[-20.0, 20.0], field_width=40.0)
    beam = BeamSequence.from_shapes(None, [1.0], jaws=[jaw])
    assert beam.mlc is None
    assert beam.segments[0].start.jaws == [jaw]


def test_segment_weight_must_not_be_negative():
    with pytest.raises(ValueError):
        Segment(weight=-1.0, start=MachineState())


def test_segment_needs_the_same_leaf_rows_at_start_and_end():
    with pytest.raises(ValueError, match="same leaf rows"):
        Segment(
            weight=1.0,
            start=MachineState(mlc=_mlc([0.0], [[-5.0, 5.0]])),
            end=MachineState(mlc=_mlc([0.0, 5.0], [[-5.0, 5.0], [-5.0, 5.0]])),
        )


@pytest.fixture
def moving_segment():
    return Segment(
        weight=3.0,
        start=MachineState(gantry_angle=350.0, mlc=_mlc([-2.5, 2.5], [[-7.5, 7.5], [-2.5, 2.5]])),
        end=MachineState(gantry_angle=10.0, mlc=_mlc([-2.5, 2.5], [[-5.5, 7.5], [-2.5, 4.5]])),
        duration=2.0,
    )


def test_a_state_carries_no_speed_the_segment_derives_it(moving_segment):
    assert moving_segment.is_dynamic
    np.testing.assert_allclose(moving_segment.leaf_travel, [[2.0, 0.0], [0.0, 2.0]])
    assert moving_segment.max_leaf_speed == pytest.approx(1.0)
    assert moving_segment.gantry_travel() == pytest.approx(20.0)
    assert moving_segment.gantry_speed() == pytest.approx(10.0)
    assert moving_segment.weight_rate == pytest.approx(1.5)


def test_speeds_are_unknown_without_a_duration(moving_segment):
    untimed = moving_segment.model_copy(update={"duration": None})
    assert untimed.max_leaf_speed is None
    assert untimed.gantry_speed() is None
    assert untimed.weight_rate is None
    np.testing.assert_allclose(untimed.leaf_travel, [[2.0, 0.0], [0.0, 2.0]])


def test_gantry_travel_follows_the_rotation_direction(moving_segment):
    assert moving_segment.gantry_travel("cw") == pytest.approx(20.0)
    assert moving_segment.gantry_travel("ccw") == pytest.approx(-340.0)


def test_state_at_interpolates_linearly_across_the_wrap(moving_segment):
    state = moving_segment.state_at(0.75)
    assert state.gantry_angle == pytest.approx(5.0)
    np.testing.assert_allclose(state.mlc.leaf_positions, [[-6.0, 7.5], [-2.5, 4.0]])

    static = Segment(weight=1.0, start=moving_segment.start)
    assert static.state_at(0.5) is static.start
    assert not static.is_dynamic
    assert static.gantry_travel() == 0.0


def test_control_points_of_step_and_shoot(sequence):
    points = sequence.beams[0].control_points(gantry_angle=30.0, couch_angle=0.0)

    assert [point.index for point in points] == [0, 1, 2, 3]
    np.testing.assert_allclose([point.cumulative_weight for point in points], [0.0, 1.0, 1.0, 3.5])
    np.testing.assert_allclose(
        [point.cumulative_meterset_weight for point in points], [0.0, 1 / 3.5, 1 / 3.5, 1.0]
    )
    assert {point.gantry_angle for point in points} == {30.0}
    assert {point.gantry_rotation_direction for point in points} == {"NONE"}
    assert points[0].mlc == points[1].mlc
    assert points[2].mlc == points[3].mlc
    assert points[1].mlc != points[2].mlc


def test_control_points_of_a_continuous_arc(moving_segment):
    onward = Segment(
        weight=1.0,
        start=moving_segment.end,
        end=moving_segment.end.model_copy(update={"gantry_angle": 30.0}),
    )
    arc = BeamSequence(segments=[moving_segment, onward], rotation="cw")

    assert arc.is_dynamic
    assert arc.is_continuous
    points = arc.control_points()
    assert [point.gantry_angle for point in points] == [350.0, 10.0, 30.0]
    np.testing.assert_allclose([point.cumulative_weight for point in points], [0.0, 3.0, 4.0])
    assert [point.gantry_rotation_direction for point in points] == ["CW", "CW", "NONE"]


def test_a_gap_between_segments_is_not_continuous(moving_segment):
    elsewhere = Segment(weight=1.0, start=moving_segment.start)
    assert not BeamSequence(segments=[moving_segment, elsewhere]).is_continuous


def test_sequence_survives_a_model_dump(sequence, moving_segment):
    sequence = sequence.model_copy(
        update={"beams": [*sequence.beams, BeamSequence(segments=[moving_segment], rotation="cw")]}
    )
    assert Sequence.model_validate(sequence.model_dump()) == sequence


def test_aperture_info_maps_segments_to_beamlets(sequence, stf):
    info = ApertureInfo.from_sequence(sequence, stf)

    assert [grid.bixel_offset for grid in info.grids] == [0, 15]
    assert info.total_num_of_bixels == 21
    assert info.total_num_of_apertures == 3
    assert info.total_num_of_leaf_pairs == 8
    np.testing.assert_allclose(info.grids[0].lim_left, [-12.5, -12.5, -12.5])
    np.testing.assert_allclose(info.grids[1].lim_right, [7.5, 7.5])

    # Row at z=-5 of beam 0: the first shape opens x < 2.5, the second [-6, 1], which
    # covers 70 % of the beamlets at -5 and 0.
    np.testing.assert_allclose(
        info.bixel_weights()[:5], [1.0, 1.0 + 2.5 * 0.7, 1.0 + 2.5 * 0.7, 0.0, 0.0]
    )
    np.testing.assert_allclose(info.aperture_fluence(1, 0), [1, 1, 1, 0, 1, 0])
    np.testing.assert_allclose(info.to_sequence().fluence, info.bixel_weights())


def test_aperture_info_counts_leaf_rows_wider_than_beamlets():
    stf = SimpleNamespace(beams=[_stf_beam([0.0], [-2.5, 2.5, 7.5])])
    wide_rows = _mlc([-5.0, 5.0], [[-2.5, 2.5], [0.0, 0.0]], leaf_width=10.0)
    sequence = Sequence(beams=[BeamSequence.from_shapes([wide_rows], [2.0], stf_beam=0)])

    info = sequence.aperture_info(stf)

    np.testing.assert_allclose(info.bixel_weights(), [2.0, 2.0, 0.0])
    np.testing.assert_allclose(info.grids[0].row_coverage, [[1, 1, 0], [0, 0, 1]])


def test_aperture_vector_round_trip_and_bounds(sequence, stf):
    info = sequence.aperture_info(stf)
    vector = info.to_vector()

    assert vector.size == 3 + 2 * 8
    np.testing.assert_allclose(vector[:3], [1.0, 2.5, 3.0])
    np.testing.assert_allclose(vector[3:6], [-12.5, -7.5, -2.5])
    np.testing.assert_allclose(vector[3 + 8 : 6 + 8], [2.5, 12.5, -2.5])
    assert info.update(vector).sequence == info.sequence

    lower, upper = info.bounds()
    np.testing.assert_array_equal(lower[:3], 0.0)
    assert np.all(np.isinf(upper[:3]))
    np.testing.assert_allclose(lower[3:6], -12.5)
    np.testing.assert_allclose(upper[3:6], 12.5)

    with pytest.raises(ValueError, match="Expected 19 variables"):
        info.update(vector[:-1])


def test_updated_vector_becomes_a_sequence(sequence, stf):
    info = sequence.aperture_info(stf)
    vector = info.to_vector()
    vector[0] = 4.0
    vector[3] = -11.0

    updated = info.update(vector).to_sequence()

    first = updated.beams[0].segments[0]
    assert first.weight == pytest.approx(4.0)
    assert first.start.mlc.leaf_positions[0, 0] == pytest.approx(-11.0)
    assert first.start.jaws == sequence.beams[0].segments[0].start.jaws
    assert updated.sequencer == "xia"
    # The beamlet at x=-10 spans [-12.5, -7.5]; the leaf at -11 leaves 70 % of it open.
    assert updated.fluence[0] == pytest.approx(4.0 * 0.7)


def test_bixel_jacobian_matches_finite_differences(sequence, stf):
    info = sequence.aperture_info(stf)
    # Leaf edges on beamlet edges are kinks of the mapping; move every edge off them.
    vector = info.to_vector()
    vector[3:11] += 0.8
    vector[11:] -= 0.6
    info = info.update(vector)

    jacobian = info.bixel_jacobian().toarray()

    step = 1e-4
    numeric = np.zeros_like(jacobian)
    for ix in range(vector.size):
        up, down = vector.copy(), vector.copy()
        up[ix] += step
        down[ix] -= step
        numeric[ix] = (info.update(up).bixel_weights() - info.update(down).bixel_weights()) / (
            2 * step
        )
    assert jacobian.shape == (19, 21)
    assert np.count_nonzero(jacobian) > 20  # noqa: PLR2004
    np.testing.assert_allclose(jacobian, numeric, atol=1e-8)


def test_jaws_cut_the_apertures(stf):
    rows = [-7.5, -2.5, 2.5]
    jaws = [
        Jaw(device_orientation="X", positions=[-8.0, 12.5], field_width=50.0),
        Jaw(device_orientation="Y", positions=[-6.0, 20.0], field_width=50.0),
    ]
    mlc = _mlc(rows, [[-12.5, 2.5], [-7.5, 12.5], [-2.5, -2.5]])
    clipped = Sequence(beams=[BeamSequence.from_shapes([mlc], [1.0], jaws=jaws, stf_beam=0)])
    single = SimpleNamespace(beams=stf.beams[:1])

    info = clipped.aperture_info(single)

    # Row at z=-5 ([-7.5, -2.5]): the Y jaw keeps 70 % of it, the X jaw at -8 cuts the
    # leaf at -12.5 so that the beamlet at -10 ([-12.5, -7.5]) is open 10 % of its width.
    np.testing.assert_allclose(info.bixel_weights()[:5], [0.7 * 0.1, 0.7, 0.7, 0.0, 0.0])
    np.testing.assert_allclose(info.bixel_weights()[5:10], [0.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(info.to_sequence().fluence, info.bixel_weights())


def test_bixel_jacobian_within_jaws_matches_finite_differences(stf):
    rows = [-7.5, -2.5, 2.5]
    jaws = [
        Jaw(device_orientation="X", positions=[-8.0, 9.0], field_width=50.0),
        Jaw(device_orientation="Y", positions=[-6.0, 3.5], field_width=50.0),
    ]
    mlc = _mlc(rows, [[-11.7, 1.8], [-5.4, 11.2], [-3.1, 3.3]])
    clipped = Sequence(beams=[BeamSequence.from_shapes([mlc], [1.5], jaws=jaws, stf_beam=0)])
    info = clipped.aperture_info(SimpleNamespace(beams=stf.beams[:1]))
    vector = info.to_vector()

    jacobian = info.bixel_jacobian().toarray()

    step = 1e-4
    numeric = np.zeros_like(jacobian)
    for ix in range(vector.size):
        up, down = vector.copy(), vector.copy()
        up[ix] += step
        down[ix] -= step
        numeric[ix] = (info.update(up).bixel_weights() - info.update(down).bixel_weights()) / (
            2 * step
        )
    # The leaves behind the jaws (left of row 0, right of row 1) change nothing.
    assert not jacobian[1].any()
    assert not jacobian[1 + 3 + 1].any()
    np.testing.assert_allclose(jacobian, numeric, atol=1e-8)


def test_aperture_info_refuses_a_collimator_angle(sequence, stf):
    segment = sequence.beams[1].segments[0]
    mlc = segment.start.mlc.model_copy(update={"device_angle": 15.0})
    rotated = sequence.model_copy(
        update={
            "beams": [
                sequence.beams[0],
                BeamSequence(
                    segments=[
                        segment.model_copy(
                            update={"start": segment.start.model_copy(update={"mlc": mlc})}
                        )
                    ]
                ),
            ]
        }
    )
    with pytest.raises(NotImplementedError, match="collimator angle"):
        rotated.aperture_info(stf)


def test_aperture_info_requires_static_segments_with_an_mlc(stf, moving_segment):
    arc = Sequence(beams=[BeamSequence(segments=[moving_segment])])
    with pytest.raises(NotImplementedError, match="move while they are delivered"):
        arc.aperture_info(SimpleNamespace(beams=stf.beams[:1]))

    jaw = Jaw(device_orientation="X", positions=[-20.0, 20.0], field_width=40.0)
    jaw_only = Sequence(beams=[BeamSequence.from_shapes(None, [1.0], jaws=[jaw], stf_beam=0)])
    with pytest.raises(ValueError, match="needs an MLC"):
        jaw_only.aperture_info(stf)


def test_aperture_info_assigns_segments_without_stf_beam_by_position(sequence, stf):
    unassigned = Sequence(
        beams=[
            BeamSequence(segments=[s.model_copy(update={"stf_beam": None}) for s in beam.segments])
            for beam in sequence.beams
        ]
    )
    np.testing.assert_allclose(
        unassigned.aperture_info(stf).bixel_weights(), sequence.aperture_info(stf).bixel_weights()
    )
    with pytest.raises(ValueError, match="one beam sequence per beam"):
        Sequence(beams=unassigned.beams[:1]).aperture_info(stf)


@pytest.mark.parametrize("with_stf", [False, True])
def test_matrad_round_trip_keeps_segments(sequence, stf, with_stf):
    fluence = sequence.aperture_info(stf).bixel_weights()
    result = PlanResult(
        fluence=np.ones(21), sequence=sequence.model_copy(update={"fluence": fluence})
    )

    exported = result.to_matrad(stf=stf if with_stf else None)

    info = exported["apertureInfo"]
    assert ("totalNumOfBixels" in info) == with_stf
    assert ("beam" in exported["sequencing"]) == with_stf
    assert info["totalNumOfShapes"] == 3  # noqa: PLR2004
    assert info["totalNumOfLeafPairs"] == 8  # noqa: PLR2004
    np.testing.assert_allclose(info["apertureVector"][:3], [1.0, 2.5, 3.0])
    if with_stf:
        beam = exported["sequencing"]["beam"][0]
        assert beam["shapes"].shape == (3, 5, 2)
        np.testing.assert_array_equal(beam["bixelIx"], np.arange(1, 16))
        np.testing.assert_array_equal(beam["shapes"][0, :, 0], [1, 1, 1, 0, 0])
        np.testing.assert_allclose(info["limMx"][3:6], [[-12.5, 12.5]] * 3)

    loaded = PlanResult.from_matrad(exported).sequence
    assert loaded.sequencer == "xia"
    assert loaded.num_of_levels == 5  # noqa: PLR2004
    np.testing.assert_allclose(loaded.fluence, fluence)
    # The jaws reach further than the beamlets and must come back as they were.
    assert loaded.beams[0].segments[0].start.jaws == sequence.beams[0].segments[0].start.jaws
    for beam, original in zip(loaded.beams, sequence.beams):
        np.testing.assert_allclose(beam.weights, original.weights)
        assert beam.stf_beams == original.stf_beams
        for segment, expected in zip(beam.segments, original.segments):
            np.testing.assert_allclose(
                segment.start.mlc.leaf_positions, expected.start.mlc.leaf_positions
            )
            np.testing.assert_allclose(
                segment.start.mlc.leaf_position_boundaries,
                expected.start.mlc.leaf_position_boundaries,
            )


def test_matrad_export_assigns_segments_without_stf_beam_by_position(sequence, stf):
    unassigned = Sequence(
        beams=[
            BeamSequence(segments=[s.model_copy(update={"stf_beam": None}) for s in beam.segments])
            for beam in sequence.beams
        ]
    )

    exported = PlanResult(fluence=np.ones(21), sequence=unassigned).to_matrad(stf=stf)

    assert exported["apertureInfo"]["totalNumOfBixels"] == 21  # noqa: PLR2004
    np.testing.assert_array_equal(exported["sequencing"]["beam"][1]["bixelIx"], np.arange(16, 22))


def test_matrad_export_keeps_leaves_wider_than_the_beamlets(stf, caplog):
    wide = _mlc([-10.0, 0.0], [[-12.5, 2.5], [-7.5, 7.5]], leaf_width=10.0)
    second = _mlc([-10.0, 0.0], [[-7.5, 7.5], [-2.5, 2.5]], leaf_width=10.0)
    sequence = Sequence(
        beams=[
            BeamSequence.from_shapes([wide], [1.0], stf_beam=0),
            BeamSequence.from_shapes([second], [2.0], stf_beam=1),
        ]
    )

    exported = PlanResult(fluence=np.ones(21), sequence=sequence).to_matrad(stf=stf)

    assert "leaves as wide as the beamlets" in caplog.text
    assert exported["apertureInfo"]["bixelWidth"] == 10.0  # noqa: PLR2004
    assert "totalNumOfBixels" not in exported["apertureInfo"]
    loaded = PlanResult.from_matrad(exported).sequence
    mlc = loaded.beams[0].segments[0].start.mlc
    assert mlc.leaf_width == 10.0  # noqa: PLR2004
    np.testing.assert_allclose(mlc.leaf_position_boundaries, [-10.0, 0.0])
    np.testing.assert_allclose(mlc.leaf_positions, wide.leaf_positions)


def test_matrad_export_refuses_beams_of_different_leaf_widths(sequence, caplog):
    segment = sequence.beams[1].segments[0]
    wide = _mlc([-10.0, 0.0], [[-7.5, 7.5], [-2.5, 2.5]], leaf_width=10.0)
    mixed = sequence.model_copy(
        update={
            "beams": [
                sequence.beams[0],
                BeamSequence(
                    segments=[
                        segment.model_copy(
                            update={"start": segment.start.model_copy(update={"mlc": wide})}
                        )
                    ]
                ),
            ]
        }
    )

    exported = PlanResult(fluence=np.ones(21), sequence=mixed).to_matrad()

    assert "apertureInfo" not in exported
    assert "one leaf width" in caplog.text


def test_matrad_import_of_a_single_beam_with_a_single_shape():
    data = {
        "w": np.ones(3),
        "apertureInfo": {
            "bixelWidth": 5.0,
            "numOfMLCLeafPairs": 80.0,
            "beam": {
                "leafPairPos": np.array([-5.0, 0.0]),
                "MLCWindow": np.array([-7.5, 7.5, -7.5, 2.5]),
                "shape": {
                    "leftLeafPos": np.array([-7.5, -2.5]),
                    "rightLeafPos": np.array([2.5, 7.5]),
                    "weight": 1.5,
                },
            },
        },
    }

    sequence = Sequence.model_validate(data, context={"from_matRad": True})

    segment = sequence.beams[0].segments[0]
    assert segment.weight == 1.5  # noqa: PLR2004
    assert segment.stf_beam == 0
    np.testing.assert_allclose(segment.start.mlc.leaf_positions, [[-7.5, 2.5], [-2.5, 7.5]])
    np.testing.assert_allclose(segment.start.mlc.leaf_position_boundaries, [-7.5, -2.5])
    assert [jaw.positions for jaw in segment.start.jaws] == [[-7.5, 7.5], [-7.5, 2.5]]


def test_matrad_import_without_aperture_info_keeps_the_weights_only(caplog):
    data = {"w": np.ones(3), "beam": {"shapes": np.ones((3, 4)), "shapesWeight": 1.0}}

    sequence = Sequence.model_validate(data, context={"from_matRad": True})

    assert sequence.beams == []
    np.testing.assert_array_equal(sequence.fluence, np.ones(3))
    assert "carries no leaf positions" in caplog.text


def test_matrad_export_leaves_out_what_matrad_cannot_hold(sequence, moving_segment, caplog):
    arc = sequence.model_copy(update={"beams": [BeamSequence(segments=[moving_segment])]})

    exported = PlanResult(fluence=np.ones(3), sequence=arc).to_matrad()

    assert "apertureInfo" not in exported
    assert exported["sequencing"] == {"sequencer": "xia", "numOfLevels": 5}
    assert "segments are not exported" in caplog.text
