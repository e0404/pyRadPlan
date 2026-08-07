"""Tests for the sequencing part of a plan result (Sequence / BeamSequence)."""

import numpy as np
import pytest

from pyRadPlan.result._beam_sequence import BeamSequence
from pyRadPlan.result._sequence import Sequence


@pytest.fixture
def beam_sequence_data():
    """Minimal valid BeamSequence input."""
    return {
        "shapes": np.ones((3, 4, 2)),
        "shapes_weight": np.array([1.5, 2.5]),
        "bixel_ix": [0, 1, 2],
        "fluence": np.array([[0.4, 1.7], [2.9, 300.5]]),
    }


def test_beam_sequence_keeps_fluence_continuous(beam_sequence_data):
    """Fluence carries continuous weights and must not be rounded or wrapped."""
    beam = BeamSequence.model_validate(beam_sequence_data)

    assert beam.fluence.dtype == float
    np.testing.assert_allclose(beam.fluence, [[0.4, 1.7], [2.9, 300.5]])


def test_beam_sequence_shapes_are_compact(beam_sequence_data):
    """Shapes are binary aperture masks, stored as uint8."""
    beam = BeamSequence.model_validate(beam_sequence_data)

    assert beam.shapes.dtype == np.uint8
    assert beam.num_of_shapes == 2
    np.testing.assert_array_equal(beam.sum_shapes, np.full((3, 4), 2))


def test_beam_sequence_adds_shape_axis_from_matrad(beam_sequence_data):
    """matRad stores a single shape as a 2D array; a shape axis is added on import."""
    beam_sequence_data["shapes"] = np.ones((3, 4))
    beam = BeamSequence.model_validate(beam_sequence_data, context={"from_matRad": True})

    assert beam.shapes.shape == (3, 4, 1)
    assert beam.num_of_shapes == 1


def _matrad_beam_struct(num_beams=2):
    """A matRad ``sequencing.beam`` struct-of-arrays for *num_beams* beams."""
    return {
        "shapes": [np.ones((3, 4, 1)) for _ in range(num_beams)],
        "shapesWeight": [np.array([1.0]) for _ in range(num_beams)],
        "bixelIx": [np.array([1, 2, 3]) for _ in range(num_beams)],
        "fluence": [np.full((3, 4), 0.5) for _ in range(num_beams)],
    }


def test_sequence_matrad_import_converts_bixel_indices():
    """matRad bixel indices are 1-based and become 0-based on import."""
    seq = Sequence.model_validate({"beam": _matrad_beam_struct()}, context={"from_matRad": True})

    assert len(seq.beam) == 2
    assert seq.beam[0].bixel_ix == [0, 1, 2]
    assert seq.total_num_of_shapes == 2


def test_sequence_matrad_import_merges_aperture_info():
    """apertureInfo entries are flattened onto the Sequence and its beams."""
    data = {
        "beam": _matrad_beam_struct(),
        "aperture_info": {
            "bixelWidth": 5.0,
            "numOfMLCLeafPairs": 40,
            "totalNumOfBixels": 6,
            "beam": {"numOfActiveLeafPairs": [3, 3]},
        },
    }
    seq = Sequence.model_validate(data, context={"from_matRad": True})

    assert seq.bixel_width == 5.0
    assert seq.num_of_mlc_leaf_pairs == 40
    assert seq.total_num_of_bixels == 6
    assert [beam.num_of_active_leaf_pairs for beam in seq.beam] == [3, 3]
    assert seq.total_num_of_leaf_pairs == 6


def test_sequence_matrad_import_without_beam_substruct():
    """An apertureInfo without a 'beam' sub-struct must not raise."""
    seq = Sequence.model_validate(
        {"aperture_info": {"bixelWidth": 5.0, "numOfMLCLeafPairs": 40}},
        context={"from_matRad": True},
    )

    assert seq.bixel_width == 5.0
    assert seq.num_of_mlc_leaf_pairs == 40
    assert seq.beam is None
    assert seq.total_num_of_leaf_pairs == 0


def test_sequence_matrad_import_mismatched_beam_count(caplog):
    """A beam-count mismatch is reported rather than crashing."""
    data = {
        "beam": _matrad_beam_struct(num_beams=2),
        "aperture_info": {"beam": {"numOfActiveLeafPairs": [3, 3, 3]}},
    }
    seq = Sequence.model_validate(data, context={"from_matRad": True})

    assert "does not match" in caplog.text
    assert [beam.num_of_active_leaf_pairs for beam in seq.beam] == [None, None]


def test_sequence_matrad_import_leaves_input_untouched():
    """Validation must not consume the caller's dictionary."""
    data = {"beam": _matrad_beam_struct(), "aperture_info": {"bixelWidth": 5.0}}
    snapshot = set(data)

    Sequence.model_validate(data, context={"from_matRad": True})

    assert set(data) == snapshot
