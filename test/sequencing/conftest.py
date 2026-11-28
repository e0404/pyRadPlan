import pytest
import numpy as np

from pyRadPlan.plan import validate_pln
from pyRadPlan.stf import validate_stf
from pyRadPlan.dij import validate_dij
from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst
from pyRadPlan.machines import MLC
from pyRadPlan.result import BeamSequence, PlanResult, Sequence
from pyRadPlan.sequencing.sequencer import get_sequencer_from_pln


@pytest.fixture
def test_data_engel(test_data_engel_seq_raw):
    pln_mat = validate_pln(test_data_engel_seq_raw["pln"])
    stf_mat = validate_stf(test_data_engel_seq_raw["stf"])
    dij_mat = validate_dij(test_data_engel_seq_raw["dij"])
    ct_mat = validate_ct(test_data_engel_seq_raw["ct"])
    cst_mat = validate_cst(test_data_engel_seq_raw["cst"], ct=ct_mat)
    result_mat = PlanResult.from_matrad(
        test_data_engel_seq_raw["resultGUI_before_seq"], ct_mat.cube_hu
    )
    result_seq_mat = PlanResult.from_matrad(test_data_engel_seq_raw["resultGUI"], ct_mat.cube_hu)

    return pln_mat, stf_mat, dij_mat, ct_mat, cst_mat, result_mat, result_seq_mat


@pytest.fixture
def test_data_siochi(test_data_siochi_seq_raw):
    pln_mat = validate_pln(test_data_siochi_seq_raw["pln"])
    stf_mat = validate_stf(test_data_siochi_seq_raw["stf"])
    dij_mat = validate_dij(test_data_siochi_seq_raw["dij"])
    ct_mat = validate_ct(test_data_siochi_seq_raw["ct"])
    cst_mat = validate_cst(test_data_siochi_seq_raw["cst"], ct=ct_mat)
    result_mat = PlanResult.from_matrad(
        test_data_siochi_seq_raw["resultGUI_before_seq"], ct_mat.cube_hu
    )
    result_seq_mat = PlanResult.from_matrad(test_data_siochi_seq_raw["resultGUI"], ct_mat.cube_hu)

    return pln_mat, stf_mat, dij_mat, ct_mat, cst_mat, result_mat, result_seq_mat


@pytest.fixture
def test_data_xia(test_data_xia_seq_raw):
    pln_mat = validate_pln(test_data_xia_seq_raw["pln"])
    stf_mat = validate_stf(test_data_xia_seq_raw["stf"])
    dij_mat = validate_dij(test_data_xia_seq_raw["dij"])
    ct_mat = validate_ct(test_data_xia_seq_raw["ct"])
    cst_mat = validate_cst(test_data_xia_seq_raw["cst"], ct=ct_mat)
    result_mat = PlanResult.from_matrad(
        test_data_xia_seq_raw["resultGUI_before_seq"], ct_mat.cube_hu
    )
    result_seq_mat = PlanResult.from_matrad(test_data_xia_seq_raw["resultGUI"], ct_mat.cube_hu)

    return pln_mat, stf_mat, dij_mat, ct_mat, cst_mat, result_mat, result_seq_mat


# TODO: Put these functions either in a common utils module or in the upper test conftest?


@pytest.fixture
def to_py():
    """
    Return a function that recursively turns any numpy array
    into a nested Python list (and walks into dicts/lists).
    """

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        else:
            return obj

    return _convert


# this here is very helpful when debugging "==" assertions
@pytest.fixture
def assert_models_equal(to_py):
    """
    Return a function for asserting model sequences equal.
    """
    try:
        from pprint import pformat
    except ImportError:
        pformat = lambda x: str(x)

    def _assert(got: list[dict], want: list[dict]):
        assert len(got) == len(want), f"number of beams differ: got {len(got)}, want {len(want)}"
        for idx, (g, w) in enumerate(zip(got, want)):
            pg, pw = to_py(g), to_py(w)
            assert pg == pw, f"beam[{idx}] mismatch:\n  got:  {pformat(pg)}\n  want: {pformat(pw)}"

    return _assert


def _matrad_structs(data: dict) -> list[dict]:
    """One dictionary per element of a matRad struct array, which is read as one list per field."""
    if all(isinstance(value, list) for value in data.values()):
        return [dict(zip(data, values)) for values in zip(*data.values())]
    return [data]


@pytest.fixture
def assert_matches_matrad():
    """
    Return a function comparing a sequenced result with the matRad reference: the imported
    reference sequence and the ``apertureInfo`` / ``sequencing`` structs of its ``resultGUI``.
    """

    def _assert(result_seq, result_seq_mat, pln, stf, result_gui, atol=1e-4):  # noqa: PLR0913
        assert isinstance(result_seq, PlanResult)
        assert isinstance(result_seq.sequence, Sequence)

        seq, seq_mat = result_seq.sequence, result_seq_mat.sequence
        info = seq.aperture_info(stf)
        aperture_info = result_gui["apertureInfo"]
        num_of_apertures = seq.total_num_of_segments

        assert num_of_apertures == seq_mat.total_num_of_segments
        assert num_of_apertures == aperture_info["totalNumOfShapes"]
        assert info.total_num_of_bixels == aperture_info["totalNumOfBixels"]
        assert info.total_num_of_leaf_pairs == aperture_info["totalNumOfLeafPairs"]

        np.testing.assert_allclose(seq.fluence, seq_mat.fluence, atol=atol)
        np.testing.assert_allclose(info.bixel_weights(), seq_mat.fluence, atol=atol)

        vector = info.to_vector()
        np.testing.assert_allclose(vector, aperture_info["apertureVector"], atol=atol)
        np.testing.assert_allclose(
            np.column_stack(info.bounds()), aperture_info["limMx"], atol=atol
        )

        # matRad's mappingMx spells out the layout of the vector: per weight its beam, per
        # leaf position (beam, aperture, aperture of the beam, leaf row), all counted from 1.
        weight_rows = [[beam_ix + 1, np.nan, np.nan, np.nan] for beam_ix, _, _ in info.apertures]
        leaf_rows = [
            [beam_ix + 1, aperture_ix + 1, segment_ix + 1, row + 1]
            for aperture_ix, (beam_ix, segment_ix, segment) in enumerate(info.apertures)
            for row in range(info.grids[segment.stf_beam].num_of_rows)
        ]
        np.testing.assert_array_equal(
            aperture_info["mappingMx"], np.array(weight_rows + 2 * leaf_rows)
        )

        sequencer = get_sequencer_from_pln(pln)
        vector_offset = num_of_apertures
        beams = zip(
            seq.beams,
            seq_mat.beams,
            info.grids,
            _matrad_structs(aperture_info["beam"]),
            _matrad_structs(result_gui["sequencing"]["beam"]),
            strict=True,
        )
        # Every beam is checked: the test data is not symmetric between the beams.
        for idx, (beam, beam_mat, grid, info_mat, sequencing_mat) in enumerate(beams):
            where = f"beam[{idx}]"
            assert isinstance(beam, BeamSequence)
            assert isinstance(beam_mat, BeamSequence)
            assert beam.num_of_segments == beam_mat.num_of_segments, where
            assert beam.num_of_segments == info_mat["numOfShapes"], where
            assert beam.stf_beams == [idx], where

            mlc = beam.mlc
            central_leaf_pair = sequencer._central_leaf_pair(grid.bixel_width)
            assert grid.bixel_width == aperture_info["bixelWidth"], where
            assert mlc.number_of_elements == aperture_info["numOfMLCLeafPairs"], where
            assert grid.num_of_rows == info_mat["numOfActiveLeafPairs"], where
            assert central_leaf_pair == info_mat["centralLeafPair"], where
            np.testing.assert_array_equal(
                mlc.is_active_leaf_pair(central_leaf_pair),
                info_mat["isActiveLeafPair"],
                err_msg=f"{where}: active leaf pairs mismatch",
            )

            np.testing.assert_array_equal(
                grid.bixel_offset + np.arange(grid.num_of_bixels),
                np.asarray(sequencing_mat["bixelIx"]) - 1,
                err_msg=f"{where}: bixel indices mismatch",
            )
            index_map, travel_min, stack_min = grid.bixel_index_map()
            np.testing.assert_array_equal(
                index_map + 1,
                np.nan_to_num(info_mat["bixelIndMap"], nan=0.0),
                err_msg=f"{where}: bixel index map mismatch",
            )
            np.testing.assert_allclose(
                [travel_min, 0.0, stack_min], info_mat["posOfCornerBixel"], atol=atol
            )

            np.testing.assert_allclose(beam.weights, beam_mat.weights, atol=atol)
            np.testing.assert_allclose(
                mlc.leaf_position_boundaries + 0.5 * mlc.leaf_width,
                info_mat["leafPairPos"],
                atol=atol,
            )
            np.testing.assert_allclose(grid.lim_left, info_mat["lim_l"], atol=atol)
            np.testing.assert_allclose(grid.lim_right, info_mat["lim_r"], atol=atol)

            has_bixel = index_map >= 0
            shapes_mat = np.atleast_3d(sequencing_mat["shapes"])
            segments = zip(
                beam.segments, beam_mat.segments, _matrad_structs(info_mat["shape"]), strict=True
            )
            for m, (segment, segment_mat, shape_mat) in enumerate(segments):
                where = f"beam[{idx}].segments[{m}]"
                positions = segment.start.mlc.leaf_positions
                assert isinstance(segment.start.mlc, MLC)
                np.testing.assert_allclose(
                    positions, segment_mat.start.mlc.leaf_positions, atol=atol
                )
                np.testing.assert_allclose(positions[:, 0], shape_mat["leftLeafPos"], atol=atol)
                np.testing.assert_allclose(positions[:, 1], shape_mat["rightLeafPos"], atol=atol)

                x_jaw, y_jaw = segment.start.jaws
                assert (x_jaw.device_orientation, y_jaw.device_orientation) == ("X", "Y"), where
                np.testing.assert_allclose(
                    [*x_jaw.positions, *y_jaw.positions], info_mat["MLCWindow"], atol=atol
                )
                assert segment.start.jaws == segment_mat.start.jaws, where

                assert shape_mat["vectorOffset"] == vector_offset + 1, where
                rows = slice(vector_offset, vector_offset + grid.num_of_rows)
                np.testing.assert_array_equal(vector[rows], positions[:, 0])
                vector_offset = rows.stop

                opened = np.zeros(index_map.shape)
                opened[has_bixel] = info.aperture_fluence(idx, m)[
                    index_map[has_bixel] - grid.bixel_offset
                ]
                np.testing.assert_array_equal(
                    opened, shapes_mat[:, :, m] > 0, err_msg=f"{where}: shape mismatch"
                )

    return _assert
