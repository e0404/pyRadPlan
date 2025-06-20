import numpy as np

from pyRadPlan.result import (
    PlanResult,
    Sequence,
)
from pyRadPlan.sequencing import (
    run_sequencing,
)
from pyRadPlan.result._beam_sequence import BeamSequence


def test_fluence_sequencing(test_data_xia):
    pln_mat, stf_mat, dij_mat, *_, result_mat, result_seq_mat = test_data_xia
    result_seq = run_sequencing(result_mat, pln_mat, stf_mat, dij=dij_mat)

    # --- basic type checks ---
    assert isinstance(result_seq, PlanResult)
    assert isinstance(result_seq.sequence, Sequence)

    seq, seq_mat = result_seq.sequence, result_seq_mat.sequence

    # --- scalar attributes ---
    scalar_attrs = [
        "bixel_width",
        "num_of_mlc_leaf_pairs",
        "total_num_of_bixels",
        "total_num_of_leaf_pairs",
        "total_num_of_shapes",
    ]
    for attr in scalar_attrs:
        got = getattr(seq, attr)
        exp = getattr(seq_mat, attr)
        assert got == exp, f"{attr}: got {got}, expected {exp}"

    # --- sequence array comparisons ---
    allclose_sequence = [
        ("fluence", 1e-4),
        ("lim_mx", 1e-4),
        ("mapping_mx", 1e-4),
        ("aperture_vector", 1e-4),
    ]

    for attr, tol in allclose_sequence:
        arr1 = getattr(seq, attr)
        arr2 = getattr(seq_mat, attr)
        np.testing.assert_allclose(arr1, arr2, atol=tol, err_msg=f"sequence.{attr} mismatch")

    # --- per‐beam checks ---
    # Check every beam. Check of one beam is not sufficient, as some asymmetries exist in the test data
    array_equal_beam = [
        "shapes",
        "sum_shapes",
        "bixel_ix",
        "fluence",
        "is_active_leaf_pair",
        "bixel_ind_map",
        "shape_map",
        "vector_offset",
    ]
    allclose_beam = [
        ("shapes_weight", 1e-4),
        ("leaf_pair_pos", 1e-4),
        ("pos_of_corner_bixel", 1e-4),
        ("mlc_window", 1e-4),
        ("lim_l", 1e-4),
        ("lim_r", 1e-4),
        ("weight", 1e-4),
        ("left_leaf_pos", 1e-4),
        ("right_leaf_pos", 1e-4),
    ]

    for idx, (beam, beam_mat) in enumerate(zip(seq.beam, seq_mat.beam)):
        assert isinstance(beam, BeamSequence)
        assert isinstance(beam_mat, BeamSequence)

        # simple scalars
        for attr in ["num_of_shapes", "num_of_active_leaf_pairs", "central_leaf_pair"]:
            got = getattr(beam, attr)
            exp = getattr(beam_mat, attr)
            assert got == exp, f"beam[{idx}].{attr}: got {got}, expected {exp}"

        # exact‐array fields
        for attr in array_equal_beam:
            arr1 = getattr(beam, attr)
            arr2 = getattr(beam_mat, attr)
            np.testing.assert_array_equal(arr1, arr2, err_msg=f"beam[{idx}].{attr} mismatch")

        # tolerance‐based fields
        for attr, tol in allclose_beam:
            arr1 = getattr(beam, attr)
            arr2 = getattr(beam_mat, attr)
            np.testing.assert_allclose(
                arr1, arr2, atol=tol, err_msg=f"beam[{idx}].{attr} mismatch"
            )
