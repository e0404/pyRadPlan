from pyRadPlan.sequencing import run_sequencing


def test_fluence_sequencing(test_data_xia, test_data_xia_seq_raw, assert_matches_matrad):
    pln_mat, stf_mat, dij_mat, *_, result_mat, result_seq_mat = test_data_xia
    result_seq = run_sequencing(result_mat, pln_mat, stf_mat, dij=dij_mat)

    assert_matches_matrad(
        result_seq, result_seq_mat, pln_mat, stf_mat, test_data_xia_seq_raw["resultGUI"]
    )
