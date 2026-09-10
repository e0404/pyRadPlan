from pyRadPlan.sequencing import run_sequencing


def test_run_sequencing_stamps_sequencer_short_name(test_data_xia):
    pln_mat, stf_mat, dij_mat, _ct_mat, _cst_mat, result_mat, _result_seq_mat = test_data_xia

    result_seq = run_sequencing(result_mat, pln_mat, stf_mat, dij=dij_mat)

    assert result_seq.sequence.sequencer == "xia"
