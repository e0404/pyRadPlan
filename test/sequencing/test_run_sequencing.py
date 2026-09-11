import logging

import numpy as np
import SimpleITK as sitk

from pyRadPlan.sequencing import run_sequencing
from pyRadPlan.sequencing.sequencer import get_sequencer_from_pln


def test_run_sequencing_stamps_sequencer_short_name(test_data_xia):
    pln_mat, stf_mat, dij_mat, _ct_mat, _cst_mat, result_mat, _result_seq_mat = test_data_xia

    result_seq = run_sequencing(result_mat, pln_mat, stf_mat, dij=dij_mat)

    assert result_seq.sequence.sequencer == "xia"


def test_sequencing_keeps_optimized_dose_and_adds_sequenced_one(test_data_xia):
    pln_mat, stf_mat, dij_mat, _ct_mat, _cst_mat, result_mat, _result_seq_mat = test_data_xia
    optimized = result_mat.quantities["physical_dose"]

    result_seq = run_sequencing(result_mat, pln_mat, stf_mat, dij=dij_mat)

    assert "physical_dose" in result_seq.quantities
    assert "physical_dose_sequenced" in result_seq.quantities
    # The optimized dose is carried through untouched...
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(result_seq.quantities["physical_dose"].values),
        sitk.GetArrayFromImage(optimized.values),
    )
    # ...and the delivered one actually differs from it (sequencing changed the fluence).
    assert not np.array_equal(
        sitk.GetArrayFromImage(result_seq.quantities["physical_dose_sequenced"].values),
        sitk.GetArrayFromImage(optimized.values),
    )


def test_run_sequencing_without_dij_sequences_and_warns(test_data_xia, caplog):
    pln_mat, stf_mat, _dij_mat, _ct_mat, _cst_mat, result_mat, _result_seq_mat = test_data_xia

    with caplog.at_level(logging.WARNING):
        result_seq = run_sequencing(result_mat, pln_mat, stf_mat)

    assert result_seq.sequence is not None
    assert "no delivered dose" in caplog.text
    assert "physical_dose" in result_seq.quantities
    assert "physical_dose_sequenced" not in result_seq.quantities


def test_run_sequencing_stamps_the_level_count(test_data_xia):
    pln_mat, stf_mat, dij_mat, _ct_mat, _cst_mat, result_mat, _result_seq_mat = test_data_xia

    result_seq = run_sequencing(result_mat, pln_mat, stf_mat, dij=dij_mat)

    assert result_seq.sequence.num_of_levels == get_sequencer_from_pln(pln_mat).num_of_levels
