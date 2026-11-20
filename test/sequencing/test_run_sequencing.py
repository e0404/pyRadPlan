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


def _dose(result, name="physical_dose_sequenced"):
    return sitk.GetArrayFromImage(result.quantities[name].values)


def test_sequenced_dose_follows_the_dose_convention_of_the_plan(test_data_xia):
    pln_mat, stf_mat, dij_mat, _ct_mat, _cst_mat, result_mat, _result_seq_mat = test_data_xia
    per_fraction = pln_mat.model_copy(update={"dose_convention": "per_fraction"})
    total = pln_mat.model_copy(update={"dose_convention": "total", "num_of_fractions": 30})

    one = run_sequencing(result_mat.model_copy(), per_fraction, stf_mat, dij=dij_mat)
    course = run_sequencing(result_mat.model_copy(), total, stf_mat, dij=dij_mat)

    np.testing.assert_allclose(_dose(course), 30 * _dose(one), rtol=1e-5)
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(course.beam_quantities["physical_dose_sequenced"][0].values),
        30 * sitk.GetArrayFromImage(one.beam_quantities["physical_dose_sequenced"][0].values),
        rtol=1e-5,
    )


def test_resequencing_without_dij_drops_the_doses_of_the_previous_sequence(test_data_xia):
    pln_mat, stf_mat, dij_mat, _ct_mat, _cst_mat, result_mat, _result_seq_mat = test_data_xia

    first = run_sequencing(result_mat, pln_mat, stf_mat, dij=dij_mat)
    assert "physical_dose_sequenced" in first.beam_quantities
    levels = first.sequence.num_of_levels
    pln_other = pln_mat.model_copy(
        update={"prop_seq": {**pln_mat.prop_seq, "num_of_levels": levels + 2}}
    )
    second = run_sequencing(first, pln_other, stf_mat)

    assert second.sequence.num_of_levels == levels + 2
    assert "physical_dose" in second.quantities
    assert "physical_dose_sequenced" not in second.quantities
    assert "physical_dose_sequenced" not in second.beam_quantities

    # With the dose influence, the doses are those of the new sequence
    third = run_sequencing(first, pln_other, stf_mat, dij=dij_mat)
    assert not np.array_equal(_dose(third), _dose(first))


def test_sequencing_leaves_the_sequencing_settings_of_the_plan(test_data_xia):
    pln_mat, stf_mat, dij_mat, _ct_mat, _cst_mat, result_mat, _result_seq_mat = test_data_xia
    pln = pln_mat.model_copy(update={"prop_seq": {"sequencer": "xia", "num_of_levels": 4}})

    run_sequencing(result_mat, pln, stf_mat, dij=dij_mat)

    assert pln.prop_seq == {"sequencer": "xia", "num_of_levels": 4}


def test_sequenced_result_survives_a_dump(test_data_xia):
    from pyRadPlan.machines import Jaw
    from pyRadPlan.result import validate_result

    pln_mat, stf_mat, dij_mat, _ct_mat, _cst_mat, result_mat, _result_seq_mat = test_data_xia
    result_seq = run_sequencing(result_mat, pln_mat, stf_mat, dij=dij_mat)

    dumped = result_seq.model_dump()
    jaw = dumped["sequence"]["beam"][0]["extra_blds"][0][0]
    assert jaw["positions"] == result_seq.sequence.beam[0].extra_blds[0][0].positions
    restored = validate_result(dumped)

    for beam, original in zip(restored.sequence.beam, result_seq.sequence.beam):
        assert all(isinstance(jaw, Jaw) for jaws in beam.extra_blds for jaw in jaws)
        assert beam.extra_blds == original.extra_blds
        assert beam.mlc_sequence == original.mlc_sequence


def test_sequenced_result_survives_a_mat_file(test_data_xia, tmp_path):
    from pyRadPlan.io import load_data, save_data

    pln_mat, stf_mat, dij_mat, ct_mat, _cst_mat, result_mat, _result_seq_mat = test_data_xia
    result_seq = run_sequencing(result_mat, pln_mat, stf_mat, dij=dij_mat)

    save_data(file_name=tmp_path / "sequenced.mat", result=result_seq, ct=ct_mat)
    loaded = load_data(str(tmp_path / "sequenced.mat"))["result"]

    assert len(loaded.sequence.beam) == len(result_seq.sequence.beam)
    for beam, original in zip(loaded.sequence.beam, result_seq.sequence.beam):
        assert beam.extra_blds == original.extra_blds
        assert beam.bixel_ix == original.bixel_ix
        for mlc, original_mlc in zip(beam.mlc_sequence, original.mlc_sequence, strict=True):
            np.testing.assert_allclose(mlc.leaf_positions, original_mlc.leaf_positions)
