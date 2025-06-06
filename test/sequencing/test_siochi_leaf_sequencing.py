import pytest
import SimpleITK as sitk
import numpy as np
from pydantic import ValidationError
from copy import deepcopy

# from scipy.io import savemat
from pyRadPlan import fluence_optimization, calc_dose_influence
from pyRadPlan.result import (
    PlanResult,
    Sequence,
    QuantityDistribution,
    validate_result,
    create_result,
)
from pyRadPlan.sequencing import (
    SiochiLeafSequencer,
    run_sequencing,
)


def test_fluence_sequencing(test_data_siochi, to_py):
    pln_mat, stf_mat, dij_mat, _, _, result_mat, result_seq_mat = test_data_siochi
    result_seq = run_sequencing(result_mat, pln_mat, stf_mat, dij=dij_mat)
    assert isinstance(result_seq, PlanResult)
    assert isinstance(result_seq.sequence, Sequence)
    assert np.array_equal(result_seq.sequence.fluence, result_seq_mat.sequence.fluence)
    assert result_seq.sequence.beam == result_seq_mat.sequence.beam


# def test_aperture_info(test_data_siochi):
#     pln_mat, stf_mat, dij_mat, ct_mat, cst_mat, result_mat, result_seq_mat = test_data_siochi
#     result_seq = run_sequencing(result_mat, pln_mat, stf_mat, dij=dij_mat)
#     assert isinstance(result_seq.sequence.aperture_info, dict)
#     assert "aperture" in result_seq.sequence.aperture_info
#     assert "aperture_weight" in result_seq.sequence.aperture_info
#     assert "aperture_shape" in result_seq.sequence.aperture_info
#     assert "aperture_shape_weight" in result_seq.sequence.aperture_info
