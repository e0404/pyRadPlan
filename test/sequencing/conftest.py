import pytest
import numpy as np

from pyRadPlan.plan import validate_pln
from pyRadPlan.stf import validate_stf
from pyRadPlan.dij import validate_dij
from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst
from pyRadPlan.result import validate_result


@pytest.fixture
def test_data_engel(test_data_engel_seq_raw):
    pln_mat = validate_pln(test_data_engel_seq_raw["pln"])
    stf_mat = validate_stf(test_data_engel_seq_raw["stf"])
    dij_mat = validate_dij(test_data_engel_seq_raw["dij"])
    ct_mat = validate_ct(test_data_engel_seq_raw["ct"])
    cst_mat = validate_cst(test_data_engel_seq_raw["cst"], ct=ct_mat)
    result_mat = validate_result(test_data_engel_seq_raw["resultGUI_before_seq"])
    result_seq_mat = validate_result(test_data_engel_seq_raw["resultGUI"])

    return pln_mat, stf_mat, dij_mat, ct_mat, cst_mat, result_mat, result_seq_mat


@pytest.fixture
def test_data_siochi(test_data_siochi_seq_raw):
    pln_mat = validate_pln(test_data_siochi_seq_raw["pln"])
    stf_mat = validate_stf(test_data_siochi_seq_raw["stf"])
    dij_mat = validate_dij(test_data_siochi_seq_raw["dij"])
    ct_mat = validate_ct(test_data_siochi_seq_raw["ct"])
    cst_mat = validate_cst(test_data_siochi_seq_raw["cst"], ct=ct_mat)
    result_mat = validate_result(test_data_siochi_seq_raw["resultGUI_before_seq"])
    result_seq_mat = validate_result(test_data_siochi_seq_raw["resultGUI"])

    return pln_mat, stf_mat, dij_mat, ct_mat, cst_mat, result_mat, result_seq_mat


@pytest.fixture
def test_data_xia(test_data_xia_seq_raw):
    pln_mat = validate_pln(test_data_xia_seq_raw["pln"])
    stf_mat = validate_stf(test_data_xia_seq_raw["stf"])
    dij_mat = validate_dij(test_data_xia_seq_raw["dij"])
    ct_mat = validate_ct(test_data_xia_seq_raw["ct"])
    cst_mat = validate_cst(test_data_xia_seq_raw["cst"], ct=ct_mat)
    result_mat = validate_result(test_data_xia_seq_raw["resultGUI_before_seq"])
    result_seq_mat = validate_result(test_data_xia_seq_raw["resultGUI"])

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
