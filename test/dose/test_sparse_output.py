"""Safe sparse index sizing, growth, promotion and shared quantity structure."""

from types import SimpleNamespace

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import cupy_available
from pyRadPlan.scenarios import ScenarioSet, ScenarioBasis, RangeUncertainty
from pyRadPlan.dose.engines import PhotonPencilBeamSVDEngine
from pyRadPlan.dose.engines._base import DoseEngineBase
from pyRadPlan.dose.engines import _sparse_output


@pytest.mark.parametrize("counts", [(10, 20, 0), (2**31 - 1, 2, 3), (2, 3, 2**31 - 1)])
def test_int32_limits(counts):
    """Dimensions and counts through the signed int32 limit fit compact storage."""
    assert _sparse_output.sparse_index_dtype(*counts) == np.int32


@pytest.mark.parametrize("counts", [(2**31, 2, 0), (2, 2**31, 0), (2, 2, 2**31)])
def test_int64_limits(counts):
    """Any dimension or count above the signed int32 limit requires int64."""
    assert _sparse_output.sparse_index_dtype(*counts) == np.int64


def test_invalid_counts():
    """Reject negative dimensions or entry counts."""
    with pytest.raises(ValueError, match="nonnegative"):
        _sparse_output.sparse_index_dtype(10, 2, -1)


def test_real_pointer_boundary():
    """Exercise the actual int32 boundary without allocating billions of entries."""
    maximum = np.iinfo(np.int32).max
    structure = dict(indices=np.array([7, 9], np.int32), indptr=np.array([0, maximum], np.int32))
    original = structure["indices"]
    _sparse_output.ensure_index_capacity(structure, int(maximum))
    assert structure["indices"] is original
    _sparse_output.ensure_index_capacity(structure, int(maximum) + 1)
    assert structure["indices"].dtype == structure["indptr"].dtype == np.int64
    np.testing.assert_array_equal(structure["indices"], [7, 9])
    np.testing.assert_array_equal(structure["indptr"], [0, maximum])
    structure["indptr"][1] = int(maximum) + 1
    assert structure["indptr"][1] == 2**31


def make_engine(rows=8, columns=3, direct=False):
    """Create the minimal collector state without unrelated patient setup."""
    engine = object.__new__(PhotonPencilBeamSVDEngine)
    engine._scenarios = ScenarioSet.from_points(
        [RangeUncertainty()],
        {"range": [[0, 0], [0, 0], [1, 0]]},
        bases=[None, ScenarioBasis(coefficients={0: 0.5, 2: 0.5}), None],
        probabilities=np.ones(3),
    )
    engine.dose_grid = SimpleNamespace(num_voxels=rows)
    engine._num_of_columns_dij = columns
    engine._calc_dose_direct = direct
    engine._computed_quantities = []
    engine.assumed_sparsity = 0.0  # Exercise growth from zero capacity.
    engine.geometric_lateral_cutoff = 50.0
    engine._dij_guarantee_nonzero = True
    engine._dij_guarantee_canonical = True
    engine.keep_rad_depth_cubes = False
    return engine


@pytest.mark.parametrize("backend", ["numpy", "cupy", "array_api_strict"])
@pytest.mark.parametrize("promote", [False, True])
def test_collect_columns_and_quantities(backend, promote, monkeypatch):
    """Growth, an empty column and promotion preserve values and shared structure."""
    if backend == "cupy" and not cupy_available():
        pytest.skip("CuPy unavailable")
    module = pytest.importorskip(backend)
    xp = array_api_compat.array_namespace(module.asarray([0.0]))
    engine = make_engine()
    dij = dict(num_of_beams=1, beam_num=np.zeros(3), ray_num=np.zeros(3), bixel_num=np.zeros(3))
    engine._allocate_quantity_matrices(dij, ["physical_dose"])
    original = dij["shared_structure"].flat[0]
    engine._allocate_quantity_matrices(dij, ["let_dose"])
    assert dij["shared_structure"].flat[0] is original
    assert dij["shared_structure"].flat[1] is None
    if promote:
        monkeypatch.setattr(_sparse_output, "_INT32_MAX", 4)
    for scenario in (0, 2):
        for column, rows in enumerate(([0, 3, 7], [], [1, 4, 6])):
            values = np.arange(len(rows), dtype=np.float32) + column + scenario + 1
            bixel = dict(
                ix=xp.asarray(rows, dtype=xp.int64),
                physical_dose=xp.asarray(values),
                let_dose=xp.asarray(values * 2),
                weight=1,
            )
            engine._fill_dij(bixel, dij, None, scenario, 0, column, 0, column)
        structure = dij["shared_structure"].flat[scenario]
        assert structure["indices"].dtype == (np.int64 if promote else np.int32)
        assert structure["indices"].dtype == structure["indptr"].dtype
        np.testing.assert_array_equal(structure["indptr"], [0, 3, 3, 6])
    monkeypatch.setattr(DoseEngineBase, "_finalize_dose", lambda self, dij: dij)
    result = engine._finalize_dose(dij)
    for scenario in (0, 2):
        dose, let = (result[q].flat[scenario] for q in ("physical_dose", "let_dose"))
        assert np.shares_memory(dose.indices, let.indices)
        assert np.shares_memory(dose.indptr, let.indptr)
        np.testing.assert_array_equal(dose.indices, [0, 3, 7, 1, 4, 6])
        np.testing.assert_array_equal(dose.data, np.array([1, 2, 3, 3, 4, 5]) + scenario)
        np.testing.assert_array_equal(let.data, 2 * dose.data)
        assert dose.has_canonical_format
        np.testing.assert_array_equal(dose.tocsr().toarray(), dose.toarray())


def test_large_grid_starts_with_int64(monkeypatch):
    """A huge sparse shape needs wide row indices even with few actual entries."""
    engine = make_engine(rows=2**31 + 3)
    dij = dict(num_of_beams=1, beam_num=np.zeros(3), ray_num=np.zeros(3), bixel_num=np.zeros(3))
    engine._allocate_quantity_matrices(dij, ["physical_dose"])
    bixel = dict(ix=np.array([2**31 + 1]), physical_dose=np.array([2.0], np.float32))
    engine._fill_dij(bixel, dij, None, 0, 0, 0, 0, 0)
    structure = dij["shared_structure"].flat[0]
    assert structure["indices"].dtype == structure["indptr"].dtype == np.int64
    assert structure["indices"][0] == 2**31 + 1
    for column in (1, 2):
        engine._fill_dij(
            dict(ix=np.array([], dtype=np.int64), physical_dose=np.array([], np.float32)),
            dij,
            None,
            0,
            0,
            column,
            0,
            column,
        )
    monkeypatch.setattr(DoseEngineBase, "_finalize_dose", lambda self, dij: dij)
    matrix = engine._finalize_dose(dij)["physical_dose"].flat[0]
    assert matrix.indices.dtype == matrix.indptr.dtype == np.int64
    assert matrix[2**31 + 1, 0] == 2.0


def test_direct_dose_unchanged():
    """Weighted direct-dose accumulation bypasses sparse index sizing."""
    engine = make_engine(direct=True)
    dij = dict(num_of_beams=1, beam_num=np.zeros(1), ray_num=np.zeros(1), bixel_num=np.zeros(1))
    engine._allocate_quantity_matrices(dij, ["physical_dose"])
    for _ in range(2):
        engine._fill_dij(
            dict(ix=np.array([1, 4]), physical_dose=np.array([2, 3]), weight=0.5),
            dij,
            None,
            0,
            0,
            0,
            0,
            0,
        )
    np.testing.assert_array_equal(dij["physical_dose"].flat[0][:, 0], [0, 2, 0, 0, 3, 0, 0, 0])
