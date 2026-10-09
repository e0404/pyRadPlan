"""Biological evaluation on a non-default array backend / device.

Not every backend device object is hashable, and not every array can be combined across
devices, so the per-calculation caches and the model masking are exercised here on a real
GPU backend when one is available.
"""

import numpy as np
import pytest

from pyRadPlan.bio_models import (
    BioEvaluationContext,
    ExactClassLookup,
    KernelBasedLQModel,
    MCNamara,
    TabulatedAlphaBetaModel,
)
from pyRadPlan.core.xp_utils import cupy_available

from test_fragment_species_selection import _kernel

pytestmark = pytest.mark.skipif(not cupy_available(), reason="CuPy / CUDA is not available.")


@pytest.fixture
def xp():
    import array_api_compat.cupy as cp  # noqa: PLC0415

    return cp


def test_exact_class_lookup_caches_reference_classes_on_gpu(xp):
    lookup = ExactClassLookup([0.1, 0.5], [0.05, 0.05])
    alpha_x = xp.asarray([0.1, 0.5, 0.0])
    beta_x = xp.asarray([0.05, 0.05, 0.0])

    first = lookup.class_index(alpha_x, beta_x)
    second = lookup.class_index(alpha_x, beta_x)

    assert np.array_equal(xp.asnumpy(first), [0, 1, 0])
    assert np.array_equal(xp.asnumpy(second), [0, 1, 0])
    assert len(lookup._reference_cache) == 1


def test_kernel_based_evaluator_gathers_on_gpu(xp):
    class _Kernel:
        alpha_x = np.asarray([0.1, 0.5])
        beta_x = np.asarray([0.05, 0.05])

    class _Machine:
        energies = [100.0]
        pb_kernels = {100.0: _Kernel()}

    voxel_params = {
        "alpha_x": xp.asarray([[0.1], [0.5]]),
        "beta_x": xp.asarray([[0.05], [0.05]]),
    }
    evaluator = KernelBasedLQModel().evaluator(_Machine(), voxel_params)
    result = evaluator.evaluate(
        BioEvaluationContext(
            {
                "alpha_x": xp.asarray([0.1, 0.5]),
                "beta_x": xp.asarray([0.05, 0.05]),
                "alpha": xp.asarray([[1.0, 1.0], [2.0, 2.0]]),
                "beta": xp.asarray([[4.0, 4.0], [5.0, 5.0]]),
            }
        )
    )

    assert np.allclose(xp.asnumpy(result["alpha"]), [1.0, 2.0])
    assert np.allclose(xp.asnumpy(result["beta"]), [4.0, 5.0])


def test_tabulated_evaluator_converts_tables_once_per_device(xp):
    class _Machine:
        energies = [100.0]
        pb_kernels = {100.0: _kernel([(1, 1.0), (6, np.nan)])}

    machine = _Machine()
    evaluator = TabulatedAlphaBetaModel().evaluator(
        machine, {"alpha_x": xp.asarray([[0.1]]), "beta_x": xp.asarray([[0.05]])}
    )
    kernel = machine.pb_kernels[100.0].to_namespace(xp)

    first = evaluator.kernel_quantities(kernel)
    evaluator.kernel_quantities(kernel)

    assert set(first) == {"alpha", "sqrt_beta"}
    assert len(evaluator._converted) == 1
    assert np.allclose(xp.asnumpy(first["alpha"][0, :]), evaluator._tables[100.0]["alpha"][0])


def test_let_model_masks_undefined_voxels_on_gpu(xp):
    evaluator = MCNamara().evaluator(machine=None, voxel_params={})
    result = evaluator.evaluate_influence(
        BioEvaluationContext(
            {
                "alpha_x": xp.asarray([0.0, 0.1]),
                "beta_x": xp.asarray([0.0, 0.05]),
                "physical_dose": xp.asarray([1.0, 1.0]),
                "let": xp.asarray([3.0, 3.0]),
            }
        )
    )

    for name in ("alpha_dose", "sqrt_beta_dose"):
        values = xp.asnumpy(result[name])
        assert np.all(np.isfinite(values))
        assert values[0] == 0.0
