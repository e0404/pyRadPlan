import numpy as np
import pytest
from pydantic import ValidationError

from pyRadPlan.machines.particles import ParticlePencilBeamKernel

N_DEPTHS = 7


def _kernel(**extra):
    depths = np.linspace(0.0, 30.0, N_DEPTHS)
    return ParticlePencilBeamKernel(
        energy=100.0, depths=depths, Z=np.ones(N_DEPTHS), sigma=np.ones(N_DEPTHS), **extra
    )


@pytest.mark.parametrize("transposed", [False, True])
def test_multi_gaussian_kernel_orientation(transposed):
    sigma_multi = np.arange(3 * N_DEPTHS, dtype=float).reshape(3, N_DEPTHS) + 1
    weight_multi = np.full((2, N_DEPTHS), 0.1)
    if transposed:
        sigma_multi, weight_multi = sigma_multi.T, weight_multi.T
    kernel = _kernel(sigma_multi=sigma_multi, weight_multi=weight_multi)
    assert kernel.sigma_multi.shape == (3, N_DEPTHS)
    assert kernel.weight_multi.shape == (2, N_DEPTHS)
    assert kernel.sigma_multi[1, 0] == pytest.approx(N_DEPTHS + 1)


def test_multi_gaussian_kernel_single_weight():
    kernel = _kernel(sigma_multi=np.ones((2, N_DEPTHS)), weight_multi=np.full(N_DEPTHS, 0.3))
    assert kernel.weight_multi.shape == (N_DEPTHS,)


def test_multi_gaussian_kernel_errors():
    with pytest.raises(ValidationError, match="depth data length"):
        _kernel(sigma_multi=np.ones((2, N_DEPTHS + 1)), weight_multi=np.ones(N_DEPTHS))
    with pytest.raises(ValidationError, match="one sigma more than weights"):
        _kernel(sigma_multi=np.ones((3, N_DEPTHS)), weight_multi=np.ones(N_DEPTHS))
