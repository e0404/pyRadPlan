"""Smooth maximum approximations against matRad's formulas."""

import numpy as np
import pytest
import array_api_strict as xp

from pyRadPlan.optimization.robustness import smooth_max

VALUES = np.array([0.7, 2.0, 1.3, 1.9])


def _fd(method, values, p=30.0, eps=1e-7):
    grad = np.zeros_like(values)
    for s in range(values.size):
        plus, minus = values.copy(), values.copy()
        plus[s] += eps
        minus[s] -= eps
        f_plus = float(smooth_max(xp.asarray(plus), method, p)[0])
        f_minus = float(smooth_max(xp.asarray(minus), method, p)[0])
        grad[s] = (f_plus - f_minus) / (2 * eps)
    return grad


def test_logsumexp_matches_matrad():
    value, grad = smooth_max(xp.asarray(VALUES), "logsumexp")
    f_max = VALUES.max()
    others = np.delete(VALUES, VALUES.argmax())
    exp_sum = np.sum(np.exp(others - f_max))
    assert float(value) == pytest.approx(f_max + np.log1p(exp_sum))
    assert float(value) == pytest.approx(np.log(np.sum(np.exp(VALUES))))
    np.testing.assert_allclose(np.asarray(grad), np.exp(VALUES - f_max) / (1 + exp_sum))
    np.testing.assert_allclose(np.asarray(grad), _fd("logsumexp", VALUES), rtol=1e-6)
    assert float(np.sum(np.asarray(grad))) == pytest.approx(1.0)


def test_logsumexp_is_stable_for_large_values():
    values = np.array([1000.0, 999.0, 0.0])
    value, grad = smooth_max(xp.asarray(values), "logsumexp")
    assert np.isfinite(float(value))
    assert float(value) == pytest.approx(1000.0 + np.log1p(np.exp(-1.0) + np.exp(-1000.0)))
    assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.parametrize("p", [30.0, 4.0])
def test_pnorm_matches_matrad(p):
    value, grad = smooth_max(xp.asarray(VALUES), "pnorm", p)
    n = VALUES.size
    f_max = VALUES.max()
    tmp = VALUES / f_max
    p_norm_val = np.sum(tmp**p) ** (1 / p)
    fac = (1 / n) ** (1 / p)
    assert float(value) == pytest.approx(f_max * fac * p_norm_val)
    assert float(value) == pytest.approx(np.mean(VALUES**p) ** (1 / p))
    np.testing.assert_allclose(np.asarray(grad), fac * (tmp / p_norm_val) ** (p - 1))
    np.testing.assert_allclose(np.asarray(grad), _fd("pnorm", VALUES, p), rtol=1e-5, atol=1e-9)


def test_pnorm_falls_back_to_exact_max_without_positive_values():
    value, grad = smooth_max(xp.asarray(np.zeros(3)), "pnorm")
    assert float(value) == 0.0
    np.testing.assert_array_equal(np.asarray(grad), [1.0, 0.0, 0.0])
    value, _ = smooth_max(xp.asarray(np.array([-2.0, -1.0])), "pnorm")
    assert float(value) == -1.0


def test_exact_max_one_hot_first_maximizer():
    value, grad = smooth_max(xp.asarray(VALUES), "none")
    assert float(value) == 2.0
    np.testing.assert_array_equal(np.asarray(grad), [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(grad), _fd("none", VALUES), atol=1e-8)
    _, grad = smooth_max(xp.asarray(np.array([1.0, 3.0, 3.0])), "none")
    np.testing.assert_array_equal(np.asarray(grad), [0.0, 1.0, 0.0])


def test_smooth_max_keeps_namespace_and_validates():
    value, grad = smooth_max(xp.asarray(VALUES), "logsumexp")
    assert type(value) is type(xp.asarray(1.0))
    assert grad.shape == (4,)
    with pytest.raises(ValueError, match="Unknown maximum approximation"):
        smooth_max(xp.asarray(VALUES), "softmax")
    with pytest.raises(ValueError, match="1-D"):
        smooth_max(xp.asarray(np.ones((2, 2))), "none")
