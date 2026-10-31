"""Smooth approximations of the maximum over scenarios (matRad's ``useMaxApprox``)."""

from typing import Literal

import array_api_compat

from ...core.xp_utils.typing import Array

MaxApproximation = Literal["logsumexp", "pnorm", "none"]


def smooth_max(
    values: Array, method: MaxApproximation = "logsumexp", p: float = 30.0
) -> tuple[Array, Array]:
    """
    Smooth maximum of per-scenario values and its gradient.

    Parameters
    ----------
    values : Array
        One-dimensional per-scenario values ``f_s``.
    method : {"logsumexp", "pnorm", "none"}
        ``"logsumexp"``: matRad's stable form ``f_max + log1p(sum_{s != argmax} exp(f_s -
        f_max))``. ``"pnorm"``: matRad's normalized form ``f_max (1/n)^(1/p) ||f / f_max||_p``,
        falling back to the exact maximum when ``f_max <= 0``. ``"none"``: the exact maximum
        with a one-hot gradient at the first maximizer.
    p : float
        Exponent of the ``"pnorm"`` approximation.

    Returns
    -------
    tuple[Array, Array]
        The scalar value and the gradient ``d value / d f_s`` (same shape as ``values``).

    Raises
    ------
    ValueError
        If ``method`` is unknown or ``values`` is not one-dimensional.
    """
    if method not in ("logsumexp", "pnorm", "none"):
        raise ValueError(
            f"Unknown maximum approximation {method!r}; use 'logsumexp', 'pnorm' or 'none'"
        )
    xp = array_api_compat.array_namespace(values)
    if values.ndim != 1:
        raise ValueError(f"smooth_max expects a 1-D array, got shape {tuple(values.shape)}")
    n = values.shape[0]
    ix_max = xp.argmax(values)
    f_max = values[ix_max]
    is_max = xp.arange(n, device=array_api_compat.device(values)) == ix_max

    if method == "logsumexp":
        tmp = xp.exp(values - f_max)
        exp_sum = xp.sum(xp.where(is_max, xp.zeros_like(tmp), tmp))
        return f_max + xp.log1p(exp_sum), tmp / (1.0 + exp_sum)

    if method == "pnorm" and float(f_max) > 0.0:
        tmp = values / f_max
        p_norm_val = xp.sum(tmp**p) ** (1.0 / p)
        fac = (1.0 / n) ** (1.0 / p)
        return f_max * fac * p_norm_val, fac * (tmp / p_norm_val) ** (p - 1.0)

    return f_max, xp.astype(is_max, values.dtype)
