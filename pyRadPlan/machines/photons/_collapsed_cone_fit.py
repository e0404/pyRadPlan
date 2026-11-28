"""Refit of collapsed cone kernels into polar bins (port of ``matRad_fitCollapsedConeKernel``)."""

import os
from typing import Union

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from ._collapsed_cone_kernel import PhotonCollapsedConeKernel, load_collapsed_cone_kernel


def load_kernel(
    filename: Union[str, os.PathLike, None],
    num_polar_angles: int,
    num_scatter_exponentials: int = 1,
) -> PhotonCollapsedConeKernel:
    """
    Load a kernel and refit it if another number of polar bins or scatter terms is requested.

    Parameters
    ----------
    filename : str or PathLike or None
        Kernel file, None for the default 6 MV kernel.
    num_polar_angles : int
        Number of polar bins.
    num_scatter_exponentials : int
        Number of exponential terms of the scatter component (1 or 2).

    Returns
    -------
    PhotonCollapsedConeKernel
        The kernel, with the nominal energy and the conical data of the file.
    """
    stored = load_collapsed_cone_kernel(filename)
    if (
        stored.num_polar_angles == num_polar_angles
        and stored.num_scatter_exponentials == num_scatter_exponentials
    ):
        return stored
    if stored.cones is None:
        raise ValueError(
            f"Kernel file {filename} cannot be refitted, it contains no conical kernel data."
        )
    kernel = fit_collapsed_cone_kernel(
        stored.cones,
        num_polar_angles,
        stored.min_radius,
        num_scatter_exponentials,
        stored.scatter_max_radius,
    )
    return kernel.model_copy(
        update={"nominal_energy": stored.nominal_energy, "cones": stored.cones}
    )


def fit_collapsed_cone_kernel(
    edk: dict,
    num_polar_angles: int,
    min_radius: float = 0.0,
    num_scatter_exponentials: int = 1,
    scatter_max_radius: float = np.inf,
) -> PhotonCollapsedConeKernel:
    """
    Collapse a conical energy deposition kernel into polar bins and fit exponentials.

    For each polar bin the primary and scatter energy fractions are taken from the Monte
    Carlo data (no renormalization) and the attenuation coefficients (and for two scatter
    terms the split of the scatter energy) are fitted to the normalized cumulative radial
    energy distribution of each component.

    Parameters
    ----------
    edk : dict
        Conical kernel with ``angleEdges`` [deg], ``radiusEdges`` [mm], ``primary`` and
        ``scatter`` (``n_cones x n_shells``).
    num_polar_angles : int
        Number of polar bins, must divide the number of cones.
    min_radius : float
        Smallest radius [mm] included in the fit and in the residuals.
    num_scatter_exponentials : int
        Number of exponential terms of the scatter component (1 or 2).
    scatter_max_radius : float
        Largest radius [mm] included in the fit of the scatter component.

    Returns
    -------
    PhotonCollapsedConeKernel
        The fitted kernel. ``residual`` holds per bin the maximum absolute error of the
        normalized cumulative energy for r >= min_radius (``primaryCum``, ``scatterCum``),
        r <= 10 mm (``...CumNear``) and r >= 100 mm (``...CumFar``), and the negative
        fraction of the modeled energy beyond the kernel range (``...Energy``).
    """
    if num_scatter_exponentials not in (1, 2):
        raise ValueError("Only one or two scatter exponentials are supported.")

    # Polar bins of cones_per_bin cones, with the solid angle bisectors as polar angles
    primary = np.atleast_2d(edk["primary"])
    n_cones = primary.shape[0]
    if n_cones % num_polar_angles != 0:
        raise ValueError(
            f"Number of polar angles ({num_polar_angles}) must divide the number of kernel "
            f"cones ({n_cones})."
        )
    cones_per_bin = n_cones // num_polar_angles

    angle_edges = np.asarray(edk["angleEdges"], dtype=np.float64).ravel()
    polar_edges = angle_edges[::cones_per_bin]
    cos_edges = np.cos(np.deg2rad(polar_edges))
    polar_angles = np.rad2deg(np.arccos(0.5 * (cos_edges[:-1] + cos_edges[1:])))

    # Residual masks: whole fit range, near (r <= 10 mm) and far (r >= 100 mm)
    r_outer = np.asarray(edk["radiusEdges"], dtype=np.float64).ravel()[1:]
    in_fit = r_outer >= min_radius
    masks = (in_fit, in_fit & (r_outer <= 10), in_fit & (r_outer >= 100))
    suffix = ("Cum", "CumNear", "CumFar")

    results = {}
    residual = {}
    for comp, num_terms, max_radius in (
        ("primary", 1, np.inf),
        ("scatter", num_scatter_exponentials, scatter_max_radius),
    ):
        val = np.atleast_2d(edk[comp])
        weight = np.zeros((num_terms, num_polar_angles))
        att = np.ones((num_terms, num_polar_angles))
        err = np.zeros((3, num_polar_angles))
        tail = np.zeros(num_polar_angles)
        for p in range(num_polar_angles):
            # Energy of the bin per radial shell and its normalized cumulative distribution
            y = val[p * cones_per_bin : (p + 1) * cones_per_bin, :].sum(axis=0)
            energy = y.sum()
            if energy <= 0:
                continue
            cum_data = np.cumsum(y) / energy
            sel = in_fit & (r_outer <= max_radius)
            frac, att[:, p] = _fit_cumulative_exponentials(r_outer[sel], cum_data[sel], num_terms)
            # The weights keep the Monte Carlo energy of the bin (no renormalization)
            weight[:, p] = energy * frac
            model = 1 - frac @ np.exp(-np.outer(att[:, p], r_outer))
            for m, mask in enumerate(masks):
                if np.any(mask):
                    err[m, p] = np.max(np.abs(model[mask] - cum_data[mask]))
            tail[p] = -frac @ np.exp(-att[:, p] * r_outer[-1])
        results[comp] = (weight, att)
        for m in range(3):
            residual[comp + suffix[m]] = err[m]
        residual[comp + "Energy"] = tail

    return PhotonCollapsedConeKernel(
        polar_angles=polar_angles,
        polar_edges=polar_edges,
        A=results["primary"][0],
        a=results["primary"][1],
        B=results["scatter"][0],
        b=results["scatter"][1],
        min_radius=float(min_radius),
        scatter_max_radius=float(scatter_max_radius),
        residual=residual,
    )


def _fminbnd(fun, lo: float, hi: float, xatol: float) -> tuple[float, float]:
    """
    Minimize a scalar function on an interval, like MATLAB's ``fminbnd``.

    Parameters
    ----------
    fun : callable
        Function of one variable.
    lo, hi : float
        Interval.
    xatol : float
        Absolute tolerance of the minimizer.

    Returns
    -------
    tuple[float, float]
        Minimizer and minimum.
    """
    if lo == hi:
        return lo, float(fun(lo))
    res = minimize_scalar(fun, bounds=(lo, hi), method="bounded", options={"xatol": xatol})
    return float(res.x), float(res.fun)


def _fit_cumulative_exponentials(
    r: np.ndarray, cum_data: np.ndarray, num_terms: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit ``1 - sum_k f_k exp(-a_k r)`` to a normalized cumulative radial distribution.

    Parameters
    ----------
    r : np.ndarray
        Outer radii of the shells [mm].
    cum_data : np.ndarray
        Normalized cumulative energy at ``r``.
    num_terms : int
        Number of exponential terms (1 or 2).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Fractions ``f_k`` (summing to 1) and attenuation coefficients ``a_k`` [1/mm], in
        descending order of the coefficients.
    """
    # One term: grid search of the log coefficient, refined on the neighboring interval
    if num_terms == 1:

        def cost(log_att):
            return float(np.sum((1 - np.exp(-np.exp(log_att) * r) - cum_data) ** 2))

        log_grid = np.linspace(np.log(1e-5), np.log(20), 300)
        i_best = int(np.argmin([cost(lg) for lg in log_grid]))
        lo = log_grid[max(i_best - 1, 0)]
        hi = log_grid[min(i_best + 1, log_grid.size - 1)]
        log_att, _ = _fminbnd(cost, lo, hi, 1e-10)
        return np.ones(1), np.array([np.exp(log_att)])

    # Two terms: grid search of both log coefficients (the fraction is solved for), refined
    # with Nelder-Mead as matRad's fminsearch
    log_grid = np.linspace(np.log(1e-5), np.log(20), 80)
    best_cost = np.inf
    best_log = np.zeros(2)
    for i in range(log_grid.size):
        for j in range(i):
            c, _ = _two_term_cost(log_grid[[i, j]], r, cum_data)
            if c < best_cost:
                best_cost = c
                best_log = log_grid[[i, j]]
    res = minimize(
        lambda la: _two_term_cost(la, r, cum_data)[0],
        best_log,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-14, "maxfev": 4000, "maxiter": 4000},
    )
    _, f1 = _two_term_cost(res.x, r, cum_data)
    att = np.exp(res.x)
    frac = np.array([f1, 1 - f1])
    order = np.argsort(-att, kind="stable")
    return frac[order], att[order]


def _two_term_cost(log_att, r, cum_data) -> tuple[float, float]:
    """
    Least squares cost of a two term fit with the optimal fraction of the first term.

    Parameters
    ----------
    log_att : array_like
        Logarithms of the two attenuation coefficients.
    r : np.ndarray
        Outer radii of the shells [mm].
    cum_data : np.ndarray
        Normalized cumulative energy at ``r``.

    Returns
    -------
    tuple[float, float]
        The cost and the fraction of the first term, clipped to [0, 1].
    """
    e1 = np.exp(-np.exp(log_att[0]) * r)
    e2 = np.exp(-np.exp(log_att[1]) * r)
    target = 1 - cum_data - e2
    basis = e1 - e2
    f1 = np.sum(target * basis) / max(np.sum(basis**2), np.finfo(float).tiny)
    f1 = min(max(f1, 0.0), 1.0)
    cost = np.sum((f1 * e1 + (1 - f1) * e2 - (1 - cum_data)) ** 2)
    return float(cost), float(f1)
