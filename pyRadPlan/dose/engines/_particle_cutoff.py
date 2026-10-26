"""Array API quadrature for particle lateral-cutoff preparation."""

from dataclasses import dataclass
from typing import Callable

import array_api_compat
import numpy as np

from pyRadPlan.core.xp_utils import to_namespace
from pyRadPlan.core.xp_utils.typing import Array


@dataclass(frozen=True)
class CutoffQuadrature:
    """Fixed radial/angular samples, in the historical integration order.

    All arrays belong to the same namespace/device. Geometry is generated on the
    host once so changing execution backends does not change the sampling grid.
    """

    radii: Array
    dr: Array
    angular_weights: Array
    radial_dist_sq: Array
    lat_dists: Array

    @classmethod
    def create(cls, lateral_model):
        """Prepare the existing 1,200-bin radial and optional 360-angle grid."""
        edges = np.concatenate(([0], np.logspace(-1, 3, 1200)))
        radii = 0.5 * (edges[:-1] + edges[1:])
        dr = np.diff(edges)
        if lateral_model == "singleXY":
            angles = np.linspace(0, 2 * np.pi, 361)[:-1]
            angular_weights = (angles[1] - angles[0]) * np.ones_like(angles)
            radius_mesh, angle_mesh = np.meshgrid(radii, angles)
            angular_weights = np.tile(angular_weights, radii.size)
            radii = radius_mesh.ravel(order="F")
            dr = np.repeat(dr, angles.size)
            theta = angle_mesh.ravel(order="F")
            radial_dist_sq = radii**2
            lat_dists = np.sqrt(radial_dist_sq)[:, None] * np.column_stack(
                (np.cos(theta), np.sin(theta))
            )
        else:
            angular_weights = np.asarray(2 * np.pi)
            radial_dist_sq = radii**2
            lat_dists = np.tile(np.sqrt(radial_dist_sq / 2)[:, None], (1, 2))
        return cls(radii, dr, angular_weights, radial_dist_sq, lat_dists)

    def to_namespace(self, xp, device=None):
        """Transfer preparation once, outside the depth-batch loop."""
        return type(self)(
            *(to_namespace(xp, value, device=device) for value in vars(self).values())
        )


def evaluate_cutoff(
    depths: Array,
    idd: Array,
    quadrature: CutoffQuadrature,
    dose_profile: Callable[[Array, CutoffQuadrature], Array],
    *,
    cutoff: float,
    method: str,
    max_samples: int = 1_000_000,
) -> Array:
    """Integrate depth batches with bounded depth-by-sample intermediates.

    Parameters
    ----------
    depths, idd : Array
        One-dimensional depths and reference integrated depth doses.
    quadrature : CutoffQuadrature
        Radial/angular samples on the same namespace and device.
    dose_profile : callable
        Called with a depth batch and the quadrature; returns physical dose with
        shape ``(n_depths, n_samples)``. Engines retain ownership of their physics.
    cutoff : float
        Requested retained fraction, in (0, 1].
    method : {"integral", "relative"}
        Existing cutoff selection rule.
    max_samples : int
        Maximum depth-by-sample entries per batch, except that one complete depth
        is always processed. This bounds each 2D temporary, not total workspace.

    Returns
    -------
    Array
        Shape ``(n_depths, 3)``: radius, compensation factor and relative integral
        error in percent. One packed result permits a single host transfer.

    Notes
    -----
    Preserves the original first-true argmax rule (including index zero when no
    sample meets the condition). Relative compensation is returned per depth;
    callers preserving the existing scalar model use the final depth's value.
    """
    xp = array_api_compat.array_namespace(depths, idd, quadrature.radii)
    device = array_api_compat.device(depths)
    if depths.ndim != 1 or depths.shape != idd.shape:
        raise ValueError("Depths and IDD must be one-dimensional with matching shapes.")
    if not 0 < cutoff <= 1 or max_samples < 1:
        raise ValueError("Cutoff must be in (0, 1] and max_samples must be positive.")
    result = xp.zeros((depths.shape[0], 3), dtype=idd.dtype, device=device)
    if cutoff == 1:
        result[:, 0] = xp.inf
        result[:, 1] = 1
        return result
    if method not in ("integral", "relative"):
        raise ValueError("Invalid Cutoff Method. Must be 'integral' or 'relative'!")
    batch_size = max(1, max_samples // quadrature.radii.shape[0])
    for start in range(0, depths.shape[0], batch_size):
        stop = min(start + batch_size, depths.shape[0])
        dose = dose_profile(depths[start:stop], quadrature)
        if dose.shape != (stop - start, quadrature.radii.shape[0]):
            raise ValueError("Cutoff dose profile has an unexpected shape.")
        if bool(xp.any(xp.isnan(dose))) or bool(xp.any(dose < 0)):
            raise ValueError("Error in particle dose calculation.")
        area = xp.cumulative_sum(
            quadrature.radii * dose * quadrature.dr * quadrature.angular_weights, axis=1
        )
        if method == "integral":
            condition = area >= idd[start:stop, None] * cutoff
        else:
            condition = dose <= (1 - cutoff) * xp.max(dose, axis=1, keepdims=True)
        indices = xp.argmax(xp.astype(condition, xp.int32), axis=1)
        result[start:stop, 0] = xp.take(quadrature.radii, indices, axis=0)
        if method == "integral":
            result[start:stop, 1] = cutoff**-1
        else:
            selected = xp.take_along_axis(area, indices[:, None], axis=1)[:, 0]
            result[start:stop, 1] = (selected / area[:, -1]) ** -1
        result[start:stop, 2] = xp.abs(area[:, -1] / idd[start:stop] - 1) * 100
    return result
