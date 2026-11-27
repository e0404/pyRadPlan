"""Array API quadrature for particle lateral-cutoff preparation."""

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Callable

import array_api_compat
import numpy as np

from pyRadPlan.core.xp_utils import to_namespace
from pyRadPlan.core.xp_utils.typing import Array


@dataclass(frozen=True)
class CutoffQuadrature:
    """Sampling definition and derived arrays for batched radial integration.

    All arrays belong to the same namespace/device. Treat their contents as
    read-only; create a new quadrature to change sampling consistently. The frozen
    dataclass prevents rebinding fields, but does not make array contents immutable.
    """

    radii: Array
    dr: Array
    angular_weights: Array
    radial_dist_sq: Array
    lat_dists: Array
    radial_edges: Array
    n_angles: int
    lateral_model: str

    @classmethod
    def create(cls, lateral_model, *, xp=None, device=None, radial_edges=None, n_angles=360):
        """Construct samples directly on the requested backend and device.

        Parameters
        ----------
        lateral_model : {"single", "double", "multi", "singleXY"}
            Only ``singleXY`` needs explicit angular sampling.
        xp : array namespace, optional
            Defaults to the namespace of array-valued ``radial_edges``, otherwise
            NumPy. Pass an Array API-compatible namespace.
        device : device, optional
            Target backend device. Defaults to the input edges' device when
            inferring their namespace, otherwise the backend default.
        radial_edges : array or sequence, optional
            Finite, strictly increasing bin boundaries in mm, starting at zero.
            Defaults to zero followed by 1,200 logarithmic points from 0.1 to
            1,000 mm. Custom edges are copied and stored as float64.
        n_angles : int, optional
            Number of uniformly spaced angles for ``singleXY``; default 360.
            Isotropic models integrate angles analytically and store ``1``.

        Notes
        -----
        Flattened angular samples are radius-major. Backend arithmetic may differ
        slightly in rounding, while sampling definitions and ordering are shared.
        """
        if lateral_model not in ("single", "double", "multi", "singleXY"):
            raise ValueError("Invalid lateral model for cutoff quadrature.")
        if isinstance(n_angles, bool) or not isinstance(n_angles, Integral) or n_angles < 1:
            raise ValueError("n_angles must be a positive integer.")
        if xp is None:
            if radial_edges is not None and array_api_compat.is_array_api_obj(radial_edges):
                xp = array_api_compat.array_namespace(radial_edges)
                if device is None:
                    device = array_api_compat.device(radial_edges)
            else:
                xp = array_api_compat.array_namespace(np.empty(0))
        if radial_edges is None:
            edges = xp.concat(
                (
                    xp.zeros(1, dtype=xp.float64, device=device),
                    10 ** xp.linspace(-1, 3, 1200, dtype=xp.float64, device=device),
                )
            )
        else:
            if (
                array_api_compat.is_array_api_obj(radial_edges)
                and array_api_compat.array_namespace(radial_edges) != xp
            ):
                radial_edges = to_namespace(xp, radial_edges, device=device)
            edges = xp.asarray(radial_edges, dtype=xp.float64, device=device, copy=True)
            if edges.ndim != 1 or edges.shape[0] < 2:
                raise ValueError("radial_edges must be a one-dimensional array of bin boundaries.")
            if (
                not bool(xp.all(xp.isfinite(edges)))
                or float(edges[0]) != 0
                or not bool(xp.all(edges[1:] > edges[:-1]))
            ):
                raise ValueError(
                    "radial_edges must be finite, strictly increasing, and start at zero."
                )
            device = array_api_compat.device(edges)
        radii = 0.5 * (edges[:-1] + edges[1:])
        dr = edges[1:] - edges[:-1]
        if lateral_model == "singleXY":
            angles = xp.linspace(0, 2 * xp.pi, n_angles + 1, dtype=xp.float64, device=device)[:-1]
            shape = (radii.shape[0], n_angles)
            radii = xp.reshape(xp.broadcast_to(radii[:, None], shape), (-1,))
            dr = xp.reshape(xp.broadcast_to(dr[:, None], shape), (-1,))
            theta = xp.reshape(xp.broadcast_to(angles[None, :], shape), (-1,))
            angular_weights = xp.full(
                radii.shape, 2 * xp.pi / n_angles, dtype=xp.float64, device=device
            )
            radial_dist_sq = radii**2
            lat_dists = xp.sqrt(radial_dist_sq)[:, None] * xp.stack(
                (xp.cos(theta), xp.sin(theta)), axis=1
            )
        else:
            n_angles = 1
            angular_weights = xp.asarray(2 * xp.pi, dtype=xp.float64, device=device)
            radial_dist_sq = radii**2
            lat_dists = xp.broadcast_to(xp.sqrt(radial_dist_sq / 2)[:, None], (radii.shape[0], 2))
        return cls(
            radii,
            dr,
            angular_weights,
            radial_dist_sq,
            lat_dists,
            edges,
            int(n_angles),
            lateral_model,
        )

    def to_namespace(self, xp, device=None):
        """Transfer preparation once, outside the depth-batch loop."""

        def move(value):
            if array_api_compat.array_namespace(value) == xp:
                return xp.asarray(value, device=device)
            return to_namespace(xp, value, device=device)

        return replace(
            self,
            **{
                name: move(getattr(self, name))
                for name in (
                    "radii",
                    "dr",
                    "angular_weights",
                    "radial_dist_sq",
                    "lat_dists",
                    "radial_edges",
                )
            },
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
