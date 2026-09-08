"""Tissue parameter lookups: how per-voxel (alpha_x, beta_x) select from per-class kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import array_api_compat
import numpy as np

from pyRadPlan.core.xp_utils import device_cache_key


class TissueParameterLookup(ABC):
    """
    Maps per-voxel reference LQ parameters onto rows of per-tissue-class kernel arrays.

    Kernel-based and tabulated models carry depth-dependent data for a finite set of
    reference ``(alpha_x, beta_x)`` pairs (tissue classes). A lookup decides how a voxel
    with arbitrary reference parameters uses that data — by exact class match today,
    possibly by nearest class or interpolation between classes in the future.

    The reference parameters are machine / table metadata and are kept as host arrays; they
    are fixed for the lifetime of the lookup.

    Parameters
    ----------
    class_alpha_x, class_beta_x : array-like, shape (n_classes,)
        Reference LQ parameters of the available tissue classes.

    Raises
    ------
    ValueError
        No tissue class is declared, or the two declarations have different lengths.
    """

    def __init__(self, class_alpha_x: Any, class_beta_x: Any):
        self.class_alpha_x = np.reshape(np.asarray(class_alpha_x, dtype=float), (-1,))
        self.class_beta_x = np.reshape(np.asarray(class_beta_x, dtype=float), (-1,))
        if self.class_alpha_x.size == 0:
            raise ValueError("A tissue lookup needs at least one reference tissue class.")
        if self.class_alpha_x.shape != self.class_beta_x.shape:
            raise ValueError(
                f"Tissue classes declare {self.class_alpha_x.size} reference alpha_x but "
                f"{self.class_beta_x.size} reference beta_x values."
            )

    @abstractmethod
    def validate(self, voxel_params: dict[str, Any]) -> None:
        """Raise ValueError if some voxel of the dose grid cannot be looked up."""

    @abstractmethod
    def gather(
        self,
        alpha_x: Any,
        beta_x: Any,
        context: Mapping[str, Any],
        fields: list[str],
    ) -> dict[str, Any]:
        """
        Select per-voxel kernel values for one bixel.

        Parameters
        ----------
        alpha_x, beta_x : Array, shape (n_voxels,)
            Reference photon LQ parameters of the voxels.
        context : mapping
            Full biological evaluation context. The entries named by ``fields`` are
            interpolated kernel arrays of shape ``(n_classes, n_voxels)``.
        fields : list[str]
            Names of the kernel arrays to gather.

        Returns
        -------
        dict[str, Array]
            One ``(n_voxels,)`` array per field.
        """


class ExactClassLookup(TissueParameterLookup):
    """
    Lookup requiring every voxel to match one tissue class exactly.

    Voxels with ``alpha_x == beta_x == 0`` (outside any structure) map to class 0.

    The reference class arrays are converted into the array namespace, device and dtype of
    the voxel parameters once per combination instead of on every bixel. Once
    :meth:`validate` has confirmed the whole dose grid, :meth:`gather` skips the per-bixel
    match check, since the bixel parameters are a subset of the validated grid and the check
    would force a device synchronization per bixel.
    """

    def __init__(self, class_alpha_x: Any, class_beta_x: Any):
        super().__init__(class_alpha_x, class_beta_x)
        self._reference_cache: dict[Any, tuple[Any, Any]] = {}
        self._grid_validated = False

    def _reference(self, like: Any) -> tuple[Any, Any, Any]:
        """Namespace and reference class arrays in ``like``'s namespace, device and dtype."""
        xp = array_api_compat.array_namespace(like)
        device = array_api_compat.device(like)
        key = (xp, device_cache_key(like), like.dtype)
        reference = self._reference_cache.get(key)
        if reference is None:
            reference = (
                xp.reshape(xp.asarray(self.class_alpha_x, dtype=like.dtype, device=device), (-1,)),
                xp.reshape(xp.asarray(self.class_beta_x, dtype=like.dtype, device=device), (-1,)),
            )
            self._reference_cache[key] = reference
        return (xp, *reference)

    def class_index(self, v_alpha_x: Any, v_beta_x: Any, validate: bool = True) -> Any:
        """
        Index of the matching tissue class per voxel.

        Parameters
        ----------
        v_alpha_x, v_beta_x : Array, shape (n_voxels,) or (n_voxels, n_ct_scen)
        validate : bool
            Check that every voxel matches a class. Only the per-bixel gather of an already
            validated dose grid may skip this.

        Raises
        ------
        ValueError
            If a voxel pair has no exact match in the reference classes.
        """
        xp, ref_alpha_x, ref_beta_x = self._reference(v_alpha_x)

        # (..., n_classes) boolean match against every class
        matches = (v_alpha_x[..., None] == ref_alpha_x) & (v_beta_x[..., None] == ref_beta_x)
        if validate:
            outside = (v_alpha_x == 0) & (v_beta_x == 0)
            unmatched = ~xp.any(matches, axis=-1) & ~outside
            if xp.any(unmatched):
                first = tuple(int(b[0]) for b in xp.nonzero(unmatched))
                a, b = v_alpha_x[first], v_beta_x[first]
                raise ValueError(
                    f"No matching tissue class for alpha_x={float(a)}, beta_x={float(b)}. "
                    f"Available classes: alpha_x={ref_alpha_x}, beta_x={ref_beta_x}"
                )
        return xp.argmax(xp.astype(matches, xp.int64), axis=-1)

    def validate(self, voxel_params: dict[str, Any]) -> None:
        missing = [name for name in ("alpha_x", "beta_x") if voxel_params.get(name) is None]
        if missing:
            raise ValueError(
                f"A tissue class lookup needs the per-voxel {sorted(missing)}, which the dose "
                "calculation did not provide."
            )
        self._grid_validated = False
        self.class_index(voxel_params["alpha_x"], voxel_params["beta_x"])
        self._grid_validated = True

    def gather(
        self,
        alpha_x: Any,
        beta_x: Any,
        context: Mapping[str, Any],
        fields: list[str],
    ) -> dict[str, Any]:
        xp = array_api_compat.array_namespace(alpha_x)
        class_ix = self.class_index(alpha_x, beta_x, validate=not self._grid_validated)
        voxel_ix = xp.arange(class_ix.shape[0], device=array_api_compat.device(class_ix))
        return {name: context[name][class_ix, voxel_ix] for name in fields}


TISSUE_LOOKUPS: dict[str, type[TissueParameterLookup]] = {"exact": ExactClassLookup}


def make_tissue_lookup(
    method: str, class_alpha_x: Any, class_beta_x: Any
) -> TissueParameterLookup:
    """Instantiate a lookup by name (``"exact"``)."""
    try:
        cls = TISSUE_LOOKUPS[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown tissue lookup '{method}'. Available: {sorted(TISSUE_LOOKUPS)}"
        ) from exc
    return cls(class_alpha_x, class_beta_x)
