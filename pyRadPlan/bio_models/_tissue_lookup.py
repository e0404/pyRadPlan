"""Tissue parameter lookups: how per-voxel (alpha_x, beta_x) select from per-class kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import array_api_compat


class TissueParameterLookup(ABC):
    """
    Maps per-voxel reference LQ parameters onto rows of per-tissue-class kernel arrays.

    Kernel-based and tabulated models carry depth-dependent data for a finite set of
    reference ``(alpha_x, beta_x)`` pairs (tissue classes). A lookup decides how a voxel
    with arbitrary reference parameters uses that data — by exact class match today,
    possibly by nearest class or interpolation between classes in the future.

    Parameters
    ----------
    class_alpha_x, class_beta_x : array-like, shape (n_classes,)
        Reference LQ parameters of the available tissue classes.
    """

    def __init__(self, class_alpha_x: Any, class_beta_x: Any):
        self.class_alpha_x = class_alpha_x
        self.class_beta_x = class_beta_x

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
    """

    def class_index(self, v_alpha_x: Any, v_beta_x: Any) -> Any:
        """
        Index of the matching tissue class per voxel.

        Parameters
        ----------
        v_alpha_x, v_beta_x : Array, shape (n_voxels,) or (n_voxels, n_ct_scen)

        Raises
        ------
        ValueError
            If a voxel pair has no exact match in the reference classes.
        """
        xp = array_api_compat.array_namespace(v_alpha_x)
        device = array_api_compat.device(v_alpha_x)
        ref_alpha_x = xp.reshape(
            xp.asarray(self.class_alpha_x, dtype=v_alpha_x.dtype, device=device), (-1,)
        )
        ref_beta_x = xp.reshape(
            xp.asarray(self.class_beta_x, dtype=v_beta_x.dtype, device=device), (-1,)
        )

        # (..., n_classes) boolean match against every class
        matches = (v_alpha_x[..., None] == ref_alpha_x) & (v_beta_x[..., None] == ref_beta_x)
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
        self.class_index(voxel_params["alpha_x"], voxel_params["beta_x"])

    def gather(
        self,
        alpha_x: Any,
        beta_x: Any,
        context: Mapping[str, Any],
        fields: list[str],
    ) -> dict[str, Any]:
        xp = array_api_compat.array_namespace(alpha_x)
        class_ix = self.class_index(alpha_x, beta_x)
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
