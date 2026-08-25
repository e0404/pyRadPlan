from typing import Any

import array_api_compat

from pyRadPlan.bio_models._base import BiologicalModel


class LQModel(BiologicalModel):
    """
    Abstract base class for linear-quadratic (LQ) biological models.

    The linear-quadratic model describes cell survival after irradiation as

    .. math::

        S = \\exp(-(\\alpha d + \\beta d^2))

    where d is the dose per fraction and alpha, beta are tissue-specific radiosensitivity
    parameters. Subclasses provide the per-voxel alpha and beta values, either as a pure
    function of the voxel parameters (:meth:`alpha_beta`) or by converting pre-computed
    tissue-class kernel rows (:meth:`alpha_beta_from_kernel_rows`).
    """

    provides_alpha_beta = True
    default_report_quantity = "rbe_x_dose"

    def alpha_beta(self, alpha_x: Any, beta_x: Any, kernels: dict[str, Any]) -> tuple[Any, Any]:
        """
        Per-voxel (alpha, beta) as a pure function of reference parameters and kernels.

        Parameters
        ----------
        alpha_x, beta_x : Array, shape (n_voxels,)
            Reference photon LQ parameters of the voxels.
        kernels : dict
            Kernel values interpolated at the voxels (e.g. ``kernels["let"]``).
        """
        raise NotImplementedError(f"Model '{self.model}' is not a parametric LQ model.")

    def alpha_beta_from_kernel_rows(self, rows: dict[str, Any]) -> tuple[Any, Any]:
        """
        Convert per-voxel gathered tissue-class kernel values into (alpha, beta).

        Parameters
        ----------
        rows : dict[str, Array]
            One ``(n_voxels,)`` array per kernel field the model requested.
        """
        raise NotImplementedError(f"Model '{self.model}' has no tissue-class kernels.")

    @staticmethod
    def match_tissue_classes(
        v_alpha_x: Any, v_beta_x: Any, ref_alpha_x: Any, ref_beta_x: Any
    ) -> Any:
        """
        Map per-voxel reference (alpha_x, beta_x) pairs to tissue-class indices.

        Parameters
        ----------
        v_alpha_x, v_beta_x : Array, shape (n_voxels,) or (n_voxels, n_ct_scen)
            Reference photon LQ parameters per voxel (and CT scenario).
        ref_alpha_x, ref_beta_x : array-like, shape (n_classes,)
            Reference pairs of the tissue classes available in the base data / table.

        Returns
        -------
        Array, same shape as ``v_alpha_x``
            Index of the matching tissue class per voxel. Voxels with
            ``alpha_x == beta_x == 0`` (outside any structure) get index 0.

        Raises
        ------
        ValueError
            If a voxel pair has no exact match in the reference classes.
        """
        xp = array_api_compat.array_namespace(v_alpha_x)
        ref_alpha_x = xp.reshape(xp.asarray(ref_alpha_x, dtype=v_alpha_x.dtype), (-1,))
        ref_beta_x = xp.reshape(xp.asarray(ref_beta_x, dtype=v_beta_x.dtype), (-1,))

        # (..., n_classes) boolean match against every class
        matches = (v_alpha_x[..., None] == ref_alpha_x) & (v_beta_x[..., None] == ref_beta_x)
        outside = (v_alpha_x == 0) & (v_beta_x == 0)
        unmatched = ~xp.any(matches, axis=-1) & ~outside
        if xp.any(unmatched):
            bad = xp.nonzero(unmatched)
            first = tuple(int(b[0]) for b in bad)
            a, b = v_alpha_x[first], v_beta_x[first]
            raise ValueError(
                f"No matching tissue class for alpha_x={float(a)}, beta_x={float(b)}. "
                f"Available classes: alpha_x={ref_alpha_x}, beta_x={ref_beta_x}"
            )
        return xp.argmax(xp.astype(matches, xp.int64), axis=-1)
