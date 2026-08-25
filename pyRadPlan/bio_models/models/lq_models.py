from typing import Any

import array_api_compat

from pyRadPlan.bio_models._base import BiologicalModelBase


class LQModel(BiologicalModelBase):
    """
    Abstract base class for linear-quadratic (LQ) biological models.

    The linear-quadratic model describes cell survival after irradiation as:

    .. math::

        S = exp(-(alpha d + beta d^2)

    where d is the dose per fraction, and alpha and
    beta are tissue-specific radiosensitivity parameters.
    Subclasses are responsible for providing the actual per-voxel
    alpha and beta values via calc_biological_quantities_for_bixel`.

    """

    default_report_quantity = "rbe_x_dose"  # Suggested quantity for display and planning

    def calc_biological_quantities_for_bixel(self, bixel: dict, kernels: dict) -> dict:
        """Subclasses fill in per-voxel ``alpha`` and ``beta`` here."""
        return bixel

    @staticmethod
    def match_tissue_classes(
        v_alpha_x: Any, v_beta_x: Any, ref_alpha_x: Any, ref_beta_x: Any
    ) -> Any:
        """
        Map per-voxel reference (alpha_x, beta_x) pairs to tissue-class indices.

        Parameters
        ----------
        v_alpha_x, v_beta_x : Array, shape (n_voxels, n_ct_scen)
            Reference photon LQ parameters per voxel and CT scenario.
        ref_alpha_x, ref_beta_x : array-like, shape (n_classes,)
            Reference pairs of the tissue classes available in the base data / table.

        Returns
        -------
        Array, shape (n_voxels, n_ct_scen)
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

        # (n_voxels, n_ct_scen, n_classes) boolean match against every class
        matches = (v_alpha_x[..., None] == ref_alpha_x) & (v_beta_x[..., None] == ref_beta_x)
        outside = (v_alpha_x == 0) & (v_beta_x == 0)
        unmatched = ~xp.any(matches, axis=-1) & ~outside
        if xp.any(unmatched):
            bad = xp.nonzero(unmatched)
            a, b = v_alpha_x[bad[0][0], bad[1][0]], v_beta_x[bad[0][0], bad[1][0]]
            raise ValueError(
                f"No matching tissue class for alpha_x={float(a)}, beta_x={float(b)}. "
                f"Available classes: alpha_x={ref_alpha_x}, beta_x={ref_beta_x}"
            )
        return xp.argmax(xp.astype(matches, xp.int64), axis=-1)
