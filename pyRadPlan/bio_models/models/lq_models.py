from typing import Any

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
