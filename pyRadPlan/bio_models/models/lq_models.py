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
        """
        Initialise alpha/beta arrays to NaN and compute the alpha/beta ratio.
        Subclasses call super() then fill in the actual values.
        """
        xp = array_api_compat.array_namespace(bixel["rad_depths"])
        n = xp.unique_values(bixel["rad_depths"]).shape[0]
        bixel["alpha"] = xp.full(n, xp.nan)
        bixel["beta"] = xp.full(n, xp.nan)
        return bixel
