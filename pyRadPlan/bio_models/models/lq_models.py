from pyRadPlan.bio_models._base import BiologicalModelBase
import array_api_strict as xp


class LQModel(BiologicalModelBase):
    """
    Abstract base class for linear-quadratic (LQ) models, which are commonly used
    to describe the relationship between radiation dose and biological effect.

    """

    default_report_quantity = "RBExDose"  # Suggested quantity for display and planning

    def __init__(self):
        self.default_alpha_x: float = 0.1
        self.default_beta_x: float = 0.05
        super().__init__()

    def calc_biological_quantities_for_bixel(self, bixel: dict, kernels: dict) -> dict:
        """
        Initialise alpha/beta arrays to NaN and compute the alpha/beta ratio.
        Subclasses call super() then fill in the actual values.
        """
        n = len(bixel["rad_depths"])
        bixel["alpha"] = xp.full(n, xp.nan)
        bixel["beta"] = xp.full(n, xp.nan)
        return bixel
