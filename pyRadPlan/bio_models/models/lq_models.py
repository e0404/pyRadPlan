from pyRadPlan.bio_models._base import BiologicalModelBase
import array_api_strict as xp


class LQModel(BiologicalModelBase):
    """
    Abstract base class for linear-quadratic (LQ) models, which are commonly used
    to describe the relationship between radiation dose and biological effect.

    """

    default_report_quantity = "RBExDose"  # Suggested quantity for display and planning

    def __init__(self, pln):
        self.default_alpha_x: float = 0.1
        self.default_beta_x: float = 0.05
        super().__init__(pln)

    def calc_biological_quantities_for_bixel(self, bixel: dict) -> dict:
        """
        Initialise alpha/beta arrays to NaN and compute the alpha/beta ratio.
        Subclasses call super() then fill in the actual values.
        """
        n = len(bixel["rad_depths"])
        v_ab_ratio = bixel["v_alpha_x"] / bixel["v_beta_x"]
        bixel = bixel.model_copy(
            update={
                "alpha": xp.full(n, xp.nan),
                "beta": xp.full(n, xp.nan),
                "v_ab_ratio": v_ab_ratio,
            }
        )
