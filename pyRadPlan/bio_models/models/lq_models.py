from ._base import BiologicalModelBase


class LQModel(BiologicalModelBase):
    """
    Abstract base class for linear-quadratic (LQ) models, which are commonly used
    to describe the relationship between radiation dose and biological effect.

    """

    default_report_quantity = "RBExDose"  # Suggested quantity for display and planning
