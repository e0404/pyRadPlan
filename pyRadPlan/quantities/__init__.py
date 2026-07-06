"""Physical and biological quantities for treatment planning."""

from ._base import FluenceDependentQuantity, RTQuantity
from ._dose import Dose
from ._let_x_dose import LETxDose
from ._alpha_dose import AlphaDose
from ._sqrt_beta_dose import SqrtBetaDose
from ._effect import Effect
from ._rbe_x_dose import RBExDose
from ._let import DoseWeightedLET

QUANTITIES = {
    Dose.identifier: Dose,
    LETxDose.identifier: LETxDose,
    RBExDose.identifier: RBExDose,
    Effect.identifier: Effect,
    AlphaDose.identifier: AlphaDose,
    SqrtBetaDose.identifier: SqrtBetaDose,
    DoseWeightedLET.identifier: DoseWeightedLET,
}


def get_available_quantities() -> dict[str, RTQuantity]:
    """
    Obtain the available quantities in planning.

    Returns
    -------
    dict
        Dictionary with the available quantities
    """
    return QUANTITIES


def get_quantity(identifier: str) -> RTQuantity:
    """
    Obtain the quantity from name.

    Parameters
    ----------
    identifier : str
        The identifier of the quantity

    Returns
    -------
    RTQuantity
        The quantity
    """
    return QUANTITIES[identifier]


# Import after QUANTITIES is defined to avoid circular imports.
from ._resolver import QuantityResolver  # noqa: E402

__all__ = [
    "FluenceDependentQuantity",
    "RTQuantity",
    "QuantityResolver",
    "Dose",
    "LETxDose",
    "RBExDose",
    "Effect",
    "AlphaDose",
    "SqrtBetaDose",
    "DoseWeightedLET",
    "get_available_quantities",
    "get_quantity",
    "QUANTITIES",
]
