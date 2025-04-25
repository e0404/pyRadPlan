from ._base import FluenceDependentQuantity, RTQuantity
from ._dose import Dose
from ._let_x_dose import LETxDose

QUANTITIES = {Dose.identifier: Dose, LETxDose.identifier: LETxDose}


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


__all__ = [
    "FluenceDependentQuantity",
    "Dose",
    "RTQuantity",
    "get_available_quantities",
    "get_quantity",
]
