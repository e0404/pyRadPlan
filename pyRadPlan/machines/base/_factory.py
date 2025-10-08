"""Factory methods to manage available Machines."""

from typing import Union, Optional, Type, Dict
import warnings
import logging


from ._base import Machine
from pyRadPlan.plan import Plan, validate_pln

logger = logging.getLogger(__name__)


MACHINES: Dict[str, Type[Machine]] = {}


def register_machine(machine_cls: Type[Machine]) -> None:
    """
    Register a new machine.

    Parameters
    ----------
    prob_cls : type
        A Machine class.
    """
    if not issubclass(machine_cls, Machine):
        raise ValueError("Machine must be a subclass of Machine.")

    if not hasattr(machine_cls, "_possible_radiation_modes"):
        raise ValueError("Machine must define the '_possible_radiation_modes' class variable.")

    radiation_modes = getattr(machine_cls, "_possible_radiation_modes", [])

    for mode in radiation_modes:
        if mode in MACHINES:
            warnings.warn(
                f"Machine '{mode}' is already registered. Make sure radiation_mode is unique in all machine classes."
            )
        MACHINES[mode] = machine_cls


def get_machine(radiation_mode: Optional[str] = None) -> Type[Machine]:
    """Retrieve a registered Machine class by name (optionally verifying compatibility)."""
    for k, cls in MACHINES.items():
        if k == radiation_mode:
            return cls


def get_machine_from_pln(pln: Union[Plan, dict]) -> Machine:
    """Obtain a Machine instance from a Plan (or plan dict).

    The Plan's ``machine`` field can be one of the accepted ``machine_desc`` types.
    """
    _pln = validate_pln(pln)
    return get_machine(_pln.radiation_mode)
