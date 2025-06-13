"""Factory methods to manage available Sequencer."""

import warnings
import logging
from typing import Union, Type

from pyRadPlan.plan import Plan, validate_pln
from ._base import Sequencer

SEQUENCER = {}

logger = logging.getLogger(__name__)


def register_sequencer(sequencer_cls: Type[Sequencer]) -> None:
    """
    Register a new sequencer.

    Parameters
    ----------
    prob_cls : type
        A Sequencer class.
    """
    if not issubclass(sequencer_cls, Sequencer):
        raise ValueError("Sequencer must be a subclass of Sequencer.")

    if sequencer_cls.short_name is None:
        raise ValueError("Sequencer must have a 'short_name' attribute.")

    if sequencer_cls.name is None:
        raise ValueError("Sequencer must have a 'name' attribute.")

    prob_name = sequencer_cls.short_name
    if prob_name in SEQUENCER:
        warnings.warn(f"Sequencer '{prob_name}' is already registered.")
    else:
        SEQUENCER[prob_name] = sequencer_cls


def get_available_sequencers(pln: Union[Plan, dict[str]]) -> dict[str, Type[Sequencer]]:
    """
    Get a list of available Sequencer based on the plan.

    Parameters
    ----------
    pln : Union[Plan,dict]
        A Plan object.

    Returns
    -------
    list
        A list of available sequencer.
    """
    pln = validate_pln(pln)
    return {
        name: cls
        for name, cls in SEQUENCER.items()
        if pln.radiation_mode in cls.possible_radiation_modes
    }


def get_sequencer_from_pln(pln: Union[Plan, dict]) -> Sequencer:
    """
    Return an appropriate sequencer based on the plan.

    Parameters
    ----------
    pln : Plan
        A Plan object.

    Returns
    -------
    Sequencer
        A Sequencer object.
    """
    pln = validate_pln(pln)

    # Available sequencers based on the radiation mode
    sequencer = get_available_sequencers(pln)

    if len(sequencer) <= 0:
        raise ValueError(f"No sequencer available for radiation mode '{pln.radiation_mode}'.")

    sequencer_names = list(sequencer.keys())

    # Did the user provide a sequencer in the pln?
    if isinstance(pln.prop_seq, Sequencer):
        # The user provided a sequencer object, so lets use it
        # but warn the user if it is not in the available engines
        sequencer_name = pln.prop_seq.short_name
        if sequencer_name not in sequencer:
            warnings.warn(f"Engine '{sequencer_name}' seems not to be valid for Plan setup.")
        return pln.prop_seq

    if isinstance(pln.prop_seq, dict):
        # The user provided a dictionary with engine parameters, so we need to find the engine name
        if "sequencer" in pln.prop_seq:
            if pln.prop_seq["sequencer"] in sequencer:
                return sequencer[pln.prop_seq["sequencer"]](pln)
            warnings.warn(f"sequencer '{pln.prop_seq['sequencer']}' not available for Plan.")

        # If no engine name was found, we choose the first as default
        logger.warning(
            "No sequencer specified in Plan. Using first available sequencer %s.",
            sequencer_names[0],
        )
        return sequencer[sequencer_names[0]](pln)

    raise ValueError(f"No sequencer available for radiation mode '{pln.radiation_mode}'.")
