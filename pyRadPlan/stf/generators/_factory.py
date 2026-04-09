import warnings
import logging
from typing import Union, Type
from pyRadPlan.plan import validate_pln, Plan

from ._base import StfGeneratorBase


STF_GENERATORS = {}

logger = logging.getLogger(__name__)


def register_generator(generator_cls: Type[StfGeneratorBase]) -> None:
    """
    Register a new stf generator for irradiation geometry.

    Parameters
    ----------
    generator_cls : type
        A Generator class.
    """
    if not issubclass(generator_cls, StfGeneratorBase):
        raise ValueError("Generator must be a subclass of StfGeneratorBase.")

    if generator_cls.short_name is None:
        raise ValueError("Generator must have a 'short_name' attribute.")

    if generator_cls.name is None:
        raise ValueError("Generator must have a 'name' attribute.")

    if generator_cls.possible_radiation_modes is None:
        raise ValueError("Generator must have a 'possible_radiation_modes' attribute.")

    generator_name = generator_cls.short_name
    if generator_name in STF_GENERATORS:
        warnings.warn(f"Generator '{generator_name}' is already registered.")
    else:
        STF_GENERATORS[generator_name] = generator_cls


def get_available_generators(pln: Union[Plan, dict[str]]) -> dict[str, Type[StfGeneratorBase]]:
    """
    Get a list of available stf generators based on the plan.

    Parameters
    ----------
    pln : Union[Plan,dict]
        A Plan object.

    Returns
    -------
    list
        A list of available generators.
    """
    pln = validate_pln(pln)
    return {
        name: cls
        for name, cls in STF_GENERATORS.items()
        if pln.radiation_mode in cls.possible_radiation_modes
    }


def get_generator(pln: Union[Plan, dict]) -> StfGeneratorBase:
    """
    Get the appropriate generator based on the plan.

    Parameters
    ----------
    pln : Plan
        A Plan object.

    Returns
    -------
    Generator
        A Dose Generator object.
    """
    pln = validate_pln(pln)

    # Available generators
    generators = get_available_generators(pln)
    if len(generators) <= 0:
        raise ValueError(f"No generator available for radiation mode '{pln.radiation_mode}'.")

    generator_names = list(generators.keys())

    # Did the user provide an generator in the pln?
    if isinstance(pln.prop_stf, StfGeneratorBase):
        # The user provided an generator object, so lets use it
        # but warn the user if it is not in the available generators
        generator_name = pln.prop_stf.short_name
        if generator_name not in generators:
            warnings.warn(f"Generator '{generator_name}' seems not to be valid for Plan setup.")
        return pln.prop_stf

    if isinstance(pln.prop_stf, dict):
        # The user provided a dictionary with generator parameters, so we need to find the generator name
        if "generator" in pln.prop_stf:
            if pln.prop_stf["generator"] in generators:
                return generators[pln.prop_stf["generator"]](pln)
            warnings.warn(f"Generator '{pln.prop_stf['generator']}' not available for Plan.")

        # If no generator name was found, we choose the first as default
        # TODO: Check for if field shape was provided and choose appropriate generator
        logger.warning(
            "No generator specified in Plan. Using first available generator %s.",
            generator_names[0],
        )
        return generators[generator_names[0]](pln)

    raise ValueError(f"No generator available for radiation mode '{pln.radiation_mode}'.")
