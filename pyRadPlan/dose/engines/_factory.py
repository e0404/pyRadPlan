import warnings
import logging
from typing import Union, Type
from pyRadPlan.plan import validate_pln, Plan
from pyRadPlan.dose.engines import DoseEngineBase

DOSE_ENGINES = {}

logger = logging.getLogger(__name__)


def register_engine(engine_cls: Type[DoseEngineBase]) -> None:
    """
    Register a new engine.

    Parameters
    ----------
    engine_cls : type
        A Dose Engine class.
    """
    if not issubclass(engine_cls, DoseEngineBase):
        raise ValueError("Engine must be a subclass of DoseEngineBase.")

    if engine_cls.short_name is None:
        raise ValueError("Engine must have a 'short_name' attribute.")

    if engine_cls.name is None:
        raise ValueError("Engine must have a 'name' attribute.")

    if engine_cls.possible_radiation_modes is None:
        raise ValueError("Engine must have a 'possible_radiation_modes' attribute.")

    engine_name = engine_cls.short_name
    if engine_name in DOSE_ENGINES:
        warnings.warn(f"Engine '{engine_name}' is already registered.")
    else:
        DOSE_ENGINES[engine_name] = engine_cls


def get_available_engines(pln: Union[Plan, dict[str]]) -> dict[str, Type[DoseEngineBase]]:
    """
    Get a list of available engines based on the plan.

    Parameters
    ----------
    pln : Union[Plan,dict]
        A Plan object.

    Returns
    -------
    list
        A list of available engines.
    """
    pln = validate_pln(pln)
    return {
        name: cls
        for name, cls in DOSE_ENGINES.items()
        if pln.radiation_mode in cls.possible_radiation_modes
    }


def get_engine(pln: Union[Plan, dict]) -> DoseEngineBase:
    """
    Get the appropriate engine based on the plan.

    Parameters
    ----------
    pln : Plan
        A Plan object.

    Returns
    -------
    Engine
        A Dose Engine object.
    """
    pln = validate_pln(pln)

    # Available engines
    engines = get_available_engines(pln)
    if len(engines) <= 0:
        raise ValueError(f"No engine available for radiation mode '{pln.radiation_mode}'.")

    engine_names = list(engines.keys())

    # Did the user provide an engine in the pln?
    if isinstance(pln.prop_dose_calc, DoseEngineBase):
        # The user provided an engine object, so lets use it
        # but warn the user if it is not in the available engines
        engine_name = pln.prop_dose_calc.short_name
        if engine_name not in engines:
            warnings.warn(f"Engine '{engine_name}' seems not to be valid for Plan setup.")
        return pln.prop_dose_calc

    if isinstance(pln.prop_dose_calc, dict):
        # The user provided a dictionary with engine parameters, so we need to find the engine name
        if "engine" in pln.prop_dose_calc:
            if pln.prop_dose_calc["engine"] in engines:
                return engines[pln.prop_dose_calc["engine"]](pln)
            warnings.warn(f"Engine '{pln.prop_dose_calc['engine']}' not available for Plan.")

        # If no engine name was found, we choose the first as default
        logger.warning(
            "No engine specified in Plan. Using first available engine %s.", engine_names[0]
        )
        return engines[engine_names[0]](pln)

    raise ValueError(f"No engine available for radiation mode '{pln.radiation_mode}'.")
