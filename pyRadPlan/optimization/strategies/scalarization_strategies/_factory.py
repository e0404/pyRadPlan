"""Factory methods to manage available scalarization strategy implementations."""

import warnings
import logging
from typing import Union, Type
from ._base_scalarization_strategies import ScalarizationStrategyBase

SCALARIZATIONSTRATEGIES = {}

logger = logging.getLogger(__name__)


def register_scalarization_strategy(scalarization_cls: Type[ScalarizationStrategyBase]) -> None:
    """
    Register a new scalarization strategy.

    Parameters
    ----------
    scalarization_cls : type
        A Scalarization Strategy class.
    """
    if not issubclass(scalarization_cls, ScalarizationStrategyBase):
        raise ValueError("Scalarization strategy must be a subclass of ScalarizationStrategyBase.")

    if scalarization_cls.short_name is None:
        raise ValueError("Scalarization strategy must have a 'short_name' attribute.")

    if scalarization_cls.name is None:
        raise ValueError("Scalarization strategy must have a 'name' attribute.")

    scalarization_name = scalarization_cls.short_name
    if scalarization_name in SCALARIZATIONSTRATEGIES:
        warnings.warn(f"Scalarization strategy '{scalarization_name}' is already registered.")
    else:
        SCALARIZATIONSTRATEGIES[scalarization_name] = scalarization_cls



def get_available_scalarization_strategies() -> dict[str]:
    """
    Get a list of available scalarization strategies based on the plan.

    Returns
    -------
    list
        A list of available scalarization strategies.
    """
    return SCALARIZATIONSTRATEGIES


def get_scalarization_strategy(scalarization_desc: Union[str, dict],eval_callbacks) -> ScalarizationStrategyBase:
    """
    Returns a scalarization strategy instance based on a descriptive parameter.

    Parameters
    ----------
    scalarization_desc : Union[str, dict]
        A string with the strategy name, or a dictionary with the strategy configuration

    Returns
    -------
    ScalarizationStrategyBase
        A strategy instance
    """
    if isinstance(scalarization_desc, str):
        strategy = SCALARIZATIONSTRATEGIES[scalarization_desc](
            eval_callbacks
        )
    
    elif isinstance(scalarization_desc, dict):
        raise NotImplementedError("Scalarization strategy configuration from dictionary not implemented yet.")
    else:
        raise ValueError(f"Invalid scalarization strategy description: {scalarization_desc}")

    return strategy
