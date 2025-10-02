"""Factory method to manage available tradeoff exploration method implementations."""

import warnings
import logging
from typing import Union, Type
from ._base_tradeoff_strategies import TradeoffStrategyBase

TRADEOFFSTRATEGIES = {}

logger = logging.getLogger(__name__)


def register_tradeoff_strategy(tradeoff_cls: Type[TradeoffStrategyBase]) -> None:
    """
    Register a new tradeoff strategy.

    Parameters
    ----------
    tradeoff_cls : type
        A Tradeoff Strategy class.
    """
    if not issubclass(tradeoff_cls, TradeoffStrategyBase):
        raise ValueError("Tradeoff strategy must be a subclass of TradeoffStrategyBase.")

    if tradeoff_cls.short_name is None:
        raise ValueError("Tradeoff strategy must have a 'short_name' attribute.")

    if tradeoff_cls.name is None:
        raise ValueError("Tradeoff strategy must have a 'name' attribute.")

    tradeoff_name = tradeoff_cls.short_name
    if tradeoff_name in TRADEOFFSTRATEGIES:
        warnings.warn(f"Tradeoff strategy '{tradeoff_name}' is already registered.")
    else:
        TRADEOFFSTRATEGIES[tradeoff_name] = tradeoff_cls


def get_available_tradeoff_strategies() -> dict[str, Type[TradeoffStrategyBase]]:
    """
    Get a list of available tradeoff strategies based on the plan.

    Returns
    -------
    list
        A list of available tradeoff strategies.
    """
    return TRADEOFFSTRATEGIES


def get_tradeoff_strategy(
    tradeoff_desc: Union[str, dict],
    callbacks: dict[str, callable],
    scalarization_desc: Union[str, dict],
    scalarization_model_params,  # TODO: Define type,
    solver_desc: Union[str, dict],
) -> TradeoffStrategyBase:
    """
    Returns a tradeoff strategy based on a descriptive parameter.

    Parameters
    ----------
    tradeoff_desc : Union[str, dict]
        A string with the strategy name, or a dictionary with the strategy configuration
    callbacks : dict[str, callable]
        A dictionary with the functions in the planning problem that are required for the actual optimization
    scalarization_desc : Union[str, dict]
        A scalarization strategy instance
    solver_desc : Union[str, dict]
        A string with the solver name, or a dictionary with the solver configuration

    Returns
    -------
    TradeoffStrategyBase
        A solver instance
    """
    if isinstance(tradeoff_desc, str):
        tradeoff_strategy = TRADEOFFSTRATEGIES[tradeoff_desc](
            callbacks, scalarization_desc, scalarization_model_params, solver_desc
        )
    elif isinstance(tradeoff_desc, dict):
        raise NotImplementedError(
            "Tradeoff strategy configuration from dictionary not implemented yet."
        )
    else:
        raise ValueError(f"Invalid tradeoff strategy description: {tradeoff_desc}")

    return tradeoff_strategy
