"""Registry of robustness operators."""

from typing import Any, Optional, Type, Union
import logging

from ...scenarios import ScenarioSet
from ._base import RobustnessOperator, RobustnessOptions

logger = logging.getLogger(__name__)

_OPERATORS: dict[str, Type[RobustnessOperator]] = {}
_ALIASES: dict[str, str] = {}


def register_robustness_operator(
    operator_cls: Type[RobustnessOperator],
) -> Type[RobustnessOperator]:
    """
    Register a robustness operator class under its name and aliases.

    Usable as a decorator. Re-registering the same class is a no-op; a different class under
    an existing name raises.
    """
    if not (isinstance(operator_cls, type) and issubclass(operator_cls, RobustnessOperator)):
        raise TypeError("robustness operators must subclass RobustnessOperator")
    name = getattr(operator_cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("robustness operators must define a non-empty 'name'")
    existing = _OPERATORS.get(name)
    if existing is not None and existing is not operator_cls:
        raise ValueError(f"robustness operator {name!r} is already registered")
    for alias in (name, *operator_cls.aliases):
        owner = _ALIASES.get(alias)
        if owner is not None and owner != name:
            raise ValueError(f"robustness alias {alias!r} already refers to {owner!r}")
    _OPERATORS[name] = operator_cls
    for alias in (name, *operator_cls.aliases):
        _ALIASES[alias] = name
    return operator_cls


def available_robustness() -> list[str]:
    """Canonical names of the registered operators, ``"none"`` first."""
    names = list(_OPERATORS)
    if "none" in names:
        names.remove("none")
        names.insert(0, "none")
    return names


def resolve_robustness_name(name: str) -> str:
    """
    Canonical operator name for ``name`` or one of its aliases.

    Raises
    ------
    ValueError
        If no operator is registered under the name.
    """
    if not isinstance(name, str):
        raise ValueError(f"robustness must be a string, got {type(name).__name__}")
    canonical = _ALIASES.get(name)
    if canonical is None:
        key = name.strip().lower()
        canonical = next((v for k, v in _ALIASES.items() if k.lower() == key), None)
    if canonical is None:
        raise ValueError(f"Unknown robustness {name!r}. Available: {available_robustness()}")
    return canonical


def get_robustness_operator_class(name: str) -> Type[RobustnessOperator]:
    """Operator class registered under ``name`` or one of its aliases."""
    return _OPERATORS[resolve_robustness_name(name)]


def get_robustness_operator(
    robustness: Union[str, dict, RobustnessOperator],
    scenarios: Optional[ScenarioSet] = None,
    options: Optional[RobustnessOptions] = None,
) -> RobustnessOperator:
    """
    Create a robustness operator from a name, a configuration dictionary or an instance.

    Parameters
    ----------
    robustness : Union[str, dict, RobustnessOperator]
        The operator name (or alias), a dictionary ``{"name": ..., **parameters}`` whose
        parameters are passed to the operator's constructor, or an instance (returned as is).
    scenarios : ScenarioSet
        The scenario set the operator evaluates (required unless an instance is given).
    options : RobustnessOptions, optional
        Problem-wide operator options.
    """
    if isinstance(robustness, RobustnessOperator):
        return robustness
    params: dict[str, Any] = {}
    if isinstance(robustness, dict):
        params = dict(robustness)
        name = params.pop("name", None)
        if name is None:
            raise ValueError("robustness dictionaries need a 'name' entry")
    else:
        name = robustness
    if scenarios is None:
        raise ValueError("a scenario set is required to create a robustness operator")
    return get_robustness_operator_class(name)(scenarios, options, **params)
