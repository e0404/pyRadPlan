from typing import Any, Union

from pyRadPlan.machines import Machine
from pyRadPlan.machines.base import get_machine


def validate_machine(data: Union[dict[str, Any], Machine, None] = None, **kwargs: Any) -> Machine:
    """Create and validate a ``Machine`` instance.

    Parameters
    ----------
    data
        A pre-existing ``Machine`` instance or a dictionary payload.
    **kwargs
        Fallback keyword arguments (used only when ``data`` is not a dict) and to
        supply ``radiation_mode`` when no dictionary is provided.
    """

    # Fast path: already a Machine
    if isinstance(data, Machine):
        return data

    radiation_mode = ""

    if isinstance(data, dict):
        meta = data.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("Dictionary Structure of provided machine not valid!")
        # Prefer snake_case if both present, fall back to camelCase.
        radiation_mode = meta.get("radiation_mode") or meta.get("radiationMode") or ""
    else:
        # If something non-dict (but not None) was passed, it's invalid
        if data is not None:
            raise ValueError("Dictionary Structure of provided machine not valid!")
        radiation_mode = kwargs.get("radiation_mode") or kwargs.get("radiationMode") or ""

    machine_cls = get_machine(radiation_mode)
    if machine_cls is None:
        raise ValueError(f"Could not resolve machine class for radiation_mode='{radiation_mode}'.")

    # Mirrors original behavior: pass through ``data`` (may be None) to model_validate
    return machine_cls.model_validate(data)
