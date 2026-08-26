from __future__ import annotations
import inspect
import logging
import warnings
from typing import Any, List, Optional, Union

from pydantic.alias_generators import to_snake

from pyRadPlan.bio_models._base import BiologicalModel

BioModelSpec = Union[str, dict[str, Any], BiologicalModel]

BIO_MODELS = {}

logger = logging.getLogger(__name__)

"""
Biological model registry for pyRadPlan.

This module maintains a global registry of available biological models and
provides utilities for registering, querying, and instantiating them.

The registry maps model names (and their aliases) to their corresponding
:class:`~pyRadPlan.bio_models._base.BiologicalModel` subclasses. Models
are registered via :func:`register_model` and can be retrieved by name using
:func:`get_bio_model`. Only models compatible with a given radiation mode and
the quantities provided by the dose engine are considered available.

Registry
--------
BIO_MODELS : dict[str, type[BiologicalModel]]
    Global mapping of model name / alias strings to model classes. Populated
    at import time as models are registered with :func:`register_model`.

Functions
---------
register_model
    Add a new biological model class to the registry.
get_available_models
    Filter the registry to models compatible with a given radiation mode and
    set of provided quantities.
get_bio_model
    Look up and instantiate a biological model by name, falling back to
    :class:`~pyRadPlan.bio_models._base.EmptyModel` when no match is found.
"""


def register_model(model_cls: BiologicalModel) -> None:
    """
    Register a new model.

    Parameters
    ----------
    model_cls : type
        A Biological Model class.
    """
    if not issubclass(model_cls, BiologicalModel):
        raise ValueError("Model must be a subclass of BiologicalModel.")

    if model_cls.model is None:
        raise ValueError("Model must have a 'model' attribute.")

    if model_cls.possible_radiation_modes is None:
        raise ValueError("Model must have a 'possible_radiation_modes' attribute.")

    for name in [model_cls.model, *model_cls.model_aliases]:
        if name in BIO_MODELS:
            warnings.warn(f"Model '{name}' is already registered.")
        else:
            BIO_MODELS[name] = model_cls


def get_available_models(
    radiation_mode: str,
    provided_quantities: List[str],
) -> dict[str, type[BiologicalModel]]:
    """
    Return all registered models given the radiaiton mode and provided quantities.
    """
    result = {}
    for cls in set(BIO_MODELS.values()):
        if radiation_mode in cls.possible_radiation_modes and all(
            q in provided_quantities for q in cls.required_quantities
        ):
            result[cls.model] = cls
            for alias in cls.model_aliases:
                result[alias] = cls
    return result


def available_bio_models(radiation_mode: str) -> list[type[BiologicalModel]]:
    """
    Registered model classes supporting a radiation mode, independent of machine data.

    Canonical names only (no aliases), ``"none"`` first, then alphabetically.
    """
    classes = {
        cls for cls in BIO_MODELS.values() if radiation_mode in cls.possible_radiation_modes
    }
    return sorted(classes, key=lambda cls: (cls.model != "none", cls.model))


def create_bio_model(spec: BioModelSpec, radiation_mode: Optional[str] = None) -> BiologicalModel:
    """
    Create a biological model from a name, a specification dict or an instance.

    Parameters
    ----------
    spec : str | dict | BiologicalModel
        Model name (``"constant_rbe"``, ``"WED"``, ...), a dict ``{"model": <name>,
        **constructor arguments}`` (camelCase keys are accepted), or an existing model.
    radiation_mode : str, optional
        If given, the model must support this modality.

    Raises
    ------
    ValueError
        Unknown model name, unknown / invalid constructor arguments, or a model that does
        not support ``radiation_mode``.
    """
    if isinstance(spec, BiologicalModel):
        model = spec
    else:
        if isinstance(spec, str):
            name, kwargs = spec, {}
        elif isinstance(spec, dict):
            # camelCase keys -> snake_case; leave names like 'p1' untouched
            kwargs = {
                (to_snake(k) if any(c.isupper() for c in k) else k): v for k, v in spec.items()
            }
            name = kwargs.pop("model", None) or kwargs.pop("name", None)
            if name is None:
                raise ValueError(
                    "Biological model specification dict needs a 'model' entry, "
                    f"got keys {sorted(spec)}"
                )
        else:
            raise ValueError(
                f"Cannot create a biological model from {type(spec).__name__}; "
                "expected a name, a dict or a BiologicalModel."
            )

        if name not in BIO_MODELS:
            raise ValueError(
                f"Unknown biological model '{name}'. Registered models: {sorted(BIO_MODELS)}"
            )
        cls = BIO_MODELS[name]
        try:
            model = cls(**kwargs)
        except TypeError as exc:
            accepted = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
            raise ValueError(
                f"Invalid parameters for biological model '{name}': {exc}. "
                f"Accepted parameters: {accepted}"
            ) from exc

    if radiation_mode is not None and radiation_mode not in model.possible_radiation_modes:
        raise ValueError(
            f"Biological model '{model.model}' does not support radiation mode "
            f"'{radiation_mode}' (supports {model.possible_radiation_modes})."
        )
    return model


def get_bio_model(
    spec: BioModelSpec, radiation_mode: str, provided_quantities: List[str]
) -> BiologicalModel:
    """
    Create a biological model and check it against a radiation mode and machine data.

    Raises
    ------
    ValueError
        If the model is unknown, its parameters are invalid, or it is not available for
        the given radiation mode / provided quantities.
    """
    model = create_bio_model(spec, radiation_mode)
    ok, msg = model.is_available(radiation_mode, provided_quantities)
    if not ok:
        raise ValueError(f"Biological model '{model.model}' not available: {msg}")
    return model


def bio_model_spec_from_matrad(spec: Any) -> BioModelSpec:
    """
    Normalise a matRad ``pln.bioModel`` entry into a pyRadPlan model specification.

    matRad stores either the model name or the model object as a struct whose fields include
    the model name (``model``) next to non-parameter metadata (``possibleRadiationModes``,
    ``requiredQuantities``, ...). Metadata fields are dropped; constructor parameters are
    kept (camelCase keys are converted to snake_case, e.g. ``RBE`` to ``rbe``).

    Parameters
    ----------
    spec : str, dict or BiologicalModel
        The matRad entry. Names may be matRad names (``"constRBE"``, ``"LEM"``).

    Returns
    -------
    str, dict or BiologicalModel
        A specification accepted by :func:`create_bio_model`.
    """
    if not isinstance(spec, dict):
        return spec
    spec = dict(spec)
    name = spec.pop("model", None) or spec.pop("name", None)
    if name is None:
        raise ValueError("A matRad bioModel struct needs a 'model' field.")
    if name not in BIO_MODELS:
        raise ValueError(
            f"Unknown biological model '{name}'. Registered models: {sorted(BIO_MODELS)}"
        )
    accepted = set(inspect.signature(BIO_MODELS[name].__init__).parameters) - {"self"}
    params, dropped = {}, []
    for key, value in spec.items():
        snake = to_snake(key) if any(c.isupper() for c in key) else key
        if snake in accepted:
            params[snake] = value
        else:
            dropped.append(key)
    if dropped:
        logger.info("Ignoring matRad bioModel fields %s for model '%s'.", dropped, name)
    return {"model": name, **params}
