"""
Scenario models for uncertainty quantification, robust optimization and robustness analysis.

The module separates three concerns:

- :class:`UncertaintyDimension` subclasses (:class:`SetupUncertainty`,
  :class:`RangeUncertainty`, :class:`CTPhaseUncertainty`) describe *what* is uncertain and with
  which parameters, together with a :class:`JointDistribution`.
- A :class:`ScenarioSet` is a concrete, immutable list of realizations with probabilities,
  weights and a flat index into Dij containers; it is what dose calculation and the quantities
  consume. It can be built from a model or from explicit points.
- A :class:`ScenarioModel` (``nomScen``, ``wcScen``, ...) is the pydantic configuration that
  generates a set lazily.
"""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from typing import Any, Optional, Type, Union

import numpy as np

from ._base import ScenarioModel
from ._dimensions import (
    AFFECTS_GEOMETRY,
    AFFECTS_RANGE,
    AFFECTS_VOXEL_PARAMS,
    CTPhaseUncertainty,
    RangeUncertainty,
    SetupUncertainty,
    UncertaintyDimension,
    available_dimensions,
    create_dimension,
    register_dimension,
)
from ._distributions import (
    IndependentGaussian,
    JointDistribution,
    MultivariateGaussian,
    available_distributions,
    create_distribution,
    register_distribution,
)
from ._explicit import ExplicitScenarioModel
from ._gridded import GriddedScenarioModel
from ._importance import ImportanceScenarioModel
from ._nominal import NominalScenarioModel
from ._random import RandomScenarioModel
from ._set import (
    MATRAD_AXES,
    MATRAD_DIMENSIONS,
    TAG_NOMINAL,
    Scenario,
    ScenarioBasis,
    ScenarioLayout,
    ScenarioSet,
)
from ._worst_case import WorstCaseScenarioModel

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, Type[ScenarioModel]] = {}

#: matRad model names known but not (yet) available; reported as not implemented.
_KNOWN_MATRAD_MODELS: dict[str, str] = {}


def __getattr__(name: str) -> Any:
    if name == "NominalScenario":
        warnings.warn(
            "NominalScenario is deprecated and will be removed after one release; use "
            "NominalScenarioModel.",
            DeprecationWarning,
            stacklevel=2,
        )
        return NominalScenarioModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_scenario_model(cls: Type[ScenarioModel]) -> Type[ScenarioModel]:
    """Register a scenario model class under its ``short_name`` (class decorator)."""
    _MODEL_REGISTRY[cls.short_name] = cls
    return cls


register_scenario_model(NominalScenarioModel)
register_scenario_model(ExplicitScenarioModel)
register_scenario_model(WorstCaseScenarioModel)
register_scenario_model(ImportanceScenarioModel)
register_scenario_model(RandomScenarioModel)


def available_scenario_models() -> list[str]:
    """Short names of the available scenario models (nominal first)."""
    names = [n for n in _MODEL_REGISTRY if n not in ("nomScen", "explicit")]
    return ["nomScen", *names]


def get_scenario_model_class(model_name: str) -> Type[ScenarioModel]:
    """Return the scenario model class registered under ``model_name``."""
    if model_name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_name]
    if model_name in _KNOWN_MATRAD_MODELS:
        raise NotImplementedError(
            f"{_KNOWN_MATRAD_MODELS[model_name]} ({model_name!r}) are not implemented yet"
        )
    raise ValueError(f"Unknown scenario model: {model_name}")


_get_model_class_by_name = get_scenario_model_class


def create_scenario_model(
    model_def: Union[ScenarioModel, ScenarioSet, str, dict, None] = "nomScen",
    ct: Optional[Any] = None,
) -> ScenarioModel:
    """
    Create a scenario model.

    Parameters
    ----------
    model_def : ScenarioModel or ScenarioSet or str or dict, optional
        A model (returned as is), a scenario set (wrapped in an
        :class:`ExplicitScenarioModel`), a model short name, or a dictionary with a ``"model"``
        key and model parameters (nested dimensions or the flat matRad names). Default: nominal.
    ct : CT or dict, optional
        CT to validate the CT phases against.

    Returns
    -------
    ScenarioModel
    """
    from pyRadPlan.ct import validate_ct  # noqa: PLC0415

    if isinstance(ct, dict):
        ct = validate_ct(ct)

    if model_def is None:
        model_def = "nomScen"

    if isinstance(model_def, ScenarioModel):
        model = model_def
    elif isinstance(model_def, ScenarioSet):
        model = ExplicitScenarioModel.from_set(model_def)
    elif isinstance(model_def, str):
        model = get_scenario_model_class(model_def)()
    elif isinstance(model_def, dict):
        params = dict(model_def)
        model_name = params.pop("model", None)
        if model_name is None:
            raise ValueError("Scenario model name not provided in the model definition")
        model = get_scenario_model_class(model_name).model_validate(params)
    else:
        raise ValueError("Invalid scenario model definition")

    if ct is not None:
        model.validate_ct(ct)
    return model


def validate_scenario_model(
    model_def: Union[ScenarioModel, ScenarioSet, str, dict], ct: Optional[Any] = None
) -> ScenarioModel:
    """
    Validate a scenario model definition, converting matRad's 1-based ``ctScenProb``.

    A model instance is returned as is (not copied), so that the stf generators and the dose
    engines working from the same plan share its generated set, including unseeded random
    samples.

    Parameters
    ----------
    model_def : ScenarioModel or ScenarioSet or str or dict
        See :func:`create_scenario_model`.
    ct : CT or dict, optional
        CT to validate against.

    Returns
    -------
    ScenarioModel
    """
    if isinstance(model_def, dict):
        model_def = deepcopy(model_def)
    if isinstance(model_def, dict) and "ctScenProb" in model_def:
        # Normalize to (n, 2), so squeezed 1-D pairs loaded from .mat files, 2-D
        # arrays and Python lists of tuples all become an (index, prob) matrix.
        arr = np.atleast_2d(np.asarray(model_def["ctScenProb"], dtype=float))
        pairs = [(int(round(a)), float(b)) for a, b in arr]
        if all(a > 0 for a, _ in pairs):
            pairs = [(a - 1, b) for a, b in pairs]
            logger.info(
                "Assuming scenario model from matRad. Converting 1-based indexing to 0-based "
                "indexing..."
            )
        else:
            logger.info(
                "CamelCase was used, but assuming 0-based indexing, since some values are less "
                "than 1. No index conversion applied"
            )
        model_def["ctScenProb"] = pairs

    return create_scenario_model(model_def, ct)


__all__ = [
    "AFFECTS_GEOMETRY",
    "AFFECTS_RANGE",
    "AFFECTS_VOXEL_PARAMS",
    "MATRAD_AXES",
    "MATRAD_DIMENSIONS",
    "TAG_NOMINAL",
    "CTPhaseUncertainty",
    "ExplicitScenarioModel",
    "GriddedScenarioModel",
    "ImportanceScenarioModel",
    "IndependentGaussian",
    "JointDistribution",
    "MultivariateGaussian",
    "NominalScenario",
    "NominalScenarioModel",
    "RandomScenarioModel",
    "RangeUncertainty",
    "Scenario",
    "ScenarioBasis",
    "ScenarioLayout",
    "ScenarioModel",
    "ScenarioSet",
    "SetupUncertainty",
    "UncertaintyDimension",
    "WorstCaseScenarioModel",
    "available_dimensions",
    "available_distributions",
    "available_scenario_models",
    "create_dimension",
    "create_distribution",
    "create_scenario_model",
    "get_scenario_model_class",
    "register_dimension",
    "register_distribution",
    "register_scenario_model",
    "validate_scenario_model",
]
