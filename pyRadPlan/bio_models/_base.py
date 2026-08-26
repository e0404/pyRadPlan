from __future__ import annotations
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, get_type_hints

from ._evaluator import BioModelEvaluator, ParametricEvaluator

logger = logging.getLogger(__name__)


class BiologicalModel(ABC):
    """
    Abstract base class for biological models.

    A biological model is a lightweight, patient-agnostic bundle of parameters and pure
    functions. Everything that depends on a particular machine or patient geometry (tissue
    class lookup, pre-computed kernel tables, ...) lives in the
    :class:`~pyRadPlan.bio_models.BioModelEvaluator` created by :meth:`evaluator` for one
    dose calculation.

    Attributes
    ----------
    model : str
        Canonical name identifying the biological model (e.g. ``"none"``, ``"LEM"``).
    model_aliases : list[str]
        Alternative names by which this model can be looked up.
    matrad_name : str or None
        Name of the equivalent matRad model (``pln.bioModel``); ``None`` if it is ``model``.
    required_quantities : list[str]
        Kernel quantities the machine data must provide (e.g. ``["physical_dose", "let"]``).
    possible_radiation_modes : list[str]
        Radiation modalities this model supports.
    default_report_quantity : str
        Quantity recommended for display and planning by default.
    provides_alpha_beta : bool
        Whether the model yields per-voxel LQ parameters; if so, the dose engine computes
        ``alpha_dose`` / ``sqrt_beta_dose`` influence matrices.
    requires_let : bool
        Whether the dose engine has to provide LET kernels to evaluate the model.
    """

    model: ClassVar[str]
    model_aliases: ClassVar[list[str]] = []
    matrad_name: ClassVar[Optional[str]] = None
    required_quantities: ClassVar[list[str]] = []
    possible_radiation_modes: ClassVar[list[str]]
    default_report_quantity: ClassVar[str] = "physical_dose"
    provides_alpha_beta: ClassVar[bool] = False
    requires_let: ClassVar[bool] = False

    _parameters: dict[str, Any]

    def __new__(cls, *args: Any, **kwargs: Any):
        # Remember the explicitly passed constructor arguments so that a model can be
        # serialised back into the {"model": ..., **parameters} form accepted by the factory.
        obj = super().__new__(cls)
        bound = inspect.signature(cls.__init__).bind(obj, *args, **kwargs)
        obj._parameters = {k: v for k, v in bound.arguments.items() if k != "self"}
        return obj

    @property
    def parameters(self) -> dict[str, Any]:
        """Constructor arguments this model was created with (explicitly passed ones)."""
        return dict(self._parameters)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable specification: ``{"model": <name>, **parameters}``."""
        return {"model": self.model, **self._parameters}

    def to_matrad(self) -> str:
        """Model name as used by matRad's ``pln.bioModel``."""
        return self.matrad_name or self.model

    @classmethod
    def config_model(cls) -> type:
        """
        Pydantic model of the constructor parameters (one field per parameter).

        Generated lazily per class from the ``__init__`` signature; used to validate and
        edit parameter dicts (e.g. in the GUI) analogous to
        :meth:`~pyRadPlan.core.ConfigurableAlgorithm.config_model`.
        """
        if "_config_model" in cls.__dict__:
            return cls.__dict__["_config_model"]

        from pydantic import create_model
        from pyRadPlan.core import AlgorithmConfig

        try:
            hints = get_type_hints(cls.__init__)
        except (NameError, TypeError):
            hints = {}
        fields = {}
        for name, param in inspect.signature(cls.__init__).parameters.items():
            if name == "self" or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[name] = (hints.get(name, Any), default)
        model = create_model(f"{cls.__name__}Config", __base__=AlgorithmConfig, **fields)
        cls._config_model = model
        return model

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BiologicalModel):
            return NotImplemented
        return type(self) is type(other) and _params_equal(self._parameters, other._parameters)

    def __hash__(self) -> int:
        return hash((type(self), repr(sorted(self._parameters.items()))))

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self._parameters.items())
        return f"{type(self).__name__}({params})"

    @abstractmethod
    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        """
        Create the evaluator of this model for one machine / patient geometry.

        Parameters
        ----------
        machine
            The (validated) machine the dose is calculated with.
        voxel_params : dict[str, Array]
            Per-voxel tissue parameters on the dose grid, each of shape
            ``(num_voxels, num_ct_scenarios)``. Currently ``"alpha_x"`` and ``"beta_x"``.
        """

    def dij_scalars(self) -> dict[str, Any]:
        """Scalar entries this model contributes to the dij (e.g. a constant RBE)."""
        return {}

    def is_available(
        self,
        radiation_mode: str,
        provided_quantities: Optional[list[str]] = None,
    ) -> tuple[bool, str]:
        """
        Check compatibility with a radiation mode and the quantities a machine provides.

        Returns
        -------
        (available, message)
            ``available`` is ``True`` only when both the radiation mode and all required
            quantities are satisfied.
        """
        messages: list[str] = []

        valid_rad_mode = radiation_mode in self.possible_radiation_modes
        if not valid_rad_mode:
            messages.append(
                f"Radiation mode '{radiation_mode}' is invalid for model '{self.model}'."
            )

        valid_quantities = True
        if provided_quantities is not None and self.required_quantities:
            missing = [q for q in self.required_quantities if q not in provided_quantities]
            if missing:
                valid_quantities = False
                messages.append(
                    "Required quantities not provided by Dose Engine or selected Machine Dataset."
                )

        return (valid_rad_mode and valid_quantities), ", ".join(messages)

    def validate_for(
        self,
        radiation_mode: str,
        provided_quantities: Optional[list[str]] = None,
    ) -> None:
        """Raise ValueError if the model is not available for the given mode/quantities."""
        ok, msg = self.is_available(radiation_mode, provided_quantities)
        if not ok:
            raise ValueError(f"Biological model '{self.model}' not valid: {msg}")


def _params_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.keys() != b.keys():
        return False
    for key in a:
        try:
            equal = bool(a[key] == b[key])
        except ValueError:  # array-valued parameter
            import numpy as np

            equal = np.array_equal(a[key], b[key])
        if not equal:
            return False
    return True


# Backwards-compatible name
BiologicalModelBase = BiologicalModel


class EmptyModel(BiologicalModel):
    """
    Passthrough biological model that applies no biological weighting.

    Compatible with all modalities, requires no kernel data beyond physical dose, and
    recommends reporting physical dose.
    """

    model = "none"
    required_quantities = []
    possible_radiation_modes = ["photons", "protons", "helium", "carbon", "oxygen", "VHEE"]
    default_report_quantity = "physical_dose"

    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        return ParametricEvaluator(self)
