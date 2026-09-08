from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Optional, get_type_hints

import array_api_compat
import numpy as np
from pydantic import create_model

from pyRadPlan.core import AlgorithmConfig

from ._evaluator import BioModelEvaluator, ParametricEvaluator, _validate_name_tuple

logger = logging.getLogger(__name__)

#: Per-voxel reference photon LQ parameters a dose engine supplies to a biological model.
REFERENCE_PARAMETER_NAMES = ("alpha_x", "beta_x")


class BiologicalModel(ABC):
    """
    Abstract base class for biological models.

    A biological model is a lightweight, patient-agnostic bundle of parameters and pure
    functions. Everything that depends on a particular machine or patient geometry (tissue
    class lookup, pre-computed kernel tables, ...) lives in the
    :class:`~pyRadPlan.bio_models.BioModelEvaluator` created by :meth:`evaluator` for one
    dose calculation.

    Custom subclasses must provide immutable tuple declarations for aliases, required machine
    quantities, evaluator outputs and supported radiation modes. Register a concrete subclass
    with :func:`~pyRadPlan.bio_models.register_model`; constructor arguments are retained by
    :meth:`__new__` for :attr:`parameters` and :meth:`to_dict`, so constructor parameters should
    be serialisable.

    Attributes
    ----------
    model : str
        Canonical name identifying the biological model (e.g. ``"none"``, ``"LEM"``).
    model_aliases : tuple[str, ...]
        Alternative names by which this model can be looked up.
    matrad_name : str or None
        Name of the equivalent matRad model (``pln.bioModel``); ``None`` if it is ``model``.
    required_quantities : tuple[str, ...]
        Named dose-engine or machine quantities required to construct and evaluate the model
        (e.g. ``("physical_dose", "let")``).
    possible_radiation_modes : tuple[str, ...]
        Radiation modalities this model supports.
    default_report_quantity : str
        Quantity recommended for display and planning by default.
    output_quantities : tuple[str, ...]
        Named quantities returned by the model evaluator, such as ``("alpha", "beta")``.
    requires_positive_reference : tuple[str, ...]
        Names of the per-voxel reference photon LQ parameters (``"alpha_x"``, ``"beta_x"``)
        the model formulation is only defined for when they are strictly positive. Dose
        engines validate them over the whole dose grid before a calculation starts.
    """

    model: ClassVar[str]
    model_aliases: ClassVar[tuple[str, ...]] = ()
    matrad_name: ClassVar[Optional[str]] = None
    required_quantities: ClassVar[tuple[str, ...]] = ()
    possible_radiation_modes: ClassVar[tuple[str, ...]]
    default_report_quantity: ClassVar[str] = "physical_dose"
    output_quantities: ClassVar[tuple[str, ...]] = ()
    requires_positive_reference: ClassVar[tuple[str, ...]] = ()

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
        """Return the serialisable specification: ``{"model": <name>, **parameters}``."""
        return {"model": self.model, **self._parameters}

    def to_matrad(self) -> str:
        """Model name as used by matRad's ``pln.bioModel``."""
        return self.matrad_name or self.model

    def requires(self, quantity_name: str) -> bool:
        """Whether the model requires the named engine or machine quantity."""
        return quantity_name in self.required_quantities

    def provides(self, quantity_name: str, *additional_names: str) -> bool:
        """Whether the evaluator provides every named quantity."""
        return all(name in self.output_quantities for name in (quantity_name, *additional_names))

    @classmethod
    def validate_declarations(cls) -> None:
        """Validate the public name and capability declarations of a model class."""
        canonical_name = getattr(cls, "model", None)
        if not isinstance(canonical_name, str) or not canonical_name:
            raise ValueError("Biological model must declare a non-empty string model name.")

        _validate_name_tuple(cls.model_aliases, "model_aliases", subject="Biological model")
        _validate_name_tuple(
            getattr(cls, "possible_radiation_modes", None),
            "possible_radiation_modes",
            subject="Biological model",
            allow_empty=False,
        )
        _validate_name_tuple(
            cls.required_quantities, "required_quantities", subject="Biological model"
        )
        _validate_name_tuple(
            cls.output_quantities, "output_quantities", subject="Biological model"
        )
        if not isinstance(cls.default_report_quantity, str) or not cls.default_report_quantity:
            raise ValueError(
                "Biological model default_report_quantity must be a non-empty string."
            )
        _validate_name_tuple(
            cls.requires_positive_reference,
            "requires_positive_reference",
            subject="Biological model",
        )

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

        Return a fresh :class:`BioModelEvaluator` bound to ``self``. Dose engines validate the
        evaluator's type, model binding and name declarations once during setup.

        Parameters
        ----------
        machine
            The (validated) machine the dose is calculated with.
        voxel_params : dict[str, Array]
            Per-voxel tissue parameters on the dose grid, each of shape
            ``(num_voxels, num_ct_scenarios)``. Currently ``"alpha_x"`` and ``"beta_x"``.
        """

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
                    "Required quantities not provided by Dose Engine or selected Machine Dataset: "
                    f"{', '.join(missing)}."
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

    def validate_reference_parameters(self, voxel_params: Mapping[str, Any]) -> None:
        """
        Check the per-voxel reference photon LQ parameters against the model's domain.

        Two levels are checked. First, every supplied reference coefficient must be a valid
        LQ rate: finite and non-negative, regardless of the model. Second, the parameters
        listed in :attr:`requires_positive_reference` must be strictly positive wherever
        reference parameters exist at all, so that a model formulated in terms of
        ``alpha_x / beta_x`` never has to invent a value for a degenerate ratio. Voxels
        outside every structure carry no reference parameters
        (``alpha_x == beta_x == 0``); models evaluate to zero there and they are skipped by
        the second check.

        Together the two levels guarantee that a model receiving these parameters cannot be
        driven into a non-finite result by the reference data itself. Dose engines call this
        during setup, before any dose is computed.

        Parameters
        ----------
        voxel_params : mapping
            Per-voxel tissue parameters on the dose grid (``"alpha_x"``, ``"beta_x"``).

        Raises
        ------
        ValueError
            A supplied coefficient is negative, infinite or NaN, a declared parameter is
            missing, or a voxel inside a structure is outside the model's parameter domain.
        """
        present = [
            (name, voxel_params[name])
            for name in REFERENCE_PARAMETER_NAMES
            if voxel_params.get(name) is not None
        ]
        if not present:
            if self.requires_positive_reference:
                raise ValueError(
                    f"Biological model '{self.model}' requires the reference photon parameters "
                    f"{sorted(self.requires_positive_reference)}, which the dose calculation "
                    "did not provide."
                )
            return

        xp = array_api_compat.array_namespace(*[values for _, values in present])

        # Level 1: the reference coefficients themselves must be valid LQ rates. A negative,
        # infinite or NaN alpha_x / beta_x would otherwise reach the influence matrices.
        for name, values in present:
            invalid = ~(xp.isfinite(values) & (values >= 0.0))
            n_invalid = int(xp.sum(xp.astype(invalid, xp.int64)))
            if n_invalid:
                raise ValueError(
                    f"The reference photon parameter {name} must be finite and non-negative, "
                    f"but {n_invalid} voxel(s) are negative, infinite or NaN. Check the "
                    "alpha_x / beta_x of the structures used in this dose calculation."
                )

        names = self.requires_positive_reference
        if not names:
            return

        missing = [name for name in names if voxel_params.get(name) is None]
        if missing:
            raise ValueError(
                f"Biological model '{self.model}' requires the reference photon parameters "
                f"{sorted(missing)}, which the dose calculation did not provide."
            )

        # Level 2: a voxel outside every structure has *no* reference parameter set, so
        # "outside" is decided from all of them, not only from the ones this model needs.
        defined = present[0][1] != 0.0
        for _, values in present[1:]:
            defined = defined | (values != 0.0)

        for name in names:
            values = voxel_params[name]
            invalid = defined & ~(values > 0.0)
            n_invalid = int(xp.sum(xp.astype(invalid, xp.int64)))
            if n_invalid:
                raise ValueError(
                    f"Biological model '{self.model}' is only defined for {name} > 0, but "
                    f"{n_invalid} voxel(s) inside a structure have {name} <= 0. Set positive "
                    "reference photon alpha_x / beta_x values on every structure used in this "
                    "dose calculation."
                )


def _params_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.keys() != b.keys():
        return False
    for key, value in a.items():
        try:
            equal = bool(value == b[key])
        except ValueError:  # array-valued parameter
            equal = np.array_equal(value, b[key])
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
    possible_radiation_modes = ("photons", "protons", "helium", "carbon", "oxygen", "VHEE")
    default_report_quantity = "physical_dose"

    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        return ParametricEvaluator(self)
