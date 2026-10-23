from __future__ import annotations

import inspect
import logging
import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Optional, get_type_hints

import array_api_compat
import numpy as np
from pydantic import create_model

from pyRadPlan.core import AlgorithmConfig
from pyRadPlan.biology.tissues import TissueProperties

from ._evaluator import BioModelEvaluator, ParametricEvaluator, _validate_name_tuple
from ._kernel_sets import KernelSetInfo, machine_kernel_sets
from ._resolution import (
    BioModelResolution,
    TissueComparison,
    TissueIssue,
    compare_tissue_parameters,
    resolve_bio_model,
    select_kernel_set,
)

logger = logging.getLogger(__name__)

#: Per-voxel reference photon LQ parameters a dose engine supplies to a biological model.
REFERENCE_PARAMETER_NAMES = ("alpha_x", "beta_x")


class BiologicalModel(ABC):
    """
    Abstract base class for biological models.

    A biological model is a lightweight, patient-agnostic bundle of parameters and pure
    functions. Everything that depends on a particular machine or patient geometry (tissue
    class lookup, pre-computed kernel tables, ...) lives in the
    :class:`~pyRadPlan.biology.models.BioModelEvaluator` created by :meth:`evaluator` for one
    dose calculation.

    Custom subclasses must provide immutable tuple declarations for aliases, required machine
    quantities, evaluator outputs and supported radiation modes. Register a concrete subclass
    with :func:`~pyRadPlan.biology.models.register_model`; constructor arguments are retained by
    :meth:`__new__` for :attr:`parameters` and :meth:`to_dict`, so constructor parameters should
    be serialisable.

    Attributes
    ----------
    model : str
        Canonical identifier of the formalism, stored by plans (e.g. ``"none"``,
        ``"alpha_beta_kernels"``).
    name : str or None
        Display name for user interfaces (e.g. ``"Constant RBE"``); ``None`` if it is
        ``model``.
    description : str
        One-line description for tooltips.
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
    provides_alpha_beta : bool
        Whether the model yields per-voxel LQ parameters; if so, the dose engine computes
        ``alpha_dose`` / ``sqrt_beta_dose`` influence matrices.
    requires_let : bool
        Whether the dose engine has to provide LET kernels to evaluate the model.
    kernel_set_kind : str or None
        Kind of machine kernel set the formalism reads (``"alpha_beta"``, ``"zs"``,
        ``"let"``); None if it reads none. Which set of that kind is used is resolved by
        :meth:`select_kernel_set`; the tissue type then follows the provenance of the set.
    output_quantities : tuple[str, ...]
        Named quantities returned by the model evaluator, such as ``("alpha", "beta")``.
    requires_positive_reference : tuple[str, ...]
        Names of the per-voxel reference photon LQ parameters (``"alpha_x"``, ``"beta_x"``)
        the model formulation is only defined for when they are strictly positive. Dose
        engines validate them over the whole dose grid before a calculation starts.
    required_tissue_parameters : tuple[str, ...]
        Tissue parameters the model reads off each structure's tissue properties at runtime
        (e.g. ``("alpha_0", "beta_0")`` for z* kernels). Every tissue must carry them as
        finite numbers, see :meth:`missing_tissue_parameters`.
    """

    model: ClassVar[str]
    name: ClassVar[Optional[str]] = None
    description: ClassVar[str] = ""
    model_aliases: ClassVar[tuple[str, ...]] = ()
    matrad_name: ClassVar[Optional[str]] = None
    required_quantities: ClassVar[tuple[str, ...]] = ()
    possible_radiation_modes: ClassVar[tuple[str, ...]]
    default_report_quantity: ClassVar[str] = "physical_dose"
    provides_alpha_beta: ClassVar[bool] = False
    requires_let: ClassVar[bool] = False
    kernel_set_kind: ClassVar[Optional[str]] = None
    output_quantities: ClassVar[tuple[str, ...]] = ()
    requires_positive_reference: ClassVar[tuple[str, ...]] = ()
    required_tissue_parameters: ClassVar[tuple[str, ...]] = ()
    _legacy_tissue_type: ClassVar[type[TissueProperties]] = TissueProperties

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

    @property
    def tissue_properties_type(self) -> type[TissueProperties]:
        """
        Tissue properties class of the base data this model was written for.

        .. deprecated:: 0.6
            The tissue type follows the provenance of the kernel set a model resolves to
            (see :meth:`resolve`); a model no longer fixes it.
        """
        warnings.warn(
            "BiologicalModel.tissue_properties_type is deprecated: the tissue type follows the "
            "provenance of the kernel set (see BiologicalModel.resolve).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._legacy_tissue_type

    @classmethod
    def display_name(cls) -> str:
        """Display name of the formalism, ``model`` if it declares none."""
        return cls.name or cls.model

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
        if cls.name is not None and (not isinstance(cls.name, str) or not cls.name):
            raise ValueError("Biological model name must be None or a non-empty string.")
        if not isinstance(cls.description, str):
            raise ValueError("Biological model description must be a string.")

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
        _validate_name_tuple(
            cls.required_tissue_parameters,
            "required_tissue_parameters",
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

    def missing_tissue_parameters(self, tissue: TissueProperties) -> tuple[str, ...]:
        """
        Return the :attr:`required_tissue_parameters` a tissue does not set to a finite value.

        A parameter the tissue's type does not declare at all counts as missing, as does one
        left unset (NaN).
        """
        missing = []
        for name in self.required_tissue_parameters:
            value = getattr(tissue, name, None)
            try:
                valid = value is not None and math.isfinite(float(value))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                missing.append(name)
        return tuple(missing)

    def validate_tissue_parameters(self, tissues: Any) -> None:
        """
        Raise ValueError if a tissue lacks a parameter the model reads at runtime.

        Parameters
        ----------
        tissues : iterable of TissueProperties
            The tissues of the structures a dose calculation evaluates the model for.

        Raises
        ------
        ValueError
            Some tissue does not carry every :attr:`required_tissue_parameters` as a finite
            number, which would otherwise evaluate to NaN in every voxel of its structures.
        """
        if not self.required_tissue_parameters or tissues is None:
            return
        problems = [
            f"{tissue!r} lacks {', '.join(missing)}"
            for tissue in tissues
            if (missing := self.missing_tissue_parameters(tissue))
        ]
        if problems:
            raise ValueError(
                f"Biological model '{self.model}' reads the tissue parameters "
                f"{', '.join(self.required_tissue_parameters)} of every structure, but "
                f"{'; '.join(problems)}. Set them on the tissue properties of these structures."
            )

    # ------------------------------------------------------------ resolution
    def quantity_variant(self, quantity: str) -> Optional[str]:
        """Variant configured for a machine quantity (``quantity_variants``), if any."""
        variants = getattr(self, "quantity_variants", None) or {}
        return variants.get(quantity)

    def select_kernel_set(
        self, machine: Any, tissues: Optional[list[TissueProperties]] = None
    ) -> Optional[KernelSetInfo]:
        """
        Select the machine kernel set this model reads, None if it reads none.

        Applies the resolution rules of :func:`~pyRadPlan.biology.models.select_kernel_set`
        with the model's ``kernel_set``, ``data_model`` and ``quantity_variants``
        configuration. A LET model may run without a LET kernel set if the dose engine
        provides LET itself, so a missing LET set is not an error here.

        Parameters
        ----------
        machine : Machine or BioMachineSummary
            The machine.
        tissues : list of TissueProperties, optional
            The structures' tissues, used to choose among several fitting sets.

        Raises
        ------
        KernelSetSelectionError
            No kernel set fits, or several fit equally.
        """
        kind = self.kernel_set_kind
        if kind is None:
            return None
        sets = list(machine_kernel_sets(machine).values())
        variant = self.quantity_variant(kind)
        if kind == "let" and variant is None and not any(s.kind == kind for s in sets):
            return None
        return select_kernel_set(
            sets,
            kind=kind,
            model_name=self.model,
            kernel_set=getattr(self, "kernel_set", None),
            data_model=getattr(self, "data_model", None),
            variant=variant,
            tissues=tissues,
        )

    def kernel_set_issue(self, kernel_set: KernelSetInfo) -> Optional[TissueIssue]:
        """Reason the selected kernel set cannot be evaluated by dose engines, if any."""
        return None

    def resolve(
        self,
        machine: Any = None,
        cst: Any = None,
        *,
        provided_quantities: Optional[list[str]] = None,
        engine: Any = None,
    ) -> BioModelResolution:
        """
        Resolve the model against a machine and the tissues of a structure set.

        Selects the kernel set the model reads and, per structure, the tissue class and
        kernel column holding its base data; see :class:`BioModelResolution`.

        Parameters
        ----------
        machine : Machine, BioMachineSummary or None
            The machine, or a summary of it (:meth:`BioMachineSummary.from_machine`).
        cst : StructureSet, sequence of VOIs, mapping or sequence of TissueProperties
            The structures whose tissues must resolve.
        provided_quantities : list[str], optional
            Quantities available to the model, defaulting to the machine's as the dose
            engine sees them.
        engine : str, type, DoseEngineBase or EngineBioCapability, optional
            The dose engine, see :func:`~pyRadPlan.biology.models.resolve_bio_model`.
        """
        return resolve_bio_model(
            self, machine, cst, provided_quantities=provided_quantities, engine=engine
        )

    def check_tissues(
        self,
        machine: Any = None,
        cst: Any = None,
        *,
        provided_quantities: Optional[list[str]] = None,
        engine: Any = None,
    ) -> list[TissueIssue]:
        """
        Check that the model can be evaluated for a machine and every structure's tissue.

        Dose engines run the same check, with themselves as ``engine``, before building
        the evaluator: with ``calc_bio_dose=True`` any error raises, with ``"auto"`` the
        engine falls back to the physical dose, and ``invalid_reference`` issues always
        raise. Warnings (:attr:`TissueIssue.severity`, e.g. a tissue labelled MKM matching
        mMKM base data) are logged and reported, and the model is evaluated.

        Parameters
        ----------
        machine : Machine, BioMachineSummary or None
            The machine, or a cacheable summary of it. Without one, only the checks not
            needing base data are made (reference and runtime tissue parameters, and what
            the engine can compute).
        cst : StructureSet, sequence of VOIs, mapping or sequence of TissueProperties
            The structures; every one of them must resolve.
        provided_quantities : list[str], optional
            Quantities available to the model, defaulting to the machine's as the dose
            engine sees them.
        engine : str, type, DoseEngineBase or EngineBioCapability, optional
            The dose engine (short name, class or instance). A model the engine cannot
            compute is reported as ``model_unavailable``, e.g. α/β kernels with a Monte
            Carlo engine not using the machine's kernels, or a LET model with an engine
            neither reading LET kernels nor scoring LET. Without, only the machine counts.

        Returns
        -------
        list[TissueIssue]
            One issue per problem: those of the model and machine first, then one or more
            per affected structure. Without errors (:attr:`TissueIssue.is_error`) the model
            can be evaluated; it holds warnings only, or is empty.
        """
        return list(
            self.resolve(
                machine, cst, provided_quantities=provided_quantities, engine=engine
            ).issues
        )

    def compare_tissues(
        self,
        machine: Any = None,
        cst: Any = None,
        *,
        provided_quantities: Optional[list[str]] = None,
        engine: Any = None,
    ) -> list[TissueComparison]:
        """
        Compare every structure's tissue parameter by parameter with its tissue class.

        The per-parameter view of :meth:`check_tissues`, judging the same kernel set and
        tissue classes; see :func:`~pyRadPlan.biology.models.compare_tissue_parameters`.
        """
        return compare_tissue_parameters(
            self, machine, cst, provided_quantities=provided_quantities, engine=engine
        )

    def validate_reference_parameters(self, voxel_params: Mapping[str, Any]) -> None:
        """
        Check the per-voxel reference photon LQ parameters against the model's domain.

        Two levels are checked. First, every supplied reference coefficient must be a valid
        LQ rate: finite and non-negative, regardless of the model
        (:func:`validate_reference_coefficients`). Second, the parameters
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
        validate_reference_coefficients(voxel_params)

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


class BioModelUnavailableError(ValueError):
    """
    A biological model cannot be evaluated for the machine and patient of a dose calculation.

    Raised for model-specific reasons only: the machine lacks the quantities the model needs,
    a patient tissue has no matching tissue class in the base data, or the reference photon
    parameters are outside the model's domain. Dose engines treat it as fatal only when
    biological dose is explicitly requested (``calc_bio_dose=True``); otherwise they compute
    the physical dose alone. Invalid reference coefficients (see
    :func:`validate_reference_coefficients`) are data errors and are never downgraded.
    """


def raise_for_tissue_issues(model: BiologicalModel, issues: Any) -> list[TissueIssue]:
    """
    Raise for the errors among the issues of :meth:`BiologicalModel.check_tissues`.

    This is how dose engines act on the check. Warnings (e.g. ``provenance_differs``) do
    not raise; they are returned for the caller to log and report, as the model is
    evaluated nonetheless.

    Returns
    -------
    list[TissueIssue]
        The warnings among the issues, if there is no error.

    Raises
    ------
    ValueError
        A structure's reference α_x / β_x is invalid (``invalid_reference``); a data error
        that is never downgraded to a physical-dose calculation.
    BioModelUnavailableError
        Any other error: the model cannot be evaluated for this machine and these
        structures.
    """
    issues = list(issues)
    errors = [issue for issue in issues if issue.is_error]
    invalid = [issue for issue in errors if issue.kind == "invalid_reference"]
    if invalid:
        raise ValueError(
            "; ".join(str(issue) for issue in invalid)
            + " Check the alpha_x / beta_x of the structures used in this dose calculation."
        )
    if errors:
        raise BioModelUnavailableError(
            f"Biological model '{model.model}' cannot be evaluated for this machine and these "
            f"structures: {'; '.join(str(issue) for issue in errors)}"
        )
    return [issue for issue in issues if not issue.is_error]


def validate_reference_coefficients(voxel_params: Mapping[str, Any]) -> None:
    """
    Check that the supplied reference photon LQ coefficients are valid LQ rates.

    Independent of any biological model: every supplied ``alpha_x`` / ``beta_x`` must be
    finite and non-negative, as a negative, infinite or NaN value would otherwise reach the
    influence matrices.

    Parameters
    ----------
    voxel_params : mapping
        Per-voxel tissue parameters on the dose grid (``"alpha_x"``, ``"beta_x"``).

    Raises
    ------
    ValueError
        A supplied coefficient is negative, infinite or NaN.
    """
    present = [
        (name, voxel_params[name])
        for name in REFERENCE_PARAMETER_NAMES
        if voxel_params.get(name) is not None
    ]
    if not present:
        return
    xp = array_api_compat.array_namespace(*[values for _, values in present])
    for name, values in present:
        invalid = ~(xp.isfinite(values) & (values >= 0.0))
        n_invalid = int(xp.sum(xp.astype(invalid, xp.int64)))
        if n_invalid:
            raise ValueError(
                f"The reference photon parameter {name} must be finite and non-negative, "
                f"but {n_invalid} voxel(s) are negative, infinite or NaN. Check the "
                "alpha_x / beta_x of the structures used in this dose calculation."
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
    name = "None (physical dose)"
    description = "No biological weighting; only the physical dose is computed."
    possible_radiation_modes = ("photons", "protons", "helium", "carbon", "oxygen", "VHEE")
    default_report_quantity = "physical_dose"

    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        return ParametricEvaluator(self)
