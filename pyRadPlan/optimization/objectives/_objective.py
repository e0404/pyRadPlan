"""Base Implementation for objective functions."""

from abc import abstractmethod
from functools import lru_cache
from typing import Annotated, ClassVar, Any, Literal, Union, Optional
import logging

import numpy as np
from pydantic import (
    computed_field,
    Field,
    field_validator,
    model_validator,
    PrivateAttr,
    TypeAdapter,
)
import SimpleITK as sitk
import array_api_compat

from ...core.xp_utils.typing import Array
from ...core.xp_utils import to_numpy, from_numpy
from ...core.datamodel import PyRadPlanBaseModel
from ...quantities import get_available_quantities, get_quantity
from ...core import Grid
from ...core.resample import resample_array
from ..robustness import available_robustness, resolve_robustness_name

ParameterType = Union[
    Literal["reference", "numeric", "relative_volume", "image_reference"], list[str]
]

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _parameter_adapter(cls: type, param: str) -> TypeAdapter:
    """Build the validator of a single objective parameter field, constraints included."""
    field = cls.model_fields[param]
    return TypeAdapter(Annotated[(field.annotation, *field.metadata)])


class ParameterMetadata:
    """
    Parameter Metadata to attach to objective function parameters to designate configurability and type.

    Parameters
    ----------
    configurable : bool, optional, default=True
        Whether the parameter is intended to be configurable (e.g. a reference dose value for
        optimization)
    kind : str, optional, default='numeric'
        Type/Meaning of the parameter. Options are 'reference' (relates to the input vector, e.g.
        the dose when you optimize on dose. Used for normalization), 'numeric', 'relative_volume'.
        The kind is used in case the parameter needs to pre transformed in context.

    Attributes
    ----------
    configurable : bool, optional, default=True
        Whether the parameter is intended to be configurable (e.g. a reference dose value for
        optimization)
    kind : str, optional, default='numeric'
        Type/Meaning of the parameter. Options are 'reference' (relates to the input vector, e.g.
        the dose when you optimize on dose. Used for normalization), 'numeric', 'relative_volume'.
        The kind is used in case the parameter needs to pre transformed in context.

    Note
    ----
    Intended for use with 'Annotated' when defining an objective attribute.
    """

    configurable: bool
    kind: Optional[ParameterType]

    """Configurable Parameter."""

    def __init__(self, configurable: bool = True, kind: Optional[ParameterType] = "numeric"):
        self.configurable = configurable
        self.kind = kind

    def __repr__(self):
        return f"{self.__class__}({self.__dict__})"


class Objective(PyRadPlanBaseModel):
    """
    Base class for objective functions in the optimization problem.

    Attributes
    ----------
    name : str
        Name of the objective function. Concrete objectives narrow this to a
        ``Literal`` with a default, which makes it usable as a discriminator in
        tagged unions and for round-tripping serialized objectives.
    has_hessian : ClassVar[bool]
        Whether the objective function has a Hessian implementation.
    scenario_aware : ClassVar[bool]
        Whether the objective evaluates all scenarios at once: ``compute_objective`` /
        ``compute_gradient`` then receive the values stacked over the scenarios, shape
        ``(S, n)``, and the scenario ``weights``, and the gradient has the same shape.
    priority : float
        Weight/Priority assigned to the objective function.
    quantity : str
        The quantity this objective is connected to (e.g. 'physical_dose', 'RBExDose').
    robustness : str
        Name of the :mod:`~pyRadPlan.optimization.robustness` operator combining the
        objective's values over the scenarios of the dose influence data (``"none"``: the
        nominal scenario only). matRad's names (``STOCH``, ``OWC``, ...) are accepted as
        aliases and normalized.
    """

    name: str = Field(description="Name identifying the objective function type.")
    has_hessian: ClassVar[bool] = False
    scenario_aware: ClassVar[bool] = False
    priority: float = Field(
        default=1.0,
        ge=0.0,
        alias="penalty",
        description="Weight/priority of the objective in the optimization problem.",
    )
    quantity: str = Field(
        default="physical_dose",
        description="Quantity the objective is evaluated on (e.g. 'physical_dose').",
    )
    robustness: str = Field(
        default="none",
        description=(
            "Robustness operator combining the objective over the scenarios of the dose "
            "influence data ('none': nominal scenario only)."
        ),
    )

    _resampled_image_reference_cache: dict[str, Array] = PrivateAttr(default_factory=dict)

    def preprocess_image_reference_parameters(
        self, target_grid: Grid, index_list: Optional[Array] = None
    ):
        """
        Preprocess image reference parameters if existing in the objective definition.

        Preprocessing of reference image parameters is necessary to align with the corresponding
        target dose/optimization grid. The function will resample the reference image to the
        target grid and cache it.

        Parameters
        ----------
        target_grid : Grid
            The target grid that the parameter should match.
        index_list : Optional[Array], optional
                    Array containing indices for which the objective needs to be cached
        """

        # A fresh dict: shallow model copies share the cache with the objective they were
        # copied from.
        cache = dict(self._resampled_image_reference_cache)
        for param_name, param_type in zip(self.parameter_names, self.parameter_types):
            if param_type == "image_reference":
                # Check cache first
                cache_key = param_name

                param_value = getattr(self, param_name)

                # Get Array and Grid from parameter value
                if isinstance(param_value, sitk.Image):
                    ref_grid = Grid.from_sitk_image(param_value)
                    param_value = sitk.GetArrayViewFromImage(param_value)
                elif isinstance(param_value, tuple):
                    param_value, ref_grid = param_value

                # Get array namespace from parameter value
                xp = array_api_compat.array_namespace(param_value, index_list)

                # Resample the parameter to target grid
                resampled_array = resample_array(
                    input_array=to_numpy(param_value),
                    reference_grid=ref_grid,
                    target_grid=target_grid,
                )
                resampled_array = xp.reshape(
                    from_numpy(xp, resampled_array),
                    (array_api_compat.size(resampled_array),),
                    copy=False,
                )
                if index_list is not None:
                    resampled_array = resampled_array[index_list]

                cache[cache_key] = xp.asarray(resampled_array)
        self._resampled_image_reference_cache = cache

    def _image_reference(self, param_name: str, values: Array) -> Array:
        """Get cached image reference aligned to the namespace and device of *values*.

        Preprocessing builds the cache with numpy (image references live in numpy /
        SimpleITK), while *values* carry the optimization backend (e.g. cupy or
        torch). The first evaluation converts the cached array once and re-caches
        it, so subsequent iterations only pay the namespace/device check.
        """
        ref = self._resampled_image_reference_cache[param_name]
        xp = array_api_compat.array_namespace(values)
        if array_api_compat.array_namespace(ref) is not xp or array_api_compat.device(
            ref
        ) != array_api_compat.device(values):
            ref = xp.asarray(to_numpy(ref), device=array_api_compat.device(values))
            self._resampled_image_reference_cache[param_name] = ref
        return ref

    def normalize_to_fraction_num(self, num_of_fractions: int):
        """
        Convert total-course reference parameters to one of ``num_of_fractions`` fractions.

        The optimization runs on per-fraction quantities. Scalar (``"reference"``) and image
        (``"image_reference"``) parameters given for the total course are divided by the
        quantity's :meth:`~pyRadPlan.quantities.RTQuantity.total_course_factor`, i.e. by
        ``num_of_fractions`` for dose-like quantities and not at all for LET. Image references
        are scaled through their resampled cache only, so
        :meth:`preprocess_image_reference_parameters` must run first; the image parameter
        itself is left unchanged.

        Parameters
        ----------
        num_of_fractions : int
            The number of fractions to normalize to.

        Returns
        -------
        Objective
            The objective itself.

        Raises
        ------
        ValueError
            If the objective has reference parameters and ``num_of_fractions`` is not
            positive or the objective's quantity declares no fraction scaling, or if an image
            reference has not been preprocessed.
        """
        references = [
            (param_name, param_type)
            for param_name, param_type in zip(self.parameter_names, self.parameter_types)
            if param_type in ("reference", "image_reference")
        ]
        if not references:
            return self
        factor = get_quantity(self.quantity).total_course_factor(num_of_fractions)
        if factor == 1.0:
            return self
        cache = dict(self._resampled_image_reference_cache)
        for param_name, param_type in references:
            if param_type == "reference":
                setattr(self, param_name, getattr(self, param_name) / factor)
                continue
            if param_name not in cache:
                raise ValueError(
                    f"Image reference {param_name!r} has no resampled cache entry; "
                    "preprocess_image_reference_parameters must run before "
                    "normalize_to_fraction_num."
                )
            ref = cache[param_name]
            xp = array_api_compat.array_namespace(ref)
            if not xp.isdtype(ref.dtype, "real floating"):
                ref = xp.astype(ref, xp.float64)
            cache[param_name] = ref / factor
        self._resampled_image_reference_cache = cache
        return self

    @abstractmethod
    def compute_objective(self, values):
        """Compute the objective function."""

    @abstractmethod
    def compute_gradient(self, values):
        """Compute the objective gradient."""

    def compute_hessian(self, values):
        """Compute the objective Hessian."""
        return

    @computed_field
    @property
    def parameter_names(self) -> list[str]:
        """List[str]: Parameter names."""
        return self._parameter_names()

    @classmethod
    def _parameter_names(cls) -> list[str]:
        """List[str]: Parameter names as classmethod."""
        return [
            name
            for name, info in cls.model_fields.items()
            if any(isinstance(meta, ParameterMetadata) for meta in info.metadata)
        ]

    @computed_field
    @property
    def parameter_types(self) -> list[ParameterType]:
        """List[str]: Parameter types."""
        return self._parameter_types()

    @classmethod
    def _parameter_types(cls) -> list[ParameterType]:
        """List[str]: Parameter types as classmethod."""
        return [
            meta.kind
            for name in cls._parameter_names()
            for meta in cls.model_fields[name].metadata
            if isinstance(meta, ParameterMetadata)
        ]

    @computed_field
    @property
    def parameters(self) -> list[Any]:
        """List[str]: Parameter values."""
        return [getattr(self, name) for name in self.parameter_names]

    def to_matrad(self, context: str = "mat-file") -> Optional[dict]:
        """
        Serialize the objective into a matRad-compatible struct.

        Returns
        -------
        Optional[dict]
            A ``{"className", "parameters", "penalty"}`` dict matRad understands, or
            ``None`` when this objective has no matRad equivalent (in which case it is
            skipped on export). The shape mirrors what :func:`get_objective` consumes on
            import, so objectives round-trip through a ``.mat`` file.
        """
        if context != "mat-file":
            raise ValueError(f"Context {context} not supported")

        # Deferred import to avoid a circular import with the factory module.
        from ._factory import get_matrad_class_name  # noqa: PLC0415

        class_name = get_matrad_class_name(self)
        if class_name is None:
            logger.warning(
                "Objective '%s' has no matRad equivalent and is skipped on export.", self.name
            )
            return None

        # Deferred import to avoid a circular import with the factory module.
        from ..robustness import get_robustness_operator_class  # noqa: PLC0415

        matrad_robustness = get_robustness_operator_class(self.robustness).matrad_name
        if matrad_robustness is None:
            logger.warning(
                "Robustness '%s' of objective '%s' has no matRad equivalent and is exported as "
                "'none'.",
                self.robustness,
                self.name,
            )
            matrad_robustness = "none"

        # matRad stores choice parameters as one-based indices into their options.
        parameters = [
            options.index(value) + 1
            if (options := self._parameter_options(name)) is not None and value in options
            else value
            for name, value in zip(self.parameter_names, self.parameters)
        ]

        return {
            "className": class_name,
            "parameters": parameters,
            "penalty": self.priority,
            "robustness": matrad_robustness,
        }

    @classmethod
    def available_robustness(cls) -> list[str]:
        """
        Names of the robustness operators this objective accepts.

        Scenario-aware objectives evaluate the scenario stack themselves and accept only
        ``"scenario_stack"``; all other objectives accept every registered operator.
        """
        if cls.scenario_aware:
            return ["scenario_stack"]
        return [name for name in available_robustness() if name != "scenario_stack"]

    @field_validator("robustness")
    @classmethod
    def _validate_robustness(cls, v):
        """Normalize aliases to the canonical operator name and check it is applicable."""
        if cls.scenario_aware:
            if v not in ("none", "scenario_stack"):
                logger.warning(
                    "Objective '%s' evaluates the scenarios itself; robustness '%s' is replaced "
                    "by 'scenario_stack'.",
                    cls.model_fields["name"].default,
                    v,
                )
            return "scenario_stack"
        name = resolve_robustness_name(v)
        if name == "scenario_stack":
            raise ValueError("'scenario_stack' is reserved for scenario-aware objectives")
        if v.strip().upper() == "PROB":
            logger.warning(
                "matRad's 'PROB' robustness of objective '%s' is used as 'expected_quantity' "
                "without its variance term; add a 'Variance' objective to penalize the "
                "variance over the scenarios.",
                cls.model_fields["name"].default,
            )
        return name

    @field_validator("quantity")
    @classmethod
    def _validate_quantity(cls, v):
        """Validate the quantity attribute."""
        if v not in get_available_quantities():
            raise ValueError(
                f"Quantity {v} not available. Choose from {get_available_quantities()}"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def _validate_model(cls, data: Any) -> Any:
        """Pre-validate the input and perform conversions if necessary."""

        # Check if this is a matRad-like objective
        if isinstance(data, dict) and "className" in data:
            data = data.copy()

            # Should we confirm once more we have the correct objective?
            data.pop("className")

            # Copy into a fresh list: data.copy() is shallow, so popping from the
            # original ``parameters`` list would mutate the caller's input and break
            # a second validation of the same objective definition.
            params = data.get("parameters", [])
            if isinstance(params, list):
                params = list(params)
            else:
                # A single scalar (or numpy array) parameter arrives unwrapped.
                params = [params]

            # obtain the parameter names
            param_names = cls._parameter_names()
            objective_name = cls.model_fields["name"].default

            if len(params) != len(param_names):
                logger.warning(
                    "Objective '%s' expects %d parameters, but %d were provided; "
                    "missing parameters keep their defaults, extra ones are ignored.",
                    objective_name,
                    len(param_names),
                    len(params),
                )

            # matRad objectives may carry fewer parameters than their pyRadPlan
            # counterpart (matRad_EUD has no f_diff), so unmatched names keep defaults.
            for param, value in zip(param_names, params):
                try:
                    data[param] = cls._import_matrad_parameter(param, value)
                except ValueError as exc:
                    logger.warning(
                        "Objective '%s': invalid value %r for parameter '%s' (%s); "
                        "using the default %r.",
                        objective_name,
                        value,
                        param,
                        exc,
                        cls.model_fields[param].get_default(call_default_factory=True),
                    )

            data.pop("parameters", None)

        return data

    @classmethod
    def _parameter_options(cls, param: str) -> Optional[list[str]]:
        """Return the options of a choice parameter, or ``None`` for other kinds."""
        kind = cls._parameter_kind(param)
        return kind if isinstance(kind, list) else None

    @classmethod
    def _import_matrad_parameter(cls, param: str, value: Any) -> Any:
        """
        Convert and validate a single parameter value of a matRad objective struct.

        matRad stores choice parameters (e.g. ``f_diff``) as one-based indices into
        their option list; these are mapped to the option names, and names are matched
        case-insensitively.

        Raises
        ------
        ValueError
            If the value is not valid for the parameter.
        """
        options = cls._parameter_options(param)
        if options is not None:
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()
            if isinstance(value, str):
                matches = [opt for opt in options if opt.lower() == value.strip().lower()]
                if not matches:
                    raise ValueError(f"expected one of {options}")
                value = matches[0]
            elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
                value, bool
            ):
                if not float(value).is_integer() or not 1 <= int(value) <= len(options):
                    raise ValueError(f"expected an index from 1 to {len(options)}")
                value = options[int(value) - 1]

        if cls._parameter_kind(param) == "image_reference":
            return value
        return _parameter_adapter(cls, param).validate_python(value)

    @classmethod
    def _parameter_kind(cls, param: str) -> Optional[ParameterType]:
        for meta in cls.model_fields[param].metadata:
            if isinstance(meta, ParameterMetadata):
                return meta.kind
        return None
