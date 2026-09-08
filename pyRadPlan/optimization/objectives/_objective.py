"""Base Implementation for objective functions."""

from abc import abstractmethod
from typing import ClassVar, Any, Literal, Union, Optional
import logging

from pydantic import computed_field, Field, field_validator, model_validator, PrivateAttr
import SimpleITK as sitk
import array_api_compat

from ...core.xp_utils.typing import Array
from ...core.xp_utils import to_numpy, from_numpy
from ...core.datamodel import PyRadPlanBaseModel
from ...quantities import get_available_quantities
from ...core import Grid
from ...core.resample import resample_numpy_array

ParameterType = Union[
    Literal["reference", "numeric", "relative_volume", "image_reference"], list[str]
]

logger = logging.getLogger(__name__)


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
    priority : float
        Weight/Priority assigned to the objective function.
    quantity : str
        The quantity this objective is connected to (e.g. 'physical_dose', 'RBExDose').
    """

    name: str = Field(description="Name identifying the objective function type.")
    has_hessian: ClassVar[bool] = False
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
                resampled_array = resample_numpy_array(
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

                # Cache the resampled value
                self._resampled_image_reference_cache[cache_key] = xp.asarray(resampled_array)

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

        return {
            "className": class_name,
            "parameters": list(self.parameters),
            "penalty": self.priority,
        }

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

            if len(params) != len(param_names):
                logger.warning(
                    "Objective '%s' expects %d parameters, but %d were provided.",
                    cls.model_fields["name"].default,
                    len(param_names),
                    len(params),
                )

            for param in param_names:
                data[param] = params.pop(0)

            data.pop("parameters", None)

        return data
