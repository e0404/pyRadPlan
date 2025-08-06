"""Base Implementation for objective functions."""

from abc import abstractmethod
from typing import ClassVar, Any, Literal, Union, Optional
import logging

from pydantic import computed_field, Field, field_validator, model_validator

from pyRadPlan.core.datamodel import PyRadPlanBaseModel
from pyRadPlan.quantities import get_available_quantities

ParameterType = Union[Literal["reference", "numeric", "relative_volume"], list[str]]

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
    name : ClassVar[str]
        Name of the objective function.
    has_hessian : ClassVar[bool]
        Whether the objective function has a Hessian implementation.
    priority : float
        Weight/Priority assigned to the objective function.
    quantity : str
        The quantity this objective is connected to (e.g. 'physical_dose', 'RBExDose').
    """

    name: ClassVar[str]
    has_hessian: ClassVar[bool] = False
    priority: float = Field(default=1.0, ge=0.0, alias="penalty")
    quantity: str = Field(default="physical_dose")

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

            params = data.get("parameters", [])

            # If there are not more than one parameter,
            # it will usually not be in a list so we put it into one
            if not isinstance(params, list):
                params = [params]

            # obtain the parameter names
            param_names = cls._parameter_names()

            if len(params) != len(param_names):
                logger.warning(
                    "Objective '%s' expects %d parameters, but %d were provided.",
                    cls.name,
                    len(param_names),
                    len(params),
                )

            for param in param_names:
                data[param] = params.pop(0)

            data.pop("parameters", None)

        return data
