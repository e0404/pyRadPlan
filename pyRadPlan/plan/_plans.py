"""
Contains the definition of the Plan class and its derived classes.

Available spezialized Plan classes are PhotonPlan and IonPlan.
"""

from abc import ABC
from typing import Dict, Any, List, Union, ClassVar, Optional
from copy import deepcopy

from pydantic import (
    Field,
    field_validator,
    field_serializer,
    SerializationInfo,
    ValidationError,
    model_validator,
)
from pydantic.alias_generators import to_snake
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.scenarios import ScenarioModel, create_scenario_model, validate_scenario_model
from pyRadPlan.bio_models import BiologicalModel, create_bio_model

default_bio_models: dict[str, str] = {
    "photons": "none",
    "protons": "constant_rbe",
    "helium": "HEL",
    "carbon": "kernel_based_lq",
    "oxygen": "kernel_based_lq",
    "VHEE": "none",
}


class Plan(PyRadPlanBaseModel, ABC):
    """
    Base class representing a treatment plan.

    Attributes
    ----------
    prop_stf : Dict[str, Any]
        Properties of the stf.
    prop_opt : Dict[str, Any]
        Properties of the optimization.
    prop_dose_calc : Dict[str, Any]
        Properties of the dose calculation.
    prop_seq : Dict[str, Any]
        Properties for the sequencer
    num_of_fractions : int
        Number of fractions in the plan.
    machine : str
        Machine used for the plan.
    prescribed_dose : float
        Prescribed dose for the plan. Serves mainly as normalization value.
    radiation_mode : str
        Will return the radiation modality (e.g. photons or protons).
    """

    prop_stf: Dict[str, Any] = Field(default_factory=dict)
    prop_opt: Dict[str, Any] = Field(default_factory=dict)
    prop_dose_calc: Dict[str, Any] = Field(default_factory=dict)
    prop_seq: Dict[str, Any] = Field(default_factory=dict)
    num_of_fractions: int = Field(default=30, gt=0)
    machine: Union[Dict, str] = Field(default="Generic")
    prescribed_dose: float = Field(default=60.0, gt=0.0)
    mult_scen: ScenarioModel = Field(default_factory=create_scenario_model)
    bio_model: Optional[Any] = Field(
        default=None,
        description="Biological model: name, {'model': name, **parameters} or instance. "
        "Defaults per radiation mode.",
    )

    radiation_mode: str

    @field_validator("radiation_mode", mode="after")
    @classmethod
    def validate_radiation_mode(cls, v: str) -> str:
        """
        Validate the radiation mode.

        Parameters
        ----------
        v : str
            The radiation mode value to be validated.

        Raises
        ------
        NotImplementedError
            This method should be overridden in derived classes.
        """
        raise NotImplementedError("This method should be overridden in derived classes")

    @model_validator(mode="after")
    def validate_bio_model(self) -> "Plan":
        """
        Resolve ``bio_model`` into a :class:`BiologicalModel` instance.

        Accepts a model name, a ``{"model": name, **parameters}`` dict or an instance;
        falls back to the per-modality default. The model must support the plan's
        radiation mode; availability against the machine data is checked by the dose
        engine.
        """
        spec = self.bio_model
        if spec is None:
            spec = default_bio_models.get(self.radiation_mode, "none")
        if isinstance(spec, BiologicalModel):
            create_bio_model(spec, self.radiation_mode)  # radiation mode check only
        else:
            self.bio_model = create_bio_model(spec, self.radiation_mode)
        return self

    @field_serializer("bio_model")
    def _serialize_bio_model(self, value: Any, info: SerializationInfo) -> Any:
        if not isinstance(value, BiologicalModel):
            return value
        context = info.context or {}
        if context.get("matRad"):
            return value.model
        return value.to_dict()

    @field_validator("mult_scen", mode="before")
    @classmethod
    def _validate_mult_scen(
        cls, v: Union[Union[Dict[str, Any], ScenarioModel], str]
    ) -> ScenarioModel:
        """
        Validate the mult_scen attribute.

        Parameters
        ----------
        v : Union[Dict[str, Any], ScenarioModel]
            The mult_scen attribute to be validated.

        Returns
        -------
        ScenarioModel
            The validated mult_scen attribute.

        Raises
        ------
        ValueError
            If the mult_scen attribute is not a valid ScenarioModel object.
        """

        try:
            return validate_scenario_model(v)
        except ValueError as exc:
            raise ValidationError(
                "mult_scen must be a ScenarioModel object or respective dictionary"
            ) from exc

    @field_validator("prop_stf", "prop_opt", "prop_dose_calc", "prop_seq", mode="before")
    @classmethod
    def _coerce_empty_prop(cls, v: Any) -> Dict[str, Any]:
        """Coerce missing/empty property blocks to an empty dict.

        MATLAB ``.mat`` round-trips turn an empty ``{}`` struct into ``None`` (or an
        empty array); accept those so a saved plan re-validates.
        """
        if v is None or (hasattr(v, "__len__") and len(v) == 0):
            return {}
        return v

    @field_validator("prop_stf", "prop_opt", "prop_dose_calc", "prop_seq", mode="after")
    @classmethod
    def validate_prop(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the workflow property dictionaries.

        Will try to convert to snake_case if camelCase is used.

        Parameters
        ----------
        v : Dict[str, Any]
            The properties of the plan to be validated.

        Returns
        -------
        Dict[str, Any]
            The validated properties of the plan.
        """

        if not v:
            return {}

        # Convert camelCase to snake_case
        return {to_snake(k): v for k, v in v.items()}

    def to_matrad(self, context: str = "mat-file") -> Any:
        """
        Create a dictionary ready to save the Plan model to a mat-file.

        Returns
        -------
            Dict: A dictionary containing the data of the Plan model in a format suitable for
            saving to a mat-file.
        """

        pln_dict = super().to_matrad(context=context)
        pln_dict["numOfFractions"] = float(pln_dict["numOfFractions"])
        return pln_dict


class PhotonPlan(Plan):
    """
    Class representing a photon treatment plan.

    Attributes
    ----------
    Inherits all attributes from Plan.

    Methods
    -------
    radiation_mode : str
        Returns the radiation mode as 'photons'.
    """

    radiation_mode: str = "photons"

    @field_validator("radiation_mode", mode="after")
    @classmethod
    def validate_radiation_mode(cls, v: str) -> str:
        """
        Validate the radiation mode for a PhotonPlan.

        Parameters
        ----------
        v : str
            The radiation mode to be validated.

        Returns
        -------
        str
            The validated radiation mode.

        Raises
        ------
        ValueError
            If the radiation mode is not "photons".
        """
        if v != "photons":
            raise ValueError('radiation_mode for PhotonPlan must be "photons"')
        return v


class IonPlan(Plan):
    """
    Class representing an ion treatment plan.

    Attributes
    ----------
    ionType : str
        Type of ion used in the plan.
    Inherits all other attributes from Plan.

    Methods
    -------
    radiation_mode : str
        Returns the radiation mode as the ion type.
    """

    available_radiation_modes: ClassVar[List[str]] = [
        "protons",
        "helium",
        "carbon",
        "oxygen",
        "VHEE",
    ]

    radiation_mode: str = Field(
        default="protons", pattern="^(protons|helium|carbon|oxygen|VHEE)$", validate_default=True
    )

    @field_validator("radiation_mode", mode="after")
    @classmethod
    def validate_radiation_mode(cls, v: str) -> str:
        """
        Validate the radiation mode for IonPlan.

        Parameters
        ----------
        cls : class
            The class object.
        v : str
            The radiation mode to be validated.

        Returns
        -------
        str
            The validated radiation mode.

        Raises
        ------
        ValueError
            If the radiation mode is not one of the available radiation modes.
        """
        if v not in cls.available_radiation_modes:
            raise ValueError(
                f"radiation_mode for IonPlan must be one of {cls.available_radiation_modes}"
            )
        return v


def create_pln(data: Union[Dict[str, Any], Plan, None] = None, **kwargs) -> Plan:
    """
    Create a Plan object (factory function).

    Parameters
    ----------
    data : Union[Dict[str, Any], None]
        Dictionary containing the data to create the Plan object.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    Plan
        A Plan object.

    Raises
    ------
    ValueError
        If the radiation mode is unknown or empty.
    """
    data = deepcopy(data)
    if data:
        # If data is already a Plan object, return it directly
        if isinstance(data, Plan):
            return data

        # obtain the radiation mode if we have a dictionary at our hands
        radiation_mode = data.get("radiation_mode")
        data["bio_model"] = data.get("bio_model") or default_bio_models.get(radiation_mode, "none")

        # Since we also allow camelCase, try to get radiationMode if radiation_mode is not set
        if radiation_mode is None:
            radiation_mode = data.get("radiationMode")

        if radiation_mode == "photons":
            return PhotonPlan.model_validate(data)
        # radiation_mode in ['protons', 'helium', 'carbon', 'oxygen']:
        return IonPlan.model_validate(data)
        # raise ValueError(f"Unknown radiation mode: {radiation_mode}")
    radiation_mode = kwargs.get("radiation_mode", "")
    if radiation_mode == "photons":
        return PhotonPlan(**kwargs)
    if radiation_mode in ["protons", "helium", "carbon", "oxygen"]:
        return IonPlan(**kwargs)
    raise ValueError(f"Unknown radiation mode: {radiation_mode}")


def validate_pln(plan: Union[Dict[str, Any], Plan, None] = None, **kwargs) -> Plan:
    """
    Validate a Plan object.

    Synonym to create_pln but should be used in validation context.

    Parameters
    ----------
    plan : Union[Dict[str, Any], Plan, None], optional
        Dictionary containing the data to create the Plan object, by default None.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    Plan
        A validated Plan object.

    Raises
    ------
    ValueError
        If the radiation mode is unknown or empty.
    """
    return create_pln(plan, **kwargs)
