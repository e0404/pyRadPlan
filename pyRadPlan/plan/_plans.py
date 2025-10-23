"""
Contains the definition of the Plan class and its derived classes.

Available spezialized Plan classes are PhotonPlan and IonPlan.
"""

from abc import ABC
from typing import Dict, Any, List, Union, ClassVar
from copy import deepcopy

from pydantic import (
    Field,
    field_validator,
    ValidationError,
)
from pydantic.alias_generators import to_snake
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.scenarios import ScenarioModel, create_scenario_model, validate_scenario_model


class Plan(PyRadPlanBaseModel, ABC):
    """
    Abstract base class for a treatment plan using PyRadPlanBaseModel.

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

    # Abstract property handled by below validator
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

    @field_validator("prop_stf", "prop_opt", "prop_dose_calc", "prop_opt", mode="after")
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
    Class for a photon treatment plan.

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
    Class for an ion treatment plan.

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
    Validate and create a Plan object.

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


if __name__ == "__main__":
    scen = create_scenario_model("nomScen")
    scen_dict_camel = scen.to_matrad()
    scen_dict_snake = scen.to_dict()

    pln_dict_camel = {
        "radiationMode": "photons",  # either photons / protons / carbon
        "machine": "Generic",
        "numOfFractions": 30,
        "prescribedDose": 60.0,
        "propStf": {},
        # dose calculation settings
        "propDoseCalc": {},
        # optimization settings
        "propOpt": {},
        "propSeq": {},
        "multScen": scen_dict_camel,
    }

    pln = create_pln(pln_dict_camel)
