"""Base class for sequencing."""

import logging

from abc import ABC
from typing import ClassVar, Union
import warnings

from pydantic.alias_generators import to_snake
from abc import abstractmethod

from pyRadPlan.result import PlanResult
from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.dij import Dij

logger = logging.getLogger(__name__)


class Sequencer(ABC):
    name: ClassVar[str]
    short_name: ClassVar[str]

    def __init__(self, pln: Union[Plan, dict] = None):
        self.resultGUI: PlanResult
        self.aperture_info = None

        if pln is not None:
            self.assign_properties_from_pln(pln, True)

    @abstractmethod
    def sequence(self, result: PlanResult, stf: SteeringInformation, dij: Dij = None):
        """
        Abstract method to be implemented by subclasses for sequencing.

        Parameters
        ----------
        result : PlanResult
            The result object to store the sequencing results.
        stf : SteeringInformation
            The steering information structure.
        dij : Dij, optional
            The dij matrix or related input.

        Returns
        -------
        PlanResult
            The updated result after sequencing.
        """
        raise NotImplementedError("Sequencing method must be implemented in subclasses.")

    def assign_properties_from_pln(self, pln: Plan, warn_when_property_changed: bool = False):
        """
        Assign properties from a Plan object to the Sequencer.

        This includes the Scenario Model and the Biological Model, and any
        other properties that
        can be stored in the prop_seq dictionary within the Plan object.
        This function will check if a property exists for the dose engine and,
        if yes, set it.

        Parameters
        ----------
        pln : Plan
            The Plan object to assign properties from.
        warn_when_property_changed : bool
            Whether to warn when properties are changed.
        """

        pln = validate_pln(pln)

        # Set Scenario Model
        if hasattr(pln, "mult_scen"):
            self.mult_scen = pln.mult_scen

        if not isinstance(warn_when_property_changed, bool):
            warn_when_property_changed = False

        # Overwrite default properties within the engine
        # with the ones given in the prop_seq dict
        if pln.prop_seq is not None:
            prop_dict = pln.prop_seq
            if (
                "sequencer" in prop_dict
                and prop_dict["sequencer"]
                and prop_dict["sequencer"] != self.short_name
            ):
                raise ValueError(
                    f"Inconsistent sequencer given! pln asks for '{prop_dict['sequencer']}', "
                    f"but you are using '{self.short_name}'!"
                )
            prop_dict.pop("sequencer", None)
            if "sequencing_level" in prop_dict:
                self.num_of_levels = prop_dict.pop("sequencing_level")
            elif "sequencingLevel" in prop_dict:
                self.num_of_levels = prop_dict.pop("sequencingLevel")
        else:
            prop_dict = {}

        fields = prop_dict.keys()

        # Set up warning message
        if warn_when_property_changed:
            warning_msg = "Property in Dose Engine overwritten from pln.prop_seq"
        else:
            warning_msg = None

        for field in fields:
            if not hasattr(self, field) or not hasattr(self, to_snake(field)):
                warnings.warn('Property "{}" not found in Sequencer!'.format(field))
            elif warn_when_property_changed and warning_msg:
                logger.warning(warning_msg + f": {field}")

            setattr(self, to_snake(field), prop_dict[field])
