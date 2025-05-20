"""Base class for sequencing."""

import logging

from abc import ABC, abstractmethod
from typing import ClassVar, Union
import warnings

from pyRadPlan.result import PlanResult, ApertureInfo
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.plan import Plan, validate_pln

logger = logging.getLogger(__name__)


class Sequencer(ABC):
    name: ClassVar[str]
    short_name: ClassVar[str]

    def __init__(self, pln: Union[Plan, dict] = None):
        self.resultGUI: PlanResult
        self.aperture_info = None
        self.num_of_levels = 5
        if pln is not None:
            self.assign_properties_from_pln(pln, True)

    @abstractmethod
    def sequence(self, result: PlanResult, stf: SteeringInformation, dij) -> PlanResult:
        """
        Execute the sequencing algorithm.

        Parameters
        ----------
        resultGUI : object
            The result object or dictionary to store sequencing results.
        stf : object
            The steering information structure.
        dij : object
            The dij matrix or related input.
        num_of_levels : int
            The number of intensity levels used for stratification.

        Returns
        -------
        object
            The updated resultGUI after sequencing.
        """
        pass

    def sequencing_to_aperture_info(self, sequencing, stf) -> ApertureInfo:
        """
        Convert the sequencing result to an ApertureInfo object.

        Parameters
        ----------
        sequencing : object
            The sequencing result.
        stf : object
            The steering information.

        Returns
        -------
        ApertureInfo
            The aperture information.
        """
        pass

    def assign_properties_from_pln(self, pln: Plan, warn_when_property_changed: bool = False):
        """
        Assign properties from a Plan object to the Sequencer.

        This includes the Scenario Model and the Biological Model, and any
        other properties that
        can be stored in the prop_dose_calc dictionary within the Plan object.
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

        # Assign Biologival Model
        if hasattr(pln, "bio_param"):
            self.bio_param = pln.bio_param  # TODO: No bio_param yet

        if not isinstance(warn_when_property_changed, bool):
            warn_when_property_changed = False

        # Overwrite default properties within the engine
        # with the ones given in the prop_dose_calc dict
        if pln.prop_seq is not None:
            prop_dict = pln.prop_seq
            if (
                "sequencer" in prop_dict
                and prop_dict["sequencer"]
                and prop_dict["sequencer"] != self.short_name
            ):
                raise ValueError(
                    f"Inconsistent dose engines given! pln asks for '{prop_dict['sequencer']}', "
                    f"but you are using '{self.short_name}'!"
                )
            prop_dict.pop("sequencer", None)
        else:
            prop_dict = {}

        fields = prop_dict.keys()

        # Set up warning message
        if warn_when_property_changed:
            warning_msg = "Property in Dose Engine overwritten from pln.propDoseCalc"
        else:
            warning_msg = None

        for field in fields:
            if not hasattr(self, field):
                warnings.warn('Property "{}" not found in Dose Engine!'.format(field))
            elif warn_when_property_changed and warning_msg:
                logger.warning(warning_msg + f": {field}")

            setattr(self, field, prop_dict[field])
