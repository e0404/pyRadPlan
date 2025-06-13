"""Base class for ion sequencing."""

import logging
from abc import abstractmethod
from typing import Union

from ._base import Sequencer
from pyRadPlan.plan import Plan
from pyRadPlan.result import PlanResult
from pyRadPlan.stf import SteeringInformation

logger = logging.getLogger(__name__)


class IonSequencer(Sequencer):
    def __init__(self, pln: Union[Plan, dict] = None):
        self.num_of_levels = 5
        raise NotImplementedError("IonSequencer has yet to be implemented.")
        super().__init__(pln)

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
        dij (Optional): object
            The dij matrix or related input.

        Returns
        -------
        object
            The updated resultGUI after sequencing.
        """
        pass
