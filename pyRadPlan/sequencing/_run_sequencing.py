"""Run sequencing based on the plan and steering information."""

import logging

from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.dij import Dij, validate_dij
from pyRadPlan.stf import SteeringInformation, validate_stf
from pyRadPlan.result import PlanResult
from .sequencer import get_sequencer_from_pln

logger = logging.getLogger(__name__)


def run_sequencing(
    result: PlanResult, pln: Plan, stf: SteeringInformation, dij: Dij = None
) -> PlanResult:
    """
    Trigger sequencing using the configuration stored in the pln object.

    Parameters
    ----------
    ct : CT
        CT object.
    cst : StructureSet
        StructureSet object.
    stf : SteeringInformation
        SteeringInformation object.
    dij : Dij
        Dij object.
    pln : Plan
        Plan object.

    Returns
    -------
    PlanResult
        Result object containing the requested dose cubes and information.
    """

    _stf = validate_stf(stf)
    _pln = validate_pln(pln)
    if dij is not None:
        _dij = validate_dij(dij)

    sequencing_prob = get_sequencer_from_pln(_pln)

    if sequencing_prob is None:
        logger.error("No sequencing found in the plan. Running without...")
        return result

    result = sequencing_prob.sequence(result, _stf, _dij)

    return result
