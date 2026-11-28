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
    result : PlanResult
        Result object to store the sequencing output.
    pln : Plan
        Plan object containing the sequencing configuration.
    stf : SteeringInformation
        SteeringInformation object for sequencing.
    dij : Dij, optional
        Dij object, if available.

    Returns
    -------
    PlanResult
        Updated result object containing the sequencing output.
    """

    _stf = validate_stf(stf)
    _pln = validate_pln(pln)
    _dij = validate_dij(dij) if dij is not None else None
    if _dij is None:
        logger.warning(
            "No dij given: sequencing will produce deliverable shapes, but no delivered dose "
            "can be computed from the sequenced fluence, so the result carries no "
            "'*_sequenced' dose quantities."
        )

    sequencing_prob = get_sequencer_from_pln(_pln)

    if sequencing_prob is None:
        logger.error("No sequencing found in the plan. Running without...")
        return result

    result = sequencing_prob.sequence(result, _stf, _dij)
    if result.sequence is not None:
        result.sequence.sequencer = sequencing_prob.short_name
        result.sequence.num_of_levels = sequencing_prob.num_of_levels

    return result
