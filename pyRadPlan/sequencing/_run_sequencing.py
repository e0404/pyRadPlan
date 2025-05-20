import logging

from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.dij import Dij, validate_dij
from pyRadPlan.stf import SteeringInformation, validate_stf
from pyRadPlan.result import PlanResult
from pyRadPlan.sequencing import (
    Sequencer,
    XiaLeafSequencer,
    SiochiLeafSequencer,
    EngelLeafSequencer,
)

logger = logging.getLogger(__name__)


def get_sequencer_from_pln(pln: Plan) -> Sequencer:
    """Get the sequencer from the plan object."""

    if pln.prop_seq is None:
        pln.prop_seq["run_sequencing"] = False

    elif hasattr(pln, "prop_seq") and "sequencer" in pln.prop_seq:
        pln.prop_seq["run_sequencing"] = True

    # For photon radiation mode and when either sequencing or DAO is enabled.
    if pln.radiation_mode == "photons" and (
        pln.prop_seq.get("run_sequencing", False) or pln.prop_opt.get("runDAO", False)
    ):
        # Check for sequencer setting, otherwise default to "siochi".
        if "sequencer" not in pln.prop_seq:
            pln.prop_seq["sequencer"] = "siochi"
            logger.warning(
                "pln.prop_seq.sequencer not specified. Using siochi leaf sequencing (default)."
            )
        # Check for sequencingLevel setting, otherwise default to 5.
        if "sequencing_level" not in pln.prop_seq:
            pln.prop_seq["sequencing_level"] = 5
            logger.warning(
                "pln.prop_seq.sequencing_level not specified. Using 5 sequencing levels (default)."
            )

        sequencer = pln.prop_seq["sequencer"]

        # Choose and call the appropriate sequencing algorithm.
        if sequencer == "xia":
            logger.info("Calling xia leaf sequencing algorithm.")
            sequencer = XiaLeafSequencer(pln)
        elif sequencer == "engel":
            logger.info("Calling engel leaf sequencing algorithm.")
            sequencer = EngelLeafSequencer(pln)
        elif sequencer == "siochi":
            logger.info("Calling siochi leaf sequencing algorithm.")
            sequencer = SiochiLeafSequencer(pln)
        else:
            logger.error(f"Could not find specified sequencing algorithm '{sequencer}'.")

    # If sequencing or DAO is requested but radiation mode is not photons.
    elif (
        pln.prop_seq.get("run_sequencing", False) or pln.prop_opt.get("runDAO", False)
    ) and pln.radiationMode != "photons":
        logger.warning(
            'Sequencing is only specified for pln.radiationMode = "photons". Continuing without sequencing ...'
        )
    return sequencer


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

    result = sequencing_prob.sequence(result, _stf, _dij)

    return result
