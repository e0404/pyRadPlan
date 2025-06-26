"""Provides functionality for handling sequencing."""

from ._base import Sequencer
from ._base_photon import PhotonSequencer
from ._engel_leaf_sequencing import EngelLeafSequencer
from ._siochi_leaf_sequencing import SiochiLeafSequencer
from ._xia_leaf_sequencing import XiaLeafSequencer
from ._factory import get_available_sequencers, get_sequencer_from_pln, register_sequencer

# Register sequencers in alphabetical order
register_sequencer(SiochiLeafSequencer)
register_sequencer(XiaLeafSequencer)
# TODO: register EngelLeafSequencer once implemented
# register_sequencer(EngelLeafSequencer)

__all__ = [
    "EngelLeafSequencer",
    "PhotonSequencer",
    "Sequencer",
    "SiochiLeafSequencer",
    "XiaLeafSequencer",
    "get_available_sequencers",
    "get_sequencer_from_pln",
    "register_sequencer",
]
