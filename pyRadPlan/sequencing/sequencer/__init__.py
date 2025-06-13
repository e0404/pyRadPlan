"""Provides functionality for handling sequencing."""

from ._base import Sequencer
from ._base_photon import PhotonSequencer
from ._engel_leaf_sequencing import EngelLeafSequencer
from ._factory import get_sequencer_from_pln, get_available_sequencers, register_sequencer

from ._siochi_leaf_sequencing import SiochiLeafSequencer
from ._xia_leaf_sequencing import XiaLeafSequencer

register_sequencer(XiaLeafSequencer)
register_sequencer(SiochiLeafSequencer)
# TODO: Update the following as soon as EngelLeafSequencer is implemented
# register_sequencer(EngelLeafSequencer)

__all__ = [
    "Sequencer",
    "PhotonSequencer",
    "EngelLeafSequencer",
    "SiochiLeafSequencer",
    "XiaLeafSequencer",
    "get_sequencer_from_pln",
    "get_available_sequencers",
    "register_sequencer",
]
