"""A simple Qt window to visualize a plan's delivery sequence."""

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from pyRadPlan.result import Sequence
from pyRadPlan.stf import SteeringInformation, validate_stf
from pyRadPlan.gui.widgets.result.aperture_widget import (
    ApertureViewerWidget,
    SequenceViewContext,
)


class SequenceWindow(QMainWindow):
    """A simple main window hosting a delivery-sequence viewer as central widget.

    Currently always an :class:`ApertureViewerWidget` (MLC shapes), since
    pyRadPlan only implements sequencing for photon plans. Other radiation
    modalities would get their own delivery-specific viewer here once implemented
    (e.g. a spot-map widget for ion sequencing).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("pyRadPlan Sequence Viewer")
        self.viewer = ApertureViewerWidget(self)
        self.setCentralWidget(self.viewer)
        self.resize(900, 700)


def show_sequence_viewer(
    sequence: Sequence,
    stf: Optional[SteeringInformation] = None,
    initial_beam_ix: Optional[int] = None,
    context: Optional[SequenceViewContext] = None,
) -> Optional[SequenceWindow]:
    """Create and show a non-modal :class:`SequenceWindow` for *sequence*.

    Parameters
    ----------
    sequence : Sequence
        A sequenced plan's delivery sequence.
    stf : SteeringInformation, optional
        Steering information used to label each beam's tab with its gantry
        angle, if available.
    initial_beam_ix : int, optional
        Beam tab to select initially (e.g. the beam currently shown in the
        dose viewer).

    Returns
    -------
    SequenceWindow or None
        The created window, or ``None`` if there is no sequencing data.
    """
    if not isinstance(sequence, Sequence) or not sequence.beam:
        return None

    if stf is not None:
        stf = validate_stf(stf)

    win = SequenceWindow()
    win.viewer.set_sequence(sequence, stf=stf, context=context)
    if initial_beam_ix is not None:
        win.viewer.set_current_beam(initial_beam_ix)

    win.show()
    return win


def _launch_sequence_window(sequence: Sequence, stf: Optional[SteeringInformation] = None) -> None:
    """Launch the Qt sequence viewer for a sequenced plan's ``Sequence``.

    Parameters
    ----------
    sequence : Sequence
        A sequenced plan's delivery sequence (``run_sequencing()``'s
        ``result.sequence``, or the return value of ``run_sequencing()``
        applied to a bare ``Sequence``).
    stf : SteeringInformation, optional
        Steering information used to label each beam's tab with its gantry
        angle, if available.
    """
    if not isinstance(sequence, Sequence) or not sequence.beam:
        raise ValueError("No sequencing data to visualize.")

    app = QApplication.instance() or QApplication(sys.argv)

    if stf is not None:
        stf = validate_stf(stf)

    win = SequenceWindow()
    win.viewer.set_sequence(sequence, stf=stf)

    win.show()
    app.exec()
