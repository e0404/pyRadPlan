"""The delivery sequence of one beam: segments between machine states."""

from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, computed_field, field_validator, model_validator

from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.machines import MLC, Jaw

_FULL_CIRCLE = 360.0


def _angle_span(start: float, end: float, rotation: Optional[str]) -> float:
    """Signed angle travelled from *start* to *end*, positive clockwise (ascending angles)."""
    span = float(end) - float(start)
    if rotation == "cw":
        return span % _FULL_CIRCLE
    if rotation == "ccw":
        return -((-span) % _FULL_CIRCLE)
    # No direction given: shorter way, in [-180, 180)
    return (span + _FULL_CIRCLE / 2) % _FULL_CIRCLE - _FULL_CIRCLE / 2


class MachineState(PyRadPlanBaseModel):
    """The machine at one instant: where the gantry, the couch, the leaves and the jaws are.

    A state is a snapshot without any speed. Everything that moves is described by the
    :class:`Segment` between two states.

    Attributes
    ----------
    gantry_angle : float, optional
        Gantry angle in degrees. ``None`` stands for the angle of the beam the segment is
        planned on.
    couch_angle : float, optional
        Couch angle in degrees. ``None`` stands for the angle of the beam.
    mlc : MLC, optional
        The multi-leaf collimator with its leaf positions. Its ``device_angle`` is the
        collimator angle.
    jaws : list[Jaw]
        The jaws, at most one per axis.
    """

    gantry_angle: Optional[float] = None
    couch_angle: Optional[float] = None
    mlc: Optional[MLC] = None
    jaws: list[Jaw] = Field(default_factory=list)

    @field_validator("jaws", mode="before")
    @classmethod
    def _jaw_list(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, (Jaw, dict)):
            return [v]
        return list(v)

    def interpolate(
        self, other: "MachineState", fraction: float, rotation: Optional[str] = None
    ) -> "MachineState":
        """State a *fraction* (0 to 1) of the way to *other*, all axes moving linearly.

        The gantry turns in the direction *rotation* (``"cw"``, ``"ccw"``, or ``None`` for the
        shorter way), the couch always the shorter way; both angles stay within [0, 360).
        """

        def lerp(a: Any, b: Any) -> Any:
            if a is None or b is None:
                return a
            return a + fraction * (np.asarray(b) - np.asarray(a))

        def turn(a: Optional[float], b: Optional[float], direction: Optional[str]) -> Any:
            if a is None or b is None:
                return a
            return (a + fraction * _angle_span(a, b, direction)) % _FULL_CIRCLE

        mlc = self.mlc
        if mlc is not None and other.mlc is not None:
            mlc = mlc.model_copy(
                update={"leaf_positions": lerp(mlc.leaf_positions, other.mlc.leaf_positions)}
            )
        jaws = list(self.jaws)
        if len(jaws) == len(other.jaws):
            jaws = [
                jaw.model_copy(
                    update={"positions": [float(p) for p in lerp(jaw.positions, to.positions)]}
                )
                for jaw, to in zip(jaws, other.jaws)
            ]
        return MachineState(
            gantry_angle=turn(self.gantry_angle, other.gantry_angle, rotation),
            couch_angle=turn(self.couch_angle, other.couch_angle, None),
            mlc=mlc,
            jaws=jaws,
        )


class Segment(PyRadPlanBaseModel):
    """A part of a beam that delivers ``weight`` while the machine is in or between states.

    With only ``start``, nothing moves while the beam is on (step and shoot). With ``end``,
    every axis moves linearly from ``start`` to ``end`` while the weight is delivered at a
    constant rate (sliding window, arc).

    Attributes
    ----------
    weight : float
        Fluence weight delivered by the segment, in the unit of the bixel weights.
    start : MachineState
        The machine when the segment starts.
    end : MachineState, optional
        The machine when the segment ends. ``None``: the machine stays at ``start``.
    duration : float, optional
        Time the segment takes in seconds. Leaf speed, gantry speed and dose rate follow
        from it; without it only the travelled distances are known.
    stf_beam : int, optional
        Index of the beam of the steering information the segment is planned on, whose
        beamlets its fluence refers to.
    """

    weight: float = Field(ge=0.0)
    start: MachineState
    end: Optional[MachineState] = None
    duration: Optional[float] = Field(default=None, gt=0.0)
    stf_beam: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _matching_leaf_rows(self) -> "Segment":
        if self.end is None or self.start.mlc is None or self.end.mlc is None:
            return self
        start, end = self.start.mlc.leaf_positions, self.end.mlc.leaf_positions
        if start is not None and end is not None and np.shape(start) != np.shape(end):
            raise ValueError("The start and the end of a segment need the same leaf rows.")
        return self

    @property
    def is_dynamic(self) -> bool:
        """Whether the machine moves while the segment is delivered."""
        return self.end is not None and self.end != self.start

    @property
    def leaf_travel(self) -> Optional[NDArray]:
        """Distance each leaf travels during the segment in mm, ``(rows, 2)``, signed."""
        if self.start.mlc is None or self.start.mlc.leaf_positions is None:
            return None
        start = np.asarray(self.start.mlc.leaf_positions, dtype=float)
        if self.end is None or self.end.mlc is None:
            return np.zeros_like(start)
        return np.asarray(self.end.mlc.leaf_positions, dtype=float) - start

    def gantry_travel(self, rotation: Optional[str] = None) -> float:
        """Angle the gantry travels during the segment in degrees, positive clockwise."""
        if self.end is None or self.start.gantry_angle is None or self.end.gantry_angle is None:
            return 0.0
        return _angle_span(self.start.gantry_angle, self.end.gantry_angle, rotation)

    @property
    def max_leaf_speed(self) -> Optional[float]:
        """Speed of the fastest leaf in mm/s, ``None`` without a duration."""
        travel = self.leaf_travel
        if self.duration is None or travel is None:
            return None
        return float(np.max(np.abs(travel), initial=0.0)) / self.duration

    def gantry_speed(self, rotation: Optional[str] = None) -> Optional[float]:
        """Gantry speed in degrees/s, ``None`` without a duration."""
        if self.duration is None:
            return None
        return abs(self.gantry_travel(rotation)) / self.duration

    @property
    def weight_rate(self) -> Optional[float]:
        """Weight delivered per second (the dose rate in weight units), ``None`` without a duration."""
        return None if self.duration is None else self.weight / self.duration

    def state_at(self, fraction: float, rotation: Optional[str] = None) -> MachineState:
        """Return the machine a *fraction* (0 to 1) of the way through the segment.

        *rotation* is the direction the gantry turns in, see :meth:`MachineState.interpolate`.
        """
        if self.end is None:
            return self.start
        return self.start.interpolate(self.end, fraction, rotation)


class ControlPoint(PyRadPlanBaseModel):
    """A machine state with the weight delivered up to it, as in a DICOM RT Plan.

    Attributes
    ----------
    index : int
        Position in the control point list, from 0.
    cumulative_weight : float
        Weight delivered when the machine reaches this control point.
    cumulative_meterset_weight : float
        ``cumulative_weight`` as a fraction of the weight of the beam (0 to 1).
    gantry_angle, couch_angle : float, optional
        Angles in degrees.
    gantry_rotation_direction : str
        Direction the gantry moves in towards the next control point: ``"CW"``, ``"CC"`` or
        ``"NONE"``.
    mlc : MLC, optional
        The multi-leaf collimator with its leaf positions.
    jaws : list[Jaw]
        The jaws.
    """

    index: int
    cumulative_weight: float
    cumulative_meterset_weight: float
    gantry_angle: Optional[float] = None
    couch_angle: Optional[float] = None
    gantry_rotation_direction: Literal["CW", "CC", "NONE"] = "NONE"
    mlc: Optional[MLC] = None
    jaws: list[Jaw] = Field(default_factory=list)


class BeamSequence(PyRadPlanBaseModel):
    """What the machine does to deliver one beam: a static field or one arc.

    All positions are in mm in the beam's eye view frame of the ray matrix (LPS +x and +z at
    gantry and couch angle 0, see :class:`~pyRadPlan.machines.BeamLimitingDevice`).

    Attributes
    ----------
    segments : list[Segment]
        The segments in the order they are delivered.
    rotation : str, optional
        Direction the gantry rotates in during an arc, ``"cw"`` (ascending angles) or
        ``"ccw"``. ``None`` takes the shorter way between two gantry angles.
    """

    segments: list[Segment] = Field(default_factory=list)
    rotation: Optional[Literal["cw", "ccw"]] = None

    @classmethod
    def from_shapes(
        cls,
        mlcs: Optional[list[MLC]],
        weights: Any,
        jaws: Optional[list] = None,
        stf_beam: Optional[int] = None,
    ) -> "BeamSequence":
        """Step-and-shoot sequence from one MLC and one weight per segment.

        Parameters
        ----------
        mlcs : list[MLC], optional
            The MLC of each segment; ``None`` for a field shaped by the jaws only.
        weights : array-like
            The weight of each segment.
        jaws : list, optional
            The jaws of the beam (a list of ``Jaw``), or one such list per segment. A jaw
            also bounds the field across its axis at ``±field_width / 2``, so a field with a
            single jaw needs a ``field_width`` that covers the field.
        stf_beam : int, optional
            Index of the beam of the steering information the segments are planned on.
        """
        weights = np.asarray(weights, dtype=float).reshape(-1)
        num = len(weights)
        if mlcs is not None and len(mlcs) != num:
            raise ValueError(f"Got {len(mlcs)} MLC shapes for {num} weights.")
        jaws = list(jaws or [])
        per_segment = bool(jaws) and all(isinstance(j, (list, tuple)) for j in jaws)
        if per_segment and len(jaws) != num:
            raise ValueError(f"Got jaws for {len(jaws)} segments, expected {num}.")
        return cls(
            segments=[
                Segment(
                    weight=weights[m],
                    start=MachineState(
                        mlc=None if mlcs is None else mlcs[m],
                        jaws=jaws[m] if per_segment else jaws,
                    ),
                    stf_beam=stf_beam,
                )
                for m in range(num)
            ]
        )

    @computed_field
    @property
    def num_of_segments(self) -> int:
        """Number of segments of the beam."""
        return len(self.segments)

    @property
    def weights(self) -> NDArray:
        """Weight of each segment."""
        return np.array([segment.weight for segment in self.segments], dtype=float)

    @property
    def total_weight(self) -> float:
        """Weight the beam delivers in total."""
        return float(self.weights.sum())

    @property
    def is_dynamic(self) -> bool:
        """Whether the machine moves while any segment is delivered."""
        return any(segment.is_dynamic for segment in self.segments)

    @property
    def is_continuous(self) -> bool:
        """Whether every segment starts in the state the one before ended in."""
        return all(
            (before.end or before.start) == after.start
            for before, after in zip(self.segments, self.segments[1:])
        )

    @property
    def stf_beams(self) -> list[int]:
        """Indices of the beams of the steering information the segments are planned on."""
        return sorted({s.stf_beam for s in self.segments if s.stf_beam is not None})

    @property
    def mlc(self) -> Optional[MLC]:
        """The MLC of the first segment, for the geometry all segments share."""
        return self.segments[0].start.mlc if self.segments else None

    def control_points(
        self, gantry_angle: Optional[float] = None, couch_angle: Optional[float] = None
    ) -> list[ControlPoint]:
        """List the sequence as control points with cumulative weights, as in a DICOM RT Plan.

        A segment contributes the state it starts in at the weight delivered before it, and
        the state it ends in at the weight delivered after it. A control point equal to the
        one before it (same state, same cumulative weight) is left out: a continuous arc of
        ``n`` segments gives ``n + 1`` control points, ``n`` step-and-shoot segments of
        different shapes and positive weights give ``2 n``, and a segment without weight or
        a repeated shape adds fewer. Between two control points, rising weight means the
        beam is on, unchanged weight that the machine moves with the beam off.

        Parameters
        ----------
        gantry_angle, couch_angle : float, optional
            Angles of the beam, used where a state does not give its own.
        """
        total = self.total_weight
        points: list[dict] = []

        def add(state: MachineState, cumulative: float) -> None:
            point = {
                "cumulative_weight": cumulative,
                "cumulative_meterset_weight": cumulative / total if total > 0 else 0.0,
                "gantry_angle": gantry_angle if state.gantry_angle is None else state.gantry_angle,
                "couch_angle": couch_angle if state.couch_angle is None else state.couch_angle,
                "mlc": state.mlc,
                "jaws": state.jaws,
            }
            if points and all(points[-1][key] == value for key, value in point.items()):
                return
            points.append(point)

        # Add start and end state of each segment (step and shoot: start twice) and
        # track the cumulative weight
        cumulative = 0.0
        for segment in self.segments:
            add(segment.start, cumulative)
            cumulative += segment.weight
            add(segment.end or segment.start, cumulative)

        for point, after in zip(points, points[1:]):
            if point["gantry_angle"] is None or after["gantry_angle"] is None:
                continue
            span = _angle_span(point["gantry_angle"], after["gantry_angle"], self.rotation)
            if span != 0:
                point["gantry_rotation_direction"] = "CW" if span > 0 else "CC"
        return [ControlPoint(index=ix, **point) for ix, point in enumerate(points)]
