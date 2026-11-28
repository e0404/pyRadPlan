"""Mapping between a delivery sequence, the beamlets and the variables of an optimization."""

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy import sparse

from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.result._beam_sequence import Segment
from pyRadPlan.result._sequence import Sequence

if TYPE_CHECKING:
    from pyRadPlan.stf import SteeringInformation


def _same_leaf_rows(mlc: Any, other: Any) -> bool:
    """Whether two MLCs have the same leaf rows: orientation, boundaries and leaf width."""
    if mlc.device_orientation != other.device_orientation:
        return False
    if not np.array_equal(mlc.leaf_position_boundaries, other.leaf_position_boundaries):
        return False
    if mlc.leaf_width is None or other.leaf_width is None:
        return mlc.leaf_width == other.leaf_width
    return bool(np.isclose(mlc.leaf_width, other.leaf_width))


class ApertureGrid(PyRadPlanBaseModel):
    """The beamlets of one beam of the steering information and the leaf rows over them.

    Positions are in mm on the axes of the MLC: ``travel`` along the leaves, ``stack``
    across the leaf rows.

    Attributes
    ----------
    stf_beam : int
        Index of the beam in the steering information.
    orientation : str
        Axis the leaves travel along, ``"X"`` or ``"Y"``.
    bixel_width : float
        Width of a beamlet.
    bixel_offset : int
        Index of the first beamlet of the beam among all beamlets (columns of the dose
        influence matrix).
    travel, stack : NDArray
        Position of the center of each beamlet, in ray order.
    row_boundaries : NDArray
        Lower edges of the leaf rows, ascending.
    leaf_width : float
        Width of a leaf row.
    lim_left, lim_right : NDArray
        Per leaf row the outer edges of the beamlets it covers, the range leaf positions
        are meaningful in. A row without beamlets has both at the lowest beamlet edge.
    """

    stf_beam: int
    orientation: Literal["X", "Y"] = "X"
    bixel_width: float
    bixel_offset: int
    travel: NDArray
    stack: NDArray
    row_boundaries: NDArray
    leaf_width: float
    lim_left: NDArray
    lim_right: NDArray

    @classmethod
    def from_beam(cls, beam: Any, stf_beam: int, bixel_offset: int, mlc: Any) -> "ApertureGrid":
        """Grid of a steering-information *beam* under the leaf rows of *mlc* (or none)."""
        orientation = "X" if mlc is None else mlc.device_orientation
        travel_axis, stack_axis = (0, 2) if orientation == "X" else (2, 0)
        positions = np.array([ray.ray_pos_bev for ray in beam.rays], dtype=float).reshape(-1, 3)
        bixel_width = float(beam.bixel_width)
        travel, stack = positions[:, travel_axis], positions[:, stack_axis]

        if mlc is None or mlc.leaf_position_boundaries is None:
            boundaries, leaf_width = np.zeros(0), bixel_width
        else:
            boundaries = np.asarray(mlc.leaf_position_boundaries, dtype=float)
            leaf_width = float(mlc.leaf_width)

        grid = cls(
            stf_beam=stf_beam,
            orientation=orientation,
            bixel_width=bixel_width,
            bixel_offset=bixel_offset,
            travel=travel,
            stack=stack,
            row_boundaries=boundaries,
            leaf_width=leaf_width,
            lim_left=np.zeros(len(boundaries)),
            lim_right=np.zeros(len(boundaries)),
        )
        covered = grid.row_coverage > 0
        lowest = travel.min() - 0.5 * bixel_width if travel.size else 0.0
        left = np.full(len(boundaries), lowest)
        right = np.full(len(boundaries), lowest)
        # Outer beamlet edges in each row (rows without beamlets stay closed)
        for row in np.flatnonzero(covered.any(axis=1)):
            left[row] = travel[covered[row]].min() - 0.5 * bixel_width
            right[row] = travel[covered[row]].max() + 0.5 * bixel_width
        grid.lim_left, grid.lim_right = left, right
        return grid

    @property
    def num_of_bixels(self) -> int:
        """Number of beamlets of the beam."""
        return int(self.travel.size)

    @property
    def num_of_rows(self) -> int:
        """Number of leaf rows."""
        return int(self.row_boundaries.size)

    @property
    def row_coverage(self) -> NDArray:
        """Fraction of each beamlet's extent across the rows a leaf row covers.

        ``(rows, beamlets)``, 1 for a beamlet that lies within the row.
        """
        return self.row_coverage_between(-np.inf, np.inf)

    def row_coverage_between(self, stack_min: float, stack_max: float) -> NDArray:
        """:attr:`row_coverage` within a jaw window ``[stack_min, stack_max]`` across the rows."""
        # Overlap of leaf row, jaw window and beamlet: highest lower to lowest upper edge
        low = np.maximum(
            np.maximum(self.row_boundaries[:, None], stack_min),
            self.stack[None, :] - 0.5 * self.bixel_width,
        )
        high = np.minimum(
            np.minimum(self.row_boundaries[:, None] + self.leaf_width, stack_max),
            self.stack[None, :] + 0.5 * self.bixel_width,
        )
        coverage = np.clip(high - low, 0.0, None) / self.bixel_width
        coverage[coverage < 1e-9] = 0.0
        return coverage

    def travel_coverage_between(self, left: NDArray, right: NDArray) -> NDArray:
        """Fraction of each beamlet's extent along travel between the edges of each row.

        The counterpart of :meth:`row_coverage_between` along the leaves, with one window per
        row: *left* and *right* hold the ``(rows,)`` edges of the opening, e.g. the leaf
        positions within the jaws. Returns ``(rows, beamlets)``; the product with the row
        coverage is the open fraction of each beamlet's area per row (see
        :meth:`open_fraction`).
        """
        low = np.maximum(left[:, None], self.travel[None, :] - 0.5 * self.bixel_width)
        high = np.minimum(right[:, None], self.travel[None, :] + 0.5 * self.bixel_width)
        return np.clip(high - low, 0.0, None) / self.bixel_width

    def open_fraction(
        self, leaf_positions: NDArray, window: tuple[float, float, float, float]
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Part of each beamlet's area each leaf row leaves open within a jaw window.

        Parameters
        ----------
        leaf_positions : NDArray
            ``(rows, 2)`` left and right leaf of every row.
        window : tuple
            ``(travel_min, travel_max, stack_min, stack_max)`` of the jaws.

        Returns
        -------
        tuple
            ``(edges, row_coverage, open_fraction)``: the ``(rows, 2)`` leaf positions cut to
            the jaws along travel, the row coverage within the jaws and the open fraction,
            both ``(rows, beamlets)``.
        """
        travel_min, travel_max, stack_min, stack_max = window
        edges = np.column_stack(
            (
                np.maximum(leaf_positions[:, 0], travel_min),
                np.minimum(leaf_positions[:, 1], travel_max),
            )
        )
        row_coverage = self.row_coverage_between(stack_min, stack_max)
        travel = self.travel_coverage_between(edges[:, 0], edges[:, 1])
        return edges, row_coverage, row_coverage * travel

    def bixel_index_map(self) -> tuple[NDArray, float, float]:
        """Place the beamlets on their dense grid.

        Returns
        -------
        tuple
            ``(index, travel_min, stack_min)``: ``index`` holds per ``(stack, travel)`` grid
            position the index of the beamlet among all beamlets, ``-1`` where the beam has
            none; the two positions are those of the first grid column and row.
        """
        travel_min, stack_min = float(self.travel.min()), float(self.stack.min())
        cols = np.rint((self.travel - travel_min) / self.bixel_width).astype(int)
        rows = np.rint((self.stack - stack_min) / self.bixel_width).astype(int)
        index = np.full((rows.max() + 1, cols.max() + 1), -1, dtype=int)
        index[rows, cols] = self.bixel_offset + np.arange(self.num_of_bixels)
        return index, travel_min, stack_min


class ApertureInfo(PyRadPlanBaseModel):
    """A delivery sequence together with the beamlets its apertures open.

    This is the bridge between the three views of a deliverable plan: the
    :class:`Sequence` the machine delivers, the beamlet weights the dose calculation
    works on, and the variable vector of an optimization over apertures (weights and leaf
    positions). It is built from a sequence and the steering information and is not part
    of a result.

    The variable vector is ``[weights | left leaf positions | right leaf positions]``, the
    apertures in the order of the beams and their segments, the leaves in the order of the
    leaf rows. The jaws of a segment are fixed: they cut away what lies outside their
    window, and a leaf behind a jaw does not change the beamlet weights.

    Attributes
    ----------
    sequence : Sequence
        The delivery sequence.
    grids : list[ApertureGrid]
        One grid per beam of the steering information.
    """

    sequence: Sequence
    grids: list[ApertureGrid] = Field(default_factory=list)

    @classmethod
    def from_sequence(cls, sequence: Sequence, stf: "SteeringInformation") -> "ApertureInfo":
        """Relate the segments of *sequence* to the beamlets of *stf*.

        A segment without ``stf_beam`` belongs to the beam at the position of its
        ``BeamSequence``, which requires as many beam sequences as beams.

        Raises
        ------
        NotImplementedError
            For segments that move while they are delivered, or whose MLC or jaws are
            rotated by a collimator angle.
        ValueError
            For segments without an MLC, segments of one beam with different leaf rows, or
            an assignment to beams the steering information does not have.
        """
        stf_beams = list(stf.beams)
        aligned = len(sequence.beams) == len(stf_beams)
        beams = []
        for beam_ix, beam in enumerate(sequence.beams):
            segments = []
            for segment in beam.segments:
                if segment.is_dynamic:
                    raise NotImplementedError(
                        "Segments that move while they are delivered cannot be mapped to "
                        "beamlets yet."
                    )
                devices = [segment.start.mlc, *segment.start.jaws]
                if any(
                    device is not None and (device.device_angle or 0.0) % 360.0 != 0.0
                    for device in devices
                ):
                    raise NotImplementedError(
                        "Segments with a collimator angle cannot be mapped to beamlets yet."
                    )
                stf_beam = segment.stf_beam
                if stf_beam is None:
                    if not aligned:
                        raise ValueError(
                            "Segments without stf_beam need one beam sequence per beam of the "
                            "steering information."
                        )
                    stf_beam = beam_ix
                if stf_beam >= len(stf_beams):
                    raise ValueError(f"The steering information has no beam {stf_beam}.")
                segments.append(segment.model_copy(update={"stf_beam": stf_beam}))
            beams.append(beam.model_copy(update={"segments": segments}))
        sequence = sequence.model_copy(update={"beams": beams})

        mlcs: dict[int, Any] = {}
        for beam in sequence.beams:
            for segment in beam.segments:
                mlc = segment.start.mlc
                if mlc is None:
                    raise ValueError("Every segment needs an MLC to be mapped to beamlets.")
                # The grid of a beam follows the first MLC on it, all others have to match
                known = mlcs.setdefault(segment.stf_beam, mlc)
                if not _same_leaf_rows(known, mlc):
                    raise ValueError(
                        f"The segments on beam {segment.stf_beam} do not share their leaf rows."
                    )

        grids, offset = [], 0
        for stf_beam, beam in enumerate(stf_beams):
            grids.append(ApertureGrid.from_beam(beam, stf_beam, offset, mlcs.get(stf_beam)))
            offset += len(beam.rays)
        return cls(sequence=sequence, grids=grids)

    @property
    def apertures(self) -> list[tuple[int, int, Segment]]:
        """Every aperture as ``(beam sequence index, segment index, segment)``."""
        return [
            (beam_ix, segment_ix, segment)
            for beam_ix, beam in enumerate(self.sequence.beams)
            for segment_ix, segment in enumerate(beam.segments)
        ]

    @property
    def total_num_of_bixels(self) -> int:
        """Number of beamlets of all beams."""
        return sum(grid.num_of_bixels for grid in self.grids)

    @property
    def total_num_of_apertures(self) -> int:
        """Number of apertures (segments) of all beams."""
        return len(self.apertures)

    @property
    def total_num_of_leaf_pairs(self) -> int:
        """Number of leaf pairs over all apertures."""
        return sum(self.grids[s.stf_beam].num_of_rows for _, _, s in self.apertures)

    def _leaf_offsets(self) -> NDArray:
        """Start of each aperture's leaf pairs among all leaf pairs, ``(apertures + 1,)``.

        Aperture ``i`` holds leaf pairs ``offsets[i]:offsets[i + 1]``. The last entry is the
        total number of leaf pairs.
        """
        rows = [self.grids[s.stf_beam].num_of_rows for _, _, s in self.apertures]
        return np.concatenate(([0], np.cumsum(rows))).astype(int)

    def to_vector(self) -> NDArray:
        """Return the variables ``[weights | left leaf positions | right leaf positions]``."""
        apertures = self.apertures
        weights = [segment.weight for _, _, segment in apertures]
        leaves = [np.asarray(s.start.mlc.leaf_positions, dtype=float) for _, _, s in apertures]
        left = [positions[:, 0] for positions in leaves]
        right = [positions[:, 1] for positions in leaves]
        return np.concatenate([np.asarray(weights, dtype=float), *left, *right])

    def bounds(self) -> tuple[NDArray, NDArray]:
        """Lower and upper bound of every variable.

        Weights are not negative; leaf positions stay within the beamlets of their row.
        """
        apertures = self.apertures
        num = len(apertures)
        lim_left = [self.grids[s.stf_beam].lim_left for _, _, s in apertures]
        lim_right = [self.grids[s.stf_beam].lim_right for _, _, s in apertures]
        lower = np.concatenate([np.zeros(num), *lim_left, *lim_left])
        upper = np.concatenate([np.full(num, np.inf), *lim_right, *lim_right])
        return lower, upper

    def update(self, vector: Any) -> "ApertureInfo":
        """Return the aperture info with weights and leaf positions of a variable vector.

        Negative weights, which a solver may return within its bound tolerance, are stored
        as 0: a segment cannot deliver a negative weight.
        """
        vector = np.asarray(vector, dtype=float).reshape(-1)
        apertures = self.apertures
        num, offsets = len(apertures), self._leaf_offsets()
        total = int(offsets[-1])
        if vector.size != num + 2 * total:
            raise ValueError(f"Expected {num + 2 * total} variables, got {vector.size}.")

        beams = [list(beam.segments) for beam in self.sequence.beams]
        for ix, (beam_ix, segment_ix, segment) in enumerate(apertures):
            # The segment's left leaves follow the weights and the leaves of the segments
            # before it; the right leaves follow all left leaves
            rows = slice(num + offsets[ix], num + offsets[ix + 1])
            left, right = vector[rows], vector[rows.start + total : rows.stop + total]
            mlc = segment.start.mlc.model_copy(
                update={"leaf_positions": np.column_stack((left, right))}
            )
            state = segment.start.model_copy(update={"mlc": mlc})
            beams[beam_ix][segment_ix] = segment.model_copy(
                update={"weight": max(float(vector[ix]), 0.0), "start": state, "end": None}
            )
        sequence = self.sequence.model_copy(
            update={
                "beams": [
                    beam.model_copy(update={"segments": segments})
                    for beam, segments in zip(self.sequence.beams, beams)
                ]
            }
        )
        return self.model_copy(update={"sequence": sequence})

    @staticmethod
    def _jaw_window(segment: Segment, orientation: str) -> tuple[float, float, float, float]:
        """Return the jaw window of a segment as ``(travel_min, travel_max, stack_min, stack_max)``.

        Several jaws on one axis open their intersection.
        """
        window = [-np.inf, np.inf, -np.inf, np.inf]
        for jaw in segment.start.jaws:
            low, high = (float(p) for p in jaw.positions)
            ix = 0 if jaw.device_orientation == orientation else 2
            window[ix], window[ix + 1] = max(window[ix], low), min(window[ix + 1], high)
        return tuple(window)

    def _coverage(self, segment: Segment) -> tuple[ApertureGrid, NDArray]:
        """Grid of a segment's beam and the per-row open fraction of its beamlets."""
        grid = self.grids[segment.stf_beam]
        positions = np.asarray(segment.start.mlc.leaf_positions, dtype=float)
        window = self._jaw_window(segment, grid.orientation)
        return grid, grid.open_fraction(positions, window)[2]

    def aperture_fluence(self, beam_ix: int, segment_ix: int) -> NDArray:
        """Fraction of each beamlet's area the segment leaves open within its jaws (0 to 1).

        One value per beamlet of the segment's beam, in ray order, without the weight of the
        segment.
        """
        segment = self.sequence.beams[beam_ix].segments[segment_ix]
        return self._coverage(segment)[1].sum(axis=0)

    def bixel_weights(self) -> NDArray:
        """Weight every beamlet receives from all apertures.

        A beamlet counts by the fraction of its area an aperture leaves open within the
        jaws, times the aperture's weight, summed over all apertures, so leaf positions
        between beamlet edges and leaf rows wider than the beamlets are handled.
        """
        weights = np.zeros(self.total_num_of_bixels)
        for _, _, segment in self.apertures:
            grid, coverage = self._coverage(segment)
            bixels = slice(grid.bixel_offset, grid.bixel_offset + grid.num_of_bixels)
            weights[bixels] += segment.weight * coverage.sum(axis=0)
        return weights

    def bixel_jacobian(self) -> sparse.csr_matrix:
        """Derive the beamlet weights with respect to the variables.

        A beamlet's weight is the sum over segments of segment weight × open fraction, so
        its derivative by a segment weight is the beamlet's open fraction in that segment.
        Its derivative by a leaf position is segment weight × row coverage / bixel width
        while the leaf tip lies inside the beamlet and moving it changes the opening:
        negative for a left leaf, which closes the beamlet as its position increases,
        positive for a right leaf. It is 0 for a leaf behind a jaw and for a leaf past the
        other leaf of its row. Where both leaves of a row meet, the derivative is the one of
        opening the row, so that a gradient can reopen a closed leaf pair.

        Returns
        -------
        scipy.sparse.csr_matrix
            ``(variables, beamlets)``: row ``i`` holds the change of each beamlet weight per
            unit change of variable ``i`` of :meth:`to_vector`.
        """
        apertures = self.apertures
        num, offsets = len(apertures), self._leaf_offsets()
        total = int(offsets[-1])
        rows, cols, values = [], [], []
        for ix, (_, _, segment) in enumerate(apertures):
            grid = self.grids[segment.stf_beam]
            positions = np.asarray(segment.start.mlc.leaf_positions, dtype=float)
            window = self._jaw_window(segment, grid.orientation)
            edges, row_coverage, coverage = grid.open_fraction(positions, window)
            travel_min, travel_max = window[:2]
            offset = grid.bixel_offset

            per_bixel = coverage.sum(axis=0)
            hit = np.flatnonzero(per_bixel)
            rows.append(np.full(hit.size, ix))
            cols.append(offset + hit)
            values.append(per_bixel[hit])

            half = 0.5 * grid.bixel_width
            scale = segment.weight * row_coverage / grid.bixel_width
            left, right = positions[:, :1], positions[:, 1:]
            travel = grid.travel[None, :]
            for inside, sign, start in (
                (
                    (left > travel_min)
                    & (left > travel - half)
                    & (left < travel + half)
                    & (left <= edges[:, 1:]),
                    -1.0,
                    num,
                ),
                (
                    (right < travel_max)
                    & (right > travel - half)
                    & (right < travel + half)
                    & (right >= edges[:, :1]),
                    1.0,
                    num + total,
                ),
            ):
                row, bixel = np.nonzero(inside & (row_coverage > 0))
                rows.append(start + offsets[ix] + row)
                cols.append(offset + bixel)
                values.append(sign * scale[row, bixel])

        shape = (num + 2 * total, self.total_num_of_bixels)
        if not rows:  # no apertures: np.concatenate can't take an empty list
            return sparse.csr_matrix(shape)
        return sparse.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=shape
        )

    def to_sequence(self) -> Sequence:
        """Return the delivery sequence with the beamlet weights it delivers as fluence."""
        return self.sequence.model_copy(update={"fluence": self.bixel_weights()})
