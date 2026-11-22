import logging

import numpy as np
from typing import Any, Optional

from ._externalbeam import StfGeneratorExternalBeamRayBixel, StfGeneratorSingleBixel
from .._beamlet import FieldShape, FieldShapeAsMask, FieldShapeAsBLD, FieldShapeComposite
from pyRadPlan.machines import PhotonLINAC
from pyRadPlan.result._beam_sequence import BeamSequence

logger = logging.getLogger(__name__)


class StfGeneratorPhotonIMRT(StfGeneratorExternalBeamRayBixel):
    """Class representing a Photon IMRT Geometry Stf Generator.

    Attributes
    ----------
    name : str
        The name of the generator ("Photon IMRT Geometry").
    short_name : str
        The short name of the generator ("photonIMRT").
    possible_radiation_modes : list[str]
        A list of possible radiation modes (["photons"]).
    """

    name = "Photon IMRT Geometry"
    short_name = "photonIMRT"
    possible_radiation_modes = ["photons"]

    energy: float = 6.0

    def __init__(self, pln=None):
        self.radiation_mode = "photons"  # Radiation mode must be photons
        super().__init__(pln)

    def _initialize(self):
        super()._initialize()

        if not isinstance(self.machine, PhotonLINAC):
            raise ValueError("Machine must be an instance of PhotonLINAC.")

        energy_ix = self.machine.get_closest_energy_index(self.energy)

        machine_energy = self.machine.energies[energy_ix]

        if machine_energy != self.energy:
            logger.warning(
                "Selected energy not available in machine. Using closest available energy %g.",
                machine_energy,
            )
            self.energy = machine_energy

    def _generate_source_geometry(self):
        """Generate the source geometry for the photon IMRT geometry."""
        stf = super()._generate_source_geometry()
        return stf

    def _create_rays(self, beam: dict) -> list[dict]:
        """Create the rays for the photon IMRT geometry.

        Parameters
        ----------
        beam : dict
            The beam dictionary.

        Returns
        -------
        list[dict]
            A list of ray dictionaries
        """
        rays = super()._create_rays(beam)

        beamlet_template = {"energy": self.energy}
        rays = [
            {
                **ray,
                "beamlets": [beamlet_template.copy()],
                "target_point": 2 * ray["ray_pos"] - beam["source_point"],
                "target_point_bev": 2 * ray["ray_pos_bev"] - beam["source_point_bev"],
            }
            for ray in rays
        ]

        return rays


class StfGeneratorPhotonCollimatedSquareFields(StfGeneratorPhotonIMRT):
    """Class representing a Photon Collimated Square Fields Stf Generator.

    Attributes
    ----------
    name : str
        The name of the generator ("Photon Collimated Square Fields").
    short_name : str
        The short name of the generator ("photonCollimatedSquareFields").
    possible_radiation_modes : list[str]
        A list of possible radiation modes (["photons"]).
    """

    name = "Photon Open Fields Geometry"
    short_name = "photonOpenFields"
    possible_radiation_modes = ["photons"]

    # Alias for field_width
    @property
    def field_width(self) -> float:
        """Alias for the bixel_width property."""
        return self.bixel_width

    @field_width.setter
    def field_width(self, value: float):
        self.bixel_width = value

    def __init__(self, pln=None):
        self.radiation_mode = "photons"
        super().__init__(pln)

    def _generate_ray_positions_in_isocenter_plane(self, beam):
        """Generate the ray positions in the isocenter plane.

        As we have a square field, this is a single ray to the isocenter.

        Parameters
        ----------
        beam : dict
            The beam dictionary. Ignored in this case

        Returns
        -------
        np.ndarray
            A numpy array with the single ray position of (0, 0, 0) in the isocenter.
        """

        return np.zeros((3, 1), dtype=float)

    def _generate_source_geometry(self):
        """Generate the source geometry for photon collimated square fields."""
        stf = super()._generate_source_geometry()

        for i, field in enumerate(stf):
            field["field_width"] = self.field_width

        return stf


# TODO: Remove duplication with StfGeneratorPhotonIMRT
class StfGeneratorPhotonSingleBixel(StfGeneratorSingleBixel):
    """Class representing a Photon IMRT Geometry Stf Generator.

    Attributes
    ----------
    name : str
        The name of the generator ("Photon Single Bixel Geometry").
    short_name : str
        The short name of the generator ("photonSingleBixel").
    possible_radiation_modes : list[str]
        A list of possible radiation modes (["photons"]).
    beam_sequences : list[BeamSequence]
        The delivery sequence of each beam (e.g. a static field: one segment of weight
        1.0). A segment whose leaves or jaws move is sampled at ``DYNAMIC_SAMPLES``
        intermediate states; gantry or couch motion within a segment is not supported.
        If not given, each beam gets a single ray with no field shape.
    """

    name = "Photon Single Bixel Geometry"
    short_name = "photonSingleBixel"
    possible_radiation_modes = ["photons"]

    energy: float = 6.0

    #: Number of intermediate states a moving segment is sampled at.
    DYNAMIC_SAMPLES: int = 5

    beam_sequences: Optional[list[BeamSequence]]

    def __init__(self, pln=None):
        self.radiation_mode = "photons"  # Radiation mode must be photons
        self.beam_sequences = None
        super().__init__(pln)

    def _initialize(self):
        super()._initialize()

        if not isinstance(self.machine, PhotonLINAC):
            raise ValueError("Machine must be an instance of PhotonLINAC.")

        energy_ix = self.machine.get_closest_energy_index(self.energy)

        machine_energy = self.machine.energies[energy_ix]

        if machine_energy != self.energy:
            logger.warning(
                "Selected energy not available in machine. Using closest available energy %g.",
                machine_energy,
            )
            self.energy = machine_energy

        if self.beam_sequences is not None:
            if not hasattr(self, "resolution"):
                raise ValueError(
                    "Resolution must be given to calculate transmission masks from beam "
                    "limiting devices"
                )
            if len(self.beam_sequences) != self.num_of_beams:
                raise ValueError(
                    f"beam_sequences has {len(self.beam_sequences)} entries, expected one per "
                    f"beam ({self.num_of_beams})."
                )

    def _generate_source_geometry(self):
        """Generate the source geometry for the photon IMRT geometry."""
        stf = super()._generate_source_geometry()
        return stf

    def _segment_states(self, beam_sequence: BeamSequence) -> list[tuple[float, Any, list]]:
        """List the fields a beam sequence delivers as ``(weight, mlc, jaws)``.

        A static segment is one field; a segment with moving leaves or jaws is split into
        ``DYNAMIC_SAMPLES`` fields of equal weight at evenly spaced intermediate states.
        """
        states = []
        for segment in beam_sequence.segments:
            if not segment.is_dynamic:
                states.append((segment.weight, segment.start.mlc, segment.start.jaws))
                continue
            start, end = segment.start, segment.end
            if start.gantry_angle != end.gantry_angle or start.couch_angle != end.couch_angle:
                raise NotImplementedError(
                    "A segment with gantry or couch motion cannot be delivered from one beam."
                )
            num = self.DYNAMIC_SAMPLES
            for k in range(num):
                state = segment.state_at((k + 0.5) / num)
                states.append((segment.weight / num, state.mlc, state.jaws))
        return states

    @staticmethod
    def _group_by_jaws(states: list[tuple[float, Any, list]]) -> list[list[int]]:
        """Group the indices of fields that share value-identical jaws."""
        groups: list[list[int]] = []
        for m, (_, _, jaws) in enumerate(states):
            for group in groups:
                if jaws == states[group[0]][2]:
                    group.append(m)
                    break
            else:
                groups.append([m])
        return groups

    def _composite_shape_group(
        self, mlcs: list, weights: list[float], extra_blds: Optional[list]
    ) -> FieldShape:
        """Build one field shape for a group of fields that share identical jaws."""
        extra_shapes = [
            FieldShapeAsBLD(energy=self.energy, bld=bld, resolution=self.resolution)
            for bld in (extra_blds or [])
        ]
        if all(mlc is None for mlc in mlcs):
            if not extra_shapes:
                raise ValueError("A segment needs an MLC or jaws to shape its field.")
            beamlet = (
                extra_shapes[0]
                if len(extra_shapes) == 1
                else FieldShapeComposite(energy=self.energy, shapes=extra_shapes)
            )
            beamlet.weight = float(sum(weights))
            return beamlet
        if any(mlc is None for mlc in mlcs):
            raise ValueError("Segments with and without an MLC cannot share one beam.")
        mlc_shapes = [
            FieldShapeAsBLD(energy=self.energy, bld=mlc, resolution=self.resolution)
            for mlc in mlcs
        ]

        if len(mlc_shapes) == 1:
            shapes = [mlc_shapes[0], *extra_shapes]
            beamlet = (
                shapes[0]
                if len(shapes) == 1
                else FieldShapeComposite(energy=self.energy, shapes=shapes)
            )
            beamlet.weight = float(weights[0])
            return beamlet

        field_width = max(s.field_width for s in (*mlc_shapes, *extra_shapes))
        grid = FieldShape._build_grid(self.resolution, field_width)

        combined_mask = sum(
            w * s.resample(new_grid=grid).mask for w, s in zip(weights, mlc_shapes)
        )
        for extra_shape in extra_shapes:
            combined_mask = combined_mask * extra_shape.resample(new_grid=grid).mask

        return FieldShapeAsMask(energy=self.energy, mask=combined_mask, grid=grid, weight=1.0)

    def _merge_group_shapes(self, group_shapes: list[FieldShape]) -> FieldShape:
        """Combine the composite field shape of every jaw group into one beam-wide mask."""
        if len(group_shapes) == 1:
            return group_shapes[0]

        field_width = max(s.field_width for s in group_shapes)
        grid = FieldShape._build_grid(self.resolution, field_width)

        combined_mask = sum(s.weight * s.resample(new_grid=grid).mask for s in group_shapes)

        return FieldShapeAsMask(energy=self.energy, mask=combined_mask, grid=grid, weight=1.0)

    def _create_rays(self, beam: dict) -> list[dict]:
        """Create the rays for the photon IMRT geometry.

        Parameters
        ----------
        beam : dict
            The beam dictionary.

        Returns
        -------
        list[dict]
            A single ray combining every shape of the beam when
            ``beam_sequences`` is set (see ``_merge_group_shapes``), else the
            single ray from the base class.
        """
        rays = super()._create_rays(beam)
        ray = rays[0]  # single ray for single-bixel generator

        if self.beam_sequences is not None:
            states = self._segment_states(self.beam_sequences[beam["beam_ix"]])
            group_shapes = [
                self._composite_shape_group(
                    [states[m][1] for m in group],
                    [float(states[m][0]) for m in group],
                    states[group[0]][2],
                )
                for group in self._group_by_jaws(states)
            ]

            beamlet = self._merge_group_shapes(group_shapes)
            return [
                {
                    **ray,
                    "beamlets": [beamlet],
                    "target_point": 2 * ray["ray_pos"] - beam["source_point"],
                    "target_point_bev": 2 * ray["ray_pos_bev"] - beam["source_point_bev"],
                }
            ]

        beamlet = {"energy": self.energy}
        return [
            {
                **ray,
                "beamlets": [beamlet],
                "target_point": 2 * ray["ray_pos"] - beam["source_point"],
                "target_point_bev": 2 * ray["ray_pos_bev"] - beam["source_point_bev"],
            }
        ]
