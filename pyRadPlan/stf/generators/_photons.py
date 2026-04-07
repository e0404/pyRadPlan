import logging

import numpy as np

from ._externalbeam import StfGeneratorExternalBeamRayBixel
from pyRadPlan.machines import PhotonLINAC

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
