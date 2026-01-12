"""VHEE (Very High Energy Electrons) steering information generator."""

import warnings

from pyRadPlan.ct._hlut import default_hlut
from pyRadPlan.machines import VHEEAccelerator
from pyRadPlan.stf._beamlet import Beamlet
from pyRadPlan.stf._ray import Ray
from pyRadPlan.stf._rangeshifter import RangeShifter

from ._externalbeam import StfGeneratorExternalBeamRayBixel


class StfGeneratorVHEE(StfGeneratorExternalBeamRayBixel):
    """
    VHEE (Very High Energy Electrons) Geometry Generator.

    This generator creates steering information for Very High Energy Electron beams.
    Unlike ion beams, VHEE uses a single fixed energy per beam.

    Attributes
    ----------
    name : str
        The name of the generator ("VHEE Geometry Generator").
    short_name : str
        The short name of the generator ("VHEE").
    possible_radiation_modes : list[str]
        A list of possible radiation modes (["VHEE"]).
    energy : float, optional
        The energy for VHEE beams in MeV. If not specified, defaults to 200 MeV.
    """

    name = "VHEE Geometry Generator"
    short_name = "VHEE"
    possible_radiation_modes = ["VHEE"]

    use_given_wet_image: bool

    def __init__(self, pln=None):
        self.energy = None  # Will default to 200 MeV if not specified
        self.radiation_mode = "VHEE"
        self.use_given_wet_image = False
        self._wet_image = None
        super().__init__(pln)

    def _initialize(self):
        """Initialize the VHEE generator."""
        super()._initialize()

        # Validate machine type
        if not isinstance(self.machine, VHEEAccelerator):
            raise ValueError("Machine must be an instance of VHEEAccelerator")

        self._available_energies = self.machine.energies
        if self.use_given_wet_image:
            if self._ct.cube is None:
                warnings.warn("No WET CT provided provided in CT. Cannot use given WET image.")
                self.use_given_wet_image = False

        if self.use_given_wet_image:
            self._wet_image = self._ct.cube
        else:
            # TODO: load HLUT for given scanner
            self._wet_image = self._ct.compute_wet(default_hlut(self.radiation_mode))

    def _init_beam_data(self, beam):
        """
        Initialize beam data with VHEE-specific parameters.

        Parameters
        ----------
        beam : dict
            The beam dictionary to initialize.

        Returns
        -------
        dict
            The initialized beam with VHEE energy set.
        """
        # Set default energy if not specified
        if self.energy is None:
            beam["vhee_energy"] = 200.0  # Default to 200 MeV
        else:
            beam["vhee_energy"] = self.energy

        # Optional: check if the energy is available in machine data
        if hasattr(self.machine, "energies") and self.machine.energies is not None:
            if beam["vhee_energy"] not in self.machine.energies:
                warnings.warn(
                    f"The specified VHEE energy ({beam['vhee_energy']} MeV) "
                    f"is not found in machine.data.energies!"
                )

        return beam

    def _create_rays(self, beam) -> list[dict]:
        """
        Create rays for VHEE beam.

        For VHEE, each ray gets a single beamlet with the fixed energy.

        Parameters
        ----------
        beam : dict
            The beam information.

        Returns
        -------
        list[dict]
            List of ray dictionaries with VHEE beamlets.
        """
        # Get the base rays from the parent class
        rays = super()._create_rays(beam)

        # Initialize beam data with VHEE specifics
        beam = self._init_beam_data(beam)

        # For each ray, create a single beamlet with the VHEE energy
        for ray in rays:
            beamlets = []

            # Create a single beamlet for VHEE
            beamlet = Beamlet(
                energy=beam["vhee_energy"],
                num_particles_per_mu=1.0e6,  # Standard value
                min_mu=0.0,
                max_mu=float("inf"),
                range_shifter=RangeShifter(),  # No range shifter for VHEE
                focus_ix=0,  # Default focus
            )
            beamlets.append(beamlet)

            ray["beamlets"] = beamlets

        # Validate rays - remove any empty rays
        rays = [ray for ray in rays if len(ray["beamlets"]) > 0]

        # Validate each ray using the Ray model
        for r, ray in enumerate(rays):
            rays[r] = Ray.model_validate(ray)

        return rays

    def _generate_source_geometry(self):
        """
        Generate the source geometry for VHEE.

        Returns
        -------
        list
            List of beam dictionaries representing the steering information.
        """
        stf = super()._generate_source_geometry()

        # Add VHEE-specific beam finalization
        stf = [self._finalize_beam(beam) for beam in stf]

        return stf

    def _finalize_beam(self, beam):
        """
        Finalize beam data for VHEE.

        Remove any rays that don't have valid beamlets.

        Parameters
        ----------
        beam : dict
            The beam to finalize.

        Returns
        -------
        dict
            The finalized beam.
        """
        # Remove rays without energy/beamlets
        valid_rays = []
        for ray in beam["rays"]:
            if hasattr(ray, "beamlets") and len(ray.beamlets) > 0:
                # Check if beamlets have valid energy
                if ray.beamlets[0].energy is not None:
                    valid_rays.append(ray)

        beam["rays"] = valid_rays

        return beam
