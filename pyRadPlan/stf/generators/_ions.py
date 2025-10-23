"""Provides basic classes for generating STF for ion beams."""

import warnings
import logging
from typing import cast

import numpy as np

from pyRadPlan.raytracer import RayTracerSiddon
from pyRadPlan.machines import IonAccelerator
from pyRadPlan.ct import default_hlut

from .._rangeshifter import RangeShifter
from .._beamlet import IonSpot
from .._ray import Ray
from ._externalbeam import StfGeneratorExternalBeamRayBixel

logger = logging.getLogger(__name__)


class StfGeneratorIonRayBixel(StfGeneratorExternalBeamRayBixel):
    """
    Intermediate Interface for an Ion-Ray-Bixel Geometry Generator.

    Attributes
    ----------
    use_range_shifter : bool
        Use range shifter when finding energies for irradiation.
    use_given_wet_image : bool
        Use a provided WET image for range determination.
    """

    use_range_shifter: bool
    use_given_wet_image: bool

    def __init__(self, pln=None):
        self.use_range_shifter = False
        self.use_given_wet_image = False
        self._wet_image = None
        super().__init__(pln)

    def _initialize(self):
        super()._initialize()

        if not isinstance(self.machine, IonAccelerator):
            raise ValueError("Machine must be an instance of IonAccelerator")

        self._available_peak_positions = self.machine.peak_positions
        self._available_energies = self.machine.energies

        if self.use_range_shifter:
            range_shifter_wepl = np.round(self._available_peak_positions.min() * 1.25)
            self._available_peak_positions_range_shifter = (
                self.machine.peak_positions - range_shifter_wepl
            )
            warnings.warn(
                "Use of range shifter was enabled."
                "For now, pyRadPlan will generate generic range shifter "
                f"with WEPL {range_shifter_wepl} mm to enable ranges below "
                "the minimum energy in the machine."
            )

        if self.use_given_wet_image:
            if self._ct.cube is None:
                warnings.warn("No WET CT provided provided in CT. Cannot use given WET image.")
                self.use_given_wet_image = False

        if self.use_given_wet_image:
            self._wet_image = self._ct.cube
        else:
            # TODO: load HLUT for given scanner
            self._wet_image = self._ct.compute_wet(default_hlut(self.radiation_mode))


class StfGeneratorIonSingleSpot(StfGeneratorIonRayBixel):
    """Class representing a Ion Single Spot Geometry Stf Generator.

    Attributes
    ----------
    name : str
        The name of the generator ("Ion Single Spot").
    short_name : str
        The short name of the generator ("ionSingleSpot").
    possible_radiation_modes : list[str]
        A list of possible radiation modes.
    """

    name = "Ion Single Spot"
    short_name = "ionSingleSpot"
    possible_radiation_modes = ["protons", "helium", "carbon", "oxygen"]

    fix_energy: float

    def __init__(self, pln=None):
        self.fix_energy = None
        super().__init__(pln)

    def _generate_ray_positions_in_isocenter_plane(self, beam):
        """Generate the ray positions in the isocenter plane.

        As we have a single spot, this is a single ray to the isocenter.

        Parameters
        ----------
        beam : dict
            The beam dictionary. Ignored in this case

        Returns
        -------
        np.ndarray
            A numpy array with the single ray position of (0, 0, 0) in the
            isocenter.
        """

        return np.zeros((3, 1), dtype=float)

    def _generate_source_geometry(self):
        """Generate the source geometry for the ion single spot geometry."""
        stf = super()._generate_source_geometry()

        return stf

    def _create_rays(self, beam: dict) -> list[dict]:
        ray = super()._create_rays(beam)

        beamlets = []
        tmp_machine = cast(IonAccelerator, self.machine)
        closest_energy_ix = tmp_machine.get_closest_energy_index(self.fix_energy)
        closest_energy = tmp_machine.energies[closest_energy_ix]
        if closest_energy != self.fix_energy:
            logger.info(
                "No exact energy match for requested energy %f.Chose closest energy of %f.",
                self.fix_energy,
                closest_energy,
            )
        beamlet = IonSpot(
            energy=closest_energy,
        )

        beamlets.append(beamlet)

        target_point = 2 * (ray[0]["ray_pos"] - beam["source_point"]) + beam["source_point"]
        target_points_bev = (
            2 * (ray[0]["ray_pos_bev"] - beam["source_point_bev"]) + beam["source_point_bev"]
        )
        ray[0]["target_point"] = target_point
        ray[0]["target_point_bev"] = target_points_bev

        ray[0]["beamlets"] = beamlets
        ray[0] = Ray.model_validate(ray[0])

        return ray


class StfGeneratorIMPT(StfGeneratorIonRayBixel):
    """Class representing an Ion IMPT Geometry Stf Generator.

    Attributes
    ----------
    name : str
        The name of the generator ("Photon IMRT Geometry").
    short_name : str
        The short name of the generator ("photonIMRT").
    possible_radiation_modes : list[str]
        A list of possible radiation modes (["photons"]).
    """

    name = "IMPT"
    short_name = "IMPT"
    possible_radiation_modes = ["protons", "helium", "carbon", "oxygen"]

    # Alias for bixel_width
    @property
    def lateral_spot_spacing(self) -> float:
        """Alias for the bixel_width property."""
        return self.bixel_width

    @lateral_spot_spacing.setter
    def lateral_spot_spacing(self, value: float):
        self.bixel_width = value

    def __init__(self, pln=None):
        self.radiation_mode = "protons"
        self.longitudinal_spot_spacing = 3.0  # Longitudinal spot spacing in mm
        super().__init__(pln)

    def _generate_source_geometry(self):
        """Generate the source geometry for the photon IMRT geometry."""

        # TODO: Get available Energies
        stf = super()._generate_source_geometry()
        return stf

    def _create_rays(self, beam: dict) -> list[dict]:
        rays = super()._create_rays(beam)

        # Perform vectorized Ray Tracing from source through ray_pos through whole image
        trace_cubes = [self._wet_image, self._target_mask]
        rt = RayTracerSiddon(trace_cubes)

        ray_pos_all = np.zeros((len(rays), 3))
        for i, ray in enumerate(rays):
            ray_pos_all[i, :] = ray["ray_pos"]

        ray_pos_all_bev = np.zeros((len(rays), 3))
        for i, ray in enumerate(rays):
            ray_pos_all_bev[i, :] = ray["ray_pos_bev"]

        target_points = 2 * (ray_pos_all - beam["source_point"]) + beam["source_point"]
        target_points_bev = (
            2 * (ray_pos_all_bev - beam["source_point_bev"]) + beam["source_point_bev"]
        )
        # TODO: do we need to check scenarios here? Compare to matrad...
        _, lengths, rhos, _, _ = rt.trace_rays(
            beam["iso_center"],
            beam["source_point"].reshape((1, 3)),
            target_points,
        )

        for r, ray in enumerate(rays):
            # check if ray hit target

            # valid_alphas = np.isnan(alphas[r, :]) == False
            valid_ixs = np.isfinite(rhos[0][r, :])
            # alpha = alphas[r, valid_alphas]
            length = lengths[r, valid_ixs]
            rho = [cube_rho[r, valid_ixs] for cube_rho in rhos]
            # d12 = d12s[r]
            # ix = ixs[valid_values]

            ray_hit_target = np.count_nonzero(rho[1]) > 0

            ray_energies = np.array([])

            if ray_hit_target:
                ray["target_point"] = target_points[r]
                ray["target_point_bev"] = target_points_bev[r]
                # ct_entry_point = alpha[0] * d12

                # Obtain radiological depths / WET
                # rad_depths = np.cumsum(l * rho[0])
                rad_depths = np.cumsum(length * rho[0]) - length * rho[0] / 2.0

                rho_enter = np.insert(rho[1], 0, 0)
                # Use diff to find next voxel with different value
                diff_voi_enter = np.diff(rho_enter)
                entry_ix = np.argwhere(diff_voi_enter == 1)
                # Manage Edge case for last voxel
                entry_ix[entry_ix > len(rho[1])] = len(rho[1]) - 1
                # target entry at beginning of voxel
                target_entry = rad_depths[entry_ix] - length[entry_ix] * rho[0][entry_ix] / 2.0
                target_entry = target_entry.squeeze(axis=1)

                rho_exit = np.append(rho[1], 0)
                diff_voi_exit = np.diff(rho_exit)
                exit_ix = np.argwhere(diff_voi_exit == -1)
                # Manage Edge case for last voxel
                exit_ix[exit_ix > len(rho[1])] = len(rho[1]) - 1
                target_exit = rad_depths[exit_ix] + length[entry_ix] * rho[0][entry_ix] / 2.0
                target_exit = target_exit.squeeze(axis=1)

                # Sanity Check
                if target_entry.size != target_exit.size:
                    raise ValueError(
                        "Did not find same number of target entries and exits! "
                        "Can not assign energies for IMPT geometry."
                    )

                for i in range(target_entry.size):
                    # TODO: use range shifter

                    # Select energies
                    select_energies = self._available_energies[
                        np.logical_and(
                            self._available_peak_positions >= target_entry[i],
                            self._available_peak_positions <= target_exit[i],
                        )
                    ]

                    ray_energies = np.append(ray_energies, select_energies)

                # TODO: finalize range shifters
                ray_range_shifters = [RangeShifter() for _ in ray_energies]

                # TODO: select foci from available ones using the LUT
                ray_foci = np.zeros_like(ray_energies, dtype=int)

                # TODO: finalize MU settings
                ray_num_particles_per_mu = np.ones_like(ray_energies) * 1.0e6
                ray_min_mu = np.zeros_like(ray_energies)
                ray_max_mu = np.ones_like(ray_energies) * float("inf")
            else:
                ray_energies = np.array([])
                ray_foci = np.array([])

            beamlets = []
            # Create / Validate Beamlets
            for i, energy in enumerate(ray_energies):
                beamlet = IonSpot(
                    energy=energy,
                    num_particles_per_mu=ray_num_particles_per_mu[i],
                    min_mu=ray_min_mu[i],
                    max_mu=ray_max_mu[i],
                    range_shifter=ray_range_shifters[i],
                    focus_ix=ray_foci[i],
                )
                beamlets.append(beamlet)

            ray["beamlets"] = beamlets

        # Clean out empty rays with no energy
        rays = [ray for ray in rays if len(ray["beamlets"]) > 0]

        # Validate rays
        for r, ray in enumerate(rays):
            rays[r] = Ray.model_validate(ray)

        return rays
