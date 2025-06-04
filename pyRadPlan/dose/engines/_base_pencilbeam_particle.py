"""Base class for particle pencil beam dose calculation algorithms."""

import warnings
from abc import abstractmethod
from typing import cast, Literal, Any
import logging
import time
from copy import deepcopy

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

from pyRadPlan.ct import CT
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.machines import IonAccelerator, IonPencilBeamKernel, LateralCutOff
from pyRadPlan.cst import StructureSet
from ._base_pencilbeam import PencilBeamEngineAbstract


logger = logging.getLogger(__name__)


class ParticlePencilBeamEngineAbstract(PencilBeamEngineAbstract):
    """
    Abstract interface for Particle Pencil-Beam dose calculation.

    This class extends PencilBeamEngineAbstract by adding infrastructure for particles spots and
    quantities like LET and biological dose for variable RBE calculations.

    Attributes
    ----------
    calc_let : bool
        Boolean which defines if LET should be calculated.
    calc_bio_dose : bool
        Boolean to query biological dose calculation.
    air_offset_correction : bool
        Corrects WEPL for SSD difference to kernel database.
    lateral_model : Literal["auto", "single", "double", "multi", "fastest"]
        Lateral Model used. 'auto' uses the most accurate model available ( multiple Gaussians).
        "fastest" uses the most simple, inaccurate model (e.g., single Gaussian).
    cut_off_method : Literal["integral", "relative"]
        Method used to determine lateral cut off. 'integral' uses the integral of the lateral
        dose profile to obtain the cut-off distance. 'relative' finds the lateral distance at which
        the dose drops to the requested cutoff
    """

    calc_let: bool
    calc_bio_dose: bool
    air_offset_correction: bool
    lateral_model: Literal["auto", "single", "double", "multi", "fastest"]
    cut_off_method: Literal["integral", "relative"]

    def __init__(self, pln):
        self.calc_let = True
        self.calc_bio_dose = False
        self.air_offset_correction = True
        self.lateral_model = "fastest"
        self.cut_off_method = "integral"

        # Protected properties with public get access
        self._constant_rbe = None  # constant RBE value
        self._v_tissue_index = None  # Stores tissue indices available in the matRad base data
        self._v_alpha_x = None  # Stores Photon Alpha
        self._v_beta_x = None  # Stores Photon Beta

        super().__init__(pln)

    @abstractmethod
    def _calc_particle_bixel(self, bixel: dict):
        pass

    def _choose_lateral_model(self):
        """Choose & validate the lateral beam model."""

        available_models = {
            "single": self._machine.has_single_gaussian_kernel,
            "double": self._machine.has_double_gaussian_kernel,
            "multi": self._machine.has_multi_gaussian_kernel,
        }

        # First we validate the users choice
        if self.lateral_model in available_models:
            if not available_models[self.lateral_model]:
                logger.warning(
                    "Chosen Machine does not support a %s Gaussian Pencil-Beam model!",
                    self.lateral_model,
                )
                self.lateral_model = "auto"
        elif self.lateral_model not in ["auto", "fastest"]:
            logger.warning(
                "Invalid lateral model %s. Using auto instead.",
                self.lateral_model,
            )
            self.lateral_model = "auto"

        # Now if auto, we choose the most accurate one (i.e. the last one in the dictionary that
        # is available)

        if self.lateral_model == "auto":
            for key, available in available_models.items():
                if available:
                    self.lateral_model = key

        # if it is "auto"
        if self.lateral_model == "fastest":
            for key, available in available_models.items():
                if available:
                    self.lateral_model = key
                    break

        logger.info("Using a %s Gaussian pencil-beam kernel model!\n", self.lateral_model)

    def _compute_bixel(self, curr_ray: dict, k: int) -> dict:
        """
        Compute the bixel for the given ray and index.

        Parameters
        ----------
        curr_ray : dict
            The current ray data.
        k : int
            The index of the bixel.

        Returns
        -------
        dict
            The computed bixel.
        """
        # Initialize Bixel Geometry
        bixel = self._init_bixel(curr_ray, k)

        # Compute Bixel
        self._calc_particle_bixel(bixel)

        return bixel

    def _init_bixel(self, curr_ray, k):
        """
        Initialize general bixel geometry for particle dose calculation.

        Parameters
        ----------
        curr_ray : dict
            The current ray data.
        k : int
            The index of the bixel.

        Returns
        -------
        dict
            The initialized bixel.
        """

        bixel = curr_ray["beamlets"][k]
        bixel["beam_index"] = curr_ray["beam_index"]
        bixel["ray_index"] = curr_ray["ray_index"]
        bixel["bixel_index"] = k

        # First we get metadata: MU, corresponding base data entry, etc.
        # Extract MU data if present (checks for downwards compatibility)
        bixel["min_mu"] = 0
        if "min_mu" in curr_ray:
            bixel["min_mu"] = curr_ray["min_mu"][k]

        bixel["max_mu"] = float("inf")
        if "max_mu" in curr_ray:
            bixel["max_mu"] = curr_ray["max_mu"][k]

        bixel["num_particles_per_mu"] = 1e6
        if "num_particles_per_mu" in curr_ray:
            bixel["num_particles_per_mu"] = curr_ray["num_particles_per_mu"][k]

        # Find energy index in base data
        energy = curr_ray["beamlets"][k]["energy"]
        energy_ix = self._machine.get_energy_index(energy, 4)

        if energy_ix.size > 1:
            raise ValueError("Multiple energies found in base data for one bixel!")

        energy_ix = np.int64(energy_ix)

        bixel["energy_ix"] = energy_ix

        # Get the kernel for the current energy
        tmp_machine = cast(IonAccelerator, self._machine)
        bixel["kernel"] = tmp_machine.get_kernel_by_index(energy_ix)

        bixel["range_shifter"] = curr_ray["beamlets"][k]["range_shifter"]
        bixel["SSD"] = curr_ray["SSD"]
        bixel["rad_depth_offset"] = curr_ray["rad_depth_offset"]

        # Compute initial spot width
        bixel["sigma_ini_sq"] = curr_ray["sigma_ini"][k] ** 2

        # Apply beam modifiers
        self._get_beam_modifiers(bixel)

        # Gets bixel.ix (voxel indices) and bixel.subIx (logical
        # indices to be used) after cutoff. Storing these allows us to
        # use indexing for performance and avoid too many copies
        self._get_bixel_indices_on_ray(bixel, curr_ray)
        if "ix" not in bixel or bixel["ix"].size == 0:
            return bixel

        # Get quantities 1:1 from ray. Here we trust Python's memory
        # management to not copy the arrays until they are modified.
        # This allows us to efficiently access them by indexing in the
        # bixel computation
        bixel["radial_dist_sq"] = curr_ray["radial_dist_sq"][bixel["sub_ix"]]
        bixel["rad_depths"] = curr_ray["rad_depths"][bixel["sub_ix"]]
        # TODO:
        # if self.calc_bio_dose:
        #     bixel["v_tissue_index"] = curr_ray["v_tissue_index"][bixel["sub_ix"]]
        #     bixel["v_alpha_x"] = curr_ray["v_alpha_x"][bixel["sub_ix"]]
        #     bixel["v_beta_x"] = curr_ray["v_beta_x"][bixel["sub_ix"]]

        return bixel

    def _interpolate_kernels_in_depth(self, bixel):
        kernel = cast(IonPencilBeamKernel, bixel["kernel"])
        depths = kernel.depths

        # Add potential offset
        depths = depths + kernel.offset - bixel["rad_depth_offset"]

        # Conversion factor from MeV cm^2/g per primary to Gy mm^2 per 1e6 primaries
        conversion_factor = 1.6021766208e-02

        # Find all values we need to interpolate
        used_kernels = {}
        used_kernels["idd"] = conversion_factor * kernel.idd

        # Lateral Kernel Model
        if self.lateral_model == "single":
            used_kernels["sigma"] = kernel.sigma
        elif self.lateral_model == "double":
            used_kernels["sigma_1"] = kernel.sigma_1
            used_kernels["sigma_2"] = kernel.sigma_2
            used_kernels["weight"] = kernel.weight
        elif self.lateral_model == "multi":
            used_kernels["weight_multi"] = kernel.weight_multi
            used_kernels["sigma_multi"] = kernel.sigma_multi
        else:
            raise ValueError("Invalid Lateral Model")

        # LET
        if self.calc_let:
            used_kernels["let"] = kernel.let

        # bioDose
        # TODO:
        if self.calc_bio_dose:
            used_kernels["alpha"] = kernel.alpha
            used_kernels["beta"] = kernel.beta

        # Interpolate all fields in X
        kernel_interp = {}
        for key, kernel_data in used_kernels.items():
            if kernel_data.ndim > 1:
                tmp_kernel_data = np.apply_along_axis(
                    lambda x: np.interp(bixel["rad_depths"], depths, x), axis=1, arr=kernel_data
                )
            else:
                tmp_kernel_data = np.interp(bixel["rad_depths"], depths, kernel_data)

            kernel_interp[key] = tmp_kernel_data

        return kernel_interp

    def _get_ray_geometry_from_beam(self, ray: dict[str], beam_info: dict[str]):
        lateral_ray_cutoff = self._get_lateral_distance_from_dose_cutoff_on_ray(ray)

        # Ray tracing for beam i and ray j
        ix, radial_dist_sq, _, _ = self.calc_geo_dists(
            beam_info["bev_coords"],
            ray["source_point_bev"],
            ray["target_point_bev"],
            ray["sad"],
            beam_info["valid_coords_all"],
            lateral_ray_cutoff,
        )

        # Subindex given the relevant indices from the geometric distance calculation
        ray["valid_coords"] = [beam_ix & ix for beam_ix in beam_info["valid_coords"]]
        ray["ix"] = [self._vdose_grid[ix_in_grid] for ix_in_grid in ray["valid_coords"]]

        ray["radial_dist_sq"] = [
            radial_dist_sq[beam_ix[ix]] for beam_ix in beam_info["valid_coords"]
        ]

        ray["valid_coords_all"] = np.any(np.vstack(ray["valid_coords"]), axis=1)

        ray["geo_depths"] = [
            rD[ix] for rD, ix in zip(beam_info["geo_depths"], ray["valid_coords"])
        ]  # usually not needed for particle beams
        ray["rad_depths"] = [
            rD[ix] for rD, ix in zip(beam_info["rad_depths"], ray["valid_coords"])
        ]

    def _get_bixel_indices_on_ray(self, curr_bixel, curr_ray):
        kernel = cast(IonPencilBeamKernel, curr_bixel["kernel"])

        # Create offset vector to account for additional offsets modeled in the base data
        # and a potential range shifter
        tmp_offset = kernel.offset - curr_bixel["rad_depth_offset"]

        # Find depth-dependent lateral cutoff
        if self.dosimetric_lateral_cutoff == 1:
            curr_ix = curr_ray["rad_depths"] <= kernel.depths[-1] + tmp_offset
        elif 0 < self.dosimetric_lateral_cutoff < 1:
            cutoff_info = kernel.lateral_cut_off

            if cutoff_info.cut_off.size > 1:
                curr_ix = (
                    np.interp(
                        curr_ray["rad_depths"],
                        cutoff_info.depths + tmp_offset,
                        cutoff_info.cut_off**2,
                        left=np.nan,
                        right=np.nan,
                    )
                    >= curr_ray["radial_dist_sq"]
                ) & (curr_ray["rad_depths"] <= kernel.depths[-1] + tmp_offset)
            else:
                curr_ix = (cutoff_info.cut_off[0] ** 2 >= curr_ray["radial_dist_sq"]) & (
                    curr_ray["rad_depths"] <= kernel.depths[-1] + tmp_offset
                )
        else:
            raise ValueError("dosimetric_lateral_cutoff must be a value > 0 and <= 1!")

        curr_bixel["sub_ix"] = curr_ix
        curr_bixel["ix"] = curr_ray["ix"][curr_ix]

    def _get_beam_modifiers(self, curr_bixel):
        add_sigma_sq = 0
        rad_depth_offset = 0

        # Consider range shifter for protons if applicable
        if curr_bixel["range_shifter"]["eq_thickness"] > 0:
            # TODO: We should do this check in dose calc initialization instead to spare some time
            if self._machine["meta"]["radiationMode"] != "protons":
                warnings.warn(
                    "Range shifter not valid for irradiation with particle other than protons!"
                )

            # Compute!
            sigma_rashi = self._calc_sigma_rashi(
                curr_bixel["kernel"]["energy"], curr_bixel["range_shifter"], curr_bixel["SSD"]
            )

            # Add to initial sigma in quadrature
            add_sigma_sq += sigma_rashi**2
            rad_depth_offset += curr_bixel["range_shifter"]["eq_thickness"]

        curr_bixel["add_sigma_sq"] = add_sigma_sq
        curr_bixel["rad_depth_offset"] += rad_depth_offset

    def _init_dose_calc(self, ct, cst, stf):
        """
        Initialize dose calculation.

        Modified inherited method of the superclass DoseEngine,
        containing intialization which are specificly needed for
        pencil beam calculation and not for other engines.
        """

        dij = super()._init_dose_calc(ct, cst, stf)

        # Choose the lateral pencil beam model
        self._choose_lateral_model()

        # Toggles correction of small difference of current SSD to distance used
        # in generation of base data (e.g. phantom surface at isocenter)

        # TODO: this is a dummy. bio_param not implemented yet...
        self.bio_param = {"bioOpt": False}

        # Omit field checks of fit_air_offset and BAMStoIsoDist as validated through machine model

        # biology
        if hasattr(self, "_constant_rbe") and self._constant_rbe is not None:
            dij["RBE"] = self._constant_rbe

        # TODO: (Comment from matlab): This is clumsy and needs to be changed with the biomodel
        # update
        if self.bio_param["bioOpt"]:
            self.calc_bio_dose = True

        # Load biologicla base data if needed
        if self.calc_bio_dose:
            dij = self._load_biological_kernel(cst, dij)  # TODO: Not fully implemented yet
            # allocate alpha and beta dose container and sparse matrices in the dij struct,
            # for more informations see corresponding method
            dij = self._allocate_bio_dose_container(dij)  # TODO: Not fully implemented yet

        # Allocate LET containner and let sparse matrix in dij struct
        if self.calc_let:
            if self._machine.has_let_kernel:
                dij = self._allocate_let_container(dij)
            else:
                logger.warning(
                    "No LET data found in machine data. LET calculation will be skipped."
                )
                self.calc_let = False

        self._effective_lateral_cutoff = self.geometric_lateral_cutoff

        return dij

    def _init_beam(
        self, dij: dict, ct: CT, cst: StructureSet, stf: SteeringInformation, i
    ) -> dict:
        """
        Initialize beam for dose calculation.

        Extends the inherited method with particle-specific initialization.

        Parameters
        ----------
        dij : dict
            The dose influence matrix dictionary.
        ct : CT
            The CT object.
        _cst : StructureSet
            The structure set object. Unused here
        stf : SteeringInformation
            The steering information object.
        i : int
            Index of the beam.

        Returns
        -------
        dict
            Updated Beam Information dictionary.
        """
        beam_info = super()._init_beam(dij, ct, cst, stf, i)

        # Sanity Check
        assert isinstance(self._machine, IonAccelerator)

        # Assuming currBeam is part of beam_info
        curr_beam = beam_info["beam"]

        # Find max energy
        max_energy = max(
            beamlet["energy"] for ray in curr_beam["rays"] for beamlet in ray["beamlets"]
        )

        max_energy_ix = self._machine.get_energy_index(max_energy, 4)

        max_energy_kernel = self._machine.get_kernel_by_index(max_energy_ix)

        # Find minimum range shifter equivalent thickness
        min_ra_shi = min(
            beamlet["range_shifter"]["eq_thickness"]
            for ray in curr_beam["rays"]
            for beamlet in ray["beamlets"]
        )
        rad_depth_offset = max_energy_kernel.offset + min_ra_shi

        # Apply limit in depth
        sub_select_ix = [
            rD < (max_energy_kernel.depths[-1] - rad_depth_offset)
            for rD in beam_info["rad_depths"]
        ]

        for scen in range(len(beam_info["valid_coords"])):
            beam_info["valid_coords"][scen] = np.array(
                [
                    True if ss & vc else False
                    for ss, vc in zip(sub_select_ix[scen], beam_info["valid_coords"][scen])
                ]
            )

        beam_info["valid_coords_all"] = np.any(np.column_stack(beam_info["valid_coords"]), axis=1)

        # Precompute CutOff
        logger.info("Calculating lateral cutoffs for beam %d...", i)
        t_start = time.time()
        self._calc_lateral_particle_cut_off(self.dosimetric_lateral_cutoff, beam_info)
        t_elapsed = time.time() - t_start
        logger.info("Done in %f seconds.", t_elapsed)

        return beam_info

    def _load_biological_kernel(self, _cst: StructureSet, dij: dict[str, Any]):
        """
        Load / organize biological kernel data from the machine.

        Stores information in the dose influence matrix dictionary.

        Parameters
        ----------
        _cst : StructureSet
            The structure set object.
        dij : dict
            The dose influence matrix dictionary.

        Returns
        -------
        dict
            The updated dose influence matrix dictionary with biological information.
        """
        logger.warning(
            "Biological Kernel not implemented yet. "
            "Biological dose calculation will not honor tissue settings."
        )
        return dij

    def _allocate_bio_dose_container(self, dij: dict[str, Any]):
        """
        Allocate space for container used in LET calculation.

        Parameters
        ----------
        dij : dict
            The dose influence matrix dictionary.

        Returns
        -------
        dict
            The updated dose influence matrix dictionary with LET containers allocated.
        """

        if self._machine.has_alpha_beta_kernels:
            dij = self._allocate_quantity_matrices(dij, ["alpha_dose", "sqrt_beta_dose"])
        else:
            warnings.warn(
                "Biological kernels not available. Biological dose will not be calculated."
            )

        return dij

    def _allocate_let_container(self, dij):
        """
        Allocate space for container used in LET calculation.

        Parameters
        ----------
        dij : dict
            The dose influence matrix dictionary.

        Returns
        -------
        dict
            The updated dose influence matrix dictionary with LET containers allocated.
        """
        # Get MatRad Config instance for displaying warnings

        if self._machine.has_let_kernel:
            dij = self._allocate_quantity_matrices(dij, ["let_dose"])
        else:
            warnings.warn("LET not available in the machine data. LET will not be calculated.")

        return dij

    def _calc_lateral_particle_cut_off(self, cut_off_level, stf_element):
        # Sanity Checks
        assert isinstance(self._machine, IonAccelerator)

        if len(stf_element) > 1 and not isinstance(stf_element, dict):
            raise ValueError(
                "CutOff can only be precalculated for a single element, "
                "but you provided steering information for multiple beams!"
            )

        if cut_off_level <= 0.98:
            logger.warning(
                "A lateral cut off below 0.98 may result in an inaccurate dose calculation"
            )

        if cut_off_level < 0 or cut_off_level > 1:
            warnings.warn("Lateral cutoff is out of range - using default cut off of 0.99")
            cut_off_level = 0.99

        # conversion factor from kernel units (MeV cm^2/g per primary) to Gy mm^2 per 1e6 primaries
        # TODO: this factor should probably be in the machine data and translate to MU
        conversion_factor = 1.6021766208e-02

        # Define some variables needed for the cutoff calculation
        v_x = np.concatenate(([0], np.logspace(-1, 3, 1200)))  # [mm]

        # Integration steps
        r_mid = 0.5 * (v_x[:-1] + v_x[1:])  # [mm]
        dr = np.diff(v_x)
        radial_dist_sq = r_mid**2

        # Number of depth points for which a lateral cutoff is determined
        num_depth_val = 35

        # Extract SSD for each bixel
        v_ssd = np.ones(
            len(
                [
                    beamlet["energy"]
                    for ray in stf_element["beam"]["rays"]
                    for beamlet in ray["beamlets"]
                ]
            )
        )

        cnt = 0
        # TODO: this can be done more efficiently. (instead of calc len() all the time)
        # TODO: these for loops can be simplified!
        for ray in stf_element["beam"]["rays"]:
            len_ray_energy = len([beamlet["energy"] for beamlet in ray["beamlets"]])
            v_ssd[cnt : cnt + len_ray_energy] = ray["SSD"]
            # TODO: whats the purpose of this? in matRad it counts the len(v_ssd)+1 for some reason
            cnt += len_ray_energy

        # Setup energy, focus index, sigma look up table - only consider unique rows
        energy_sigma_lut, ix_unique = np.unique(
            np.column_stack(
                (
                    [
                        beamlet["energy"]
                        for ray in stf_element["beam"]["rays"]
                        for beamlet in ray["beamlets"]
                    ],
                    [
                        beamlet["focus_ix"]
                        for ray in stf_element["beam"]["rays"]
                        for beamlet in ray["beamlets"]
                    ],
                    v_ssd,
                )
            ),
            axis=0,
            return_index=True,
        )
        # add extra dimension for energy_sigma_lut
        # TODO: there must be a simpler way to do this
        energy_sigma_lut = np.hstack(
            (energy_sigma_lut, np.full((len(energy_sigma_lut), 1), np.nan))
        )

        range_shifter_lut = [
            beamlet["range_shifter"]
            for ray in stf_element["beam"]["rays"]
            for beamlet in ray["beamlets"]
        ]
        range_shifter_lut = [range_shifter_lut[i] for i in ix_unique]

        # Find the largest initial beam width considering focus index, SSD and range shifter
        # for each individual energy
        for i in range(len(energy_sigma_lut)):
            # find index of maximum used energy (round to keV for numerical reasons)
            energy_ix = self._machine.get_energy_index(energy_sigma_lut[i, 0], 4)
            # Get the kernel entry
            kernel = self._machine.get_kernel_by_index(energy_ix)
            # Get the available beam foci
            foci = self._machine.get_foci_by_index(energy_ix)

            focus_ix = int(energy_sigma_lut[i, 1])
            focus = foci[focus_ix]

            sigma_ini = np.interp(energy_sigma_lut[i, 2], focus.dist, focus.sigma)

            sigma_ini_sq = sigma_ini**2

            # Consider range shifter for protons if applicable
            if (
                self._machine.radiation_mode == "protons"
                and range_shifter_lut[i]["eq_thickness"] > 0
            ):
                # TODO: this is not tested yet as only Generic is used
                sigma_rashi = self._calc_sigma_rashi(
                    self._machine.energies[energy_ix],
                    range_shifter_lut[i],
                    energy_sigma_lut[i, 2],
                )
                sigma_ini_sq += sigma_rashi**2

            energy_sigma_lut[i, 3] = sigma_ini_sq

        # Find for each individual energy the broadest initial beam width
        unique_energies = np.unique(energy_sigma_lut[:, 0])
        largest_sigma_sq4unique_energies = np.full(len(unique_energies), np.nan)
        ix_max = np.full(len(unique_energies), np.nan)
        for i, energy in enumerate(unique_energies):
            largest_sigma_sq4unique_energies[i] = np.max(
                energy_sigma_lut[energy == energy_sigma_lut[:, 0], 3]
            )
            ix_max[i] = np.argmax(energy_sigma_lut[energy == energy_sigma_lut[:, 0], 3])

        # Get energy indices for looping
        v_energies_ix = np.where(np.isin(self._machine.energies, unique_energies))[0]
        cnt = 0

        # Loop over all entries in the machine.data struct
        for energy_ix in v_energies_ix:
            # Get the kernel entry
            kernel = self._machine.get_kernel_by_index(energy_ix)

            # set default depth cut off - finite value will be set during first iteration
            depth_dose_cut_off = np.inf

            # Get the current integrated depth dose profile
            idd_org = conversion_factor * kernel.idd
            peak_ix_org = np.argmax(idd_org)

            # Get indices for which a lateral cutoff should be calculated
            cum_int_energy = cumulative_trapezoid(idd_org, kernel.depths, initial=0)

            peak_tail_relation = 0.5
            num_depth_val_to_peak = int(np.ceil(num_depth_val * peak_tail_relation))
            num_depth_val_tail = int(np.ceil(num_depth_val * (1 - peak_tail_relation)))
            energy_steps_to_peak = cum_int_energy[peak_ix_org] / num_depth_val_to_peak
            energy_steps_tail = (
                cum_int_energy[-1] - cum_int_energy[peak_ix_org]
            ) / num_depth_val_tail

            v_energy_steps = np.unique(
                np.concatenate(
                    (
                        np.arange(0, cum_int_energy[peak_ix_org], energy_steps_to_peak),
                        [cum_int_energy[peak_ix_org]],
                        np.arange(
                            cum_int_energy[peak_ix_org + 1], cum_int_energy[-1], energy_steps_tail
                        ),
                        [cum_int_energy[-1]],
                    )
                )
            )

            cum_int_energy, ix = np.unique(cum_int_energy, return_index=True)

            depth_values = np.interp(v_energy_steps, cum_int_energy, kernel.depths[ix])
            idd = conversion_factor * np.interp(depth_values, kernel.depths, kernel.idd)

            cnt += 1

            # Create a default cutoff and store it in the kernel
            kernel.lateral_cut_off = LateralCutOff(
                comp_fac=1.0, depths=depth_values, cut_off=np.inf * np.ones_like(depth_values)
            )

            base_kernel = deepcopy(kernel)  # similar to base_data in matRad

            # TODO: this could probably be done for all depths with one calculation without a loop
            for j, current_depth in enumerate(depth_values):
                # If there's no cut-off set, we do not need to find it. Set it to infinity
                if cut_off_level == 1:
                    continue
                # Create a dummy bixel calculating radial dose distribution for the current
                # depth
                bixel = {
                    "energy_ix": energy_ix,
                    "kernel": base_kernel,
                    "radial_dist_sq": radial_dist_sq,
                    "sigma_ini_sq": largest_sigma_sq4unique_energies[
                        cnt - 1
                    ],  # TODO: check if this result is correct (rounded but may be right)
                    "rad_depths": (current_depth + base_kernel.offset)
                    * np.ones_like(radial_dist_sq),
                    "v_tissue_index": np.ones_like(radial_dist_sq),
                    "v_alpha_x": 0.5 * np.ones_like(radial_dist_sq),
                    "v_beta_x": 0.05 * np.ones_like(radial_dist_sq),
                    "sub_ray_ix": np.ones_like(radial_dist_sq, dtype=bool),
                    "ix": np.arange(len(radial_dist_sq)),
                    "rad_depth_offset": 0,
                    "add_sigma_sq": 0,
                }

                self._calc_particle_bixel(bixel)
                dose_r = bixel["physical_dose"]

                # Do an integration check that the radial dose integrates to the tabulated
                # integrated depth dose
                cum_area = np.cumsum(2 * np.pi * r_mid * dose_r * dr)
                relative_tolerance = 0.5  # in [%]

                if abs((cum_area[-1] / idd[j]) - 1) * 100 > relative_tolerance:
                    warnings.warn("Shell integration in cut-off calculation is inconsistent!")

                # obtain the cut-off compensation factor dending on the cut-off method
                if self.cut_off_method == "integral":
                    ix = (cum_area >= idd[j] * cut_off_level).argmax()
                    kernel.lateral_cut_off.comp_fac = cut_off_level**-1
                elif self.cut_off_method == "relative":
                    ix = (dose_r <= (1 - cut_off_level) * np.max(dose_r)).argmax()
                    rel_fac = cum_area[ix] / cum_area[-1]
                    kernel.lateral_cut_off.comp_fac = rel_fac**-1
                else:
                    raise ValueError("Invalid Cutoff Method. Must be 'integral' or 'relative'!")
                # Nothing was cut-off -> warn the user
                if ix == len(cum_area):
                    depth_dose_cut_off = np.inf
                    warnings.warn("Couldn't find lateral cut off!")
                else:
                    depth_dose_cut_off = r_mid[ix]

                kernel.lateral_cut_off.cut_off[j] = depth_dose_cut_off

            self._machine.update_kernel_at_index(energy_ix, kernel)

    def _init_ray(self, beam_info: dict[str], j: int) -> dict[str]:
        ray = super()._init_ray(beam_info, j)

        self._machine = cast(IonAccelerator, self._machine)

        # Calculate initial sigma for all bixels on the current ray
        # TODO: here [ray] since calc_sigma_ini takes multiple rays (why?)
        ray["sigma_ini"] = self._calc_sigma_ini_on_ray(ray)

        # Since pyRadPlan's ray cast starts at the skin and base data
        # is generated at some source to phantom distance
        # we can explicitly correct for the nozzle to air WEPL in
        # the current case.
        if self.air_offset_correction:
            nozzle_to_skin = (ray["SSD"] + self._machine.bams_to_iso_dist) - self._machine.sad
            ray["rad_depth_offset"] = 0.0011 * (nozzle_to_skin - self._machine.fit_air_offset)
        else:
            ray["rad_depth_offset"] = 0

        # Just use tissue classes of voxels found by ray tracer
        # TODO: not tested
        # if self.calc_bio_dose:
        #     for s in range(len(self._v_tissue_index)):
        #         ray["vtissue_index"][s] = self._v_tissue_index[s][ray["valid_coords"][s], :]
        #         ray["valpha_x"][s] = self._v_alpha_x[s][ray["valid_coords"][s]]
        #         ray["vbeta_x"][s] = self._v_beta_x[s][ray["valid_coords"][s]]

        return ray

    def _extract_single_scenario_ray(self, ray, scen_idx):
        scen_ray = super()._extract_single_scenario_ray(ray, scen_idx)
        # TODO: Add multscen support
        # Gets number of scenario
        scen_num = 1  # self.mult_scen['scenNum'][scen_idx]
        ct_scen = self.mult_scen.linear_mask[0][scen_num]

        if "vTissueIndex" in scen_ray:
            scen_ray["vTissueIndex"] = scen_ray["vTissueIndex"][ct_scen]
            scen_ray["vAlphaX"] = scen_ray["vAlphaX"][ct_scen]
            scen_ray["vBetaX"] = scen_ray["vBetaX"][ct_scen]

        return scen_ray

    def _fill_dij(
        self, bixel, dij, stf, scen_idx, curr_beam_idx, curr_ray_idx, curr_bixel_idx, bixel_counter
    ):
        super()._fill_dij(
            bixel, dij, stf, scen_idx, curr_beam_idx, curr_ray_idx, curr_bixel_idx, bixel_counter
        )

        # Add MU information
        if not self._calc_dose_direct:
            dij["min_mu"][bixel_counter] = bixel["min_mu"]
            dij["max_mu"][bixel_counter] = bixel["max_mu"]
            dij["num_of_particles_per_mu"][bixel_counter] = bixel["num_particles_per_mu"]

    def _get_lateral_distance_from_dose_cutoff_on_ray(self, ray: dict):
        # Find index of maximum used energy (round to keV for numerical reasons)
        self._machine = cast(IonAccelerator, self._machine)

        max_energy = max([beamlet["energy"] for beamlet in ray["beamlets"]])
        kernel = self._machine.get_kernel_by_energy(max_energy)
        lat_cut_off = kernel.lateral_cut_off

        # Get the lateral cutoff distance
        lateral_ray_cut_off = lat_cut_off.cut_off.max()

        return lateral_ray_cut_off

    def _calc_sigma_ini_on_ray(self, ray: dict, ssd: np.floating = None):
        """
        Get initial beam width (sigma) at the sruface.

        Parameters
        ----------
        ray : dict
            The ray data.
        ssd : float
            The source to surface distance. Can be omitted if ray has SSD stored

        Returns
        -------
        np.ndarray
            The initial sigma for the ray energies
        """

        if ssd is None:
            if "SSD" not in ray:
                raise ValueError("SSD not provided and not stored in ray data!")
            ssd = ray["SSD"]

        # energies on ray
        sigma_ini = np.zeros((len(ray["beamlets"]),), dtype=np.float64)
        for i, beamlet in enumerate(ray["beamlets"]):
            # obtain focus
            energy = beamlet["energy"]
            focus_ix = beamlet["focus_ix"]
            sigma_ini[i] = self._calc_sigma_ini_focus_ix(energy, focus_ix, ssd)
        return sigma_ini

    def _calc_sigma_ini_focus_ix(
        self, energy: np.float64, focus_ix: int, ssd: np.floating
    ) -> np.float64:
        """
        Get the initial beam width for a specific focus.

        Parameters
        ----------
        energy : np.float64
            The energy of the beam.
        focus_ix : int
            The focus index.
        ssd : np.floating
            The source to surface distance.

        Returns
        -------
        np.float64
            The initial sigma for the given energy and focus index.
        """
        focus = self._machine.foci[energy][focus_ix]
        interpolator = interp1d(focus.dist, focus.sigma, fill_value="extrapolate")
        return interpolator(ssd)

    def _calc_sigma_rashi(self, bd_entry, range_shifter: dict, ssd: float):
        # Distance of range shifter to patient surface
        rashi_dist = ssd - range_shifter["sourceRashiDistance"]

        # Convert to cm
        z_rs_cm = rashi_dist / 10.0  # [cm]
        t = range_shifter["eqThickness"] / 10.0  # [cm]

        # Constants
        a1 = 0.21  # [1]
        a2 = 0.77  # [1]
        c0 = 0.0191027  # [cm]
        c1 = 0.0204539  # [1]
        alpha = 0.0022  # [cm MeV ^ (-p)]
        p = 1.77  # [1]

        # Determine range
        if "range" in bd_entry:
            range_in_water = bd_entry["range"]
        else:
            energy = bd_entry["energy"]
            range_in_water = alpha * (energy**p)

        # Check if valid computation possible or range shifter too thick
        if t / range_in_water >= 0.95:
            raise ValueError(
                "Computation of range shifter sigma invalid. Range shifter is too thick."
            )

        # Improved HONG's Formula [1, Eq. 21]
        s = t / range_in_water
        sigma_t = (a1 * s + a2 * s**2) * (c0 + c1 * range_in_water)

        # Constants for further calculations
        c1 = 13.6  # MeV
        c2 = 0.038
        lr_water = 36.08  # rad. length [cm]
        lr_pmma = 40.55  # rad. length [cm]
        rsp_pmma = 1.165  # rSP [rel. u.]
        prsf_pmma = np.sqrt(
            rsp_pmma**3 * lr_water / lr_pmma * (1 + 2 * c2 * np.log(rsp_pmma * lr_water / lr_pmma))
        )

        # Calculate F1 and F2
        f1_part1 = (2 * c1**2 * alpha ** (2 / p)) / (4 * lr_pmma * (2 / p - 1))
        f1_part2 = ((1 - s) ** (2 - 2 / p) - 1) / (2 / p - 2) - s
        f1_part3 = range_in_water ** (2 - 2 / p)
        f1 = f1_part1 * f1_part2 * f1_part3

        f2_part1 = (c1**2 * alpha ** (2 / p)) / (4 * lr_pmma * (2 / p - 1))
        f2_part2 = ((1 - s) ** (1 - 2 / p) - 1) / 1
        f2_part3 = range_in_water ** (1 - 2 / p)
        f2 = f2_part1 * f2_part2 * f2_part3

        sigma_proj_sq = prsf_pmma**2 * (
            sigma_t**2 + f1 * z_rs_cm * rsp_pmma + f2 * (z_rs_cm * rsp_pmma) ** 2
        )  # [cm ^ 2]

        # Convert to mm
        sigma_rashi = 10.0 * np.sqrt(sigma_proj_sq)  # [mm]

        return sigma_rashi

    def round2(self, a: np.floating, b: int) -> np.floating:
        """
        Round a number stably for energy selection (helper function).

        Parameters
        ----------
        a : np.floating
            The number to round.
        b : int
            The number of decimal places to round to.

        Returns
        -------
        np.floating
            The rounded number.
        """
        return np.round(a * 10**b) / 10**b
