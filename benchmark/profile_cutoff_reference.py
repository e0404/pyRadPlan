"""Frozen cutoff/physical-dose reference from 115e6ca8 for numerical comparisons."""

# ruff: noqa
import logging
import inspect
import textwrap
import warnings
from copy import deepcopy
import numpy as np
import array_api_compat
from scipy.integrate import cumulative_trapezoid
from pyRadPlan.machines.particles import ParticleAccelerator, LateralCutOff

logger = logging.getLogger(__name__)


def make_hoisted_cutoff():
    """Build the isolated conversion-hoisting variant from this frozen source."""
    source = textwrap.dedent(inspect.getsource(legacy_cutoff))
    source = source.replace('"kernel": base_kernel.to_namespace(np),', '"kernel": base_kernel_np,')
    source = source.replace(
        "for j, current_depth in enumerate(depth_values):",
        "base_kernel_np = base_kernel.to_namespace(np)\n"
        "        for j, current_depth in enumerate(depth_values):",
    )
    namespace = dict(globals())
    exec(compile(source, "<frozen hoisted cutoff>", "exec"), namespace)
    return namespace["legacy_cutoff"]


def legacy_cutoff(self, cut_off_level, stf_element):
    # Sanity Checks
    assert isinstance(self._machine, ParticleAccelerator)

    if len(stf_element) > 1 and not isinstance(stf_element, dict):
        raise ValueError(
            "CutOff can only be precalculated for a single element, "
            "but you provided steering information for multiple beams!"
        )

    if cut_off_level <= 0.98:
        logger.warning("A lateral cut off below 0.98 may result in an inaccurate dose calculation")

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

    # Handle different lateral models for integration
    if self.lateral_model == "singleXY":
        v_theta = np.linspace(0, 2 * np.pi, 361)
        v_theta = v_theta[:-1]  # remove last to avoid duplication at 2*pi
        # angular bin size
        d_theta_angle = (v_theta[1] - v_theta[0]) * np.ones_like(v_theta)

        # Store original radial mid vector length before expansion
        n_radial_bins = r_mid.size  # equals numel(vR)-1 in MATLAB
        n_angles = v_theta.size

        r_mid_mesh, theta_mesh = np.meshgrid(r_mid, v_theta)

        # Flatten r_mid_mesh
        r_mid = r_mid_mesh.ravel(order="F")
        radial_dist_sq = r_mid**2

        # Expand dr /repeat
        dr = np.repeat(dr, n_angles)

        # Flatten theta
        theta = theta_mesh.ravel(order="F")

        # Expand d_theta: replicate angular step sequence for each radial bin
        d_theta = np.tile(d_theta_angle, n_radial_bins)

        # Lateral distances per sample point (vector of radii times unit vectors)
        sqrt_r2 = np.sqrt(radial_dist_sq)
        lat_dists = sqrt_r2[:, np.newaxis] * np.column_stack((np.cos(theta), np.sin(theta)))
    else:
        lat_dists = np.tile(np.sqrt(radial_dist_sq / 2)[:, np.newaxis], (1, 2))
        d_theta = 2 * np.pi

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
        # TODO: what's the purpose of this? in matRad it counts the len(v_ssd)+1 for some reason
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
    energy_sigma_lut = np.hstack((energy_sigma_lut, np.full((len(energy_sigma_lut), 1), np.nan)))

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
        if self._machine.radiation_mode == "protons" and range_shifter_lut[i]["eq_thickness"] > 0:
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
        energy_steps_tail = (cum_int_energy[-1] - cum_int_energy[peak_ix_org]) / num_depth_val_tail

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
            # depth (physical dose only: no tissue parameters, hence no bio evaluation)
            bixel = {
                "energy_ix": energy_ix,
                "kernel": base_kernel.to_namespace(np),
                "radial_dist_sq": radial_dist_sq,
                "lat_dists": lat_dists,
                "sigma_ini_sq": largest_sigma_sq4unique_energies[
                    cnt - 1
                ],  # TODO: check if this result is correct (rounded but may be right)
                "rad_depths": (current_depth + base_kernel.offset) * np.ones_like(radial_dist_sq),
                "sub_ray_ix": np.ones_like(radial_dist_sq, dtype=bool),
                "ix": np.arange(len(radial_dist_sq)),
                "rad_depth_offset": 0,
                "add_sigma_sq": 0,
            }

            self._calc_particle_bixel(bixel)
            dose_r = bixel["physical_dose"]

            cum_area = np.cumsum(r_mid * dose_r * dr * d_theta)

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


def legacy_particle_bixel(self, bixel):
    kernels = self._interpolate_kernels_in_depth(bixel)

    xp = array_api_compat.array_namespace(bixel["radial_dist_sq"])
    pb_kernel = bixel[
        "kernel"
    ]  # {"lateral_cut_off":{"comp_fac": xp.asarray(bixel["kernel"]["lateral_cut_off"]["comp_fac"])}}

    # Lateral Component
    if self.lateral_model == "single":
        # Compute lateral sigma
        sigma_sq = kernels["sigma"] ** 2 + bixel["sigma_ini_sq"]
        lateral = xp.exp(-bixel["radial_dist_sq"] / (2 * sigma_sq)) / (2 * xp.pi * sigma_sq)
    elif self.lateral_model == "double":
        # Compute lateral sigmas
        sigma_sq_narrow = kernels["sigma_1"] ** 2 + bixel["sigma_ini_sq"]
        sigma_sq_broad = kernels["sigma_2"] ** 2 + bixel["sigma_ini_sq"]

        # Calculate lateral profile
        l_narr = xp.exp(-bixel["radial_dist_sq"] / (2 * sigma_sq_narrow)) / (
            2 * xp.pi * sigma_sq_narrow
        )
        l_bro = xp.exp(-bixel["radial_dist_sq"] / (2 * sigma_sq_broad)) / (
            2 * xp.pi * sigma_sq_broad
        )
        lateral = (1 - kernels["weight"]) * l_narr + kernels["weight"] * l_bro
    elif self.lateral_model == "multi":
        # Kernels are interpolated along depth: sigma_multi (m + 1, n_vox) and
        # weight_multi (m, n_vox) or (n_vox,); the first sigma carries the remaining weight
        radial_dist_sq = bixel["radial_dist_sq"]
        sigma_sq = kernels["sigma_multi"] ** 2 + bixel["sigma_ini_sq"]
        w = xp.reshape(kernels["weight_multi"], (-1, radial_dist_sq.shape[0]))
        weights_full = xp.concat((1 - xp.sum(w, axis=0, keepdims=True), w), axis=0)
        lateral = xp.sum(
            weights_full
            * xp.exp(-xp.expand_dims(radial_dist_sq, axis=0) / (2 * sigma_sq))
            / (2 * xp.pi * sigma_sq),
            axis=0,
        )
    elif self.lateral_model == "singleXY":
        # Extract squared distances in the two lateral axes
        x_sq = bixel["lat_dists"][:, 0] ** 2
        y_sq = bixel["lat_dists"][:, 1] ** 2

        sigma_sq_x = kernels["sigma_x"] ** 2 + bixel["sigma_ini_sq"]
        sigma_sq_y = kernels["sigma_y"] ** 2 + bixel["sigma_ini_sq"]
        sigma_x = xp.sqrt(sigma_sq_x)
        sigma_y = xp.sqrt(sigma_sq_y)

        # Anisotropic 2D Gaussian
        lateral = xp.exp(-(x_sq / (2 * sigma_sq_x) + y_sq / (2 * sigma_sq_y))) / (
            2 * xp.pi * sigma_x * sigma_y
        )

    else:
        raise ValueError("Invalid Lateral Model")

    bixel["physical_dose"] = pb_kernel["lateral_cut_off"]["comp_fac"] * lateral * kernels["idd"]

    # Check if we have valid dose values
    if xp.any(xp.isnan(bixel["physical_dose"])) or xp.any(bixel["physical_dose"] < 0):
        raise ValueError("Error in particle dose calculation.")

    if self._calc_let:
        bixel["let_dose"] = bixel["physical_dose"] * kernels["let"]

    if self._calc_bio_dose and "v_alpha_x" in bixel:
        bixel.update(self._evaluate_bio_influence(bixel, kernels))
