from typing import cast
import numpy as np
from ._base_pencilbeam_particle import ParticlePencilBeamEngineAbstract
from pyRadPlan.machines import ParticlePencilBeamKernel


class ParticleHongPencilBeamEngine(ParticlePencilBeamEngineAbstract):
    # constants
    short_name = "HongPB"
    name = "Hong Particle Pencil-Beam"
    possible_radiation_modes = ["protons", "helium", "carbon", "VHEE"]

    # private methods
    def _calc_particle_bixel(self, bixel):
        kernels = self._interpolate_kernels_in_depth(bixel)

        pb_kernel = cast(ParticlePencilBeamKernel, bixel["kernel"])

        # Lateral Component
        if self.lateral_model == "single":
            # Compute lateral sigma
            sigma_sq = kernels["sigma"] ** 2 + bixel["sigma_ini_sq"]
            lateral = np.exp(-bixel["radial_dist_sq"] / (2 * sigma_sq)) / (2 * np.pi * sigma_sq)
        elif self.lateral_model == "double":
            # Compute lateral sigmas
            sigma_sq_narrow = kernels["sigma_1"] ** 2 + bixel["sigma_ini_sq"]
            sigma_sq_broad = kernels["sigma_2"] ** 2 + bixel["sigma_ini_sq"]

            # Calculate lateral profile
            l_narr = np.exp(-bixel["radial_dist_sq"] / (2 * sigma_sq_narrow)) / (
                2 * np.pi * sigma_sq_narrow
            )
            l_bro = np.exp(-bixel["radial_dist_sq"] / (2 * sigma_sq_broad)) / (
                2 * np.pi * sigma_sq_broad
            )
            lateral = (1 - kernels["weight"]) * l_narr + kernels["weight"] * l_bro
        elif self.lateral_model == "multi":
            sigma_sq = kernels["sigma_multi"] ** 2 + bixel["sigma_ini_sq"]
            lateral = np.sum(
                (
                    np.column_stack(
                        (1 - np.sum(kernels["weight_multi"], axis=1), kernels["weight_multi"])
                    )
                    * np.exp(-bixel["radial_dist_sq"][:, np.newaxis] / (2 * sigma_sq))
                    / (2 * np.pi * sigma_sq)
                ),
                axis=1,
            )
        elif self.lateral_model == "singleXY":
            # Extract squared distances in the two lateral axes
            x_sq = bixel["lat_dists"][:, 0] ** 2
            y_sq = bixel["lat_dists"][:, 1] ** 2

            sigma_sq_x = kernels["sigma_x"] ** 2 + bixel["sigma_ini_sq"]
            sigma_sq_y = kernels["sigma_y"] ** 2 + bixel["sigma_ini_sq"]
            sigma_x = np.sqrt(sigma_sq_x)
            sigma_y = np.sqrt(sigma_sq_y)

            # Anisotropic 2D Gaussian
            lateral = np.exp(-(x_sq / (2 * sigma_sq_x) + y_sq / (2 * sigma_sq_y))) / (
                2 * np.pi * sigma_x * sigma_y
            )

        else:
            raise ValueError("Invalid Lateral Model")

        bixel["physical_dose"] = pb_kernel.lateral_cut_off.comp_fac * lateral * kernels["idd"]

        # Check if we have valid dose values
        if np.any(np.isnan(bixel["physical_dose"])) or np.any(bixel["physical_dose"] < 0):
            raise ValueError("Error in particle dose calculation.")

        if self.calc_let:
            bixel["let_dose"] = bixel["physical_dose"] * kernels["let"]

        if self.calc_bio_dose:
            # TODO: correct / adaptive alpha / beta values given tissue indices
            bixel_alpha = kernels["alpha"][0]
            bixel_beta = kernels["beta"][0]

            # Multiple with dose
            bixel["alpha_dose"] = bixel["physical_dose"] * bixel_alpha
            bixel["sqrt_beta_dose"] = bixel["physical_dose"] * np.sqrt(bixel_beta)

    @staticmethod
    def is_available(pln, machine):
        available, msg = [], []
        return available, msg
