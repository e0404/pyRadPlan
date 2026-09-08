from typing import ClassVar

import array_api_compat

from ._base_pencilbeam_particle import ParticlePencilBeamEngineAbstract


class ParticleHongPencilBeamEngine(ParticlePencilBeamEngineAbstract):
    # constants
    short_name = "HongPB"
    name = "Hong Particle Pencil-Beam"
    possible_radiation_modes = ["protons", "helium", "carbon", "oxygen", "VHEE"]

    _dij_guarantee_canonical: ClassVar[bool] = True
    _dij_guarantee_nonzero: ClassVar[bool] = True

    # private methods
    def _calc_particle_bixel(self, bixel):
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

        bixel["physical_dose"] = (
            pb_kernel["lateral_cut_off"]["comp_fac"] * lateral * kernels["idd"]
        )

        # Check if we have valid dose values
        if xp.any(xp.isnan(bixel["physical_dose"])) or xp.any(bixel["physical_dose"] < 0):
            raise ValueError("Error in particle dose calculation.")

        if self._calc_let:
            bixel["let_dose"] = bixel["physical_dose"] * kernels["let"]

        if self._calc_bio_dose and "v_alpha_x" in bixel:
            bixel.update(self._evaluate_bio_influence(bixel, kernels))

    @staticmethod
    def is_available(pln, machine):
        available, msg = [], []
        return available, msg
