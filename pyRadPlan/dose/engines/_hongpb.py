from typing import ClassVar

import array_api_compat

from ...core.xp_utils.compat import interp1d as array_interp
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

        lateral = self._lateral_profile(
            bixel["radial_dist_sq"], bixel.get("lat_dists"), bixel["sigma_ini_sq"], kernels
        )

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

    def _lateral_profile(self, radial_dist_sq, lat_dists, sigma_ini_sq, kernels):
        """Evaluate the lateral model for voxels or broadcast depth/sample batches."""
        xp = array_api_compat.array_namespace(radial_dist_sq)
        # Lateral Component
        if self.lateral_model == "single":
            # Compute lateral sigma
            sigma_sq = kernels["sigma"] ** 2 + sigma_ini_sq
            lateral = xp.exp(-radial_dist_sq / (2 * sigma_sq)) / (2 * xp.pi * sigma_sq)
        elif self.lateral_model == "double":
            # Compute lateral sigmas
            sigma_sq_narrow = kernels["sigma_1"] ** 2 + sigma_ini_sq
            sigma_sq_broad = kernels["sigma_2"] ** 2 + sigma_ini_sq

            # Calculate lateral profile
            l_narr = xp.exp(-radial_dist_sq / (2 * sigma_sq_narrow)) / (
                2 * xp.pi * sigma_sq_narrow
            )
            l_bro = xp.exp(-radial_dist_sq / (2 * sigma_sq_broad)) / (2 * xp.pi * sigma_sq_broad)
            lateral = (1 - kernels["weight"]) * l_narr + kernels["weight"] * l_bro
        elif self.lateral_model == "multi":
            # Kernels are interpolated along depth: sigma_multi (m + 1, n_vox) and
            # weight_multi (m, n_vox) or (n_vox,); the first sigma carries the remaining weight
            sigma_sq = kernels["sigma_multi"] ** 2 + sigma_ini_sq
            w = kernels["weight_multi"]
            if w.ndim == sigma_sq.ndim - 1:
                w = xp.expand_dims(w, axis=0)
            weights_full = xp.concat((1 - xp.sum(w, axis=0, keepdims=True), w), axis=0)
            lateral = xp.sum(
                weights_full
                * xp.exp(-xp.expand_dims(radial_dist_sq, axis=0) / (2 * sigma_sq))
                / (2 * xp.pi * sigma_sq),
                axis=0,
            )
        elif self.lateral_model == "singleXY":
            # Extract squared distances in the two lateral axes
            x_sq = lat_dists[..., 0] ** 2
            y_sq = lat_dists[..., 1] ** 2

            sigma_sq_x = kernels["sigma_x"] ** 2 + sigma_ini_sq
            sigma_sq_y = kernels["sigma_y"] ** 2 + sigma_ini_sq
            sigma_x = xp.sqrt(sigma_sq_x)
            sigma_y = xp.sqrt(sigma_sq_y)

            # Anisotropic 2D Gaussian
            lateral = xp.exp(-(x_sq / (2 * sigma_sq_x) + y_sq / (2 * sigma_sq_y))) / (
                2 * xp.pi * sigma_x * sigma_y
            )

        else:
            raise ValueError("Invalid Lateral Model")

        return lateral

    def _cutoff_dose_profile(self, depths, kernel, sigma_ini_sq, quadrature):
        """Interpolate once per depth, then broadcast over quadrature samples."""
        if (
            getattr(self._calc_particle_bixel, "__func__", None)
            is not ParticleHongPencilBeamEngine._calc_particle_bixel
            or getattr(self._interpolate_kernels_in_depth, "__func__", None)
            is not ParticlePencilBeamEngineAbstract._interpolate_kernels_in_depth
        ):
            # Preserve custom engines' physical-dose overrides unless they provide this hook.
            return super()._cutoff_dose_profile(depths, kernel, sigma_ini_sq, quadrature)

        xp = array_api_compat.array_namespace(depths)
        used = {"idd": 1.6021766208e-02 * kernel["idd"]}
        used.update({name: kernel[name] for name in self._lateral_kernel_names()})
        kernels = array_interp(
            depths + kernel["offset"], kernel["depths"] + kernel["offset"], used
        )
        kernels = {name: xp.expand_dims(value, axis=-1) for name, value in kernels.items()}
        lateral = self._lateral_profile(
            quadrature.radial_dist_sq[None, :], quadrature.lat_dists, sigma_ini_sq, kernels
        )
        return kernel["lateral_cut_off"]["comp_fac"] * lateral * kernels["idd"]

    @staticmethod
    def is_available(pln, machine):
        available, msg = [], []
        return available, msg
