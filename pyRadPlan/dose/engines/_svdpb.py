from typing import TypedDict, Annotated, ClassVar, Literal, Any, cast, Callable, Optional
import logging
import random

import numpy as np
import array_api_compat
from pydantic import Field

from scipy import fft
from scipy.interpolate import RegularGridInterpolator

from pyRadPlan.plan import PhotonPlan
from pyRadPlan.stf import FieldShape

# from pyRadPlan.stf import Beam
from pyRadPlan.machines import PhotonLINAC, PhotonSVDKernel
from pyRadPlan.core.xp_utils import to_numpy
from ._base_pencilbeam import PencilBeamEngineAbstract


logger = logging.getLogger(__name__)


class DijSamplingConfig(TypedDict, total=False):
    """Properties for Dij sampling configuration."""

    rel_dose_threshold: float
    lat_cut_off: float
    type: Literal["radius", "depth"]
    delta_rad_depth: float
    force_penumbra: Optional[float]
    force_uniform_fluence: bool


class PhotonPencilBeamSVDEngine(PencilBeamEngineAbstract):
    """
    Class representing a pencil beam dose calculation engine for photons.

    The implementation is based on the Singular-value decomposition (SVD)
    method by Bortfeld, Schlegel & Rhein (1993).

    Parameters
    ----------
    pln : PhotonPlan
        A photon plan object.

    Attributes
    ----------
    use_custom_primary_photon_fluence : bool
        Use custom primary photon fluence.
    kernel_cutoff : float
        Kernel cutoff.
    random_seed : int
        Random seed.
    int_conv_resolution : float
        Intensity convolution resolution.
    enable_dij_sampling : bool
        Enable Dij sampling.
    dij_sampling : DijSamplingConfig
        Dij sampling configuration.
    """

    short_name = "SVDPB"
    name = "SVD Pencil Beam"
    possible_radiation_modes = ["photons"]

    _dij_guarantee_canonical: ClassVar[bool] = True
    _dij_guarantee_nonzero: ClassVar[bool] = True

    use_custom_primary_photon_fluence: bool = False
    kernel_cutoff: Annotated[float, Field(gt=0.0, description="Lateral kernel cutoff [mm]")] = (
        np.inf
    )
    random_seed: int = 0
    int_conv_resolution: Annotated[
        float, Field(gt=0.0, description="Intensity convolution resolution [mm]")
    ] = 0.5
    force_penumbra: Annotated[Optional[float], Field(gt=0.0)] = None
    force_uniform_fluence: bool = False
    enable_dij_sampling: bool = True
    dij_sampling: DijSamplingConfig = DijSamplingConfig(
        rel_dose_threshold=0.01, lat_cut_off=20, type="radius", delta_rad_depth=5
    )

    def __init__(self, pln: PhotonPlan = None):
        super().__init__(pln)

    def _init_dose_calc(self, ct, cst, stf) -> dict[str, Any]:
        dij = super()._init_dose_calc(ct, cst, stf)

        # dij = []

        # checks of values
        # matrad here tests the kernel cutoff against the tabulated kernels
        # pyRadPlan, however, is more flexible and allows energy-specific kernels,
        # so we can't check this here but only when we load the energy-kernel?
        if self.kernel_cutoff < self.geometric_lateral_cutoff:
            logger.warning(
                "Kernel cutoff smaller than the geometric lateral cutoff. Using geometric cutoff."
            )

        # matRad does the gaussian filter here, but can we do that here?
        # We should be as flexible as to allow beams with different energies / penumbras / kernels
        # moved the kernel filtering to beam initialization

        # Initialize random number generator
        random.seed(self.random_seed)

        return dij

    def _init_beam(self, beam_info, ct, cst, stf, i):
        """
        Initialize a beam for pencil beam dose calculation.

        Parameters
        ----------
        beam_info : dict
            Beam information struct.
        ct : np.ndarray
            MatRad CT struct.
        cst : np.ndarray
            MatRad steering information struct.
        stf : np.ndarray
            MatRad steering information struct.
        i : int
            Index of beam.

        Returns
        -------
        dict
            Updated beam information struct.
        """

        field_based_dose_calc = False
        field_width = 0
        field_shape_idx = []
        for j, ray in enumerate(stf.beams[i].rays):
            for k, beamlet in enumerate(ray.beamlets):
                if isinstance(beamlet, FieldShape):
                    field_shape_idx.append([j, k])
                    # field_based_dose_calc set to true if only one beamlet has a field shape
                    if not field_based_dose_calc:
                        field_based_dose_calc = True

                    # find largest field_width
                    field_width = max(beamlet.field_width, field_width)

        # TODO: it would probably be much faster to not use the full field extent, but maximum jaw positions
        # as field width here

        if field_based_dose_calc:
            logger.debug("Enabling field-based dose calculation for beam %d!", i)
            self._effective_lateral_cutoff = self.geometric_lateral_cutoff + field_width / np.sqrt(
                2
            )
        else:
            logger.debug("Enabling bixel-based dose calculation for beam %d!", i)

        beam_info = super()._init_beam(beam_info, ct, cst, stf, i)

        if not field_based_dose_calc:
            field_width = beam_info["beam"]["bixel_width"]

        beam_info["field_based_dose_calc"] = field_based_dose_calc
        beam_info["effective_lateral_cut_off"] = self._effective_lateral_cutoff

        field_limit = np.ceil(field_width / (2 * self.int_conv_resolution))
        field_grid = self.int_conv_resolution * np.arange(-field_limit, field_limit + 1)
        beam_info["f_x"], beam_info["f_z"] = np.meshgrid(field_grid, field_grid, indexing="xy")

        # TODO: resampling should directly change object, not create new one?
        # TODO: the model_dump here is unfortunate?
        for j, k in field_shape_idx:
            stf.beams[i].rays[j].beamlets[k] = (
                stf.beams[i].rays[j].beamlets[k].resample(new_grid=field_grid)
            )
            beam_info["beam"]["rays"][j]["beamlets"][k] = (
                stf.beams[i].rays[j].beamlets[k].model_dump()
            )
            # TODO: add mask as computed field?
            beam_info["beam"]["rays"][j]["beamlets"][k]["mask"] = np.rot90(
                stf.beams[i].rays[j].beamlets[k].mask, k=-1
            )

        # Get the kernel
        beamlets = [beamlet for ray in beam_info["beam"]["rays"] for beamlet in ray["beamlets"]]

        energies = np.unique([beamlet["energy"] for beamlet in beamlets])

        if len(energies) > 1:
            raise ValueError("Different energies in one photon beam not supported yet.")

        energy = energies[0]

        kernel = cast(PhotonLINAC, self._machine).get_kernel_by_energy(energy)

        min_kernel_spacing = np.min(np.diff(kernel.kernel_pos))
        if self.int_conv_resolution > min_kernel_spacing:
            logger.warning(
                "Chosen kernel convolution resolution of %f mm is larger than minimum kernel "
                "spacing of %f mm. This can strongly affect absolute dosimetry.",
                self.int_conv_resolution,
                min_kernel_spacing,
            )

        if self.kernel_cutoff > kernel.kernel_pos[-1]:
            logger.info(
                "Kernel cutoff (%f mm) is larger than the beam's kernel range (%f mm)."
                " Using kernel range.",
                self.kernel_cutoff,
                kernel.kernel_pos[-1],
            )
            kernel_cutoff = kernel.kernel_pos[-1]
        else:
            kernel_cutoff = self.kernel_cutoff

        if self.force_penumbra is not None:
            penumbra = self.force_penumbra
            logger.info(
                "Using forced penumbra of %f mm for beam %d instead of kernel penumbra of %f mm.",
                penumbra,
                i,
                kernel.penumbra,
            )
        else:
            penumbra = kernel.penumbra
            logger.info("Kernel penumbra: %f mm for beam %d.", penumbra, i)

        sigma_gauss = penumbra / np.sqrt(8 * np.log(2))  # [mm]

        # use 5 times sigma as the limits for the gaussian convolution
        gauss_limit = np.ceil(5 * sigma_gauss / self.int_conv_resolution)
        gauss_grid = self.int_conv_resolution * np.arange(-gauss_limit, gauss_limit)

        gauss_filter_x, gauss_filter_z = np.meshgrid(gauss_grid, gauss_grid, indexing="xy")
        # Scaling with int_conv_resolution^2 for correct convolution integral in mm units
        gauss_filter = (
            self.int_conv_resolution**2
            / (2 * np.pi * sigma_gauss**2)
            * np.exp(-(gauss_filter_x**2 + gauss_filter_z**2) / (2 * sigma_gauss**2))
        )

        gauss_conv_size = 2 * (field_limit + gauss_limit).astype(int)

        beam_info["gauss_conv_size"] = gauss_conv_size
        beam_info["gauss_filter"] = gauss_filter

        # get kernel size and distances
        kernel_limit = np.ceil(kernel_cutoff / self.int_conv_resolution)
        kernel_grid = self.int_conv_resolution * np.arange(-kernel_limit, kernel_limit)

        kernel_x, kernel_z = np.meshgrid(kernel_grid, kernel_grid, indexing="xy")

        # calculate also the total size and distance as we need this during convolution extensively
        kernel_conv_limit = field_limit + gauss_limit + kernel_limit
        kernel_conv_grid = self.int_conv_resolution * np.arange(
            -kernel_conv_limit, kernel_conv_limit
        )
        conv_mx_x, conv_mx_z = np.meshgrid(kernel_conv_grid, kernel_conv_grid, indexing="xy")

        kernel_conv_size = 2 * kernel_conv_limit.astype(int)

        if not field_based_dose_calc:
            n = np.floor(field_width / self.int_conv_resolution).astype(int)
            f_pre = np.ones((n, n), dtype=np.float32)

            if not self.use_custom_primary_photon_fluence:
                f_pre = fft.ifft2(
                    fft.fft2(f_pre, (gauss_conv_size, gauss_conv_size))
                    * fft.fft2(gauss_filter, (gauss_conv_size, gauss_conv_size))
                )
                f_pre = np.real(f_pre)

        # get index of central ray or closest to the central ray
        xp = self.xp
        ray_pos_bev = xp.stack(
            [xp.asarray(ray["ray_pos_bev"]) for ray in beam_info["beam"]["rays"]]
        )
        center = int(xp.argmin(xp.sum(ray_pos_bev**2, axis=1)))

        center_ssd = beam_info["beam"]["rays"][center]["SSD"]

        # get correct kernel for given SSD at central ray
        kernels_at_ssd = kernel.get_kernels_at_ssd(center_ssd)

        # Display console message
        logger.info(
            "Kernel SSD = %g mm using %d components", center_ssd, kernel.num_kernel_components
        )

        # Get Interpolators
        # TODO: need scipy interpolate here probably
        # Kernel has units 1/mm^2, scaled with convolution resolution for correct normalization
        kernel_mxs = np.apply_along_axis(
            lambda x: (
                self.int_conv_resolution**2
                * np.interp(
                    np.sqrt(kernel_x**2 + kernel_z**2), kernel.kernel_pos, x, left=0.0, right=0.0
                )
            ),
            axis=1,
            arr=kernels_at_ssd,
        )

        beam_info["kernel"] = kernel
        beam_info["kernel_mxs"] = kernel_mxs
        # beam_info["kernel_xz"] = (kernel_x.ravel(), kernel_z.ravel())
        # beam_info["conv_mx_xz"] = (conv_mx_x, conv_mx_z)
        beam_info["kernel_conv_grid"] = kernel_conv_grid
        beam_info["kernel_conv_size"] = kernel_conv_size

        if not field_based_dose_calc and not self.use_custom_primary_photon_fluence:
            beam_info["f_pre"] = f_pre
            kernel_interpolators = self._get_kernel_interpolators(beam_info, f_pre)
            beam_info["kernel_interpolators"] = kernel_interpolators

        return beam_info

    def _compute_bixel(self, curr_ray: dict[str], k: int) -> dict[str, Any]:
        """
        PyRadPlan photon dose calculation for a single bixel.

        call
            bixel = self.computeBixel(currRay,k)
        """
        bixel = {}

        kernel = cast(PhotonSVDKernel, curr_ray["kernel"])

        m = kernel.m
        betas = kernel.kernel_betas
        rd = curr_ray["rad_depths"]
        interpolators = cast(list[RegularGridInterpolator], curr_ray["kernel_interpolators"])
        iso_lat_dists = curr_ray["iso_lat_dists"]
        geo_depths = curr_ray["geo_depths"]
        sad = curr_ray["sad"]

        xp = array_api_compat.array_namespace(rd)

        # Reshape for broadcasting: betas (n_components, 1), rd (1, n_voxels)
        betas = xp.reshape(xp.asarray(betas), (-1, 1))
        rd = xp.reshape(rd, (1, -1))
        m_arr = xp.asarray(m, dtype=rd.dtype)

        dose_component = betas / (betas - m_arr) * (xp.exp(-m_arr * rd) - xp.exp(-betas * rd))

        # scipy interpolators require NumPy arrays
        iso_lat_dists_np = to_numpy(iso_lat_dists)
        interpolated_kernels = [interp(iso_lat_dists_np) for interp in interpolators]

        for c, interp in enumerate(interpolated_kernels):
            dose_component[c, :] *= xp.asarray(interp)

        bixel_dose = xp.sum(dose_component, axis=0)

        bixel_dose = bixel_dose * ((sad / geo_depths) ** 2)

        bixel["physical_dose"] = bixel_dose
        bixel["weight"] = curr_ray["beamlets"][k]["weight"]
        bixel["ix"] = curr_ray["ix"]

        return bixel

    def _get_kernel_interpolators(self, beam_info: dict[str], f: np.ndarray) -> list[Callable]:
        """Get kernel interpolator for photon dose calculation."""

        num_kernels = cast(PhotonSVDKernel, beam_info["kernel"]).num_kernel_components
        conv_size = beam_info["kernel_conv_size"]
        kernel_mxs = beam_info["kernel_mxs"]
        conv_grid = beam_info["kernel_conv_grid"]

        interpolators = [None] * num_kernels
        for c in range(num_kernels):
            conv_mx = np.real(
                fft.ifft2(
                    fft.fft2(f, (conv_size, conv_size))
                    * fft.fft2(kernel_mxs[c], (conv_size, conv_size))
                )
            )
            interpolators[c] = RegularGridInterpolator((conv_grid, conv_grid), conv_mx)

        return interpolators

    def _sample_dij(self, ix, bixel_dose, rad_depth_v, rad_distances_sq, bixel_width):
        """Perform lateral sampling of the beam."""

        raise NotImplementedError("This method is not implemented yet.")

    def _init_ray(self, beam_info: dict[str], j: int) -> dict[str]:
        """Initialize the current ray."""

        ray = super()._init_ray(beam_info, j)

        ray["kernel"] = beam_info["kernel"]

        if self.use_custom_primary_photon_fluence or beam_info["field_based_dose_calc"]:
            print("Calculating custom kernel interpolators for ray ", j)
            if beam_info["field_based_dose_calc"]:
                f = ray["beamlets"][0]["mask"]
                # multiply masks of beamlets on a ray to get field mask for ray
                # TODO: ensure matching grids? should be handled in ersampling above
                for k, beamlet in enumerate(ray["beamlets"]):
                    if k == 0:
                        continue
                    f *= ray["beamlets"][k]["mask"]
            else:
                # TODO: Not saved yet
                f = beam_info["f_pre"]

            f = cast(np.ndarray, f)  # Typing

            primary_fluence = cast(PhotonSVDKernel, beam_info["kernel"]).primary_fluence
            r = np.sqrt(
                (beam_info["f_x"] - ray["ray_pos_bev"][0]) ** 2
                + (beam_info["f_z"] - ray["ray_pos_bev"][2]) ** 2
            )

            if not (f.shape == beam_info["f_x"].shape == beam_info["f_z"].shape == r.shape):
                raise ValueError("Shape mismatch in kernel interpolation!")

            if self.force_uniform_fluence:
                fx = f
            else:
                fx = f * np.interp(r, primary_fluence[:, 0], primary_fluence[:, 1])

            n = beam_info["gauss_conv_size"]
            gauss_filter = beam_info["gauss_filter"]

            fx = np.real(fft.ifft2(fft.fft2(fx, (n, n)) * fft.fft2(gauss_filter, (n, n))))

            ray["kernel_interpolators"] = self._get_kernel_interpolators(beam_info, fx)
        else:
            ray["kernel_interpolators"] = beam_info["kernel_interpolators"]

        return ray
