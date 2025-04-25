"""
Implementation of a pencil beam dose calculation engine for photon beams
based on the Singular-value
decomposition (SVD) method by Bortfeld.
"""

from typing import TypedDict, Literal, Any, cast, Callable
import logging
import random

import numpy as np
from scipy import fft
from scipy.interpolate import RegularGridInterpolator

from pyRadPlan.plan import PhotonPlan

# from pyRadPlan.stf import Beam
from pyRadPlan.machines import PhotonLINAC, PhotonSVDKernel
from ._base_pencilbeam import PencilBeamEngineAbstract


logger = logging.getLogger(__name__)


class DijSamplingConfig(TypedDict):
    """Properties for Dij sampling configuration."""

    rel_dose_threshold: float
    lat_cut_off: float
    type: Literal["radius", "depth"]
    delta_rad_depth: float


class PhotonPencilBeamSVDEngine(PencilBeamEngineAbstract):
    """
    Implementation of a pencil beam dose calculation engine for photons.

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

    use_custom_primary_photon_fluence: bool
    kernel_cutoff: float
    random_seed: int
    int_conv_resolution: float = 0.5
    enable_dij_sampling: bool = True
    dij_sampling: DijSamplingConfig

    def __init__(self, pln: PhotonPlan):
        self.use_custom_primary_photon_fluence = False
        self.kernel_cutoff = np.inf
        self.random_seed = 0
        self.int_conv_resolution = 0.5
        self.enable_dij_sampling = True
        self.dij_sampling = DijSamplingConfig(
            rel_dose_threshold=0.01, lat_cut_off=20, type="radius", delta_rad_depth=5
        )

        super().__init__(pln)

        # Protected/Private attributes (equivalent to SetAccess = protected)
        self._is_field_based_dose_calc = None  # will be set
        self._field_width = None  # will be obtained during calculation

        self._collimation = None  # collimation structure from DICOM import

    def _init_dose_calc(self, ct, cst, stf) -> dict[str, Any]:
        # TODO: What is "s" here?
        # self._is_field_based_dose_calc = any(str(s['bixelWidth']) == 'field' for s in stf)

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

        beam_info = super()._init_beam(beam_info, ct, cst, stf, i)

        field_based_dose_calc = False
        if self._is_field_based_dose_calc:
            logger.debug("Enabling field-based dose calculation for beam %d!", i)
            self.int_conv_resolution = self._collimation["conv_resolution"]
            field_width = self._collimation["field_width"]
            field_based_dose_calc = True
        else:
            logger.debug("Enabling bixel-based dose calculation for beam %d!", i)
            field_width = beam_info["beam"]["bixel_width"]

        beam_info["field_based_dose_calc"] = field_based_dose_calc

        # TODO: obtain maximum field limits

        field_limit = np.ceil(field_width / (2 * self.int_conv_resolution))
        # TODO: should this be +1 as end or is it correct? In matRad it is equivalent to no +1
        field_grid = self.int_conv_resolution * np.arange(-field_limit, field_limit)
        beam_info["f_x"], beam_info["f_z"] = np.meshgrid(field_grid, field_grid, indexing="xy")

        # Get the kernel
        beamlets = [beamlet for ray in beam_info["beam"]["rays"] for beamlet in ray["beamlets"]]

        energies = np.unique([beamlet["energy"] for beamlet in beamlets])

        if len(energies) > 1:
            raise ValueError("Different energies in one photon beam not supported yet.")

        energy = energies[0]

        kernel = cast(PhotonLINAC, self._machine).get_kernel_by_energy(energy)

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

        sigma_gauss = kernel.penumbra / np.sqrt(8 * np.log(2))  # [mm]

        # use 5 times sigma as the limits for the gaussian convolution
        gauss_limit = np.ceil(5 * sigma_gauss / self.int_conv_resolution)
        gauss_grid = self.int_conv_resolution * np.arange(-gauss_limit, gauss_limit)

        gauss_filter_x, gauss_filter_z = np.meshgrid(gauss_grid, gauss_grid, indexing="xy")
        gauss_filter = (
            1.0
            / (2 * np.pi * sigma_gauss**2 / self.int_conv_resolution**2)
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

        effective_lateral_cut_off = self.geometric_lateral_cutoff + field_width / np.sqrt(2)
        beam_info["effective_lateral_cut_off"] = effective_lateral_cut_off

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
        center = np.argmin(
            np.sum(
                np.array([ray["ray_pos_bev"] for ray in beam_info["beam"]["rays"]]) ** 2, axis=1
            )
        )

        center_ssd = beam_info["beam"]["rays"][center]["SSD"]

        # get correct kernel for given SSD at central ray
        kernels_at_ssd = kernel.get_kernels_at_ssd(center_ssd)

        # Display console message
        logger.info(
            "Kernel SSD = %g mm using %d components", center_ssd, kernel.num_kernel_components
        )

        # Get Interpolators
        # TODO: need scipy interpolate here probably
        kernel_mxs = np.apply_along_axis(
            lambda x: np.interp(
                np.sqrt(kernel_x**2 + kernel_z**2), kernel.kernel_pos, x, left=0.0, right=0.0
            ),
            axis=1,
            arr=kernels_at_ssd,
        )

        beam_info["kernel"] = kernel
        beam_info["kernel_mxs"] = kernel_mxs
        beam_info["f_pre"] = f_pre
        # beam_info["kernel_xz"] = (kernel_x.ravel(), kernel_z.ravel())
        # beam_info["conv_mx_xz"] = (conv_mx_x, conv_mx_z)
        beam_info["kernel_conv_grid"] = kernel_conv_grid
        beam_info["kernel_conv_size"] = kernel_conv_size

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
        betas = kernel.kernel_betas.reshape((-1, 1))
        rd = curr_ray["rad_depths"].reshape((1, -1))
        interpolators = cast(list[RegularGridInterpolator], curr_ray["kernel_interpolators"])
        iso_lat_dists = curr_ray["iso_lat_dists"]
        geo_depths = curr_ray["geo_depths"]
        sad = curr_ray["sad"]

        dose_component = betas / (betas - m) * (np.exp(-m * rd) - np.exp(-betas * rd))

        interpolated_kernels = [interp(iso_lat_dists) for interp in interpolators]

        for c, interp in enumerate(interpolated_kernels):
            dose_component[c, :] *= interp

        bixel_dose = np.sum(dose_component, axis=0)

        bixel_dose *= ((sad) / geo_depths) ** 2

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
            if beam_info["field_based_dose_calc"]:
                f = ray["shape"]
            else:
                f = beam_info["f_pre"]

            f = cast(np.ndarray, f)  # Typing

            primary_fluence = cast(PhotonSVDKernel, beam_info["kernel"]).primary_fluence
            r = np.sqrt(
                (beam_info["f_x"] - ray["ray_pos_bev"][0]) ** 2
                + (beam_info["f_x"] - ray["ray_pos_bev"][2]) ** 2
            )
            fx = f * np.interp(r, primary_fluence[:, 0], primary_fluence[:, 1])

            n = beam_info["gauss_conv_size"]
            gauss_filter = beam_info["gauss_filter"]

            fx = np.real(fft.ifft2(fft.fft2(fx, (n, n)) * fft.fft2(gauss_filter, (n, n))))

            ray["kernel_interpolators"] = self._get_kernel_interpolators(beam_info, fx)
        else:
            ray["kernel_interpolators"] = beam_info["kernel_interpolators"]

        return ray
