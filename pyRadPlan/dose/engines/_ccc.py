from contextlib import nullcontext
from typing import Annotated, Any, ClassVar, Literal, Optional, cast
import logging
import time

import array_api_compat
import numpy as np
import SimpleITK as sitk
from pydantic import Field
from tqdm.contrib.logging import logging_redirect_tqdm

from pyRadPlan._settings import get_settings
from pyRadPlan.core import resample_image
from pyRadPlan.core.xp_utils import free_gpu_memory, import_backend, to_namespace, to_numpy
from pyRadPlan.core.xp_utils.typing import Array
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.machines import PhotonLINAC, PhotonSVDKernel
from pyRadPlan.plan import PhotonPlan
from pyRadPlan.scenarios import Scenario
from pyRadPlan.stf import FieldShape, SteeringInformation

from ._base import ct_phase_images
from ._base_pencilbeam import PencilBeamEngineAbstract
from ._svdpb import DijSamplingConfig, PhotonPencilBeamSVDEngine
from . import _collapsed_cone as cc
from ._collapsed_cone_cuda import gpu_available

logger = logging.getLogger(__name__)


class PhotonCollapsedConeEngine(PencilBeamEngineAbstract):
    """
    Photon dose calculation with collapsed cone convolution/superposition.

    Port of matRad's ``matRad_PhotonCollapsedConeEngine``. For each bixel the TERMA is
    computed on the dose grid voxels selected by the ray tracer and transported with a set
    of collapsed cones [1]_ of a polyenergetic energy deposition kernel, whose polar axis is
    tilted along the beam axis or the bixel's ray. The absolute dose is normalized to the
    analytical SVD pencil beam depth dose [2]_ of the machine at a reference depth in water.

    TERMA and transport run on the GPU with CuPy when a CUDA device is available (see
    ``use_gpu``), otherwise on the CPU with NumPy, compiled with numba when installed.

    Parameters
    ----------
    pln : PhotonPlan
        A photon plan object.

    Attributes
    ----------
    num_polar_angles : int
        Number of polar cone bins, must divide the 48 cones of the kernel data.
    num_azimuthal_angles : int
        Number of azimuthal cones per polar bin.
    scatter_cutoff : float
        Lateral margin [mm] of the transport sub-volume around the TERMA of each bixel. Dose
        is stored within ``max(geometric_lateral_cutoff, scatter_cutoff)`` around each ray.
    normalization_depth : float
        Depth [mm] of the absolute dose normalization to the machine depth dose.
    kernel_file : str, optional
        Kernel data file (matRad format). Default: 6 MV water kernel.
    use_gpu : bool, optional
        Compute TERMA and transport on the GPU with CuPy. None (default) follows
        ``settings.xp.prefer_gpu``; True requests the GPU. Without CuPy and a CUDA device the
        CPU is used.
    use_numba : bool
        Use the numba transport on the CPU if numba is installed.
    num_threads : int
        Number of numba threads, 0 for numba's default.
    hardening_correction : bool
        Use the depth dependent effective attenuation ``mu0 / (1 + beta d)`` of the TERMA,
        with ``mu0`` and ``beta`` fitted so that the broad field depth dose matches the
        machine data.
    num_scatter_exponentials : int
        Number of exponential terms of the scatter kernel (1 or 2); the stored kernel is
        refitted if it differs.
    kernel_tilt : str
        Polar axis of the cones: ``"beam"`` (beam axis, same transport lattice for all
        bixels of a beam) or ``"ray"`` (ray of each bixel, which causes lattice aliasing in
        the summed dose).
    beam_forward : bool
        Forward dose with a single transport of the weighted TERMA of all bixels of a beam
        (only with ``kernel_tilt="beam"``). Without the cut-offs this equals the dose
        influence matrix times the weights.
    enable_dij_sampling : bool
        Enable the lateral sampling of the dose influence matrix.
    dij_sampling : DijSamplingConfig
        Dij sampling configuration.
    random_seed : int
        Random seed used for the dij sampling.
    ignore_invalid_values : bool
        Do not raise an error for NaN, infinite or negative doses.

    Notes
    -----
    The spectral hardening of the TERMA is approximated by a depth dependent effective
    attenuation coefficient (no off-axis softening, no spectral kernel) and the kernel is
    computed in water and scaled with the relative electron density.

    References
    ----------
    .. [1] Ahnesjo, Med. Phys. 16, 577 (1989)
    .. [2] Bortfeld, Schlegel, Rhein, Med. Phys. 20, 311 (1993)
    """

    short_name = "CCC"
    name = "Collapsed Cone Convolution"
    possible_radiation_modes = ["photons"]

    _dij_guarantee_canonical: ClassVar[bool] = True
    # Voxels of a ray outside the transport sub-volume get a dose of exactly zero
    _dij_guarantee_nonzero: ClassVar[bool] = False

    num_polar_angles: Annotated[int, Field(gt=0, description="Number of polar cone bins")] = 12
    num_azimuthal_angles: Annotated[
        int, Field(gt=0, description="Number of azimuthal cones per polar bin")
    ] = 8
    scatter_cutoff: Annotated[
        float, Field(gt=0.0, description="Lateral transport margin around the TERMA [mm]")
    ] = 150.0
    normalization_depth: Annotated[
        float, Field(gt=0.0, description="Depth of the absolute dose normalization [mm]")
    ] = 100.0
    kernel_file: Annotated[
        Optional[str], Field(description="Kernel data file, default: 6 MV water kernel")
    ] = None
    use_gpu: Annotated[
        Optional[bool],
        Field(description="Use the GPU (CuPy) if available, default: follow the settings"),
    ] = None
    use_numba: bool = True
    num_threads: Annotated[int, Field(ge=0, description="Number of numba threads (0: all)")] = 0
    hardening_correction: bool = True
    num_scatter_exponentials: Literal[1, 2] = 1
    kernel_tilt: Literal["beam", "ray"] = "beam"
    beam_forward: bool = True
    enable_dij_sampling: bool = True
    dij_sampling: DijSamplingConfig = DijSamplingConfig(
        rel_dose_threshold=0.01, lat_cut_off=20, type="radius", delta_rad_depth=5
    )
    random_seed: int = 0
    ignore_invalid_values: bool = False

    # Same lateral dij sampling as the SVD pencil beam engine
    _sample_dij = PhotonPencilBeamSVDEngine._sample_dij

    def __init__(self, pln: PhotonPlan = None):
        self._kernel: Optional[cc.CollapsedConeKernel] = None  # fitted kernel
        self._kernel_meta: dict = {}  # meta information of the kernel file
        self._local_directions: Optional[np.ndarray] = None  # cone directions, polar axis z
        self._direction_weights: Optional[np.ndarray] = None  # kernel weights per direction
        self._direction_att: Optional[np.ndarray] = None  # attenuation per direction
        self._density_cubes: list[Array] = []  # electron density on the dose grid per phase
        self._energy_params: dict[float, dict] = {}  # hardening / normalization per energy
        self._dose_dims: Optional[tuple[int, ...]] = None  # dose grid shape (z, y, x)
        self._dose_res: Optional[np.ndarray] = None  # voxel size along (z, y, x) [mm]
        self._on_gpu = False  # TERMA and transport on the GPU
        self._cx = np  # array module of TERMA and transport (numpy or cupy)
        super().__init__(pln)
        self._rng = np.random.default_rng(self.random_seed)

    @staticmethod
    def is_available(pln: Optional[PhotonPlan], machine: Any) -> tuple[bool, str]:
        """
        Check whether the engine can compute the plan with the machine.

        Parameters
        ----------
        pln : PhotonPlan, optional
            The plan, whose ``prop_dose_calc["kernel_file"]`` selects the kernel.
        machine : Any
            The machine.

        Returns
        -------
        tuple[bool, str]
            Availability and a message with the reason or the warnings.
        """
        available = False
        if not isinstance(machine, PhotonLINAC) or not machine.pb_kernels:
            return available, "Machine has no photon pencil beam data."

        kernel_file = None
        if pln is not None and getattr(pln, "prop_dose_calc", None):
            kernel_file = pln.prop_dose_calc.get("kernel_file")
        try:
            meta = cc.load_kernel_file(kernel_file).meta
        except (FileNotFoundError, ValueError) as err:
            return available, str(err)

        # The kernel is computed for one nominal energy, which every beam energy has to match
        msgs = []
        for kernel in machine.pb_kernels.values():
            ok, energy_msg = cc.check_kernel_energy(meta, kernel.energy)
            if not ok:
                return available, energy_msg
            if energy_msg:
                msgs.append(energy_msg)
        if not cc.NUMBA_AVAILABLE and not gpu_available():
            msgs.append("numba not installed, the slower NumPy transport will be used!")
        return True, " ".join(msgs)

    def _init_dose_calc(
        self, ct: CT, cst: StructureSet, stf: SteeringInformation
    ) -> dict[str, Any]:
        """
        Initialize the dose calculation.

        Loads (and if needed refits) the kernel, sets up the cone directions, selects the
        device of TERMA and transport and interpolates the relative electron densities of
        all CT phases on the dose grid.

        Parameters
        ----------
        ct : CT
            The CT object.
        cst : StructureSet
            The structure set object.
        stf : SteeringInformation
            The steering information object.

        Returns
        -------
        dict
            The dose influence matrix dictionary.
        """
        for beam in stf.beams:
            for ray in beam.rays:
                if any(isinstance(b, FieldShape) for b in ray.beamlets):
                    raise ValueError(
                        f"Field based dose calculation is not supported by the {self.name} engine."
                    )
        if len({beam.bixel_width for beam in stf.beams}) > 1:
            raise ValueError("Different bixel widths per beam are not supported!")

        dij = super()._init_dose_calc(ct, cst, stf)

        if self.scatter_cutoff < 50:
            logger.warning(
                "Scatter cut-off of %g mm is small and truncates the scatter dose.",
                self.scatter_cutoff,
            )

        # Kernel and cone directions in the local frame (polar axis along z)
        self._kernel, self._kernel_meta = cc.load_kernel(
            self.kernel_file, self.num_polar_angles, self.num_scatter_exponentials
        )
        (
            self._local_directions,
            self._direction_weights,
            self._direction_att,
        ) = cc.cone_directions(self._kernel, self.num_azimuthal_angles)

        # Device of TERMA and transport
        wants_gpu = get_settings().xp.prefer_gpu if self.use_gpu is None else self.use_gpu
        self._on_gpu = bool(wants_gpu) and gpu_available()
        if self._on_gpu:
            self._cx = import_backend("cupy")
            logger.info("Collapsed cone TERMA and transport on the GPU (CuPy).")
        else:
            self._cx = np
            if self.use_gpu:
                logger.warning("No CuPy with a CUDA device found, collapsed cone runs on the CPU.")
            elif self.use_numba and not cc.NUMBA_AVAILABLE:
                logger.warning(
                    "numba is not installed, using the slower NumPy collapsed cone transport."
                )

        # Relative electron density of every CT phase on the full dose grid, since the
        # transport also crosses voxels outside the structures
        self._density_cubes = []
        for phase in ct_phase_images(self.cube_wed):
            rho = sitk.GetArrayFromImage(
                resample_image(
                    input_image=sitk.Cast(phase, sitk.sitkFloat64),
                    interpolator=sitk.sitkLinear,
                    target_grid=self.dose_grid,
                )
            ).astype(np.float64)
            rho[~np.isfinite(rho) | (rho < 0)] = 0.0
            self._density_cubes.append(self._cx.asarray(rho))
        self._dose_dims = self._density_cubes[0].shape

        # Voxel size along the numpy axes (z, y, x)
        res = self.dose_grid.resolution
        self._dose_res = np.array([res["z"], res["y"], res["x"]], dtype=np.float64)

        # Hardening and normalization are fitted per beam energy when the beam is initialized
        self._energy_params = {}

        # Initialize random number generator for the dij sampling
        self._rng = np.random.default_rng(self.random_seed)

        return dij

    def _get_energy_params(self, kernel: PhotonSVDKernel) -> dict[str, Any]:
        """
        Get the TERMA parameters and the absolute normalization of a beam energy.

        The effective attenuation is fitted so that the collapsed cone broad field depth dose
        matches the analytical SVD pencil beam depth dose of the machine, which also
        normalizes the absolute dose at ``normalization_depth``. The parameters are cached
        per energy.

        Parameters
        ----------
        kernel : PhotonSVDKernel
            The machine's pencil beam kernel of the beam energy.

        Returns
        -------
        dict
            ``mu0`` and ``beta`` of the effective attenuation [1/mm], the normalization
            factor ``normalization``, the penumbra ``sigma`` [mm] and the radial
            ``primary_fluence``.
        """
        energy = float(kernel.energy)
        if energy in self._energy_params:
            return self._energy_params[energy]

        ok, msg = cc.check_kernel_energy(self._kernel_meta, energy)
        if not ok:
            raise ValueError(msg)
        if msg:
            logger.warning(msg)

        # Reference depth dose from the kernels at the SSD closest to the SAD
        sad = float(self._machine.sad)
        ssd_ix = int(np.argmin(np.abs(np.asarray(kernel.kernel_ssds) - sad)))

        def reference(depths: np.ndarray) -> np.ndarray:
            return cc.analytical_depth_dose(
                kernel.kernel_data[ssd_ix],
                kernel.kernel_pos,
                kernel.kernel_betas,
                kernel.m,
                depths,
            )

        dz = float(np.min(self._dose_res))
        mu0, beta = float(kernel.m), 0.0
        if self.hardening_correction:
            mu0, beta = cc.fit_hardening(
                self._kernel, reference, float(kernel.m), dz, self.normalization_depth
            )
            logger.info(
                "Collapsed cone effective attenuation mu0 = %g 1/mm, beta = %g 1/mm.", mu0, beta
            )

        # Absolute dose normalization to the machine depth dose
        depth = np.array([self.normalization_depth])
        dd_svd = reference(depth)[0]
        dd_ccc = cc.broad_field_depth_dose(self._kernel, mu0, depth, dz, beta)[0]
        normalization = dd_svd / dd_ccc
        logger.info(
            "Collapsed cone normalization factor %g at %g mm depth.",
            normalization,
            self.normalization_depth,
        )

        params = {
            "mu0": mu0,
            "beta": beta,
            "normalization": normalization,
            "sigma": float(kernel.penumbra) / np.sqrt(8 * np.log(2)),
            "primary_fluence": np.asarray(kernel.primary_fluence, dtype=np.float64),
        }
        self._energy_params[energy] = params
        return params

    def _world_to_array_dirs(self, dirs_world: np.ndarray) -> np.ndarray:
        """
        Convert directions in world coordinates to the numpy axes of the dose grid.

        Parameters
        ----------
        dirs_world : np.ndarray
            ``(3, n_dirs)`` directions in world coordinates (x, y, z).

        Returns
        -------
        np.ndarray
            ``(3, n_dirs)`` directions along the numpy axes (z, y, x) of the dose grid.
        """
        index_dirs = np.asarray(self.dose_grid.direction, dtype=np.float64).T @ dirs_world
        return index_dirs[::-1, :]

    def _tilted_directions(self, rot_mat_t: np.ndarray, axis_bev: np.ndarray) -> np.ndarray:
        """
        Tilt the cones so that their polar axis is along a direction in beam's eye view.

        Parameters
        ----------
        rot_mat_t : np.ndarray
            The beam's ``rot_mat_system_T``.
        axis_bev : np.ndarray
            Polar axis in beam's eye view (the beam axis is ``[0, 1, 0]``).

        Returns
        -------
        np.ndarray
            ``(3, n_dirs)`` cone directions along the numpy axes of the dose grid.
        """
        # BEV coordinates are world coordinates times rot_mat_t, so world = bev @ rot_mat_t.T
        axis_world = np.asarray(axis_bev, dtype=np.float64) @ rot_mat_t.T
        lateral_world = np.array([1.0, 0.0, 0.0]) @ rot_mat_t.T
        dirs_world = cc.tilt_directions(self._local_directions, axis_world, lateral_world)
        return self._world_to_array_dirs(dirs_world)

    def _init_beam(
        self, beam_info: dict, ct: CT, cst: StructureSet, stf: SteeringInformation, i: int
    ) -> dict:
        """
        Initialize a beam for collapsed cone dose calculation.

        Parameters
        ----------
        beam_info : dict
            Beam information struct.
        ct : CT
            The CT object.
        cst : StructureSet
            The structure set object.
        stf : SteeringInformation
            The steering information object.
        i : int
            Index of beam.

        Returns
        -------
        dict
            Updated beam information struct, with the TERMA parameters, the rotation matrix
            and the cone directions tilted along the beam axis in ``beam_info["ccc"]``.
        """
        # The rays select the voxels within the scatter cut-off, since dose is transported
        # there; the half-diagonal of the bixel is added as in the SVD pencil beam engine
        bixel_width = stf.beams[i].bixel_width
        self._effective_lateral_cutoff = max(
            self.geometric_lateral_cutoff, self.scatter_cutoff
        ) + bixel_width / np.sqrt(2)

        beam_info = super()._init_beam(beam_info, ct, cst, stf, i)
        beam_info["effective_lateral_cut_off"] = self._effective_lateral_cutoff

        # Get the kernel
        beamlets = [beamlet for ray in beam_info["beam"]["rays"] for beamlet in ray["beamlets"]]
        energies = np.unique([beamlet["energy"] for beamlet in beamlets])
        if len(energies) > 1:
            raise ValueError("Different energies in one photon beam not supported yet.")
        kernel = cast(PhotonLINAC, self._machine).get_kernel_by_energy(energies[0])

        rot_mat_t = np.asarray(to_numpy(beam_info["rot_mat_system_T"]), dtype=np.float64)
        beam_info["ccc"] = {
            **self._get_energy_params(kernel),
            "rot_mat_t": rot_mat_t,
            "beam_dirs": self._tilted_directions(rot_mat_t, np.array([0.0, 1.0, 0.0])),
        }
        return beam_info

    def _init_ray(self, beam_info: dict, j: int) -> dict:
        """
        Initialize a ray and its cone directions.

        Parameters
        ----------
        beam_info : dict
            Beam information struct. Keys used here:
            - ``ccc``: the TERMA parameters and cone directions (see ``_init_beam``)
        j : int
            The index of the current ray.

        Returns
        -------
        dict
            The initialized ray, with ``ccc`` and the cone directions ``ccc_dirs`` (along
            the beam axis, or along the ray for ``kernel_tilt="ray"``).
        """
        ray = super()._init_ray(beam_info, j)
        params = beam_info["ccc"]
        ray["ccc"] = params
        if self.kernel_tilt == "beam":
            ray["ccc_dirs"] = params["beam_dirs"]
        else:
            ray_dir = to_numpy(ray["target_point_bev"]) - to_numpy(ray["source_point_bev"])
            ray["ccc_dirs"] = self._tilted_directions(params["rot_mat_t"], ray_dir)
        return ray

    def _apply_scenario_to_ray(self, ray: dict, scenario: Scenario) -> dict:
        """
        Select a scenario's CT phase from a ray and apply its range shift.

        Parameters
        ----------
        ray : dict
            The ray data with one entry per CT phase for the geometric quantities.
        scenario : Scenario
            The scenario.

        Returns
        -------
        dict
            The ray of the scenario (see the base class), which also records its CT phase
            in ``ct_phase`` to select the density cube of the transport.
        """
        scen_ray = super()._apply_scenario_to_ray(ray, scenario)
        scen_ray["ct_phase"] = scenario.ct_phase
        return scen_ray

    def _as_cx(self, a: Array, dtype: Any = np.float64) -> Array:
        """
        Move an array to the device of TERMA and transport.

        Parameters
        ----------
        a : Array
            Array of any supported backend.
        dtype : Any
            NumPy dtype of the result.

        Returns
        -------
        Array
            A NumPy array, or a CuPy array if the calculation runs on the GPU. CuPy arrays
            and CUDA torch tensors are not copied to the host.
        """
        if not self._on_gpu:
            return np.asarray(to_numpy(a), dtype=dtype)
        if cc.is_cupy_array(a):
            return a.astype(dtype, copy=False)
        if array_api_compat.is_torch_array(a) and a.is_cuda:
            return self._cx.from_dlpack(a.contiguous()).astype(dtype, copy=False)
        return self._cx.asarray(to_numpy(a), dtype=dtype)

    def _ray_terma(self, ray: dict) -> Array:
        """
        Compute the TERMA of a bixel of the (scenario) ray on the ray's voxels.

        Parameters
        ----------
        ray : dict
            Ray information struct (see ``_init_ray``). Keys used here: ``ccc``, ``sad``,
            ``bixel_width``, ``ray_pos_bev``, ``rad_depths``, ``geo_depths`` and
            ``iso_lat_dists``.

        Returns
        -------
        Array
            TERMA per unit bixel weight on ``ray["ix"]``.
        """
        params = ray["ccc"]
        return cc.calc_terma(
            float(ray["sad"]),
            params["mu0"],
            params["sigma"],
            float(ray["bixel_width"]),
            params["primary_fluence"],
            np.asarray(to_numpy(ray["ray_pos_bev"]), dtype=np.float64),
            self._as_cx(ray["rad_depths"]),
            self._as_cx(ray["geo_depths"]),
            self._as_cx(ray["iso_lat_dists"]),
            params["beta"],
        )

    def _source_box(self, ix: Array, terma: Array) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the transport sub-volume around the relevant TERMA.

        Parameters
        ----------
        ix : Array
            Dose grid indices of the TERMA values.
        terma : Array
            TERMA values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Lowest and highest array subscripts (z, y, x) of the sub-volume, which extends
            ``scatter_cutoff`` around the voxels with more than 1e-6 of the maximum TERMA.
        """
        cx = self._cx
        is_source = terma > 1e-6 * terma.max()
        sub = cx.stack(cx.unravel_index(ix[is_source], self._dose_dims), axis=1)
        sub_lo = to_numpy(sub.min(axis=0))
        sub_hi = to_numpy(sub.max(axis=0))
        margin = np.ceil(self.scatter_cutoff / self._dose_res).astype(np.int64)
        box_lo = np.maximum(sub_lo - margin, 0)
        box_hi = np.minimum(sub_hi + margin, np.asarray(self._dose_dims) - 1)
        return box_lo, box_hi

    def _box_indices(self, ix: Array, box_lo: np.ndarray, box_hi: np.ndarray) -> tuple:
        """
        Select the dose grid indices inside a sub-volume.

        Parameters
        ----------
        ix : Array
            Dose grid indices.
        box_lo, box_hi : np.ndarray
            Lowest and highest array subscripts of the sub-volume.

        Returns
        -------
        tuple
            Mask of the indices inside the sub-volume and their subscripts in it.
        """
        cx = self._cx
        sub = cx.stack(cx.unravel_index(ix, self._dose_dims), axis=1)
        lo = cx.asarray(box_lo)
        in_box = cx.all((sub >= lo) & (sub <= cx.asarray(box_hi)), axis=1)
        local = sub[in_box] - lo
        return in_box, (local[:, 0], local[:, 1], local[:, 2])

    def _transport_on_box(
        self,
        source_ix: Array,
        terma: Array,
        box_lo: np.ndarray,
        box_hi: np.ndarray,
        ct_phase: int,
        dirs: np.ndarray,
        out_ix: Array,
        normalization: float,
    ) -> Array:
        """
        Transport TERMA within a sub-volume and get the absolute dose on given voxels.

        Parameters
        ----------
        source_ix : Array
            Dose grid indices of the TERMA values.
        terma : Array
            TERMA values.
        box_lo, box_hi : np.ndarray
            Lowest and highest array subscripts of the sub-volume.
        ct_phase : int
            CT phase of the density cube.
        dirs : np.ndarray
            ``(3, n_dirs)`` cone directions along the numpy axes of the dose grid.
        out_ix : Array
            Dose grid indices to return the dose on.
        normalization : float
            Absolute dose normalization factor.

        Returns
        -------
        Array
            Dose on ``out_ix``, zero outside the sub-volume.
        """
        cx = self._cx
        box = tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(box_lo, box_hi))
        rho_box = self._density_cubes[ct_phase][box]
        terma_box = cx.zeros(rho_box.shape)
        in_source, local = self._box_indices(source_ix, box_lo, box_hi)
        terma_box[local] = terma[in_source]

        # The transport lattice is anchored to the dose grid through the origin of the box,
        # so that the doses of different sub-volumes are consistent
        dose_box = cc.transport(
            terma_box,
            rho_box,
            self._dose_res,
            dirs,
            self._direction_weights,
            self._direction_att,
            origin=box_lo,
            use_numba=self.use_numba,
            num_threads=self.num_threads,
        )

        in_out, local = self._box_indices(out_ix, box_lo, box_hi)
        dose = cx.zeros(out_ix.size)
        dose[in_out] = normalization * dose_box[local]

        # Check if we have valid dose values
        if not self.ignore_invalid_values and bool(cx.any(~cx.isfinite(dose) | (dose < 0))):
            raise ValueError("Invalid numerical values in collapsed cone dose calculation.")
        return dose

    def _bixel_dose(self, ray: dict) -> Array:
        """
        Compute the collapsed cone dose of a bixel on the voxels of its ray.

        Parameters
        ----------
        ray : dict
            Ray information struct of the scenario (see ``_apply_scenario_to_ray``).

        Returns
        -------
        Array
            Dose per unit bixel weight on ``ray["ix"]``.
        """
        ix = self._as_cx(ray["ix"], dtype=np.int64)
        terma = self._ray_terma(ray)
        if not bool(self._cx.any(terma > 0)):
            return self._cx.zeros(ix.size)
        box_lo, box_hi = self._source_box(ix, terma)
        return self._transport_on_box(
            ix,
            terma,
            box_lo,
            box_hi,
            ray["ct_phase"],
            ray["ccc_dirs"],
            ix,
            ray["ccc"]["normalization"],
        )

    def _compute_bixel(self, curr_ray: dict, k: int) -> dict[str, Any]:
        """
        Collapsed cone dose calculation for a single bixel.

        The TERMA of the bixel is transported on the sub-volume around it and the dose is
        returned on the voxels of the ray. When ``enable_dij_sampling`` is True, the low-dose
        tail is resampled (see ``_sample_dij``) to sparsify the dose influence matrix.

        Parameters
        ----------
        curr_ray : dict
            Current ray information struct as prepared by ``_apply_scenario_to_ray``.
        k : int
            Index of the beamlet on the ray.

        Returns
        -------
        dict
            The computed bixel dose. Keys:
            - ``physical_dose``: the physical dose values at voxels with indices ``ix``
            - ``weight``: the weight of the beamlet corresponding to the bixel
            - ``ix``: the dose grid indices of the voxels (with ``enable_dij_sampling``,
              only the kept voxels)
        """
        ix = np.asarray(to_numpy(curr_ray["ix"]), dtype=np.int64)
        dose = to_numpy(self._bixel_dose(curr_ray), dtype=np.float32)
        bixel = {
            "weight": curr_ray["beamlets"][k]["weight"],
            "ix": ix,
            "physical_dose": dose,
        }
        if self.enable_dij_sampling and not self._calc_dose_direct:
            bixel["ix"], bixel["physical_dose"] = self._sample_dij(
                ix,
                bixel["physical_dose"],
                np.asarray(to_numpy(curr_ray["rad_depths"])),
                np.asarray(to_numpy(curr_ray["radial_dist_sq"])),
                bixel_width=curr_ray["bixel_width"],
            )
        return bixel

    def _calc_dose(self, ct: CT, cst: StructureSet, stf: SteeringInformation) -> dict:
        """
        Calculate the dose.

        Forward calculations with the cones tilted along the beam axis transport the
        weighted TERMA of all bixels of a beam at once (see ``_calc_beam_forward``), all
        other calculations go bixel by bixel as in the pencil beam engines.

        Parameters
        ----------
        ct : CT
            The CT object.
        cst : StructureSet
            The structure set object.
        stf : SteeringInformation
            The steering information object.

        Returns
        -------
        dict
            The dose influence matrix dictionary.
        """
        if not (self._calc_dose_direct and self.beam_forward and self.kernel_tilt == "beam"):
            return super()._calc_dose(ct, cst, stf)

        # Initialize
        dij = self._init_dose_calc(ct, cst, stf)

        xp = self.xp
        self._vox_world_coords = xp.asarray(
            self._vox_world_coords, dtype=xp.float32, device=self.device
        )
        self._vox_world_coords_dose_grid = xp.asarray(
            self._vox_world_coords_dose_grid, dtype=xp.float32, device=self.device
        )
        self._vdose_grid = to_namespace(xp, self._vdose_grid, device=self.device)

        groups = self._geometry_groups()
        num_computed = sum(len(scen_ix) for _, scen_ix in groups)
        scenario_progress = (
            self.progress("Scenarios", total=num_computed, console=True, unit="scen")
            if num_computed > 1
            else nullcontext(None)
        )
        with scenario_progress as scenarios_done, logging_redirect_tqdm():
            for setup_shift, scen_ix in groups:
                scen_stf = self._shift_stf(stf, setup_shift)
                self._init_geometry_group()

                # Loop over all beams
                for i in self.track(range(dij["num_of_beams"]), name="Beam", unit="b"):
                    t = time.time()
                    curr_beam = self._init_beam(dij, ct, cst, scen_stf, i)
                    self._calc_beam_forward(dij, curr_beam, scen_ix, i)
                    logger.info("Beam %d computed in %f seconds.", i + 1, time.time() - t)

                    # Free GPU memory from curr_beam
                    curr_beam = None
                    free_gpu_memory(self.xp)

                if scenarios_done is not None:
                    scenarios_done.advance(len(scen_ix))

        return self._finalize_dose(dij)

    def _calc_beam_forward(self, dij: dict, curr_beam: dict, scen_ix: list[int], i: int) -> None:
        """
        Compute the forward dose of a beam with a single transport per scenario.

        The TERMA of all bixels, weighted with the bixel weights, is accumulated on the dose
        grid together with the union of their transport sub-volumes and transported once.
        The dose is stored on the voxels of all rays of the beam.

        Parameters
        ----------
        dij : dict
            The dose influence matrix dictionary (forward dose, one column per beam).
        curr_beam : dict
            Beam information struct (see ``_init_beam``).
        scen_ix : list[int]
            Indices of the computed scenarios of the geometry group.
        i : int
            Index of the beam.
        """
        cx = self._cx
        num_vox = self.dose_grid.num_voxels
        acc = {}

        # Ray calculation
        for j in range(curr_beam["beam"]["num_of_rays"]):
            curr_ray = self._init_ray(curr_beam, j)
            weight = sum(
                float(curr_ray["beamlets"][k]["weight"]) for k in range(curr_ray["num_of_bixels"])
            )
            for s in scen_ix:
                scen_ray = self._apply_scenario_to_ray(curr_ray, self._scenarios[s])
                if s not in acc:
                    acc[s] = {
                        "terma": cx.zeros(num_vox),
                        "mask": cx.zeros(num_vox, dtype=bool),
                        "ct_phase": scen_ray["ct_phase"],
                        "box_lo": None,
                        "box_hi": None,
                    }
                ix = self._as_cx(scen_ray["ix"], dtype=np.int64)
                acc[s]["mask"][ix] = True
                if ix.size == 0 or weight == 0:
                    continue
                terma = self._ray_terma(scen_ray)
                if not bool(cx.any(terma > 0)):
                    continue
                acc[s]["terma"][ix] += weight * terma

                # Union of the sub-volumes of the single bixels
                lo, hi = self._source_box(ix, terma)
                if acc[s]["box_lo"] is None:
                    acc[s]["box_lo"], acc[s]["box_hi"] = lo, hi
                else:
                    acc[s]["box_lo"] = np.minimum(acc[s]["box_lo"], lo)
                    acc[s]["box_hi"] = np.maximum(acc[s]["box_hi"], hi)

        # One transport per scenario
        dirs = curr_beam["ccc"]["beam_dirs"]
        normalization = curr_beam["ccc"]["normalization"]
        for s, a in acc.items():
            source_ix = cx.flatnonzero(a["terma"] > 0)
            if a["box_lo"] is None or source_ix.size == 0:
                continue
            out_ix = cx.flatnonzero(a["mask"])
            dose = self._transport_on_box(
                source_ix,
                a["terma"][source_ix],
                a["box_lo"],
                a["box_hi"],
                a["ct_phase"],
                dirs,
                out_ix,
                normalization,
            )
            dij["physical_dose"][s][to_numpy(out_ix), i] += to_numpy(dose, dtype=np.float32)

        # Bookkeeping of the beam number
        dij["beam_num"][i] = i
        dij["ray_num"][i] = i
        dij["bixel_num"][i] = i
