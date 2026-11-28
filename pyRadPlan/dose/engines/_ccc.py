from contextlib import nullcontext
from typing import Annotated, Any, ClassVar, Literal, Optional, TypedDict, Union, cast
import logging
import time

import numpy as np
import SimpleITK as sitk
from pydantic import Field
from tqdm.contrib.logging import logging_redirect_tqdm

from pyRadPlan.core import resample_image
from pyRadPlan.core.xp_utils import free_gpu_memory, scatter, to_namespace, to_numpy
from pyRadPlan.core.xp_utils.typing import Array
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.machines import PhotonCollapsedConeKernel, PhotonLINAC, PhotonSVDKernel
from pyRadPlan.machines.photons import load_collapsed_cone_kernel
from pyRadPlan.machines.photons._collapsed_cone_fit import load_kernel
from pyRadPlan.plan import PhotonPlan
from pyRadPlan.scenarios import Scenario
from pyRadPlan.stf import FieldShape, SteeringInformation

from ._base import ct_phase_images
from ._base_pencilbeam import PencilBeamEngineAbstract
from ._svdpb import DijSamplingConfig, PhotonPencilBeamSVDEngine
from ._collapsed_cone import (
    analytical_depth_dose,
    broad_field_depth_dose,
    calc_terma,
    cone_directions,
    fit_hardening,
    tilt_directions,
)
from .kernels._collapsed_cone_transport import transport

logger = logging.getLogger(__name__)


class CollapsedConeEnergyParams(TypedDict):
    """TERMA parameters and absolute normalization of a beam energy."""

    mu0: float
    beta: float
    normalization: float
    sigma: float
    primary_fluence: np.ndarray


class PhotonCollapsedConeEngine(PencilBeamEngineAbstract):
    """
    Photon dose calculation with collapsed cone convolution/superposition.

    Port of matRad's ``matRad_PhotonCollapsedConeEngine``. For each bixel the TERMA is
    computed on the dose grid voxels selected by the ray tracer and transported with a set
    of collapsed cones [1]_ of a polyenergetic energy deposition kernel, whose polar axis is
    tilted along the beam axis or the bixel's ray. The absolute dose is normalized to the
    analytical SVD pencil beam depth dose [2]_ of the machine at a reference depth in water.

    TERMA and transport use Array API operations only and run in the engine's array
    namespace and device (NumPy, CuPy, PyTorch), see ``settings.xp``.

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
    hardening_correction: Annotated[
        bool, Field(description="Fit a depth dependent effective attenuation of the TERMA")
    ] = True
    num_scatter_exponentials: Annotated[
        Literal[1, 2], Field(description="Number of exponential terms of the scatter kernel")
    ] = 1
    kernel_tilt: Annotated[
        Literal["beam", "ray"], Field(description="Polar axis of the cones: beam axis or ray")
    ] = "beam"
    beam_forward: Annotated[
        bool, Field(description="Forward dose with one transport per beam")
    ] = True
    enable_dij_sampling: bool = True
    dij_sampling: DijSamplingConfig = DijSamplingConfig(
        rel_dose_threshold=0.01, lat_cut_off=20, type="radius", delta_rad_depth=5
    )
    random_seed: int = 0

    # Same lateral dij sampling as the SVD pencil beam engine
    _sample_dij = PhotonPencilBeamSVDEngine._sample_dij

    def __init__(self, pln: PhotonPlan = None):
        self._kernel: Optional[PhotonCollapsedConeKernel] = None  # fitted kernel
        self._local_directions: Optional[np.ndarray] = None  # cone directions, polar axis z
        self._direction_weights: Optional[np.ndarray] = None  # kernel weights per direction
        self._direction_att: Optional[np.ndarray] = None  # attenuation per direction
        self._density_cubes: list[Array] = []  # electron density on the dose grid per phase
        self._energy_params: dict[float, CollapsedConeEnergyParams] = {}
        self._dose_dims: Optional[tuple[int, ...]] = None  # dose grid shape (z, y, x)
        self._dose_res: Optional[np.ndarray] = None  # voxel size along (z, y, x) [mm]
        super().__init__(pln)
        self._rng = np.random.default_rng(self.random_seed)

    @staticmethod
    def is_available(pln: Union[PhotonPlan, str, None], machine: Any = None) -> tuple[bool, str]:
        """
        Check whether the engine can compute the plan (with the machine, if given).

        The engine factory calls this with ``machine=None`` and a plan or a radiation mode; the
        kernel file is then checked, the energies of the machine only when a machine is given.

        Parameters
        ----------
        pln : PhotonPlan or str, optional
            The plan, whose ``prop_dose_calc["kernel_file"]`` selects the kernel, or a
            radiation mode.
        machine : Any, optional
            The machine.

        Returns
        -------
        tuple[bool, str]
            Availability and a message with the reason or the warnings.
        """
        kernel_file = None
        prop_dose_calc = getattr(pln, "prop_dose_calc", None)
        if isinstance(prop_dose_calc, dict):
            kernel_file = prop_dose_calc.get("kernel_file")
        try:
            stored = load_collapsed_cone_kernel(kernel_file)
        except (FileNotFoundError, ValueError) as err:
            return False, str(err)
        if machine is None:
            return True, ""
        if not isinstance(machine, PhotonLINAC) or not machine.pb_kernels:
            return False, "Machine has no photon pencil beam data."

        # The kernel is computed for one nominal energy, which every beam energy has to match
        msgs = []
        for kernel in machine.pb_kernels.values():
            ok, energy_msg = stored.check_energy(kernel.energy)
            if not ok:
                return False, energy_msg
            if energy_msg:
                msgs.append(energy_msg)
        return True, " ".join(msgs)

    def _init_dose_calc(
        self, ct: CT, cst: StructureSet, stf: SteeringInformation
    ) -> dict[str, Any]:
        """
        Initialize the dose calculation.

        Loads (and if needed refits) the kernel, sets up the cone directions and
        interpolates the relative electron densities of all CT phases on the dose grid.

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
        self._kernel = load_kernel(
            self.kernel_file, self.num_polar_angles, self.num_scatter_exponentials
        )
        (
            self._local_directions,
            self._direction_weights,
            self._direction_att,
        ) = cone_directions(self._kernel, self.num_azimuthal_angles)

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
            self._density_cubes.append(self.xp.asarray(rho, device=self.device))
        self._dose_dims = tuple(int(n) for n in self._density_cubes[0].shape)

        # Voxel size along the numpy axes (z, y, x)
        res = self.dose_grid.resolution
        self._dose_res = np.array([res["z"], res["y"], res["x"]], dtype=np.float64)

        # Hardening and normalization are fitted per beam energy when the beam is initialized
        self._energy_params = {}

        # Initialize random number generator for the dij sampling
        self._rng = np.random.default_rng(self.random_seed)

        return dij

    def _get_energy_params(self, kernel: PhotonSVDKernel) -> CollapsedConeEnergyParams:
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

        ok, msg = self._kernel.check_energy(energy)
        if not ok:
            raise ValueError(msg)
        if msg:
            logger.warning(msg)

        # Reference depth dose from the kernels at the SSD closest to the SAD
        sad = float(self._machine.sad)
        ssd_ix = int(np.argmin(np.abs(np.asarray(kernel.kernel_ssds) - sad)))

        def reference(depths: np.ndarray) -> np.ndarray:
            return analytical_depth_dose(
                kernel.kernel_data[ssd_ix],
                kernel.kernel_pos,
                kernel.kernel_betas,
                kernel.m,
                depths,
            )

        dz = float(np.min(self._dose_res))
        mu0, beta = float(kernel.m), 0.0
        if self.hardening_correction:
            mu0, beta = fit_hardening(
                self._kernel, reference, float(kernel.m), dz, self.normalization_depth
            )
            logger.info(
                "Collapsed cone effective attenuation mu0 = %g 1/mm, beta = %g 1/mm.", mu0, beta
            )

        # Absolute dose normalization to the machine depth dose
        depth = np.array([self.normalization_depth])
        dd_svd = reference(depth)[0]
        dd_ccc = broad_field_depth_dose(self._kernel, mu0, depth, dz, beta)[0]
        normalization = dd_svd / dd_ccc
        logger.info(
            "Collapsed cone normalization factor %g at %g mm depth.",
            normalization,
            self.normalization_depth,
        )

        params: CollapsedConeEnergyParams = {
            "mu0": mu0,
            "beta": beta,
            "normalization": float(normalization),
            "sigma": float(kernel.penumbra) / np.sqrt(8 * np.log(2)),
            "primary_fluence": np.asarray(kernel.primary_fluence, dtype=np.float64),
        }
        self._energy_params[energy] = params
        return params

    def _world_to_array_dirs(self, dirs_world: np.ndarray) -> np.ndarray:
        """Convert ``(3, n)`` world directions to the numpy axes (z, y, x) of the dose grid."""
        index_dirs = np.asarray(self.dose_grid.direction, dtype=np.float64).T @ dirs_world
        return index_dirs[::-1, :]

    def _tilted_directions(self, rot_mat_t: np.ndarray, axis_bev: np.ndarray) -> np.ndarray:
        """Cone directions along the numpy axes with the polar axis along ``axis_bev``."""
        # BEV coordinates are world coordinates times rot_mat_t, so world = bev @ rot_mat_t.T
        axis_world = np.asarray(axis_bev, dtype=np.float64) @ rot_mat_t.T
        lateral_world = np.array([1.0, 0.0, 0.0]) @ rot_mat_t.T
        dirs_world = tilt_directions(self._local_directions, axis_world, lateral_world)
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

    def _as_xp(self, a: Array, dtype: Any = None) -> Array:
        """Cast an array of the engine's namespace to ``dtype`` (default: float64)."""
        xp = self.xp
        return xp.astype(a, xp.float64 if dtype is None else dtype, copy=False)

    def _ray_terma(self, ray: dict) -> Array:
        """TERMA per unit bixel weight on the voxels ``ray["ix"]`` of a (scenario) ray."""
        params = ray["ccc"]
        return calc_terma(
            float(ray["sad"]),
            params["mu0"],
            params["sigma"],
            float(ray["bixel_width"]),
            params["primary_fluence"],
            np.asarray(to_numpy(ray["ray_pos_bev"]), dtype=np.float64),
            self._as_xp(ray["rad_depths"]),
            self._as_xp(ray["geo_depths"]),
            self._as_xp(ray["iso_lat_dists"]),
            params["beta"],
        )

    def _subscripts(self, ix: Array) -> tuple[Array, Array, Array]:
        """Array subscripts (z, y, x) of linear C order dose grid indices."""
        _, ny, nx = self._dose_dims
        return ix // (ny * nx), (ix // nx) % ny, ix % nx

    def _source_box(self, ix: Array, terma: Array) -> tuple[np.ndarray, np.ndarray]:
        """Sub-volume within ``scatter_cutoff`` of the voxels with > 1e-6 of the maximum TERMA."""
        xp = self.xp
        is_source = terma > 1e-6 * xp.max(terma)
        num_vox = int(np.prod(self._dose_dims))
        subs = self._subscripts(ix)
        sub_lo = np.array([int(xp.min(xp.where(is_source, a, num_vox))) for a in subs])
        sub_hi = np.array([int(xp.max(xp.where(is_source, a, -1))) for a in subs])
        margin = np.ceil(self.scatter_cutoff / self._dose_res).astype(np.int64)
        box_lo = np.maximum(sub_lo - margin, 0)
        box_hi = np.minimum(sub_hi + margin, np.asarray(self._dose_dims) - 1)
        return box_lo, box_hi

    def _box_local(self, ix: Array, box_lo: np.ndarray, box_hi: np.ndarray) -> tuple:
        """Mask of the indices inside a sub-volume and their linear indices in it (0 outside)."""
        xp = self.xp
        subs = self._subscripts(ix)
        inside = xp.ones(ix.shape, dtype=xp.bool, device=self.device)
        local = xp.zeros(ix.shape, dtype=ix.dtype, device=self.device)
        for a, lo, hi in zip(subs, box_lo.tolist(), box_hi.tolist()):
            inside = inside & (a >= lo) & (a <= hi)
            local = local * (hi - lo + 1) + (a - lo)
        return inside, xp.where(inside, local, 0)

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
        """Transport TERMA within a sub-volume; absolute dose on ``out_ix``, 0 outside it."""
        xp = self.xp
        box = tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(box_lo, box_hi))
        rho_box = self._density_cubes[ct_phase][box]
        num_box = int(np.prod(rho_box.shape))

        # TERMA outside the sub-volume is written to an extra element, which is dropped
        inside, local = self._box_local(source_ix, box_lo, box_hi)
        terma_box = xp.zeros(num_box + 1, dtype=xp.float64, device=self.device)
        terma_box = scatter(terma_box, xp.where(inside, local, num_box), terma)
        terma_box = xp.reshape(terma_box[:num_box], rho_box.shape)

        # The transport lattice is anchored to the dose grid through the origin of the box,
        # so that the doses of different sub-volumes are consistent
        dose_box = transport(
            terma_box,
            rho_box,
            self._dose_res,
            dirs,
            self._direction_weights,
            self._direction_att,
            origin=box_lo,
        )

        inside, local = self._box_local(out_ix, box_lo, box_hi)
        dose = xp.take(xp.reshape(dose_box, (-1,)), local)
        dose = normalization * xp.where(inside, dose, 0.0)
        return dose

    def _bixel_dose(self, ray: dict) -> Array:
        """Dose per unit bixel weight on the voxels of the ray."""
        xp = self.xp
        ix = self._as_xp(ray["ix"], dtype=xp.int64)
        terma = self._ray_terma(ray)
        if not bool(xp.any(terma > 0)):
            return xp.zeros(ix.shape, dtype=xp.float64, device=self.device)
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
        xp = self.xp
        dose = xp.astype(self._bixel_dose(curr_ray), xp.float32)
        bixel = {
            "weight": curr_ray["beamlets"][k]["weight"],
            "ix": curr_ray["ix"],
            "physical_dose": dose,
        }
        if self.enable_dij_sampling and not self._calc_dose_direct:
            bixel["ix"], bixel["physical_dose"] = self._sample_dij(
                curr_ray["ix"],
                dose,
                curr_ray["rad_depths"],
                curr_ray["radial_dist_sq"],
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
        xp = self.xp
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
                        "terma": xp.zeros(num_vox, dtype=xp.float64, device=self.device),
                        "mask": xp.zeros(num_vox, dtype=xp.bool, device=self.device),
                        "ct_phase": scen_ray["ct_phase"],
                        "box_lo": None,
                        "box_hi": None,
                    }
                ix = self._as_xp(scen_ray["ix"], dtype=xp.int64)
                if ix.shape[0] == 0:
                    continue
                acc[s]["mask"] = scatter(acc[s]["mask"], ix, True)
                if weight == 0:
                    continue
                terma = self._ray_terma(scen_ray)
                if not bool(xp.any(terma > 0)):
                    continue
                # The voxels of a ray are unique, so gather and scatter accumulate correctly
                acc[s]["terma"] = scatter(
                    acc[s]["terma"], ix, xp.take(acc[s]["terma"], ix) + weight * terma
                )

                # Union of the sub-volumes of the single bixels
                lo, hi = self._source_box(ix, terma)
                if acc[s]["box_lo"] is None:
                    acc[s]["box_lo"], acc[s]["box_hi"] = lo, hi
                else:
                    acc[s]["box_lo"] = np.minimum(acc[s]["box_lo"], lo)
                    acc[s]["box_hi"] = np.maximum(acc[s]["box_hi"], hi)

        # One transport per scenario on the union of the sub-volumes
        dirs = curr_beam["ccc"]["beam_dirs"]
        normalization = curr_beam["ccc"]["normalization"]
        for s, a in acc.items():
            if a["box_lo"] is None:
                continue
            box = tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(a["box_lo"], a["box_hi"]))
            dose_box = normalization * transport(
                xp.reshape(a["terma"], self._dose_dims)[box],
                self._density_cubes[a["ct_phase"]][box],
                self._dose_res,
                dirs,
                self._direction_weights,
                self._direction_att,
                origin=a["box_lo"],
            )
            dose = np.zeros(self._dose_dims, dtype=np.float32)
            dose[box] = to_numpy(dose_box, dtype=np.float32)
            out_ix = np.flatnonzero(to_numpy(a["mask"]))
            dij["physical_dose"][s][out_ix, i] += dose.reshape(-1)[out_ix]

        # Bookkeeping of the beam number
        dij["beam_num"][i] = i
        dij["ray_num"][i] = i
        dij["bixel_num"][i] = i
