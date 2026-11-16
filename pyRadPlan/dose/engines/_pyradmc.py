"""Monte Carlo photon dose engine using pyRadMC.

pyRadMC is a GPU-accelerated (NVIDIA warp) photon/electron Monte Carlo engine
for beamlet-resolved dose influence calculation.

Notes
-----
Frame conventions bridged by this adapter:

- pyRadPlan works in mm on the LPS world grid; Dij rows are numpy (C-order)
  linear indices over the ``(z, y, x)``-shaped dose cube (x fastest).
- pyRadMC works in cm with ``(x, y, z)`` axes, grid origin at the lower voxel
  *corner*, and flat C-order voxel indexing over ``(nx, ny, nz)`` (z fastest).

pyRadMC scores absolute dose in Gy *per emitted history* of the respective
beamlet. Mapping histories to monitor units requires a machine-specific output
calibration which is not modelled yet: the engine scales the Dij columns by the
``num_particles_per_mu`` placeholder (default 1e10 histories/MU, chosen so Dij
entries and optimized weights are of order one for typical setups), so columns
are "Gy per MU" only up to that calibration.

The primary photon spectrum comes from the machine's Monte Carlo beam model
(``PhotonLINAC.mc_beam_model``), selected per beam by the stf beamlet energy;
the ``spectrum_beam`` engine option overrides it with a named Ali & Rogers
(2012) analytic MV spectrum.

The machine beam model also carries the beam-limiting device geometry (jaws,
MLC). Dose *influence* (Dij) calculation deliberately uses the plain divergent
beamlet source without any jaw/MLC model: beamlets are the optimization
degrees of freedom, and the aperture belongs to the delivered plan. The
devices enter only through the opt-in treatment-head pre-solve forward path
(``forward_mode='head_presolve'``), which builds a conformal static aperture
from the stf rays and includes collimator transmission and scatter (but
ignores per-bixel fluence modulation).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Callable, ClassVar, Literal, Optional, Sequence

import numpy as np
from pydantic import Field
from scipy import sparse

from ...core import Grid, ProgressHandle
from ...ct import CT
from ...cst import StructureSet
from ...geometry import lps
from ...machines.base import Jaw as JawSetting, MLC as MLCSetting  # noqa: N811
from ...machines.photons import PhotonLINACBeamModelMC
from ...stf import Beam, FieldShapeAsBLD, FieldShapeComposite, SteeringInformation

from ._base_montecarlo import MonteCarloEngineAbstract

if TYPE_CHECKING:
    from pyradmc.progress import ProgressEvent

logger = logging.getLogger(__name__)

_MM_PER_CM = 10.0

# Compiled tabulated cross-section data, keyed by (e_max, pcut, ecut). Compiling
# from the EPICS libraries parses ~120 MB of text; reuse within the process.
_TABULATED_DATA_CACHE: dict[tuple, Any] = {}

# Validated (msc_model, step_energy_fraction) pairs for pyRadMC's electron
# transport. The two are coupled and only these pairings are validated
# (R50/detour, PDD gamma, cross-backend chi2), so the engine exposes them as
# one preset: under 'gaussian' longer substeps are re-bound by an internal
# angular validity cap and buy nothing; under 'gs' the cap is lifted and the
# longer substep is part of the validated configuration.
_MSC_PRESETS: dict[str, dict[str, Any]] = {
    "gaussian": {"msc_model": "gaussian", "step_energy_fraction": 0.05},
    "gs": {"msc_model": "gs", "step_energy_fraction": 0.20},
}


def transport_grid_from_ct(ct: CT, water_only: bool = False):
    """Convert the CT into a pyRadMC transport ``VoxelGrid``.

    The engine frame is the LPS world frame scaled to cm: pyRadMC's CT adapter
    keeps the image's physical coordinates (converting mm to cm and voxel-center
    to lower-corner origin), so a world point maps into the engine frame by
    dividing by 10.

    Parameters
    ----------
    ct : CT
        The CT with ``cube_hu`` as a SimpleITK image (axis-aligned; oblique
        orientations raise inside pyRadMC).
    water_only : bool
        Map all HU to water composition (density still follows the HU ramp).
        Required for the analytic (water-only) cross-section model.
    """
    from pyradmc import WATER
    from pyradmc.adapters.ct import (
        DEFAULT_CALIBRATION,
        HounsfieldCalibration,
        grid_from_image,
    )

    calibration = DEFAULT_CALIBRATION
    if water_only:
        calibration = HounsfieldCalibration(
            density_hu=DEFAULT_CALIBRATION.density_hu,
            density_values=DEFAULT_CALIBRATION.density_values,
            material_thresholds=(),
            material_indices=(WATER,),
        )

    return grid_from_image(ct.cube_hu, calibration=calibration)


def scoring_grid_from_dose_grid(dose_grid: Grid, transport_grid):
    """Build a pyRadMC ``ScoringGrid`` matching the pyRadPlan dose grid.

    The dose grid origin is the *center* of voxel (0, 0, 0) in mm; the scoring
    grid origin is the lower *corner* in cm. Per-voxel masses are rebinned from
    the transport grid by exact voxel overlap inside pyRadMC.
    """
    from pyradmc import ScoringGrid

    if not np.allclose(dose_grid.direction, np.eye(3), atol=1e-6):
        raise ValueError(
            "pyRadMC engine only supports dose grids with identity direction cosines."
        )

    resolution = dose_grid.resolution_vector
    origin_corner_mm = dose_grid.origin - 0.5 * resolution

    return ScoringGrid.rebin(
        transport_grid,
        shape=tuple(int(d) for d in dose_grid.dimensions),
        spacing=tuple(float(r) / _MM_PER_CM for r in resolution),
        origin=tuple(float(o) / _MM_PER_CM for o in origin_corner_mm),
    )


def permute_rows_engine_to_grid(matrix: sparse.csc_array, dimensions) -> sparse.csc_array:
    """Remap Dij rows from pyRadMC voxel order to pyRadPlan linear indices.

    pyRadMC rows are C-order over the scoring grid ``(nx, ny, nz)`` (z fastest);
    pyRadPlan rows ravel the sitk-shaped ``(nz, ny, nx)`` dose cube (x fastest).

    Parameters
    ----------
    matrix : scipy.sparse.csc_array
        Sparse matrix with rows in pyRadMC voxel order.
    dimensions : tuple
        The dose grid dimensions ``(nx, ny, nz)``.
    """
    nx, ny, nz = (int(d) for d in dimensions)
    ix, iy, iz = np.unravel_index(matrix.indices, (nx, ny, nz))
    new_indices = np.ravel_multi_index((iz, iy, ix), (nz, ny, nx))

    permuted = sparse.csc_array((matrix.data, new_indices, matrix.indptr), shape=matrix.shape)
    permuted.sort_indices()
    return permuted


def spectrum_from_model(model) -> Any:
    """Build a pyRadMC ``Spectrum`` from a machine spectrum model.

    Supports the three machine parameterizations: the Ali & Rogers (2012)
    analytic MV form (evaluated by pyRadMC), an explicit histogram, and a
    monoenergetic beam (represented as one narrow histogram bin).
    """
    from pyradmc.geometry.spectrum import AliRogersMV, Spectrum, ali_rogers_mv

    kind = getattr(model, "kind", None)
    if kind == "analytic_mv":
        return ali_rogers_mv(
            AliRogersMV(
                e_e=model.fitted_electron_energy,
                c1=model.c1,
                c2=model.c2,
                c3=model.c3,
                c4=model.c4,
            ),
            e_min=model.energy_min,
            n_bins=model.num_bins,
        )
    if kind == "histogram":
        return Spectrum(model.edges, model.weights, convention=model.convention)
    if kind == "monoenergetic":
        return Spectrum((model.energy * (1.0 - 1.0e-6), model.energy), (1.0,))
    raise ValueError(f"Unsupported machine spectrum model: {model!r}")


def beamlet_source_for_beam(beam: Beam, spectrum, focal_spot_sigma: float = 0.0):
    """Build the beamlet source for one stf beam, in the engine frame.

    stf positions (``source_point``, ``ray_pos``) are isocenter-relative mm in
    LPS; world position = ``iso_center + v``, engine position = world / 10. The
    aperture plane is the isocenter plane; its lateral axes are the beam's
    rotated BEV x and z axes. Aperture centers are handed over in pyRadPlan
    bixel order (rays, then beamlets per ray), so the Dij columns come back in
    that order.

    ``focal_spot_sigma`` (mm) selects the source kind: 0 gives the ideal
    point-focus ``SpectralBeamletSource``; a positive value gives the planar
    ``GaussianSpotBeamletSource`` with a 2D-Gaussian focal spot.
    """
    rotation_matrix = lps.get_beam_rotation_matrix(
        np.asarray(beam.gantry_angle, dtype=np.float64),
        np.asarray(beam.couch_angle, dtype=np.float64),
    )

    focal_point = (beam.iso_center + beam.source_point) / _MM_PER_CM
    u_axis = rotation_matrix @ np.array([1.0, 0.0, 0.0])
    v_axis = rotation_matrix @ np.array([0.0, 0.0, 1.0])

    centers = [
        tuple((beam.iso_center + ray.ray_pos) / _MM_PER_CM)
        for ray in beam.rays
        for _ in ray.beamlets
    ]

    width = beam.bixel_width / _MM_PER_CM

    if focal_spot_sigma > 0.0:
        from pyradmc import GaussianSpotBeamletSource

        sigma = focal_spot_sigma / _MM_PER_CM
        return GaussianSpotBeamletSource(
            spectrum,
            focal_point=tuple(focal_point),
            centers=centers,
            width_u=width,
            width_v=width,
            sigma_u=sigma,
            sigma_v=sigma,
            u_axis=tuple(u_axis),
            v_axis=tuple(v_axis),
        )

    from pyradmc import SpectralBeamletSource

    return SpectralBeamletSource(
        spectrum,
        focal_point=tuple(focal_point),
        centers=centers,
        width_u=width,
        width_v=width,
        u_axis=tuple(u_axis),
        v_axis=tuple(v_axis),
    )


def _bev_axis_vector(axis: str) -> np.ndarray:
    """Return the unit vector of a machine collimator axis in BEV coordinates.

    In the matRad-heritage BEV frame the beam travels along +y; the machine
    crossplane axis 'x' is BEV x and the inplane axis 'y' is BEV z.
    """
    if axis == "x":
        return np.array([1.0, 0.0, 0.0])
    if axis == "y":
        return np.array([0.0, 0.0, 1.0])
    raise ValueError(f"Unknown collimator axis {axis!r}")


@dataclass
class BeamAperture:
    """Per-beam aperture settings at the isocenter plane, in BEV coordinates.

    All values are mm at the isocenter. Intervals are ``(lo, hi)`` along the
    machine axis' BEV direction: machine axis 'x' maps to BEV x, machine axis
    'y' to BEV z (mind that DICOM/IEC device coordinates have X flipped
    against BEV LPS; :func:`field_aperture_for_beam` handles that).

    Attributes
    ----------
    box : dict[str, tuple[float, float]]
        The overall open bounding box per machine axis - used for devices
        without an explicit setting and for the pre-solve emission field.
    jaws : dict[str, tuple[float, float]]
        Explicitly configured jaw openings per machine axis.
    mlc_travel_axis : str, optional
        Machine axis the configured MLC leaves travel along.
    mlc_strips : list, optional
        Per leaf pair: ``(strip_lo, strip_hi, opening)`` where the strip
        bounds run along the orthogonal axis and ``opening`` is the
        ``(lo, hi)`` interval along the travel axis, or None for a closed
        pair. Sorted ascending by strip bound.
    """

    box: dict[str, tuple[float, float]]
    jaws: dict[str, tuple[float, float]] = field(default_factory=dict)
    mlc_travel_axis: Optional[str] = None
    mlc_strips: Optional[list] = None


def _iec_interval_to_bev(axis: str, lo: float, hi: float) -> tuple[float, float]:
    """Map an IEC device interval to the machine axis' BEV coordinate.

    IEC X points to the patient's right while BEV LPS x points left, so
    X-oriented settings flip sign (and swap interval ends); IEC Y maps to
    BEV z unchanged.
    """
    if axis == "x":
        return (-hi, -lo)
    return (lo, hi)


def collect_field_blds(beam: Beam) -> list:
    """Collect beam-limiting device settings from a beam's field-shape beamlets.

    Field-based stf beamlets (``FieldShapeAsBLD``, or ``FieldShapeComposite``
    of such) carry the configured devices - jaw positions and MLC leaf
    positions at the isocenter plane in IEC coordinates.
    """
    blds = []
    for ray in beam.rays:
        for beamlet in ray.beamlets:
            if isinstance(beamlet, FieldShapeAsBLD):
                blds.append(beamlet.bld)
            elif isinstance(beamlet, FieldShapeComposite):
                blds.extend(
                    shape.bld for shape in beamlet.shapes if isinstance(shape, FieldShapeAsBLD)
                )
    return blds


def field_aperture_for_beam(beam: Beam, blds: Sequence = ()) -> BeamAperture:
    """Resolve one beam's aperture from BLD settings and/or the stf rays.

    Without device settings the aperture is the conformal bounding box of the
    ray positions plus half a bixel width. Configured jaws and MLC leaf
    positions (all stated at the isocenter plane, IEC coordinates) override
    and tighten it.
    """
    half = beam.bixel_width / 2.0
    ray_pos_bev = np.array([ray.ray_pos_bev for ray in beam.rays], dtype=np.float64)
    box = {
        "x": (float(ray_pos_bev[:, 0].min()) - half, float(ray_pos_bev[:, 0].max()) + half),
        "y": (float(ray_pos_bev[:, 2].min()) - half, float(ray_pos_bev[:, 2].max()) + half),
    }
    aperture = BeamAperture(box=box)

    for bld in blds:
        axis = "x" if bld.device_orientation == "X" else "y"
        if isinstance(bld, JawSetting):
            aperture.jaws[axis] = _iec_interval_to_bev(
                axis, float(min(bld.positions)), float(max(bld.positions))
            )
        elif isinstance(bld, MLCSetting):
            if aperture.mlc_strips is not None:
                logger.warning("Multiple MLC settings found for one beam; using the last one.")
            orth = "y" if axis == "x" else "x"
            edges = np.append(
                bld.leaf_position_boundaries,
                bld.leaf_position_boundaries[-1] + bld.leaf_width,
            )
            strips = []
            for i, (o_lo, o_hi) in enumerate(np.asarray(bld.leaf_positions, dtype=np.float64)):
                strip = _iec_interval_to_bev(orth, float(edges[i]), float(edges[i + 1]))
                opening = None if o_hi <= o_lo else _iec_interval_to_bev(axis, o_lo, o_hi)
                strips.append((strip[0], strip[1], opening))
            strips.sort(key=lambda s: s[0])
            aperture.mlc_travel_axis = axis
            aperture.mlc_strips = strips

    # Tighten the box: MLC open extent, then jaw intersections
    if aperture.mlc_strips is not None:
        open_strips = [s for s in aperture.mlc_strips if s[2] is not None]
        if open_strips:
            travel = aperture.mlc_travel_axis
            orth = "y" if travel == "x" else "x"
            box[travel] = (
                min(s[2][0] for s in open_strips),
                max(s[2][1] for s in open_strips),
            )
            box[orth] = (min(s[0] for s in open_strips), max(s[1] for s in open_strips))
    for axis, interval in aperture.jaws.items():
        lo = max(box[axis][0], interval[0])
        hi = min(box[axis][1], interval[1])
        box[axis] = (lo, hi) if lo < hi else interval

    return aperture


def beam_limiting_stack_for_beam(  # noqa: PLR0912, PLR0915
    beam: Beam,
    mc_model: PhotonLINACBeamModelMC,
    water_equivalent: bool = False,
    aperture: Optional[BeamAperture] = None,
):
    """Build a pyRadMC ``BeamLimitingStack`` for one stf beam.

    The machine model provides the device geometry (plane positions, leaf
    boundaries, materials); the per-field opening comes from ``aperture``
    (jaw and MLC leaf settings at the isocenter plane, see
    :func:`field_aperture_for_beam`). Devices without an explicit setting
    open conformally: jaws to the aperture bounding box, each machine MLC
    leaf pair to the stf ray extent within its strip (closed pairs are parked
    outside the field). All settings are stated at the isocenter plane and
    projected to the device mid-planes.

    The stack's local frame is right-handed with ``w`` pointing downstream and
    ``u`` along the MLC leaf travel; ``v = w x u`` may therefore be the
    *negative* of a BEV axis, which the coordinate mapping here accounts for.

    Returns ``None`` when the machine model has no devices.
    ``water_equivalent`` replaces device materials by water at unit density
    (for the analytic, water-only cross-section model - unrealistically weak
    attenuation, testing only).
    """
    from pyradmc import TUNGSTEN, WATER
    from pyradmc.geometry.collimation import (
        MLC,
        BeamFrame,
        BeamLimitingStack,
        JawPair,
        project_between_planes,
    )

    if mc_model is None or (not mc_model.jaws and mc_model.mlc is None):
        return None

    if aperture is None:
        aperture = field_aperture_for_beam(beam)

    half = beam.bixel_width / 2.0  # mm at the isocenter plane
    ray_pos_bev = np.array([ray.ray_pos_bev for ray in beam.rays], dtype=np.float64)

    travel_axis = mc_model.mlc.travel_axis if mc_model.mlc is not None else "x"
    u_bev = _bev_axis_vector(travel_axis)
    w_bev = np.array([0.0, 1.0, 0.0])
    v_bev = np.cross(w_bev, u_bev)

    q_u = ray_pos_bev @ u_bev  # mm at iso, along leaf travel
    q_v = ray_pos_bev @ v_bev

    sad_cm = beam.sad / _MM_PER_CM

    def project(setting_iso_mm: float, z_top_mm: float, z_bottom_mm: float) -> float:
        z_mid_cm = 0.5 * (z_top_mm + z_bottom_mm) / _MM_PER_CM
        return float(project_between_planes(setting_iso_mm / _MM_PER_CM, sad_cm, z_mid_cm))

    def material_of(name: str, density: float) -> tuple[int, float]:
        if water_equivalent:
            return WATER, 1.0
        return (TUNGSTEN if name == "tungsten" else WATER), density

    def to_local(
        machine_axis: str, interval: tuple[float, float], local_vec
    ) -> tuple[float, float]:
        # BEV coordinate along a machine axis -> local frame coordinate (sign flip only)
        sign = float(_bev_axis_vector(machine_axis) @ local_vec)
        lo, hi = sign * interval[0], sign * interval[1]
        return (lo, hi) if lo <= hi else (hi, lo)

    devices = []

    for jaw in mc_model.jaws:
        jaw_vec = _bev_axis_vector(jaw.axis)
        if abs(float(jaw_vec @ u_bev)) > 0.5:
            local_axis, local_vec = "u", u_bev
        else:
            local_axis, local_vec = "v", v_bev
        interval_bev = aperture.jaws.get(jaw.axis, aperture.box[jaw.axis])
        lo, hi = to_local(jaw.axis, interval_bev, local_vec)
        material, density = material_of(jaw.material, jaw.density)
        devices.append(
            JawPair(
                axis=local_axis,
                z_top=jaw.z_top / _MM_PER_CM,
                z_bottom=jaw.z_bottom / _MM_PER_CM,
                edge_neg=project(lo, jaw.z_top, jaw.z_bottom),
                edge_pos=project(hi, jaw.z_top, jaw.z_bottom),
                material=material,
                density=density,
                focused=jaw.focused,
            )
        )

    if mc_model.mlc is not None:
        mlc = mc_model.mlc
        orth_axis = "y" if mlc.travel_axis == "x" else "x"

        configured_strips = None
        if aperture.mlc_strips is not None:
            if aperture.mlc_travel_axis == mlc.travel_axis:
                configured_strips = aperture.mlc_strips
            else:
                logger.warning(
                    "Configured MLC travels along machine axis '%s' but the machine MLC "
                    "travels along '%s'; falling back to a conformal opening.",
                    aperture.mlc_travel_axis,
                    mlc.travel_axis,
                )

        if configured_strips is not None:
            # Explicit leaf settings: strips along orth axis, openings along travel
            strips_local = []
            for strip_lo, strip_hi, opening in configured_strips:
                v_lo, v_hi = to_local(orth_axis, (strip_lo, strip_hi), v_bev)
                u_open = None if opening is None else to_local(mlc.travel_axis, opening, u_bev)
                strips_local.append((v_lo, v_hi, u_open))
            strips_local.sort(key=lambda s: s[0])

            edges_iso = np.array(
                [strips_local[0][0]] + [s[1] for s in strips_local], dtype=np.float64
            )
            open_lows = [s[2][0] for s in strips_local if s[2] is not None]
            park = (min(open_lows) if open_lows else 0.0) - 30.0
            tips_neg = [park if s[2] is None else s[2][0] for s in strips_local]
            tips_pos = [park if s[2] is None else s[2][1] for s in strips_local]
        else:
            # Conformal opening from the stf rays on the machine's own strips
            sign = float(_bev_axis_vector(orth_axis) @ v_bev)
            edges_iso = np.sort(sign * mlc.leaf_edges)  # mm at iso, ascending along v

            park = float(q_u.min()) - 3.0 * half  # closed pairs park outside the field
            tips_neg, tips_pos = [], []
            for lo, hi in zip(edges_iso[:-1], edges_iso[1:]):
                in_strip = (q_v + half > lo) & (q_v - half < hi)
                if np.any(in_strip):
                    tips_neg.append(float(q_u[in_strip].min()) - half)
                    tips_pos.append(float(q_u[in_strip].max()) + half)
                else:
                    tips_neg.append(park)
                    tips_pos.append(park)

        material, density = material_of(mlc.material, mlc.density)
        devices.append(
            MLC(
                z_top=mlc.z_top / _MM_PER_CM,
                z_bottom=mlc.z_bottom / _MM_PER_CM,
                leaf_edges_v=tuple(project(e, mlc.z_top, mlc.z_bottom) for e in edges_iso),
                tips_neg=tuple(project(t, mlc.z_top, mlc.z_bottom) for t in tips_neg),
                tips_pos=tuple(project(t, mlc.z_top, mlc.z_bottom) for t in tips_pos),
                tip_radius=mlc.tip_radius / _MM_PER_CM,
                material=material,
                density=density,
            )
        )

    devices.sort(key=lambda device: device.z_top)

    rotation_matrix = lps.get_beam_rotation_matrix(
        np.asarray(beam.gantry_angle, dtype=np.float64),
        np.asarray(beam.couch_angle, dtype=np.float64),
    )
    frame = BeamFrame(
        origin=tuple((beam.iso_center + beam.source_point) / _MM_PER_CM),
        u_axis=tuple(rotation_matrix @ u_bev),
        v_axis=tuple(rotation_matrix @ v_bev),
        w_axis=tuple(rotation_matrix @ w_bev),
    )
    return BeamLimitingStack(frame=frame, devices=tuple(devices))


class PhotonPyRadMCEngine(MonteCarloEngineAbstract):
    """Photon Monte Carlo dose engine using pyRadMC.

    Computes beamlet-resolved dose influence matrices (and forward dose) with
    the pyRadMC GPU/CPU photon Monte Carlo engine.

    Attributes
    ----------
    backend : str
        'warp' (production, GPU or CPU) or 'ref' (pure-python reference
        backend; correct but orders of magnitude slower - testing only).
    warp_device : str, optional
        Warp device ('cuda:0', 'cpu'). None selects CUDA if available.
    warp_devices : list of str, optional
        Devices to spread the Dij calculation over (e.g. ['cuda:0',
        'cuda:1']); how the work is distributed is set by
        ``beam_scheduling``. None (default) runs everything on the single
        ``warp_device``.
    beam_scheduling : str
        Multi-device work distribution for Dij calculation ('warp' backend
        only). 'sharded' (default) keeps beams sequential and lets pyRadMC
        shard each beam's beamlet groups greedily over ``warp_devices`` -
        finer load balancing, best with many beamlets per beam or
        heterogeneous devices (a slow device self-limits).
        'beam_per_device' runs whole beams concurrently, beam ``b`` wholly
        on ``warp_devices[b % n]`` - coarse and fully reproducible with
        zero intra-run coordination, best with few beams on several alike
        GPUs. It requires ``warp_devices``, reduces to the sequential loop
        when only one device is configured, and does not apply to the
        head-presolve forward path.
    concurrent_batches : int
        Statistical-batch lanes per CUDA device: up to this many batches
        transport concurrently on one device. Bitwise inert (any value
        reproduces the sequential result exactly) at the cost of extra
        device memory per lane; measured neutral on laptop-class GPUs, so
        measure before raising it. Ignored on cpu devices. When
        ``beamlet_group_size`` is auto-sized (its default), each added lane
        enlarges the per-beamlet buffer, so pyRadMC shrinks the auto group
        size to compensate rather than multiplying the two into an overflow.
    seed : int
        Random seed; one pyRadMC run per beam, seeded with ``seed + beam index``.
    n_batches : int
        Statistical batches per run; histories per beamlet are rounded up to a
        multiple of this.
    correlated : bool
        Use correlated (shared-stream) beamlet sampling (pyRadMC default). Note
        that with correlated sampling, Dij columns are statistically dependent:
        combining per-column variances across columns is invalid (pyRadMC warns
        when the variance matrix is exported).
    scoring_mode : str
        'dose_to_medium' or 'dose_to_water'.
    spectrum_beam : str, optional
        Override: Ali & Rogers (2012) analytic MV spectrum key (e.g.
        'varian-6mv'; see ``pyradmc.geometry.spectrum.ALI_ROGERS_BEAMS``).
        None (default) uses the machine's ``mc_beam_model`` spectrum selected
        by the stf beamlet energy.
    forward_mode : str
        Forward (direct) dose path: 'beamlets' (default) weights the per-bixel
        Dij columns with the stf fluence weights; 'head_presolve' builds a
        conformal static jaw/MLC aperture from the machine's ``mc_beam_model``,
        runs the pyRadMC treatment-head pre-solve (collimator transmission and
        scatter included) and transports the exit phase space - physically
        richer, but it ignores per-bixel fluence modulation (uniform fields
        only) and its absolute scaling relies on the emission-area to
        bixel-area ratio. Dij calculation never models the devices.
    cross_section_model : str
        'tabulated' compiles multi-material tables from the EPICS libraries
        (~120 MB download on first use, cached in ``~/.cache/pyradmc/epics``).
        'analytic' is a water-only closed-form model: all voxels are treated as
        water with the HU-derived density (fast, no download - testing and
        water-phantom use).
    dij_truncation : float, optional
        Relative per-column truncation threshold of the sparse Dij; None uses
        the pyRadMC default.
    num_particles_per_mu : float
        Placeholder output calibration: histories per monitor unit. pyRadMC
        scores Gy per emitted history; Dij columns are scaled by this factor
        (variance by its square) so that weights are in MU-like units. A
        machine-specific output calibration is not modelled yet.
    transport_electrons : bool
        Transport secondary electrons (disable only for testing).
    msc_preset : str
        Electron multiple-scattering preset, selecting one of pyRadMC's
        validated ``(msc_model, step_energy_fraction)`` pairs. 'gaussian'
        (default) is the shipped Gaussian small-angle model with electron
        substeps capped at 5% of the CSDA range. 'gs' samples exact
        Goudsmit-Saunderson deflections with 20% substeps - an accuracy
        option for the large-angle scattering tail and penumbra (~+0.3% dose
        vs the default on a 6 MV spectral thorax benchmark; ~1.17x GPU
        open-field runtime, Dij nearly unchanged). The pair is validated as
        a whole and the step fraction is deliberately not exposed
        separately: under 'gaussian' longer substeps are re-bound by an
        internal angular cap, under 'gs' that cap is lifted. The first 'gs'
        run per machine builds a deflection-table grid (~30-45 s, one time,
        persisted to ``~/.cache/pyradmc/gs-grid``); later processes load it
        in ~40 ms. 'gs' keeps the engine's reproducibility guarantees
        (bit-identical per device, Dij group-size invariant).
    beamlet_group_size : int, optional
        Beamlets scored concurrently into one dense device buffer per device
        (memory/occupancy knob, result-invariant). None (default) lets pyRadMC
        auto-size the group to fit each device's free memory - taking the
        tightest device's fit across a multi-device set, and shrinking as
        ``concurrent_batches`` lanes are added (dense cost per beamlet is
        ``(8 * concurrent_batches + 16) * n_scoring_voxels`` bytes per device).
        Set an explicit integer to override the auto-sizing.

    Notes
    -----
    Reproducibility: over identical devices the Dij is bit-identical for
    any ``beam_scheduling``, ``warp_devices`` and ``concurrent_batches``
    (pinned by pyRadMC's tests). With 'sharded' over *mixed* devices (e.g.
    ['cuda:0', 'cpu']) pyRadMC's greedy queue makes the column-to-device
    assignment timing-dependent, so the same plan run twice may differ per
    column at float precision level; 'beam_per_device' assigns beams to
    devices statically and stays run-to-run reproducible even on mixed
    device sets.
    """

    short_name: ClassVar[str] = "pyRadMC"
    name: ClassVar[str] = "pyRadMC Photon Monte Carlo"
    possible_radiation_modes: ClassVar[list[str]] = ["photons"]

    backend: Literal["warp", "ref"] = "warp"
    warp_device: Optional[str] = None
    warp_devices: Optional[list[str]] = None
    beam_scheduling: Literal["sharded", "beam_per_device"] = "sharded"
    concurrent_batches: Annotated[int, Field(ge=1)] = 1
    seed: int = 20260715
    n_batches: int = 10
    correlated: bool = False
    scoring_mode: Literal["dose_to_medium", "dose_to_water"] = "dose_to_medium"
    spectrum_beam: Optional[str] = None
    forward_mode: Literal["beamlets", "head_presolve"] = "beamlets"
    cross_section_model: Literal["tabulated", "analytic"] = "tabulated"
    dij_truncation: Optional[float] = None
    num_particles_per_mu: float = 1e9
    transport_electrons: bool = True
    msc_preset: Literal["gaussian", "gs"] = "gaussian"
    beamlet_group_size: Optional[int] = None

    def __init__(self, pln=None):
        self._spectrum_cache = {}
        super().__init__(pln)

    def _mc_beam_model(self) -> Optional[PhotonLINACBeamModelMC]:
        return getattr(self._machine, "mc_beam_model", None)

    def _spectrum_for_beam(self, beam: Beam):
        """Resolve the primary photon spectrum for one stf beam.

        ``spectrum_beam`` (an Ali & Rogers key) overrides; otherwise the
        machine's MC beam model provides the spectrum for the beam's nominal
        energy (taken from its first beamlet).
        """
        if self.spectrum_beam is not None:
            key = ("named", self.spectrum_beam)
            if key not in self._spectrum_cache:
                from pyradmc import ali_rogers_mv

                self._spectrum_cache[key] = ali_rogers_mv(self.spectrum_beam)
            return self._spectrum_cache[key]

        energy = float(beam.rays[0].beamlets[0].energy)
        key = ("machine", energy)
        if key not in self._spectrum_cache:
            beam_model = self._mc_beam_model()
            if beam_model is None or not beam_model.spectra:
                raise ValueError(
                    "Machine carries no MC beam model spectrum; set the engine's "
                    "'spectrum_beam' option to a named Ali & Rogers spectrum instead."
                )
            self._spectrum_cache[key] = spectrum_from_model(beam_model.get_spectrum(energy))
        return self._spectrum_cache[key]

    def _make_cross_sections(self, transport_grid, e_max: float):
        """Build the cross-section source configured by ``cross_section_model``."""
        if self.cross_section_model == "analytic":
            from pyradmc import AnalyticCrossSections

            return AnalyticCrossSections(
                geometry_densities=transport_grid.max_density_by_material()
            )

        from pyradmc import ECUT_MEV, PCUT_MEV, TabulatedCrossSections
        from pyradmc.data.tabulated import build
        from pyradmc.data.tabulated.precompile import compile_materials

        cache_key = (round(e_max, 6), PCUT_MEV, ECUT_MEV)
        if cache_key not in _TABULATED_DATA_CACHE:
            logger.info(
                "Compiling tabulated cross sections from the EPICS libraries "
                "(downloads ~120 MB into ~/.cache/pyradmc/epics on first use)..."
            )
            epdl = build.download_library("epdl")
            eedl = build.download_library("eedl")
            _TABULATED_DATA_CACHE[cache_key] = compile_materials(
                epdl.read_text(encoding="latin-1"),
                eedl.read_text(encoding="latin-1"),
                e_max=e_max,
            )

        return TabulatedCrossSections(
            _TABULATED_DATA_CACHE[cache_key],
            geometry_densities=transport_grid.max_density_by_material(),
        )

    def _make_engine(self, transport_grid, cross_sections):
        """Instantiate the configured pyRadMC backend engine."""
        if self.backend == "warp":
            import warp as wp

            from pyradmc import WarpEngine

            device = self.warp_device
            if device is None:
                device = "cuda:0" if wp.is_cuda_available() else "cpu"
            logger.info("pyRadMC warp backend on device '%s'.", device)
            return WarpEngine(grid=transport_grid, cross_sections=cross_sections, device=device)

        from pyradmc import HostRNG, ReferenceEngine

        logger.info("pyRadMC reference (CPU) backend.")
        return ReferenceEngine(grid=transport_grid, cross_sections=cross_sections, rng=HostRNG())

    def _devices(self) -> list[str]:
        """Resolve ``warp_devices``; empty means the engine's own single device."""
        return list(self.warp_devices) if self.warp_devices else []

    def _effective_beam_scheduling(self) -> str:
        """Resolve ``beam_scheduling`` against the backend and configured devices."""
        if self.beam_scheduling != "beam_per_device":
            return self.beam_scheduling
        if self.backend != "warp":
            logger.info(
                "beam_scheduling='beam_per_device' applies to the 'warp' backend only; "
                "running beams sequentially."
            )
            return "sharded"
        devices = self._devices()
        if not devices:
            raise ValueError(
                "beam_scheduling='beam_per_device' requires at least one entry in 'warp_devices'."
            )
        if len(devices) == 1:
            logger.info(
                "Only one device configured in 'warp_devices'; "
                "beam_scheduling='beam_per_device' reduces to the sequential beam loop."
            )
            return "sharded"
        return "beam_per_device"

    def _progress_callback(  # noqa: PLR0913
        self,
        beam_index: int,
        num_beams: int,
        total_histories: int,
        history_handle: Optional[ProgressHandle] = None,
    ) -> Callable[["ProgressEvent"], None]:
        """Build a pyRadMC ``progress`` callback that reports beam-scoped status.

        pyRadMC ticks this from whichever thread is transporting (the calling
        thread for sequential scheduling, a worker thread per device for
        ``beam_scheduling='beam_per_device'``); ``report_status`` dispatches
        without touching the nested progress stack, so it is safe to call from
        any thread without further synchronization. For the same reason the
        console bar below is a plain ``tqdm`` instance owned by this closure
        instead of going through ``self.track``/``self.progress``: those push
        onto the engine's single shared progress stack (fine for the other
        engines' single-threaded beam/ray loops, but not safe to share across
        the concurrent per-device worker threads here).

        ``history_handle``, if given, is advanced from ticks so the GUI/console
        progress bar tracks per-beam history progress instead of just pulsing
        while busy - like the pencil-beam engines' Ray level nested under
        Beam. The caller must open it from an enclosing
        ``with self.progress("History", total=total_histories)`` scoped to
        this beam's run, and only when that beam runs alone in the calling
        thread (sequential scheduling); the concurrent
        ``beam_scheduling='beam_per_device'`` worker threads must never touch
        the shared stack and stay on ``report_status`` alone (the GUI still
        shows a busy bar for those, via the enclosing Beam level).
        """
        bar = None
        if self.console_progress:
            try:
                from tqdm import tqdm  # noqa: PLC0415

                bar = tqdm(
                    desc=f"Beam {beam_index + 1}/{num_beams}",
                    unit="h",
                    total=total_histories,
                    leave=False,
                    position=beam_index,
                )
            except ImportError:
                bar = None

        def _on_progress(event: "ProgressEvent") -> None:
            if bar is not None:
                bar.n = event.histories_done
                bar.refresh()
                if event.histories_done >= event.histories_total:
                    bar.close()

            if history_handle is not None:
                history_handle.update(event.histories_done)

            self.report_status(
                f"Beam {beam_index + 1}/{num_beams}: {event.histories_done}/"
                f"{event.histories_total} histories ({event.rate_hz:.3g} Hz)",
                beam=beam_index,
                num_beams=num_beams,
                histories_done=event.histories_done,
                histories_total=event.histories_total,
                elapsed_s=event.elapsed_s,
                rate_hz=event.rate_hz,
            )

        return _on_progress

    @staticmethod
    def _forward_weights(beam: Beam) -> np.ndarray:
        """Per-bixel fluence weights of a beam (missing weights count as 1)."""
        return np.array(
            [
                beamlet.weight if beamlet.weight is not None else 1.0
                for ray in beam.rays
                for beamlet in ray.beamlets
            ],
            dtype=np.float64,
        )

    def _histories_per_beamlet(self, total_num_of_bixels: int) -> int:
        """Histories per beamlet, rounded up to a multiple of ``n_batches``."""
        if self._calc_dose_direct:
            n = self.num_histories_direct / max(1, total_num_of_bixels)
        else:
            n = self.num_histories_per_beamlet
        return int(np.ceil(max(1.0, float(n)) / self.n_batches)) * self.n_batches

    def _aperture_for_beam(self, beam: Beam) -> BeamAperture:
        """Resolve the beam aperture from stf field-shape BLD settings or the rays."""
        blds = collect_field_blds(beam)
        if blds:
            logger.info(
                "Using %d beam-limiting device setting(s) from the stf field shapes.", len(blds)
            )
        return field_aperture_for_beam(beam, blds)

    def _stack_for_beam(self, beam: Beam, aperture: Optional[BeamAperture] = None):
        """Build the per-beam beam-limiting stack (head-presolve forward only)."""
        beam_model = self._mc_beam_model()
        if beam_model is None:
            return None
        if beam_model.mlc is not None and beam_model.mlc.leaf_end_design == "double_focused":
            logger.warning(
                "The machine declares a double-focused MLC; pyRadMC approximates the "
                "focused leaf ends by rounded tips (radius %g mm) with parallel sides.",
                beam_model.mlc.tip_radius,
            )
        water_equivalent = self.cross_section_model == "analytic"
        stack = beam_limiting_stack_for_beam(
            beam, beam_model, water_equivalent=water_equivalent, aperture=aperture
        )
        if stack is not None and water_equivalent:
            logger.warning(
                "Beam-limiting devices are modeled as water at unit density under the "
                "analytic cross-section model; attenuation is unrealistically weak."
            )
        return stack

    def _run_beam_dij(  # noqa: PLR0913
        self,
        engine,
        beam: Beam,
        n_histories: int,
        seed: int,
        scoring_grid,
        devices: Optional[Sequence[str]] = None,
        progress: Optional[Callable[["ProgressEvent"], None]] = None,
    ):
        """Run one per-beam Dij and return (dose_csc, variance_csc), MU-scaled.

        The beamlet source is the plain divergent fan - beam-limiting devices
        are deliberately not modeled in the influence matrix. Rows are already
        remapped to pyRadPlan dose-grid linear indices. ``devices`` shards the
        beamlet groups over the given warp devices (None/empty runs on the
        engine's own device). ``progress`` (see :mod:`pyradmc.progress`) is
        forwarded to ``run_dij`` verbatim; it ticks once per beamlet group
        drained, covering that group's full history count.
        """
        from pyradmc import DIJ_TRUNCATION_RELATIVE

        beam_model = self._mc_beam_model()
        focal_spot_sigma = beam_model.focal_spot_sigma if beam_model is not None else 0.0
        source = beamlet_source_for_beam(
            beam, self._spectrum_for_beam(beam), focal_spot_sigma=focal_spot_sigma
        )

        truncation = (
            DIJ_TRUNCATION_RELATIVE if self.dij_truncation is None else self.dij_truncation
        )

        kwargs = dict(_MSC_PRESETS[self.msc_preset])
        if self.backend == "warp":
            kwargs["beamlet_group_size"] = self.beamlet_group_size
            if devices:
                kwargs["devices"] = list(devices)
            kwargs["concurrent_batches"] = self.concurrent_batches

        result = engine.run_dij(
            source,
            n_histories_per_beamlet=n_histories,
            n_batches=self.n_batches,
            seed=seed,
            transport_electrons=self.transport_electrons,
            truncation=truncation,
            correlated=self.correlated,
            scoring_grid=scoring_grid,
            scoring_mode=self.scoring_mode,
            progress=progress,
            **kwargs,
        )

        dimensions = self.dose_grid.dimensions
        # Placeholder MU calibration: Gy/history -> Gy/MU (see class docstring)
        dose = permute_rows_engine_to_grid(
            result.dose_csc(unit="gy") * self.num_particles_per_mu, dimensions
        )
        # The correlated-sampling caveat warning from pyRadMC propagates here on
        # purpose: pyRadPlan combines variance columns (physical_dose_var @ w),
        # which is invalid under correlated sampling.
        variance = permute_rows_engine_to_grid(
            result.variance_csc(unit="gy") * self.num_particles_per_mu**2, dimensions
        )
        return dose, variance

    def _forward_presolve_column(  # noqa: PLR0913
        self,
        engine,
        beam: Beam,
        stack,
        aperture: BeamAperture,
        n_histories: int,
        seed: int,
        scoring_grid,
        cross_sections,
        progress: Optional[Callable[["ProgressEvent"], None]] = None,
    ):
        """Forward dose of one beam via the treatment-head pre-solve.

        Emits an open field covering the aperture bounding box (from the stf
        BLD settings, or the ray bounding box) on a plane just upstream of the
        first beam-limiting device, pre-solves the head (deterministic
        transmission + collimator scatter), and transports the exit phase
        space. Returns MU-scaled (dose, variance) column vectors on the
        pyRadPlan dose grid.

        Per-bixel fluence modulation is ignored (the mean beamlet weight
        scales the whole field); the absolute scale maps the per-field-history
        dose to the per-bixel-history convention via the emission-to-bixel
        area ratio.
        """
        from pyradmc import GY_PER_MEV_PER_G

        rotation_matrix = lps.get_beam_rotation_matrix(
            np.asarray(beam.gantry_angle, dtype=np.float64),
            np.asarray(beam.couch_angle, dtype=np.float64),
        )
        spectrum = self._spectrum_for_beam(beam)
        beam_model = self._mc_beam_model()

        # Aperture bounding box in BEV coordinates (mm at iso); a small margin
        # lets the emission cover the geometric penumbra region
        margin = 2.0
        x_lo, x_hi = aperture.box["x"]
        z_lo, z_hi = aperture.box["y"]
        lo = np.array([x_lo - margin, 0.0, z_lo - margin])
        hi = np.array([x_hi + margin, 0.0, z_hi + margin])
        center_iso_bev = 0.5 * (lo + hi)  # mm at the isocenter plane, bev y = 0
        widths_iso = hi - lo

        # Emission plane just upstream of the first device, scaled from the iso plane
        z_up_mm = min(device.z_top for device in stack.devices) * _MM_PER_CM - 10.0
        scale = z_up_mm / beam.sad
        focal_bev = np.array([0.0, -beam.sad, 0.0])
        center_bev = focal_bev + scale * (center_iso_bev - focal_bev)

        focal_point = tuple((beam.iso_center + beam.source_point) / _MM_PER_CM)
        center = tuple((beam.iso_center + rotation_matrix @ center_bev) / _MM_PER_CM)
        width_u = float(widths_iso[0]) * scale / _MM_PER_CM
        width_v = float(widths_iso[2]) * scale / _MM_PER_CM
        u_axis = tuple(rotation_matrix @ np.array([1.0, 0.0, 0.0]))
        v_axis = tuple(rotation_matrix @ np.array([0.0, 0.0, 1.0]))

        sigma = (beam_model.focal_spot_sigma if beam_model is not None else 0.0) / _MM_PER_CM
        if sigma > 0.0:
            from pyradmc import GaussianSpotBeamSource

            source = GaussianSpotBeamSource(
                spectrum, focal_point, center, width_u, width_v, sigma, sigma, u_axis, v_axis
            )
        else:
            from pyradmc import SpectralBeamSource

            source = SpectralBeamSource(
                spectrum, focal_point, center, width_u, width_v, u_axis, v_axis
            )

        exit_z = max(device.z_bottom for device in stack.devices) + 1.0  # cm, local w
        if self.backend == "warp":
            from pyradmc.backends.warp.presolve import presolve_head_device

            phase_space = presolve_head_device(
                source,
                cross_sections=cross_sections,
                n_histories=n_histories,
                seed=seed,
                exit_z=exit_z,
                stack=stack,
                device=engine.device,
                return_device_source=True,
            )
        else:
            from pyradmc.geometry.head import presolve_head

            phase_space = presolve_head(
                source,
                cross_sections=cross_sections,
                n_histories=n_histories,
                seed=seed,
                exit_z=exit_z,
                stack=stack,
            )

        result = engine.run(
            phase_space,
            n_histories,
            self.n_batches,
            seed + 1,
            transport_electrons=self.transport_electrons,
            scoring_grid=scoring_grid,
            scoring_mode=self.scoring_mode,
            progress=progress,
            **_MSC_PRESETS[self.msc_preset],
        )

        weights = self._forward_weights(beam)
        if weights.size and np.ptp(weights) > 1.0e-6 * max(1.0, np.abs(weights).max()):
            logger.warning(
                "forward_mode='head_presolve' ignores per-bixel fluence modulation; "
                "using the mean beamlet weight for beam scaling."
            )

        area_ratio = float(np.prod(widths_iso[[0, 2]])) / beam.bixel_width**2
        factor = GY_PER_MEV_PER_G * self.num_particles_per_mu * area_ratio * weights.mean()

        dose_column = result.dose.transpose(2, 1, 0).reshape(-1) * factor
        var_column = (result.dose_sigma.transpose(2, 1, 0).reshape(-1)) ** 2 * factor**2
        return dose_column, var_column

    def _run_beams_per_device(self, engine, beams, n_histories, scoring_grid, devices):
        """Run whole beams concurrently, one warp device each ('beam_per_device').

        Beam ``b`` runs wholly on ``devices[b % len(devices)]``. The static
        assignment (rather than a greedy queue) plus the per-beam seeds make
        the columns identical to the sequential loop and run-to-run
        reproducible even on mixed device sets. Blocks are returned in beam
        order regardless of completion order.
        """
        import warp as wp

        from pyradmc.backends.warp import kernels

        # Warp module loading is not thread-safe, and pyRadMC preloads only
        # when it multi-threads internally; the single-device run_dij calls
        # issued from the beam threads would otherwise race the first-launch
        # compile.
        for device in devices:
            wp.load_module(kernels, device=device)

        dose_blocks: list = [None] * len(beams)
        var_blocks: list = [None] * len(beams)

        def run_one(b: int, beam: Beam):
            return self._run_beam_dij(
                engine,
                beam,
                n_histories,
                self.seed + b,
                scoring_grid,
                devices=[devices[b % len(devices)]],
                progress=self._progress_callback(
                    b, len(beams), n_histories * beam.total_number_of_bixels
                ),
            )

        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {pool.submit(run_one, b, beam): b for b, beam in enumerate(beams)}
            for future in self.track(
                as_completed(futures), name="Beam", unit="b", total=len(beams)
            ):
                b = futures[future]
                dose, variance = future.result()
                if self._calc_dose_direct:
                    weights = self._forward_weights(beams[b])
                    dose = sparse.csc_array((dose @ weights).reshape(-1, 1))
                    variance = sparse.csc_array((variance @ weights**2).reshape(-1, 1))
                dose_blocks[b] = dose
                var_blocks[b] = variance

        return dose_blocks, var_blocks

    def _run_beams_sequential(  # noqa: PLR0913
        self, engine, stf, n_histories, scoring_grid, cross_sections, devices
    ):
        """Run the beams one after another ('sharded' scheduling).

        Each beam's beamlet groups are sharded over ``devices`` by pyRadMC
        itself (empty list: the engine's own device). This is also the only
        path supporting the head-presolve forward mode.
        """
        dose_blocks = []
        var_blocks = []
        for b, beam in enumerate(self.track(stf.beams, name="Beam", unit="b")):
            if self._calc_dose_direct and self.forward_mode == "head_presolve":
                aperture = self._aperture_for_beam(beam)
                stack = self._stack_for_beam(beam, aperture)
                if stack is None:
                    logger.warning(
                        "forward_mode='head_presolve' needs beam-limiting devices; "
                        "falling back to the beamlet-weighted forward path for beam %d.",
                        b,
                    )
                else:
                    n_presolve = (
                        int(
                            np.ceil(
                                max(1.0, self.num_histories_direct / len(stf.beams))
                                / self.n_batches
                            )
                        )
                        * self.n_batches
                    )
                    with self.progress("History", total=n_presolve) as handle:
                        dose_col, var_col = self._forward_presolve_column(
                            engine,
                            beam,
                            stack,
                            aperture,
                            n_presolve,
                            self.seed + b,
                            scoring_grid,
                            cross_sections,
                            progress=self._progress_callback(
                                b, len(stf.beams), n_presolve, history_handle=handle
                            ),
                        )
                    dose_blocks.append(sparse.csc_array(dose_col.reshape(-1, 1)))
                    var_blocks.append(sparse.csc_array(var_col.reshape(-1, 1)))
                    continue

            total_beam_histories = n_histories * beam.total_number_of_bixels
            with self.progress("History", total=total_beam_histories) as handle:
                dose, variance = self._run_beam_dij(
                    engine,
                    beam,
                    n_histories,
                    self.seed + b,
                    scoring_grid,
                    devices=devices,
                    progress=self._progress_callback(
                        b, len(stf.beams), total_beam_histories, history_handle=handle
                    ),
                )

            if self._calc_dose_direct:
                weights = self._forward_weights(beam)
                dose_blocks.append(sparse.csc_array((dose @ weights).reshape(-1, 1)))
                var_blocks.append(sparse.csc_array((variance @ weights**2).reshape(-1, 1)))
            else:
                dose_blocks.append(dose)
                var_blocks.append(variance)

        return dose_blocks, var_blocks

    def _calc_dose(self, ct: CT, cst: StructureSet, stf: SteeringInformation, dij: dict) -> dict:
        if self._scenarios.num_computed != 1:
            raise ValueError("pyRadMC engine currently only supports a single scenario.")
        scen_ix = int(self._scenarios.computed_indices()[0])

        # apply_config assigns raw values that failed validation; refuse them
        # here, before any grid or cross-section work
        if self.msc_preset not in _MSC_PRESETS:
            raise ValueError(
                f"unknown msc_preset {self.msc_preset!r}; expected one of {sorted(_MSC_PRESETS)}"
            )

        dij["num_of_particles_per_mu"] = np.full(
            self._num_of_columns_dij, self.num_particles_per_mu
        )

        # Bookkeeping columns
        if self._calc_dose_direct:
            dij["beam_num"] = np.arange(len(stf.beams), dtype=float)
            dij["ray_num"] = np.zeros(len(stf.beams))
            dij["bixel_num"] = np.zeros(len(stf.beams))
        else:
            counter = 0
            for b, beam in enumerate(stf.beams):
                for j, ray in enumerate(beam.rays):
                    for k in range(len(ray.beamlets)):
                        dij["beam_num"][counter] = b
                        dij["ray_num"][counter] = j
                        dij["bixel_num"][counter] = k
                        counter += 1

        water_only = self.cross_section_model == "analytic"
        transport_grid = transport_grid_from_ct(self._scenario_ct(ct), water_only=water_only)
        scoring_grid = scoring_grid_from_dose_grid(self.dose_grid, transport_grid)

        # This scan also fills _spectrum_cache for every beam, which the
        # beam-per-device threads later read without locking.
        e_max = 1.2 * max(self._spectrum_for_beam(beam).max_energy for beam in stf.beams)
        cross_sections = self._make_cross_sections(transport_grid, e_max)
        engine = self._make_engine(transport_grid, cross_sections)

        if self.msc_preset == "gs" and self.transport_electrons:
            logger.info(
                "Goudsmit-Saunderson electron scattering selected (msc_preset='gs', "
                "step_energy_fraction=0.20). The first GS run on this machine builds a "
                "deflection-table grid once (~30-45 s, persisted to "
                "~/.cache/pyradmc/gs-grid); later runs load it in ~40 ms."
            )

        n_histories = self._histories_per_beamlet(dij["total_num_of_bixels"])
        logger.info(
            "Running pyRadMC with %d histories per beamlet in %d batches.",
            n_histories,
            self.n_batches,
        )

        devices = self._devices()
        scheduling = self._effective_beam_scheduling()
        if (
            scheduling == "beam_per_device"
            and self._calc_dose_direct
            and self.forward_mode == "head_presolve"
        ):
            logger.warning(
                "beam_scheduling='beam_per_device' does not apply to the head-presolve "
                "forward path; running beams sequentially."
            )
            scheduling = "sharded"

        if scheduling == "beam_per_device":
            dose_blocks, var_blocks = self._run_beams_per_device(
                engine, stf.beams, n_histories, scoring_grid, devices
            )
        else:
            dose_blocks, var_blocks = self._run_beams_sequential(
                engine, stf, n_histories, scoring_grid, cross_sections, devices
            )

        physical_dose = sparse.hstack(dose_blocks, format="csc").astype(np.float32)
        physical_dose_var = sparse.hstack(var_blocks, format="csc").astype(np.float32)

        dij["physical_dose"] = np.empty((len(self._scenarios),), dtype=object)
        dij["physical_dose"][scen_ix] = physical_dose
        dij["physical_dose_var"] = np.empty((len(self._scenarios),), dtype=object)
        dij["physical_dose_var"][scen_ix] = physical_dose_var

        return dij
