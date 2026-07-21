"""Tests for the pyRadMC photon Monte Carlo dose engine.

Unit tests cover the pure frame/grid/source mapping helpers (no MC transport);
the integration tests run the pyRadMC reference (pure-python CPU) backend with
unit-scale statistics on a tiny water phantom.
"""

import numpy as np
import pytest
import SimpleITK as sitk
from scipy import sparse

pyradmc = pytest.importorskip("pyRadMC")

from pyRadPlan.ct import CT
from pyRadPlan.cst import VOI, StructureSet
from pyRadPlan.plan import create_pln
from pyRadPlan.stf import validate_stf
from pyRadPlan.stf.generators import StfGeneratorPhotonIMRT
from pyRadPlan.dij import Dij
from pyRadPlan.dose.engines import PhotonPyRadMCEngine, get_available_engines, get_engine
from pyRadPlan.dose.engines._pyradmc import (
    beamlet_source_for_beam,
    permute_rows_engine_to_grid,
    scoring_grid_from_dose_grid,
    transport_grid_from_ct,
)


CT_SHAPE_ZYX = (16, 16, 16)
CT_SPACING = (8.0, 8.0, 8.0)  # mm
CT_ORIGIN = (-60.0, -60.0, -60.0)  # mm, roughly centered around zero


@pytest.fixture
def water_ct():
    """A small water cube CT with a bone insert at a known voxel."""
    hu = np.zeros(CT_SHAPE_ZYX, dtype=np.float64)  # water everywhere
    hu[4, 5, 6] = 1000.0  # bone insert at numpy (z=4, y=5, x=6)
    image = sitk.GetImageFromArray(hu)
    image.SetOrigin(CT_ORIGIN)
    image.SetSpacing(CT_SPACING)
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return CT(cube_hu=image)


@pytest.fixture
def water_cst(water_ct):
    target_mask = np.zeros(CT_SHAPE_ZYX, dtype=np.uint8)
    target_mask[6:10, 6:10, 6:10] = 1
    target_mask = sitk.GetImageFromArray(target_mask)
    target_mask.CopyInformation(water_ct.cube_hu)

    target = VOI(name="target", voi_type="TARGET", mask=target_mask, ct_image=water_ct)
    return StructureSet(vois=[target], ct_image=water_ct)


@pytest.fixture
def photon_pln():
    return create_pln(
        {
            "radiation_mode": "photons",
            "machine": "Generic",
            "prop_stf": {"bixel_width": 16.0},
            "prop_dose_calc": {
                "engine": "pyRadMC",
                "backend": "ref",
                "cross_section_model": "analytic",
                "transport_electrons": False,
                "num_histories_per_beamlet": 40,
                "n_batches": 2,
                "seed": 42,
                "dose_grid": {"resolution": {"x": 8.0, "y": 8.0, "z": 8.0}},
            },
        }
    )


@pytest.fixture
def photon_stf(water_ct, water_cst, photon_pln):
    generator = StfGeneratorPhotonIMRT(photon_pln)
    generator.bixel_width = 16.0
    generator.gantry_angles = [0.0, 90.0]
    generator.couch_angles = [0.0, 0.0]
    return validate_stf(generator.generate(water_ct, water_cst))


def test_engine_registered():
    engines = get_available_engines("photons")
    assert "pyRadMC" in engines
    assert engines["pyRadMC"] is PhotonPyRadMCEngine


def test_engine_config_from_pln(photon_pln):
    engine = get_engine(photon_pln)
    assert isinstance(engine, PhotonPyRadMCEngine)
    assert engine.backend == "ref"
    assert engine.cross_section_model == "analytic"
    assert engine.transport_electrons is False
    assert engine.num_histories_per_beamlet == 40
    assert engine.n_batches == 2
    assert engine.seed == 42


def test_concurrency_config_defaults(photon_pln):
    engine = get_engine(photon_pln)
    assert engine.warp_devices is None
    assert engine.beam_scheduling == "sharded"
    assert engine.concurrent_batches == 1
    assert engine.beamlet_group_size == 32


def test_beam_scheduling_resolution(photon_pln):
    """'beam_per_device' requires warp_devices and reduces to 'sharded' sensibly."""
    engine = get_engine(photon_pln)  # backend 'ref'
    assert engine._effective_beam_scheduling() == "sharded"

    # non-warp backend: ignored with a log line
    engine.beam_scheduling = "beam_per_device"
    assert engine._effective_beam_scheduling() == "sharded"

    engine.backend = "warp"
    with pytest.raises(ValueError, match="warp_devices"):
        engine._effective_beam_scheduling()

    engine.warp_devices = ["cpu"]
    assert engine._effective_beam_scheduling() == "sharded"

    engine.warp_devices = ["cuda:0", "cuda:1"]
    assert engine._effective_beam_scheduling() == "beam_per_device"


def test_transport_grid_from_ct(water_ct):
    from pyRadMC.data.materials import CORTICAL_BONE, WATER

    grid = transport_grid_from_ct(water_ct)

    # (x, y, z) axis order and mm -> cm conversion
    assert grid.shape == CT_SHAPE_ZYX[::-1]
    assert grid.spacing == tuple(s / 10.0 for s in CT_SPACING)

    # origin is the lower voxel corner in cm
    expected_origin = tuple((o - 0.5 * s) / 10.0 for o, s in zip(CT_ORIGIN, CT_SPACING))
    assert np.allclose(grid.origin, expected_origin)

    # the bone insert at numpy (z=4, y=5, x=6) lands at engine (x=6, y=5, z=4)
    assert grid.material[6, 5, 4] == CORTICAL_BONE
    assert grid.material[0, 0, 0] == WATER
    assert grid.density[6, 5, 4] > 1.5
    assert np.isclose(grid.density[0, 0, 0], 1.0)


def test_transport_grid_water_only(water_ct):
    from pyRadMC.data.materials import WATER

    grid = transport_grid_from_ct(water_ct, water_only=True)
    assert np.all(grid.material == WATER)
    # density still follows the HU ramp
    assert grid.density[6, 5, 4] > 1.5


def test_scoring_grid_from_dose_grid(water_ct):
    transport_grid = transport_grid_from_ct(water_ct)
    dose_grid = water_ct.grid

    scoring = scoring_grid_from_dose_grid(dose_grid, transport_grid)

    assert scoring.shape == tuple(dose_grid.dimensions)
    assert np.allclose(scoring.spacing, transport_grid.spacing)
    assert np.allclose(scoring.origin, transport_grid.origin)
    # identical geometry: mass equals density * voxel volume
    assert np.allclose(scoring.voxel_mass, transport_grid.density * transport_grid.voxel_volume)


def test_scoring_grid_rejects_nonidentity_direction(water_ct):
    transport_grid = transport_grid_from_ct(water_ct)
    dose_grid = water_ct.grid.model_copy(
        update={"direction": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)}
    )
    with pytest.raises(ValueError, match="direction"):
        scoring_grid_from_dose_grid(dose_grid, transport_grid)


def test_row_permutation_single_entry():
    nx, ny, nz = 4, 5, 6
    ix, iy, iz = 1, 2, 3

    engine_row = np.ravel_multi_index((ix, iy, iz), (nx, ny, nz))
    matrix = sparse.csc_array(
        (np.array([1.0]), np.array([engine_row]), np.array([0, 1])),
        shape=(nx * ny * nz, 1),
    )

    permuted = permute_rows_engine_to_grid(matrix, (nx, ny, nz))

    expected_row = np.ravel_multi_index((iz, iy, ix), (nz, ny, nx))
    assert permuted.indices.tolist() == [expected_row]
    assert permuted.data.tolist() == [1.0]


def test_row_permutation_dense_roundtrip():
    """Permuting a full random matrix reorders rows exactly as cube transposition."""
    rng = np.random.default_rng(0)
    nx, ny, nz = 3, 4, 5
    cube_engine = rng.random((nx, ny, nz))  # engine-frame cube (x, y, z)

    matrix = sparse.csc_array(cube_engine.reshape(-1, 1))
    permuted = permute_rows_engine_to_grid(matrix, (nx, ny, nz))

    expected = cube_engine.transpose(2, 1, 0).reshape(-1)  # numpy (z, y, x) cube ravel
    assert np.allclose(permuted.toarray().ravel(), expected)


def test_beamlet_source_geometry(photon_stf):
    from pyRadMC.geometry.spectrum import ali_rogers_mv

    spectrum = ali_rogers_mv("varian-6mv")

    beam = photon_stf.beams[0]  # gantry 0, couch 0
    source = beamlet_source_for_beam(beam, spectrum)

    assert source.n_beamlets == beam.total_number_of_bixels

    # gantry 0 / couch 0: rotation is identity, so the focal point is
    # iso + (0, -SAD, 0) in mm, divided by 10 for cm
    expected_focal = (beam.iso_center + np.array([0.0, -beam.sad, 0.0])) / 10.0
    assert np.allclose(source._fan.focal_point, expected_focal)
    assert np.allclose(source._fan.u_axis, [1.0, 0.0, 0.0])
    assert np.allclose(source._fan.v_axis, [0.0, 0.0, 1.0])
    assert np.isclose(source._fan.width_u, beam.bixel_width / 10.0)

    # aperture centers are the ray positions in the isocenter plane
    expected_centers = np.array([(beam.iso_center + ray.ray_pos) / 10.0 for ray in beam.rays])
    assert np.allclose(np.array(source._fan.centers), expected_centers)

    # gantry 90: BEV x-axis rotates onto world -y (active ccw rotation around z)
    beam90 = photon_stf.beams[1]
    source90 = beamlet_source_for_beam(beam90, spectrum)
    assert np.allclose(source90._fan.u_axis, [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(source90._fan.v_axis, [0.0, 0.0, 1.0], atol=1e-12)


def test_calc_dose_influence(water_ct, water_cst, photon_stf, photon_pln):
    engine = get_engine(photon_pln)

    dij = engine.calc_dose_influence(water_ct, water_cst, photon_stf)

    assert isinstance(dij, Dij)
    total_bixels = photon_stf.total_number_of_bixels
    assert dij.total_num_of_bixels == total_bixels

    dose = dij.physical_dose.flat[0]
    assert dose.shape == (dij.dose_grid.num_voxels, total_bixels)

    variance = dij.physical_dose_var.flat[0]
    assert variance.shape == dose.shape

    # bookkeeping: columns per beam in stf order
    n_first = photon_stf.beams[0].total_number_of_bixels
    assert np.all(dij.beam_num[:n_first] == 0)
    assert np.all(dij.beam_num[n_first:] == 1)

    # every beamlet deposited dose, all entries positive (Gy per history)
    col_sums = np.asarray(dose.sum(axis=0)).ravel()
    assert np.all(col_sums > 0.0)
    assert np.all(dose.data > 0.0)

    # dose must be deposited inside the target region
    target_indices = water_cst.vois[0].indices_numpy
    dose_cube = np.asarray(dose.sum(axis=1)).ravel()
    assert dose_cube[target_indices].sum() > 0.0


def test_calc_dose_forward_matches_influence(water_ct, water_cst, photon_stf, photon_pln):
    """Uniform-weight forward dose equals the influence matrix row sum."""
    engine = get_engine(photon_pln)
    dij = engine.calc_dose_influence(water_ct, water_cst, photon_stf)

    total_bixels = photon_stf.total_number_of_bixels
    weights = np.ones(total_bixels)

    # same histories per beamlet in direct mode
    forward_engine = get_engine(photon_pln)
    forward_engine.num_histories_direct = forward_engine.num_histories_per_beamlet * total_bixels
    result = forward_engine.calc_dose_forward(water_ct, water_cst, photon_stf, w=weights)

    forward_dose = sitk.GetArrayFromImage(result["physical_dose"]).ravel()

    influence_dose = np.asarray((dij.physical_dose.flat[0] @ weights.astype(np.float32))).ravel()

    # identical seeds and history counts: same transport, so results match
    # up to float32 accumulation order
    assert np.allclose(forward_dose, influence_dose, rtol=1e-4, atol=1e-10)


def test_machine_mc_beam_model_defaults():
    """The Generic machine defaults to the Siemens PRIMUS-like MC beam model."""
    from pyRadPlan.dose.engines import DoseEngineBase

    machine = DoseEngineBase.load_machine("photons", "Generic")
    beam_model = machine.mc_beam_model

    spectrum = beam_model.get_spectrum(6.0)
    assert spectrum.kind == "analytic_mv"
    assert spectrum.fitted_electron_energy == pytest.approx(6.83)

    assert len(beam_model.jaws) == 1
    assert beam_model.jaws[0].axis == "y"
    assert beam_model.jaws[0].focused

    assert beam_model.mlc is not None
    assert beam_model.mlc.travel_axis == "x"
    assert beam_model.mlc.num_leaf_pairs == 29
    assert beam_model.mlc.leaf_end_design == "double_focused"
    assert beam_model.mlc.z_top > beam_model.jaws[0].z_bottom
    # the MLC mid-plane (the collimation plane for geometric penumbra) sits at scd
    assert 0.5 * (beam_model.mlc.z_top + beam_model.mlc.z_bottom) == pytest.approx(machine.scd)


def test_spectrum_from_model_variants():
    from pyRadMC.geometry.spectrum import ali_rogers_mv

    from pyRadPlan.dose.engines._pyradmc import spectrum_from_model
    from pyRadPlan.machines.photons import (
        PhotonSpectrumAnalyticMV,
        PhotonSpectrumHistogram,
        PhotonSpectrumMonoenergetic,
    )

    analytic = PhotonSpectrumAnalyticMV(
        fitted_electron_energy=6.83, c1=1.184, c2=4.840, c3=-1.161, c4=0.01416
    )
    spectrum = spectrum_from_model(analytic)
    reference = ali_rogers_mv("siemens-6mv")
    assert spectrum.max_energy == pytest.approx(reference.max_energy)
    assert np.allclose(spectrum.bin_probabilities, reference.bin_probabilities)

    histogram = PhotonSpectrumHistogram(edges=[1.0, 2.0, 3.0], weights=[1.0, 3.0])
    spectrum = spectrum_from_model(histogram)
    assert spectrum.max_energy == pytest.approx(3.0)
    assert np.allclose(spectrum.bin_probabilities, [0.25, 0.75])

    mono = PhotonSpectrumMonoenergetic(energy=2.5)
    spectrum = spectrum_from_model(mono)
    assert spectrum.max_energy == pytest.approx(2.5)
    assert spectrum.mean_energy == pytest.approx(2.5, rel=1e-5)


def test_beam_limiting_stack_geometry(photon_stf):
    from pyRadMC.geometry.collimation import MLC, JawPair, project_between_planes

    from pyRadPlan.dose.engines import DoseEngineBase
    from pyRadPlan.dose.engines._pyradmc import beam_limiting_stack_for_beam

    machine = DoseEngineBase.load_machine("photons", "Generic")
    beam_model = machine.mc_beam_model
    beam = photon_stf.beams[0]  # gantry 0, couch 0

    stack = beam_limiting_stack_for_beam(beam, beam_model)

    # one Y jaw pair + the MLC, stacked ascending in z without overlap
    assert len(stack.devices) == 2
    jaw, mlc = stack.devices
    assert isinstance(jaw, JawPair) and isinstance(mlc, MLC)
    assert jaw.z_bottom <= mlc.z_top

    # gantry 0: frame axes are the identity-rotated BEV axes (v = w x u = -BEV z)
    assert np.allclose(stack.frame.u_axis, [1.0, 0.0, 0.0])
    assert np.allclose(stack.frame.v_axis, [0.0, 0.0, -1.0])
    assert np.allclose(stack.frame.w_axis, [0.0, 1.0, 0.0])
    expected_origin = (beam.iso_center + beam.source_point) / 10.0
    assert np.allclose(stack.frame.origin, expected_origin)

    # jaw edges: the ray bounding box +- half a bixel, projected to the mid-plane
    half = beam.bixel_width / 2.0
    ray_pos_bev = np.array([r.ray_pos_bev for r in beam.rays])
    q_v = ray_pos_bev @ np.array([0.0, 0.0, -1.0])  # local v coordinate
    z_mid_cm = 0.5 * (beam_model.jaws[0].z_top + beam_model.jaws[0].z_bottom) / 10.0
    sad_cm = beam.sad / 10.0
    assert jaw.axis == "v"
    assert jaw.edge_neg == pytest.approx(
        project_between_planes((q_v.min() - half) / 10.0, sad_cm, z_mid_cm)
    )
    assert jaw.edge_pos == pytest.approx(
        project_between_planes((q_v.max() + half) / 10.0, sad_cm, z_mid_cm)
    )

    # a focal-point ray through the field center passes every device untouched,
    # a far off-axis ray is blocked by material
    focal = np.asarray(stack.frame.origin)
    iso = beam.iso_center / 10.0
    for target, blocked in ((iso, False), (iso + np.array([15.0, 0.0, 0.0]), True)):
        direction = target - focal
        direction = direction / np.linalg.norm(direction)
        lengths = stack.path_lengths(focal.reshape(1, 3), direction.reshape(1, 3))
        if blocked:
            assert lengths.sum() > 1.0
        else:
            assert lengths.sum() == pytest.approx(0.0, abs=1e-9)

    # leaf pairs far outside the field are parked closed
    assert mlc.tips_neg[0] == mlc.tips_pos[0]
    assert mlc.tips_neg[-1] == mlc.tips_pos[-1]
    open_pairs = sum(pos > neg for neg, pos in zip(mlc.tips_neg, mlc.tips_pos))
    assert open_pairs > 0


def test_calc_dose_forward_head_presolve(water_ct, water_cst, photon_pln):
    """The opt-in head-presolve forward path places a collimated field correctly."""
    photon_pln.prop_dose_calc["forward_mode"] = "head_presolve"
    photon_pln.prop_dose_calc["num_histories_direct"] = 20000

    # single gantry-0 beam: travels along +y, laterally confined by the stack
    generator = StfGeneratorPhotonIMRT(photon_pln)
    generator.bixel_width = 16.0
    generator.gantry_angles = [0.0]
    generator.couch_angles = [0.0]
    stf = validate_stf(generator.generate(water_ct, water_cst))

    engine = get_engine(photon_pln)
    assert engine.forward_mode == "head_presolve"

    weights = np.ones(stf.total_number_of_bixels)
    result = engine.calc_dose_forward(water_ct, water_cst, stf, weights)
    cube = sitk.GetArrayFromImage(result["physical_dose"])  # (z, y, x)

    assert cube.min() >= 0.0
    assert cube.max() > 0.0

    # the field is confined laterally: profile far outside the ray bounding box
    # stays below a few percent of the in-field level (collimation + divergence)
    beam = stf.beams[0]
    half = beam.bixel_width / 2.0
    ray_pos_bev = np.array([r.ray_pos_bev for r in beam.rays])
    x_world = np.arange(cube.shape[2]) * 8.0 - 60.0
    x_bev = x_world - beam.iso_center[0]
    x_profile = cube.sum(axis=(0, 1))
    in_field = (x_bev > ray_pos_bev[:, 0].min() - half) & (x_bev < ray_pos_bev[:, 0].max() + half)
    far_outside = np.abs(x_bev) > (ray_pos_bev[:, 0].max() + 3 * half)
    assert x_profile[far_outside].max() < 0.05 * x_profile[in_field].mean()

    # the beam enters at low y: the entrance half carries more dose than the exit
    y_profile = cube.sum(axis=(0, 2))
    assert y_profile[:8].sum() > y_profile[8:].sum()


def test_field_aperture_from_blds(photon_stf):
    """BLD settings (IEC, at iso) map into BEV apertures with the X flip."""
    from pyRadPlan.machines import create_bld

    from pyRadPlan.dose.engines._pyradmc import field_aperture_for_beam

    beam = photon_stf.beams[0]

    mlc = create_bld(
        {
            "device_type": "MLC",
            "device_orientation": "X",
            "leaf_position_boundaries": [-20.0, -10.0, 0.0],
            "leaf_positions": [[-30.0, 10.0], [0.0, 0.0], [-15.0, 25.0]],
            "leaf_width": 10.0,
        }
    )
    jaw = create_bld(
        {
            "device_type": "JAW",
            "device_orientation": "Y",
            "positions": [-12.0, 18.0],
            "field_width": 80.0,
        }
    )

    aperture = field_aperture_for_beam(beam, [mlc, jaw])

    # IEC Y maps to BEV z unchanged
    assert aperture.jaws["y"] == pytest.approx((-12.0, 18.0))

    # MLC travels along IEC X -> machine axis 'x'; strips along IEC Y = BEV z
    assert aperture.mlc_travel_axis == "x"
    strips = aperture.mlc_strips
    assert len(strips) == 3
    # strip bounds unchanged (Y axis); openings flip sign (IEC X = -BEV x)
    assert strips[0][:2] == pytest.approx((-20.0, -10.0))
    assert strips[0][2] == pytest.approx((-10.0, 30.0))  # [-30, 10] flipped
    assert strips[1][2] is None  # closed pair
    assert strips[2][2] == pytest.approx((-25.0, 15.0))  # [-15, 25] flipped

    # box: x from open leaf extents, z from open strips intersected with the jaw
    assert aperture.box["x"] == pytest.approx((-25.0, 30.0))
    assert aperture.box["y"] == pytest.approx((-12.0, 10.0))


def test_stack_from_configured_blds(photon_stf):
    """Configured leaf/jaw settings land on the projected device planes."""
    from pyRadMC.geometry.collimation import project_between_planes

    from pyRadPlan.dose.engines import DoseEngineBase
    from pyRadPlan.dose.engines._pyradmc import (
        beam_limiting_stack_for_beam,
        field_aperture_for_beam,
    )
    from pyRadPlan.machines import create_bld

    machine = DoseEngineBase.load_machine("photons", "Generic")
    beam = photon_stf.beams[0]  # gantry 0

    mlc = create_bld(
        {
            "device_type": "MLC",
            "device_orientation": "X",
            "leaf_position_boundaries": [-20.0, -10.0, 0.0, 10.0],
            "leaf_positions": [[-30.0, 10.0], [0.0, 0.0], [-15.0, 25.0], [-30.0, 30.0]],
            "leaf_width": 10.0,
        }
    )
    jaw = create_bld(
        {
            "device_type": "JAW",
            "device_orientation": "Y",
            "positions": [-12.0, 18.0],
            "field_width": 80.0,
        }
    )

    aperture = field_aperture_for_beam(beam, [mlc, jaw])
    stack = beam_limiting_stack_for_beam(beam, machine.mc_beam_model, aperture=aperture)

    jaw_device, mlc_device = stack.devices
    geometry = machine.mc_beam_model

    def at_plane(value_iso_mm, device_geometry):
        z_mid_cm = 0.5 * (device_geometry.z_top + device_geometry.z_bottom) / 10.0
        return project_between_planes(value_iso_mm / 10.0, beam.sad / 10.0, z_mid_cm)

    # Y jaw: local v = -BEV z, so BEV (-12, 18) becomes local (-18, 12)
    assert jaw_device.edge_neg == pytest.approx(at_plane(-18.0, geometry.jaws[0]))
    assert jaw_device.edge_pos == pytest.approx(at_plane(12.0, geometry.jaws[0]))

    # MLC: 4 configured pairs replace the machine's 29
    assert mlc_device.n_pairs == 4
    # strips flip along v (v = -BEV z) but the boundary set is symmetric: -20..20
    assert mlc_device.leaf_edges_v[0] == pytest.approx(at_plane(-20.0, geometry.mlc))
    assert mlc_device.leaf_edges_v[-1] == pytest.approx(at_plane(20.0, geometry.mlc))
    # first local strip is IEC pair 3 ([0,10) in Y): opening [-30,30] -> BEV x (-30,30)
    assert mlc_device.tips_neg[0] == pytest.approx(at_plane(-30.0, geometry.mlc))
    assert mlc_device.tips_pos[0] == pytest.approx(at_plane(30.0, geometry.mlc))
    # the closed IEC pair 1 ([-10,0) in Y -> local strip (0,10)) is parked
    assert mlc_device.tips_neg[2] == mlc_device.tips_pos[2]
