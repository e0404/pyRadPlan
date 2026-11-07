import pytest
import SimpleITK as sitk
import numpy as np

from pyRadPlan import PhotonPlan, generate_stf, load_tg119
from pyRadPlan.ct import create_ct
from pyRadPlan.cst import create_cst
from pyRadPlan.dose import calc_dose_forward
from pyRadPlan.dose.engines import get_engine
from pyRadPlan.machines import create_bld, load_from_name
from pyRadPlan.visualization._plot_slice import plot_slice


# test/data/photons_testData.mat is byte-identical to matRad's test/testData/photons_testData.mat;
# its reference dose is computed without dij sampling, which matRad's random numbers would
# make irreproducible here
def test_photons(test_data_photons):
    pln, ct, cst, stf, dij, result = test_data_photons
    pln.prop_dose_calc["enable_dij_sampling"] = False
    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])
    result_matRad = np.transpose(result["physicalDose"], (2, 0, 1))
    mask = (result_py > 0) & (result_matRad > 0)
    assert np.allclose(result_py[mask], result_matRad[mask], rtol=1e-2)


def test_photons_field_based():
    """Exercise the per-ray kernel interpolation path (_init_ray) on the configured backend.

    Field-based dose calculation is the one configuration that reaches ``_init_ray``'s
    custom fluence branch, which builds its own kernel interpolators via FFT convolution.
    The default configuration skips it entirely, so it needs its own coverage.
    """
    ct, cst = load_tg119()

    leaf_width = 5.0
    positions = [[-20.0, 20.0]] * 4
    boundaries = np.arange(-2 * leaf_width, 2 * leaf_width, leaf_width)

    mlc = create_bld(
        {
            "device_type": "MLC",
            "device_orientation": "X",
            "leaf_position_boundaries": boundaries,
            "leaf_positions": positions,
            "leaf_width": leaf_width,
            "leaf_leakage": 0.1,
        }
    )

    pln = PhotonPlan(machine="Generic")
    pln.prop_stf = {
        "gantry_angles": [0.0],
        "couch_angles": [0.0],
        "generator": "photonSingleBixel",
        "field_based": True,
        "blds": [mlc],
        "resolution": 2.0,
        "energy": 6.0,
    }
    pln.prop_dose_calc["dosimetric_lateral_cutoff"] = 0.995
    pln.prop_dose_calc["lateral_model"] = "single"

    stf = generate_stf(ct, cst, pln)
    masks = [b.mask.copy() for beam in stf.beams for ray in beam.rays for b in ray.beamlets]
    dose = sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln)["physical_dose"])

    # The field shapes are resampled for the calculation only, not in the caller's stf
    after = [b.mask for beam in stf.beams for ray in beam.rays for b in ray.beamlets]
    assert all(np.array_equal(a, b) for a, b in zip(after, masks))

    assert np.all(np.isfinite(dose))
    assert np.all(dose >= 0.0)
    assert dose.max() > 0.0

    # Keeping this for debugging:
    # plot_slice(
    #     image_volume=ct,
    #     cst=cst,
    #     overlay=result_matRad_rot,#-result_matRad_rot,
    #     view_slice=5,
    #     plane="axial",
    #     overlay_unit="Gy",
    #     plt_show = True,
    #     use_global_max = False,
    # )


def test_bixel_based_lateral_cutoff_includes_half_diagonal(test_data_photons):
    """matRad widens the lateral cutoff by bixelWidth/sqrt(2) for bixel-based dose too."""
    pln, ct, cst, stf, _, _ = test_data_photons

    engine = get_engine(pln)
    engine.calc_dose_forward(ct, cst, stf, None)

    expected = engine.geometric_lateral_cutoff + stf.beams[0].bixel_width / np.sqrt(2)
    assert engine._effective_lateral_cutoff == pytest.approx(expected)


def test_bixel_dose_is_mirror_symmetric(test_data_photons):
    """Opposed beams on the symmetric test data give a mirror-symmetric dose."""
    pln, ct, cst, stf, _, _ = test_data_photons
    pln.prop_dose_calc["enable_dij_sampling"] = False
    dose = sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln)["physical_dose"])
    assert np.allclose(dose, dose[::-1, ::-1, ::-1], atol=1e-3 * dose.max())


def test_bixel_based_custom_primary_fluence(test_data_photons):
    """A uniform fluence matches the default bixel dose, the primary fluence changes it."""
    pln, ct, cst, stf, _, _ = test_data_photons
    pln.prop_dose_calc["enable_dij_sampling"] = False
    default = sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln)["physical_dose"])

    pln.prop_dose_calc["use_custom_primary_photon_fluence"] = True
    pln.prop_dose_calc["force_uniform_fluence"] = True
    uniform = sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln)["physical_dose"])
    assert np.allclose(uniform, default)

    pln.prop_dose_calc["force_uniform_fluence"] = False
    custom = sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln)["physical_dose"])
    assert np.all(np.isfinite(custom))
    assert not np.allclose(custom, default)


def _water_cube(n, resolution, target_radius):
    """Water cube of n^3 voxels with a spherical target (radius in voxels) on the center voxel."""
    img = sitk.Image(n, n, n, sitk.sitkInt16)
    img.SetSpacing((resolution, resolution, resolution))
    ct = create_ct(cube_hu=img)

    c = (n - 1) / 2
    g = np.arange(n)
    yy, xx, zz = np.meshgrid(g, g, g, indexing="ij")
    target = (xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2 <= target_radius**2
    cst = create_cst(
        [
            [0, "Body", "OAR", np.arange(1, n**3 + 1, dtype=float), 2, []],
            [1, "Target", "TARGET", np.flatnonzero(target.ravel(order="F")) + 1.0, 1, []],
        ],
        ct=ct,
    )
    return ct, cst


@pytest.fixture(scope="module")
def water_sphere():
    """Water cube of 51^3 voxels at 2 mm with a centered spherical target of radius 34 mm."""
    return _water_cube(51, 2.0, 17)


@pytest.fixture(scope="module")
def ramp_fluence_machine():
    """Load the Generic photon machine with a primary fluence rising linearly off axis."""
    machine = load_from_name("photons", "Generic")
    r = np.arange(401.0)
    for kernel in machine.pb_kernels.values():
        kernel.primary_fluence = np.column_stack([r, r])
    return machine


def _single_bixel_dose(water_sphere, machine, gantry_angle, ray_pos_bev, custom_fluence):
    """Dose of the single 5 mm bixel at ray_pos_bev [mm] and the voxel centroid in world x, y, z."""
    ct, cst = water_sphere
    pln = PhotonPlan(machine=machine)
    pln.prop_stf = {
        "gantry_angles": [float(gantry_angle)],
        "couch_angles": [0.0],
        "bixel_width": 5.0,
    }
    pln.prop_dose_calc["enable_dij_sampling"] = False
    pln.prop_dose_calc["use_custom_primary_photon_fluence"] = custom_fluence
    pln.prop_dose_calc["dose_grid"] = {"resolution": {"x": 2.0, "y": 2.0, "z": 2.0}}

    stf = generate_stf(ct, cst, pln)
    beam = stf.beams[0]
    beam.rays = [r for r in beam.rays if np.allclose(r.ray_pos_bev, ray_pos_bev, atol=1e-6)]
    assert len(beam.rays) == 1

    image = calc_dose_forward(ct, cst, stf, pln, weights=np.ones(1))["physical_dose"]
    dose = sitk.GetArrayFromImage(image)
    zyx = (np.indices(dose.shape).reshape(3, -1) @ dose.ravel()) / dose.sum()
    centroid = np.array(image.TransformContinuousIndexToPhysicalPoint(zyx[::-1].tolist()))
    return dose, centroid


@pytest.mark.parametrize(
    "ray_pos_bev", [[-30.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 0.0, -30.0], [0.0, 0.0, 30.0]]
)
def test_custom_primary_fluence_tilt_direction(water_sphere, ramp_fluence_machine, ray_pos_bev):
    """A fluence rising off axis pulls an off-axis bixel's dose outward, along its own offset.

    At gantry 0 the beam's eye view x and z are the world x and z.
    """
    _, uniform = _single_bixel_dose(water_sphere, ramp_fluence_machine, 0, ray_pos_bev, False)
    _, custom = _single_bixel_dose(water_sphere, ramp_fluence_machine, 0, ray_pos_bev, True)
    shift = custom - uniform

    axis = 0 if ray_pos_bev[0] != 0.0 else 2
    other = 2 - axis
    assert np.sign(shift[axis]) == np.sign(ray_pos_bev[axis])
    assert abs(shift[other]) < 1e-3 * abs(shift[axis])


def test_custom_primary_fluence_gantry_invariance(water_sphere, ramp_fluence_machine):
    """In a phantom symmetric under gantry rotation, an off-axis bixel's fluence is the same."""
    integral = [
        _single_bixel_dose(water_sphere, ramp_fluence_machine, g, [30.0, 0.0, 0.0], True)[0].sum()
        for g in (0, 90, 180, 270)
    ]
    assert np.allclose(integral, integral[0], rtol=1e-3)


def _plan_with_20mm_cutoffs(prop_stf):
    """Photon plan on a 1 mm dose grid with 20 mm kernel and lateral cutoffs, no dij sampling."""
    pln = PhotonPlan(machine="Generic")
    pln.prop_stf = prop_stf
    pln.prop_dose_calc["enable_dij_sampling"] = False
    pln.prop_dose_calc["geometric_lateral_cutoff"] = 20.0
    pln.prop_dose_calc["kernel_cutoff"] = 20.0
    pln.prop_dose_calc["dose_grid"] = {"resolution": {"x": 1.0, "y": 1.0, "z": 1.0}}
    return pln


def _max_mirror_difference(dose):
    """Largest difference to the mirror image along x or z, relative to the maximum dose."""
    # Image arrays are (z, y, x)
    return max(np.abs(dose - dose[:, :, ::-1]).max(), np.abs(dose - dose[::-1]).max()) / dose.max()


def test_bixel_dose_mirror_symmetric_with_kernel_cutoff():
    """A cut-off kernel and the Gaussian penumbra filter are sampled symmetrically."""
    ct, cst = _water_cube(41, 1.0, 5)
    pln = _plan_with_20mm_cutoffs(
        {"gantry_angles": [0.0], "couch_angles": [0.0], "bixel_width": 5.0}
    )

    stf = generate_stf(ct, cst, pln)
    beam = stf.beams[0]
    beam.rays = [r for r in beam.rays if np.allclose(r.ray_pos_bev, 0.0)]
    assert len(beam.rays) == 1

    dose = calc_dose_forward(ct, cst, stf, pln, weights=np.ones(1))["physical_dose"]
    assert _max_mirror_difference(sitk.GetArrayFromImage(dose)) < 1e-5


def test_field_dose_mirror_symmetric_with_kernel_cutoff():
    """The same holds for a symmetric 20 mm x 20 mm field in field-based dose calculation."""
    ct, cst = _water_cube(41, 1.0, 5)
    leaf_width = 5.0
    mlc = create_bld(
        {
            "device_type": "MLC",
            "device_orientation": "X",
            "leaf_position_boundaries": np.arange(-2 * leaf_width, 2 * leaf_width, leaf_width),
            "leaf_positions": [[-10.0, 10.0]] * 4,
            "leaf_width": leaf_width,
            "leaf_leakage": 0.0,
        }
    )
    pln = _plan_with_20mm_cutoffs(
        {
            "gantry_angles": [0.0],
            "couch_angles": [0.0],
            "generator": "photonSingleBixel",
            "field_based": True,
            "blds": [mlc],
            "resolution": 1.0,
            "energy": 6.0,
        }
    )

    stf = generate_stf(ct, cst, pln)
    dose = calc_dose_forward(ct, cst, stf, pln)["physical_dose"]
    assert _max_mirror_difference(sitk.GetArrayFromImage(dose)) < 1e-5
