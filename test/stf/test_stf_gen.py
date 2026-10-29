import pytest
import sys

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+

import numpy as np
import SimpleITK as sitk

from pyRadPlan.io import load_patient
from pyRadPlan.machines import load_from_name
from pyRadPlan.plan import create_pln
from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import validate_cst, StructureSet, VOI

from pyRadPlan.stf import Ray, get_available_generators, get_generator, resolve_iso_centers
from pyRadPlan.stf.generators import (
    StfGeneratorBase,
    StfGeneratorExternalBeamRayBixel,
    StfGeneratorPhotonIMRT,
    StfGeneratorPhotonCollimatedSquareFields,
    StfGeneratorIonSingleSpot,
    StfGeneratorIMPT,
    StfGeneratorVHEE,
)


@pytest.fixture
def sample_ct():
    image = sitk.GetImageFromArray(np.random.rand(5, 15, 25) * 1000)  # Random HU values
    image.SetOrigin((0, 0, 0))
    image.SetSpacing((2, 3, 4))  # Irregular spacing for test
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    ct = CT(cube_hu=image)

    return ct


@pytest.fixture
def sample_cst(sample_ct):
    organ_mask = np.zeros((5, 15, 25), dtype=np.uint8)
    organ_mask[1:3, 3:11, 5:20] = 1
    organ_mask = sitk.GetImageFromArray(organ_mask)

    target_mask = np.zeros((5, 15, 25), dtype=np.uint8)
    target_mask[2, 5:9, 10:15] = 1
    target_mask = sitk.GetImageFromArray(target_mask)

    target1 = VOI(name="testtarget", voi_type="TARGET", mask=target_mask, ct_image=sample_ct)
    organ1 = VOI(name="testorgan", voi_type="OAR", mask=organ_mask, ct_image=sample_ct)

    cst = StructureSet(vois=[target1, organ1], ct_image=sample_ct)

    return cst


@pytest.fixture
def tg119():
    path = resources.files("pyRadPlan.data.phantoms")
    ct, cst = load_patient(path.joinpath("TG119.mat"))
    return {"ct": ct, "cst": cst}


@pytest.fixture
def sample_photon_pln_dict():
    pln = {
        "radiation_mode": "photons",
        "prop_stf": {"bixel_width": 2.0},
        "machine": "Generic",
    }
    return pln


@pytest.fixture
def sample_photon_pln(sample_photon_pln_dict):
    pln = create_pln(sample_photon_pln_dict)
    return pln


@pytest.fixture
def sample_proton_pln_dict():
    pln = {
        "radiation_mode": "protons",
        "prop_stf": {"bixel_width": 2.0},
        "machine": "Generic",
    }
    return pln


@pytest.fixture
def sample_proton_pln(sample_proton_pln_dict):
    pln = create_pln(sample_proton_pln_dict)
    return pln


def test_available_generators(sample_proton_pln):
    generators = get_available_generators(sample_proton_pln)
    assert isinstance(generators, dict)
    assert len(generators.items()) > 0


def test_get_generator(sample_proton_pln):
    sample_proton_pln.prop_stf["generator"] = "IMPT"
    generator = get_generator(sample_proton_pln)
    assert generator
    assert isinstance(generator, StfGeneratorBase)


def test_get_generator_default(sample_proton_pln):
    generator = get_generator(sample_proton_pln)
    assert generator
    assert isinstance(generator, StfGeneratorBase)


def test_get_generator_invalid(sample_proton_pln):
    sample_proton_pln.prop_stf["generator"] = "InvalidGenerator"
    with pytest.warns(UserWarning):
        generator = get_generator(sample_proton_pln)


def test_basic_photon_imrt_construct(sample_photon_pln_dict, sample_photon_pln):
    stf_gen = StfGeneratorPhotonIMRT()
    assert isinstance(
        stf_gen, (StfGeneratorBase, StfGeneratorExternalBeamRayBixel, StfGeneratorPhotonIMRT)
    )

    stf_gen = StfGeneratorPhotonIMRT(sample_photon_pln)
    assert isinstance(
        stf_gen, (StfGeneratorBase, StfGeneratorExternalBeamRayBixel, StfGeneratorPhotonIMRT)
    )
    assert stf_gen.bixel_width == sample_photon_pln.prop_stf["bixel_width"]

    stf_gen = StfGeneratorPhotonIMRT(sample_photon_pln_dict)
    assert isinstance(
        stf_gen, (StfGeneratorBase, StfGeneratorExternalBeamRayBixel, StfGeneratorPhotonIMRT)
    )
    assert stf_gen.bixel_width == sample_photon_pln.prop_stf["bixel_width"]


def test_basic_photon_imrt(sample_ct, sample_cst, sample_photon_pln):
    stf_gen = StfGeneratorPhotonIMRT(sample_photon_pln)

    stf_gen.gantry_angles = [90.0, 270.0]
    stf_gen.couch_angles = [0.0, 90.0]

    stf = stf_gen.generate(sample_ct, sample_cst)

    assert len(stf) == 2
    for i in range(len(stf)):
        assert isinstance(stf[i], dict)
        assert stf[i]["gantry_angle"] == stf_gen.gantry_angles[i]
        assert stf[i]["couch_angle"] == stf_gen.couch_angles[i]
        assert stf[i]["bixel_width"] == stf_gen.bixel_width
        assert "rays" in stf[i]
        assert isinstance(stf[i]["rays"], list)

        for ray in stf[i]["rays"]:
            assert isinstance(ray, dict)
            assert "ray_pos_bev" in ray
            assert isinstance(ray["ray_pos_bev"], np.ndarray)
            assert ray["ray_pos_bev"].size == 3


def test_basic_photon_collimated_field(sample_ct, sample_cst, sample_photon_pln):
    stf_gen = StfGeneratorPhotonCollimatedSquareFields(sample_photon_pln)

    stf_gen.gantry_angles = [0.0, 270.0]
    stf_gen.couch_angles = [0.0, 90.0]

    expected_source_points_bev = [
        np.array([0, -1000, 0], dtype=float),
        np.array([0, -1000, 0], dtype=float),
    ]

    expected_source_points = [
        np.array([0, -1000.0, 0], dtype=float),
        np.array(
            [
                0,
                0,
                1000.0,
            ],
            dtype=float,
        ),
    ]

    expected_ray_pos_bev = [np.array([0, 0, 0], dtype=float), np.array([0, 0, 0], dtype=float)]

    expected_ray_pos = [
        np.array([[0, 0, 0], [0, 0, 0]], dtype=float),
        np.array([[0, 0, 0], [0, 0, 0]], dtype=float),
    ]

    stf = stf_gen.generate(sample_ct, sample_cst)

    assert len(stf) == 2
    for i, field in enumerate(stf):
        assert isinstance(field, dict)
        assert field["gantry_angle"] == stf_gen.gantry_angles[i]
        assert field["couch_angle"] == stf_gen.couch_angles[i]
        assert "field_width" in field
        assert "bixel_width" in field
        assert field["bixel_width"] == stf_gen.bixel_width
        assert "source_point" in field
        assert isinstance(field["source_point"], np.ndarray)
        assert field["source_point"].size == 3
        assert "source_point_bev" in field
        assert isinstance(field["source_point_bev"], np.ndarray)
        assert field["source_point_bev"].size == 3
        assert np.isclose(field["source_point_bev"], expected_source_points_bev[i]).all()
        assert np.isclose(field["source_point"], expected_source_points[i]).all()
        assert np.isclose(field["rays"][0]["ray_pos_bev"], expected_ray_pos_bev[i]).all()
        assert np.isclose(field["rays"][0]["ray_pos"], expected_ray_pos[i]).all()


def test_single_ion_spot(sample_ct, sample_cst, sample_proton_pln):
    stf_gen = StfGeneratorIonSingleSpot(sample_proton_pln)

    stf_gen.gantry_angles = [0.0]
    stf_gen.couch_angles = [0.0]
    stf_gen.fix_energy = 120
    sad = 10000.0
    expected_source_points_bev = [np.array([0, -sad, 0], dtype=float)]

    expected_source_points = [np.array([0, -sad, 0], dtype=float)]

    expected_ray_pos_bev = np.array([0, 0, 0], dtype=float)

    expected_ray_pos = np.array([0, 0, 0], dtype=float)

    stf = stf_gen.generate(sample_ct, sample_cst)

    assert len(stf) == 1
    for i, field in enumerate(stf):
        assert isinstance(field, dict)
        assert field["gantry_angle"] == stf_gen.gantry_angles[i]
        assert field["couch_angle"] == stf_gen.couch_angles[i]
        assert "bixel_width" in field
        assert field["bixel_width"] == stf_gen.bixel_width
        assert "source_point" in field
        assert isinstance(field["source_point"], np.ndarray)
        assert field["source_point"].size == 3
        assert "source_point_bev" in field
        assert isinstance(field["source_point_bev"], np.ndarray)
        assert field["source_point_bev"].size == 3
        assert np.isclose(field["source_point_bev"], expected_source_points_bev[i]).all()
        assert np.isclose(field["source_point"], expected_source_points[i]).all()
        assert np.isclose(field["rays"][0].ray_pos_bev, expected_ray_pos_bev).all()
        assert np.isclose(field["rays"][0].ray_pos, expected_ray_pos).all()


@pytest.mark.parametrize("default_mode", [False, True], ids=["loaded", "default-mode"])
@pytest.mark.parametrize("radiation_mode", ["protons", "photons"])
def test_generate_with_round_tripped_plan_machine(
    sample_ct, sample_cst, radiation_mode, default_mode
):
    """Steering generation works with a loaded machine after a plan dump round trip."""
    machine = load_from_name(radiation_mode, "Generic")
    if default_mode:
        data = machine.model_dump(exclude_unset=True)
        del data["radiation_mode"]
        machine = type(machine).model_validate(data)
    pln = create_pln(radiation_mode=radiation_mode, machine=machine, prop_stf={"bixel_width": 5})
    pln = create_pln(pln.model_dump())

    stf_gen = get_generator(pln)
    stf_gen.gantry_angles = [0.0]
    stf_gen.couch_angles = [0.0]
    stf = stf_gen.generate(sample_ct, sample_cst)

    assert type(stf_gen.machine) is type(machine)
    assert len(stf) == 1
    assert stf[0]["machine"] == "Generic"


def test_impt(sample_ct, sample_cst, sample_proton_pln):
    stf_gen = StfGeneratorIMPT(sample_proton_pln)

    stf_gen.gantry_angles = [0.0, 270.0]
    stf_gen.couch_angles = [0.0, 90.0]

    stf = stf_gen.generate(sample_ct, sample_cst)

    assert len(stf) == 2
    for i, field in enumerate(stf):
        assert isinstance(field, dict)
        assert field["gantry_angle"] == stf_gen.gantry_angles[i]
        assert field["couch_angle"] == stf_gen.couch_angles[i]
        assert "bixel_width" in field
        assert field["bixel_width"] == stf_gen.bixel_width
        assert "source_point" in field
        assert isinstance(field["source_point"], np.ndarray)
        assert field["source_point"].size == 3
        assert "source_point_bev" in field
        assert isinstance(field["source_point_bev"], np.ndarray)
        assert field["source_point_bev"].size == 3
        assert len(field["rays"]) > 0
        for ray in field["rays"]:
            assert isinstance(ray, Ray)


def test_impt_tg119(tg119, sample_proton_pln):
    ct = tg119["ct"]
    cst = tg119["cst"]

    stf_gen = StfGeneratorIMPT(sample_proton_pln)
    stf_gen.bixel_width = 5.0
    stf = stf_gen.generate(ct, cst)

    assert isinstance(stf, list)
    assert len(stf) == 1

    for i, field in enumerate(stf):
        assert isinstance(field, dict)
        assert "bixel_width" in field
        assert field["bixel_width"] == stf_gen.bixel_width
        assert "source_point" in field
        assert isinstance(field["source_point"], np.ndarray)
        assert field["source_point"].size == 3
        assert "source_point_bev" in field
        assert isinstance(field["source_point_bev"], np.ndarray)
        assert field["source_point_bev"].size == 3
        assert len(field["rays"]) > 0
        for ray in field["rays"]:
            assert isinstance(ray, Ray)


def test_ion_generators_use_no_pb_margin(sample_proton_pln, sample_photon_pln):
    """matRad adds a bixel-width margin for photon IMRT only; ion generators use none."""
    assert StfGeneratorIMPT(sample_proton_pln)._computed_target_margin() == 0.0
    assert StfGeneratorIonSingleSpot(sample_proton_pln)._computed_target_margin() == 0.0

    photon_gen = StfGeneratorPhotonIMRT(sample_photon_pln)
    assert photon_gen._computed_target_margin() == photon_gen.bixel_width


def test_impt_longitudinal_spot_spacing_thins_energy_layers(tg119, sample_proton_pln):
    """Energy layers must be spaced by at least longitudinal_spot_spacing in peak position."""
    ct, cst = tg119["ct"], tg119["cst"]

    def spots_and_layers(spacing):
        stf_gen = StfGeneratorIMPT(sample_proton_pln)
        stf_gen.bixel_width = 5.0
        stf_gen.longitudinal_spot_spacing = spacing
        beams = stf_gen.generate(ct, cst)
        spots = sum(len(ray.beamlets) for beam in beams for ray in beam["rays"])
        layers = np.unique(
            np.concatenate([ray.energies for beam in beams for ray in beam["rays"]])
        )
        return spots, layers, stf_gen

    dense_spots, dense_layers, gen = spots_and_layers(0.0)
    sparse_spots, sparse_layers, _ = spots_and_layers(10.0)

    # A wider spacing must remove layers, not add them
    assert sparse_spots < dense_spots
    assert sparse_layers.size < dense_layers.size

    # The surviving layers really are at least the requested distance apart
    peak_pos = np.sort(
        [
            gen._available_peak_positions[np.argmin(np.abs(gen._available_energies - e))]
            for e in sparse_layers
        ]
    )
    assert np.all(np.diff(peak_pos) >= 10.0 - 1.0)


def test_impt_zero_spot_spacing_keeps_all_layers(tg119, sample_proton_pln):
    """A spacing of zero disables the thinning entirely."""
    ct, cst = tg119["ct"], tg119["cst"]

    counts = []
    for spacing in (0.0, 1e-9):
        stf_gen = StfGeneratorIMPT(sample_proton_pln)
        stf_gen.bixel_width = 5.0
        stf_gen.longitudinal_spot_spacing = spacing
        beams = stf_gen.generate(ct, cst)
        counts.append(sum(len(ray.beamlets) for beam in beams for ray in beam["rays"]))

    assert counts[0] == counts[1]


def test_target_margin_stays_inside_the_patient(sample_ct, sample_cst):
    """matRad_addMargin only grows the target into voxels that belong to some VOI."""
    pln = create_pln(
        {"radiation_mode": "photons", "machine": "Generic", "prop_stf": {"bixel_width": 5.0}}
    )
    stf_gen = StfGeneratorPhotonIMRT(pln)
    stf_gen._ct = sample_ct
    stf_gen._cst = sample_cst
    stf_gen._initialize()
    stf_gen._initialize_patient_geometry()

    target = sitk.GetArrayFromImage(stf_gen._target_mask).astype(bool)
    patient = sitk.GetArrayFromImage(stf_gen._patient_mask).astype(bool)

    assert target.any()
    assert not (target & ~patient).any()


def test_ion_energy_selection_ignores_densities_outside_patient(
    sample_ct, sample_cst, sample_proton_pln
):
    """With ignore_outside_densities, densities outside every contour are zeroed."""
    stf_gen = StfGeneratorIMPT(sample_proton_pln)
    stf_gen.ignore_outside_densities = True
    stf_gen._ct = sample_ct
    stf_gen._cst = sample_cst
    stf_gen._initialize()
    stf_gen._initialize_patient_geometry()

    wet = sitk.GetArrayFromImage(stf_gen._wet_image)
    patient = sitk.GetArrayFromImage(stf_gen._patient_mask).astype(bool)

    assert (wet[~patient] == 0).all()
    assert (wet[patient] > 0).any()


def test_ion_energy_selection_traces_outside_densities_by_default(
    sample_ct, sample_cst, sample_proton_pln
):
    """By default the energy selection traces all densities, as the dose engines do."""
    stf_gen = StfGeneratorIMPT(sample_proton_pln)
    assert stf_gen.ignore_outside_densities is False
    stf_gen._ct = sample_ct
    stf_gen._cst = sample_cst
    stf_gen._initialize()
    stf_gen._initialize_patient_geometry()

    wet = sitk.GetArrayFromImage(stf_gen._wet_image)
    patient = sitk.GetArrayFromImage(stf_gen._patient_mask).astype(bool)

    assert (wet[~patient] > 0).any()


@pytest.mark.parametrize("generator_cls", [StfGeneratorIMPT, StfGeneratorVHEE])
def test_use_given_wet_image_is_deprecated_alias(generator_cls):
    stf_gen = generator_cls()
    with pytest.warns(DeprecationWarning, match="use_given_eq_density_cube"):
        stf_gen.use_given_wet_image = True
    assert stf_gen.use_given_eq_density_cube is True
    with pytest.warns(DeprecationWarning):
        assert stf_gen.use_given_wet_image is True


class _CenterOfMassCst:
    def target_center_of_mass(self):
        return np.array([1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("iso_center", "expected"),
    [
        (None, [[1.0, 2.0, 3.0]] * 3),
        ([], [[1.0, 2.0, 3.0]] * 3),
        (np.empty((0, 3)), [[1.0, 2.0, 3.0]] * 3),
        ([4.0, 5.0, 6.0], [[4.0, 5.0, 6.0]] * 3),
        ([[4.0, 5.0, 6.0]], [[4.0, 5.0, 6.0]] * 3),
        (np.arange(9.0).reshape(3, 3), np.arange(9.0).reshape(3, 3)),
    ],
    ids=["none", "empty-list", "empty-array", "single", "single-row", "per-beam"],
)
def test_resolve_iso_centers(iso_center, expected):
    iso = resolve_iso_centers(iso_center, _CenterOfMassCst(), 3)
    assert iso.shape == (3, 3)
    assert np.allclose(iso, expected)


def test_resolve_iso_centers_without_number_of_beams():
    iso = resolve_iso_centers([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], None)
    assert np.allclose(iso, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert resolve_iso_centers(None, _CenterOfMassCst()).shape == (1, 3)


@pytest.mark.parametrize("iso_center", [[1.0, 2.0], np.zeros((2, 3))], ids=["two-coords", "rows"])
def test_resolve_iso_centers_rejects_invalid(iso_center):
    with pytest.raises(ValueError):
        resolve_iso_centers(iso_center, _CenterOfMassCst(), 3)


def test_ray_padding_is_symmetric_for_bixels_below_ct_resolution(tg119):
    # With 2 mm bixels on the 3 mm TG119 CT, every ray is padded by one bixel on all sides,
    # so a single target voxel at the isocenter yields the full 3 x 3 grid of ray positions.
    ct, cst = tg119["ct"], tg119["cst"]
    pln = create_pln(
        {
            "radiation_mode": "photons",
            "machine": "Generic",
            "prop_stf": {"gantry_angles": [0.0], "couch_angles": [0.0], "bixel_width": 2.0},
        }
    )
    generator = StfGeneratorPhotonIMRT(pln)
    generator._ct, generator._cst = ct, cst
    generator._initialize()
    generator._initialize_patient_geometry()

    iso_center = np.asarray(cst.target_center_of_mass())
    generator._target_voxel_coordinates = iso_center.reshape(3, 1)
    beam = {
        "gantry_angle": 0.0,
        "couch_angle": 0.0,
        "sad": 1000.0,
        "bixel_width": 2.0,
        "iso_center": iso_center,
    }
    ray_pos = generator._generate_ray_positions_in_isocenter_plane(beam)

    offsets = {(float(x), float(z)) for x, z in ray_pos[[0, 2]].T}
    assert offsets == {(x, z) for x in (-2.0, 0.0, 2.0) for z in (-2.0, 0.0, 2.0)}
