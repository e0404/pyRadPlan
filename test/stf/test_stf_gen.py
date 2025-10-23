import pytest
import sys

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+

import numpy as np
import SimpleITK as sitk

from pyRadPlan.io import load_patient
from pyRadPlan.plan import create_pln
from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import validate_cst, StructureSet, VOI

from pyRadPlan.stf import Ray, get_available_generators, get_generator
from pyRadPlan.stf.generators import (
    StfGeneratorBase,
    StfGeneratorExternalBeamRayBixel,
    StfGeneratorPhotonIMRT,
    StfGeneratorPhotonCollimatedSquareFields,
    StfGeneratorIonSingleSpot,
    StfGeneratorIMPT,
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
