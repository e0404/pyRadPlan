import pytest
import sys

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+

import pymatreader
import numpy as np

from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst
from pyRadPlan.plan import IonPlan
from pyRadPlan.stf import SteeringInformation, create_stf
from pyRadPlan.stf.generators import StfGeneratorIMPT


@pytest.fixture
def sample_stf_snake():
    beam = {
        "gantry_angle": 0,
        "couch_angle": 0,
        "bixel_width": 1,
        "radiation_mode": "protons",
        "machine": "Generic",
        "sad": 100000,
        "iso_center": [0, 0, 0],
        "rays": [
            {"ray_pos_bev": ([0, 1, 2]), "ray_pos": ([5, 6, 7])},
            {"ray_pos_bev": ([0, 1, 2]), "ray_pos": ([5, 6, 7])},
        ],
        "source_point_bev": [0, -10000, 0],
        "source_point": [0, 0, 0],
        "longitudinal_spot_spacing": 1,
    }
    return beam


@pytest.fixture
def sample_stf_camel():
    beam = {
        "gantryAngle": 0,
        "couchAngle": 0,
        "bixelWidth": 1,
        "radiationMode": "protons",
        "machine": "Generic",
        "sad": 100000,
        "isoCenter": [0, 0, 0],
        "rays": [
            {"rayPos_bev": ([0, 1, 2]), "rayPos": ([5, 6, 7])},
            {"rayPos_bev": ([0, 1, 2]), "rayPos": ([5, 6, 7])},
        ],
        "sourcePointBev": [0, -10000, 0],
        "sourcePoint": [0, 0, 0],
        "longitudinalSpotSpacing": 1,
    }
    return beam


def test_create_stf_from_snake_case(sample_stf_snake):
    beam1 = sample_stf_snake
    input1 = [beam1, beam1]
    input2 = {"beams": [beam1, beam1]}

    # Testing for list[Dict]
    stf1 = create_stf(input1)
    assert isinstance(stf1, SteeringInformation)
    assert stf1.bixel_beam_index_map.size == stf1.total_number_of_bixels
    assert stf1.bixel_ray_index_per_beam_map.size == stf1.total_number_of_bixels
    assert stf1.bixel_index_per_beam_map.size == stf1.total_number_of_bixels

    tot_num_bixels = 0
    for i in range(len(input1)):
        assert hasattr(stf1.beams[i], "couch_angle")
        assert hasattr(stf1.beams[i].rays[0], "ray_pos_bev")
        assert all(hasattr(ray, "ray_pos_bev") for ray in stf1.beams[i].rays)
        assert all(hasattr(ray, "beamlets") for ray in stf1.beams[i].rays)
        assert all(isinstance(ray.beamlets, list) for ray in stf1.beams[i].rays)

        assert stf1.beams[i].bixel_ray_map.size == stf1.beams[i].total_number_of_bixels
        assert np.array_equal(
            stf1.beams[i].bixel_ray_map,
            stf1.bixel_ray_index_per_beam_map[
                tot_num_bixels : tot_num_bixels + stf1.beams[i].total_number_of_bixels
            ],
        )
        tot_num_bixels += stf1.beams[i].total_number_of_bixels

    # Testing for Dict[list[Dict]]
    stf2 = create_stf(input2)
    for i in range(len(input1)):
        assert hasattr(stf2.beams[i], "couch_angle")
        assert hasattr(stf2.beams[i].rays[0], "ray_pos_bev")
        assert all(hasattr(ray, "ray_pos_bev") for ray in stf2.beams[i].rays)
        assert all(hasattr(ray, "beamlets") for ray in stf2.beams[i].rays)
        assert all(isinstance(ray.beamlets, list) for ray in stf2.beams[i].rays)


def test_create_stf_from_camel_case(sample_stf_camel):
    beam1 = sample_stf_camel
    input1 = [beam1, beam1]
    input2 = {"beams": [beam1, beam1]}

    # Testing for list[Dict]
    stf1 = create_stf(input1)
    assert isinstance(stf1, SteeringInformation)
    for i in range(len(input1)):
        assert hasattr(stf1.beams[i], "couch_angle")
        assert hasattr(stf1.beams[i].rays[0], "ray_pos_bev")
        assert all(hasattr(ray, "ray_pos_bev") for ray in stf1.beams[i].rays)
        assert all(hasattr(ray, "beamlets") for ray in stf1.beams[i].rays)
        assert all(isinstance(ray.beamlets, list) for ray in stf1.beams[i].rays)

    # Testing for Dict[list[Dict]]
    stf2 = create_stf(input2)
    for i in range(len(input1)):
        assert hasattr(stf2.beams[i], "couch_angle")
        assert hasattr(stf2.beams[i].rays[0], "ray_pos_bev")
        assert all(hasattr(ray, "ray_pos_bev") for ray in stf2.beams[i].rays)
        assert all(hasattr(ray, "beamlets") for ray in stf2.beams[i].rays)
        assert all(isinstance(ray.beamlets, list) for ray in stf2.beams[i].rays)


def test_create_stf_from_stf(sample_stf_snake):
    beam1 = sample_stf_snake
    beam2 = beam1.copy()
    beam2["gantry_angle"] = 90
    input1 = [beam1, beam2]

    # Creating stf
    stf = SteeringInformation(beams=input1)
    for i, beam in enumerate(stf.beams):
        assert hasattr(beam, "couch_angle")
        assert hasattr(beam.rays[0], "ray_pos_bev")
        assert all(hasattr(ray, "ray_pos_bev") for ray in beam.rays)
        assert all(hasattr(ray, "beamlets") for ray in beam.rays)
        assert all(isinstance(ray.beamlets, list) for ray in beam.rays)

    # Creating stf from existing stf by SteeringInformation()
    stf = create_stf(stf)
    for i, beam in enumerate(stf.beams):
        assert hasattr(beam, "couch_angle")
        assert hasattr(beam.rays[0], "ray_pos_bev")
        assert all(hasattr(ray, "ray_pos_bev") for ray in beam.rays)
        assert all(hasattr(ray, "beamlets") for ray in beam.rays)
        assert all(isinstance(ray.beamlets, list) for ray in beam.rays)

    stf = create_stf(input1)
    assert isinstance(stf, SteeringInformation)
    for i, beam in enumerate(stf.beams):
        assert hasattr(beam, "couch_angle")
        assert hasattr(beam.rays[0], "ray_pos_bev")
        assert all(hasattr(ray, "ray_pos_bev") for ray in beam.rays)
        assert all(hasattr(ray, "beamlets") for ray in beam.rays)
        assert all(isinstance(ray.beamlets, list) for ray in beam.rays)


def test_stfgen_to_stf():
    files = resources.files("pyRadPlan.data.phantoms")
    path = files.joinpath("TG119.mat")
    tmp = pymatreader.read_mat(path)

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    pln = IonPlan(radiation_mode="protons", machine="Generic")

    stfgen = StfGeneratorIMPT(pln)
    stfgen.bixel_width = 5.0
    stfgen.gantry_angles = [0.0]

    stf_native = stfgen.generate(ct, cst)
    stf = create_stf(stf_native)

    assert stf is not None
    assert isinstance(stf, SteeringInformation)

    for i, beam in enumerate(stf.beams):
        assert hasattr(beam, "couch_angle")
        assert hasattr(beam.rays[0], "ray_pos_bev")
        assert all(hasattr(ray, "ray_pos_bev") for ray in beam.rays)
        assert all(hasattr(ray, "beamlets") for ray in beam.rays)
        assert all(isinstance(ray.beamlets, list) for ray in beam.rays)


def test_from_and_to_matrad():
    # from scipy.io import savemat

    files = resources.files("pyRadPlan.data.stf")
    path = files.joinpath("matRad_stf.mat")
    path_export = files.joinpath("export_stf.mat")
    matRad_file = pymatreader.read_mat(path)
    stf_from_matRad = create_stf(matRad_file)
    stf_to_matrad = stf_from_matRad.to_matrad()

    assert isinstance(stf_to_matrad, np.recarray)
    # savemat(path_export, stf_to_matrad, format='5', oned_as='column')


def test_from_matlab_one_beam():
    # Special case of one beam since pymatreader handles 1D arrays differently
    files = resources.files("pyRadPlan.data.stf")
    path = files.joinpath("matRad_stf_onebeam.mat")
    matRad_file = pymatreader.read_mat(path)
    stf_from_matRad = create_stf(matRad_file)
    assert stf_from_matRad.beams[0].gantry_angle == 0
    assert stf_from_matRad is not None

    assert stf_from_matRad.bixel_beam_index_map.size == stf_from_matRad.total_number_of_bixels
    assert (
        stf_from_matRad.bixel_ray_index_per_beam_map.size == stf_from_matRad.total_number_of_bixels
    )
    assert stf_from_matRad.bixel_index_per_beam_map.size == stf_from_matRad.total_number_of_bixels


def test_from_matlab_multi_beam():
    files = resources.files("pyRadPlan.data.stf")
    path = files.joinpath("matRad_stf.mat")
    matRad_file = pymatreader.read_mat(path)
    stf_from_matRad = create_stf(matRad_file)
    assert stf_from_matRad.beams[0].gantry_angle == 0
    assert stf_from_matRad is not None

    assert stf_from_matRad.bixel_beam_index_map.size == stf_from_matRad.total_number_of_bixels
    assert (
        stf_from_matRad.bixel_ray_index_per_beam_map.size == stf_from_matRad.total_number_of_bixels
    )
    assert stf_from_matRad.bixel_index_per_beam_map.size == stf_from_matRad.total_number_of_bixels

    tot_num_bixels = 0
    for i in range(stf_from_matRad.num_of_beams):
        assert (
            stf_from_matRad.beams[i].bixel_ray_map.size
            == stf_from_matRad.beams[i].total_number_of_bixels
        )
        assert np.array_equal(
            stf_from_matRad.beams[i].bixel_ray_map,
            stf_from_matRad.bixel_ray_index_per_beam_map[
                tot_num_bixels : tot_num_bixels + stf_from_matRad.beams[i].total_number_of_bixels
            ],
        )
        assert np.all(
            stf_from_matRad.bixel_beam_index_map[
                tot_num_bixels : tot_num_bixels + stf_from_matRad.beams[i].total_number_of_bixels
            ]
            == i
        )
        assert np.all(
            stf_from_matRad.bixel_index_per_beam_map[
                tot_num_bixels : tot_num_bixels + stf_from_matRad.beams[i].total_number_of_bixels
            ]
            == np.arange(stf_from_matRad.beams[i].total_number_of_bixels)
        )
        assert np.all(
            stf_from_matRad.beams[i].bixel_ray_map
            == stf_from_matRad.bixel_ray_index_per_beam_map[
                tot_num_bixels : tot_num_bixels + stf_from_matRad.beams[i].total_number_of_bixels
            ]
        )

        tot_num_bixels += stf_from_matRad.beams[i].total_number_of_bixels


# def test_create_stf_kwargs_snake():
#     stf1 = create_stf(Beam={"Beam":{"gantry_angle":0}})
#     assert isinstance(stf1, SteeringInformation)
#     assert stf1.gantry_angle == 0
