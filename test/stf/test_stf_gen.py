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
from pyRadPlan.scenarios import (
    CTPhaseUncertainty,
    RangeUncertainty,
    ScenarioSet,
    SetupUncertainty,
)

from pyRadPlan.stf import (
    Ray,
    get_available_generators,
    get_generator,
    resolve_iso_centers,
    FieldShapeAsBLD,
    FieldShapeAsMask,
    FieldShapeComposite,
)
from pyRadPlan.stf.generators import (
    StfGeneratorBase,
    StfGeneratorExternalBeamRayBixel,
    StfGeneratorPhotonIMRT,
    StfGeneratorPhotonCollimatedSquareFields,
    StfGeneratorPhotonSingleBixel,
    StfGeneratorIonSingleSpot,
    StfGeneratorIMPT,
    StfGeneratorVHEE,
)
from pyRadPlan.machines import MLC, Jaw
from pyRadPlan.result import BeamSequence, MachineState, Segment


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


def _make_mlc(leaf_width: float, opening: float) -> MLC:
    boundaries = np.array([-10.0, 0.0, 10.0])
    positions = np.array([[-opening, opening]] * 3)
    return MLC(
        device_type="MLCX",
        leaf_position_boundaries=boundaries,
        leaf_positions=positions,
        leaf_width=leaf_width,
    )


def _make_beam_sequence(n_shapes: int, weight_start: float = 1.0) -> BeamSequence:
    mlcs = [_make_mlc(leaf_width=10.0, opening=5.0 + s) for s in range(n_shapes)]
    return BeamSequence.from_shapes(mlcs, [weight_start + s for s in range(n_shapes)])


def _with_jaws(beam_sequence: BeamSequence, jaws_per_segment: list) -> BeamSequence:
    return BeamSequence.from_shapes(
        [segment.start.mlc for segment in beam_sequence.segments],
        beam_sequence.weights,
        jaws=jaws_per_segment,
    )


def _mlcs(beam_sequence: BeamSequence) -> list:
    return [segment.start.mlc for segment in beam_sequence.segments]


def test_photon_single_bixel_bare_pencil_beam_without_beam_sequences(
    sample_ct, sample_cst, sample_photon_pln_dict
):
    sample_photon_pln_dict["prop_stf"] = {"generator": "photonSingleBixel", "energy": 6.0}
    pln = create_pln(sample_photon_pln_dict)
    stf_gen = StfGeneratorPhotonSingleBixel(pln)
    stf_gen.gantry_angles = [0.0, 90.0]
    stf_gen.couch_angles = [0.0, 0.0]

    stf = stf_gen.generate(sample_ct, sample_cst)

    assert len(stf) == 2
    for field in stf:
        assert len(field["rays"]) == 1
        beamlet = field["rays"][0]["beamlets"][0]
        assert isinstance(beamlet, dict)
        assert beamlet["energy"] == 6.0


def _states(beam_sequence: BeamSequence) -> list:
    return StfGeneratorPhotonSingleBixel()._segment_states(beam_sequence)


def test_group_by_jaws_without_jaws_groups_everything():
    groups = StfGeneratorPhotonSingleBixel._group_by_jaws(_states(_make_beam_sequence(3)))
    assert groups == [[0, 1, 2]]


def test_group_by_jaws_splits_on_differing_devices():
    jaw_a = Jaw(device_orientation="X", positions=[-20.0, 20.0], field_width=40.0)
    jaw_b = Jaw(device_orientation="X", positions=[-15.0, 15.0], field_width=30.0)
    beam_sequence = _with_jaws(_make_beam_sequence(3), [[jaw_a], [jaw_a], [jaw_b]])
    groups = StfGeneratorPhotonSingleBixel._group_by_jaws(_states(beam_sequence))
    assert groups == [[0, 1], [2]]


def test_segment_states_sample_a_moving_segment():
    start, end = _make_mlc(leaf_width=10.0, opening=5.0), _make_mlc(leaf_width=10.0, opening=15.0)
    moving = Segment(weight=2.0, start=MachineState(mlc=start), end=MachineState(mlc=end))
    generator = StfGeneratorPhotonSingleBixel()

    states = generator._segment_states(BeamSequence(segments=[moving]))

    assert len(states) == generator.DYNAMIC_SAMPLES
    assert sum(weight for weight, _, _ in states) == pytest.approx(2.0)
    travelled = [
        (mlc.leaf_positions - start.leaf_positions) / (end.leaf_positions - start.leaf_positions)
        for _, mlc, _ in states
    ]
    np.testing.assert_allclose([np.nanmean(t) for t in travelled], [0.1, 0.3, 0.5, 0.7, 0.9])


def test_segment_states_refuse_gantry_motion():
    mlc = _make_mlc(leaf_width=10.0, opening=5.0)
    arc = Segment(
        weight=1.0,
        start=MachineState(gantry_angle=0.0, mlc=mlc),
        end=MachineState(gantry_angle=4.0, mlc=mlc),
    )
    with pytest.raises(NotImplementedError, match="gantry or couch motion"):
        StfGeneratorPhotonSingleBixel()._segment_states(BeamSequence(segments=[arc]))


def test_photon_single_bixel_beam_sequences(sample_ct, sample_cst, sample_photon_pln_dict):
    sample_photon_pln_dict["prop_stf"] = {
        "generator": "photonSingleBixel",
        "resolution": 0.5,
        "energy": 6.0,
    }
    pln = create_pln(sample_photon_pln_dict)
    stf_gen = StfGeneratorPhotonSingleBixel(pln)
    stf_gen.gantry_angles = [0.0, 90.0]
    stf_gen.couch_angles = [0.0, 0.0]
    stf_gen.beam_sequences = [_make_beam_sequence(2), _make_beam_sequence(3, weight_start=2.0)]

    stf = stf_gen.generate(sample_ct, sample_cst)

    assert len(stf) == 2
    # all shapes of each beam share no jaws -> one group -> one ray
    assert len(stf[0]["rays"]) == 1
    assert len(stf[1]["rays"]) == 1

    for field, beam_sequence in zip(stf, stf_gen.beam_sequences):
        beamlet = field["rays"][0]["beamlets"][0]
        assert isinstance(beamlet, FieldShapeAsMask)
        assert beamlet.weight == pytest.approx(1.0)

        expected_mask = sum(
            w
            * FieldShapeAsBLD(energy=6.0, bld=mlc, resolution=0.5)
            .resample(new_grid=beamlet.grid)
            .mask
            for w, mlc in zip(beam_sequence.weights, _mlcs(beam_sequence))
        )
        np.testing.assert_allclose(beamlet.mask, expected_mask)


def test_photon_single_bixel_beam_sequences_single_shape_stays_a_plain_composite(
    sample_ct, sample_cst, sample_photon_pln_dict
):
    sample_photon_pln_dict["prop_stf"] = {
        "generator": "photonSingleBixel",
        "resolution": 0.5,
        "energy": 6.0,
    }
    pln = create_pln(sample_photon_pln_dict)
    stf_gen = StfGeneratorPhotonSingleBixel(pln)
    stf_gen.gantry_angles = [0.0]
    stf_gen.couch_angles = [0.0]

    beam_sequence = _make_beam_sequence(1)
    stf_gen.beam_sequences = [beam_sequence]

    stf = stf_gen.generate(sample_ct, sample_cst)

    assert len(stf[0]["rays"]) == 1
    beamlet = stf[0]["rays"][0]["beamlets"][0]
    assert isinstance(beamlet, FieldShapeAsBLD)
    assert beamlet.bld is _mlcs(beam_sequence)[0]
    assert beamlet.weight == pytest.approx(beam_sequence.weights[0])


def test_photon_single_bixel_beam_sequences_with_jaws(
    sample_ct, sample_cst, sample_photon_pln_dict
):
    sample_photon_pln_dict["prop_stf"] = {
        "generator": "photonSingleBixel",
        "resolution": 0.5,
        "energy": 6.0,
    }
    pln = create_pln(sample_photon_pln_dict)
    stf_gen = StfGeneratorPhotonSingleBixel(pln)
    stf_gen.gantry_angles = [0.0]
    stf_gen.couch_angles = [0.0]

    beam_sequence = _make_beam_sequence(2)
    jaw_x = Jaw(device_orientation="X", positions=[-20.0, 20.0], field_width=40.0)
    jaw_y = Jaw(device_orientation="Y", positions=[-15.0, 15.0], field_width=30.0)
    beam_sequence = _with_jaws(beam_sequence, [[jaw_x, jaw_y]] * 2)
    stf_gen.beam_sequences = [beam_sequence]

    stf = stf_gen.generate(sample_ct, sample_cst)

    assert len(stf[0]["rays"]) == 1
    beamlet = stf[0]["rays"][0]["beamlets"][0]
    assert isinstance(beamlet, FieldShapeAsMask)
    assert beamlet.weight == pytest.approx(1.0)


def test_photon_single_bixel_beam_sequences_differing_jaws_are_merged_into_one_ray(
    sample_ct, sample_cst, sample_photon_pln_dict
):
    sample_photon_pln_dict["prop_stf"] = {
        "generator": "photonSingleBixel",
        "resolution": 0.5,
        "energy": 6.0,
    }
    pln = create_pln(sample_photon_pln_dict)
    stf_gen = StfGeneratorPhotonSingleBixel(pln)
    stf_gen.gantry_angles = [0.0]
    stf_gen.couch_angles = [0.0]

    beam_sequence = _make_beam_sequence(2)
    jaw_a = Jaw(device_orientation="X", positions=[-20.0, 20.0], field_width=40.0)
    jaw_b = Jaw(device_orientation="X", positions=[-15.0, 15.0], field_width=30.0)
    beam_sequence = _with_jaws(beam_sequence, [[jaw_a], [jaw_b]])
    stf_gen.beam_sequences = [beam_sequence]

    stf = stf_gen.generate(sample_ct, sample_cst)

    assert len(stf[0]["rays"]) == 1
    beamlet = stf[0]["rays"][0]["beamlets"][0]
    assert isinstance(beamlet, FieldShapeAsMask)
    assert beamlet.weight == pytest.approx(1.0)

    expected_mask = sum(
        w
        * FieldShapeAsBLD(energy=6.0, bld=mlc, resolution=0.5).resample(new_grid=beamlet.grid).mask
        * FieldShapeAsBLD(energy=6.0, bld=jaw, resolution=0.5).resample(new_grid=beamlet.grid).mask
        for w, mlc, jaw in zip(beam_sequence.weights, _mlcs(beam_sequence), [jaw_a, jaw_b])
    )
    np.testing.assert_allclose(beamlet.mask, expected_mask, atol=1e-6)


def test_photon_single_bixel_beam_sequences_length_mismatch(
    sample_ct, sample_cst, sample_photon_pln_dict
):
    sample_photon_pln_dict["prop_stf"] = {
        "generator": "photonSingleBixel",
        "resolution": 0.5,
        "energy": 6.0,
    }
    pln = create_pln(sample_photon_pln_dict)
    stf_gen = StfGeneratorPhotonSingleBixel(pln)
    stf_gen.gantry_angles = [0.0, 90.0]
    stf_gen.couch_angles = [0.0, 0.0]
    stf_gen.beam_sequences = [_make_beam_sequence(2)]  # only 1 entry, but 2 beams

    with pytest.raises(ValueError):
        stf_gen.generate(sample_ct, sample_cst)


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


def _imrt_target_mask(ct, cst, pln, scenarios):
    stf_gen = StfGeneratorPhotonIMRT(pln)
    stf_gen.mult_scen = scenarios
    stf_gen._ct = ct
    stf_gen._cst = cst
    stf_gen._initialize()
    stf_gen._initialize_patient_geometry()
    return stf_gen, sitk.GetArrayFromImage(stf_gen._target_mask).astype(bool)


def _num_rays(ct, cst, pln, scenarios):
    stf_gen = StfGeneratorPhotonIMRT(pln)
    stf_gen.gantry_angles = [0.0]
    stf_gen.couch_angles = [0.0]
    stf_gen.mult_scen = scenarios
    return sum(len(beam["rays"]) for beam in stf_gen.generate(ct, cst))


def _scenario_set(points):
    return ScenarioSet.from_points(
        (CTPhaseUncertainty(), SetupUncertainty(), RangeUncertainty()), points
    )


def test_target_margin_uses_setup_extent(sample_ct, sample_cst, sample_photon_pln):
    """A 5 mm x shift widens the target margin along x only (numpy axis 2)."""
    _, nominal = _imrt_target_mask(sample_ct, sample_cst, sample_photon_pln, "nomScen")
    shift_set = _scenario_set({"setup": [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)]})
    stf_gen, shifted = _imrt_target_mask(sample_ct, sample_cst, sample_photon_pln, shift_set)

    max_iso_shift, range_margin = stf_gen._scenario_margins()
    np.testing.assert_array_equal(max_iso_shift, [5.0, 0.0, 0.0])
    assert range_margin == 0.0

    assert not (nominal & ~shifted).any()
    assert shifted.sum() > nominal.sum()
    np.testing.assert_array_equal(shifted.any(axis=2), nominal.any(axis=2))

    rays_nominal = _num_rays(sample_ct, sample_cst, sample_photon_pln, "nomScen")
    rays_shifted = _num_rays(sample_ct, sample_cst, sample_photon_pln, shift_set)
    assert rays_shifted > rays_nominal


def test_range_margin_uses_both_range_components(sample_ct, sample_cst, sample_photon_pln):
    """The relative component enters with its own magnitude, not the absolute one."""
    range_set = _scenario_set({"range": [(0.0, 0.0), (-1.0, 0.05), (0.5, -0.02)]})
    stf_gen, _ = _imrt_target_mask(sample_ct, sample_cst, sample_photon_pln, range_set)

    max_iso_shift, range_margin = stf_gen._scenario_margins()
    np.testing.assert_array_equal(max_iso_shift, np.zeros(3))
    assert range_margin == pytest.approx(1.0 + 0.05)


def test_margins_without_setup_and_range_dimensions(sample_ct, sample_cst, sample_photon_pln):
    phase_only = ScenarioSet.from_points((CTPhaseUncertainty(),), {"ct_phase": [0]})
    stf_gen, _ = _imrt_target_mask(sample_ct, sample_cst, sample_photon_pln, phase_only)

    max_iso_shift, range_margin = stf_gen._scenario_margins()
    np.testing.assert_array_equal(max_iso_shift, np.zeros(3))
    assert range_margin == 0.0


def test_generator_takes_the_plans_scenario_model(sample_photon_pln):
    """``pln.scenarios`` reaches the generator (``"mult_scen" in pln`` never did)."""
    from pyRadPlan.stf.generators import get_generator

    sample_photon_pln.scenarios = {"model": "wcScen", "combinations": "none"}
    generator = get_generator(sample_photon_pln)
    assert getattr(generator.mult_scen, "short_name", None) == "wcScen"


def test_scenario_model_in_prop_stf_takes_precedence(sample_photon_pln):
    from pyRadPlan.stf.generators import get_generator

    sample_photon_pln.scenarios = {"model": "wcScen", "combinations": "none"}
    sample_photon_pln.prop_stf["mult_scen"] = "nomScen"
    assert get_generator(sample_photon_pln).mult_scen == "nomScen"

    sample_photon_pln.prop_stf.pop("mult_scen")
    sample_photon_pln.prop_stf["scenarios"] = "rndScen"
    assert get_generator(sample_photon_pln).mult_scen == "rndScen"


def test_target_margin_adds_shift_range_and_beamlet_margin(
    sample_ct, sample_cst, sample_photon_pln
):
    """matRad: margin = max(resolution, max |shift| + range margin + beamlet margin) per axis."""
    shift_set = _scenario_set(
        {
            "setup": [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, -4.0, 0.0)],
            "range": [(0.0, 0.0), (1.0, 0.02), (0.0, 0.0)],
        }
    )
    stf_gen, _ = _imrt_target_mask(sample_ct, sample_cst, sample_photon_pln, shift_set)
    added = stf_gen._computed_target_margin()
    range_margin = 1.0 + 0.02
    res = sample_ct.resolution
    expected = (
        max(res["x"], 10.0 + range_margin + added),
        max(res["y"], 4.0 + range_margin + added),
        max(res["z"], range_margin + added),
    )
    assert stf_gen._target_margins() == pytest.approx(expected)
    # the sum exceeds the former max(shift, range + beamlet) rule along the shifted axes
    assert stf_gen._target_margins()[0] > max(res["x"], 10.0, range_margin + added)


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
