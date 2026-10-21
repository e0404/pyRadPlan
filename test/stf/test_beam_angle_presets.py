import pytest
from pydantic import ValidationError

from pyRadPlan.stf import BeamAnglePreset, get_beam_angle_presets, register_beam_angle_preset
from pyRadPlan.stf import _presets
from pyRadPlan.stf.generators import StfGeneratorIMPT, StfGeneratorPhotonIMRT


@pytest.fixture
def restore_registry():
    snapshot = dict(_presets._PRESETS)
    yield
    _presets._PRESETS.clear()
    _presets._PRESETS.update(snapshot)


def _names(presets):
    return [p.name for p in presets]


def test_builtin_presets():
    presets = {p.name: p for p in get_beam_angle_presets()}
    assert presets["Lateral opposing"].gantry_angles == (90.0, 270.0)
    assert presets["4-field box"].gantry_angles == (0.0, 90.0, 180.0, 270.0)
    for n in (5, 7, 9, 12):
        star = presets[f"{n}-field IMRT star"]
        assert star.num_of_beams == n
        assert star.gantry_angles[1] == pytest.approx(360 / n, abs=1e-4)
        assert star.couch_angles == (0.0,) * n


def test_presets_typical_for_the_mode_come_first():
    photons = _names(get_beam_angle_presets("photons"))
    protons = _names(get_beam_angle_presets("protons"))
    assert photons.index("5-field IMRT star") < photons.index("Lateral opposing")
    assert protons.index("Lateral opposing") < protons.index("5-field IMRT star")
    assert sorted(photons) == sorted(protons)


def test_presets_apply_to_external_beam_generators():
    preset = get_beam_angle_presets()[0]
    assert preset.applies_to(StfGeneratorPhotonIMRT)
    assert preset.applies_to(StfGeneratorIMPT)
    assert not preset.applies_to(object)
    assert get_beam_angle_presets(generator=object) == []


def test_register_preset_from_dict(restore_registry):
    preset = register_beam_angle_preset(
        {"name": "Vertex", "gantry_angles": [0, 90], "couch_angles": [0, 90]}
    )
    assert preset.prop_stf() == {"gantry_angles": [0.0, 90.0], "couch_angles": [0.0, 90.0]}
    assert "Vertex" in _names(get_beam_angle_presets())
    with pytest.raises(ValueError, match="already registered"):
        register_beam_angle_preset({"name": "Vertex", "gantry_angles": [0]})
    register_beam_angle_preset({"name": "Vertex", "gantry_angles": [0]}, overwrite=True)


def test_preset_validation():
    with pytest.raises(ValidationError):
        BeamAnglePreset(name="bad", gantry_angles=[0, 90], couch_angles=[0])
    with pytest.raises(ValidationError):
        BeamAnglePreset(name="empty", gantry_angles=[])
