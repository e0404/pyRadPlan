import pytest

from pyRadPlan.dose.engines import (
    AIBeamletEngine,
    ParticleHongPencilBeamEngine,
    get_available_engines,
    get_engine,
)
from pyRadPlan.plan import validate_pln


def test_available_engines(test_data_protons):
    engines = get_available_engines(test_data_protons[0])
    assert isinstance(engines, dict)
    assert len(engines.items()) > 0


def test_get_engine(test_data_protons):
    engine = get_engine(test_data_protons[0])
    assert engine
    assert isinstance(engine, ParticleHongPencilBeamEngine)


def test_get_engine_default(test_data_protons):
    engine = get_engine(test_data_protons[0])
    assert engine
    assert isinstance(engine, ParticleHongPencilBeamEngine)


def test_get_engine_invalid(test_data_protons):
    test_data_protons[0].prop_dose_calc["engine"] = "InvalidEngine"
    with pytest.raises(ValueError):
        engine = get_engine(test_data_protons[0])


def test_unknown_model_key_does_not_break_other_engines():
    """A 'model' entry only means something to the AI engine; others treat it as any unknown key."""
    pln = validate_pln(
        {
            "radiation_mode": "protons",
            "machine": "Generic",
            "prop_dose_calc": {"engine": "HongPB", "model": "x"},
        }
    )
    with pytest.warns(UserWarning, match='Property "model" not found'):
        engine = get_engine(pln)
    assert isinstance(engine, ParticleHongPencilBeamEngine)


def test_ai_engine_settings_come_from_prop_dose_calc():
    """The AI engine's model and inference settings are ordinary config fields."""
    pln = validate_pln(
        {
            "radiation_mode": "protons",
            "machine": "Generic",
            "prop_dose_calc": {
                "engine": "AIBeamlet",
                "model": "org/model",
                "max_batch_size": 8,
                "ensemble_size": 4,
                "dose_threshold": 1e-5,
            },
        }
    )
    engine = AIBeamletEngine(pln)
    assert engine.model == "org/model"
    assert engine.max_batch_size == 8
    assert engine.ensemble_size == 4
    assert engine.dose_threshold == 1e-5

    defaults = AIBeamletEngine()
    assert defaults.model is None
    assert defaults.max_batch_size == 32
    assert defaults.ensemble_size is None
    assert defaults.assumed_sparsity == 1e-2
