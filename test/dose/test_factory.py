import pytest

from pyRadPlan.dose.engines import (
    ParticleHongPencilBeamEngine,
    get_available_engines,
    get_engine,
)


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
