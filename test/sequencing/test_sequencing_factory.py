import pytest

from pyRadPlan.sequencing.sequencer import (
    EngelLeafSequencer,
    Sequencer,
    SiochiLeafSequencer,
    XiaLeafSequencer,
    get_available_sequencers,
    get_sequencer_from_pln,
    register_sequencer,
)
from pyRadPlan.sequencing.sequencer import _factory


def test_available_sequencers(test_data_engel):
    sequencers = get_available_sequencers(test_data_engel[0])
    assert isinstance(sequencers, dict)
    assert sequencers == {
        "engel": EngelLeafSequencer,
        "siochi": SiochiLeafSequencer,
        "xia": XiaLeafSequencer,
    }


def test_available_sequencers_from_radiation_mode():
    assert set(get_available_sequencers("photons")) == {"engel", "siochi", "xia"}
    assert get_available_sequencers("protons") == {}


@pytest.mark.parametrize(
    "short_name, cls",
    [("engel", EngelLeafSequencer), ("siochi", SiochiLeafSequencer), ("xia", XiaLeafSequencer)],
)
def test_get_sequencer(test_data_engel, short_name, cls):
    test_data_engel[0].prop_seq = {"sequencer": short_name}
    sequencer = get_sequencer_from_pln(test_data_engel[0])
    assert isinstance(sequencer, cls)


def test_get_sequencer_default(test_data_engel):
    test_data_engel[0].prop_seq = {}
    sequencer = get_sequencer_from_pln(test_data_engel[0])
    assert isinstance(sequencer, EngelLeafSequencer)


def test_get_sequencer_passes_properties(test_data_engel):
    test_data_engel[0].prop_seq = {"sequencer": "siochi", "sequencing_level": 5}
    sequencer = get_sequencer_from_pln(test_data_engel[0])
    assert sequencer.num_of_levels == 5


def test_get_sequencer_invalid(test_data_engel):
    test_data_engel[0].prop_seq = {"sequencer": "InvalidSequencer"}
    with pytest.warns(UserWarning, match="InvalidSequencer"), pytest.raises(ValueError):
        get_sequencer_from_pln(test_data_engel[0])


def test_get_sequencer_unavailable_radiation_mode():
    with pytest.raises(ValueError, match="protons"):
        get_sequencer_from_pln({"radiation_mode": "protons", "machine": "Generic"})


def test_register_sequencer(monkeypatch):
    monkeypatch.setattr(_factory, "SEQUENCER", {})

    class DummySequencer(EngelLeafSequencer):
        name = "dummy_sequencer"
        short_name = "dummy"

    register_sequencer(DummySequencer)
    assert get_available_sequencers("photons") == {"dummy": DummySequencer}

    with pytest.warns(UserWarning, match="already registered"):
        register_sequencer(DummySequencer)


def test_register_sequencer_invalid(monkeypatch):
    monkeypatch.setattr(_factory, "SEQUENCER", {})

    with pytest.raises(ValueError):
        register_sequencer(object)

    class NoShortName(Sequencer):
        name = "no_short_name"
        short_name = None

    with pytest.raises(ValueError):
        register_sequencer(NoShortName)
