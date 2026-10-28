import pytest

from pyRadPlan.machines import (
    add_machine_file,
    find_machine_file,
    get_available_machines,
    load_from_name,
    remove_machine_file,
    user_machine_dir,
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def photon_file():
    return str(find_machine_file("photons", "Generic"))


def test_user_machine_dir_lives_in_data_dir(data_dir):
    assert user_machine_dir() == data_dir / "machines"
    assert not user_machine_dir().exists()
    assert get_available_machines(user_only=True) == {}


def test_add_machine_file_makes_machine_available(data_dir, photon_file):
    target = add_machine_file(photon_file, machine_name="MyLinac")

    assert target == data_dir / "machines" / "photons_MyLinac.mat"
    assert "MyLinac" in get_available_machines("photons")["photons"]
    assert get_available_machines(user_only=True) == {"photons": ["MyLinac"]}
    assert load_from_name("photons", "MyLinac").radiation_mode == "photons"


def test_add_machine_file_cannot_shadow_bundled_machine(data_dir, photon_file):
    with pytest.raises(FileExistsError, match="ships with pyRadPlan"):
        add_machine_file(photon_file)


def test_add_machine_file_overwrite(data_dir, photon_file):
    add_machine_file(photon_file, machine_name="MyLinac")
    with pytest.raises(FileExistsError, match="already added"):
        add_machine_file(photon_file, machine_name="MyLinac")
    add_machine_file(photon_file, machine_name="MyLinac", overwrite=True)


def test_add_machine_file_with_known_radiation_mode_skips_reading(data_dir, tmp_path):
    fake = tmp_path / "anything.mat"
    fake.write_bytes(b"not read")
    target = add_machine_file(fake, machine_name="Copied", radiation_mode="protons")
    assert target.read_bytes() == b"not read"


@pytest.mark.parametrize("name", ["", "a/b", "a\\b", "x:y", "tab\there"])
def test_add_machine_file_rejects_invalid_names(data_dir, photon_file, name):
    with pytest.raises(ValueError):
        add_machine_file(photon_file, machine_name=name, radiation_mode="photons")


def test_add_machine_file_rejects_invalid_file(data_dir, tmp_path):
    bad = tmp_path / "photons_Bad.txt"
    bad.write_text("")
    with pytest.raises(ValueError, match="Unsupported"):
        add_machine_file(bad)


def test_remove_machine_file(data_dir, photon_file):
    add_machine_file(photon_file, machine_name="MyLinac")
    remove_machine_file("photons", "MyLinac")
    assert "MyLinac" not in get_available_machines("photons")["photons"]

    with pytest.raises(FileNotFoundError):
        remove_machine_file("photons", "MyLinac")
    with pytest.raises(FileNotFoundError):
        remove_machine_file("photons", "Generic")
    assert find_machine_file("photons", "Generic") is not None


def test_dose_engine_finds_user_machine(data_dir, photon_file):
    from pyRadPlan.dose.engines import DoseEngineBase

    add_machine_file(photon_file, machine_name="MyLinac")

    assert DoseEngineBase.load_machine("photons", "MyLinac").radiation_mode == "photons"
