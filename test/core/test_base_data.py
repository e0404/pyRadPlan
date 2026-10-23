import os

import pytest

from pyRadPlan.biology.models import (
    TabulatedAlphaBetaModel,
    add_rbe_table,
    add_stopping_power_table,
)
from pyRadPlan.core import (
    BASE_DATA_KINDS,
    add_base_data_file,
    find_base_data,
    list_base_data,
    remove_base_data_file,
    shared_base_data_dirs,
    user_base_data_dir,
)
from pyRadPlan._settings import get_settings

RBE_TABLE = "RBEtable_LEMI_Scholz06_AX01_BX005.mat"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture
def shared_roots(tmp_path):
    """Two shared base data roots configured through ``settings.base_data_path``."""
    settings = get_settings()
    previous = settings.base_data_path
    roots = [tmp_path / "course", tmp_path / "site"]
    settings.base_data_path = os.pathsep.join(str(root) for root in roots)
    yield roots
    settings.base_data_path = previous


def _shipped(kind, name):
    return find_base_data(kind, name).path


@pytest.mark.parametrize(
    "kind, folder, shipped",
    [
        ("machines", "machines", "photons_Generic.mat"),
        ("rbe_tables", "rbe", RBE_TABLE),
        ("stopping_power_tables", "physics/stopping", "SPtable.mat"),
        ("phantoms", "phantoms", "TG119.mat"),
    ],
)
def test_user_folder_mirrors_the_shipped_one(data_dir, kind, folder, shipped):
    assert BASE_DATA_KINDS[kind].folder == folder
    assert user_base_data_dir(kind) == data_dir / folder
    entry = find_base_data(kind, shipped)
    assert entry.source == "bundled"
    assert entry.path.is_file()
    assert [e.source for e in list_base_data(kind)] == ["bundled"] * len(list_base_data(kind))


def test_file_copied_by_hand_is_found_like_an_added_one(data_dir):
    folder = user_base_data_dir("stopping_power_tables", create=True)
    (folder / "SP_mine.mat").write_bytes(
        _shipped("stopping_power_tables", "SPtable.mat").read_bytes()
    )

    entry = find_base_data("stopping_power_tables", "SP_mine.mat")
    assert entry.source == "user"
    assert [e.name for e in list_base_data("stopping_power_tables", source="user")] == [
        "SP_mine.mat"
    ]


def test_add_and_remove_base_data_file(data_dir):
    source = _shipped("stopping_power_tables", "SPtable.mat")
    with pytest.raises(FileExistsError, match="ships with pyRadPlan"):
        add_base_data_file("stopping_power_tables", source)

    target = add_base_data_file("stopping_power_tables", source, name="SP_mine.mat")
    assert target == data_dir / "physics" / "stopping" / "SP_mine.mat"
    with pytest.raises(FileExistsError, match="already added"):
        add_base_data_file("stopping_power_tables", source, name="SP_mine.mat")
    add_base_data_file("stopping_power_tables", source, name="SP_mine.mat", overwrite=True)

    remove_base_data_file("stopping_power_tables", "SP_mine.mat")
    assert find_base_data("stopping_power_tables", "SP_mine.mat") is None
    with pytest.raises(FileNotFoundError):
        remove_base_data_file("stopping_power_tables", "SPtable.mat")


def test_shared_folders_are_searched_after_the_user_folder(data_dir, shared_roots):
    course, site = shared_roots
    sp_table = _shipped("stopping_power_tables", "SPtable.mat").read_bytes()
    assert shared_base_data_dirs("stopping_power_tables") == [
        course / "physics" / "stopping",
        site / "physics" / "stopping",
    ]
    for root, name in [(course, "SP_course.mat"), (site, "SP_course.mat"), (site, "SP_site.mat")]:
        (root / "physics" / "stopping").mkdir(parents=True, exist_ok=True)
        (root / "physics" / "stopping" / name).write_bytes(sp_table)
    # a shared file never replaces a shipped one
    (course / "physics" / "stopping" / "SPtable.mat").write_bytes(sp_table)

    entries = {e.name: e for e in list_base_data("stopping_power_tables")}
    assert entries["SPtable.mat"].source == "bundled"
    assert entries["SP_course.mat"].source == "shared"
    assert entries["SP_course.mat"].path == course / "physics" / "stopping" / "SP_course.mat"
    assert entries["SP_site.mat"].source == "shared"
    assert [e.name for e in list_base_data("stopping_power_tables", source="shared")] == [
        "SP_course.mat",
        "SP_site.mat",
    ]

    # adding a file under a shared name puts the user's own copy first
    target = add_base_data_file(
        "stopping_power_tables", course / "physics" / "stopping" / "SP_course.mat"
    )
    assert target == data_dir / "physics" / "stopping" / "SP_course.mat"
    assert find_base_data("stopping_power_tables", "SP_course.mat").source == "user"
    with pytest.raises(FileNotFoundError):
        remove_base_data_file("stopping_power_tables", "SP_site.mat")


def test_missing_shared_folders_are_skipped(data_dir, shared_roots):
    assert [e.source for e in list_base_data("phantoms")] == ["bundled"]
    assert find_base_data("phantoms", "nothing.mat") is None


@pytest.mark.parametrize("name", ["", ".mat", "a/b.mat", "x:y.mat", "table.txt", "tab\there.mat"])
def test_add_base_data_file_rejects_invalid_names(data_dir, name):
    with pytest.raises(ValueError):
        add_base_data_file("rbe_tables", _shipped("rbe_tables", RBE_TABLE), name=name)


def test_unknown_kind():
    with pytest.raises(ValueError, match="Unknown base data kind"):
        list_base_data("nope")


def test_added_rbe_table_is_available_to_the_model(data_dir):
    add_rbe_table(_shipped("rbe_tables", RBE_TABLE), name="RBEtable_mine.mat")
    add_stopping_power_table(_shipped("stopping_power_tables", "SPtable.mat"), name="SP_mine.mat")

    assert "RBEtable_mine.mat" in TabulatedAlphaBetaModel.available_quantity_tables()
    model = TabulatedAlphaBetaModel(
        quantity_table_name="RBEtable_mine.mat", sp_table_name="SP_mine.mat"
    )
    assert model.table_kernel_set.provenance_label == "LEM I"


def test_add_rbe_table_rejects_other_files(data_dir):
    with pytest.raises(ValueError, match="RBEtable"):
        add_rbe_table(_shipped("stopping_power_tables", "SPtable.mat"), name="RBE_bad.mat")
    with pytest.raises(ValueError, match="SPtable"):
        add_stopping_power_table(_shipped("rbe_tables", RBE_TABLE), name="SP_bad.mat")
    assert list_base_data("rbe_tables", source="user") == []


def test_missing_table_is_reported():
    with pytest.raises(FileNotFoundError, match="RBE table 'missing.mat'"):
        TabulatedAlphaBetaModel(quantity_table_name="missing.mat")


def test_replaced_table_metadata_is_reread(data_dir, tmp_path):
    import numpy as np
    from scipy.io import savemat

    def write_table(path, alpha_x):
        savemat(
            path,
            {
                "RBEtable": {
                    "meta": {
                        "model": "MKM",
                        "modelParameters": {"alphaX": alpha_x, "betaX": 0.05},
                    },
                    "data": {
                        "A": 1.0,
                        "Z": 1.0,
                        "energies": np.ones(2),
                        "alpha": 1.0,
                        "beta": 1.0,
                    },
                }
            },
        )

    first, second = tmp_path / "first.mat", tmp_path / "second.mat"
    write_table(first, 0.1)
    write_table(second, 0.3)

    target = add_rbe_table(first, name="RBEtable_x.mat")
    classes = TabulatedAlphaBetaModel.quantity_table_tissue_classes("RBEtable_x.mat")
    assert classes[0].alpha_x == 0.1
    stat = os.stat(target)
    add_rbe_table(second, name="RBEtable_x.mat", overwrite=True)
    # A replacement within one tick of the file system clock keeps size and mtime
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert os.stat(target).st_size == stat.st_size
    classes = TabulatedAlphaBetaModel.quantity_table_tissue_classes("RBEtable_x.mat")
    assert classes[0].alpha_x == 0.3
