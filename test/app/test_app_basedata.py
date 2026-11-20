import pytest

from pyRadPlan.app._revisions import base_data_revision
from pyRadPlan.core import find_base_data

RBE_TABLE = "RBEtable_LEMI_Scholz06_AX01_BX005.mat"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def _shipped_bytes(kind, name):
    return find_base_data(kind, name).path.read_bytes()


def _add(client, kind, filename, data, **params):
    return client.call(
        "base_data.add",
        {"kind": kind, "filename": filename, "data": {"$array": 0}, **params},
        [data],
        session=False,
    )


def _rows(client, kind):
    return client.ok("base_data.list", {"kind": kind}, session=False)["result"]["rows"]


@pytest.mark.parametrize(
    "kind, expected",
    [
        (
            "machines",
            {"name": "Generic", "radiation_mode": "photons", "file": "photons_Generic.mat"},
        ),
        ("rbe_tables", {"file": RBE_TABLE}),
        ("stopping_power_tables", {"file": "SPtable.mat"}),
        ("phantoms", {"name": "TG119", "file": "TG119.mat"}),
    ],
)
def test_list_shows_the_shipped_files(client, data_dir, kind, expected):
    listing = client.ok("base_data.list", {"kind": kind}, session=False)["result"]
    assert listing["kind"] == kind
    assert [f["source"] for f in listing["folders"]] == ["bundled", "user"]
    assert listing["folders"][1]["path"].startswith(str(data_dir))
    assert "MultiplexedPath" not in listing["folders"][0]["path"]
    row = next(r for r in listing["rows"] if r["file"] == expected["file"])
    assert row.items() >= {**expected, "source": "bundled", "removable": False}.items()
    if kind == "rbe_tables":
        assert row["model"] and "α_x=" in row["tissue_classes"] and row["problem"] is None


def test_unknown_kind(client):
    response = client.call("base_data.list", {"kind": "nope"}, session=False)
    assert response["error"]["type"] == "ValueError"


def test_add_and_remove_a_stopping_power_table(client, data_dir):
    data = _shipped_bytes("stopping_power_tables", "SPtable.mat")
    revision = base_data_revision()

    dry = _add(client, "stopping_power_tables", "SPtable.mat", data, dry_run=True)["result"]
    assert dry == {
        "row": None,
        "suggested_name": "SPtable.mat",
        "radiation_mode": None,
        "clash": {"file": "SPtable.mat", "source": "bundled"},
    }
    clash = _add(client, "stopping_power_tables", "SPtable.mat", data, overwrite=True)
    assert clash["error"]["code"] == "name_clash"
    assert base_data_revision() == revision

    added = _add(client, "stopping_power_tables", "SPtable.mat", data, name="mine")["result"]
    assert added["row"]["file"] == "mine.mat" and added["row"]["source"] == "user"
    assert added["row"]["removable"] and added["clash"] is None
    assert (data_dir / "physics" / "stopping" / "mine.mat").read_bytes() == data
    assert base_data_revision() > revision
    assert "mine.mat" in [r["file"] for r in _rows(client, "stopping_power_tables")]

    again = _add(client, "stopping_power_tables", "x.mat", data, name="mine.mat")
    assert again["error"]["code"] == "name_clash"
    replaced = _add(client, "stopping_power_tables", "x.mat", data, name="mine", overwrite=True)
    assert replaced["ok"] and replaced["result"]["clash"]["source"] == "user"

    params = {"kind": "stopping_power_tables", "file": "SPtable.mat"}
    response = client.call("base_data.remove", params, session=False)
    assert response["error"]["code"] == "not_removable"
    params["file"] = "mine.mat"
    assert client.ok("base_data.remove", params, session=False)["result"] == {}
    assert "mine.mat" not in [r["file"] for r in _rows(client, "stopping_power_tables")]
    response = client.call("base_data.remove", params, session=False)
    assert response["error"]["type"] == "FileNotFoundError"


def test_add_a_machine_under_another_name(client, data_dir):
    data = _shipped_bytes("machines", "photons_Generic.mat")
    dry = _add(client, "machines", "upload.mat", data, dry_run=True)["result"]
    assert dry["suggested_name"] == "Generic" and dry["radiation_mode"] == "photons"
    assert dry["clash"] == {"file": "photons_Generic.mat", "source": "bundled"}

    added = _add(client, "machines", "upload.mat", data, name="Mine")["result"]
    assert added["row"] | {"path": None} == {
        "name": "Mine",
        "file": "photons_Mine.mat",
        "source": "user",
        "removable": True,
        "path": None,
        "radiation_mode": "photons",
    }
    from pyRadPlan.machines import get_available_machines

    assert "Mine" in get_available_machines()["photons"]
    client.ok("base_data.remove", {"kind": "machines", "file": "photons_Mine.mat"}, session=False)


def test_add_an_rbe_table(client, data_dir):
    data = _shipped_bytes("rbe_tables", RBE_TABLE)
    added = _add(client, "rbe_tables", RBE_TABLE, data, name="copy")["result"]
    assert added["row"]["file"] == "copy.mat" and added["row"]["model"]


def test_add_a_phantom(client, data_dir):
    data = _shipped_bytes("phantoms", "TG119.mat")
    added = _add(client, "phantoms", "TG119.mat", data, name="MyPhantom")["result"]
    assert added["row"]["name"] == "MyPhantom" and added["row"]["file"] == "MyPhantom.mat"
    assert "MyPhantom" in client.ok("app.info", session=False)["result"]["phantoms"]


def test_unreadable_files_are_refused(client, data_dir):
    for kind in ("machines", "rbe_tables", "stopping_power_tables", "phantoms"):
        response = _add(client, kind, "broken.mat", b"no matlab file")
        assert response["error"]["type"] == "ValueError", kind
    assert not (data_dir / "machines").exists()


@pytest.mark.parametrize(
    "name", ["../outside.mat", "..\\outside.mat", "sub/outside.mat", "/abs/outside.mat", ".."]
)
def test_remove_refuses_names_leaving_the_user_folder(client, data_dir, name):
    victim = data_dir / "outside.mat"
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_bytes(b"keep")
    (data_dir / "rbe_tables").mkdir(parents=True, exist_ok=True)
    (data_dir / "rbe_tables" / "inside.mat").write_bytes(b"x")
    response = client.call("base_data.remove", {"kind": "rbe_tables", "file": name}, session=False)
    assert not response["ok"] and response["error"]["type"] == "ValueError"
    assert victim.read_bytes() == b"keep"


def test_core_lookups_reject_path_separators(data_dir):
    from pyRadPlan.core import remove_base_data_file

    victim = data_dir / "outside.mat"
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_bytes(b"keep")
    for name in ("../outside.mat", "a/b.mat", "..", "a\\b.mat"):
        with pytest.raises(ValueError):
            find_base_data("rbe_tables", name)
        with pytest.raises(ValueError):
            remove_base_data_file("rbe_tables", name)
    assert victim.read_bytes() == b"keep"
