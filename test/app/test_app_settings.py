import logging

import pytest

from pyRadPlan._settings import get_settings
from pyRadPlan.app import _settings as app_settings

FEATURES = {
    "image_file_io",
    "dicom_import",
    "dicom_export",
    "ai",
    "ipopt",
    "gpu",
    "numba",
    "pickle_load",
    "monte_carlo",
    "pause",
    "open_folder",
    "host_paths",
    "base_data_persistence",
}


@pytest.fixture
def restore_settings(monkeypatch):
    settings = get_settings()
    saved = {
        "gui": settings.gui.model_dump(),
        "base_data_path": settings.base_data_path,
    }
    for name in ("PYRADPLAN_GUI_UI_MODE", "PYRADPLAN_BASE_DATA_PATH"):
        monkeypatch.delenv(name, raising=False)
    yield settings
    settings.gui.ui_mode = saved["gui"]["ui_mode"]
    settings.base_data_path = saved["base_data_path"]


def test_runtime_info(client):
    info = client.ok("app.info", session=False)["result"]
    assert info["version"] and info["phantoms"]
    assert info["runtime"] == "native" and info["default_profile"] == "native"
    features = info["capabilities"]["features"]
    assert set(features) == FEATURES
    for name, feature in features.items():
        assert feature["available"] is (feature["reason"] is None), name
    assert features["image_file_io"]["available"] and features["pause"]["available"]
    quantities = {q["identifier"]: q for q in info["quantities"]}
    assert quantities["physical_dose"] == {
        "identifier": "physical_dose",
        "name": "dose",
        "meta_label": "Dose",
        "unit": "Gy",
    }
    assert quantities["rbe_x_dose"]["unit"] == "Gy (RBE)"


def test_ai_reason_without_the_dependency(monkeypatch):
    monkeypatch.setattr(app_settings, "module_missing", lambda *names: "missing")
    assert app_settings.ai_disabled_reason() == app_settings.AI_MISSING


def test_pyodide_features_are_reported_unavailable(monkeypatch):
    monkeypatch.setattr(app_settings, "running_in_pyodide", lambda: True)
    reasons = app_settings.feature_reasons()
    for name in ("monte_carlo", "pause", "open_folder", "host_paths", "base_data_persistence"):
        assert reasons[name], name
    assert app_settings.runtime() == "pyodide"


def test_licenses(client):
    rows = client.ok("app.licenses", session=False)["result"]
    names = [row["name"].lower() for row in rows]
    assert names == sorted(names) and "numpy" in names
    numpy = client.ok("app.licenses", {"name": "NumPy"}, session=False)["result"]
    assert numpy["name"].lower() == "numpy" and numpy["version"]
    assert numpy["text"] and "copyright" in numpy["text"].lower()
    response = client.call("app.licenses", {"name": "no-such-dist"}, session=False)
    assert response["error"]["type"] == "KeyError"


def _section(result, section):
    return next(s for s in result["sections"] if s["id"] == section)


def test_settings_forms(client, restore_settings):
    result = client.ok("app.settings", session=False)["result"]
    assert [s["id"] for s in result["sections"]] == ["general", "xp", "ai", "gui"]
    assert _section(result, "xp")["title"] == "XP (Backend)"
    general = _section(result, "general")["form"]
    assert [f["name"] for f in general["fields"]] == ["base_data_path"]
    gui = _section(result, "gui")["form"]
    (ui_mode,) = gui["fields"]
    assert ui_mode["type"] == "choice"
    assert [c["value"] for c in ui_mode["choices"]] == ["default", "educational"]
    assert gui["values"] == {"ui_mode": restore_settings.gui.ui_mode}
    xp = _section(result, "xp")["form"]
    assert xp["values"]["prefer_gpu"] == restore_settings.xp.prefer_gpu


def test_settings_set_writes_the_singleton_and_the_environment(client, restore_settings):
    import os

    params = {"section": "gui", "values": {"ui_mode": "educational"}}
    result = client.ok("app.settings_set", params, session=False)["result"]
    assert _section(result, "gui")["form"]["values"] == {"ui_mode": "educational"}
    assert restore_settings.gui.ui_mode == "educational"
    assert os.environ["PYRADPLAN_GUI_UI_MODE"] == "educational"


def test_settings_set_rejects_invalid_values(client, restore_settings):
    before = restore_settings.gui.ui_mode
    response = client.call(
        "app.settings_set", {"section": "gui", "values": {"ui_mode": "expert"}}, session=False
    )
    assert response["error"]["code"] == "invalid"
    assert [d["field"] for d in response["error"]["details"]] == ["ui_mode"]
    assert restore_settings.gui.ui_mode == before
    response = client.call(
        "app.settings_set", {"section": "gui", "values": {"nope": 1}}, session=False
    )
    assert response["error"]["code"] == "invalid"
    response = client.call("app.settings_set", {"section": "x", "values": {}}, session=False)
    assert response["error"]["type"] == "ValueError"


def test_settings_locked_without_host_paths(client, restore_settings, monkeypatch):
    monkeypatch.setattr(app_settings, "running_in_pyodide", lambda: True)
    result = client.ok("app.settings", session=False)["result"]
    (field,) = _section(result, "general")["form"]["fields"]
    assert field["editable"] is False and field["reason"]
    params = {"section": "general", "values": {"base_data_path": "/somewhere"}}
    response = client.call("app.settings_set", params, session=False)
    assert response["error"]["code"] == "locked"
    assert restore_settings.base_data_path != "/somewhere"


def test_session_log_level(client):
    assert client.ok("session.log_level", {"level": "debug"})["result"] == {"level": "debug"}
    assert client.manager.get(client.session).log_level == logging.DEBUG
    response = client.call("session.log_level", {"level": "loud"})
    assert response["error"]["type"] == "ValueError"
