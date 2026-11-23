import pytest

from pyRadPlan.app import PreconditionError, Session, SessionManager, _io, dispatch


def _call(manager, method, params=None, buffers=None, session=None):
    request = {"id": 1, "method": method, "params": params or {}, "buffers": buffers or []}
    if session:
        request["session"] = session
    return dispatch(request, manager=manager)


@pytest.fixture
def browser(monkeypatch):
    monkeypatch.setattr(_io, "running_in_pyodide", lambda: True)


def _refused(response):
    assert not response["ok"]
    assert response["error"]["type"] == "PreconditionError"
    assert response["error"]["code"] == "not_available"


def test_pickle_loading_is_refused_in_the_browser(browser):
    manager = SessionManager()
    session_id = _call(manager, "session.create")["result"]["session"]
    payload = {"filename": "evil.pkl", "data": {"$array": 0}}
    _refused(_call(manager, "data.load_file", payload, [b"x"], session_id))
    _refused(_call(manager, "data.load_dij", payload, [b"x"], session_id))
    session = manager.get(session_id)
    with pytest.raises(PreconditionError):
        session.load_file("evil.pickle", b"x")
    with pytest.raises(PreconditionError):
        session.load_path("evil.pkl")
    with pytest.raises(PreconditionError):
        session.prepare("load_dij", path="evil.pkl")
    add = {"kind": "phantoms", "filename": "evil.pkl", "data": {"$array": 0}}
    _refused(_call(manager, "base_data.add", add, [b"x"]))
    _refused(_call(manager, "base_data.add", {**add, "dry_run": True}, [b"x"]))


def test_capabilities_report_pickle_loading(browser):
    info = _call(SessionManager(), "app.info")["result"]
    feature = info["capabilities"]["features"]["pickle_load"]
    assert not feature["available"] and "browser" in feature["reason"]
    row = next(r for r in info["capabilities"]["formats"]["import"] if r["format"] == "pickle")
    assert not row["available"] and row["reason"] == feature["reason"]
    exports = info["capabilities"]["formats"]["export"]
    assert any(e["format"] == "pickle" and e["available"] for e in exports)


def test_pickle_loading_stays_available_natively():
    info = _call(SessionManager(), "app.info")["result"]
    assert info["capabilities"]["features"]["pickle_load"] == {"available": True, "reason": None}


def test_manager_can_forbid_pickles():
    manager = SessionManager(allow_pickle=False)
    info = _call(manager, "app.info")["result"]
    assert not info["capabilities"]["features"]["pickle_load"]["available"]
    session = manager.create()
    assert session.allow_pickle is False
    with pytest.raises(PreconditionError):
        session.load_file("x.pkl", b"x")
    assert SessionManager().create().allow_pickle is True


def test_global_settings_can_be_forbidden(tmp_path, monkeypatch):
    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path))
    manager = SessionManager(allow_global_settings=False)
    _refused(_call(manager, "app.settings_set", {"section": "gui", "values": {}}))
    add = {"kind": "rbe_tables", "filename": "x.mat", "data": {"$array": 0}}
    _refused(_call(manager, "base_data.add", add, [b"x"]))
    _refused(_call(manager, "base_data.remove", {"kind": "rbe_tables", "file": "x.mat"}))
    assert _call(manager, "app.settings")["ok"]
    assert _call(manager, "base_data.list", {"kind": "rbe_tables"})["ok"]


def test_in_process_uploads_apply_the_image_checks(monkeypatch):
    monkeypatch.setattr(_io, "sitk_missing", lambda *names: "no image file IO here")
    session = Session(auto_plan=False)
    with pytest.raises(PreconditionError) as error:
        session.load_file("scan.nii.gz", b"x")
    assert error.value.code == "not_available"
    with pytest.raises(PreconditionError) as error:
        session.load_file("scan.nii.gz", b"x", kind="ct_new")
    assert error.value.code == "not_available"
    with pytest.raises(ValueError):
        session.load_file("scan.mat", b"x", kind="bogus")
