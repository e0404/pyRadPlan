"""File and base-data methods against the Pyodide SimpleITK stand-in, natively in a subprocess."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHIMS = ROOT / "web" / "pyodide" / "shims"

SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(1, sys.argv[2])
import SimpleITK as sitk
assert sitk.IS_PYRADPLAN_SHIM
from pyRadPlan.app import SessionManager, dispatch

manager = SessionManager()

def call(method, params=None, session=None, buffers=None):
    request = {"id": 1, "method": method, "params": params or {}, "session": session}
    if buffers is not None:
        request["buffers"] = buffers
    return dispatch(request, manager=manager, with_traceback=False)

def new_session():
    return call("session.create")["result"]["session"]

out = {}
info = call("app.info")["result"]
out["features"] = {k: v["available"] for k, v in info["capabilities"]["features"].items()}
out["export"] = {
    f"{e['object']}:{e['format']}:{e.get('structure_format') or ''}": e["available"]
    for e in info["capabilities"]["formats"]["export"]
}
out["import"] = {e["format"]: e["available"] for e in info["capabilities"]["formats"]["import"]}

source = new_session()
call("data.load_phantom", {"name": "TG119"}, source)
session = manager.get(source)
dose = sitk.Cast(session.workspace.ct.cube_hu, sitk.sitkFloat64) * 0.0 + 2.0
session.workspace.result = {"physical_dose": dose}

saved = {}
for obj, fmt, extra in [
    ("pln", "mat", {}), ("pln", "pickle", {}), ("cst", "pickle", {}), ("cst", "npz", {}),
    ("cst", "dcm", {"structure_format": "seg"}), ("cst", "nifti", {}),
    ("result", "npz", {}), ("result", "mat", {}), ("result", "dcm", {}), ("result", "nrrd", {}),
    ("workspace", "pickle", {}),
]:
    response = call("export.save", {"object": obj, "format": fmt, **extra}, source)
    key = f"{obj}:{fmt}"
    if response["ok"]:
        saved[key] = (response["result"]["filename"], bytes(response["buffers"][0]))
        out[key] = "ok"
    else:
        out[key] = response["error"].get("code") or response["error"]["type"]

target = new_session()
for key in ("workspace:pickle", "cst:npz", "result:npz", "pln:mat"):
    filename, data = saved[key]
    response = call("data.load_file", {"filename": filename, "data": {"$array": 0}}, target,
                    [data])
    state = response.get("result")
    out[f"load {key}"] = (
        {**state["has"], "results": state["results"]["names"]} if response["ok"] else response["error"]
    )
response = call("data.load_file", {"filename": "ct.nii", "data": {"$array": 0}}, target, [b"x"])
out["load nii"] = response["error"].get("code")
print("RESULT" + json.dumps(out))
"""


@pytest.fixture(scope="module")
def shim_run():
    pytest.importorskip("pydicom")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(SHIMS), str(ROOT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT")]
    assert lines, completed.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT") :])


def test_capabilities_detect_the_missing_image_io(shim_run):
    features = shim_run["features"]
    assert not features["image_file_io"] and not features["dicom_import"]
    export = shim_run["export"]
    for obj in ("cst", "result"):
        for fmt in ("nifti", "nrrd", "meta"):
            assert export[f"{obj}:{fmt}:"] is False
    for key in (
        "pln:mat:",
        "pln:pickle:",
        "dij:mat:",
        "cst:npz:",
        "cst:dcm:seg",
        "result:npz:",
        "result:mat:",
        "result:dcm:",
        "workspace:pickle:",
    ):
        assert export[key] is True, key
    imports = shim_run["import"]
    assert imports["mat"] and imports["npz"] and imports["pickle"]
    assert not imports["nifti"] and not imports["dcm"]


def test_browser_formats_save_and_image_formats_are_refused(shim_run):
    for key in (
        "pln:mat",
        "pln:pickle",
        "cst:pickle",
        "cst:npz",
        "cst:dcm",
        "result:npz",
        "result:mat",
        "result:dcm",
        "workspace:pickle",
    ):
        assert shim_run[key] == "ok", key
    assert shim_run["cst:nifti"] == "not_available"
    assert shim_run["result:nrrd"] == "not_available"
    assert shim_run["load nii"] == "not_available"


def test_files_saved_with_the_stand_in_load_back(shim_run):
    workspace = shim_run["load workspace:pickle"]
    assert workspace["ct"] and workspace["cst"] and workspace["pln"] and workspace["result"]
    assert shim_run["load cst:npz"]["cst"]
    assert shim_run["load result:npz"]["results"] == ["imported"]
    assert shim_run["load pln:mat"]["pln"]
