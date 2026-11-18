import io
import zipfile

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.app._io import EXPORT_FORMATS


def _save(client, params):
    response = client.ok("export.save", params)
    result = response["result"]
    data = bytes(response["buffers"][result["data"]["$array"]])
    return result, data


def _upload(data):
    return {"$array": 0, "dtype": "uint8"}, [data]


def _load(client, method, filename, data, **params):
    descriptor, buffers = _upload(data)
    return client.call(method, {"filename": filename, "data": descriptor, **params}, buffers)


def _patient_client(client):
    client.ok("data.load_phantom", {"name": "TG119"})
    return client.manager.get(client.session)


def _unzip(data, folder):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(folder)
        return sorted(archive.namelist())


@pytest.fixture(scope="module")
def pipeline(tg119_pipeline):
    client, _, _ = tg119_pipeline
    return client, client.manager.get(client.session)


def test_every_export_format_is_listed_and_available_natively(client):
    info = client.ok("app.info", session=False)["result"]
    exports = info["capabilities"]["formats"]["export"]
    assert len(exports) == len(EXPORT_FORMATS)
    for entry in exports:
        assert entry["available"], entry
        assert entry["mime"] in (
            "application/x-matlab-data",
            "application/octet-stream",
            "application/zip",
        )
    dicom = [e for e in exports if e["object"] == "cst" and e["format"] == "dcm"]
    assert {e["structure_format"] for e in dicom} == {"rtstruct", "seg"}
    cst_formats = {e["format"]: e.get("keeps_objectives") for e in exports if e["object"] == "cst"}
    assert cst_formats["mat"] and cst_formats["pickle"] and not cst_formats["npz"]
    imports = {e["format"]: e for e in info["capabilities"]["formats"]["import"]}
    assert imports["mat"]["available"] and ".pkl" in imports["pickle"]["extensions"]


@pytest.mark.parametrize("fmt", ["mat", "pickle"])
def test_plan_round_trip(pipeline, client, fmt):
    source_client, source = pipeline
    result, data = _save(source_client, {"object": "pln", "format": fmt})
    assert result["filename"] == f"pln{'.mat' if fmt == 'mat' else '.pkl'}"
    _patient_client(client)
    client.ok("plan.set", {"gantry_angles": [45.0]})
    response = _load(client, "data.load_file", result["filename"], data)
    assert response["ok"], response.get("error")
    assert response["result"]["plan"] == source_client.ok("plan.get")["result"]


@pytest.mark.parametrize("fmt", ["mat", "pickle"])
def test_dij_round_trip_through_load_dij(pipeline, client, fmt):
    source_client, source = pipeline
    result, data = _save(source_client, {"object": "dij", "format": fmt, "basename": "influence"})
    assert result["filename"].startswith("influence.")
    session = _patient_client(client)
    response = _load(client, "data.load_dij", result["filename"], data)
    assert response["ok"], response.get("error")
    assert response["result"]["has"]["dij"]
    loaded, original = session.workspace.dij, source.workspace.dij
    assert loaded.total_num_of_bixels == original.total_num_of_bixels


def test_load_dij_without_dij_fails(pipeline, client):
    source_client, _ = pipeline
    result, data = _save(source_client, {"object": "pln", "format": "mat"})
    _patient_client(client)
    response = _load(client, "data.load_dij", result["filename"], data)
    assert response["error"]["type"] == "PreconditionError"
    assert response["error"]["code"] == "no_dij"


def _names(cst):
    return sorted(voi.name for voi in cst.vois)


@pytest.mark.parametrize("fmt", ["pickle", "npz"])
def test_structure_set_round_trip(pipeline, client, fmt):
    source_client, source = pipeline
    result, data = _save(source_client, {"object": "cst", "format": fmt})
    session = _patient_client(client)
    session.workspace.cst = None
    response = _load(client, "data.load_file", result["filename"], data)
    assert response["ok"], response.get("error")
    loaded = session.workspace.cst
    assert _names(loaded) == _names(source.workspace.cst)
    for voi in source.workspace.cst.vois:
        other = next(v for v in loaded.vois if v.name == voi.name)
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(other.mask), sitk.GetArrayFromImage(voi.mask)
        )


def test_structure_set_mat_holds_the_cst(pipeline, tmp_path):
    from pyRadPlan.cst import validate_cst
    from pyRadPlan.io.matlab import _matfile

    source_client, source = pipeline
    result, data = _save(source_client, {"object": "cst", "format": "mat"})
    assert result["mime"] == "application/x-matlab-data"
    path = tmp_path / result["filename"]
    path.write_bytes(data)
    cst = validate_cst(_matfile.load(path)["cst"], source.workspace.ct)
    assert _names(cst) == _names(source.workspace.cst)


@pytest.mark.parametrize("structure_format", ["rtstruct", "seg"])
def test_structure_set_dicom_zip_loads_back(pipeline, tmp_path, structure_format):
    from pyRadPlan.io import load_data

    source_client, source = pipeline
    result, data = _save(
        source_client, {"object": "cst", "format": "dcm", "structure_format": structure_format}
    )
    assert result["mime"] == "application/zip" and result["filename"] == "cst.zip"
    members = _unzip(data, tmp_path)
    assert members == result["files"]
    assert any(name.startswith("CT_") for name in members)
    loaded = load_data(tmp_path)
    assert set(_names(loaded["cst"])) <= set(_names(source.workspace.cst))
    assert loaded["cst"].vois


@pytest.mark.parametrize("fmt", ["nifti", "nrrd", "meta"])
def test_structure_set_label_map_zip_loads_back(pipeline, tmp_path, fmt):
    from pyRadPlan.io import load_data

    source_client, source = pipeline
    result, data = _save(source_client, {"object": "cst", "format": fmt})
    assert result["mime"] == "application/zip"
    _unzip(data, tmp_path)
    loaded = load_data(tmp_path)
    assert _names(loaded["cst"]) == _names(source.workspace.cst)


def _dose_array(session, key="physical_dose"):
    return sitk.GetArrayFromImage(session.workspace.result[key])


def test_result_npz_round_trip(pipeline, client):
    source_client, source = pipeline
    result, data = _save(source_client, {"object": "result", "format": "npz"})
    assert result["filename"] == "physical_dose.npz"
    session = _patient_client(client)
    response = _load(client, "data.load_file", result["filename"], data)
    assert response["ok"], response.get("error")
    np.testing.assert_allclose(_dose_array(session), _dose_array(source))


def test_result_mat_holds_every_selected_quantity_in_result_gui(pipeline, tmp_path):
    from pyRadPlan.io.matlab import _matfile

    source_client, source = pipeline
    keys = ["physical_dose", "physical_dose_beam 0"]
    result, data = _save(source_client, {"object": "result", "format": "mat", "quantities": keys})
    path = tmp_path / result["filename"]
    path.write_bytes(data)
    gui = _matfile.load(path)["resultGUI"]
    expected = np.transpose(_dose_array(source), (1, 2, 0))
    np.testing.assert_allclose(gui["physicalDose"], expected)
    beam = sitk.GetArrayFromImage(source.workspace.result["physical_dose_beam"][0])
    np.testing.assert_allclose(gui["physical_dose_beam_0"], np.transpose(beam, (1, 2, 0)))


def test_result_dicom_zip_holds_one_rtdose_per_quantity(pipeline, tmp_path):
    import pydicom

    source_client, source = pipeline
    single, data = _save(source_client, {"object": "result", "format": "dcm"})
    assert _unzip(data, tmp_path / "one") == ["RTDOSE.dcm"]
    ds = pydicom.dcmread(tmp_path / "one" / "RTDOSE.dcm")
    dose = ds.pixel_array * float(ds.DoseGridScaling)
    np.testing.assert_allclose(dose.max(), _dose_array(source).max(), rtol=1e-6)
    keys = ["physical_dose", "physical_dose_beam 1"]
    several, data = _save(source_client, {"object": "result", "format": "dcm", "quantities": keys})
    assert several["files"] == ["physical_dose/RTDOSE.dcm", "physical_dose_beam_1/RTDOSE.dcm"]


@pytest.mark.parametrize("fmt", ["nifti", "nrrd", "meta"])
def test_result_image_files(pipeline, tmp_path, fmt):
    source_client, source = pipeline
    result, data = _save(source_client, {"object": "result", "format": fmt})
    path = tmp_path / result["filename"]
    path.write_bytes(data)
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(sitk.ReadImage(str(path))), _dose_array(source)
    )
    keys = ["physical_dose", "physical_dose_beam 0"]
    several, data = _save(source_client, {"object": "result", "format": fmt, "quantities": keys})
    assert several["filename"] == "result.zip" and len(several["files"]) == 2


def test_workspace_pickle_round_trip_restores_snapshot_tags(client):
    session = _patient_client(client)
    dose = sitk.Image(session.workspace.ct.cube_hu)
    session.workspace.result = {
        "physical_dose": dose,
        "physical_dose_plan1": dose,
        "rbe_x_dose": dose,
        "rbe_x_dose_plan1": dose,
    }
    session.set_snapshot_tags(["plan1"])
    assert session.snapshot_tags() == ["plan1"]
    result, data = _save(client, {"object": "workspace", "format": "pickle"})
    assert result["filename"] == "workspace.pkl"

    other = type(client)()
    other.session = other.ok("session.create", session=False)["result"]["session"]
    try:
        response = _load(other, "data.load_file", result["filename"], data)
        assert response["ok"], response.get("error")
        assert response["result"]["has"] == dict.fromkeys(response["result"]["has"], False) | {
            "ct": True,
            "cst": True,
            "pln": True,
            "result": True,
        }
        assert other.manager.get(other.session).snapshot_tags() == ["plan1"]
        assert response["result"]["results"]["tags"] == ["plan1"]
    finally:
        other.manager.close_all()


def test_export_preconditions(client):
    response = client.call("export.save", {"object": "workspace", "format": "pickle"})
    assert response["error"]["code"] == "no_data"
    response = client.call("export.save", {"object": "pln", "format": "nifti"})
    assert response["error"]["type"] == "ValueError"
    response = client.call("export.save", {"object": "stf", "format": "mat"})
    assert response["error"]["type"] == "ValueError"
    _patient_client(client)
    response = client.call("export.save", {"object": "result", "format": "npz"})
    assert response["error"]["code"] == "no_data"


def test_unknown_result_quantity(pipeline):
    source_client, _ = pipeline
    response = source_client.call(
        "export.save", {"object": "result", "format": "npz", "quantities": ["nope"]}
    )
    assert response["error"]["type"] == "KeyError"


def test_load_file_kinds(client):
    descriptor, buffers = _upload(b"not an image")
    for kind in ("ct_new", "ct_replace", "structures", "dose"):
        response = client.call(
            "data.load_file", {"filename": "x.nii", "data": descriptor, "kind": kind}, buffers
        )
        assert response["error"]["code"] == "not_available"
    response = client.call(
        "data.load_file", {"filename": "x.mat", "data": descriptor, "kind": "other"}, buffers
    )
    assert response["error"]["type"] == "ValueError"
