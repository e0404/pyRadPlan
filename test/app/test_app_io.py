import io
import os
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


def _dose_array(session, key="physical_dose", result="current"):
    return sitk.GetArrayFromImage(session.workspace.results[result][key].values)


def test_result_npz_round_trip(pipeline, client):
    source_client, source = pipeline
    result, data = _save(source_client, {"object": "result", "format": "npz"})
    assert result["filename"] == "physical_dose.npz"
    session = _patient_client(client)
    response = _load(client, "data.load_file", result["filename"], data)
    assert response["ok"], response.get("error")
    assert list(session.workspace.results) == ["imported"]
    assert not session.workspace.has("result")
    np.testing.assert_allclose(_dose_array(session, result="imported"), _dose_array(source))


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
    beam = sitk.GetArrayFromImage(source.workspace.result.beam.physical_dose[0].values)
    np.testing.assert_allclose(gui["physical_dose_beam_0"], np.transpose(beam, (1, 2, 0)))


def test_result_mat_holds_a_dose_of_another_grid_on_the_ct_grid(tmp_path):
    from types import SimpleNamespace

    from pyRadPlan.app._io import _export_format, write_export
    from pyRadPlan.io.matlab import _matfile

    cube = sitk.GetImageFromArray(np.zeros((6, 8, 10), dtype=np.float32))
    cube.SetSpacing((2.0, 2.0, 2.0))
    cube.SetOrigin((-10.0, -8.0, -6.0))
    on_ct = sitk.Cast(cube, sitk.sitkFloat64) + 1.0
    coarse = sitk.GetImageFromArray(np.full((3, 4, 5), 2.0))
    coarse.SetSpacing((4.0, 4.0, 4.0))
    coarse.SetOrigin((-9.0, -7.0, -5.0))

    path = tmp_path / "result.mat"
    images = {"physical_dose": on_ct, "imported/physical_dose": coarse}
    entry = _export_format("result", "mat", None)
    write_export(entry, images, SimpleNamespace(cube_hu=cube), str(path))

    gui = _matfile.load(path)["resultGUI"]
    assert gui["physicalDose"].shape == (8, 10, 6)
    assert gui["physicalDose_imported"].shape == (8, 10, 6)
    np.testing.assert_allclose(gui["physicalDose"], 1.0)
    np.testing.assert_allclose(gui["physicalDose_imported"][2:6, 2:8, 2:4], 2.0)


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


def test_workspace_pickle_round_trip_restores_every_result(client):
    session = _patient_client(client)
    dose = sitk.Image(session.workspace.ct.cube_hu)
    coarse = sitk.Image([4, 4, 4], sitk.sitkFloat32)
    coarse.SetSpacing((20.0, 20.0, 20.0))
    session.workspace.set_many(
        results={
            "plan1": {"physical_dose": dose, "rbe_x_dose": dose},
            "plan2": {"physical_dose": dose, "w": np.ones(3)},
            "measured": {"physical_dose": coarse},
        },
        active_result="plan2",
    )
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
        loaded = other.manager.get(other.session).workspace
        assert list(loaded.results) == ["plan1", "plan2", "measured"]
        assert loaded.active_result == "plan2"
        np.testing.assert_array_equal(loaded.result.fluence, np.ones(3))
        assert loaded.results["measured"]["physical_dose"].values.GetSize() == (4, 4, 4)
        assert response["result"]["results"]["names"] == ["plan1", "plan2", "measured"]
        assert response["result"]["results"]["active"] == "plan2"
        assert response["result"]["results"]["tags"] == ["plan1", "measured"]
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


# ----------------------------------------------------------------------
# Paths (the Qt GUI): the same writers as the downloads
# ----------------------------------------------------------------------


def _save_path(session, **params):
    return session.execute(session.prepare("save_path", **params))


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ""])
def test_workspace_saved_to_a_path_loads_back(pipeline, tmp_path, suffix):
    from pyRadPlan.app import Session

    _, source = pipeline
    path = tmp_path / f"ws{suffix}"
    written = _save_path(source, object="workspace", format="pickle", path=path)
    assert written == [str(path) if suffix else str(path) + ".pkl"]
    restored = Session(auto_plan=False)
    restored.load_path(written[0])
    assert restored.workspace.present() == source.workspace.present()
    assert restored.workspace.dij.total_num_of_bixels == source.workspace.dij.total_num_of_bixels


@pytest.mark.parametrize("name", ["ws.mat", "ws.npz", "ws.txt"])
def test_a_path_with_another_ending_is_refused(pipeline, tmp_path, name):
    _, source = pipeline
    with pytest.raises(
        ValueError, match=r"Use \.pkl or \.pickle to save the complete workspace\."
    ):
        source.prepare("save_path", object="workspace", format="pickle", path=tmp_path / name)
    assert not (tmp_path / name).exists()


def test_label_maps_are_written_to_a_folder(pipeline, tmp_path):
    from pyRadPlan.io import load_data

    _, source = pipeline
    written = _save_path(source, object="cst", format="nrrd", path=tmp_path / "labels")
    assert written == [str(tmp_path / "labels")]
    assert _names(load_data(tmp_path / "labels")["cst"]) == _names(source.workspace.cst)


EXPORT_EXT = {"nifti": ".nii.gz", "nrrd": ".nrrd", "meta": ".mha"}


@pytest.mark.parametrize("fmt", ["nifti", "nrrd", "meta"])
def test_result_quantities_saved_to_paths_match_the_downloads(pipeline, tmp_path, fmt):
    source_client, source = pipeline
    _, download = _save(source_client, {"object": "result", "format": fmt})
    (tmp_path / "download").mkdir()
    downloaded = tmp_path / "download" / f"dose{EXPORT_EXT[fmt]}"
    downloaded.write_bytes(download)
    (single,) = _save_path(
        source, object="result", format=fmt, path=tmp_path / "dose", quantities=["physical_dose"]
    )
    assert single == str(tmp_path / f"dose{EXPORT_EXT[fmt]}")
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(sitk.ReadImage(single)),
        sitk.GetArrayFromImage(sitk.ReadImage(str(downloaded))),
    )
    keys = ["physical_dose", "physical_dose_beam 0"]
    several = _save_path(
        source, object="result", format=fmt, path=tmp_path / "all", quantities=keys
    )
    assert sorted(os.path.basename(p) for p in several) == sorted(
        f"{k.replace(' ', '_')}{EXPORT_EXT[fmt]}" for k in keys
    )


def test_dialog_filters_come_from_the_format_tables():
    from pyRadPlan.app._io import entry_for_filter, export_entries, export_filter, load_filter

    load = load_filter().split(";;")
    assert load[0].startswith("All supported (") and "*.MAT" in load[0]
    assert load[-1] == "All files (*)"
    assert "MATLAB / matRad (*.mat *.MAT)" in load
    assert not any(entry.startswith("DICOM") for entry in load)
    assert export_filter(export_entries("workspace")) == "pyRadPlan workspace (*.pkl *.pickle)"
    images = export_entries("result", ("nifti", "nrrd", "meta"))
    text = export_filter(images)
    assert text == "NIfTI (*.nii.gz *.nii);;NRRD (*.nrrd);;MetaImage (*.mha *.mhd)"
    assert entry_for_filter("NRRD (*.nrrd)", images).format == "nrrd"
    assert entry_for_filter("", images).format == "nifti"


# ----------------------------------------------------------------------
# Native imports: image files, binary folders, DICOM folders
# ----------------------------------------------------------------------


def _image(path, array, spacing=(2.0, 2.0, 3.0)):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))
    return str(path)


@pytest.fixture
def image_files(tmp_path):
    ct = np.full((4, 8, 8), -1000, dtype=np.int16)
    ct[1:3, 2:6, 2:6] = 0
    mask = np.zeros((4, 8, 8), dtype=np.uint8)
    mask[1:3, 3:5, 3:5] = 1
    labels = mask.copy()
    labels[1:3, 2, 2] = 2
    return {
        "ct": _image(tmp_path / "ct.nii.gz", ct),
        "other_ct": _image(tmp_path / "other.nii.gz", ct[:, :6, :6]),
        "mask": _image(tmp_path / "PTV.nrrd", mask),
        "labels": _image(tmp_path / "labels.nrrd", labels),
        "dose": _image(tmp_path / "plan dose.mha", np.full((4, 8, 8), 1.5, dtype=np.float32)),
        "coarse_dose": _image(
            tmp_path / "coarse.mha", np.full((2, 4, 4), 2.5, dtype=np.float32), (4.0, 4.0, 6.0)
        ),
    }


def _import(session, path, mode, **params):
    return session.execute(session.prepare("import_image", path=path, mode=mode, **params))


def test_image_files_import_as_ct_structures_and_dose(image_files):
    from pyRadPlan.app import Session
    from pyRadPlan.app._io import inspect_image

    session = Session(auto_plan=False)
    _import(session, image_files["ct"], "ct_new")
    ct = session.workspace.ct
    assert ct.cube_hu.GetSize() == (8, 8, 4)

    info = inspect_image(image_files["labels"], ct.cube_hu)
    assert (info.num_labels, info.grid_matches, info.inferred) == (2, True, "structures")
    _import(session, image_files["mask"], "structures")
    _import(session, image_files["mask"], "structures")
    assert [voi.name for voi in session.workspace.cst.vois] == ["PTV", "PTV_2"]

    assert inspect_image(image_files["dose"]).default_name == "plan_dose"
    assert _import(session, image_files["dose"], "dose") == "plan_dose"
    assert _import(session, image_files["dose"], "dose") == "plan_dose_2"
    assert (
        _import(session, image_files["coarse_dose"], "dose", name="coarse dose") == "coarse_dose"
    )
    ws = session.workspace
    assert list(ws.results) == ["plan_dose", "plan_dose_2", "coarse_dose"]
    assert not ws.has("result") and ws.active_result == "current"
    image = ws.results["plan_dose"]["physical_dose"].values
    assert sitk.GetArrayFromImage(image).max() == pytest.approx(1.5)
    assert image.GetSize() == ct.cube_hu.GetSize()
    coarse = ws.results["coarse_dose"]["physical_dose"].values
    assert coarse.GetSize() == (4, 4, 2) and coarse.GetSpacing() == (4.0, 4.0, 6.0)
    state = session.state()
    assert state["quantities"] == [
        "plan_dose/physical_dose",
        "plan_dose_2/physical_dose",
        "coarse_dose/physical_dose",
    ]
    assert state["actions"]["save_result"] and not state["actions"]["snapshot"]
    volume = session.volume("coarse_dose/physical_dose")
    assert volume["data"].shape == (4, 8, 8) and volume["unit"] == "Gy"
    assert volume["data"].max() == pytest.approx(2.5)

    _import(session, image_files["ct"], "ct_replace")
    assert session.workspace.has("cst") and session.workspace.ct is not ct
    assert len(session.workspace.results) == 3
    _import(session, image_files["other_ct"], "ct_replace")
    assert session.workspace.present() == {"ct"} and session.workspace.results == {}


def test_imported_dose_leaves_the_active_result_and_its_staleness(image_files):
    from pyRadPlan.app import Session

    session = Session(auto_plan=False)
    _import(session, image_files["ct"], "ct_new")
    dose = sitk.Cast(session.workspace.ct.cube_hu, sitk.sitkFloat32)
    session.workspace.result = {"physical_dose": dose}
    active = session.workspace.result
    session.workspace.pln = object()
    assert session.tracker.result_stale

    assert _import(session, image_files["coarse_dose"], "dose", name="current") == "current_2"
    assert session.snapshot("kept") == "kept"
    assert session.workspace.result is active
    assert session.tracker.result_stale and session.stages()["optimization"] == "stale"
    assert list(session.workspace.results) == ["current", "current_2", "kept"]

    session.workspace.result = {"physical_dose": dose}
    assert not session.tracker.result_stale
    assert list(session.workspace.results) == ["current", "current_2", "kept"]


def test_image_imports_need_a_ct_and_sitk_file_io(image_files, monkeypatch):
    from pyRadPlan.app import PreconditionError, Session
    from pyRadPlan.app import _io

    session = Session(auto_plan=False)
    with pytest.raises(PreconditionError) as error:
        session.prepare("import_image", path=image_files["mask"], mode="structures")
    assert error.value.code == "no_data"
    monkeypatch.setattr(_io, "sitk_missing", lambda *names: "no file IO here")
    for action, params in (
        ("import_image", {"path": image_files["ct"], "mode": "ct_new"}),
        ("import_binary", {"ct_file": image_files["ct"], "selections": []}),
        ("scan_dicom", {"directory": "."}),
    ):
        with pytest.raises(PreconditionError) as error:
            session.prepare(action, **params)
        assert error.value.code == "not_available", action


def test_binary_folder_imports_the_ct_and_masks(image_files, tmp_path):
    from pyRadPlan.app import Session
    from pyRadPlan.app._io import binary_folder_files, folder_kind

    folder = tmp_path / "patient"
    (folder / "masks").mkdir(parents=True)
    ct = _image(folder / "ct.nii.gz", sitk.GetArrayFromImage(sitk.ReadImage(image_files["ct"])))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_files["mask"]))
    _image(folder / "masks" / "PTV.nrrd", mask)
    assert folder_kind(folder) == "images"
    ct_file, masks = binary_folder_files(folder)
    assert ct_file == ct and [os.path.basename(m) for m in masks] == ["PTV.nrrd"]
    session = Session(auto_plan=False)
    selections = [{"path": masks[0], "name": "Target", "voi_type": "TARGET"}]
    session.execute(session.prepare("import_binary", ct_file=ct_file, selections=selections))
    assert [voi.name for voi in session.workspace.cst.vois] == ["Target"]
    selections[0]["voi_type"] = "IGNORED"
    session.execute(session.prepare("import_binary", ct_file=ct_file, selections=selections))
    assert session.workspace.present() == {"ct"}


def test_dicom_folder_is_scanned_then_imported():
    from pathlib import Path

    from pyRadPlan.app import Session
    from pyRadPlan.app._io import folder_kind

    folder = Path(__file__).parents[1] / "data" / "dicom"
    assert folder_kind(folder) == "dicom"
    session = Session(auto_plan=False)
    scan = session.execute(session.prepare("scan_dicom", directory=folder))
    catalog = scan["catalog"]
    assert catalog["series"] and catalog["structs"]
    selection = {
        "series_uid": catalog["series"][0]["series_uid"],
        "struct_file": catalog["structs"][0]["path"],
        "load_dose": False,
        "dose_file": None,
    }
    session.execute(
        session.prepare("import_dicom", importer=scan["importer"], selection=selection)
    )
    assert session.workspace.has("ct", "cst") and not session.workspace.has("result")
