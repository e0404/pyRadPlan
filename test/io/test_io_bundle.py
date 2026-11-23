"""Tests for the supported object kinds of the exporters and for ExportBundle."""

import os
import json
import warnings

import h5py
import numpy as np
import pymatreader
import pytest
import SimpleITK as sitk

from pyRadPlan import IonPlan, calc_dose_forward, calc_dose_influence, generate_stf, load_tg119
from pyRadPlan.io import ExportBundle, load_data, save_data
from pyRadPlan.io._factory import EXPORTERS, get_exporter
from pyRadPlan.io.base import ANY_OBJECT

ALL_KINDS = {"ct", "cst", "dose", "pln", "stf", "dij", "result", "weights"}


@pytest.fixture(scope="module")
def workspace():
    ct, cst = load_tg119()
    pln = IonPlan(radiation_mode="protons", machine="Generic", num_of_fractions=30)
    pln.prop_stf = {"gantry_angles": [0], "couch_angles": [0], "bixel_width": 20}
    pln.prop_dose_calc["dose_grid"] = {"resolution": {"x": 10.0, "y": 10.0, "z": 10.0}}
    pln.dose_convention = "per_fraction"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stf = generate_stf(ct, cst, pln)
        dij = calc_dose_influence(ct, cst, stf, pln)
        weights = np.ones(dij.total_num_of_bixels)
        result = calc_dose_forward(ct, cst, stf, pln, weights)
    result.fluence = weights
    return {
        "ct": ct,
        "cst": cst,
        "dose": result["physical_dose"].values,
        "pln": pln,
        "stf": stf,
        "dij": dij,
        "result": result,
        "weights": weights,
    }


def _objects(workspace, kinds):
    return {k: workspace[k] for k in kinds}


def _save_all(workspace, path, fmt):
    """Save every kind, collecting the dropped names from the warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save_data(file_name=path, format=fmt, **workspace)
    messages = [str(w.message) for w in caught if "cannot store" in str(w.message)]
    assert len(messages) <= 1
    dropped = set()
    if messages:
        listed = messages[0].split("cannot store ")[1].split(";")[0]
        dropped = {name.strip() for name in listed.split(",")}
    return dropped


def test_every_registered_exporter_declares_supports():
    for fmt, cls in EXPORTERS.items():
        assert isinstance(cls.supports, frozenset) and cls.supports, fmt
    assert get_exporter("pickle").supports == {ANY_OBJECT}
    assert get_exporter("hdf5").supports == {ANY_OBJECT}
    assert get_exporter("mat").supports == {ANY_OBJECT}
    assert get_exporter("npz").supports == {"ct", "cst", "dose"}
    for fmt in ("nifti", "nrrd", "meta"):
        assert get_exporter(fmt).supports == {"ct", "cst", "dose"}
    assert get_exporter("dcm").supports == {
        "ct",
        "cst",
        "dose",
        "result",
        "pln",
        "stf",
        "weights",
    }


def test_npz_stores_exactly_the_declared_kinds(workspace, tmp_path):
    path = tmp_path / "ws.npz"
    dropped = _save_all(workspace, path, "npz")
    with np.load(path) as fh:
        contents = set(json.loads(str(fh["meta"]))["contents"])
    assert contents == get_exporter("npz").supports
    assert dropped == ALL_KINDS - contents


@pytest.mark.parametrize("fmt", ["nifti", "nrrd", "meta"])
def test_image_formats_store_exactly_the_declared_kinds(workspace, tmp_path, fmt):
    out = tmp_path / fmt
    dropped = _save_all(workspace, out, fmt)
    stems = {p.name.split(".")[0] for p in out.iterdir()}
    assert {"ct", "cst", "dose"} <= stems
    assert stems <= {"ct", "cst", "dose"}
    assert dropped == ALL_KINDS - {"ct", "cst", "dose"}


def test_mat_stores_every_kind(workspace, tmp_path):
    path = tmp_path / "ws.mat"
    assert _save_all(workspace, path, "mat") == set()
    keys = set(pymatreader.read_mat(str(path)))
    # the result (and the dose) are stored as matRad's resultGUI
    assert {"ct", "cst", "resultGUI", "pln", "stf", "dij", "weights"} <= keys


def test_pickle_stores_every_kind(workspace, tmp_path):
    path = tmp_path / "ws.pkl"
    extras = {"saved_result_tags": ["a"]}
    save_data(file_name=path, **workspace, **extras)
    assert set(load_data(path)) == ALL_KINDS | set(extras)


def test_hdf5_stores_every_kind(workspace, tmp_path):
    path = tmp_path / "ws.h5"
    assert _save_all({**workspace, "saved_result_tags": ["a"]}, path, "hdf5") == set()
    with h5py.File(path) as fh:
        contents = set(json.loads(fh.attrs["contents"]))
    assert contents == ALL_KINDS | {"saved_result_tags"}


def test_dicom_stores_exactly_the_declared_kinds(workspace, tmp_path):
    out = tmp_path / "dcm"
    dropped = _save_all(workspace, out, "dcm")
    names = {p.name for p in out.iterdir()}
    import pydicom

    assert dropped == {"dij"}
    assert any(n.startswith("CT_") for n in names)
    assert any(n.startswith("RTSTRUCT") for n in names)
    assert "RTDOSE.dcm" in names  # dose
    assert "RTDOSE_physical_dose.dcm" in names  # result
    assert "RTPLAN.dcm" in names  # stf with weights
    # pln selects the dose summation type
    assert pydicom.dcmread(out / "RTDOSE.dcm").DoseSummationType == "FRACTION"

    # weights are needed for the ion plan, and a missing pln gives the default summation type
    without = tmp_path / "dcm_without"
    with pytest.warns(UserWarning):
        save_data(
            file_name=without,
            format="dcm",
            **_objects(workspace, ["ct", "dose", "stf"]),
        )
    assert not (without / "RTPLAN.dcm").exists()
    assert pydicom.dcmread(without / "RTDOSE.dcm").DoseSummationType == "PLAN"


def test_single_warning_lists_dropped_objects(workspace, tmp_path):
    objects = _objects(workspace, ["ct", "pln", "stf", "dij"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save_data(file_name=tmp_path / "ws.npz", **objects)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert messages == ["The npz format cannot store pln, stf, dij; they were not saved."]
    assert os.path.normcase(caught[0].filename) == os.path.normcase(__file__)


def test_single_warning_without_file_name(workspace, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    objects = _objects(workspace, ["ct", "pln", "dij"])
    with pytest.warns(UserWarning, match="cannot store pln, dij") as record:
        written = save_data(format="npz", **objects)
    assert len(record) == 1
    assert written == ["ct.npz"]


def test_dicom_warns_instead_of_logging(workspace, tmp_path, caplog):
    with pytest.warns(UserWarning, match="dcm format cannot store dij"):
        save_data(file_name=tmp_path / "dcm", format="dcm", ct=workspace["ct"], dij=1)
    assert "ignores unsupported" not in caplog.text


@pytest.mark.parametrize("fmt, name", [("npz", "ws.npz"), ("mat", "ws.mat"), ("pickle", "ws.pkl")])
def test_no_warning_when_everything_is_supported(workspace, tmp_path, fmt, name):
    kinds = ["ct", "cst", "dose"] if fmt == "npz" else ["ct", "cst", "dose", "pln", "weights"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_data(file_name=tmp_path / name, format=fmt, **_objects(workspace, kinds))


def test_none_values_are_not_dropped(workspace, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_data(
            {"ct": workspace["ct"], "pln": None},
            file_name=tmp_path / "ws.npz",
            dij=None,
            stf=None,
        )
    with np.load(tmp_path / "ws.npz") as fh:
        assert json.loads(str(fh["meta"]))["contents"] == ["ct"]


def test_nothing_supported_raises(workspace, tmp_path):
    with pytest.raises(ValueError, match="Nothing to export"):
        save_data(file_name=tmp_path / "ws.npz", pln=workspace["pln"], dij=workspace["dij"])
    assert not (tmp_path / "ws.npz").exists()
    with pytest.raises(ValueError, match="Nothing to save"):
        save_data(file_name=tmp_path / "ws.npz", pln=None)


def test_custom_exporter_receives_extras_by_default(workspace, tmp_path, monkeypatch):
    from pyRadPlan.io import _factory
    from pyRadPlan.io.base import BaseExporter

    received = {}

    class CustomExporter(BaseExporter):
        format = "custom_test"
        extensions = (".customtest",)

        def save(self, *, ct=None, cst=None, dose=None, **extra):
            received.update(ct=ct, cst=cst, dose=dose, **extra)

    assert CustomExporter.supports == frozenset({ANY_OBJECT})
    monkeypatch.setitem(_factory.EXPORTERS, "custom_test", CustomExporter)
    monkeypatch.setitem(_factory._EXTENSION_MAP, ".customtest", "custom_test")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_data(
            file_name=tmp_path / "x.customtest",
            ct=workspace["ct"],
            pln=workspace["pln"],
            stf=workspace["stf"],
        )
    assert received["pln"] is workspace["pln"]
    assert received["stf"] is workspace["stf"]


def test_dict_calls_stay_backward_compatible(workspace, tmp_path):
    ct, cst = workspace["ct"], workspace["cst"]
    save_data({"ct": ct, "cst": cst}, file_name=tmp_path / "a.mat")
    save_data(file_name=tmp_path / "b.h5", ct=ct, cst=cst, result=workspace["result"])
    save_data(file_name=tmp_path / "c.pkl", format="pickle", ct=ct)
    assert {"ct", "cst"} <= set(load_data(tmp_path / "a.mat"))
    assert {"ct", "cst", "result"} <= set(load_data(tmp_path / "b.h5"))
    assert set(load_data(tmp_path / "c.pkl")) == {"ct"}
    with pytest.raises(TypeError):
        save_data([ct], file_name=tmp_path / "d.pkl")


def test_bundle_fields_and_to_dict(workspace):
    bundle = ExportBundle(ct=workspace["ct"], weights=workspace["weights"], extras={"note": "x"})
    assert set(bundle.to_dict()) == {"ct", "weights", "note"}
    assert bundle.cst is None and bundle.dose is None
    assert list(ExportBundle(extras={"a": None}).to_dict()) == []
    with pytest.raises(ValueError, match="ct"):
        ExportBundle(extras={"ct": 1})


def test_bundle_hdf5_roundtrip(workspace, tmp_path):
    bundle = ExportBundle(**workspace, extras={"saved_result_tags": ["saved"], "note": "x"})
    path = tmp_path / "bundle.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert bundle.save(path) == str(path)

    loaded = ExportBundle.load(path)
    assert loaded.ct == bundle.ct
    assert loaded.cst == bundle.cst
    assert loaded.pln == bundle.pln
    assert isinstance(loaded.dose, sitk.Image)
    np.testing.assert_array_equal(loaded.weights, bundle.weights)
    assert (loaded.dij.physical_dose[0] != bundle.dij.physical_dose[0]).nnz == 0
    assert loaded.result.stored_quantities == bundle.result.stored_quantities
    np.testing.assert_array_equal(loaded.result.fluence, bundle.result.fluence)
    assert loaded.extras == {"saved_result_tags": ["saved"], "note": "x"}


def test_bundle_save_warns_once_and_forwards_options(workspace, tmp_path):
    bundle = ExportBundle(ct=workspace["ct"], pln=workspace["pln"], stf=workspace["stf"])
    with pytest.warns(UserWarning, match="cannot store pln, stf") as record:
        bundle.save(tmp_path / "bundle.npz")
    assert len(record) == 1
    with np.load(tmp_path / "bundle.npz") as fh:
        assert json.loads(str(fh["meta"]))["contents"] == ["ct"]

    out = tmp_path / "dcm"
    bundle.save(out, format="dcm", anonymize=True, export_plan=False)
    with pytest.raises(FileExistsError):
        bundle.save(out, format="dcm")
    bundle.save(out, format="dcm", overwrite=True, export_plan=False)


def test_save_data_accepts_bundle(workspace, tmp_path):
    bundle = ExportBundle(ct=workspace["ct"], cst=workspace["cst"], extras={"note": "n"})
    path = tmp_path / "bundle.pkl"
    save_data(bundle, file_name=path)
    assert set(load_data(path)) == {"ct", "cst", "note"}

    save_data(bundle, file_name=tmp_path / "extra.pkl", result={"w": np.ones(2)})
    assert set(load_data(tmp_path / "extra.pkl")) == {"ct", "cst", "note", "result"}

    with pytest.warns(UserWarning, match="cannot store note"):
        save_data(bundle, file_name=tmp_path / "bundle.npz")
