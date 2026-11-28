"""Tests for the HDF5 workspace import/export backend."""

import math
import warnings

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import SimpleITK as sitk

from pyRadPlan import IonPlan, PhotonPlan, calc_dose_forward, calc_dose_influence, generate_stf
from pyRadPlan import load_tg119
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.ct import CT, DicomInfo
from pyRadPlan.io import Hdf5Handler, load_data, save_data
from pyRadPlan.io._factory import detect_format, is_container_format
from pyRadPlan.io.hdf5._codec import COMPRESSION_THRESHOLD, INLINE_THRESHOLD
from pyRadPlan.machines import create_bld
from pyRadPlan.result import BeamSequence, validate_result


def assert_same(a, b, path="obj"):
    """Strict deep comparison of models, arrays, sparse matrices and images (types included)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        assert a == b or (math.isnan(a) and math.isnan(b)), path
        return
    assert type(a) is type(b), f"{path}: {type(a)} != {type(b)}"
    if isinstance(a, PyRadPlanBaseModel):
        fields = type(a).model_fields
        assert_same(
            {k: v for k, v in sorted(a.__dict__.items()) if k in fields},
            {k: v for k, v in sorted(b.__dict__.items()) if k in fields},
            path,
        )
    elif isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), path
        for key in a:
            assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, sitk.Image):
        assert a.GetPixelIDTypeAsString() == b.GetPixelIDTypeAsString(), path
        assert a.GetSize() == b.GetSize(), path
        assert a.GetOrigin() == b.GetOrigin(), path
        assert a.GetSpacing() == b.GetSpacing(), path
        assert a.GetDirection() == b.GetDirection(), path
        np.testing.assert_array_equal(sitk.GetArrayFromImage(a), sitk.GetArrayFromImage(b), path)
    elif sp.issparse(a):
        assert a.shape == b.shape and a.dtype == b.dtype, path
        assert (a != b).nnz == 0, path
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        if a.dtype == object:
            for i, (x, y) in enumerate(zip(a.flat, b.flat)):
                assert_same(x, y, f"{path}[{i}]")
        else:
            np.testing.assert_array_equal(a, b, path)
    else:
        assert a == b, path


@pytest.fixture(scope="module")
def tg119():
    return load_tg119()


@pytest.fixture(scope="module")
def ion_workspace(tg119):
    ct, cst = tg119
    ct = ct.model_copy(deep=True)
    ct.dicom_info = DicomInfo(patient_id="P1", kvp=120.0, sop_instance_uids=["1.2", "1.3"])
    cst = cst.model_copy(deep=True)
    cst.ct_image = ct
    for voi in cst.vois:
        if voi.voi_type == "TARGET":
            voi.objectives = [{"name": "Squared Deviation", "priority": 100, "d_ref": 60.0}]
        elif voi.voi_type == "OAR":
            voi.objectives = [{"name": "Squared Overdosing", "priority": 10, "d_ref": 20.0}]
    pln = IonPlan(radiation_mode="protons", machine="Generic", num_of_fractions=30)
    pln.prop_stf = {"gantry_angles": [0, 90], "couch_angles": [0, 0], "bixel_width": 15}
    pln.prop_dose_calc["dose_grid"] = {"resolution": {"x": 8.0, "y": 8.0, "z": 8.0}}
    stf = generate_stf(ct, cst, pln)
    dij = calc_dose_influence(ct, cst, stf, pln)
    weights = np.ones(dij.total_num_of_bixels)
    result = calc_dose_forward(ct, cst, stf, pln, weights)
    result.fluence = weights
    result.add_scenario_quantity("physical_dose", [result.quantities["physical_dose"], None])
    return {
        "ct": ct,
        "cst": cst,
        "pln": pln,
        "stf": stf,
        "dij": dij,
        "result": result,
        "weights": weights,
        "saved_result_tags": ["a", "b"],
    }


def test_named_results_roundtrip(tmp_path, ion_workspace):
    result = ion_workspace["result"]
    measured = validate_result({"physical_dose": result.physical_dose.values * 2.0})
    out = tmp_path / "results.h5"
    save_data(
        file_name=str(out),
        ct=ion_workspace["ct"],
        results={"current": result, "measured": measured},
        active_result="current",
    )
    with h5py.File(out) as fh:
        assert fh["results"].attrs["kind"] == "results"
    data = load_data(out)
    assert list(data["results"]) == ["current", "measured"]
    assert data["results"]["current"] == result
    assert data["results"]["measured"] == measured
    assert data["active_result"] == "current"


def test_factory_knows_hdf5(tmp_path):
    for ext in (".h5", ".hdf5"):
        f = tmp_path / f"x{ext}"
        f.write_bytes(b"")
        assert detect_format(f) == "hdf5"
    assert is_container_format("hdf5") is True


def test_full_ion_workspace_roundtrip(tmp_path, ion_workspace):
    out = tmp_path / "ws.h5"
    save_data(file_name=str(out), **ion_workspace)

    with h5py.File(out) as fh:
        assert fh.attrs["pyradplan_format"] == "workspace"
        assert fh.attrs["format_version"] == 1
        assert fh["dij"].attrs["kind"] == "dij"
        assert fh["result"].attrs["kind"] == "result"
        assert fh["dij/tree"].dtype.kind == "O"
        assert fh["cst"].attrs["ct_ref"] == "ct"

    data = load_data(out)
    assert set(data) == set(ion_workspace)
    for key, value in ion_workspace.items():
        assert_same(data[key], value, key)
    assert data["result"].scenario_quantities["physical_dose"][1] is None
    assert sp.issparse(data["dij"].physical_dose.flat[0])
    assert isinstance(data["ct"].dicom_info, DicomInfo)
    assert data["cst"].ct_image is data["ct"]


def test_cst_on_other_ct_keeps_its_own_ct(tmp_path, ion_workspace):
    out = tmp_path / "ws.h5"
    ct, cst = ion_workspace["ct"], ion_workspace["cst"]
    other = ct.model_copy(update={"dicom_info": None})
    save_data(file_name=str(out), ct=other, cst=cst)
    with h5py.File(out) as fh:
        assert "ct_ref" not in fh["cst"].attrs
    data = load_data(out)
    assert_same(data["cst"], cst)
    assert data["cst"].ct_image is not data["ct"]
    assert_same(Hdf5Handler(out).load_cst(), cst)


def test_handler_and_single_loaders(tmp_path, ion_workspace):
    out = tmp_path / "ws.hdf5"
    Hdf5Handler(out).save(ct=ion_workspace["ct"], cst=ion_workspace["cst"])
    handler = Hdf5Handler(out)
    assert_same(handler.load_ct(), ion_workspace["ct"])
    assert_same(handler.load_cst(), ion_workspace["cst"])
    assert handler.load_dose() is None
    assert list(handler.load_data()) == ["ct", "cst"]


def test_dose_roundtrip_and_overwrite(tmp_path, tg119):
    ct, _ = tg119
    arr = np.random.default_rng(0).random(sitk.GetArrayFromImage(ct.cube_hu).shape, np.float32)
    dose = sitk.GetImageFromArray(arr)
    dose.CopyInformation(ct.cube_hu)
    out = tmp_path / "dose.h5"
    save_data(ct=ct, cst=None, dose=dose, file_name=str(out))
    save_data(dose=dose, file_name=str(out))
    data = load_data(out)
    assert list(data) == ["dose"]
    assert_same(data["dose"], dose)


def test_photon_field_based_stf(tmp_path, tg119):
    ct, cst = tg119
    jaw = create_bld(
        {
            "device_type": "JAW",
            "device_orientation": "Y",
            "positions": [-20, 12],
            "field_width": 120,
        }
    )
    beam = BeamSequence.from_shapes(None, [1.0], jaws=[jaw])
    pln = PhotonPlan(machine="Generic", num_of_fractions=3)
    pln.prop_stf = {
        "gantry_angles": [0, 90],
        "couch_angles": [0, 0],
        "generator": "photonSingleBixel",
        "beam_sequences": [beam, beam],
        "resolution": 2.0,
        "energy": 6.0,
    }
    stf = generate_stf(ct, cst, pln)
    out = tmp_path / "photon.h5"
    save_data(file_name=str(out), ct=ct, stf=stf)
    assert_same(load_data(out)["stf"], stf)

    save_data(file_name=str(out), pln=pln)
    assert_same(load_data(out)["pln"], pln)


def test_compression_applied_to_large_datasets_only(tmp_path):
    big = np.zeros((128, 128), dtype=np.float32)
    assert big.nbytes >= COMPRESSION_THRESHOLD
    small = np.arange(INLINE_THRESHOLD // 8 + 1.0)
    assert small.nbytes < COMPRESSION_THRESHOLD
    out = tmp_path / "c.h5"
    save_data(file_name=str(out), tree={"big": big, "small": small})
    with h5py.File(out) as fh:
        assert fh["tree/arrays/0"].compression == "gzip"
        assert fh["tree/arrays/1"].compression is None
    assert out.stat().st_size < big.nbytes // 2
    data = load_data(out)["tree"]
    assert_same(data["big"], big)
    assert_same(data["small"], small)


def test_non_finite_empty_and_exotic_arrays(tmp_path):
    obj = np.empty(2, dtype=object)
    obj[0] = sp.random(4, 3, density=0.5, format="csc", random_state=0)
    obj[1] = None
    tree = {
        "nan": float("nan"),
        "inf": [float("inf"), -math.inf],
        "arr": np.array([np.nan, np.inf, -np.inf, 1.5], dtype=np.float32),
        "empty": np.zeros((0, 3)),
        "scalar": np.array(2.5),
        "complex": np.array([1 + 2j, 3j]),
        "bool": np.array([True, False]),
        "int": np.arange(6, dtype=np.uint16).reshape(2, 3),
        "str": np.array(["a", "bc"]),
        "obj": obj,
        "sparse_array": sp.csc_array(np.eye(3, dtype=np.float32)),
        "vector": sitk.GetImageFromArray(np.random.rand(2, 3, 4, 3).astype(np.float32), True),
        "int16": sitk.Cast(sitk.GetImageFromArray(np.arange(24).reshape(2, 3, 4)), sitk.sitkInt16),
        "nested": {"x": [1, "y", None]},
    }
    out = tmp_path / "e.h5"
    save_data(file_name=str(out), tree=tree)
    loaded = load_data(out)["tree"]
    assert math.isnan(loaded["nan"])
    assert loaded["inf"] == [math.inf, -math.inf]
    for key in tree:
        if key not in ("nan", "inf"):
            assert_same(loaded[key], tree[key], key)


def test_unknown_version_and_foreign_files_rejected(tmp_path):
    out = tmp_path / "v.h5"
    save_data(file_name=str(out), note=["x"])
    with h5py.File(out, "r+") as fh:
        fh.attrs["format_version"] = 99
    with pytest.raises(ValueError, match="unsupported format version"):
        load_data(out)

    foreign = tmp_path / "f.h5"
    with h5py.File(foreign, "w") as fh:
        fh.create_dataset("x", data=1)
    with pytest.raises(ValueError, match="not a pyRadPlan"):
        load_data(foreign)

    garbage = tmp_path / "g.h5"
    garbage.write_bytes(b"not hdf5")
    with pytest.raises(ValueError, match="not a pyRadPlan"):
        load_data(garbage)


def test_unsupported_objects_are_skipped_with_warning(tmp_path):
    class Unsupported:
        pass

    out = tmp_path / "u.h5"
    with pytest.warns(UserWarning) as record:
        save_data(
            file_name=str(out),
            keep=[1, 2],
            model=DicomInfo(patient_id="x"),
            thing=Unsupported(),
            nested={"a": Unsupported()},
        )
    messages = " ".join(str(w.message) for w in record)
    assert "'model'" in messages and "'thing'" in messages and "'nested'" in messages
    assert list(load_data(out)) == ["keep"]


def _datasets(path):
    names = []

    def collect(name, obj):
        if isinstance(obj, h5py.Dataset) and "/arrays/" in name:
            names.append(name)

    with h5py.File(path) as fh:
        fh.visititems(collect)
    return names


def test_small_arrays_are_inline_and_large_ones_datasets(tmp_path):
    small = np.arange(10, dtype=np.float32)
    odd = np.array([[1.5, np.nan], [np.inf, -np.inf]])
    flags = np.array([True, False, True])
    cplx = np.array([1 + 2j, 3 - 1j])
    large = np.arange(INLINE_THRESHOLD // 8 + 1, dtype=np.float64)
    out = tmp_path / "i.h5"
    save_data(
        file_name=str(out),
        tree={"small": small, "odd": odd, "flags": flags, "cplx": cplx, "large": large},
    )
    assert _datasets(out) == ["tree/arrays/0"]
    with h5py.File(out) as fh:
        text = fh["tree/tree"][()].decode()
    assert '"data"' in text and text.count('"dataset"') == 1

    loaded = load_data(out)["tree"]
    for key, ref in [("small", small), ("odd", odd), ("flags", flags), ("cplx", cplx)]:
        assert loaded[key].dtype == ref.dtype and loaded[key].shape == ref.shape
        np.testing.assert_array_equal(loaded[key], ref)
    np.testing.assert_array_equal(loaded["large"], large)


def test_files_without_inline_arrays_still_load(tmp_path, ion_workspace, monkeypatch):
    import pyRadPlan.io.hdf5._codec as codec

    monkeypatch.setattr(codec, "INLINE_THRESHOLD", 0)
    out = tmp_path / "old.h5"
    save_data(file_name=str(out), **ion_workspace)
    assert len(_datasets(out)) > 100
    monkeypatch.undo()

    data = load_data(out)
    for key, value in ion_workspace.items():
        assert_same(data[key], value, key)

    inline = tmp_path / "new.h5"
    save_data(file_name=str(inline), **ion_workspace)
    assert len(_datasets(inline)) < len(_datasets(out))
    with h5py.File(inline) as fh:
        assert fh.attrs["format_version"] == 1
    for key, value in ion_workspace.items():
        assert_same(load_data(inline)[key], value, key)


def test_strict_exporter_raises_instead_of_skipping(tmp_path):
    class Unsupported:
        pass

    out = tmp_path / "s.h5"
    with pytest.raises(TypeError, match="'thing'"):
        save_data(
            file_name=str(out),
            format="hdf5",
            exporter_options={"strict": True},
            keep=[1, 2],
            thing=Unsupported(),
        )
    with pytest.raises(ValueError, match="'model'"):
        save_data(
            file_name=str(out),
            format="hdf5",
            exporter_options={"strict": True},
            model=DicomInfo(patient_id="x"),
        )
    assert not out.exists() and not (tmp_path / "s.h5.tmp").exists()

    save_data(file_name=str(out), exporter_options={"strict": True}, keep=[1, 2], other={"a": 1})
    assert list(load_data(out)) == ["keep", "other"]


def test_load_cst_uses_the_passed_ct(tmp_path, ion_workspace):
    from pyRadPlan.io import load_patient

    out = tmp_path / "ws.h5"
    save_data(file_name=str(out), ct=ion_workspace["ct"], cst=ion_workspace["cst"])
    ct, cst = load_patient(out)
    assert cst.ct_image is ct
    assert_same(cst, ion_workspace["cst"])

    other = ion_workspace["ct"].model_copy(deep=True)
    assert Hdf5Handler(out).load_cst(other).ct_image is other


def test_nothing_to_export_raises(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            Hdf5Handler(tmp_path / "n.h5").save()


def test_ct_type_preserved(tmp_path, tg119):
    ct, _ = tg119
    out = tmp_path / "ct.h5"
    save_data(ct=ct, file_name=str(out))
    assert isinstance(load_data(out)["ct"], CT)
