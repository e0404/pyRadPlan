"""Tests for the lossless (JSON) serialization of the core data models."""

import json
import math

import numpy as np
import pytest
import pymatreader
import scipy.sparse as sp
import SimpleITK as sitk

from pyRadPlan import PhotonPlan, calc_dose_forward, generate_stf, load_tg119
from pyRadPlan.core import (
    ARRAY_TAG,
    PyRadPlanBaseModel,
    decode_input,
    decode_tree,
    encode_tree,
    json_leaf_decoder,
    json_leaf_encoder,
)
from pyRadPlan.cst import StructureSet, validate_cst, validate_voi
from pyRadPlan.ct import CT, DicomInfo, validate_ct
from pyRadPlan.dij import validate_dij
from pyRadPlan.machines import create_bld
from pyRadPlan.plan import IonPlan, validate_pln
from pyRadPlan.result import validate_result
from pyRadPlan.stf import validate_stf


def assert_same(a, b, path="obj"):
    """Strict deep comparison of models, arrays, sparse matrices and images (types included)."""
    numeric = (int, float)
    if isinstance(a, numeric) and isinstance(b, numeric) and not isinstance(a, bool):
        assert a == b or (math.isnan(a) and math.isnan(b)), path
        return
    assert type(a) is type(b), f"{path}: {type(a)} != {type(b)}"
    if isinstance(a, PyRadPlanBaseModel):
        fields = type(a).model_fields
        assert_same(
            {k: v for k, v in a.__dict__.items() if k in fields},
            {k: v for k, v in b.__dict__.items() if k in fields},
            path,
        )
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
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
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b), path
    else:
        assert a == b, path


def json_roundtrip(obj):
    return json.loads(json.dumps(obj, allow_nan=False))


# --------------------------------------------------------------------------
# Leaves
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(12, dtype=np.int32).reshape(3, 4),
        np.linspace(0, 1, 7, dtype=np.float32),
        np.array([[True, False], [False, True]]),
        np.array(3.5),
        np.zeros((0, 3)),
        np.array([0.1, 1 / 3, 1e-300, 1e300]),
        np.array([1 + 2j, -3j]),
        np.array(["a", "bc"]),
        np.arange(24, dtype=np.uint8).reshape(2, 3, 4)[:, ::2],
    ],
)
def test_ndarray_leaf_roundtrip(arr):
    leaf = json_roundtrip(json_leaf_encoder(arr))
    assert leaf[ARRAY_TAG] == "ndarray"
    assert leaf["shape"] == list(arr.shape)
    assert leaf["dtype"] == str(arr.dtype)
    assert_same(json_leaf_decoder(leaf), np.asarray(arr))


def test_leaf_format_is_flat_c_order():
    leaf = json_leaf_encoder(np.arange(6).reshape(2, 3))
    assert leaf["data"] == [0, 1, 2, 3, 4, 5]
    assert leaf["shape"] == [2, 3]


def test_non_finite_values_roundtrip_as_standard_json():
    arr = np.array([np.nan, np.inf, -np.inf, 1.5], dtype=np.float32)
    leaf = json_roundtrip(json_leaf_encoder(arr))
    assert leaf["data"] == ["NaN", "Infinity", "-Infinity", 1.5]
    out = json_leaf_decoder(leaf)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, arr)

    tree = {"a": float("nan"), "b": [float("inf"), -math.inf], "c": np.float64("nan")}
    decoded = decode_tree(json_roundtrip(encode_tree(tree)))
    assert math.isnan(decoded["a"]) and math.isnan(decoded["c"])
    assert decoded["b"] == [math.inf, -math.inf]


def test_object_array_with_none_and_sparse():
    mat = sp.random(5, 4, density=0.4, format="csr", random_state=1)
    arr = np.empty((2, 2), dtype=object)
    arr[0, 0] = mat
    arr[0, 1] = None
    arr[1, 0] = np.arange(3.0)
    arr[1, 1] = "text"
    leaf = json_roundtrip(json_leaf_encoder(arr))
    assert leaf["dtype"] == "object"
    assert leaf["data"][1] is None
    out = json_leaf_decoder(leaf)
    assert out.shape == (2, 2) and out.dtype == object
    assert (out[0, 0] != mat).nnz == 0
    assert out[0, 1] is None
    np.testing.assert_array_equal(out[1, 0], np.arange(3.0))
    assert out[1, 1] == "text"


@pytest.mark.parametrize("container", [sp.csc_matrix, sp.csr_matrix, sp.csc_array, sp.coo_array])
def test_sparse_leaf_roundtrip(container):
    dense = np.array([[0, 1.5, 0], [2, 0, np.nan], [0, 0, 0]], dtype=np.float32)
    mat = container(dense)
    leaf = json_roundtrip(json_leaf_encoder(mat))
    assert leaf["format"] == "csc" and set(leaf) >= {"data", "indices", "indptr", "shape"}
    out = json_leaf_decoder(leaf)
    assert isinstance(out, sp.csc_matrix if isinstance(mat, sp.spmatrix) else sp.csc_array)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out.toarray(), dense)


def test_sparse_empty():
    out = json_leaf_decoder(json_roundtrip(json_leaf_encoder(sp.csc_matrix((4, 3)))))
    assert out.shape == (4, 3) and out.nnz == 0


@pytest.mark.parametrize(
    "pixel_type", [sitk.sitkFloat32, sitk.sitkInt16, sitk.sitkUInt8, sitk.sitkFloat64]
)
def test_image_leaf_roundtrip(pixel_type):
    image = sitk.Cast(sitk.GetImageFromArray(np.random.rand(3, 4, 5) * 100), pixel_type)
    image.SetOrigin((1.0, -2.0, 3.5))
    image.SetSpacing((0.5, 1.0, 2.0))
    image.SetDirection((0, 1, 0, 1, 0, 0, 0, 0, -1))
    leaf = json_roundtrip(json_leaf_encoder(image))
    assert leaf[ARRAY_TAG] == "sitk.Image" and "components" not in leaf
    assert_same(json_leaf_decoder(leaf), image)


def test_vector_image_leaf_roundtrip():
    image = sitk.GetImageFromArray(np.random.rand(2, 3, 4, 3).astype(np.float32), isVector=True)
    image.SetSpacing((1, 2, 3))
    leaf = json_roundtrip(json_leaf_encoder(image))
    assert leaf["components"] == 3
    out = json_leaf_decoder(leaf)
    assert out.GetNumberOfComponentsPerPixel() == 3
    assert_same(out, image)


def test_4d_image():
    image = sitk.GetImageFromArray(np.random.rand(2, 3, 4, 5))
    assert_same(json_leaf_decoder(json_roundtrip(json_leaf_encoder(image))), image)


def test_torch_leaf_decodes_to_numpy():
    torch = pytest.importorskip("torch")
    arr = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = json_leaf_decoder(json_roundtrip(json_leaf_encoder(arr)))
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    np.testing.assert_array_equal(out, arr.numpy())


def test_unknown_tag_raises():
    with pytest.raises(ValueError):
        decode_tree({ARRAY_TAG: "mystery"})


def test_walker_is_pluggable_and_copies_containers():
    store = []

    def encoder(leaf):
        store.append(leaf)
        return {ARRAY_TAG: "stored", "index": len(store) - 1}

    def decoder(leaf):
        return store[leaf["index"]]

    arr = np.arange(3)
    tree = {"a": (arr, [1, np.int64(2)]), "b": {"c": "s", "d": None}}
    encoded = encode_tree(tree, encoder)
    assert encoded == {
        "a": [{ARRAY_TAG: "stored", "index": 0}, [1, 2]],
        "b": {"c": "s", "d": None},
    }
    assert type(encoded["a"][1][1]) is int
    assert decode_tree(encoded, decoder)["a"][0] is arr
    plain = {"x": [1, {"y": 2}]}
    assert decode_tree(plain) is plain


def test_result_dict_roundtrip():
    image = sitk.GetImageFromArray(np.random.rand(3, 4, 5).astype(np.float32))
    result = {
        "physical_dose": image,
        "physical_dose_beam": [image, image],
        "per_scenario": [[image], [None]],
        "w": np.array([0.0, 1.5, np.nan]),
        "n": np.int64(3),
        "label": "x",
    }
    out = decode_tree(json_roundtrip(encode_tree(result)))
    assert_same(out["physical_dose"], image)
    assert len(out["physical_dose_beam"]) == 2
    assert_same(out["physical_dose_beam"][1], image)
    assert out["per_scenario"][1] == [None]
    np.testing.assert_array_equal(out["w"], result["w"])
    assert out["n"] == 3 and out["label"] == "x"


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def _both(obj, factory, **kwargs):
    """Round trip through python dump, JSON dict and JSON string; compare strictly."""
    assert_same(factory(obj.model_dump(**kwargs)), obj)
    dumped = obj.model_dump(mode="json", **kwargs)
    assert_same(factory(json_roundtrip(dumped)), obj)
    text = obj.model_dump_json(**kwargs)

    def reject(constant):
        raise AssertionError(f"non-standard JSON constant {constant}")

    json.loads(text, parse_constant=reject)
    assert json.loads(text) == json_roundtrip(dumped)
    assert_same(factory(json.loads(text)), obj)
    assert_same(type(obj).model_validate_json(text), obj)
    if "by_alias" not in kwargs:
        aliased = json_roundtrip(obj.model_dump(mode="json", by_alias=True, **kwargs))
        assert_same(factory(aliased), obj)
    return dumped


@pytest.fixture(scope="module")
def tg119():
    return load_tg119()


@pytest.fixture(scope="module")
def small_case():
    shape = (6, 10, 12)
    image = sitk.GetImageFromArray(np.random.default_rng(0).uniform(-1000, 500, shape))
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetOrigin((-10.0, -15.0, -20.0))
    ct = CT(cube_hu=sitk.Cast(image, sitk.sitkFloat32))
    vois = []
    for name, voi_type, region in [
        ("target", "TARGET", (slice(2, 4), slice(3, 7), slice(4, 9))),
        ("organ", "OAR", (slice(0, 3), slice(0, 5), slice(0, 5))),
        ("body", "EXTERNAL", (slice(None), slice(None), slice(None))),
        ("helper", "HELPER", (slice(4, 6), slice(6, 9), slice(6, 10))),
    ]:
        mask = np.zeros(shape, dtype=np.uint8)
        mask[region] = 1
        vois.append(
            validate_voi(
                name=name, voi_type=voi_type, mask=sitk.GetImageFromArray(mask), ct_image=ct
            )
        )
    return ct, StructureSet(vois=vois, ct_image=ct)


@pytest.fixture(scope="module")
def proton_raw():
    return pymatreader.read_mat("test/data/protons_testData.mat")


def test_ct_with_dicom_info(small_case):
    ct, _ = small_case
    ct = ct.model_copy(deep=True)
    ct.dicom_info = DicomInfo(
        patient_id="P1",
        kvp=120.0,
        sop_instance_uids=["1.2", "1.3"],
        additional={"A": [1, 2]},
    )
    _both(ct, validate_ct)
    restored = validate_ct(json_roundtrip(ct.model_dump(mode="json")))
    assert isinstance(restored.dicom_info, DicomInfo)
    assert restored.dicom_info.patient_id == "P1"


def test_ct_by_alias(small_case):
    ct, _ = small_case
    dumped = ct.model_dump(mode="json", by_alias=True)
    assert "cubeHU" in dumped or "cubeHu" in dumped
    assert_same(validate_ct(json_roundtrip(dumped)), ct)


def test_cst_with_objectives(small_case):
    ct, cst = small_case
    cst = cst.model_copy(deep=True)
    objectives = [
        {"name": "Squared Deviation", "priority": 100, "d_ref": 50.0},
        {"name": "Squared Overdosing", "priority": 10, "d_ref": 30.0},
        {"name": "Mean Dose", "priority": 5, "d_ref": 12.0},
        {"name": "EUD", "priority": 2, "d_ref": 20.0},
    ]
    for voi, objective in zip(cst.vois, objectives):
        voi.objectives = [objective]
    assert len({type(v) for v in cst.vois}) > 1
    _both(cst, validate_cst)
    restored = validate_cst(json_roundtrip(cst.model_dump(mode="json")))
    assert isinstance(restored, StructureSet)
    assert [type(v) for v in restored.vois] == [type(v) for v in cst.vois]
    assert [type(o) for v in restored.vois for o in v.objectives] == [
        type(o) for v in cst.vois for o in v.objectives
    ]
    assert_same(restored.ct_image, cst.ct_image)


def test_pln_dump_options():
    pln = validate_pln({"radiation_mode": "protons", "machine": "Generic"})
    assert "machine" not in pln.model_dump(mode="json", exclude={"machine"})
    assert set(pln.model_dump(mode="json", include={"machine"})) == {"machine"}
    assert "radiationMode" in pln.model_dump(mode="json", by_alias=True)
    assert "radiation_mode" in pln.model_dump(mode="json", by_alias=False)


def test_pln_ion_and_photon():
    ion = validate_pln(
        {
            "radiation_mode": "protons",
            "machine": "Generic",
            "num_of_fractions": 5,
            "prop_stf": {"bixel_width": 3.0, "gantry_angles": [0, 90]},
            "scenarios": {"model": "wcScen", "combinations": "all", "shift_sd": (2.0, 2.0, 2.0)},
        }
    )
    assert isinstance(ion, IonPlan)
    _both(ion, validate_pln)
    photon = validate_pln({"radiation_mode": "photons", "machine": "Generic"})
    assert isinstance(photon, PhotonPlan)
    _both(photon, validate_pln)


def test_ion_stf_with_range_shifter(proton_raw):
    stf = validate_stf(proton_raw["stf"])
    _both(stf, validate_stf)
    spot = stf.beams[0].rays[0].beamlets[0]
    spot.range_shifter = {"id": 3, "eq_thickness": 12.5, "source_rashi_distance": 400.0}
    _both(stf, validate_stf)
    restored = validate_stf(json_roundtrip(stf.model_dump(mode="json")))
    restored_spot = restored.beams[0].rays[0].beamlets[0]
    assert restored_spot.range_shifter.eq_thickness == 12.5
    assert type(restored_spot) is type(spot)
    assert restored_spot.max_mu == spot.max_mu


def _beamlets(stf):
    return [bl for beam in stf.beams for ray in beam.rays for bl in ray.beamlets]


@pytest.mark.parametrize("n_blds,expected", [(1, "FieldShapeAsBLD"), (2, "FieldShapeComposite")])
def test_field_based_photon_stf(tg119, n_blds, expected):
    ct, cst = tg119
    blds = [
        create_bld(
            {
                "device_type": "MLC",
                "device_orientation": "X",
                "leaf_position_boundaries": np.arange(-20.0, 20.0, 5.0),
                "leaf_positions": [[-10, 10], [-15, 20], [-5, 25], [0, 30]] * 2,
                "leaf_width": 5.0,
                "leaf_leakage": 0.0,
            }
        ),
        create_bld(
            {
                "device_type": "JAW",
                "device_orientation": "Y",
                "positions": [-20, 12],
                "field_width": 120,
            }
        ),
    ][::-1][:n_blds]
    pln = PhotonPlan(machine="Generic", num_of_fractions=3)
    pln.prop_stf = {
        "gantry_angles": [0, 90],
        "couch_angles": [0, 0],
        "generator": "photonSingleBixel",
        "field_based": True,
        "blds": blds,
        "resolution": 2.0,
        "energy": 6.0,
    }
    stf = generate_stf(ct, cst, pln)
    assert {type(b).__name__ for b in _beamlets(stf)} == {expected}
    _both(stf, validate_stf)
    restored = validate_pln(json_roundtrip(pln.model_dump(mode="json")))
    restored_blds = restored.prop_stf.pop("blds")
    assert_same([create_bld(b) for b in restored_blds], pln.prop_stf["blds"])
    pln = pln.model_copy(deep=True)
    pln.prop_stf.pop("blds")
    assert_same(restored, pln)


def test_generated_stfs_keep_their_beamlet_types(tg119):
    from pyRadPlan.stf import IonSpot, PhotonBixel

    ct, cst = tg119
    for pln, expected in [
        (IonPlan(radiation_mode="protons", machine="Generic"), IonSpot),
        (PhotonPlan(machine="Generic"), PhotonBixel),
    ]:
        pln.prop_stf = {"gantry_angles": [0, 90], "couch_angles": [0, 0], "bixel_width": 20}
        stf = generate_stf(ct, cst, pln)
        assert {type(b) for b in _beamlets(stf)} == {expected}
        _both(stf, validate_stf)
        restored = validate_stf(json_roundtrip(stf.model_dump(mode="json")))
        assert restored == stf
        assert validate_stf(stf.model_dump()) == stf


def test_matrad_stf_beamlets_follow_the_radiation_mode(proton_raw):
    from pyRadPlan.stf import IonSpot

    stf = validate_stf(proton_raw["stf"])
    assert {type(b) for b in _beamlets(stf)} == {IonSpot}
    assert validate_stf(json_roundtrip(stf.model_dump(mode="json"))) == stf


def test_validate_decodes_tagged_leaves_next_to_native_ones(small_case):
    ct, cst = small_case
    vois = json_roundtrip(cst.model_dump(mode="json"))["vois"]
    restored = validate_cst({"vois": vois, "ct_image": ct})
    assert all(isinstance(v.mask, sitk.Image) for v in restored.vois)
    assert_same(restored.vois[0].mask, cst.vois[0].mask)

    mixed = {"native": np.zeros(2), "tagged": json_roundtrip(encode_tree({"a": np.arange(3)}))}
    out = decode_input(mixed)
    assert isinstance(out["tagged"]["a"], np.ndarray)
    assert out["native"] is mixed["native"]
    assert decode_input([np.zeros(2), mixed["tagged"]])[1]["a"].tolist() == [0, 1, 2]


def test_ion_dij(proton_raw):
    dij = validate_dij(proton_raw["dij"])
    _both(dij, validate_dij)
    restored = validate_dij(json_roundtrip(dij.model_dump(mode="json")))
    assert sp.issparse(restored.physical_dose.flat[0])
    assert restored.physical_dose.dtype == object


def test_forward_result_roundtrip(tg119):
    ct, cst = tg119
    pln = PhotonPlan(machine="Generic", num_of_fractions=1)
    pln.prop_stf = {"gantry_angles": [0], "couch_angles": [0], "bixel_width": 20}
    pln.prop_dose_calc["dose_grid"] = {"resolution": {"x": 8.0, "y": 8.0, "z": 8.0}}
    stf = generate_stf(ct, cst, pln)
    result = calc_dose_forward(ct, cst, stf, pln)
    out = validate_result(json.loads(result.model_dump_json()))
    assert out == result
    assert out.stored_beam_quantities == result.stored_beam_quantities
    for key, value in result.quantities.items():
        assert_same(out[key].values, value.values, key)


def test_python_dump_unchanged(proton_raw):
    dij = validate_dij(proton_raw["dij"])
    dumped = dij.model_dump()
    assert isinstance(dumped["physical_dose"], np.ndarray)
    assert isinstance(dij.to_matrad()["physicalDose"].flat[0], sp.csc_matrix)
