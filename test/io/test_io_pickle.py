"""Tests for the pickle import/export backend."""

import numpy as np
import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.io import load_data, save_data, load_tg119, PickleHandler
from pyRadPlan.io._factory import detect_format, is_container_format


def _ct_close(a: CT, b: CT, atol=1e-6) -> bool:
    return (
        a.cube_hu.GetSize() == b.cube_hu.GetSize()
        and np.allclose(a.cube_hu.GetSpacing(), b.cube_hu.GetSpacing())
        and np.allclose(a.cube_hu.GetOrigin(), b.cube_hu.GetOrigin())
        and np.allclose(
            sitk.GetArrayFromImage(a.cube_hu), sitk.GetArrayFromImage(b.cube_hu), atol=atol
        )
    )


def _voxel_sets(cst: StructureSet) -> dict:
    return {v.name.lower(): set(v.indices_numpy.tolist()) for v in cst.vois}


def test_factory_knows_pickle(tmp_path):
    f = tmp_path / "x.pkl"
    f.write_bytes(b"")
    assert detect_format(f) == "pickle"
    assert is_container_format("pickle") is True


def test_pickle_roundtrip_ct_and_cst(tmp_path):
    ct, cst = load_tg119()
    out = tmp_path / "patient.pkl"

    save_data(ct=ct, cst=cst, file_name=str(out))
    assert out.exists()

    data = load_data(out)
    ct2, cst2 = data["ct"], data["cst"]
    assert isinstance(ct2, CT)
    assert isinstance(cst2, StructureSet)
    assert _ct_close(ct, ct2)

    # Full fidelity: per-VOI voxels preserved exactly (overlaps included).
    assert _voxel_sets(cst) == _voxel_sets(cst2)


def test_pickle_roundtrip_dose(tmp_path):
    ct, _ = load_tg119()
    arr = np.random.default_rng(0).random(
        sitk.GetArrayFromImage(ct.cube_hu).shape, dtype=np.float32
    )
    dose = sitk.GetImageFromArray(arr)
    dose.CopyInformation(ct.cube_hu)
    out = tmp_path / "withdose.pkl"

    save_data(ct=ct, dose=dose, file_name=str(out))
    data = load_data(out)
    assert isinstance(data["dose"], sitk.Image)
    assert np.array_equal(sitk.GetArrayFromImage(data["dose"]), arr)


def test_pickle_passes_through_extras(tmp_path):
    ct, _ = load_tg119()
    out = tmp_path / "extra.pkl"
    save_data({"ct": ct, "note": "hello"}, file_name=str(out))

    data = load_data(out)
    assert data["note"] == "hello"
    assert isinstance(data["ct"], CT)


def test_pickle_handler_direct(tmp_path):
    ct, cst = load_tg119()
    out = tmp_path / "viahandler.pkl"
    PickleHandler(out).save(ct=ct, cst=cst)

    handler = PickleHandler(out)
    assert _ct_close(ct, handler.load_ct())
    assert _voxel_sets(handler.load_cst()) == _voxel_sets(cst)
