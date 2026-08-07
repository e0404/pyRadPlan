"""Tests for the NumPy ``.npz`` import/export backend."""

import numpy as np
import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.io import load_data, save_data, load_tg119, NpzHandler
from pyRadPlan.io._factory import detect_format, is_container_format


def _ct_arrays_close(a: CT, b: CT, atol=1e-6) -> bool:
    return (
        a.cube_hu.GetSize() == b.cube_hu.GetSize()
        and np.allclose(a.cube_hu.GetSpacing(), b.cube_hu.GetSpacing())
        and np.allclose(a.cube_hu.GetOrigin(), b.cube_hu.GetOrigin())
        and np.allclose(a.cube_hu.GetDirection(), b.cube_hu.GetDirection())
        and np.allclose(
            sitk.GetArrayFromImage(a.cube_hu), sitk.GetArrayFromImage(b.cube_hu), atol=atol
        )
    )


def _voxel_sets(cst: StructureSet) -> dict:
    return {v.name.lower(): set(v.indices_numpy.tolist()) for v in cst.vois}


def _dummy_dose(ct: CT) -> sitk.Image:
    arr = np.random.default_rng(0).random(
        sitk.GetArrayFromImage(ct.cube_hu).shape, dtype=np.float32
    )
    dose = sitk.GetImageFromArray(arr)
    dose.CopyInformation(ct.cube_hu)
    return dose


def test_factory_knows_npz(tmp_path):
    f = tmp_path / "x.npz"
    np.savez(f, a=np.zeros(1))
    assert detect_format(f) == "npz"
    assert is_container_format("npz") is True


def test_npz_roundtrip_ct_and_cst(tmp_path):
    ct, cst = load_tg119()
    out = tmp_path / "patient.npz"

    save_data(ct=ct, cst=cst, file_name=str(out))
    assert out.exists()

    data = load_data(out)
    ct2, cst2 = data["ct"], data["cst"]

    assert isinstance(ct2, CT)
    assert isinstance(cst2, StructureSet)
    assert _ct_arrays_close(ct, ct2)

    before, after = _voxel_sets(cst), _voxel_sets(cst2)
    assert set(before) == set(after)
    for name, voxels in before.items():
        assert voxels == after[name], f"VOI '{name}' changed after roundtrip"


def test_npz_roundtrip_dose(tmp_path):
    ct, _ = load_tg119()
    dose = _dummy_dose(ct)
    out = tmp_path / "withdose.npz"

    save_data(ct=ct, dose=dose, file_name=str(out))
    data = load_data(out)

    assert isinstance(data["dose"], sitk.Image)
    assert data["dose"].GetSize() == dose.GetSize()
    assert np.allclose(data["dose"].GetSpacing(), dose.GetSpacing())
    assert np.allclose(data["dose"].GetOrigin(), dose.GetOrigin())
    assert np.allclose(
        sitk.GetArrayFromImage(data["dose"]), sitk.GetArrayFromImage(dose), atol=1e-6
    )


def test_npz_cst_only_is_self_contained(tmp_path):
    # Saving a cst alone must still produce a loadable file (ct pulled from cst.ct_image).
    ct, cst = load_tg119()
    out = tmp_path / "cst_only.npz"

    save_data(cst=cst, file_name=str(out))
    data = load_data(out)

    assert isinstance(data["ct"], CT)
    assert _voxel_sets(data["cst"]) == _voxel_sets(cst)


def test_npz_handler_direct(tmp_path):
    ct, cst = load_tg119()
    out = tmp_path / "viahandler.npz"

    NpzHandler(out).save(ct=ct, cst=cst)
    handler = NpzHandler(out)
    assert _ct_arrays_close(ct, handler.load_ct())
    assert _voxel_sets(handler.load_cst()) == _voxel_sets(cst)
