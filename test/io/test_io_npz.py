"""Tests for the NumPy ``.npz`` import/export backend."""

import numpy as np
import SimpleITK as sitk

from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst, validate_voi
from pyRadPlan.io import load_data, save_data, load_tg119, NpzHandler
from pyRadPlan.io._factory import detect_format, is_container_format
from pyRadPlan.biology.tissues import LEMTissueProperties, MKMTissueProperties, TissueProperties


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


def test_npz_legacy_visible_color_key_is_read(tmp_path):
    """Archives written before the ``visible_color`` -> ``color`` rename keep their colors."""
    import json

    ct, cst = load_tg119()
    cst.vois[0].color = (7, 8, 9)
    out = tmp_path / "patient.npz"
    save_data(ct=ct, cst=cst, file_name=str(out))

    # Rewrite the archive with the pre-rename metadata key.
    with np.load(out, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    meta = json.loads(arrays.pop("meta").item())
    for vmeta in meta["cst"]["vois"]:
        vmeta["visible_color"] = vmeta.pop("color")
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)

    cst2 = load_data(out)["cst"]
    assert tuple(cst2.vois[0].color) == (7, 8, 9)
    for before, after in zip(cst.vois, cst2.vois):
        assert tuple(after.color) == tuple(before.color)


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


def _cst_with_tissues() -> StructureSet:
    ct = validate_ct(
        cube_hu=np.zeros((4, 5, 6), dtype=np.float32), resolution={"x": 2, "y": 2, "z": 3}
    )
    tissues = {
        "lem": TissueProperties.from_preset("HSG", "LEM_II"),
        "mkm": TissueProperties.from_preset("CHO", "MKM"),
        "nan": LEMTissueProperties(alpha_x=0.2, beta_x=0.04, version="IV"),
        "reference": TissueProperties(),
        "override": MKMTissueProperties(r_domain=0.2, alpha_0=0.228, beta_0=0.02, alpha_x=0.3),
    }
    vois = []
    for i, (name, tissue) in enumerate(tissues.items()):
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[:, i] = 1
        vois.append(
            validate_voi(
                name=name, voi_type="OAR", mask=mask, ct_image=ct, tissue_properties=tissue
            )
        )
    return validate_cst(vois, ct)


def test_npz_roundtrip_tissue_properties(tmp_path):
    cst = _cst_with_tissues()
    out = tmp_path / "tissues.npz"
    save_data(cst=cst, file_name=str(out))

    cst2 = load_data(out)["cst"]
    for before, after in zip(cst.vois, cst2.vois):
        assert type(after.tissue_properties) is type(before.tissue_properties)
        assert after.tissue_properties == before.tissue_properties
        assert (
            after.tissue_properties.reference_sources == before.tissue_properties.reference_sources
        )
        assert (after.alpha_x, after.beta_x) == (before.alpha_x, before.beta_x)
    sources = [voi.tissue_properties.reference_source for voi in cst2.vois]
    assert sources == ["published", "published", "explicit", "default", "estimated"]
    assert cst2.vois[4].tissue_properties.reference_sources == {
        "alpha_x": "explicit",
        "beta_x": "estimated",
    }
    preset_sources = [voi.tissue_properties.cell_line_source for voi in cst2.vois]
    assert preset_sources == [
        cst.vois[0].tissue_properties.cell_line_source,
        cst.vois[1].tissue_properties.cell_line_source,
        None,
        None,
        None,
    ]
    assert "Kase et al." in preset_sources[1]


def test_npz_without_tissue_properties_still_loads(tmp_path):
    """Archives written before tissue properties were stored keep their alpha/beta."""
    import json

    cst = _cst_with_tissues()
    out = tmp_path / "legacy.npz"
    save_data(cst=cst, file_name=str(out))

    with np.load(out, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    meta = json.loads(arrays.pop("meta").item())
    for vmeta in meta["cst"]["vois"]:
        del vmeta["tissue_properties"]
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)

    cst2 = load_data(out)["cst"]
    for before, after in zip(cst.vois, cst2.vois):
        assert type(after.tissue_properties) is TissueProperties
        assert after.tissue_properties.alpha_x == before.alpha_x
        assert after.tissue_properties.beta_x == before.beta_x
