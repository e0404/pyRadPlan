"""Tests for the SimpleITK-based backends (NIfTI, NRRD, MetaImage)."""

import numpy as np
import SimpleITK as sitk
import pytest

from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_voi, validate_cst
from pyRadPlan.io import load_data, load_patient, save_data, NiftiHandler
from pyRadPlan.io._factory import detect_format, is_container_format
from pyRadPlan.biology.tissues import MKMTissueProperties, TissueProperties

FORMATS = ["nifti", "nrrd", "meta"]


# --------------------------------------------------------------------------
# Helpers / synthetic data
# --------------------------------------------------------------------------


def _make_ct() -> CT:
    arr = np.arange(10 * 12 * 8, dtype=np.float32).reshape(10, 12, 8)  # (z, y, x)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 3.0, 4.0))
    img.SetOrigin((-5.0, -10.0, -2.0))
    return validate_ct(cube_hu=img)


def _make_cst(ct: CT) -> StructureSet:
    shape = sitk.GetArrayFromImage(ct.cube_hu).shape
    m_target = np.zeros(shape, dtype=np.uint8)
    m_target[1:3, 1:4, 1:3] = 1
    m_core = np.zeros(shape, dtype=np.uint8)
    m_core[5:7, 6:9, 4:6] = 1  # disjoint from target -> lossless label-map round-trip

    target = validate_voi(
        name="Target", voi_type="TARGET", mask=m_target, ct_image=ct, alpha_x=0.2, beta_x=0.03
    )
    core = validate_voi(name="Core", voi_type="OAR", mask=m_core, ct_image=ct)
    return validate_cst([target, core], ct)


def _dummy_dose(ct: CT) -> sitk.Image:
    arr = np.random.default_rng(0).random(
        sitk.GetArrayFromImage(ct.cube_hu).shape, dtype=np.float32
    )
    dose = sitk.GetImageFromArray(arr)
    dose.CopyInformation(ct.cube_hu)
    return dose


def _ct_close(a: CT, b: CT) -> bool:
    return (
        a.cube_hu.GetSize() == b.cube_hu.GetSize()
        and np.allclose(a.cube_hu.GetSpacing(), b.cube_hu.GetSpacing())
        and np.allclose(a.cube_hu.GetOrigin(), b.cube_hu.GetOrigin())
        and np.allclose(a.cube_hu.GetDirection(), b.cube_hu.GetDirection())
        and np.allclose(sitk.GetArrayFromImage(a.cube_hu), sitk.GetArrayFromImage(b.cube_hu))
    )


def _voxel_sets(cst: StructureSet) -> dict:
    return {v.name.lower(): set(v.indices_numpy.tolist()) for v in cst.vois}


# --------------------------------------------------------------------------
# Import of the bundled test data (same patient as the DICOM/mat reference)
# --------------------------------------------------------------------------


def test_import_nrrd_data(nrrd_dir):
    data = load_data(nrrd_dir)
    ct, cst = data["ct"], data["cst"]

    assert ct.cube_hu.GetSize() == (15, 20, 10)
    assert np.allclose(ct.cube_hu.GetSpacing(), (10.0, 15.0, 5.0))
    assert np.allclose(ct.cube_hu.GetOrigin(), (-75.0, -150.0, -25.0))

    # NRRD carries embedded pyradplan/Slicer metadata, so names/types survive.
    by_name = {v.name: v for v in cst.vois}
    assert set(by_name) == {"Target", "Body", "Core"}
    assert by_name["Target"].voi_type == "TARGET"

    # Label map -> disjoint, non-empty VOIs.
    sets = list(_voxel_sets(cst).values())
    assert all(len(s) > 0 for s in sets)
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j])


def test_import_meta_data(meta_dir):
    data = load_data(meta_dir)
    ct, cst = data["ct"], data["cst"]

    assert ct.cube_hu.GetSize() == (15, 20, 10)
    assert np.allclose(ct.cube_hu.GetOrigin(), (-75.0, -150.0, -25.0))

    # MetaImage carries embedded metadata (no sidecar), so names/types survive.
    by_name = {v.name: v for v in cst.vois}
    assert set(by_name) == {"Target", "Body", "Core"}
    assert by_name["Target"].voi_type == "TARGET"


def test_import_nifti_data(nifti_dir):
    data = load_data(nifti_dir)
    ct, cst = data["ct"], data["cst"]
    assert ct.cube_hu.GetSize() == (15, 20, 10)
    # NIfTI cannot store arbitrary metadata and there is no sidecar -> 3 VOIs, generic names.
    assert len(cst.vois) == 3


def test_real_data_ct_identical_across_formats(nifti_dir, nrrd_dir, meta_dir):
    arrays = [
        sitk.GetArrayFromImage(load_data(d)["ct"].cube_hu) for d in (nifti_dir, nrrd_dir, meta_dir)
    ]
    assert np.array_equal(arrays[0], arrays[1])
    assert np.array_equal(arrays[0], arrays[2])


def test_load_patient_single_nifti_file(nifti_dir):
    ct, cst = load_patient(nifti_dir / "ct.nii.gz")
    assert isinstance(ct, CT)
    assert ct.cube_hu.GetSize() == (15, 20, 10)
    assert cst is None  # a lone image file is just a CT


# --------------------------------------------------------------------------
# Round-trip (export + import), parametrized over all three formats
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_roundtrip_ct(fmt, tmp_path):
    ct = _make_ct()
    out = tmp_path / f"patient_{fmt}"
    save_data(ct=ct, format=fmt, file_name=str(out))

    assert detect_format(out) == fmt
    assert is_container_format(fmt) is False
    assert _ct_close(ct, load_data(out)["ct"])


@pytest.mark.parametrize("fmt", FORMATS)
def test_roundtrip_dose(fmt, tmp_path):
    ct = _make_ct()
    dose = _dummy_dose(ct)
    out = tmp_path / f"withdose_{fmt}"
    save_data(ct=ct, dose=dose, format=fmt, file_name=str(out))

    data = load_data(out)
    assert isinstance(data["dose"], sitk.Image)
    assert np.allclose(
        sitk.GetArrayFromImage(data["dose"]), sitk.GetArrayFromImage(dose), atol=1e-5
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_roundtrip_cst(fmt, tmp_path):
    ct = _make_ct()
    cst = _make_cst(ct)
    out = tmp_path / f"cst_{fmt}"
    save_data(ct=ct, cst=cst, format=fmt, file_name=str(out))

    cst2 = load_data(out)["cst"]
    assert isinstance(cst2, StructureSet)
    # Disjoint VOIs -> exact label-map round-trip; sidecar preserves metadata (all formats).
    assert _voxel_sets(cst) == _voxel_sets(cst2)
    by_name = {v.name: v for v in cst2.vois}
    assert set(by_name) == {"Target", "Core"}
    assert by_name["Target"].voi_type == "TARGET"
    assert by_name["Target"].alpha_x == pytest.approx(0.2)
    assert by_name["Target"].beta_x == pytest.approx(0.03)


def test_single_file_ct_roundtrip(tmp_path):
    ct = _make_ct()
    out = tmp_path / "ct.nii.gz"
    written = save_data(ct=ct, file_name=str(out))
    assert str(written).endswith("ct.nii.gz")
    assert detect_format(out) == "nifti"
    ct2, _ = load_patient(out)
    assert _ct_close(ct, ct2)


def test_single_file_target_rejects_cst(tmp_path):
    ct = _make_ct()
    cst = _make_cst(ct)
    with pytest.raises(ValueError):
        NiftiHandler(tmp_path / "seg.nii.gz").save(cst=cst)


def test_single_file_target_rejects_ct_and_dose(tmp_path):
    # A single image file can hold only one image; refuse rather than drop the dose.
    ct = _make_ct()
    dose = sitk.Cast(ct.cube_hu, sitk.sitkFloat32)
    with pytest.raises(ValueError):
        NiftiHandler(tmp_path / "image.nii.gz").save(ct=ct, dose=dose)


@pytest.mark.parametrize(
    "geometry, expected_index",
    [
        ("origin", (2, 1, 1)),
        ("spacing", (2, 1, 1)),
        ("direction", (5, 1, 1)),
        ("size", (2, 1, 1)),
    ],
)
def test_label_import_preserves_world_coordinates_and_metadata(tmp_path, geometry, expected_index):
    from pyRadPlan.io import NrrdHandler

    reference = sitk.Image([7, 6, 5], sitk.sitkFloat32)
    label = sitk.Image([4, 3, 3] if geometry == "size" else [7, 6, 5], sitk.sitkUInt8)
    label[1, 1, 1] = 5
    if geometry in ("origin", "size"):
        label.SetOrigin((1.0, 0.0, 0.0))
    elif geometry == "spacing":
        label.SetSpacing((2.0, 1.0, 1.0))
    else:
        label.SetOrigin((6.0, 0.0, 0.0))
        label.SetDirection((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    label.SetMetaData("pyradplan_voi_names", "a|b|c|d|target")
    label.SetMetaData("pyradplan_voi_types", "OAR|OAR|OAR|OAR|TARGET")
    sitk.WriteImage(reference, str(tmp_path / "ct.nrrd"))
    sitk.WriteImage(label, str(tmp_path / "cst.nrrd"))

    voi = NrrdHandler(tmp_path).load_data()["cst"].vois[0]
    assert voi.name == "target" and voi.voi_type == "TARGET"
    assert voi.mask.GetSize() == reference.GetSize()
    assert voi.mask.GetOrigin() == reference.GetOrigin()
    assert voi.mask.GetDirection() == reference.GetDirection()
    assert voi.mask[expected_index] == 1
    assert voi.mask.TransformIndexToPhysicalPoint(
        expected_index
    ) == label.TransformIndexToPhysicalPoint((1, 1, 1))
    assert voi.mask[0, 0, 0] == 0
    assert set(np.unique(sitk.GetArrayFromImage(voi.mask))) == {0, 1}


@pytest.mark.parametrize("outside_ct", [False, True])
def test_label_import_warns_on_resampling_and_empty_labels(caplog, outside_ct):
    from pyRadPlan.io.sitk_based.base._serialize import label_image_to_vois

    ct = CT(cube_hu=sitk.Image([5, 4, 3], sitk.sitkFloat32))
    label = sitk.Image([5, 4, 3], sitk.sitkUInt8)
    label[1, 1, 1] = 5
    if outside_ct:
        label.SetOrigin((100.0, 0.0, 0.0))
    sidecar = {"vois": {"5": {"name": "target", "voi_type": "TARGET"}}}

    with caplog.at_level("WARNING", logger="pyRadPlan.io.sitk_based.base._serialize"):
        vois = label_image_to_vois(label, ct, sidecar)

    assert len(vois) == 1 and vois[0].name == "target"
    messages = [r.getMessage() for r in caplog.records if r.name.endswith("base._serialize")]
    if outside_ct:
        assert not np.any(sitk.GetArrayViewFromImage(vois[0].mask))
        assert len(messages) == 2
        assert "resampling labels onto the CT grid" in messages[0]
        assert "Label 5 (target) is empty" in messages[1]
    else:
        assert np.any(sitk.GetArrayViewFromImage(vois[0].mask))
        assert messages == []


def test_label_import_reads_legacy_visible_color_key():
    """Sidecars written before the rename store the color under ``visible_color``."""
    from pyRadPlan.io.sitk_based.base._serialize import label_image_to_vois

    ct = CT(cube_hu=sitk.Image([5, 4, 3], sitk.sitkFloat32))
    label = sitk.Image([5, 4, 3], sitk.sitkUInt8)
    label[1, 1, 1] = 5
    sidecar = {"vois": {"5": {"name": "target", "voi_type": "TARGET", "visible_color": [7, 8, 9]}}}

    vois = label_image_to_vois(label, ct, sidecar)

    assert tuple(vois[0].color) == (7, 8, 9)
    assert "color" in vois[0].model_fields_set


def _make_cst_with_tissues(ct: CT) -> StructureSet:
    cst = _make_cst(ct)
    tissues = {
        "Target": TissueProperties.from_preset("HSG", "LEM_II"),
        "Core": MKMTissueProperties(
            cell_line="CHO", r_nucleus=4.1, r_domain=0.2, alpha_0=0.228, beta_0=0.02, alpha_x=0.3
        ),
    }
    vois = [voi.model_copy(update={"tissue_properties": tissues[voi.name]}) for voi in cst.vois]
    return validate_cst(vois, ct)


def _assert_same_tissues(cst: StructureSet, cst2: StructureSet) -> None:
    by_name = {v.name: v for v in cst2.vois}
    for voi in cst.vois:
        restored = by_name[voi.name]
        assert type(restored.tissue_properties) is type(voi.tissue_properties)
        assert restored.tissue_properties == voi.tissue_properties
        assert (
            restored.tissue_properties.reference_sources == voi.tissue_properties.reference_sources
        )
        assert (restored.alpha_x, restored.beta_x) == (voi.alpha_x, voi.beta_x)
    assert by_name["Core"].tissue_properties.reference_sources == {
        "alpha_x": "explicit",
        "beta_x": "estimated",
    }


@pytest.mark.parametrize("fmt", FORMATS)
def test_roundtrip_cst_tissue_properties(fmt, tmp_path):
    ct = _make_ct()
    cst = _make_cst_with_tissues(ct)
    out = tmp_path / f"tissues_{fmt}"
    save_data(ct=ct, cst=cst, format=fmt, file_name=str(out))

    _assert_same_tissues(cst, load_data(out)["cst"])


def test_label_import_reads_tissue_properties_from_header():
    """Without a sidecar, the tissue properties come from the ``pyradplan_*`` header."""
    from pyRadPlan.io.sitk_based.base._serialize import (
        apply_interop_metadata,
        cst_to_label_image,
        label_image_to_vois,
    )

    ct = _make_ct()
    cst = _make_cst_with_tissues(ct)
    label, extents = cst_to_label_image(cst)
    apply_interop_metadata(label, cst, extents)

    _assert_same_tissues(cst, validate_cst(label_image_to_vois(label, ct, None), ct))


def test_label_import_without_tissue_properties_keeps_alpha_beta():
    """Sidecars written before tissue properties were stored only carry alpha/beta."""
    from pyRadPlan.io.sitk_based.base._serialize import (
        build_sidecar,
        cst_to_label_image,
        label_image_to_vois,
    )

    ct = _make_ct()
    cst = _make_cst(ct)
    label, _ = cst_to_label_image(cst)
    sidecar = build_sidecar(cst)
    for meta in sidecar["vois"].values():
        del meta["tissue_properties"]

    by_name = {v.name: v for v in label_image_to_vois(label, ct, sidecar)}
    assert type(by_name["Target"].tissue_properties) is TissueProperties
    assert by_name["Target"].tissue_properties.alpha_x == pytest.approx(0.2)
    assert by_name["Target"].beta_x == pytest.approx(0.03)
