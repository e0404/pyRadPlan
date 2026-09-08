"""Tests for the format-agnostic binary (CT + per-file mask) importer."""

import numpy as np
import SimpleITK as sitk
import pytest

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan import load_binary_patient
from pyRadPlan.io import list_image_files
from pyRadPlan.io.sitk_based import (
    masks_to_cst,
    read_ct_image,
    mask_file_to_voi,
    image_file_to_vois,
    infer_image_kind,
)


# --------------------------------------------------------------------------
# Synthetic dataset: a CT plus on-grid and off-grid masks in mixed formats
# --------------------------------------------------------------------------


def _write_ct(path) -> sitk.Image:
    img = sitk.GetImageFromArray(np.zeros((10, 20, 20), dtype=np.float32))  # (z, y, x)
    img.SetSpacing((2.0, 2.0, 3.0))
    img.SetOrigin((-5.0, -7.0, 1.0))
    sitk.WriteImage(img, str(path), useCompression=True)
    return img


def _write_on_grid_mask(path, ref: sitk.Image) -> None:
    arr = np.zeros((10, 20, 20), dtype=np.uint8)
    arr[2:6, 4:10, 4:10] = 1
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ref)
    sitk.WriteImage(mask, str(path), useCompression=True)


def _write_off_grid_mask(path, ref: sitk.Image) -> None:
    # Finer grid (1 mm vs 2 mm) covering the same physical extent as the CT.
    arr = np.zeros((10, 40, 40), dtype=np.uint8)
    arr[2:6, 8:20, 8:20] = 1
    mask = sitk.GetImageFromArray(arr)
    mask.SetSpacing((1.0, 1.0, 3.0))
    mask.SetOrigin(ref.GetOrigin())
    sitk.WriteImage(mask, str(path), useCompression=True)


@pytest.fixture
def binary_dataset(tmp_path):
    """Write a CT (.nii.gz) and a structures/ folder with mixed-format masks."""
    ct_file = tmp_path / "patient_ct.nii.gz"
    ref = _write_ct(ct_file)

    struct_dir = tmp_path / "structures"
    struct_dir.mkdir()
    _write_on_grid_mask(struct_dir / "PTV.nrrd", ref)  # TARGET, on grid
    _write_on_grid_mask(struct_dir / "Body.nii.gz", ref)  # EXTERNAL, mixed format
    _write_off_grid_mask(struct_dir / "Heart.nrrd", ref)  # OAR, off grid
    return {"root": tmp_path, "ct_file": ct_file, "struct_dir": struct_dir, "ref": ref}


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_load_binary_patient_folder(binary_dataset):
    ct, cst = load_binary_patient(binary_dataset["ct_file"], binary_dataset["struct_dir"])

    assert isinstance(ct, CT)
    assert isinstance(cst, StructureSet)
    assert ct.cube_hu.GetSize() == (20, 20, 10)

    by_name = {v.name: v for v in cst.vois}
    assert set(by_name) == {"PTV", "Body", "Heart"}
    # Types come from the filename heuristic.
    assert by_name["PTV"].voi_type == "TARGET"
    assert by_name["Body"].voi_type == "EXTERNAL"
    assert by_name["Heart"].voi_type == "OAR"
    # Every VOI mask is aligned to the CT grid.
    for voi in cst.vois:
        assert voi.mask.GetSize() == ct.cube_hu.GetSize()


def test_off_grid_mask_is_resampled_non_empty(binary_dataset):
    ct = read_ct_image(binary_dataset["ct_file"])
    voi = mask_file_to_voi(ct, binary_dataset["struct_dir"] / "Heart.nrrd")
    assert voi.mask.GetSize() == ct.cube_hu.GetSize()
    assert len(voi.indices_numpy) > 0  # survived resampling onto the coarser CT grid


def test_load_binary_patient_file_list(binary_dataset):
    files = [
        binary_dataset["struct_dir"] / "PTV.nrrd",
        binary_dataset["struct_dir"] / "Heart.nrrd",
    ]
    _, cst = load_binary_patient(binary_dataset["ct_file"], files)
    assert {v.name for v in cst.vois} == {"PTV", "Heart"}


def test_selections_override_name_and_type(binary_dataset):
    ct = read_ct_image(binary_dataset["ct_file"])
    selections = [
        {
            "path": str(binary_dataset["struct_dir"] / "PTV.nrrd"),
            "name": "Tumor",
            "voi_type": "TARGET",
        },
        {"path": str(binary_dataset["struct_dir"] / "Body.nii.gz"), "voi_type": "IGNORED"},
        {
            "path": str(binary_dataset["struct_dir"] / "Heart.nrrd"),
            "name": "Heart",
            "voi_type": "OAR",
        },
    ]
    cst = masks_to_cst(ct, None, selections=selections)
    names = {v.name for v in cst.vois}
    assert names == {"Tumor", "Heart"}  # renamed PTV, Body dropped via IGNORED
    assert next(v for v in cst.vois if v.name == "Tumor").voi_type == "TARGET"


def test_list_image_files(binary_dataset):
    files = list_image_files(binary_dataset["struct_dir"])
    assert {f.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for f in files} == {
        "PTV.nrrd",
        "Body.nii.gz",
        "Heart.nrrd",
    }


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        # (sitk has no bool pixel type; on-disk boolean masks arrive as uint8)
        (np.zeros((2, 2, 2), dtype=np.uint8), "structures"),  # unsigned -> mask
        (np.full((2, 2, 2), -1000, dtype=np.int16), "ct"),  # negatives -> HU
        (np.arange(8, dtype=np.int16).reshape(2, 2, 2), "structures"),  # few labels
        (np.arange(1000, dtype=np.int16).reshape(10, 10, 10), "ct"),  # many values
        (np.full((2, 2, 2), 1.5, dtype=np.float32), "dose"),  # positive float
        (np.linspace(-100, 100, 8, dtype=np.float32).reshape(2, 2, 2), "ct"),  # HU float
    ],
)
def test_infer_image_kind(array, expected):
    assert infer_image_kind(sitk.GetImageFromArray(array)) == expected


def test_masks_to_cst_empty_raises(binary_dataset):
    ct = read_ct_image(binary_dataset["ct_file"])
    with pytest.raises(ValueError):
        masks_to_cst(ct, [], selections=None)


def test_image_file_to_vois_single_mask(binary_dataset):
    ct = read_ct_image(binary_dataset["ct_file"])
    vois = image_file_to_vois(ct, binary_dataset["struct_dir"] / "PTV.nrrd")
    assert len(vois) == 1
    assert vois[0].name == "PTV"
    assert vois[0].voi_type == "TARGET"


def test_image_file_to_vois_labelmap(binary_dataset, tmp_path):
    ct = read_ct_image(binary_dataset["ct_file"])

    # Two disjoint labels in one file -> two VOIs with fallback names.
    arr = np.zeros((10, 20, 20), dtype=np.uint8)
    arr[2:4, 2:6, 2:6] = 1
    arr[6:8, 10:14, 10:14] = 2
    label_image = sitk.GetImageFromArray(arr)
    label_image.CopyInformation(ct.cube_hu)
    labelmap = tmp_path / "seg.nrrd"
    sitk.WriteImage(label_image, str(labelmap), useCompression=True)

    vois = image_file_to_vois(ct, labelmap)
    assert {v.name for v in vois} == {"Segment_1", "Segment_2"}
    assert all(len(v.indices_numpy) > 0 for v in vois)


def test_image_file_to_vois_labelmap_with_sidecar(binary_dataset, tmp_path):
    import json

    ct = read_ct_image(binary_dataset["ct_file"])
    arr = np.zeros((10, 20, 20), dtype=np.uint8)
    arr[2:4, 2:6, 2:6] = 1
    arr[6:8, 10:14, 10:14] = 2
    label_image = sitk.GetImageFromArray(arr)
    label_image.CopyInformation(ct.cube_hu)
    labelmap = tmp_path / "seg.nrrd"
    sitk.WriteImage(label_image, str(labelmap), useCompression=True)
    sidecar = {
        "vois": {
            "1": {"name": "Tumor", "voi_type": "TARGET"},
            "2": {"name": "Cord", "voi_type": "OAR"},
        }
    }
    (tmp_path / "seg.json").write_text(json.dumps(sidecar), encoding="utf-8")

    vois = image_file_to_vois(ct, labelmap)
    by_name = {v.name: v for v in vois}
    assert set(by_name) == {"Tumor", "Cord"}
    assert by_name["Tumor"].voi_type == "TARGET"
