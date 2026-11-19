"""4D CT/dose export, 4D structure sets and overlap-preserving segmentations."""

import gzip
import warnings

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst, validate_voi
from pyRadPlan.io import load_data, load_patient, save_data
from pyRadPlan.io.sitk_based.nrrd import NrrdExporter
from pyRadPlan.io.sitk_based.base._serialize import (
    cst_to_label_image,
    label_image_to_vois,
    masks_to_label_image,
    phase_masks,
    phase_reference,
)

FORMATS = ["nifti", "nrrd", "meta"]
NUM_PHASES = 3
SHAPE = (10, 20, 20)  # (z, y, x)


def _make_ct_4d() -> CT:
    rng = np.random.default_rng(1)
    phases = []
    for _ in range(NUM_PHASES):
        image = sitk.GetImageFromArray(rng.integers(-1000, 1000, SHAPE).astype(np.float32))
        phases.append(image)
    cube = sitk.JoinSeries(phases)
    cube.SetSpacing((2.0, 3.0, 4.0, 1.0))
    cube.SetOrigin((-5.0, -10.0, -2.0, 0.0))
    return validate_ct(cube_hu=cube)


def _box(mask, z, y, x):
    mask[z[0] : z[1], y[0] : y[1], x[0] : x[1]] = 1


def _masks_3d() -> dict[str, np.ndarray]:
    body = np.zeros(SHAPE, np.uint8)
    _box(body, (0, 10), (2, 18), (2, 18))
    ptv = np.zeros(SHAPE, np.uint8)
    _box(ptv, (3, 8), (6, 14), (6, 14))
    ctv = np.zeros(SHAPE, np.uint8)
    _box(ctv, (4, 7), (8, 12), (8, 12))
    oar = np.zeros(SHAPE, np.uint8)
    _box(oar, (1, 3), (3, 6), (3, 6))
    return {"Body": body, "PTV": ptv, "CTV": ctv, "OAR": oar}


_TYPES = {"Body": "EXTERNAL", "PTV": "TARGET", "CTV": "TARGET", "OAR": "OAR"}


def _make_cst(ct: CT, masks=None, phased=False) -> StructureSet:
    """Overlapping structures; with ``phased`` the masks move by one voxel per phase."""
    masks = masks or _masks_3d()
    vois = []
    for name, mask in masks.items():
        if ct.num_of_ct_scen > 1:
            shift = (lambda p: p) if phased else (lambda p: 0)
            mask = np.stack([np.roll(mask, shift(p), axis=2) for p in range(NUM_PHASES)])
        vois.append(validate_voi(name=name, voi_type=_TYPES[name], mask=mask, ct_image=ct))
    return validate_cst(vois, ct)


def _make_ct_3d() -> CT:
    return validate_ct(cube_hu=_make_ct_4d().cube_hu[:, :, :, 0])


def _dose_4d(ct: CT) -> sitk.Image:
    arr = np.random.default_rng(2).random((NUM_PHASES, *SHAPE), dtype=np.float32)
    dose = sitk.GetImageFromArray(arr, isVector=False)
    dose.CopyInformation(ct.cube_hu)
    return dose


def _assert_same_image(a: sitk.Image, b: sitk.Image, exact=True):
    assert a.GetDimension() == b.GetDimension()
    assert a.GetSize() == b.GetSize()
    assert np.allclose(a.GetSpacing(), b.GetSpacing())
    assert np.allclose(a.GetOrigin(), b.GetOrigin())
    assert np.allclose(a.GetDirection(), b.GetDirection())
    x, y = sitk.GetArrayFromImage(a), sitk.GetArrayFromImage(b)
    assert np.array_equal(x, y) if exact else np.allclose(x, y, atol=1e-5)


def _assert_same_cst(a: StructureSet, b: StructureSet, order_matters=False):
    assert len(a.vois) == len(b.vois)
    by_name = {v.name: v for v in b.vois}
    assert set(by_name) == {v.name for v in a.vois}
    if order_matters:
        assert [v.name for v in a.vois] == [v.name for v in b.vois]
    for voi in a.vois:
        other = by_name[voi.name]
        assert other.voi_type == voi.voi_type
        assert tuple(other.color) == tuple(voi.color)
        assert other.mask.GetSize() == voi.mask.GetSize()
        assert np.array_equal(sitk.GetArrayFromImage(other.mask), sitk.GetArrayFromImage(voi.mask))


# --------------------------------------------------------------------------
# 4D CT and dose
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_4d_ct_and_dose_roundtrip(fmt, tmp_path):
    ct = _make_ct_4d()
    dose = _dose_4d(ct)
    save_data(ct=ct, dose=dose, format=fmt, file_name=str(tmp_path / "p"))

    data = load_data(tmp_path / "p")
    assert data["ct"].num_of_ct_scen == NUM_PHASES
    _assert_same_image(ct.cube_hu, data["ct"].cube_hu)
    _assert_same_image(dose, data["dose"])

    ct2, cst2 = load_patient(tmp_path / "p")
    assert ct2.num_of_ct_scen == NUM_PHASES and cst2 is None


@pytest.mark.parametrize("fmt", FORMATS)
def test_4d_ct_oblique_direction_roundtrip(fmt, tmp_path):
    ct = _make_ct_4d()
    cube = ct.cube_hu
    direction = np.eye(4)
    direction[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cube.SetDirection(direction.ravel().tolist())
    save_data(ct=CT(cube_hu=cube), format=fmt, file_name=str(tmp_path / "p"))
    _assert_same_image(cube, load_data(tmp_path / "p")["ct"].cube_hu)


def test_4d_single_file_roundtrip(tmp_path):
    ct = _make_ct_4d()
    save_data(ct=ct, file_name=str(tmp_path / "ct.nii.gz"))
    ct2, _ = load_patient(tmp_path / "ct.nii.gz")
    _assert_same_image(ct.cube_hu, ct2.cube_hu)


def test_npz_4d_roundtrip(tmp_path):
    ct = _make_ct_4d()
    cst = _make_cst(ct, phased=True)
    dose = _dose_4d(ct)
    save_data(ct=ct, cst=cst, dose=dose, file_name=str(tmp_path / "p.npz"))

    data = load_data(tmp_path / "p.npz")
    assert data["ct"].num_of_ct_scen == NUM_PHASES
    _assert_same_image(ct.cube_hu, data["ct"].cube_hu)
    _assert_same_image(dose, data["dose"])
    _assert_same_cst(cst, data["cst"], order_matters=True)
    assert data["cst"].num_of_scenarios == NUM_PHASES


# --------------------------------------------------------------------------
# Segmentations: modes, overlaps
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_layers_keep_overlaps(fmt, tmp_path):
    ct = _make_ct_3d()
    cst = _make_cst(ct)
    out = tmp_path / "p"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_data(
            ct=ct,
            cst=cst,
            format=fmt,
            file_name=str(out),
            exporter_options={"segmentation": "layers"},
        )

    _assert_same_cst(cst, load_data(out)["cst"], order_matters=True)


@pytest.mark.parametrize("fmt", FORMATS)
def test_auto_picks_layers_for_overlaps_and_labelmap_for_disjoint(fmt, tmp_path):
    ct = _make_ct_3d()
    auto = {"segmentation": "auto"}
    save_data(
        ct=ct,
        cst=_make_cst(ct),
        format=fmt,
        file_name=str(tmp_path / "overlap"),
        exporter_options=auto,
    )
    _assert_same_cst(_make_cst(ct), load_data(tmp_path / "overlap")["cst"])
    names = {p.name for p in (tmp_path / "overlap").iterdir()}
    layered = "cst.seg.nrrd" if fmt == "nrrd" else f"cst{'.nii.gz' if fmt == 'nifti' else '.mha'}"
    assert layered in names

    masks = _masks_3d()
    del masks["Body"], masks["CTV"]
    disjoint = _make_cst(ct, masks)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_data(
            ct=ct,
            cst=disjoint,
            format=fmt,
            file_name=str(tmp_path / "disjoint"),
            exporter_options=auto,
        )
    names = {p.name for p in (tmp_path / "disjoint").iterdir()}
    assert "cst.seg.nrrd" not in names
    assert any(n.startswith("cst.") and not n.endswith(".json") for n in names)
    labels = load_data(tmp_path / "disjoint")["cst"]
    _assert_same_cst(disjoint, labels)


@pytest.mark.parametrize("fmt", FORMATS)
def test_labelmap_warns_about_lost_voxels(fmt, tmp_path):
    ct = _make_ct_3d()
    cst = _make_cst(ct)
    with pytest.warns(UserWarning, match="overlapping voxels of") as rec:
        save_data(
            ct=ct,
            cst=cst,
            format=fmt,
            file_name=str(tmp_path / "p"),
            exporter_options={"segmentation": "labelmap"},
        )
    message = str(rec[0].message)
    # PTV is listed before CTV and wins the voxels they share; BODY always comes last.
    assert "Body" in message and "CTV" in message
    assert "PTV" not in message and "OAR" not in message

    restored = {v.name: v for v in load_data(tmp_path / "p")["cst"].vois}
    assert restored["PTV"].indices.size == cst.vois[1].indices.size
    assert "CTV" not in restored  # completely covered by the PTV, so no label is left
    assert restored["Body"].indices.size < cst.vois[0].indices.size


@pytest.mark.parametrize("fmt", FORMATS)
def test_default_segmentation_is_labelmap(fmt, tmp_path):
    ct = _make_ct_3d()
    with pytest.warns(UserWarning, match="overlapping voxels of"):
        save_data(ct=ct, cst=_make_cst(ct), format=fmt, file_name=str(tmp_path / "p"))
    names = {p.name for p in (tmp_path / "p").iterdir()}
    assert "cst.seg.nrrd" not in names
    assert any(n.startswith("cst.") and not n.endswith(".json") for n in names)


def _save(ct, cst, fmt, path, mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_data(
            ct=ct,
            cst=cst,
            format=fmt,
            file_name=str(path),
            exporter_options={"segmentation": mode},
        )


@pytest.mark.parametrize("fmt", FORMATS)
def test_reexport_removes_stale_segmentation_variants(fmt, tmp_path):
    ct = _make_ct_3d()
    out = tmp_path / "p"
    _save(ct, _make_cst(ct), fmt, out, "layers")

    masks = _masks_3d()
    del masks["Body"], masks["CTV"]
    second = _make_cst(ct, masks)
    _save(ct, second, fmt, out, "labelmap")

    _assert_same_cst(second, load_data(out)["cst"])
    stems = {p.name for p in out.iterdir() if p.name.startswith("cst")}
    assert not any(".seg." in n for n in stems)


@pytest.mark.parametrize("fmt", FORMATS)
def test_reexport_3d_after_4d_removes_phase_files(fmt, tmp_path):
    ct4 = _make_ct_4d()
    out = tmp_path / "p"
    _save(ct4, _make_cst(ct4, phased=True), fmt, out, "labelmap")
    assert any(p.name.startswith("cst_phase") for p in out.iterdir())

    ct3 = _make_ct_3d()
    cst3 = _make_cst(ct3)
    _save(ct3, cst3, fmt, out, "layers")

    assert not any(p.name.startswith("cst_phase") for p in out.iterdir())
    loaded = load_data(out)["cst"]
    assert loaded.num_of_scenarios == 1
    _assert_same_cst(cst3, loaded, order_matters=True)


def test_reexport_keeps_unrelated_files(tmp_path):
    ct = _make_ct_3d()
    out = tmp_path / "p"
    out.mkdir()
    (out / "cst_notes.txt").write_text("keep")
    (out / "cst_backup.json").write_text("{}")
    _save(ct, _make_cst(ct), "nrrd", out, "labelmap")
    assert (out / "cst_notes.txt").exists() and (out / "cst_backup.json").exists()


def test_invalid_segmentation_mode(tmp_path):
    with pytest.raises(ValueError, match="segmentation"):
        NrrdExporter(tmp_path / "x", segmentation="bogus")


@pytest.mark.parametrize("fmt", FORMATS)
def test_layers_roundtrip_with_tissue_properties_and_metadata(fmt, tmp_path):
    ct = _make_ct_3d()
    cst = _make_cst(ct)
    vois = [
        v.model_copy(update={"overlap_priority": 5 + i, "visible": i != 1})
        for i, v in enumerate(cst.vois)
    ]
    cst = validate_cst(vois, ct)
    save_data(
        ct=ct,
        cst=cst,
        format=fmt,
        file_name=str(tmp_path / "p"),
        exporter_options={"segmentation": "layers"},
    )
    restored = {v.name: v for v in load_data(tmp_path / "p")["cst"].vois}
    for voi in cst.vois:
        other = restored[voi.name]
        assert other.overlap_priority == voi.overlap_priority
        assert other.visible == voi.visible
        assert other.tissue_properties == voi.tissue_properties


def test_old_labelmap_folders_still_import(nifti_dir, nrrd_dir, meta_dir):
    for directory in (nifti_dir, nrrd_dir, meta_dir):
        cst = load_data(directory)["cst"]
        assert len(cst.vois) == 3
        assert cst.num_of_scenarios == 1


def test_legacy_sidecar_without_encoding_is_a_labelmap(tmp_path):
    ct = _make_ct_3d()
    masks = _masks_3d()
    del masks["Body"], masks["CTV"]
    cst = _make_cst(ct, masks)
    save_data(ct=ct, cst=cst, format="nifti", file_name=str(tmp_path / "p"))
    sidecar = tmp_path / "p" / "cst.json"
    import json

    data = json.loads(sidecar.read_text())
    assert data["encoding"] == "labelmap"
    del data["encoding"]
    sidecar.write_text(json.dumps(data))
    _assert_same_cst(cst, load_data(tmp_path / "p")["cst"])


def test_labelmap_helpers_on_4d_structure_set():
    ct = _make_ct_4d()
    cst = _make_cst(ct, phased=True)
    image, extents = cst_to_label_image(cst, phase=2)
    assert image.GetDimension() == 3 and set(extents) == {0, 1, 2, 3}
    reference = phase_reference(cst)
    labels, _, lost = masks_to_label_image(cst, phase_masks(cst, 2), reference)
    assert lost == ["CTV", "Body"]
    vois = label_image_to_vois(labels, validate_ct(cube_hu=reference), None)
    assert len(vois) == 3  # the CTV has no voxel left


# --------------------------------------------------------------------------
# Slicer segmentation format
# --------------------------------------------------------------------------


def _read_header(path) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read()
    header = raw.split(b"\n\n", 1)[0].decode()
    lines = header.split("\n")
    assert lines[0] == "NRRD0004"
    fields = {}
    for line in lines[1:]:
        if ":=" in line:
            key, value = line.split(":=", 1)
        else:
            key, value = line.split(": ", 1)
        fields[key] = value
    return fields


def test_slicer_seg_nrrd_layout(tmp_path):
    ct = _make_ct_3d()
    cst = _make_cst(ct)
    save_data(
        ct=ct,
        cst=cst,
        format="nrrd",
        file_name=str(tmp_path / "p"),
        exporter_options={"segmentation": "layers"},
    )
    path = tmp_path / "p" / "cst.seg.nrrd"
    h = _read_header(path)

    assert h["dimension"] == "4"
    assert h["kinds"] == "list domain domain domain"
    assert h["space"] == "left-posterior-superior"
    assert h["space directions"].startswith("none (2.0,0.0,0.0) (0.0,3.0,0.0)")
    assert h["Segmentation_MasterRepresentation"] == "Binary labelmap"
    layers = int(h["sizes"].split()[0])
    assert layers == 3  # Body > PTV > CTV nest; OAR fits next to Body in layer 0
    seen_layers = set()
    for i, voi in enumerate(cst.vois):
        assert h[f"Segment{i}_Name"] == voi.name
        assert h[f"Segment{i}_ID"] == f"Segment_{i + 1}"
        assert h[f"Segment{i}_LabelValue"] == str(i + 1)
        assert len(h[f"Segment{i}_Color"].split()) == 3
        assert len(h[f"Segment{i}_Extent"].split()) == 6
        seen_layers.add(int(h[f"Segment{i}_Layer"]))
    assert seen_layers == {0, 1, 2}

    image = sitk.ReadImage(str(path))
    assert image.GetNumberOfComponentsPerPixel() == layers
    assert np.allclose(image.GetSpacing(), ct.cube_hu.GetSpacing())
    assert np.allclose(image.GetOrigin(), ct.cube_hu.GetOrigin())
    body_layer = int(h["Segment0_Layer"])
    layer = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(image, body_layer))
    assert np.array_equal(layer == 1, _masks_3d()["Body"] > 0)


def test_slicer_seg_nrrd_single_layer_is_3d(tmp_path):
    ct = _make_ct_3d()
    masks = _masks_3d()
    del masks["Body"], masks["CTV"]
    cst = _make_cst(ct, masks)
    save_data(
        ct=ct,
        cst=cst,
        format="nrrd",
        file_name=str(tmp_path / "p"),
        exporter_options={"segmentation": "layers"},
    )
    h = _read_header(tmp_path / "p" / "cst.seg.nrrd")
    assert h["dimension"] == "3" and h["kinds"] == "domain domain domain"
    _assert_same_cst(cst, load_data(tmp_path / "p")["cst"], order_matters=True)


def test_slicer_seg_nrrd_import_without_sidecar_or_pyradplan_keys(tmp_path):
    ct = _make_ct_3d()
    cst = _make_cst(ct)
    save_data(
        ct=ct,
        cst=cst,
        format="nrrd",
        file_name=str(tmp_path / "p"),
        exporter_options={"segmentation": "layers"},
    )
    path = tmp_path / "p" / "cst.seg.nrrd"
    (tmp_path / "p" / "cst.json").unlink()
    raw = path.read_bytes()
    header, data = raw.split(b"\n\n", 1)
    stripped = b"\n".join(l for l in header.split(b"\n") if not l.startswith(b"pyradplan_"))
    path.write_bytes(stripped + b"\n\n" + data)

    restored = load_data(tmp_path / "p")["cst"]
    assert [v.name for v in restored.vois] == [v.name for v in cst.vois]
    assert [tuple(v.color) for v in restored.vois] == [tuple(v.color) for v in cst.vois]
    for a, b in zip(cst.vois, restored.vois):
        assert np.array_equal(sitk.GetArrayFromImage(a.mask), sitk.GetArrayFromImage(b.mask))


def test_slicer_seg_nrrd_oblique_geometry(tmp_path):
    ct = _make_ct_3d()
    cube = ct.cube_hu
    cube.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    ct = CT(cube_hu=cube)
    cst = _make_cst(ct)
    save_data(
        ct=ct,
        cst=cst,
        format="nrrd",
        file_name=str(tmp_path / "p"),
        exporter_options={"segmentation": "layers"},
    )
    image = sitk.ReadImage(str(tmp_path / "p" / "cst.seg.nrrd"))
    assert np.allclose(image.GetDirection(), cube.GetDirection())
    assert np.allclose(image.GetOrigin(), cube.GetOrigin())
    assert np.allclose(image.GetSpacing(), cube.GetSpacing())
    _assert_same_cst(cst, load_data(tmp_path / "p")["cst"])


def test_slicer_seg_nrrd_data_is_gzip(tmp_path):
    ct = _make_ct_3d()
    save_data(
        ct=ct,
        cst=_make_cst(ct),
        format="nrrd",
        file_name=str(tmp_path / "p"),
        exporter_options={"segmentation": "layers"},
    )
    raw = (tmp_path / "p" / "cst.seg.nrrd").read_bytes().split(b"\n\n", 1)[1]
    assert len(gzip.decompress(raw)) == 3 * np.prod(SHAPE)


# --------------------------------------------------------------------------
# 4D structure sets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("segmentation", ["labelmap", "layers", "auto"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_4d_cst_roundtrip_per_phase_files(fmt, segmentation, tmp_path):
    ct = _make_ct_4d()
    cst = _make_cst(ct, phased=True)
    out = tmp_path / "p"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_data(
            ct=ct,
            cst=cst,
            format=fmt,
            file_name=str(out),
            exporter_options={"segmentation": segmentation},
        )

    names = {p.name.split(".")[0] for p in out.iterdir()}
    assert {"cst_phase1", "cst_phase2", "cst_phase3"} <= names and "cst" not in names

    restored = load_data(out)
    assert restored["ct"].num_of_ct_scen == NUM_PHASES
    cst2 = restored["cst"]
    assert cst2.num_of_scenarios == NUM_PHASES
    if segmentation == "labelmap":
        # voxels shared with another VOI are lost; compare the phase-wise union instead
        for voi in cst2.vois:
            assert voi.mask.GetSize() == ct.cube_hu.GetSize()
        ptv = {v.name: v for v in cst2.vois}["PTV"]
        assert np.array_equal(
            sitk.GetArrayFromImage(ptv.mask), sitk.GetArrayFromImage(cst.vois[1].mask)
        )
    else:
        _assert_same_cst(cst, cst2, order_matters=True)


def test_3d_cst_on_4d_ct_roundtrip_is_repeated_on_all_phases(tmp_path):
    ct4d = _make_ct_4d()
    cst = _make_cst(ct4d)
    save_data(
        ct=ct4d,
        cst=cst,
        format="nifti",
        file_name=str(tmp_path / "p"),
        exporter_options={"segmentation": "layers"},
    )
    assert (tmp_path / "p" / "cst_phase3.nii.gz").exists()
    _assert_same_cst(cst, load_data(tmp_path / "p")["cst"], order_matters=True)

    ct3d = _make_ct_3d()
    save_data(
        ct=ct3d,
        cst=_make_cst(ct3d),
        format="nifti",
        file_name=str(tmp_path / "q"),
        exporter_options={"segmentation": "layers"},
    )
    cst_single = load_patient(tmp_path / "q")[1]
    ct4d_only = tmp_path / "q4"
    save_data(ct=ct4d, format="nifti", file_name=str(ct4d_only))
    for name in ("cst.nii.gz", "cst.json"):
        (ct4d_only / name).write_bytes((tmp_path / "q" / name).read_bytes())
    repeated = load_data(ct4d_only)["cst"]
    assert repeated.num_of_scenarios == NUM_PHASES
    for voi in repeated.vois:
        phases = voi.scenario_indices(order="numpy")
        assert all(np.array_equal(phases[0], p) for p in phases)
    assert len(repeated.vois) == len(cst_single.vois)


def test_4d_cst_phase_count_must_match_ct(tmp_path):
    ct = _make_ct_4d()
    save_data(ct=ct, cst=_make_cst(ct, phased=True), format="nifti", file_name=str(tmp_path / "p"))
    (tmp_path / "p" / "cst_phase3.nii.gz").unlink()
    with pytest.raises(ValueError, match="phases"):
        load_data(tmp_path / "p")
