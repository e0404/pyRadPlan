"""Tests for the DICOM import/export backend.

The bundled DICOM data (CT series, RTSTRUCT, RTDOSE) describes the same patient
as ``dicom_testData.mat`` (a matRad export), which lets us cross-validate the
DICOM importer against a known-good reference, and verify import/export
round-trips.
"""

import os
import shutil
from unittest import mock

import numpy as np
import pydicom
import SimpleITK as sitk
import pytest

from pyRadPlan.core import ProgressReport, observe_reports
from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.io import load_patient, load_data
from pyRadPlan.io.matlab import _matfile as matfile
from pyRadPlan.io.dicom import DicomImporter, DicomExporter, DicomHandler
from pyRadPlan.io.dicom._import_dose import import_dose
from pyRadPlan.io.dicom._helpers import determine_structure_type, generate_colors


def _voxel_sets(cst: StructureSet) -> dict:
    return {v.name.lower(): set(v.indices_numpy.tolist()) for v in cst.vois}


def _dice(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / (len(a) + len(b))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("PTV", "TARGET"),
        ("Target", "TARGET"),
        ("CTV_boost", "TARGET"),
        ("Body", "EXTERNAL"),
        ("External", "EXTERNAL"),
        ("Core", "OAR"),
        ("Lung_L", "OAR"),
    ],
)
def test_determine_structure_type(name, expected):
    assert determine_structure_type(name) == expected


def test_generate_colors():
    colors = generate_colors(5)
    assert len(colors) == 5
    assert all(len(c) == 3 and all(0 <= v <= 255 for v in c) for c in colors)


# --------------------------------------------------------------------------
# Import (cross-validated against the matRad reference)
# --------------------------------------------------------------------------


def test_import_ct_matches_reference(dicom_dir, dicom_reference_mat):
    ref_ct = validate_ct(matfile.load(dicom_reference_mat)["ct"])
    ct = DicomImporter(dicom_dir).load_ct()

    assert isinstance(ct, CT)
    assert ct.cube_hu.GetSize() == ref_ct.cube_hu.GetSize()
    assert np.allclose(ct.cube_hu.GetSpacing(), ref_ct.cube_hu.GetSpacing())
    assert np.allclose(ct.cube_hu.GetOrigin(), ref_ct.cube_hu.GetOrigin())
    assert np.allclose(
        sitk.GetArrayFromImage(ct.cube_hu),
        sitk.GetArrayFromImage(ref_ct.cube_hu),
        atol=1e-4,
    )


def test_import_cst_matches_reference(dicom_dir, dicom_reference_mat):
    m = matfile.load(dicom_reference_mat)
    ref_ct = validate_ct(m["ct"])
    ref_cst = validate_cst(m["cst"], ref_ct)

    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)

    assert isinstance(cst, StructureSet)

    sets = _voxel_sets(cst)
    ref_sets = _voxel_sets(ref_cst)
    assert set(sets) == set(ref_sets)
    for name, voxels in ref_sets.items():
        # The RTSTRUCT contour->mask conversion must reproduce the reference voxels exactly.
        assert voxels == sets[name], f"VOI '{name}' voxels differ from reference"


def test_import_dose_matches_reference(dicom_dir, dicom_reference_mat):
    ref_dose = np.asarray(matfile.load(dicom_reference_mat)["resultGUI"]["physicalDose"])
    ref_zyx = np.transpose(ref_dose, (2, 0, 1))  # matRad (y,x,z) -> sitk (z,y,x)

    dose = import_dose(str(dicom_dir / "RTDose_6_physicalDose.dcm"))
    assert isinstance(dose, sitk.Image)
    arr = sitk.GetArrayFromImage(dose)
    assert arr.shape == ref_zyx.shape
    assert np.allclose(arr, ref_zyx, atol=1e-3)


def test_list_ct_series(dicom_dir):
    series = DicomImporter(dicom_dir).list_ct_series()
    assert len(series) >= 1
    entry = series[0]
    assert entry["num_slices"] == len(entry["files"])
    assert entry["series_uid"]


def test_list_structure_sets(dicom_dir):
    sets = DicomImporter(dicom_dir).list_structure_sets()
    assert len(sets) >= 1
    # The RTSTRUCT enumerates its ROI names.
    rtstructs = [s for s in sets if s["modality"] == "RTSTRUCT"]
    assert rtstructs
    names = {n for s in rtstructs for n in s["structure_names"]}
    assert {"Target", "Body", "Core"} <= names


def test_list_doses_and_selective_load(dicom_dir):
    importer = DicomImporter(dicom_dir)
    doses = importer.list_doses()
    # matRad exports per-beam / LET distributions too, so there are several.
    assert len(doses) > 1
    assert all("summation" in d and "description" in d for d in doses)

    # Explicitly loading a chosen file returns exactly that distribution.
    physical = next(d for d in doses if d["path"].endswith("RTDose_6_physicalDose.dcm"))
    dose = importer.load_dose(dose_file=physical["path"])
    assert isinstance(dose, sitk.Image)


def test_load_patient_from_dicom_folder(dicom_dir):
    ct, cst = load_patient(dicom_dir)
    assert isinstance(ct, CT)
    assert isinstance(cst, StructureSet)
    assert {v.name for v in cst.vois} == {"Target", "Body", "Core"}


def test_load_data_from_dicom_folder(dicom_dir, dicom_reference_mat):
    data = load_data(dicom_dir)
    assert isinstance(data["ct"], CT)
    assert isinstance(data["cst"], StructureSet)
    assert isinstance(data["dose"], sitk.Image)

    # The folder holds per-beam doses and LET cubes (all modality RTDOSE);
    # load_data must select the plan-level physical dose, not an arbitrary one.
    ref_dose = np.asarray(matfile.load(dicom_reference_mat)["resultGUI"]["physicalDose"])
    ref_zyx = np.transpose(ref_dose, (2, 0, 1))
    arr = sitk.GetArrayFromImage(data["dose"])
    assert arr.shape == ref_zyx.shape
    assert np.allclose(arr, ref_zyx, atol=1e-3)


def test_dicom_import_reports_progress(dicom_dir):
    """A folder import reports nested, determinate progress for each step."""
    reports = []

    with observe_reports(lambda r: reports.append(r)):
        load_data(dicom_dir)

    stacks = [
        tuple((lvl.name, lvl.total) for lvl in r.levels)
        for r in reports
        if isinstance(r, ProgressReport) and r.levels
    ]
    assert stacks

    # Every step nests under the overall "Importing" level.
    nested = {s for s in stacks if len(s) > 1}
    assert {
        (("Importing", 3), ("Scanning files", 22)),
        (("Importing", 3), ("Reading CT slices", 10)),
        (("Importing", 3), ("Structure", 3)),
        (("Importing", 3), ("Reading dose cube", 1)),
    } <= nested

    # Every level is determinate, so a consumer can drive a real progress bar.
    assert all(lvl.total for r in reports if isinstance(r, ProgressReport) for lvl in r.levels)

    # The outer level runs to completion.
    outer = [
        r.levels[0]
        for r in reports
        if isinstance(r, ProgressReport) and r.levels and r.levels[0].name == "Importing"
    ]
    assert outer[-1].current == outer[-1].total == 3


def test_dicom_importer_caches_headers(dicom_dir):
    """The folder is scanned once; the listings reuse the cached headers."""
    importer = DicomImporter(dicom_dir)
    assert not importer._headers  # nothing read before the first classification

    series = importer.list_ct_series()
    scanned = dict(importer._headers)
    assert scanned

    with mock.patch("pydicom.dcmread", side_effect=AssertionError("re-read the headers")):
        assert importer.list_ct_series() == series
        importer.list_structure_sets()
        importer.list_doses()


# --------------------------------------------------------------------------
# Export / round-trip
# --------------------------------------------------------------------------


def test_dicom_roundtrip_ct(dicom_dir, tmp_path):
    ct = DicomImporter(dicom_dir).load_ct()
    DicomExporter(tmp_path).save(ct=ct)

    ct2 = DicomImporter(tmp_path).load_ct()
    assert ct2.cube_hu.GetSize() == ct.cube_hu.GetSize()
    assert np.allclose(ct2.cube_hu.GetSpacing(), ct.cube_hu.GetSpacing())
    assert np.allclose(ct2.cube_hu.GetOrigin(), ct.cube_hu.GetOrigin())
    assert np.allclose(
        sitk.GetArrayFromImage(ct2.cube_hu), sitk.GetArrayFromImage(ct.cube_hu), atol=1e-3
    )


def test_dicom_roundtrip_dose(dicom_dir, tmp_path):
    dose = import_dose(str(dicom_dir / "RTDose_6_physicalDose.dcm"))
    DicomExporter(tmp_path).save(ct=DicomImporter(dicom_dir).load_ct(), dose=dose)

    dose2 = DicomImporter(tmp_path).load_dose()
    a = sitk.GetArrayFromImage(dose)
    b = sitk.GetArrayFromImage(dose2)
    assert b.shape == a.shape
    # uint32 quantization -> values reproduced to high precision.
    assert np.allclose(a, b, atol=1e-4 * max(a.max(), 1.0))


def test_dicom_roundtrip_structures(dicom_dir, tmp_path):
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)

    DicomHandler(tmp_path).save(ct=ct, cst=cst)

    cst2 = DicomImporter(tmp_path).load_cst()
    sets = _voxel_sets(cst)
    sets2 = _voxel_sets(cst2)
    assert set(sets) == set(sets2)
    for name, voxels in sets.items():
        # Contour extraction then re-rasterization is near-lossless (boundary voxels).
        assert _dice(voxels, sets2[name]) > 0.95, f"VOI '{name}' degraded in roundtrip"


def test_dicom_rtstruct_references_ct(dicom_dir, tmp_path):
    # The RTSTRUCT must reference the exported CT series and slices so external
    # viewers can associate structures with the CT (not just via frame of reference).
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)

    DicomHandler(tmp_path).save(ct=ct, cst=cst)

    rtstruct = pydicom.dcmread(tmp_path / "RTSTRUCT.dcm")
    ref_series = (
        rtstruct.ReferencedFrameOfReferenceSequence[0]
        .RTReferencedStudySequence[0]
        .RTReferencedSeriesSequence[0]
    )
    ct_slices = [pydicom.dcmread(f, stop_before_pixels=True) for f in tmp_path.glob("CT_*.dcm")]
    ct_series_uids = {s.SeriesInstanceUID for s in ct_slices}
    ct_sop_uids = {s.SOPInstanceUID for s in ct_slices}

    assert ref_series.SeriesInstanceUID in ct_series_uids
    referenced = {img.ReferencedSOPInstanceUID for img in ref_series.ContourImageSequence}
    assert referenced == ct_sop_uids


def test_dicom_seg_is_binary_packed(dicom_dir, tmp_path):
    # SegmentationType BINARY requires 1-bit packed pixel data; verify the file is
    # conformant and that pydicom still unpacks it to the expected frame shape.
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)

    DicomExporter(tmp_path, structure_format="seg").save(ct=ct, cst=cst)

    seg = pydicom.dcmread(tmp_path / "SEG.dcm")
    assert seg.BitsAllocated == 1
    assert seg.pixel_array.shape == (seg.NumberOfFrames, seg.Rows, seg.Columns)
    assert set(np.unique(seg.pixel_array)) <= {0, 1}


def test_dicom_roundtrip_structures_as_seg(dicom_dir, tmp_path):
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)

    # SEG stores the mask directly, so the roundtrip is exact (unlike RTSTRUCT contours).
    DicomExporter(tmp_path, structure_format="seg").save(ct=ct, cst=cst)
    assert (tmp_path / "SEG.dcm").exists()

    cst2 = DicomImporter(tmp_path).load_cst()
    sets = _voxel_sets(cst)
    sets2 = _voxel_sets(cst2)
    assert set(sets) == set(sets2)
    for name, voxels in sets.items():
        assert voxels == sets2[name], f"VOI '{name}' changed in SEG roundtrip"


def test_dicom_exporter_rejects_bad_structure_format(tmp_path):
    with pytest.raises(ValueError):
        DicomExporter(tmp_path, structure_format="bogus")


def test_dicom_full_roundtrip_via_handler(dicom_dir, tmp_path):
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)
    dose = import_dose(str(dicom_dir / "RTDose_6_physicalDose.dcm"))

    handler = DicomHandler(tmp_path)
    handler.save(ct=ct, cst=cst, dose=dose)

    data = DicomImporter(tmp_path).load_data()
    assert isinstance(data["ct"], CT)
    assert isinstance(data["cst"], StructureSet)
    assert isinstance(data["dose"], sitk.Image)


def test_dicom_handler_honors_structure_format(dicom_dir, tmp_path):
    # The handler must forward structure_format to its exporter half (not silently
    # fall back to the rtstruct default from the cooperative __init__ chain).
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)

    DicomHandler(tmp_path, structure_format="seg").save(ct=ct, cst=cst)
    assert (tmp_path / "SEG.dcm").exists()


def test_dicom_exporter_requires_data(tmp_path):
    with pytest.raises(ValueError):
        DicomExporter(tmp_path).save()


# --------------------------------------------------------------------------
# Multiple structure sets in one folder: warn and use the first (no merging)
# --------------------------------------------------------------------------


def test_load_cst_multiple_structure_sets_uses_first(dicom_dir, tmp_path):
    # Copy the CT series and the RTSTRUCT, then add a second RTSTRUCT (fresh UIDs).
    rtstruct = None
    for f in os.listdir(dicom_dir):
        if f.startswith("ct_slice") or f == "RTstruct.dcm":
            shutil.copy(dicom_dir / f, tmp_path / f)
            if f == "RTstruct.dcm":
                rtstruct = tmp_path / f

    ds = pydicom.dcmread(rtstruct)
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.save_as(tmp_path / "RTstruct2.dcm", enforce_file_format=True)

    importer = DicomImporter(tmp_path)
    ct = importer.load_ct()
    with pytest.warns(UserWarning, match="Multiple structure sets"):
        cst = importer.load_cst(ct)

    # One structure set is loaded (3 VOIs), not both merged (6).
    assert [v.name for v in cst.vois] == ["Target", "Body", "Core"]


# --------------------------------------------------------------------------
# Contour rasterization
# --------------------------------------------------------------------------


def _rasterization_ct(num_x=64, num_y=48, num_z=3) -> CT:
    """A small CT grid to rasterize synthetic contours into."""
    arr = np.zeros((num_z, num_y, num_x), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.5, 2.0, 3.0))
    img.SetOrigin((-20.0, -30.0, -5.0))
    return validate_ct(cube_hu=img)


def _full_slice_mask(vertices, ct):
    """Reference rasterization: fill the whole slice, without the bounding-box window."""
    from pyRadPlan.io.dicom._import_cst import _fill_polygon

    return _fill_polygon(vertices, 0, ct.cube_dim[0], 0, ct.cube_dim[1])


@pytest.mark.parametrize(
    "polygon_world",
    [
        # Well inside the grid, so the window is a small sub-box of the slice.
        [(-10.0, -20.0), (0.0, -20.0), (0.0, -10.0), (-10.0, -10.0)],
        # Straddling the grid edge: the window must be clipped, not go out of bounds.
        [(-25.0, -35.0), (-10.0, -35.0), (-10.0, -20.0), (-25.0, -20.0)],
        # A concave (L-shaped) contour, where the bounding box is far from tight.
        [
            (-18.0, -28.0),
            (-2.0, -28.0),
            (-2.0, -22.0),
            (-10.0, -22.0),
            (-10.0, -8.0),
            (-18.0, -8.0),
        ],
        # Entirely outside the grid.
        [(500.0, 500.0), (510.0, 500.0), (510.0, 510.0), (500.0, 510.0)],
        # Extremes exactly on voxel centres (integer voxel indices), which is where
        # the boundary-inclusion rule decides whether the edge voxels are in or out.
        [(-20.0, -30.0), (-5.0, -30.0), (-5.0, -10.0), (-20.0, -10.0)],
        # A single-voxel-wide sliver, whose window has an extent of one in x.
        [(-11.0, -26.0), (-10.4, -26.0), (-10.4, -14.0), (-11.0, -14.0)],
    ],
)
def test_contour_window_matches_full_slice(polygon_world):
    """Rasterizing only the contour's bounding box must equal testing the whole slice.

    The window is a pure optimization (a voxel outside the contour's bounding box
    cannot be inside the contour), so it must not change a single voxel.
    """
    from pyRadPlan.io.dicom._import_cst import _axis_interpolators, _create_slice_mask

    ct = _rasterization_ct()
    x_interp, y_interp = _axis_interpolators(ct)

    z_pos = float(ct.z[1])
    points = np.array([[x, y, z_pos] for x, y in polygon_world])
    window, x_start, y_start, _ = _create_slice_mask(points, ct, x_interp, y_interp)

    vertices = np.column_stack((x_interp(points[:, 0]), y_interp(points[:, 1])))
    expected = _full_slice_mask(vertices, ct)

    actual = np.zeros(ct.cube_dim[:2], dtype=bool)
    if window is not None:
        actual[x_start : x_start + window.shape[0], y_start : y_start + window.shape[1]] = window

    assert np.array_equal(actual, expected)


def test_contour_with_non_finite_vertices_is_skipped():
    """A contour that cannot be placed on the grid is dropped, not raised over."""
    from pyRadPlan.io.dicom._import_cst import _axis_interpolators, _create_slice_mask

    ct = _rasterization_ct()
    x_interp, y_interp = _axis_interpolators(ct)

    z_pos = float(ct.z[1])
    points = np.array([[np.nan, 0.0, z_pos], [0.0, 0.0, z_pos], [0.0, 5.0, z_pos]])
    window, _, _, _ = _create_slice_mask(points, ct, x_interp, y_interp)

    assert window is None


# --------------------------------------------------------------------------
# The contour fill rule, on a grid small enough to read
# --------------------------------------------------------------------------
#
# There is no single "correct" rasterization of a contour onto voxels: every
# tool picks a rule for voxel centres landing exactly on the contour. pyRadPlan
# fills a voxel when its *centre* lies inside the polygon, counting a centre
# exactly on the boundary as inside.
#
# The end-to-end consequences of that rule are covered by
# test_import_cst_matches_reference, which requires the bundled RTSTRUCT to
# reproduce a matRad export voxel for voxel (and half of that phantom's contour
# coordinates sit exactly on voxel centres, so it exercises the ties). The tests
# here pin the rule itself on a grid small enough to write the answer out by
# hand, so a future change to the fill routine says which case it broke.
#
# The same conversion was also checked against label maps exported from MITK
# Workbench for a 512x512x297 CT: the two agreed to 3 voxels out of 17.6 million
# across four structures, with two of them voxel-identical. That data is too
# large to vendor, hence these miniature stand-ins.


def _unit_ct(num_x=8, num_y=8, num_z=3) -> CT:
    """A CT whose voxel indices equal its world coordinates, so masks read literally."""
    arr = np.zeros((num_z, num_y, num_x), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return validate_ct(cube_hu=img)


def _drawn(rows):
    """Parse an ASCII picture into a boolean mask indexed ``[x, y]``."""
    return np.array([[char == "#" for char in row.split()] for row in rows], dtype=bool)


def _fill(vertices, size=8):
    from pyRadPlan.io.dicom._import_cst import _fill_polygon

    return _fill_polygon(np.asarray(vertices, dtype=float), 0, size, 0, size)


def test_fill_polygon_off_lattice_rectangle():
    """The unambiguous case: centres strictly inside, no ties to break."""
    # Corners at x,y = 2.5 and 5.5, so the centres inside are 3, 4 and 5 on both axes.
    mask = _fill([(2.5, 2.5), (5.5, 2.5), (5.5, 5.5), (2.5, 5.5)])
    assert np.array_equal(
        mask,
        _drawn(
            [
                ". . . . . . . .",
                ". . . . . . . .",
                ". . . . . . . .",
                ". . . # # # . .",
                ". . . # # # . .",
                ". . . # # # . .",
                ". . . . . . . .",
                ". . . . . . . .",
            ]
        ),
    )


def test_fill_polygon_on_lattice_rectangle_keeps_inherited_tie_break():
    """Corners exactly on voxel centres: the tie-break is asymmetric between the axes.

    Both x boundaries are kept, only the upper y boundary is. That asymmetry is
    inherited from the per-voxel ``matplotlib`` test this replaced, and is what
    keeps the bundled matRad phantom (half of whose coordinates are on voxel
    centres) importing unchanged. It is pinned here so it cannot drift silently.
    """
    mask = _fill([(2, 2), (5, 2), (5, 5), (2, 5)])
    assert np.array_equal(
        mask,
        _drawn(
            [
                ". . . . . . . .",
                ". . . . . . . .",
                ". . . # # # . .",
                ". . . # # # . .",
                ". . . # # # . .",
                ". . . # # # . .",
                ". . . . . . . .",
                ". . . . . . . .",
            ]
        ),
    )
    assert mask[:, 2].sum() == 0, "the lower y boundary is excluded"
    assert mask[2].any() and mask[5].any(), "both x boundaries are included"


def test_fill_polygon_keeps_structure_narrower_than_a_voxel():
    """A contour thinner than one voxel must still produce voxels, not vanish.

    Requiring the centre to be *strictly* inside would empty this structure
    entirely, silently dropping small VOIs (e.g. thin applicators) on import.
    """
    mask = _fill([(2.0, 1.0), (2.4, 1.0), (2.4, 6.0), (2.0, 6.0)])
    assert mask.any(), "a sub-voxel-wide contour was rasterized away"
    assert np.array_equal(np.flatnonzero(mask.any(axis=1)), [2])
    assert np.array_equal(np.flatnonzero(mask.any(axis=0)), [2, 3, 4, 5, 6])


def test_fill_polygon_concave_contour():
    """A concave outline must not be filled as its convex hull."""
    mask = _fill([(1, 1), (6, 1), (6, 3), (3, 3), (3, 6), (1, 6)])
    assert np.array_equal(
        mask,
        _drawn(
            [
                ". . . . . . . .",
                ". . # # # # # .",
                ". . # # # # # .",
                ". . # # # # # .",
                ". . # # . . . .",
                ". . # # . . . .",
                ". . # # . . . .",
                ". . . . . . . .",
            ]
        ),
    )


def test_fill_polygon_clips_to_the_window():
    """A contour reaching past the grid fills up to the edge, without wrapping."""
    mask = _fill([(-4.5, -4.5), (3.5, -4.5), (3.5, 3.5), (-4.5, 3.5)])
    assert np.array_equal(np.flatnonzero(mask.any(axis=1)), [0, 1, 2, 3])
    assert np.array_equal(np.flatnonzero(mask.any(axis=0)), [0, 1, 2, 3])
    assert not mask[4:].any() and not mask[:, 4:].any()


def test_fill_polygon_outside_the_window_is_empty():
    assert not _fill([(20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)]).any()


def test_fill_polygon_degenerate_contours_are_empty_not_an_error():
    """Zero-area contours must rasterize to nothing rather than raise."""
    assert not _fill([(3.0, 3.0), (3.0, 3.0), (3.0, 3.0)]).any()  # a point
    assert not _fill([(1.5, 3.5), (5.5, 3.5), (1.5, 3.5)]).any()  # a line, there and back


def test_compute_segment_mask_ors_contours_on_the_same_slice():
    """Two contours on one slice are combined, and land on the slice they name."""
    from pyRadPlan.io.dicom._import_cst import _axis_interpolators, _compute_segment_mask

    ct = _unit_ct()
    z_pos = float(ct.z[1])

    def contour(x0, x1, y0, y1):
        item = mock.Mock()
        item.ContourGeometricType = "CLOSED_PLANAR"
        item.ContourData = [
            x0,
            y0,
            z_pos,
            x1,
            y0,
            z_pos,
            x1,
            y1,
            z_pos,
            x0,
            y1,
            z_pos,
        ]
        return item

    roi = mock.Mock()
    roi.ContourSequence = [contour(1.5, 3.5, 1.5, 3.5), contour(4.5, 6.5, 4.5, 6.5)]

    cube = _compute_segment_mask(ct, roi, _axis_interpolators(ct))

    assert cube[:, :, 0].sum() == 0 and cube[:, :, 2].sum() == 0, "only slice 1 is touched"
    filled = cube[:, :, 1]
    assert filled[2, 2] and filled[3, 3], "first contour"
    assert filled[5, 5] and filled[6, 6], "second contour"
    assert filled.sum() == 8, "both contours, nothing between them"


# --------------------------------------------------------------------------
# Dose grid handling
# --------------------------------------------------------------------------
#
# RTDOSE is stored on its own grid: typically coarser than the CT and covering
# only the irradiated region. The importer resamples it onto the CT grid,
# because everything downstream indexes quantities with CT voxel indices - the
# viewer overlays a dose slice on the CT slice of the same index, so an
# unresampled cube is drawn at the wrong scale and raises IndexError as soon as
# the index passes the dose cube's extent.


def _offset_dose_cube(ct, value=2.0):
    """A coarse dose cube covering only part of *ct*, on its own grid."""
    reference = ct.cube_hu
    spacing = tuple(2.5 * s for s in reference.GetSpacing())
    size = tuple(max(2, n // 4) for n in reference.GetSize())
    origin = tuple(o + 2 * s for o, s in zip(reference.GetOrigin(), reference.GetSpacing()))

    cube = sitk.Image(size, sitk.sitkFloat32)
    cube.SetSpacing(spacing)
    cube.SetOrigin(origin)
    cube.SetDirection(reference.GetDirection())
    cube += value
    return cube


def test_load_dose_is_resampled_onto_the_ct_grid(dicom_dir, monkeypatch):
    """A dose on its own grid must come back on the CT grid, not as-is."""
    from pyRadPlan.io.dicom import _importer as importer_module

    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cube = _offset_dose_cube(ct)
    assert cube.GetSize() != ct.cube_hu.GetSize(), "the fixture must actually differ"

    monkeypatch.setattr(importer_module, "import_dose", lambda *a, **k: cube)
    dose = importer.load_dose()

    assert dose.GetSize() == ct.cube_hu.GetSize()
    assert np.allclose(dose.GetSpacing(), ct.cube_hu.GetSpacing())
    assert np.allclose(dose.GetOrigin(), ct.cube_hu.GetOrigin())
    assert np.allclose(dose.GetDirection(), ct.cube_hu.GetDirection())


def test_load_dose_keeps_values_and_leaves_uncovered_voxels_at_zero(dicom_dir, monkeypatch):
    """Resampling preserves the dose where the cube reaches, and invents none where it does not."""
    from pyRadPlan.io.dicom import _importer as importer_module

    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cube = _offset_dose_cube(ct, value=2.0)

    monkeypatch.setattr(importer_module, "import_dose", lambda *a, **k: cube)
    values = sitk.GetArrayFromImage(importer.load_dose())

    assert np.isfinite(values).all()
    # Inside the cube the constant value survives; outside it must stay zero
    # rather than being nearest-neighbour smeared across the rest of the patient.
    assert np.isclose(values.max(), 2.0)
    assert np.isclose(values.min(), 0.0)
    assert (values > 0).any() and (values == 0).any()


def test_load_dose_on_the_ct_grid_is_not_resampled(dicom_dir):
    """The bundled RTDOSE already shares the CT grid, so it must pass through untouched."""
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    dose = importer.load_dose()

    assert dose.GetSize() == ct.cube_hu.GetSize()
    assert np.allclose(dose.GetOrigin(), ct.cube_hu.GetOrigin())


def test_load_data_dose_shares_the_ct_grid(dicom_dir):
    """The bulk loader must hand out a dose the viewer can index with CT indices."""
    data = load_data(dicom_dir)
    ct, dose = data["ct"], data["dose"]

    assert dose.GetSize() == ct.cube_hu.GetSize()
    # An index valid for the CT must be valid for the dose, on every axis.
    array = sitk.GetArrayFromImage(dose)
    assert array.shape == sitk.GetArrayFromImage(ct.cube_hu).shape
