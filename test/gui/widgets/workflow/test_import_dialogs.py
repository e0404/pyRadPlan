"""Tests for the folder-import dialogs and Load Folder routing."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import pytest

from pyRadPlan.gui.widgets.workflow._binary_import_dialog import BinaryImportDialog
from pyRadPlan.gui.widgets.workflow._dicom_import_dialog import DicomImportDialog
from pyRadPlan.gui.widgets.workflow._workflow_widget import _folder_has_images
from pyRadPlan.gui.widgets.workflow import WorkflowWidget
from pyRadPlan.gui.workspace import WorkspaceManager

DICOM_DIR = Path(__file__).parents[4] / "test" / "data" / "dicom"


def _write_image(path, size=(8, 8, 4), value=0, spacing=(2.0, 2.0, 3.0), dtype=np.uint8):
    arr = np.full(size[::-1], value, dtype=dtype)  # (z, y, x)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(path), useCompression=True)


@pytest.fixture
def binary_folder(tmp_path):
    _write_image(tmp_path / "patient_ct.nii.gz", value=0)
    structures = tmp_path / "structures"
    structures.mkdir()
    _write_image(structures / "PTV.nrrd", value=1)
    _write_image(structures / "Heart.nrrd", value=1)
    return tmp_path


# --------------------------------------------------------------------------
# Binary dialog
# --------------------------------------------------------------------------


def test_binary_dialog_prepopulates(qapp, binary_folder):
    dialog = BinaryImportDialog(str(binary_folder))
    assert dialog.ct_file().endswith("patient_ct.nii.gz")

    sels = dialog.selections()
    by_name = {s["name"]: s for s in sels}
    assert set(by_name) == {"PTV", "Heart"}
    assert by_name["PTV"]["voi_type"] == "TARGET"  # auto-classified
    assert by_name["Heart"]["voi_type"] == "OAR"


def test_binary_dialog_edit_name_and_ignore(qapp, binary_folder):
    dialog = BinaryImportDialog(str(binary_folder))
    # Row 0: rename; row 1: mark IGNORED via its combo.
    dialog._table.item(0, 1).setText("Tumor")
    dialog._table.cellWidget(1, 2).setCurrentText("IGNORED")

    sels = {s["name"] or s["path"]: s["voi_type"] for s in dialog.selections()}
    assert "Tumor" in sels
    assert any(v == "IGNORED" for v in sels.values())


# --------------------------------------------------------------------------
# DICOM dialog
# --------------------------------------------------------------------------


def test_dicom_dialog_populates_and_selects(qapp):
    from pyRadPlan.io.dicom import DicomImporter

    dialog = DicomImportDialog(DicomImporter(str(DICOM_DIR)))
    assert dialog._ct_combo.count() >= 1

    sel = dialog.selection()
    assert sel["series_uid"] is not None
    assert sel["struct_file"] is not None  # defaults to the first structure set
    assert sel["load_dose"] is True  # defaults to Auto
    assert sel["dose_file"] is None  # Auto => importer picks the plan physical dose


def test_dicom_dialog_uses_precomputed_catalog(qapp):
    """The dialog fills its combos from a scan done elsewhere (the worker thread)."""
    from pyRadPlan.gui.widgets.workflow._dicom_import_dialog import scan_folder
    from pyRadPlan.io.dicom import DicomImporter

    importer = DicomImporter(str(DICOM_DIR))
    catalog = scan_folder(importer)
    assert catalog["series"] and catalog["structs"] and catalog["doses"]

    scanned = DicomImportDialog(importer, catalog=catalog)
    reference = DicomImportDialog(DicomImporter(str(DICOM_DIR)))
    assert scanned.selection() == reference.selection()


def test_dicom_import_scans_in_worker_thread(qapp, monkeypatch):
    """Load Folder scans first (off the GUI thread), then opens the dialog."""
    from pyRadPlan.gui.widgets.workflow import _workflow_widget as wf

    w = WorkflowWidget(WorkspaceManager())
    started = []
    monkeypatch.setattr(
        wf.WorkflowWidget,
        "_run_in_thread",
        lambda self, fn, *a, **kw: started.append((fn, a, kw)),
    )

    w._open_dicom_import_dialog(str(DICOM_DIR))
    fn, args, kwargs = started.pop()
    assert kwargs["busy_text"] == "Scanning DICOM folder…"

    # The scan produces the catalog the dialog is then built from.
    catalog = fn(*args)
    assert {"series", "structs", "doses"} == set(catalog)
    assert catalog["series"]


def test_dicom_import_warns_without_ct_series(qapp, monkeypatch):
    from pyRadPlan.gui.widgets.workflow import _workflow_widget as wf
    from pyRadPlan.io.dicom import DicomImporter

    w = WorkflowWidget(WorkspaceManager())
    warned = []
    monkeypatch.setattr(wf.QMessageBox, "warning", lambda *a: warned.append(a))

    w._start_dicom_import(
        DicomImporter(str(DICOM_DIR)), {"series": [], "structs": [], "doses": []}
    )
    assert warned


# --------------------------------------------------------------------------
# Routing / merge helpers
# --------------------------------------------------------------------------


def test_folder_has_images_detects_subfolder(qapp, binary_folder):
    assert _folder_has_images(str(binary_folder))


def test_folder_has_images_false_when_empty(qapp, tmp_path):
    assert not _folder_has_images(str(tmp_path))


def test_image_import_dialog_no_ct_only_new(qapp, binary_folder):
    from pyRadPlan.gui.widgets.workflow._image_import_dialog import ImageImportDialog

    dialog = ImageImportDialog(str(binary_folder / "patient_ct.nii.gz"), has_ct=False)
    assert dialog._rb_ct_new.isChecked()
    assert not dialog._rb_ct_replace.isEnabled()
    assert not dialog._rb_structures.isEnabled()
    assert not dialog._rb_dose.isEnabled()
    assert dialog.selection() == {"mode": "ct_new"}


def test_image_import_dialog_modes_with_ct(qapp, binary_folder):
    import SimpleITK as sitk_local

    from pyRadPlan.gui.widgets.workflow._image_import_dialog import ImageImportDialog

    ct_image = sitk_local.ReadImage(str(binary_folder / "patient_ct.nii.gz"))
    path = str(binary_folder / "structures" / "PTV.nrrd")
    dialog = ImageImportDialog(path, has_ct=True, ct_image=ct_image)
    assert dialog._rb_ct_replace.isEnabled()

    dialog._rb_ct_replace.setChecked(True)
    sel = dialog.selection()
    assert sel["mode"] == "ct_replace"
    assert sel["grid_matches"] is True  # written on the same grid as the CT

    dialog._rb_dose.setChecked(True)
    sel = dialog.selection()
    assert sel == {"mode": "dose", "name": "PTV"}

    dialog._rb_structures.setChecked(True)
    assert dialog.selection() == {"mode": "structures"}


def test_image_import_dialog_grid_mismatch_flag(qapp, binary_folder, tmp_path):
    import SimpleITK as sitk_local

    from pyRadPlan.gui.widgets.workflow._image_import_dialog import ImageImportDialog

    other = tmp_path / "other_ct.nii.gz"
    _write_image(other, size=(6, 6, 3), spacing=(1.0, 1.0, 1.0))
    ct_image = sitk_local.ReadImage(str(binary_folder / "patient_ct.nii.gz"))

    dialog = ImageImportDialog(str(other), has_ct=True, ct_image=ct_image)
    dialog._rb_ct_replace.setChecked(True)
    assert dialog.selection()["grid_matches"] is False


def test_image_import_dialog_preselects_from_dtype(qapp, binary_folder, tmp_path):
    import SimpleITK as sitk_local

    from pyRadPlan.gui.widgets.workflow._image_import_dialog import ImageImportDialog

    ct_image = sitk_local.ReadImage(str(binary_folder / "patient_ct.nii.gz"))

    # Unsigned-integer mask -> structures preselected.
    dialog = ImageImportDialog(
        str(binary_folder / "structures" / "PTV.nrrd"), has_ct=True, ct_image=ct_image
    )
    assert dialog._rb_structures.isChecked()

    # Positive float -> dose preselected, name field visible.
    dose_file = tmp_path / "my_dose.nii.gz"
    _write_image(dose_file, value=1.5, dtype=np.float32)
    dialog = ImageImportDialog(str(dose_file), has_ct=True, ct_image=ct_image)
    assert dialog._rb_dose.isChecked()
    assert dialog._dose_name.isVisibleTo(dialog)

    # Negative values (HU) on the same grid -> CT replace preselected.
    hu_file = tmp_path / "another_ct.nii.gz"
    _write_image(hu_file, value=-1000, dtype=np.int16)
    dialog = ImageImportDialog(str(hu_file), has_ct=True, ct_image=ct_image)
    assert dialog._rb_ct_replace.isChecked()

    # Same HU file with no CT loaded -> falls back to new patient.
    dialog = ImageImportDialog(str(hu_file), has_ct=False)
    assert dialog._rb_ct_new.isChecked()


def test_unique_name_suffixes():
    from pyRadPlan.gui.widgets.workflow._workflow_widget import _unique_name

    assert _unique_name("Heart", set()) == "Heart"
    assert _unique_name("Heart", {"Heart"}) == "Heart_2"
    assert _unique_name("Heart", {"Heart", "Heart_2"}) == "Heart_3"


def test_merge_loaded_data_wraps_dose_into_result(qapp):
    ws = WorkspaceManager()
    w = WorkflowWidget(ws)
    dose = sitk.GetImageFromArray(np.zeros((4, 8, 8), dtype=np.float32))

    w._merge_loaded_data({"ct": object(), "dose": dose})
    assert ws.has("ct")
    assert isinstance(ws.result, dict) and "physical_dose" in ws.result
