"""File import and export of a session (protocol v1 ``data.*`` and ``export.save``).

Mirrors the load and save paths of ``gui/widgets/workflow/_workflow_widget.py``, with
bytes instead of paths: uploads are written to a temporary file and read with
:func:`pyRadPlan.io.load_data`; exports are written to a temporary folder and returned
as one file (several files and folders zipped). Which formats work depends on the
runtime's SimpleITK: formats needing ``sitk.WriteImage`` / ``sitk.ReadImage`` are
reported unavailable (with the reason) when it lacks them, e.g. under the Pyodide
SimpleITK stand-in.
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ._session import Operation, PreconditionError
from ._results import result_components
from ._views import result_entry, result_quantities

MIME_MAT = "application/x-matlab-data"
MIME_BINARY = "application/octet-stream"
MIME_ZIP = "application/zip"

#: matRad ``resultGUI`` field names of the quantities that have one.
MATRAD_RESULT_NAMES = {"physical_dose": "physicalDose", "rbe_x_dose": "RBExDose", "let": "LET"}

#: The ``kind`` values of ``data.load_file`` reserved for image files.
IMAGE_LOAD_KINDS = ("ct_new", "ct_replace", "structures", "dose")

_ALL_OBJECTS = ["ct", "cst", "pln", "stf", "dij", "result"]


# ----------------------------------------------------------------------
# Runtime support
# ----------------------------------------------------------------------


def _sitk() -> Any:
    import SimpleITK as sitk  # noqa: PLC0415

    return sitk


def is_sitk_shim() -> bool:
    """Whether ``SimpleITK`` is the numpy/scipy stand-in of the Pyodide runtime."""
    return bool(getattr(_sitk(), "IS_PYRADPLAN_SHIM", False))


def _probe_write(sitk: Any, folder: str) -> None:
    sitk.WriteImage(sitk.Image([2, 2, 2], sitk.sitkUInt8), os.path.join(folder, "probe.mha"))


def _probe_read(sitk: Any, folder: str) -> None:
    sitk.ReadImage(os.path.join(folder, "missing.mha"))


def _probe_series(sitk: Any, folder: str) -> None:
    sitk.ImageSeriesReader()


_SITK_PROBES = {
    "WriteImage": _probe_write,
    "ReadImage": _probe_read,
    "ImageSeriesReader": _probe_series,
}

#: Probe results by (SimpleITK module, function name).
_SITK_SUPPORT: dict[tuple[int, str], bool] = {}


def sitk_supports(name: str) -> bool:
    """Whether the runtime's SimpleITK implements the file IO function *name*.

    Probed by calling it once (on a temporary folder): a stand-in that defines the name
    but raises :class:`NotImplementedError` does not count, any other outcome (e.g. a
    missing file) does.
    """
    sitk = _sitk()
    key = (id(sitk), name)
    if key not in _SITK_SUPPORT:
        supported = hasattr(sitk, name)
        if supported and name in _SITK_PROBES:
            folder = tempfile.mkdtemp(prefix="pyradplan-probe-")
            try:
                _SITK_PROBES[name](sitk, folder)
            except NotImplementedError:
                supported = False
            except Exception:  # noqa: BLE001, S110 - implemented, the probe input is just bad
                pass
            finally:
                shutil.rmtree(folder, ignore_errors=True)
        _SITK_SUPPORT[key] = supported
    return _SITK_SUPPORT[key]


def sitk_missing(*names: str) -> Optional[str]:
    """Why the runtime's SimpleITK lacks the functions *names*, ``None`` if it has them."""
    missing = [name for name in names if not sitk_supports(name)]
    if not missing:
        return None
    which = "the SimpleITK stand-in of this runtime" if is_sitk_shim() else "SimpleITK"
    return f"Needs SimpleITK file IO ({', '.join(missing)}), which {which} does not provide"


def module_missing(*modules: str) -> Optional[str]:
    """Why one of the Python *modules* cannot be used, ``None`` if all are installed."""
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if not missing:
        return None
    return f"Needs the Python package(s) {', '.join(missing)}, which are not installed"


def image_extensions() -> tuple[str, ...]:
    """Return the endings of bare image files (Qt opens the Import Image dialog for them)."""
    from pyRadPlan.io.sitk_based._binary_import import IMAGE_EXTENSIONS  # noqa: PLC0415

    return IMAGE_EXTENSIONS


# ----------------------------------------------------------------------
# Format tables
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExportFormat:
    """One entry of the export table (``capabilities.formats.export``)."""

    object: str
    format: str
    label: str
    extension: str
    mime: str
    archive: bool = False
    keeps_objectives: Optional[bool] = None
    structure_format: Optional[str] = None
    note: Optional[str] = None
    needs_sitk: tuple[str, ...] = ()
    needs_modules: tuple[str, ...] = ()

    def reason(self) -> Optional[str]:
        """Why the format cannot be written in this runtime, ``None`` if it can."""
        return sitk_missing(*self.needs_sitk) or module_missing(*self.needs_modules)

    def describe(self) -> dict[str, Any]:
        reason = self.reason()
        entry = {
            "object": self.object,
            "format": self.format,
            "label": self.label,
            "extension": self.extension,
            "mime": self.mime,
            "available": reason is None,
            "reason": reason,
            "archive": self.archive,
        }
        if self.keeps_objectives is not None:
            entry["keeps_objectives"] = self.keeps_objectives
        if self.structure_format is not None:
            entry["structure_format"] = self.structure_format
        note = self.note
        if self.object == "workspace" and _pickles_runtime_specific():
            note = (
                "Workspace pickles of this runtime do not load in native pyRadPlan and "
                "vice versa (the SimpleITK stand-in pickles its images differently)"
            )
        if note:
            entry["note"] = note
        return entry


def _pickles_runtime_specific() -> bool:
    sitk = _sitk()
    return is_sitk_shim() and getattr(sitk.Image, "__module__", "") != "SimpleITK.SimpleITK"


_WRITE = ("WriteImage",)
_SEVERAL_ZIPPED = "Several quantities are zipped, one file each"

EXPORT_FORMATS: tuple[ExportFormat, ...] = (
    ExportFormat("pln", "mat", "MATLAB (*.mat)", ".mat", MIME_MAT),
    ExportFormat("pln", "pickle", "Pickle (*.pkl)", ".pkl", MIME_BINARY),
    ExportFormat("dij", "mat", "MATLAB (*.mat)", ".mat", MIME_MAT),
    ExportFormat("dij", "pickle", "Pickle (*.pkl)", ".pkl", MIME_BINARY),
    ExportFormat(
        "cst",
        "mat",
        "MATLAB (*.mat)",
        ".mat",
        MIME_MAT,
        keeps_objectives=True,
        note="Loading it again needs the CT (a cst-only .mat file does not load)",
    ),
    ExportFormat("cst", "pickle", "Pickle (*.pkl)", ".pkl", MIME_BINARY, keeps_objectives=True),
    ExportFormat(
        "cst",
        "npz",
        "NumPy (*.npz)",
        ".npz",
        MIME_BINARY,
        keeps_objectives=False,
        note="Includes the CT",
    ),
    ExportFormat(
        "cst",
        "nifti",
        "NIfTI label map (zip)",
        ".zip",
        MIME_ZIP,
        archive=True,
        keeps_objectives=False,
        needs_sitk=_WRITE,
    ),
    ExportFormat(
        "cst",
        "nrrd",
        "NRRD label map (zip)",
        ".zip",
        MIME_ZIP,
        archive=True,
        keeps_objectives=False,
        needs_sitk=_WRITE,
    ),
    ExportFormat(
        "cst",
        "meta",
        "MetaImage label map (zip)",
        ".zip",
        MIME_ZIP,
        archive=True,
        keeps_objectives=False,
        needs_sitk=_WRITE,
    ),
    ExportFormat(
        "cst",
        "dcm",
        "DICOM RTSTRUCT (zip)",
        ".zip",
        MIME_ZIP,
        archive=True,
        keeps_objectives=False,
        structure_format="rtstruct",
        note="With the CT series",
        needs_modules=("pydicom", "contourpy"),
    ),
    ExportFormat(
        "cst",
        "dcm",
        "DICOM SEG (zip)",
        ".zip",
        MIME_ZIP,
        archive=True,
        keeps_objectives=False,
        structure_format="seg",
        note="With the CT series",
        needs_modules=("pydicom",),
    ),
    ExportFormat(
        "result",
        "npz",
        "NumPy (*.npz)",
        ".npz",
        MIME_BINARY,
        note=f"{_SEVERAL_ZIPPED}; a file loads back as the physical dose of the current patient",
    ),
    ExportFormat(
        "result",
        "mat",
        "MATLAB resultGUI (*.mat)",
        ".mat",
        MIME_MAT,
        note="All selected quantities in one resultGUI; loading it again needs the CT",
    ),
    ExportFormat(
        "result",
        "dcm",
        "DICOM RTDOSE (zip)",
        ".zip",
        MIME_ZIP,
        archive=True,
        note="One RTDOSE per quantity",
        needs_modules=("pydicom",),
    ),
    ExportFormat(
        "result",
        "nifti",
        "NIfTI (*.nii.gz)",
        ".nii.gz",
        MIME_BINARY,
        note=_SEVERAL_ZIPPED,
        needs_sitk=_WRITE,
    ),
    ExportFormat(
        "result",
        "nrrd",
        "NRRD (*.nrrd)",
        ".nrrd",
        MIME_BINARY,
        note=_SEVERAL_ZIPPED,
        needs_sitk=_WRITE,
    ),
    ExportFormat(
        "result",
        "meta",
        "MetaImage (*.mha)",
        ".mha",
        MIME_BINARY,
        note=_SEVERAL_ZIPPED,
        needs_sitk=_WRITE,
    ),
    ExportFormat("workspace", "pickle", "pyRadPlan workspace (*.pkl)", ".pkl", MIME_BINARY),
)


def import_formats() -> list[dict[str, Any]]:
    """Return the import table (``capabilities.formats.import``)."""
    read = sitk_missing("ReadImage")
    entries = [
        ("mat", "MATLAB / matRad (*.mat)", [".mat"], _ALL_OBJECTS, None, None),
        ("npz", "NumPy (*.npz)", [".npz"], ["ct", "cst", "dose"], None, None),
        (
            "pickle",
            "pyRadPlan workspace (*.pkl)",
            [".pkl", ".pickle"],
            [*_ALL_OBJECTS, "dose"],
            None,
            "Unpickling runs code from the file: load only files you trust",
        ),
        (
            "nifti",
            "NIfTI image (*.nii, *.nii.gz)",
            [".nii", ".nii.gz"],
            ["ct", "cst", "dose"],
            read,
            None,
        ),
        (
            "nrrd",
            "NRRD image (*.nrrd, *.nhdr)",
            [".nrrd", ".nhdr"],
            ["ct", "cst", "dose"],
            read,
            None,
        ),
        ("meta", "MetaImage (*.mha, *.mhd)", [".mha", ".mhd"], ["ct", "cst", "dose"], read, None),
        (
            "dcm",
            "DICOM folder",
            [".dcm"],
            ["ct", "cst", "dose"],
            sitk_missing("ImageSeriesReader", "ReadImage") or module_missing("pydicom"),
            "A folder (Load Folder), not a single file",
        ),
    ]
    rows = []
    for fmt, label, extensions, objects, reason, note in entries:
        row = {
            "format": fmt,
            "label": label,
            "extensions": extensions,
            "objects": objects,
            "available": reason is None,
            "reason": reason,
        }
        if note:
            row["note"] = note
        rows.append(row)
    return rows


def export_formats() -> list[dict[str, Any]]:
    """Return the export table (``capabilities.formats.export``)."""
    return [entry.describe() for entry in EXPORT_FORMATS]


def _export_format(obj: str, fmt: str, structure_format: Optional[str]) -> ExportFormat:
    candidates = [e for e in EXPORT_FORMATS if e.object == obj and e.format == fmt]
    if not candidates:
        known = sorted({e.format for e in EXPORT_FORMATS if e.object == obj})
        if not known:
            raise ValueError(
                f"Unknown object {obj!r}; use one of pln, dij, cst, result, workspace"
            )
        raise ValueError(f"Cannot save {obj} as {fmt!r}; formats: {', '.join(known)}")
    if structure_format is not None:
        chosen = [e for e in candidates if e.structure_format == structure_format]
        if not chosen:
            raise ValueError(f"Unknown structure_format {structure_format!r}")
        return chosen[0]
    return candidates[0]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def _upload_name(filename: str) -> str:
    return os.path.basename(str(filename).replace("\\", "/")) or "upload"


def _payload_bytes(data: Any) -> bytes:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return np.asarray(data).tobytes()


def _load_upload(filename: str, payload: bytes) -> dict:
    from pyRadPlan.io import load_data  # noqa: PLC0415

    directory = tempfile.mkdtemp(prefix="pyradplan-upload-")
    path = os.path.join(directory, filename)
    try:
        with open(path, "wb") as handle:
            handle.write(payload)
        try:
            return load_data(path)
        except ValueError:
            # The importers need a CT; a dose-only archive (Save Result as npz) merges
            # onto the loaded patient instead.
            if not filename.lower().endswith(".npz"):
                raise
            from pyRadPlan.io.npz import NpzImporter  # noqa: PLC0415

            dose = NpzImporter(path).load_dose()
            if dose is None:
                raise
            return {"dose": dose}
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _check_image_upload(basename: str, kind: str) -> None:
    is_image = basename.lower().endswith(image_extensions())
    if kind in IMAGE_LOAD_KINDS:
        reason = sitk_missing("ReadImage") if is_image else None
        raise PreconditionError(
            reason
            or f"Importing an image as {kind!r} (Qt Import Image dialog) is not available "
            "through the protocol yet",
            "not_available",
        )
    if kind != "auto":
        raise ValueError(f"Unknown kind {kind!r}; use 'auto' or one of {list(IMAGE_LOAD_KINDS)}")
    if is_image:
        reason = sitk_missing("ReadImage")
        if reason:
            raise PreconditionError(f"Cannot load {basename}: {reason}", "not_available")


def load_file(session: Any, filename: str, data: Any, kind: str = "auto") -> None:
    """Load an uploaded file into the session (Qt "Load File").

    ``.mat``, ``.npz`` and ``.pkl`` / ``.pickle`` containers load directly; a workspace
    pickle also restores its snapshot tags (``saved_result_tags``) when it carries a CT or
    a result, as the Qt widget does.
    """
    basename = _upload_name(filename)
    _check_image_upload(basename, kind)
    payload = _payload_bytes(data)

    def compute() -> dict:
        return _load_upload(basename, payload)

    session.execute(Operation("load_file", f"Loading {basename}", compute, session._commit_loaded))


def load_dij(session: Any, filename: str, data: Any) -> None:
    """Merge the steering information and dose influence of a file (Qt "Load Dij")."""
    basename = _upload_name(filename)
    payload = _payload_bytes(data)

    def compute() -> dict:
        loaded = _load_upload(basename, payload)
        found = {key: loaded[key] for key in ("stf", "dij") if loaded.get(key) is not None}
        if "dij" not in found:
            raise PreconditionError(f"No dose-influence matrix found in {basename}", "no_dij")
        return found

    def commit(found: dict) -> None:
        session.workspace.set_many(**found)

    session.execute(
        Operation("load_dij", f"Loading dose influence from {basename}", compute, commit)
    )


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("._") or "export"


def _quantity_image(value: Any, ct: Any) -> Any:
    """Convert a result quantity into a CT-aligned ``sitk.Image`` (Qt ``_quantity_to_image``)."""
    sitk = _sitk()
    if isinstance(value, sitk.Image):
        return value
    array = np.asarray(value)
    # Raw 3-D arrays come from an imported matRad resultGUI, stored as (y, x, z).
    image = sitk.GetImageFromArray(np.transpose(array, (2, 0, 1)))
    if ct is not None and image.GetSize() == ct.cube_hu.GetSize():
        image.CopyInformation(ct.cube_hu)
    return image


def _zip_folder(folder: str) -> tuple[bytes, list[str]]:
    members = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(folder):
            for name in sorted(files):
                path = os.path.join(root, name)
                member = os.path.relpath(path, folder).replace(os.sep, "/")
                archive.write(path, member)
                members.append(member)
    return buffer.getvalue(), sorted(members)


def _file_result(filename: str, mime: str, data: bytes, files=None) -> dict[str, Any]:
    result = {"filename": filename, "mime": mime, "data": np.frombuffer(data, dtype=np.uint8)}
    if files is not None:
        result["files"] = files
    return result


def _result_keys(session: Any, quantities: Optional[Sequence[str]]) -> list[str]:
    result = session.workspace.result
    if quantities is None:
        keys = result_quantities(result)
    else:
        if isinstance(quantities, str):
            raise TypeError("quantities must be a list of component keys")
        keys = list(dict.fromkeys(quantities))
        unknown = [key for key in keys if result_entry(result, key) is None]
        if unknown:
            raise KeyError(
                f"No volume quantity {unknown} in the result; available: "
                f"{list(result_components(result))}"
            )
    if not keys:
        raise PreconditionError("The result has no exportable image quantities", "no_data")
    return keys


def _write_result(
    entry: ExportFormat, images: dict[str, Any], folder: str, basename: Optional[str]
) -> tuple[str, str, bool]:
    """Write the result *images*; returns the file or folder written, its name, zipped."""
    from pyRadPlan.io import save_data  # noqa: PLC0415

    if entry.format == "mat":
        sitk = _sitk()
        result_gui = {}
        for key, image in images.items():
            field = MATRAD_RESULT_NAMES.get(key) or re.sub(r"\W+", "_", key).strip("_")
            result_gui[field] = np.transpose(sitk.GetArrayFromImage(image), (1, 2, 0))
        name = f"{basename or 'result'}.mat"
        path = os.path.join(folder, name)
        save_data(file_name=path, format="mat", resultGUI=result_gui)
        return path, name, False
    if entry.format == "dcm":
        from pyRadPlan.io.dicom import DicomExporter  # noqa: PLC0415

        target = os.path.join(folder, "dicom")
        for key, image in images.items():
            subfolder = target if len(images) == 1 else os.path.join(target, _safe_name(key))
            DicomExporter(subfolder).save(dose=image)
        return target, f"{basename or 'result'}_dicom.zip", True
    target = os.path.join(folder, "files")
    os.makedirs(target)
    for key, image in images.items():
        save_data(
            file_name=os.path.join(target, f"{_safe_name(key)}{entry.extension}"),
            format=entry.format,
            dose=image,
        )
    if len(images) == 1:
        key = next(iter(images))
        name = f"{basename or _safe_name(key)}{entry.extension}"
        return os.path.join(target, f"{_safe_name(key)}{entry.extension}"), name, False
    return target, f"{basename or 'result'}.zip", True


def _require(obj: Any, what: str) -> None:
    if obj is None:
        raise PreconditionError(f"No {what} to save", "no_data")


def save(
    session: Any,
    object: str,  # noqa: A002 - protocol parameter name
    format: str,  # noqa: A002 - protocol parameter name
    quantities: Optional[Sequence[str]] = None,
    structure_format: Optional[str] = None,
    basename: Optional[str] = None,
) -> dict[str, Any]:
    """Write a workspace object to a ``FileResult`` (``export.save``, Qt File > Save ...).

    Pending plan-form edits are the frontend's to apply first, as the Qt widget asks for
    them before Save Plan and Save Workspace.
    """
    entry = _export_format(object, format, structure_format)
    reason = entry.reason()
    if reason is not None:
        raise PreconditionError(
            f"Cannot save {object} as {entry.label}: {reason}", "not_available"
        )
    ws = session.workspace
    base = _safe_name(os.path.splitext(basename)[0]) if basename else None
    if base and basename.lower().endswith(".nii.gz"):
        base = _safe_name(basename[: -len(".nii.gz")])

    if object == "workspace":
        _require(ws.ct, "CT (a workspace needs one)")
        objects = {key: value for key, value in ws.snapshot().items() if value is not None}
        objects["saved_result_tags"] = session.snapshot_tags()
    elif object == "result":
        keys = _result_keys(session, quantities)
        result = ws.result
        objects = {key: result_entry(result, key) for key in keys}
    else:
        _require(
            getattr(ws, object),
            {"pln": "plan", "dij": "dose influence", "cst": "structure set"}[object],
        )
        objects = {object: getattr(ws, object)}
    ct = ws.ct

    def compute() -> dict[str, Any]:
        from pyRadPlan.io import save_data  # noqa: PLC0415

        folder = tempfile.mkdtemp(prefix="pyradplan-export-")
        try:
            if object == "result":
                images = {key: _quantity_image(value, ct) for key, value in objects.items()}
                path, name, zipped = _write_result(entry, images, folder, base)
            elif entry.archive:
                path = os.path.join(folder, "export")
                name = f"{base or object}{entry.extension}"
                zipped = True
                if entry.format == "dcm":
                    from pyRadPlan.io.dicom import DicomExporter  # noqa: PLC0415

                    DicomExporter(path, structure_format=entry.structure_format).save(**objects)
                else:
                    save_data(file_name=path, format=entry.format, **objects)
            else:
                name = f"{base or object}{entry.extension}"
                path = os.path.join(folder, name)
                zipped = False
                save_data(file_name=path, format=entry.format, **objects)
            if zipped:
                data, members = _zip_folder(path)
                return _file_result(name, MIME_ZIP, data, members)
            with open(path, "rb") as handle:
                return _file_result(name, entry.mime, handle.read())
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    label = {"workspace": "Saving workspace", "result": "Saving result"}.get(
        object, f"Saving {object}"
    )
    return session.execute(Operation("export_save", label, compute, lambda value: value))
