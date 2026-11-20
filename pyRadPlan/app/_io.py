"""File import and export of a session (protocol v1 ``data.*`` and ``export.save``).

The load and save steps of both frontends. The Qt GUI writes to and reads from real paths
(:func:`prepare_save_path`, :func:`prepare_import_image`, :func:`prepare_import_binary`,
:func:`prepare_scan_dicom`, :func:`prepare_import_dicom`, run as session operations); the
browser works with bytes: uploads are written to a temporary file and read with
:func:`pyRadPlan.io.load_data`, exports are written by the same code into a temporary
folder and returned as one file (several files and folders zipped). The format tables
behind both (``capabilities.formats``, the Qt file dialog filters) are defined here once.
Which formats work depends on the runtime's SimpleITK: formats needing ``sitk.WriteImage``
/ ``sitk.ReadImage`` are reported unavailable (with the reason) when it lacks them, e.g.
under the Pyodide SimpleITK stand-in; the image, binary and DICOM imports need them too.
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
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ._profiles import running_in_pyodide
from ._session import Operation, PreconditionError, dij_payload
from ._results import full_quantities, result_components
from ._workspace import unique_result_name

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
        if self.note:
            entry["note"] = self.note
        return entry


#: File endings :func:`pyRadPlan.io.load_data` unpickles.
PICKLE_EXTENSIONS = (".pkl", ".pickle")


def pickle_load_reason(allow_pickle: bool = True) -> Optional[str]:
    """Why pickles cannot be loaded here, ``None`` if they can.

    Unpickling runs code from the file. In the browser that code could read and send out
    the patient data in the page, so loading is disabled there (saving pickles is not).
    """
    if running_in_pyodide():
        return (
            "Loading a pickle runs code from the file, which could read the data in this "
            "page; it is disabled in the browser. Use .mat or .npz instead"
        )
    if not allow_pickle:
        return "This server does not load pickle files"
    return None


def check_pickle_load(name: Any, allow_pickle: bool = True) -> None:
    """Raise ``PreconditionError("not_available")`` if *name* is a pickle that cannot load."""
    if not os.fspath(name).lower().endswith(PICKLE_EXTENSIONS):
        return
    reason = pickle_load_reason(allow_pickle)
    if reason:
        raise PreconditionError(
            f"Cannot load {os.path.basename(os.fspath(name))}: {reason}", "not_available"
        )


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


_OBJECT_NAMES = {
    "workspace": "the complete workspace",
    "pln": "the plan",
    "dij": "the dose influence",
    "cst": "the structure set",
    "result": "the result",
}


def export_entries(obj: str, formats: Optional[Sequence[str]] = None) -> list[ExportFormat]:
    """Return the export table's entries of *obj* this runtime can write, in table order.

    *formats* restricts them to these format keys.
    """
    return [
        entry
        for entry in EXPORT_FORMATS
        if entry.object == obj
        and (formats is None or entry.format in formats)
        and entry.reason() is None
    ]


def path_label(entry: ExportFormat) -> str:
    """Return the label of *entry* written to a path, where archives are folders."""
    return entry.label.replace("(zip)", "(folder)") if entry.archive else entry.label


def path_extensions(entry: ExportFormat) -> tuple[str, ...]:
    """Return the endings a file written as *entry* may have, the default first."""
    from pyRadPlan.io import get_exporter  # noqa: PLC0415

    try:
        extensions = tuple(get_exporter(entry.format).extensions)
    except ValueError:
        extensions = ()
    return tuple(dict.fromkeys((entry.extension, *extensions)))


def writes_folder(entry: ExportFormat, count: int = 1) -> bool:
    """Whether *entry* writes a folder for *count* objects or quantities (else one file)."""
    if entry.object == "cst":
        return entry.archive
    if entry.object == "result":
        return entry.format == "dcm" or (entry.format != "mat" and count > 1)
    return False


def _plain_label(label: str) -> str:
    return re.sub(r"\s*\([^()]*\)\s*$", "", label)


def _patterns(extensions: Sequence[str], both_cases: bool) -> list[str]:
    # GTK file dialogs (Linux) match patterns case-sensitively.
    cases = (lambda e: dict.fromkeys((e, e.upper()))) if both_cases else (lambda e: (e,))
    return [f"*{case}" for ext in extensions for case in cases(ext)]


def dialog_filter(
    rows: Sequence[tuple[str, Sequence[str]]], *, both_cases: bool = False, catch_all: bool = False
) -> str:
    """Return a file dialog filter (``"Label (*.a *.b);;..."``) of ``(label, extensions)``.

    A trailing pattern list in a label is replaced by the row's own. *catch_all* frames the
    rows with "All supported" and "All files"; *both_cases* adds upper-case patterns.
    """
    entries = [
        f"{_plain_label(label)} ({' '.join(_patterns(extensions, both_cases))})"
        for label, extensions in rows
    ]
    if catch_all:
        every = sorted({p for _, ext in rows for p in _patterns(ext, both_cases)})
        entries = [f"All supported ({' '.join(every)})", *entries, "All files (*)"]
    return ";;".join(entries)


def export_filter(entries: Sequence[ExportFormat]) -> str:
    """Return the save dialog filter of single-file export *entries*."""
    return dialog_filter([(entry.label, path_extensions(entry)) for entry in entries])


def entry_for_filter(selected: str, entries: Sequence[ExportFormat]) -> ExportFormat:
    """Return the entry of the filter *selected* in a dialog built by :func:`export_filter`.

    Falls back to the first entry when the filter cannot be matched.
    """
    for entry in entries:
        if selected and selected.startswith(_plain_label(entry.label) + " ("):
            return entry
    return entries[0]


def load_filter() -> str:
    """Return the Load File dialog filter: every file format this runtime can import.

    DICOM is a folder import (Load Folder) and not listed; "All files" still opens one.
    """
    rows = [
        (row["label"], row["extensions"])
        for row in import_formats()
        if row["available"] and row["format"] != "dcm"
    ]
    return dialog_filter(rows, both_cases=True, catch_all=True)


def import_formats(allow_pickle: bool = True) -> list[dict[str, Any]]:
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
            pickle_load_reason(allow_pickle),
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


def load_path_data(path: Any, allow_pickle: bool = True) -> dict:
    """Read everything a file or folder holds (:func:`pyRadPlan.io.load_data`).

    The importers need a CT; a dose-only archive (Save Result as npz) is returned as a bare
    ``dose`` instead, which merges onto the loaded patient. Pickles are refused (see
    :func:`pickle_load_reason`).
    """
    from pyRadPlan.io import load_data  # noqa: PLC0415

    check_pickle_load(path, allow_pickle)
    try:
        return load_data(path)
    except ValueError:
        if not os.fspath(path).lower().endswith(".npz"):
            raise
        from pyRadPlan.io.npz import NpzImporter  # noqa: PLC0415

        dose = NpzImporter(path).load_dose()
        if dose is None:
            raise
        return {"dose": dose}


def _load_upload(filename: str, payload: bytes, allow_pickle: bool = True) -> dict:
    check_pickle_load(filename, allow_pickle)
    directory = tempfile.mkdtemp(prefix="pyradplan-upload-")
    path = os.path.join(directory, filename)
    try:
        with open(path, "wb") as handle:
            handle.write(payload)
        return load_path_data(path, allow_pickle)
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


def prepare_load_file(session: Any, filename: str, data: Any, kind: str = "auto") -> Operation:
    """Return the operation loading an uploaded file (Qt "Load File")."""
    basename = _upload_name(filename)
    _check_image_upload(basename, kind)
    check_pickle_load(basename, session.allow_pickle)
    payload = _payload_bytes(data)
    allow_pickle = session.allow_pickle

    def compute() -> dict:
        return _load_upload(basename, payload, allow_pickle)

    return Operation("load_file", f"Loading {basename}", compute, session._commit_loaded)


def load_file(session: Any, filename: str, data: Any, kind: str = "auto") -> None:
    """Load an uploaded file into the session (Qt "Load File").

    ``.mat``, ``.npz`` and ``.pkl`` / ``.pickle`` containers load directly (pickles not in
    the browser); a workspace pickle brings all of its named results.
    """
    session.execute(prepare_load_file(session, filename, data, kind))


def prepare_load_dij(
    session: Any, name: str, load: Callable[[], dict], label: Optional[str] = None
) -> Operation:
    """Return the operation merging the steering information and dose influence *load* reads."""

    def compute() -> dict:
        return dij_payload(load(), name)

    def commit(found: dict) -> None:
        session.workspace.set_many(**found)

    return Operation("load_dij", label or f"Loading dose influence from {name}", compute, commit)


def prepare_load_dij_upload(session: Any, filename: str, data: Any) -> Operation:
    """Return the operation loading the dose influence of an uploaded file."""
    basename = _upload_name(filename)
    check_pickle_load(basename, session.allow_pickle)
    payload = _payload_bytes(data)
    allow_pickle = session.allow_pickle
    return prepare_load_dij(
        session, basename, lambda: _load_upload(basename, payload, allow_pickle)
    )


def load_dij(session: Any, filename: str, data: Any) -> None:
    """Merge the steering information and dose influence of a file (Qt "Load Dij")."""
    session.execute(prepare_load_dij_upload(session, filename, data))


# ----------------------------------------------------------------------
# Native imports: image files, binary folders, DICOM folders
# ----------------------------------------------------------------------


def is_image_file(path: Any) -> bool:
    """Whether *path* is a bare image file (CT, structures or dose: the user decides)."""
    return os.fspath(path).lower().endswith(image_extensions())


def unique_name(name: str, taken: Any) -> str:
    """Return *name*, or ``name_2`` / ``name_3`` / ... if it collides with *taken*."""
    if name not in taken:
        return name
    i = 2
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


def same_grid(a: Any, b: Any) -> bool:
    """Whether two images (or image file readers) share size, spacing, origin and direction."""
    return (
        a.GetSize() == b.GetSize()
        and a.GetSpacing() == b.GetSpacing()
        and a.GetOrigin() == b.GetOrigin()
        and a.GetDirection() == b.GetDirection()
    )


def _require_sitk_io(*names: str) -> None:
    reason = sitk_missing(*names)
    if reason:
        raise PreconditionError(reason, "not_available")


def default_dose_name(path: Any) -> str:
    """Return the result name of a dose image file: its sanitized file name."""
    from pyRadPlan.io.sitk_based._binary_import import _file_stem  # noqa: PLC0415

    return re.sub(r"\W+", "_", _file_stem(os.fspath(path))).strip("_") or "imported_dose"


@dataclass(frozen=True)
class ImageInfo:
    """What the Import Image choice needs to know about an image file."""

    num_labels: int
    grid_matches: bool
    inferred: str
    default_name: str


def inspect_image(path: Any, ct_image: Any = None) -> ImageInfo:
    """Read an image file: its label count, whether it lies on *ct_image*'s grid, its kind.

    The kind (``"ct"``, ``"structures"`` or ``"dose"``) is a guess from the pixel type and
    values (:func:`pyRadPlan.io.sitk_based.infer_image_kind`).
    """
    from pyRadPlan.io.sitk_based import infer_image_kind  # noqa: PLC0415

    sitk = _sitk()
    image = sitk.ReadImage(os.fspath(path))
    values = np.unique(sitk.GetArrayViewFromImage(image))
    return ImageInfo(
        num_labels=int(np.count_nonzero(values)),
        grid_matches=ct_image is not None and same_grid(image, ct_image),
        inferred=infer_image_kind(image),
        default_name=default_dose_name(path),
    )


def _file_grid(path: str) -> Any:
    reader = _sitk().ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    return reader


def prepare_import_image(
    session: Any, path: Any, mode: str, name: Optional[str] = None
) -> Operation:
    """Import a bare image file (Qt Import Image) as *mode*.

    ``ct_new`` starts a new patient; ``ct_replace`` swaps the CT and, if the grid differs,
    clears the structures, dose influence and results built on the old one; ``structures``
    adds the mask or every label of a label map to the structure set (clashing names get a
    numeric suffix); ``dose`` adds the image as a result of its own, with the image as
    its ``physical_dose`` on the grid of the file, named *name* (default: the file name),
    suffixed if taken. The commit of ``dose`` returns the name of that result; its
    quantity is listed as ``"<name>/physical_dose"``.
    """
    path = os.fspath(path)
    _require_sitk_io("ReadImage")
    ws = session.workspace
    ct = ws.ct
    if mode not in IMAGE_LOAD_KINDS:
        raise ValueError(f"Unknown mode {mode!r}; use one of {list(IMAGE_LOAD_KINDS)}")
    if mode != "ct_new" and ct is None:
        raise PreconditionError(f"Importing an image as {mode!r} needs a CT", "no_data")
    basename = os.path.basename(path)

    if mode in ("ct_new", "ct_replace"):

        def compute_ct() -> tuple:
            from pyRadPlan.io.sitk_based import read_ct_image  # noqa: PLC0415

            matches = ct is not None and same_grid(_file_grid(path), ct.cube_hu)
            return read_ct_image(path), matches

        def commit_ct(value: tuple) -> None:
            new_ct, matches = value
            if mode == "ct_new":
                session._commit_loaded({"ct": new_ct})
            elif matches:
                ws.ct = new_ct
            else:
                ws.set_many(ct=new_ct, cst=None, dij=None, results={})

        return Operation("import_image", f"Importing CT from {basename}", compute_ct, commit_ct)

    if mode == "structures":
        existing = list(ws.cst.vois) if ws.cst is not None else []

        def compute_cst() -> Any:
            from pyRadPlan.cst import validate_cst  # noqa: PLC0415
            from pyRadPlan.io.sitk_based import image_file_to_vois  # noqa: PLC0415

            taken = {voi.name for voi in existing}
            merged = list(existing)
            for voi in image_file_to_vois(ct, path):
                unique = unique_name(voi.name, taken)
                taken.add(unique)
                merged.append(
                    voi if unique == voi.name else voi.model_copy(update={"name": unique})
                )
            return validate_cst(merged, ct)

        return Operation(
            "import_image",
            f"Importing structures from {basename}",
            compute_cst,
            lambda cst: ws.set("cst", cst),
        )

    key_name = name or default_dose_name(path)

    def compute_dose() -> Any:
        return _sitk().ReadImage(path)

    def commit_dose(image: Any) -> str:
        result_name = unique_result_name(key_name, {*ws.results, ws.active_result})
        ws.add_result(result_name, {"physical_dose": image})
        return result_name

    return Operation("import_image", f"Importing dose from {basename}", compute_dose, commit_dose)


def folder_has_images(directory: Any) -> bool:
    """Whether a folder (or an immediate subfolder) holds image files SimpleITK reads."""
    from pyRadPlan.io import list_image_files  # noqa: PLC0415

    directory = os.fspath(directory)
    if list_image_files(directory):
        return True
    return any(
        os.path.isdir(os.path.join(directory, entry))
        and bool(list_image_files(os.path.join(directory, entry)))
        for entry in sorted(os.listdir(directory))
    )


def folder_kind(directory: Any) -> Optional[str]:
    """``"dicom"`` for a DICOM folder, ``"images"`` for image files, ``None`` for neither."""
    from pyRadPlan.io.dicom import DicomImporter  # noqa: PLC0415

    if DicomImporter.handles_directory(os.fspath(directory)):
        return "dicom"
    if folder_has_images(directory):
        return "images"
    return None


def binary_folder_files(directory: Any) -> tuple[Optional[str], list[str]]:
    """Propose the CT file and the mask files of a binary patient folder.

    The first top-level image is the CT candidate; the other top-level images and every
    image in an immediate subfolder are the masks.
    """
    from pyRadPlan.io import list_image_files  # noqa: PLC0415

    directory = os.fspath(directory)
    top = list_image_files(directory)
    masks = list(top[1:])
    for entry in sorted(os.listdir(directory)):
        full = os.path.join(directory, entry)
        if os.path.isdir(full):
            masks.extend(list_image_files(full))
    return (top[0] if top else None), masks


def prepare_import_binary(
    session: Any, ct_file: Any, selections: Sequence[Mapping[str, Any]]
) -> Operation:
    """Import a CT image file and binary masks (Qt Import Binary Patient), as a load does.

    *selections* are ``{path, name, voi_type}`` rows; ``voi_type`` ``"IGNORED"`` drops one.
    """
    _require_sitk_io("ReadImage")
    ct_file = os.fspath(ct_file)
    selections = [dict(selection) for selection in selections]

    def compute() -> dict:
        from pyRadPlan.io import load_binary_patient  # noqa: PLC0415
        from pyRadPlan.io.sitk_based import read_ct_image  # noqa: PLC0415

        active = [s for s in selections if str(s.get("voi_type", "")).upper() != "IGNORED"]
        if not active:
            return {"ct": read_ct_image(ct_file)}
        ct, cst = load_binary_patient(ct_file, [], selections=selections)
        return {"ct": ct, "cst": cst}

    return Operation("import_binary", "Importing binary data", compute, session._commit_loaded)


def dicom_catalog(importer: Any) -> dict[str, list]:
    """Return the CT series, structure sets and doses of a DICOM folder (headers only)."""
    return {
        "series": importer.list_ct_series(),
        "structs": importer.list_structure_sets(),
        "doses": importer.list_doses(),
    }


def _require_dicom() -> None:
    reason = sitk_missing("ImageSeriesReader", "ReadImage") or module_missing("pydicom")
    if reason:
        raise PreconditionError(reason, "not_available")


def prepare_scan_dicom(session: Any, directory: Any) -> Operation:
    """Scan a DICOM folder; the commit returns ``{"importer": ..., "catalog": ...}``.

    The importer goes with the user's choice to :func:`prepare_import_dicom`.
    """
    from pyRadPlan.io.dicom import DicomImporter  # noqa: PLC0415

    _require_dicom()
    importer = DicomImporter(os.fspath(directory))
    return Operation(
        "scan_dicom",
        "Scanning DICOM folder",
        lambda: dicom_catalog(importer),
        lambda catalog: {"importer": importer, "catalog": catalog},
    )


def prepare_import_dicom(session: Any, importer: Any, selection: Mapping[str, Any]) -> Operation:
    """Import the chosen CT series, structure set and dose of a scanned DICOM folder.

    *selection*: ``series_uid`` (``None``: the importer's choice), ``struct_file`` (``None``
    skips structures), ``load_dose`` and ``dose_file`` (``None``: the plan's physical dose).
    The dose is read on the CT grid. Merged as a load is.
    """
    _require_dicom()
    sel = dict(selection)

    def compute() -> dict:
        # One step per object, so the progress reflects the whole import and each loader's
        # own progress advances it by that step's share.
        steps = 1 + bool(sel.get("struct_file")) + bool(sel.get("load_dose"))
        data: dict[str, Any] = {}
        with importer.progress("Importing DICOM", total=steps) as step:
            data["ct"] = importer.load_ct(series_uid=sel.get("series_uid"))
            step.advance()
            if sel.get("struct_file"):
                cst = importer.load_cst(ct=data["ct"], struct_file=sel["struct_file"])
                if cst is not None:
                    data["cst"] = cst
                step.advance()
            if sel.get("load_dose"):
                dose = importer.load_dose(dose_file=sel.get("dose_file"), ct=data["ct"])
                if dose is not None:
                    data["dose"] = dose
                step.advance()
        return data

    return Operation("import_dicom", "Importing DICOM data", compute, session._commit_loaded)


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("._") or "export"


def image_keys(results: Any) -> list[str]:
    """Keys of the quantities that can be saved as images.

    These are the whole-volume quantities of all results (the entries of
    :meth:`Workspace.result_entries`), ``"[<result>/]<quantity>"``; per-beam and
    per-scenario lists are not offered.
    """
    return list(full_quantities(results))


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


def _result_images(session: Any, quantities: Optional[Sequence[str]]) -> dict[str, Any]:
    """Return the images to save by key.

    Without *quantities* these are the whole-volume quantities of all results; a chosen
    key may also name a per-beam or per-scenario component.
    """
    entries = session.workspace.result_entries()
    available = result_components(entries)
    if quantities is None:
        keys = list(full_quantities(entries))
    else:
        if isinstance(quantities, str):
            raise TypeError("quantities must be a list of component keys")
        keys = list(dict.fromkeys(quantities))
        unknown = [key for key in keys if key not in available]
        if unknown:
            raise KeyError(
                f"No volume quantity {unknown} in the results; available: {list(available)}"
            )
    if not keys:
        raise PreconditionError("The results have no exportable image quantities", "no_data")
    return {key: available[key] for key in keys}


def _require(obj: Any, what: str) -> None:
    if obj is None:
        raise PreconditionError(f"No {what} to save", "no_data")


def _save_inputs(
    session: Any,
    obj: str,
    fmt: str,
    quantities: Optional[Sequence[str]],
    structure_format: Optional[str],
) -> tuple[ExportFormat, dict[str, Any], Any]:
    """Check a save and capture what it writes: the format entry, the objects, the CT."""
    entry = _export_format(obj, fmt, structure_format)
    reason = entry.reason()
    if reason is not None:
        raise PreconditionError(f"Cannot save {obj} as {entry.label}: {reason}", "not_available")
    ws = session.workspace
    if obj == "workspace":
        _require(ws.ct, "CT (a workspace needs one)")
        objects = {key: value for key, value in ws.snapshot().items() if value is not None}
        # ``result`` is the active one of ``results``, for readers of a single result.
        objects.update(results=ws.results, active_result=ws.active_result)
    elif obj == "result":
        objects = _result_images(session, quantities)
    else:
        _require(
            getattr(ws, obj),
            {"pln": "plan", "dij": "dose influence", "cst": "structure set"}[obj],
        )
        objects = {obj: getattr(ws, obj)}
    return entry, objects, ws.ct


def matrad_field(key: str) -> str:
    """``resultGUI`` field of a quantity key: matRad's name, ``<name>_<result>`` for other results."""
    from pyRadPlan.quantities._display_names import split_result_key  # noqa: PLC0415

    tag, quantity = split_result_key(key)
    field = MATRAD_RESULT_NAMES.get(quantity) or re.sub(r"\W+", "_", quantity).strip("_")
    return f"{field}_{_safe_name(tag)}" if tag else field


def _write_result(entry: ExportFormat, images: dict[str, Any], target: str) -> list[str]:
    from pyRadPlan.io import save_data  # noqa: PLC0415

    if entry.format == "mat":
        sitk = _sitk()
        result_gui = {}
        for key, image in images.items():
            result_gui[matrad_field(key)] = np.transpose(sitk.GetArrayFromImage(image), (1, 2, 0))
        save_data(file_name=target, format="mat", resultGUI=result_gui)
        return [target]
    if entry.format == "dcm":
        from pyRadPlan.io.dicom import DicomExporter  # noqa: PLC0415

        for key, image in images.items():
            subfolder = target if len(images) == 1 else os.path.join(target, _safe_name(key))
            DicomExporter(subfolder).save(dose=image)
        return [target]
    if len(images) == 1:
        (image,) = images.values()
        return [save_data(file_name=target, format=entry.format, dose=image)]
    os.makedirs(target, exist_ok=True)
    return [
        save_data(
            file_name=os.path.join(target, f"{_safe_name(key)}{entry.extension}"),
            format=entry.format,
            dose=image,
        )
        for key, image in images.items()
    ]


def write_export(entry: ExportFormat, objects: Mapping[str, Any], ct: Any, target: str) -> list:
    """Write *objects* as *entry* to *target* and return the paths written.

    *target* is a file, or a folder where :func:`writes_folder` says so. For a result,
    *objects* are the images to write by quantity key; each is written on its own grid.
    """
    from pyRadPlan.io import save_data  # noqa: PLC0415

    if entry.object == "result":
        return _write_result(entry, dict(objects), target)
    if entry.archive and entry.format == "dcm":
        from pyRadPlan.io.dicom import DicomExporter  # noqa: PLC0415

        DicomExporter(target, structure_format=entry.structure_format).save(**objects)
        return [target]
    return [save_data(file_name=target, format=entry.format, **objects) or target]


def _save_label(obj: str) -> str:
    return {"workspace": "Saving workspace", "result": "Saving result"}.get(obj, f"Saving {obj}")


def _file_target(path: str, entry: ExportFormat) -> str:
    extensions = path_extensions(entry)
    if path.lower().endswith(tuple(e.lower() for e in extensions)):
        return path
    if not os.path.splitext(path)[1]:
        return path + entry.extension
    raise ValueError(f"Use {' or '.join(extensions)} to save {_OBJECT_NAMES[entry.object]}.")


def prepare_save_path(  # noqa: PLR0913 - the save options
    session: Any,
    object: str,  # noqa: A002 - the export.save parameter name
    format: str,  # noqa: A002 - the export.save parameter name
    path: Any,
    quantities: Optional[Sequence[str]] = None,
    structure_format: Optional[str] = None,
    label: Optional[str] = None,
) -> Operation:
    """Write a workspace object to *path* (Qt File > Save ...); returns the paths written.

    *path* is a file, or a folder where :func:`writes_folder` says so (structure label maps
    and DICOM, several result quantities, result DICOM). A file without an ending gets the
    format's; another format's ending is refused. Pending plan-form edits are the
    frontend's to apply first.
    """
    entry, objects, ct = _save_inputs(session, object, format, quantities, structure_format)
    target = os.fspath(path)
    if not writes_folder(entry, len(objects)):
        target = _file_target(target, entry)

    def compute() -> list:
        return write_export(entry, objects, ct, target)

    return Operation("save_path", label or _save_label(object), compute, lambda written: written)


def _download_target(
    entry: ExportFormat, objects: Mapping[str, Any], folder: str, base: Optional[str]
) -> tuple[str, str, bool]:
    """Where a download is written in *folder*: the target, the file name, whether zipped."""
    obj = entry.object
    if obj != "result":
        name = f"{base or obj}{entry.extension}"
        if entry.archive:
            return os.path.join(folder, "export"), name, True
        return os.path.join(folder, name), name, False
    if entry.format == "mat":
        name = f"{base or 'result'}.mat"
        return os.path.join(folder, name), name, False
    if entry.format == "dcm":
        return os.path.join(folder, "dicom"), f"{base or 'result'}_dicom.zip", True
    if len(objects) == 1:
        name = f"{base or _safe_name(next(iter(objects)))}{entry.extension}"
        return os.path.join(folder, name), name, False
    return os.path.join(folder, "files"), f"{base or 'result'}.zip", True


def save(
    session: Any,
    object: str,  # noqa: A002 - protocol parameter name
    format: str,  # noqa: A002 - protocol parameter name
    quantities: Optional[Sequence[str]] = None,
    structure_format: Optional[str] = None,
    basename: Optional[str] = None,
) -> dict[str, Any]:
    """Write a workspace object to a ``FileResult`` (``export.save``, Qt File > Save ...).

    Written by :func:`write_export` into a temporary folder, as :func:`prepare_save_path`
    writes to a path. Pending plan-form edits are the frontend's to apply first, as the Qt
    widget asks for them before Save Plan and Save Workspace.
    """
    entry, objects, ct = _save_inputs(session, object, format, quantities, structure_format)
    base = _safe_name(os.path.splitext(basename)[0]) if basename else None
    if base and basename.lower().endswith(".nii.gz"):
        base = _safe_name(basename[: -len(".nii.gz")])

    def compute() -> dict[str, Any]:
        folder = tempfile.mkdtemp(prefix="pyradplan-export-")
        try:
            target, name, zipped = _download_target(entry, objects, folder, base)
            written = write_export(entry, objects, ct, target)
            if zipped:
                data, members = _zip_folder(target)
                return _file_result(name, MIME_ZIP, data, members)
            with open(written[0], "rb") as handle:
                return _file_result(name, entry.mime, handle.read())
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    return session.execute(Operation("export_save", _save_label(object), compute, lambda v: v))
