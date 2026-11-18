"""Base data listing, adding and removing (protocol v1 ``base_data.*``).

Mirrors the Qt Base Data dialog and menu (``gui/menus/_base_data.py``) and the phantom
list of the File menu (``gui/menus/_file.py``) on :mod:`pyRadPlan.core._base_data`.
Files are added from uploaded bytes, validated by the domain functions
(:func:`pyRadPlan.machines.add_machine_file`, :func:`pyRadPlan.biology.models.add_rbe_table`,
...) and copied into the user's base data folder; under Pyodide that folder lives in the
in-memory file system, so added files last until the page is closed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Callable, Optional

from ._revisions import bump_base_data_revision
from ._session import PreconditionError

KINDS = ("machines", "rbe_tables", "stopping_power_tables", "phantoms")

_FOLDER_LABELS = {
    "bundled": "Shipped with pyRadPlan",
    "user": "Added by you",
    "shared": "Shared",
}


def _path_text(path: Any) -> str:
    """Return a folder as text (a ``MultiplexedPath`` as its first folder)."""
    paths = getattr(path, "_paths", None)
    if paths:
        return str(paths[0])
    return str(path)


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"Unknown base data kind {kind!r}; use one of {', '.join(KINDS)}")


def _describe(entry: Any) -> dict[str, Any]:
    """One row of the Qt Base Data dialog (or phantom menu) for a base data file."""
    kind = entry.kind.name
    stem = os.path.splitext(entry.name)[0]
    row: dict[str, Any] = {
        "name": entry.name,
        "file": entry.name,
        "source": entry.source,
        "removable": entry.source == "user",
        "path": _path_text(entry.path),
    }
    if kind == "machines":
        mode, _, name = stem.partition("_")
        row.update(name=name, radiation_mode=mode)
    elif kind == "phantoms":
        row["name"] = stem
    elif kind == "rbe_tables":
        from pyRadPlan.biology.models import TabulatedAlphaBetaModel  # noqa: PLC0415

        try:
            table = TabulatedAlphaBetaModel.quantity_table_kernel_set(entry.name)
        except Exception as exc:  # noqa: BLE001 - an unreadable table is listed as such
            row.update(model="unreadable", tissue_classes="", reference="", problem=str(exc))
        else:
            row.update(
                model=table.provenance_label,
                tissue_classes=", ".join(
                    f"α_x={c.tissue.alpha_x:g}, β_x={c.tissue.beta_x:g}"
                    for c in table.tissue_classes
                ),
                reference=table.reference,
                problem="\n".join(table.problems) or None,
            )
    return row


def list_kind(kind: str) -> dict[str, Any]:
    """Folders and files of a base data kind, in order of precedence."""
    from pyRadPlan.core import (  # noqa: PLC0415
        bundled_base_data_dir,
        list_base_data,
        shared_base_data_dirs,
        user_base_data_dir,
    )

    _check_kind(kind)
    folders = [
        {
            "source": "bundled",
            "label": _FOLDER_LABELS["bundled"],
            "path": _path_text(bundled_base_data_dir(kind)),
        },
        {"source": "user", "label": _FOLDER_LABELS["user"], "path": str(user_base_data_dir(kind))},
        *(
            {"source": "shared", "label": _FOLDER_LABELS["shared"], "path": str(folder)}
            for folder in shared_base_data_dirs(kind)
        ),
    ]
    if kind == "phantoms":
        from pyRadPlan.io import list_phantoms  # noqa: PLC0415

        entries = list_phantoms()
    else:
        entries = list_base_data(kind)
    return {"kind": kind, "folders": folders, "rows": [_describe(entry) for entry in entries]}


def _mat_name(name: str) -> str:
    return name if name.lower().endswith(".mat") else f"{name}.mat"


def _validator(
    kind: str, path: str, name: Optional[str], radiation_mode: Optional[str]
) -> tuple[str, str, Optional[str], Callable[[str, bool], Any]]:
    """Validate the file at *path*; returns suggested name, file name, mode and the adder."""
    if kind == "machines":
        from pyRadPlan.machines import add_machine_file, load_machine  # noqa: PLC0415

        machine = load_machine(path)
        mode = radiation_mode or machine.radiation_mode
        suggested = machine.name
        chosen = (name or suggested).strip()
        if not chosen:
            raise ValueError("A machine needs a name")

        def add_machine(file_path: str, overwrite: bool) -> Any:
            return add_machine_file(
                file_path, machine_name=chosen, radiation_mode=mode, overwrite=overwrite
            )

        return suggested, f"{mode}_{chosen}{os.path.splitext(path)[1]}", mode, add_machine

    upload = os.path.basename(path)
    if kind == "phantoms":
        from pyRadPlan.core import add_base_data_file  # noqa: PLC0415
        from pyRadPlan.io import load_data  # noqa: PLC0415

        if load_data(path).get("ct") is None:
            raise ValueError(f"{upload} holds no CT, so it is no phantom")
        ending = os.path.splitext(upload)[1]
        chosen = name if name is None or name.lower().endswith(ending.lower()) else name + ending
        file_name = chosen or upload

        def add_phantom(file_path: str, overwrite: bool) -> Any:
            return add_base_data_file("phantoms", file_path, name=file_name, overwrite=overwrite)

        return upload, file_name, None, add_phantom

    from pyRadPlan.biology.models._models import tabulated_rbe_models as tables  # noqa: PLC0415

    file_name = _mat_name(name or upload)
    if kind == "rbe_tables":
        model = tables.TabulatedAlphaBetaModel
        tables._check_table_data(
            path, model.table_root_key, ("A", "Z", "energies", *model.quantities_in_table)
        )
        try:
            tables._read_table_meta(path, model.table_root_key, None)
        except Exception as exc:
            raise ValueError(f"Could not read the metadata of {upload}: {exc}") from exc
        add_table = tables.add_rbe_table
    else:
        tables._check_table_data(path, "SPtable", ("A", "Z", "energies", "dEdx"))
        add_table = tables.add_stopping_power_table

    def add(file_path: str, overwrite: bool) -> Any:
        return add_table(file_path, name=file_name, overwrite=overwrite)

    return upload, file_name, None, add


def add(
    kind: str,
    filename: str,
    data: Any,
    name: Optional[str] = None,
    radiation_mode: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate an uploaded base data file and add it to the user's folder.

    A file of that name that ships with pyRadPlan cannot be replaced; one the user added
    only with *overwrite*. Both fail with ``PreconditionError(code="name_clash")``;
    ``dry_run`` reports the clash (and the suggested name) without writing anything, as
    the Qt dialog does before it asks for another name.
    """
    import numpy as np  # noqa: PLC0415

    from pyRadPlan.core import (  # noqa: PLC0415
        check_base_data_file_name,
        find_base_data,
        get_base_data_kind,
    )

    _check_kind(kind)
    label = get_base_data_kind(kind).label
    upload = os.path.basename(str(filename).replace("\\", "/")) or "upload.mat"
    payload = data if isinstance(data, (bytes, bytearray)) else np.asarray(data).tobytes()
    directory = tempfile.mkdtemp(prefix="pyradplan-basedata-")
    try:
        path = os.path.join(directory, upload)
        with open(path, "wb") as handle:
            handle.write(payload)
        try:
            suggested, file_name, mode, adder = _validator(kind, path, name, radiation_mode)
        except Exception as exc:  # noqa: BLE001 - any read failure (Qt reports it the same)
            raise ValueError(f"Could not read a {label} from {upload}: {exc}") from exc
        check_base_data_file_name(kind, file_name)
        existing = find_base_data(kind, file_name)
        clash = None if existing is None else {"file": existing.name, "source": existing.source}
        refused = existing is not None and (
            existing.source == "bundled" or (existing.source == "user" and not overwrite)
        )
        if dry_run:
            row = None
        elif refused:
            where = "ships with pyRadPlan" if existing.source == "bundled" else "was already added"
            raise PreconditionError(
                f"A {label} named {file_name!r} {where}; "
                "choose another name" + ("" if existing.source == "bundled" else " or overwrite"),
                "name_clash",
            )
        else:
            target = adder(path, overwrite)
            bump_base_data_revision()
            row = _describe(find_base_data(kind, os.path.basename(str(target))))
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    return {"row": row, "suggested_name": suggested, "radiation_mode": mode, "clash": clash}


def remove(kind: str, file: str) -> dict[str, Any]:
    """Delete a file the user added; shipped and shared files cannot be removed."""
    from pyRadPlan.core import find_base_data, remove_base_data_file  # noqa: PLC0415

    _check_kind(kind)
    entry = find_base_data(kind, file)
    if entry is None:
        raise FileNotFoundError(f"No base data file {file!r} of kind {kind!r}")
    if entry.source != "user":
        raise PreconditionError(
            f"{entry.name!r} is a {entry.source} file; only files you added can be removed",
            "not_removable",
        )
    remove_base_data_file(kind, entry.name)
    bump_base_data_revision()
    return {}
