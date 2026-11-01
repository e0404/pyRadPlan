"""Base data files shipped with pyRadPlan, added by the user or shared.

Base data (machine files, RBE tables, stopping power tables, phantoms) is looked up by file
name in these folders per kind, in order of precedence:

1. the folder shipped with pyRadPlan, ``pyRadPlan/data/<folder>``;
2. the same folder below the user's data root (see :func:`get_data_dir`), e.g.
   ``~/.pyradplan/<folder>``;
3. the same folder below each shared base data root listed in
   ``pyRadPlan.settings.base_data_path`` (``PYRADPLAN_BASE_DATA_PATH``, separated by
   :data:`os.pathsep`), e.g. a read-only folder with the phantoms of a course.

Files in the user folder are found exactly like the shipped ones, whether they were added
through :func:`add_base_data_file` or copied there by hand. A file never replaces one of the
same name in a folder of higher precedence, so a user file named like a shared one is used
instead of it.

This module only knows the folders; checking that a file holds valid data of its kind is
up to the domain functions (e.g. :func:`pyRadPlan.machines.add_machine_file`).
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources

from ._paths import get_data_subdir

#: Where a base data file comes from.
BaseDataSource = Literal["bundled", "user", "shared"]


@dataclass(frozen=True)
class BaseDataKind:
    """A kind of base data and the folders it is looked up in.

    Attributes
    ----------
    name : str
        Identifier of the kind (e.g. ``"machines"``).
    label : str
        Human-readable name of one file of the kind (e.g. ``"machine file"``).
    folder : str
        Folder below ``pyRadPlan/data`` and below the user's data root, ``/``-separated.
    endings : tuple[str, ...]
        File endings of the kind, in order of preference.
    """

    name: str
    label: str
    folder: str
    endings: tuple[str, ...] = (".mat",)

    @property
    def package(self) -> str:
        """Package of the shipped folder (e.g. ``"pyRadPlan.data.physics.stopping"``)."""
        return "pyRadPlan.data." + self.folder.replace("/", ".")


@dataclass(frozen=True)
class BaseDataFile:
    """A base data file found in one of the folders of its kind.

    Attributes
    ----------
    kind : BaseDataKind
        Kind of the file.
    name : str
        File name, which is what the file is selected by.
    path : Traversable
        Location of the file.
    source : {"bundled", "user", "shared"}
        Whether the file ships with pyRadPlan, is in the user's folder or in a shared one.
    """

    kind: BaseDataKind
    name: str
    path: object
    source: BaseDataSource


#: The kinds of base data, by name.
BASE_DATA_KINDS: dict[str, BaseDataKind] = {
    kind.name: kind
    for kind in (
        BaseDataKind("machines", "machine file", "machines"),
        BaseDataKind("rbe_tables", "RBE table", "rbe"),
        BaseDataKind("stopping_power_tables", "stopping power table", "physics/stopping"),
        BaseDataKind("phantoms", "phantom", "phantoms", (".mat", ".npz")),
    )
}

_INVALID_FILE_NAME_CHARACTERS = '\\/:*?"<>|'


def get_base_data_kind(kind: Union[str, BaseDataKind]) -> BaseDataKind:
    """Return the :class:`BaseDataKind` named ``kind`` (a kind is returned as is)."""
    if isinstance(kind, BaseDataKind):
        return kind
    try:
        return BASE_DATA_KINDS[kind]
    except KeyError:
        raise ValueError(
            f"Unknown base data kind {kind!r}; known are {', '.join(BASE_DATA_KINDS)}."
        ) from None


def bundled_base_data_dir(kind: Union[str, BaseDataKind]):
    """Folder of the ``kind`` files shipped with pyRadPlan (a Traversable)."""
    return resources.files(get_base_data_kind(kind).package)


def user_base_data_dir(kind: Union[str, BaseDataKind], create: bool = False) -> Path:
    """Folder of the ``kind`` files added by the user, below the data root.

    Parameters
    ----------
    kind : str or BaseDataKind
        Kind of base data.
    create : bool, optional, default=False
        Create the folder if it does not exist yet.
    """
    return get_data_subdir(get_base_data_kind(kind).folder, create=create)


def shared_base_data_roots() -> list[Path]:
    """Shared base data roots listed in ``pyRadPlan.settings.base_data_path``."""
    # Deferred: pyRadPlan._settings imports pyRadPlan.core while it is being imported.
    from pyRadPlan._settings import get_settings  # noqa: PLC0415

    entries = get_settings().base_data_path.split(os.pathsep)
    return [Path(entry.strip()).expanduser() for entry in entries if entry.strip()]


def shared_base_data_dirs(kind: Union[str, BaseDataKind]) -> list[Path]:
    """Folders of the shared ``kind`` files, one per shared root, whether they exist or not."""
    folder = get_base_data_kind(kind).folder
    return [root / folder for root in shared_base_data_roots()]


def _sourced_dirs(kind: Union[str, BaseDataKind]) -> list[tuple[BaseDataSource, object]]:
    return [
        ("bundled", bundled_base_data_dir(kind)),
        ("user", user_base_data_dir(kind)),
        *(("shared", folder) for folder in shared_base_data_dirs(kind)),
    ]


def base_data_dirs(kind: Union[str, BaseDataKind]) -> list:
    """Folders searched for ``kind`` files, in order of precedence."""
    return [folder for _, folder in _sourced_dirs(kind)]


def _folder_entries(kind: BaseDataKind, folder: Path) -> list[Path]:
    """Return the ``kind`` files directly in ``folder``, sorted by name."""
    if not folder.is_dir():
        return []
    return sorted(
        (
            entry
            for entry in folder.iterdir()
            if entry.is_file() and os.path.splitext(entry.name)[1] in kind.endings
        ),
        key=lambda entry: entry.name.casefold(),
    )


def list_base_data(
    kind: Union[str, BaseDataKind], source: Optional[BaseDataSource] = None
) -> list[BaseDataFile]:
    """Return the base data files of a kind in order of precedence, each folder sorted by name.

    A file named like one in a folder of higher precedence, in any case, is not listed: it is
    never found by name.

    Parameters
    ----------
    kind : str or BaseDataKind
        Kind of base data.
    source : {"bundled", "user", "shared"}, optional
        Only list the files of this source.
    """
    kind = get_base_data_kind(kind)
    files: list[BaseDataFile] = []
    seen: set[str] = set()
    for folder_source, folder in _sourced_dirs(kind):
        entries = _folder_entries(kind, folder)
        for entry in entries:
            if entry.name.casefold() in seen:
                continue
            if source is None or source == folder_source:
                files.append(BaseDataFile(kind, entry.name, entry, folder_source))
        seen.update(entry.name.casefold() for entry in entries)
    return files


def find_base_data(kind: Union[str, BaseDataKind], name: str) -> Optional[BaseDataFile]:
    """Locate the ``kind`` file named ``name`` without reading it, ``None`` if absent."""
    kind = get_base_data_kind(kind)
    folded = name.casefold()
    for folder_source, folder in _sourced_dirs(kind):
        path = folder.joinpath(name)
        if path.is_file():
            return BaseDataFile(kind, name, path, folder_source)
        # Case-sensitive file systems (Linux) need this to match names like Windows and macOS
        # do, within the folder so that every platform selects a file of the same folder
        for entry in _folder_entries(kind, folder):
            if entry.name.casefold() == folded:
                return BaseDataFile(kind, entry.name, entry, folder_source)
    return None


def check_base_data_file_name(kind: Union[str, BaseDataKind], name: str) -> None:
    """Raise ``ValueError`` unless ``name`` is a valid file name for a ``kind`` file."""
    kind = get_base_data_kind(kind)
    stem, ending = os.path.splitext(name)
    if ending not in kind.endings:
        raise ValueError(f"A {kind.label} must end in {' or '.join(kind.endings)}, not {name!r}.")
    if not stem or not name.isprintable() or any(c in name for c in _INVALID_FILE_NAME_CHARACTERS):
        raise ValueError(f"Invalid file name {name!r} for a {kind.label}.")


#: Single-element list holding the generation counter, see base_data_generation().
_generation = [0]


def base_data_generation() -> int:
    """Count of base data files added or removed through pyRadPlan in this process.

    Caches of data read from base data files include it in their key, so a file replaced
    through :func:`add_base_data_file` is read again even if its size and modification time
    stay the same (a copy within one tick of the file system clock keeps them).

    Returns
    -------
    int
        Increases with every :func:`add_base_data_file` and :func:`remove_base_data_file`.
    """
    return _generation[0]


def _changed() -> None:
    _generation[0] += 1


def add_base_data_file(
    kind: Union[str, BaseDataKind],
    filename: Union[os.PathLike, str],
    name: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Copy a file into the user folder of its kind, making it available by name.

    A shared file of the same name is not replaced but no longer used. The file's content is
    not checked; use the function of the kind for that (e.g.
    :func:`pyRadPlan.machines.add_machine_file`).

    Parameters
    ----------
    kind : str or BaseDataKind
        Kind of base data.
    filename : os.PathLike or str
        File to add.
    name : str, optional
        File name to add it under. Defaults to the name of ``filename``.
    overwrite : bool, optional, default=False
        Replace a user file of the same name.

    Returns
    -------
    Path
        The added file.

    Raises
    ------
    FileExistsError
        If a file of that name exists already. Shipped files cannot be replaced, user
        files only with ``overwrite``.
    """
    kind = get_base_data_kind(kind)
    source = Path(filename)
    name = source.name if name is None else name
    check_base_data_file_name(kind, name)

    target = user_base_data_dir(kind) / name
    existing = find_base_data(kind, name)
    if existing is not None and existing.source != "shared":
        if existing.source == "bundled":
            raise FileExistsError(
                f"A {kind.label} named {name!r} ships with pyRadPlan and cannot be "
                "replaced; choose another name."
            )
        if not overwrite:
            raise FileExistsError(f"A {kind.label} named {name!r} was already added.")
        if target.resolve() == source.resolve():
            return target

    user_base_data_dir(kind, create=True)
    shutil.copyfile(source, target)
    _changed()
    return target


def remove_base_data_file(kind: Union[str, BaseDataKind], name: str) -> None:
    """Delete a file of the user folder of its kind.

    Raises
    ------
    FileNotFoundError
        If the user folder has no such file; shipped files cannot be removed.
    """
    kind = get_base_data_kind(kind)
    target = user_base_data_dir(kind) / name
    if not target.is_file():
        raise FileNotFoundError(f"No user {kind.label} named {name!r}.")
    target.unlink()
    _changed()
