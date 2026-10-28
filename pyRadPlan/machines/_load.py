import os
from pathlib import Path
from typing import Optional, Union

from pymatreader import read_mat
from pyRadPlan.core import (
    BASE_DATA_KINDS,
    add_base_data_file,
    find_base_data,
    list_base_data,
    remove_base_data_file,
    user_base_data_dir,
)
from pyRadPlan.machines import Machine, validate_machine

_KIND = BASE_DATA_KINDS["machines"]

#: File endings recognized as machine files, in order of preference.
MACHINE_FILE_ENDINGS = _KIND.endings


def user_machine_dir(create: bool = False) -> Path:
    """Folder holding the machine files added by the user.

    It is ``<data_dir>/machines`` (see :func:`pyRadPlan.core.get_data_dir`) and is
    searched after the machines shipped with pyRadPlan. Files land there through
    :func:`add_machine_file`, or can be copied there by hand following the
    ``<radiation_mode>_<machine_name>.mat`` naming convention.

    Parameters
    ----------
    create : bool, optional, default=False
        Create the folder if it does not exist yet.
    """
    return user_base_data_dir(_KIND, create=create)


def load_machine(filename: str) -> Machine:
    if filename.endswith(".mat"):
        machine_dict = load_machine_from_mat(filename)
    else:
        raise ValueError("Unsupported file type for machine loading:", filename)

    return validate_machine(machine_dict)


def load_machine_from_mat(filename: Union[os.PathLike, str]) -> dict:
    try:
        machine = read_mat(filename)
        return machine["machine"]
    except FileNotFoundError as exc:
        raise FileNotFoundError("Could not find the following machine file:", filename) from exc
    except KeyError as exc:
        raise FileNotFoundError(
            "Could not find the machine key in the following file:", filename
        ) from exc


def get_available_machines(
    radiation_mode: Optional[str] = None, user_only: bool = False
) -> dict[str, list[str]]:
    """Machine names available per radiation mode.

    Machines are discovered from the search paths by their file name, which follows
    the ``<radiation_mode>_<machine_name>`` convention.

    Parameters
    ----------
    radiation_mode : str, optional
        If given, only machines for this radiation mode are returned.
    user_only : bool, optional, default=False
        Only list the machines added by the user (see :func:`user_machine_dir`).

    Returns
    -------
    dict[str, list[str]]
        Sorted machine names per radiation mode. Radiation modes without a machine
        file are absent, except for an explicitly requested one, which maps to an
        empty list.
    """
    machines: dict[str, list[str]] = {} if radiation_mode is None else {radiation_mode: []}

    for entry in list_base_data(_KIND, source="user" if user_only else None):
        stem = os.path.splitext(entry.name)[0]
        if "_" not in stem:
            continue
        mode, machine_name = stem.split("_", 1)
        if radiation_mode is not None and mode != radiation_mode:
            continue
        if machine_name not in machines.setdefault(mode, []):
            machines[mode].append(machine_name)

    for names in machines.values():
        names.sort(key=str.casefold)

    return machines


def find_machine_file(radiation_mode: str, machine_name: str):
    """Locate the base data file of a machine without reading it.

    Machine files follow the ``<radiation_mode>_<machine_name>`` convention, see
    :func:`get_available_machines`. The search paths are traversed per file ending, in
    the order of preference of :data:`MACHINE_FILE_ENDINGS`.

    Parameters
    ----------
    radiation_mode : str
        Radiation mode of the machine (``"protons"``, ``"carbon"``, ...).
    machine_name : str
        Name of the machine (``"Generic"``, ...).

    Returns
    -------
    Traversable or None
        Path of the first matching file, ``None`` if no search path holds one. Reading it
        is what costs, so callers only needing the file's identity (e.g. to decide whether
        a cached machine is still current) can stat it instead.
    """
    for ending in MACHINE_FILE_ENDINGS:
        found = find_base_data(_KIND, radiation_mode + "_" + machine_name + ending)
        if found is not None:
            return found.path

    return None


def load_from_name(radiation_mode: str, machine_name: str) -> Machine:
    filename = find_machine_file(radiation_mode, machine_name)

    if filename is None:
        composed_filename = radiation_mode + "_" + machine_name + MACHINE_FILE_ENDINGS[-1]
        raise FileNotFoundError("Could not find the following machine file:", composed_filename)

    return load_machine(filename.as_posix())


def add_machine_file(
    filename: Union[os.PathLike, str],
    machine_name: Optional[str] = None,
    radiation_mode: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Add a machine file to the machines available by name.

    The file is copied into :func:`user_machine_dir` as
    ``<radiation_mode>_<machine_name>.mat``, so the machine persists across sessions
    and is found by :func:`load_from_name` and :func:`get_available_machines`.

    Parameters
    ----------
    filename : os.PathLike or str
        Machine file to add.
    machine_name : str, optional
        Name to list the machine under. Defaults to the name stored in the file.
    radiation_mode : str, optional
        Radiation mode of the file's machine. When omitted, the file is read and
        validated and both the radiation mode and (if not given) the machine name are
        taken from it; pass it to copy an already validated file without reading it
        again.
    overwrite : bool, optional, default=False
        Replace a machine of the same name previously added by the user.

    Returns
    -------
    Path
        The added file.

    Raises
    ------
    FileExistsError
        If a machine of that radiation mode and name exists already. Machines shipped
        with pyRadPlan cannot be replaced; user machines only with ``overwrite``.
    """
    source = Path(filename)
    _, ending = os.path.splitext(source.name)
    if ending not in MACHINE_FILE_ENDINGS:
        raise ValueError(f"Unsupported file type for machine loading: {source}")

    if radiation_mode is None:
        machine = load_machine(str(source))
        radiation_mode = machine.radiation_mode
        if machine_name is None:
            machine_name = machine.name
    if not machine_name:
        raise ValueError("A machine name is needed when the radiation mode is given.")
    if not radiation_mode or "_" in radiation_mode:
        raise ValueError(f"Invalid radiation mode {radiation_mode!r} for a machine file name.")

    return add_base_data_file(
        _KIND, source, name=f"{radiation_mode}_{machine_name}{ending}", overwrite=overwrite
    )


def remove_machine_file(radiation_mode: str, machine_name: str) -> None:
    """Remove a machine added through :func:`add_machine_file`.

    Raises
    ------
    FileNotFoundError
        If the user has no such machine; machines shipped with pyRadPlan cannot be
        removed.
    """
    for ending in MACHINE_FILE_ENDINGS:
        name = f"{radiation_mode}_{machine_name}{ending}"
        if (user_machine_dir() / name).is_file():
            remove_base_data_file(_KIND, name)
            return
    raise FileNotFoundError(f"No user {radiation_mode} machine named {machine_name!r}.")
