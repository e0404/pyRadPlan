import os
import sys
from typing import Union

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+

from pymatreader import read_mat
from pyRadPlan.machines import Machine, validate_machine


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


def load_from_name(radiation_mode: str, machine_name: str) -> Machine:
    possible_endings = [".mat"]

    # Search resources
    # TODO: allow customization of search path
    search_paths = [resources.files("pyRadPlan.data.machines")]

    for ending in possible_endings:
        composed_filename = radiation_mode + "_" + machine_name + ending

        # We traverse all search paths
        for current_path in search_paths:
            filename = current_path.joinpath(composed_filename)
            if filename.exists():
                return load_machine(filename.as_posix())

    raise FileNotFoundError("Could not find the following machine file:", composed_filename)
