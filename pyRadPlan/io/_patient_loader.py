from typing import Optional
import os
import warnings
import logging
import sys

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+

from pyRadPlan.ct import validate_ct, CT
from pyRadPlan.cst import validate_cst, StructureSet
from pyRadPlan.stf import validate_stf
from pyRadPlan.plan import validate_pln
from pyRadPlan.dij import validate_dij
from . import matfile

logger = logging.getLogger(__name__)


def available_phantoms() -> dict[str, os.PathLike]:
    """
    List the patient phantoms bundled with pyRadPlan.

    Enumerates the ``*.mat`` files shipped in ``pyRadPlan.data.phantoms``.

    Returns
    -------
    dict[str, os.PathLike]
        Mapping of phantom shorthand names (the file stem, e.g. ``"TG119"``)
        to their package-resource paths, sorted by name.
    """
    files = resources.files("pyRadPlan.data.phantoms")
    phantoms = {
        entry.name[: -len(".mat")]: entry
        for entry in files.iterdir()
        if entry.is_file() and entry.name.lower().endswith(".mat")
    }
    return dict(sorted(phantoms.items()))


def resolve_phantom(name: str) -> os.PathLike:
    """
    Resolve a phantom shorthand name to its bundled data file.

    Matching is case-insensitive and tolerates a trailing ``.mat``.

    Parameters
    ----------
    name : str
        Shorthand name of a bundled phantom, e.g. ``"TG119"``.

    Returns
    -------
    os.PathLike
        Package-resource path of the phantom file.

    Raises
    ------
    ValueError
        If no bundled phantom matches *name*.
    """
    phantoms = available_phantoms()
    stem = name[: -len(".mat")] if name.lower().endswith(".mat") else name
    for phantom_name, path in phantoms.items():
        if phantom_name.lower() == stem.lower():
            return path
    raise ValueError(
        f"Unknown phantom {name!r}. Available phantoms: {', '.join(phantoms) or 'none'}"
    )


def load_phantom(
    name: str,
    extra_plan_data: Optional[dict] = None,
    extra_data: Optional[dict] = None,
) -> tuple[CT, StructureSet]:
    """
    Load a bundled phantom by its shorthand name.

    Parameters
    ----------
    name : str
        Shorthand name of a bundled phantom (see :func:`available_phantoms`).
    extra_plan_data : Optional[dict]
        Filled with additional plan data structures found in the file.
    extra_data : Optional[dict]
        Filled with any remaining raw data from the file.

    Returns
    -------
    tuple[CT, StructureSet]
        The CT and StructureSet objects.
    """
    return load_patient(resolve_phantom(name), extra_plan_data, extra_data)


def load_tg119() -> tuple[CT, StructureSet]:
    """
    Load the included TG119 phantom.

    This is a helper function to load the TG119 phantom included in the
    package data using importlib.resources.

    Returns
    -------
    tuple[CT, StructureSet]
        The CT and StructureSet objects.
    """
    return load_phantom("TG119")


def load_patient(
    filename: os.PathLike,
    extra_plan_data: Optional[dict] = None,
    extra_data: Optional[dict] = None,
) -> tuple[CT, StructureSet]:
    """
    Load a patient from a file.

    Chooses loader from the extension.

    Parameters
    ----------
    filename : os.PathLike
        Path to the file.
    additional_plan_data : Optional[dict]
        Additional data structures known to pyRadPlan that were present in the mat file
    """

    # Sanitize path and check if file exists
    path = os.path.normpath(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Patient file not found: {path}")

    # Get the file extension
    ext = os.path.splitext(path)[1].lower()

    # Load the patient
    if ext == ".mat":
        mdict = matfile.load(path)
        patient_dict = validate_matrad_patient(mdict)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # we require ct
    ct = patient_dict.pop("ct", None)
    if ct is None:
        raise ValueError("ct is missing from the patient file.")

    # we issue a warning when getting the cst
    cst = patient_dict.pop("cst", None)
    if cst is None:
        warnings.warn("cst/StructureSet is missing from the patient file.")

    extra_plan_data.update(patient_dict) if isinstance(extra_plan_data, dict) else None
    extra_data.update(mdict) if isinstance(extra_data, dict) else None

    return ct, cst


def validate_matrad_patient(mdict: dict[str], remove_matrad_structures: bool = True) -> dict[str]:
    """
    Load a matRad-like patient from a mat file.

    Assumes that the file uses matRad's data structures and tries to validates them.

    Parameters
    ----------
    mdict : dict[str]
        Dictionary imported from a .mat file. Modified if remove_matrad_structures is True.
    remove_matrad_structures : bool
        Pop the input data recognized/named as matRad structures from the dictionary.

    Returns
    -------
    dict[str]
        A dictionary with the validated data.

    """

    patient_dict = {}

    ct = mdict.pop("ct", None)
    cst = mdict.pop("cst", None)

    if ct is not None:
        patient_dict["ct"] = validate_ct(ct)

    if cst is not None:
        patient_dict["cst"] = validate_cst(cst, patient_dict["ct"])

    for key, validator in [("pln", validate_pln), ("stf", validate_stf), ("dij", validate_dij)]:
        value = mdict.get(key, None)
        if value is not None:
            try:
                patient_dict[key] = validator(value)
                if remove_matrad_structures:
                    mdict.pop(key)
            except ValueError:
                logger.warning(f"{key} present but could not be validated.")

    result = mdict.get("resultGUI", None)
    if result is not None:
        # TODO: validation as soon as result structure is implemented
        patient_dict["result"] = result
        if remove_matrad_structures:
            mdict.pop("resultGUI")

    return patient_dict
