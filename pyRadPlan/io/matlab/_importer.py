"""Importer for matRad-style MATLAB ``.mat`` files."""

import logging
from typing import ClassVar, Optional

import numpy as np
import SimpleITK as sitk

from pyRadPlan.ct import validate_ct, CT
from pyRadPlan.cst import validate_cst, StructureSet
from pyRadPlan.stf import validate_stf
from pyRadPlan.plan import validate_pln
from pyRadPlan.dij import validate_dij

from ..base import BaseImporter
from . import _matfile

logger = logging.getLogger(__name__)


def validate_matrad_patient(mdict: dict[str], remove_matrad_structures: bool = True) -> dict[str]:
    """
    Load a matRad-like patient from a dictionary.

    Assumes that the dictionary uses matRad's data structures and tries to validate them.

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


def _result_dose_to_sitk(result: dict, ct: CT) -> Optional[sitk.Image]:
    """Extract a physical dose cube from a matRad resultGUI dict as a SimpleITK image."""
    if not isinstance(result, dict):
        return None

    cube = result.get("physicalDose", result.get("physical_dose", None))
    if cube is None:
        return None

    cube = np.asarray(cube, dtype=float)
    # matRad cubes are stored as (y, x, z); SimpleITK expects (z, y, x).
    dose_image = sitk.GetImageFromArray(np.transpose(cube, (2, 0, 1)))
    if dose_image.GetSize() == ct.cube_hu.GetSize():
        dose_image.CopyInformation(ct.cube_hu)
    return dose_image


class MatlabImporter(BaseImporter):
    """Importer for matRad-style MATLAB ``.mat`` files."""

    format: ClassVar[str] = "mat"
    name: ClassVar[str] = "MATLAB Importer"
    extensions: ClassVar[tuple[str, ...]] = (".mat",)

    def __init__(self, path):
        super().__init__(path)
        self._mdict = None

    @property
    def mdict(self) -> dict:
        """Lazily load and cache the raw MATLAB dictionary."""
        if self._mdict is None:
            self._mdict = _matfile.load(self._require_path())
        return self._mdict

    def load_ct(self) -> CT:
        ct = self.mdict.get("ct", None)
        if ct is None:
            raise ValueError(f"No 'ct' found in {self.path}.")
        return validate_ct(ct)

    def load_cst(self, ct: Optional[CT] = None) -> Optional[StructureSet]:
        cst = self.mdict.get("cst", None)
        if cst is None:
            return None
        if ct is None:
            ct = self.load_ct()
        return validate_cst(cst, ct)

    def load_dose(self) -> Optional[sitk.Image]:
        result = self.mdict.get("resultGUI", None)
        if result is None:
            return None
        return _result_dose_to_sitk(result, self.load_ct())

    def load_data(self) -> dict:
        # Operate on a copy so the cached dictionary is not consumed.
        mdict = dict(self.mdict)
        patient_dict = validate_matrad_patient(mdict, remove_matrad_structures=True)

        result = patient_dict.pop("result", None)
        if result is not None and "ct" in patient_dict:
            dose = _result_dose_to_sitk(result, patient_dict["ct"])
            if dose is not None:
                patient_dict["dose"] = dose
            patient_dict["result"] = result

        return patient_dict
