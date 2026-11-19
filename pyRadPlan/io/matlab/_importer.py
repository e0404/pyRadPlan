"""Importer for matRad-style MATLAB ``.mat`` files."""

import logging
from typing import Any, ClassVar, Optional

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


def _validate_dij(value: Any, pln: Any) -> Any:
    """Validate a Dij with the scenario model of the plan in the same file, if it has several."""
    model = getattr(pln, "scenarios", None)
    if model is None or len(model.scenarios()) <= 1:
        return validate_dij(value)
    try:
        return validate_dij(value, scenario_model=model)
    except ValueError as exc:
        logger.warning(
            f"dij does not match the scenario model of the plan, imported without: {exc}"
        )
        return validate_dij(value)


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

    for key, validator in [("pln", validate_pln), ("stf", validate_stf), ("dij", _validate_dij)]:
        value = mdict.get(key, None)
        if value is not None:
            try:
                if key == "dij":
                    patient_dict[key] = validator(value, patient_dict.get("pln"))
                else:
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


def _cube_to_sitk(cube: Any, ct: CT) -> Optional[sitk.Image]:
    """Convert a matRad (y, x, z) cube to a SimpleITK image on the CT grid, ``None`` if it differs."""
    cube = np.asarray(cube, dtype=float)
    reference = ct.cube_hu
    if reference.GetDimension() == 4:
        reference = reference[:, :, :, 0]
    if cube.ndim != 3:
        return None
    image = sitk.GetImageFromArray(np.transpose(cube, (2, 0, 1)))
    if image.GetSize() != reference.GetSize():
        return None
    image.CopyInformation(reference)
    return image


def _normalize_result(result: Any, ct: CT) -> Any:
    """Convert the (y, x, z) cubes of a matRad resultGUI to images on the CT grid.

    Cubes that do not match the CT grid are dropped, as their order cannot be verified.
    """
    if not isinstance(result, dict):
        return result
    normalized = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            image = _cube_to_sitk(value, ct)
            if image is None:
                logger.warning(f"resultGUI['{key}'] does not match the CT grid, ignored.")
                continue
            normalized[key] = image
        else:
            normalized[key] = value
    return normalized


def _result_dose_to_sitk(result: dict, ct: CT) -> Optional[sitk.Image]:
    """Extract a physical dose cube from a matRad resultGUI dict as a SimpleITK image."""
    if not isinstance(result, dict):
        return None

    cube = result.get("physicalDose", result.get("physical_dose", None))
    if cube is None:
        return None

    if isinstance(cube, sitk.Image):
        return cube

    dose_image = _cube_to_sitk(cube, ct)
    if dose_image is None:
        cube = np.asarray(cube, dtype=float)
        dose_image = sitk.GetImageFromArray(np.transpose(cube, (2, 0, 1)))
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
            patient_dict["result"] = _normalize_result(result, patient_dict["ct"])

        return patient_dict
