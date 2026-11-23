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
from pyRadPlan.result import PlanResult

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
        try:
            patient_dict["result"] = _validate_matrad_result(result, patient_dict.get("ct"))
            if remove_matrad_structures:
                mdict.pop("resultGUI")
        except ValueError as exc:
            logger.warning("resultGUI present but could not be validated: %s", exc)

    return patient_dict


def _validate_matrad_result(result: Any, ct: Optional[CT] = None) -> PlanResult:
    """Validate a matRad resultGUI, placing its cubes on the grid of the CT (if any).

    Cubes that do not match the CT grid are dropped, as their order cannot be verified.
    """
    if ct is None:
        return PlanResult.from_matrad(result)
    # matRad stores its result cubes on the CT grid, but without any geometry.
    reference = ct.cube_hu
    if reference.GetDimension() == 4:
        reference = reference[:, :, :, 0]
    if isinstance(result, dict):
        # matRad cubes are (y, x, z)
        nx, ny, nz = reference.GetSize()
        mismatching = [
            key
            for key, value in result.items()
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape != (ny, nx, nz)
        ]
        if mismatching:
            logger.warning("resultGUI fields %s do not match the CT grid, ignored.", mismatching)
            result = {key: value for key, value in result.items() if key not in mismatching}
    return PlanResult.from_matrad(result, reference_image=reference)


def _result_dose_to_sitk(result: Optional[PlanResult]) -> Optional[sitk.Image]:
    """Extract the physical dose of a plan result as a SimpleITK image."""
    if result is None or "physical_dose" not in result:
        return None
    return result["physical_dose"].values


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
        return _result_dose_to_sitk(self.load_result())

    def load_result(self) -> Optional[PlanResult]:
        result = self.mdict.get("resultGUI", None)
        if result is None:
            return None
        ct = self.load_ct() if self.mdict.get("ct", None) is not None else None
        return _validate_matrad_result(result, ct)

    def load_data(self) -> dict:
        # Operate on a copy so the cached dictionary is not consumed.
        mdict = dict(self.mdict)
        patient_dict = validate_matrad_patient(mdict, remove_matrad_structures=True)

        # Expose the result's physical dose as "dose" as well, for consumers that
        # only handle a single image.
        dose = _result_dose_to_sitk(patient_dict.get("result"))
        if dose is not None:
            patient_dict["dose"] = dose

        return patient_dict
