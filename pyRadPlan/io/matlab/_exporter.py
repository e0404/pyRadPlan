"""Exporter for matRad-style MATLAB ``.mat`` files."""

from typing import ClassVar, Optional

import numpy as np
import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.result import PlanResult, validate_result

from ..base import ANY_OBJECT, BaseExporter
from . import _matfile


def _dose_to_matrad_cube(dose: sitk.Image) -> np.ndarray:
    """Convert a dose SimpleITK image to a matRad cube (y, x, z)."""
    # SimpleITK array is (z, y, x); matRad expects (y, x, z).
    return np.transpose(sitk.GetArrayFromImage(dose), (1, 2, 0))


class MatlabExporter(BaseExporter):
    """Exporter that writes pyRadPlan objects into a single matRad ``.mat`` file."""

    format: ClassVar[str] = "mat"
    name: ClassVar[str] = "MATLAB Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".mat",)
    container: ClassVar[bool] = True
    supports: ClassVar[frozenset[str]] = frozenset({ANY_OBJECT})

    def save(
        self,
        *,
        ct: Optional[CT] = None,
        cst: Optional[StructureSet] = None,
        dose: Optional[sitk.Image] = None,
        result: Optional[PlanResult] = None,
        **extra,
    ) -> None:
        mdict = {}

        if ct is not None:
            mdict["ct"] = ct.to_matrad(context="mat-file")

        if cst is not None:
            mdict["cst"] = cst.to_matrad(context="mat-file")

        # matRad stores the result under "resultGUI"; a full result wins over a bare dose.
        if result is not None:
            mdict["resultGUI"] = validate_result(result).to_matrad(
                context="mat-file", stf=extra.get("stf")
            )
        elif dose is not None:
            mdict["resultGUI"] = {"physicalDose": _dose_to_matrad_cube(dose)}

        for key, value in extra.items():
            if value is None:
                continue
            if hasattr(value, "to_matrad"):
                mdict[key] = value.to_matrad(context="mat-file")
            else:
                mdict[key] = value

        if not mdict:
            raise ValueError("Nothing to export: no ct, cst, dose, result or extra data provided.")

        _matfile.save(self.path, mdict)
