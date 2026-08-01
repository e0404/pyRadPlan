"""Exporter for matRad-style MATLAB ``.mat`` files."""

from typing import ClassVar, Optional

import numpy as np
import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet

from ..base import BaseExporter
from . import _matfile


# TODO: This will be deprecated as soon as result object is done!!!
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

    def save(
        self,
        *,
        ct: Optional[CT] = None,
        cst: Optional[StructureSet] = None,
        dose: Optional[sitk.Image] = None,
        **extra,
    ) -> None:
        mdict = {}

        if ct is not None:
            mdict["ct"] = ct.to_matrad(context="mat-file")

        if cst is not None:
            mdict["cst"] = cst.to_matrad(context="mat-file")

        if dose is not None:
            mdict["resultGUI"] = {"physicalDose": _dose_to_matrad_cube(dose)}

        for key, value in extra.items():
            if value is None:
                continue
            if hasattr(value, "to_matrad"):
                mdict[key] = value.to_matrad(context="mat-file")
            else:
                mdict[key] = value

        if not mdict:
            raise ValueError("Nothing to export: no ct, cst, dose or extra data provided.")

        _matfile.save(self.path, mdict)
