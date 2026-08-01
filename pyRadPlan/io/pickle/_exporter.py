"""Exporter for the pickle backend."""

import pickle
from typing import ClassVar, Optional

import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet

from ..base import BaseExporter


class PickleExporter(BaseExporter):
    """Exporter pickling pyRadPlan objects into a single ``.pkl`` file."""

    format: ClassVar[str] = "pickle"
    name: ClassVar[str] = "Pickle Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".pkl", ".pickle")
    container: ClassVar[bool] = True

    def save(
        self,
        *,
        ct: Optional[CT] = None,
        cst: Optional[StructureSet] = None,
        dose: Optional[sitk.Image] = None,
        **extra,
    ) -> None:
        objects: dict = {}
        if ct is not None:
            objects["ct"] = ct
        if cst is not None:
            objects["cst"] = cst
        if dose is not None:
            objects["dose"] = dose
        objects.update({k: v for k, v in extra.items() if v is not None})

        if not objects:
            raise ValueError("Nothing to export: provide ct, cst, dose or extra objects.")

        with open(self.path, "wb") as fh:
            pickle.dump(objects, fh, protocol=pickle.HIGHEST_PROTOCOL)
