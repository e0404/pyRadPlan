"""Exporter for the HDF5 backend."""

import json
import os
import warnings
from typing import Any, ClassVar, Optional

import h5py
import SimpleITK as sitk

import pyRadPlan
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.cst import StructureSet
from pyRadPlan.ct import CT
from pyRadPlan.dij import Dij
from pyRadPlan.plan import Plan
from pyRadPlan.stf import SteeringInformation

from ..base import BaseExporter
from ._codec import FORMAT_NAME, FORMAT_VERSION, write_tree

_MODEL_KINDS = (
    (CT, "ct"),
    (StructureSet, "cst"),
    (Plan, "pln"),
    (SteeringInformation, "stf"),
    (Dij, "dij"),
)


class Hdf5Exporter(BaseExporter):
    """Exporter writing pyRadPlan objects losslessly into a single HDF5 file.

    See :mod:`pyRadPlan.io.hdf5` for the file layout. Objects that cannot be stored (pydantic
    models without a known kind, values that are not plain data) are skipped with a warning.
    """

    format: ClassVar[str] = "hdf5"
    name: ClassVar[str] = "HDF5 Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".h5", ".hdf5")
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

        tmp_path = f"{self.path}.tmp"
        try:
            contents = []
            with h5py.File(tmp_path, "w") as fh:
                for name, obj in objects.items():
                    if self._write_object(fh, name, obj, objects.get("ct")):
                        contents.append(name)
                fh.attrs["pyradplan_format"] = FORMAT_NAME
                fh.attrs["format_version"] = FORMAT_VERSION
                fh.attrs["pyradplan_version"] = pyRadPlan.__version__
                fh.attrs["contents"] = json.dumps(contents)
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _write_object(fh: h5py.File, name: str, obj: Any, ct: Optional[CT]) -> bool:
        if not name or "/" in name:
            warnings.warn(f"Skipping object with invalid name {name!r}.", stacklevel=3)
            return False
        kind = next((k for cls, k in _MODEL_KINDS if isinstance(obj, cls)), "tree")
        if kind == "tree" and isinstance(obj, PyRadPlanBaseModel):
            warnings.warn(
                f"Skipping {name!r}: {type(obj).__name__} cannot be rebuilt from an HDF5 file.",
                stacklevel=3,
            )
            return False
        group = fh.create_group(name)
        group.attrs["kind"] = kind
        try:
            if kind == "tree":
                write_tree(group, obj)
            elif kind == "cst" and ct is not None and (obj.ct_image is ct or obj.ct_image == ct):
                # the reference CT is stored once, in the "ct" group
                group.attrs["ct_ref"] = "ct"
                write_tree(group, obj.model_dump(exclude={"ct_image"}), jsonable=True)
            else:
                write_tree(group, obj.model_dump(), jsonable=True)
        except (TypeError, ValueError) as err:
            del fh[name]
            warnings.warn(f"Skipping {name!r}: {err}", stacklevel=3)
            return False
        return True
