"""Exporter for the HDF5 backend."""

import json
import os
import warnings
from collections.abc import Mapping
from typing import Any, ClassVar, Optional

import h5py
import SimpleITK as sitk

import pyRadPlan
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.cst import StructureSet
from pyRadPlan.ct import CT
from pyRadPlan.dij import Dij
from pyRadPlan.plan import Plan
from pyRadPlan.result import PlanResult
from pyRadPlan.stf import SteeringInformation

from ..base import ANY_OBJECT, BaseExporter
from ._codec import FORMAT_NAME, FORMAT_VERSION, write_tree

_MODEL_KINDS = (
    (CT, "ct"),
    (StructureSet, "cst"),
    (Plan, "pln"),
    (SteeringInformation, "stf"),
    (Dij, "dij"),
    (PlanResult, "result"),
)


def _is_result_collection(obj: Any) -> bool:
    """Whether *obj* is a non-empty mapping of names to plan results (``Workspace.results``)."""
    return (
        isinstance(obj, Mapping)
        and bool(obj)
        and all(isinstance(k, str) and isinstance(v, PlanResult) for k, v in obj.items())
    )


class Hdf5Exporter(BaseExporter):
    """Exporter writing pyRadPlan objects losslessly into a single HDF5 file.

    See :mod:`pyRadPlan.io.hdf5` for the file layout. Objects that cannot be stored (pydantic
    models without a known kind, values that are not plain data) are skipped with a warning,
    or raise a ``ValueError`` (``TypeError`` for unstorable values) if ``strict`` is set.

    Parameters
    ----------
    path : str or os.PathLike
        Target file.
    strict : bool, default False
        Raise instead of skipping objects that cannot be stored.
    """

    format: ClassVar[str] = "hdf5"
    name: ClassVar[str] = "HDF5 Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".h5", ".hdf5")
    container: ClassVar[bool] = True
    supports: ClassVar[frozenset[str]] = frozenset({ANY_OBJECT})

    #: Default for the combined handler, whose cooperative __init__ chain skips ours.
    strict: bool = False

    def __init__(self, path, strict: bool = False):
        super().__init__(path)
        self.strict = strict

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

    def _skip(self, message: str, error: Optional[Exception] = None) -> bool:
        if self.strict:
            if error is not None:
                raise type(error)(message) from error
            raise ValueError(message)
        warnings.warn(message, stacklevel=4)
        return False

    def _write_object(self, fh: h5py.File, name: str, obj: Any, ct: Optional[CT]) -> bool:
        if not name or "/" in name:
            return self._skip(f"Skipping object with invalid name {name!r}.")
        kind = next((k for cls, k in _MODEL_KINDS if isinstance(obj, cls)), "tree")
        if kind == "tree" and _is_result_collection(obj):
            kind = "results"
        if kind == "tree" and isinstance(obj, PyRadPlanBaseModel):
            return self._skip(
                f"Skipping {name!r}: {type(obj).__name__} cannot be rebuilt from an HDF5 file."
            )
        group = fh.create_group(name)
        group.attrs["kind"] = kind
        try:
            if kind == "tree":
                write_tree(group, obj)
            elif kind == "results":
                write_tree(
                    group, {key: value.model_dump() for key, value in obj.items()}, jsonable=True
                )
            elif kind == "cst" and ct is not None and (obj.ct_image is ct or obj.ct_image == ct):
                # the reference CT is stored once, in the "ct" group
                group.attrs["ct_ref"] = "ct"
                write_tree(group, obj.model_dump(exclude={"ct_image"}), jsonable=True)
            else:
                write_tree(group, obj.model_dump(), jsonable=True)
        except (TypeError, ValueError) as err:
            del fh[name]
            return self._skip(f"Skipping {name!r}: {err}", err)
        return True
