"""Importer for the HDF5 backend."""

import json
from typing import Any, Callable, ClassVar, Optional

import h5py
import SimpleITK as sitk

from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.dij import validate_dij
from pyRadPlan.machines import create_bld
from pyRadPlan.plan import validate_pln
from pyRadPlan.stf import validate_stf

from ..base import BaseImporter
from ._codec import FORMAT_NAME, FORMAT_VERSION, read_tree


def _validate_pln(tree: Any) -> Any:
    prop_stf = tree.get("prop_stf") if isinstance(tree, dict) else None
    if isinstance(prop_stf, dict) and isinstance(prop_stf.get("blds"), list):
        prop_stf["blds"] = [create_bld(bld) for bld in prop_stf["blds"]]
    return validate_pln(tree)


_FACTORIES: dict[str, Callable[[Any], Any]] = {
    "ct": validate_ct,
    "cst": validate_cst,
    "pln": _validate_pln,
    "stf": validate_stf,
    "dij": validate_dij,
    "tree": lambda tree: tree,
}


def _attr(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class Hdf5Importer(BaseImporter):
    """Importer for pyRadPlan HDF5 workspaces (see :mod:`pyRadPlan.io.hdf5`).

    Loading never executes code.
    """

    format: ClassVar[str] = "hdf5"
    name: ClassVar[str] = "HDF5 Importer"
    extensions: ClassVar[tuple[str, ...]] = (".h5", ".hdf5")

    def __init__(self, path):
        super().__init__(path)
        self._data: Optional[dict] = None

    def _open(self) -> h5py.File:
        try:
            fh = h5py.File(self._require_path(), "r")
        except OSError as err:
            raise ValueError(f"{self.path} is not a pyRadPlan HDF5 workspace: {err}") from err
        try:
            if _attr(fh.attrs.get("pyradplan_format")) != FORMAT_NAME:
                raise ValueError(f"{self.path} is not a pyRadPlan HDF5 workspace.")
            version = fh.attrs.get("format_version")
            if version is None or int(version) != FORMAT_VERSION:
                raise ValueError(
                    f"{self.path} has unsupported format version {version!r}; "
                    f"this version of pyRadPlan reads version {FORMAT_VERSION}."
                )
        except Exception:
            fh.close()
            raise
        return fh

    @staticmethod
    def _names(fh: h5py.File) -> list[str]:
        return json.loads(_attr(fh.attrs["contents"]))

    def _load_object(self, fh: h5py.File, name: str, loaded: Optional[dict] = None) -> Any:
        group = fh[name]
        kind = _attr(group.attrs["kind"])
        if kind not in _FACTORIES:
            raise ValueError(f"Unknown object kind {kind!r} for {name!r} in {self.path}.")
        ct_ref = _attr(group.attrs.get("ct_ref"))
        if kind == "cst" and ct_ref is not None:
            if loaded is not None and ct_ref in loaded:
                ct = loaded[ct_ref]
            else:
                ct = self._load_object(fh, ct_ref)
            return validate_cst(read_tree(group), ct=ct)
        return _FACTORIES[kind](read_tree(group))

    @property
    def data(self) -> dict:
        """Lazily read and cache all objects of the workspace."""
        if self._data is None:
            loaded: dict = {}
            with self._open() as fh:
                for name in self._names(fh):
                    loaded[name] = self._load_object(fh, name, loaded)
            self._data = loaded
        return self._data

    def _single(self, name: str) -> Any:
        if self._data is not None:
            return self._data.get(name)
        with self._open() as fh:
            if name not in self._names(fh):
                return None
            return self._load_object(fh, name)

    def load_ct(self) -> CT:
        ct = self._single("ct")
        if ct is None:
            raise ValueError(f"No 'ct' found in {self.path}.")
        return ct

    def load_cst(self, ct: Optional[CT] = None) -> Optional[StructureSet]:
        return self._single("cst")

    def load_dose(self) -> Optional[sitk.Image]:
        return self._single("dose")

    def load_data(self) -> dict:
        return dict(self.data)
