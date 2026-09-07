"""Importer for the pickle backend."""

import pickle
from typing import ClassVar, Optional

import SimpleITK as sitk

from pyRadPlan.ct import validate_ct, CT
from pyRadPlan.cst import validate_cst, StructureSet

from ..base import BaseImporter


class PickleImporter(BaseImporter):
    """Importer for pyRadPlan ``.pkl`` archives.

    .. warning::
        Unpickling executes arbitrary code. Only load files from trusted sources.
    """

    format: ClassVar[str] = "pickle"
    name: ClassVar[str] = "Pickle Importer"
    extensions: ClassVar[tuple[str, ...]] = (".pkl", ".pickle")

    def __init__(self, path):
        super().__init__(path)
        self._data = None

    @property
    def data(self) -> dict:
        """Lazily unpickle and cache the archive contents."""
        if self._data is None:
            with open(self._require_path(), "rb") as fh:
                self._data = pickle.load(fh)
        return self._data

    def load_ct(self) -> CT:
        ct = self.data.get("ct")
        if ct is None:
            raise ValueError(f"No 'ct' found in {self.path}.")
        return validate_ct(ct)

    def load_cst(self, ct: Optional[CT] = None) -> Optional[StructureSet]:
        cst = self.data.get("cst")
        if cst is None:
            return None
        return validate_cst(cst)

    def load_dose(self) -> Optional[sitk.Image]:
        return self.data.get("dose")

    def load_data(self) -> dict:
        # Return the full unpickled mapping (passes through extras like pln/stf).
        return dict(self.data)
