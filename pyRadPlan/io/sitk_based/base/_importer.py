"""Shared importer base for SimpleITK-based backends (NIfTI, NRRD, MetaImage)."""

import os
from typing import ClassVar, Optional

from pyRadPlan.ct import validate_ct, CT
from pyRadPlan.cst import validate_cst, StructureSet

from ...base import BaseImporter
from ._serialize import read_image, read_sidecar, label_image_to_vois, _check_3d


class SitkImporterBase(BaseImporter):
    """Importer for a folder of SimpleITK images (or a single image file -> CT).

    A folder holds ``ct.<ext>``, ``dose.<ext>`` and ``segmentation.<ext>`` (+
    ``segmentation.json``). A single file with a known extension is read as a CT.
    Concrete subclasses set ``format`` and ``extensions``.
    """

    format: ClassVar[Optional[str]] = None
    name: ClassVar[str] = "SimpleITK Importer"
    extensions: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def handles_directory(cls, path) -> bool:
        try:
            entries = os.listdir(path)
        except OSError:
            return False
        return any(e.lower().endswith(ext) for e in entries for ext in cls.extensions)

    def _is_single_file(self) -> bool:
        return os.path.isfile(self._require_path())

    def _find(self, *stems: str) -> tuple[Optional[str], Optional[str]]:
        """Return ``(<dir>/<stem><ext>, ext)`` for the first match, else ``(None, None)``."""
        for stem in stems:
            for ext in self.extensions:
                candidate = os.path.join(self.path, f"{stem}{ext}")
                if os.path.isfile(candidate):
                    return candidate, ext
        return None, None

    def load_ct(self) -> CT:
        path = self._require_path()
        if os.path.isfile(path):
            image = read_image(path)
        else:
            ct_file, _ = self._find("ct")
            if ct_file is None:
                raise ValueError(f"No 'ct' image found in {self.path}.")
            image = read_image(ct_file)
        _check_3d(image)
        return validate_ct(cube_hu=image)

    def load_cst(self, ct: Optional[CT] = None) -> Optional[StructureSet]:
        if self._is_single_file():
            return None
        seg_file, ext = self._find("cst", "segmentation")
        if seg_file is None:
            return None
        if ct is None:
            ct = self.load_ct()
        label_image = read_image(seg_file)
        sidecar = read_sidecar(seg_file[: -len(ext)] + ".json")
        vois = label_image_to_vois(label_image, ct, sidecar)
        if not vois:
            return None
        return validate_cst(vois, ct)

    def load_dose(self):
        if self._is_single_file():
            return None
        dose_file, _ = self._find("dose")
        if dose_file is None:
            return None
        return read_image(dose_file)
