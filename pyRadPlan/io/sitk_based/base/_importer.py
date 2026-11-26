"""Shared importer base for SimpleITK-based backends (NIfTI, NRRD, MetaImage)."""

import os
from typing import ClassVar, Optional

from pyRadPlan.ct import validate_ct, CT
from pyRadPlan.cst import validate_cst, StructureSet

from ...base import BaseImporter
from ._serialize import (
    decode_segments,
    phase_reference_of,
    read_image,
    read_sidecar,
    segments_to_vois,
)


class SitkImporterBase(BaseImporter):
    """Importer for a folder of SimpleITK images (or a single image file -> CT).

    A folder holds ``ct.<ext>``, ``dose.<ext>`` and the structure set (``cst.<ext>`` or
    ``segmentation.<ext>``, also as ``.seg.<ext>``, + a ``.json`` sidecar). A 4D ``ct``
    is read as a 4D CT. The structure set is a label map, a 3D Slicer segmentation with
    layers or a 4D one-hot image; one file per phase (``cst_phase1.<ext>``, ...) gives 4D
    masks. A single file with a known extension is read as a CT. Concrete subclasses set
    ``format`` and ``extensions``.
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
        return validate_ct(cube_hu=image)

    def _cst_files(self) -> list[tuple[str, str]]:
        """Structure set files as ``(path, ext)``: one per phase, in phase order."""
        single, ext = self._find("cst.seg", "cst", "segmentation.seg", "segmentation")
        if single is not None:
            return [(single, ext)]
        files = []
        while True:
            n = len(files) + 1
            phase_file, ext = self._find(f"cst_phase{n}.seg", f"cst_phase{n}")
            if phase_file is None:
                return files
            files.append((phase_file, ext))

    def load_cst(self, ct: Optional[CT] = None) -> Optional[StructureSet]:
        if self._is_single_file():
            return None
        files = self._cst_files()
        if not files:
            return None
        if ct is None:
            ct = self.load_ct()
        reference = phase_reference_of(ct)
        per_phase = []
        for seg_file, ext in files:
            base = seg_file[: -len(ext)]
            base = base[: -len(".seg")] if base.endswith(".seg") else base
            sidecar = read_sidecar(base + ".json")
            per_phase.append(decode_segments(read_image(seg_file), reference, sidecar))
        vois = segments_to_vois(per_phase, ct)
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
