"""Shared exporter base for SimpleITK-based backends (NIfTI, NRRD, MetaImage)."""

import os
from typing import ClassVar, Optional

import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet

from ...base import BaseExporter
from ._serialize import (
    write_image,
    cst_to_label_image,
    build_sidecar,
    write_sidecar,
    apply_interop_metadata,
    _check_3d,
)


class SitkExporterBase(BaseExporter):
    """Exporter writing CT/dose/cst as SimpleITK images.

    If the target path ends with a known extension (e.g. ``ct.nii.gz``) a single
    image is written (the CT, or the dose). Otherwise the target is a directory
    holding ``ct.<ext>``, ``dose.<ext>`` and a ``segmentation.<ext>`` label map
    (+ ``segmentation.json``). Concrete subclasses set ``format``/``extensions``.
    """

    format: ClassVar[Optional[str]] = None
    name: ClassVar[str] = "SimpleITK Exporter"
    extensions: ClassVar[tuple[str, ...]] = ()
    container: ClassVar[bool] = False
    #: Stamp pyradplan/Slicer keys into the label image (NRRD/MetaImage only).
    stamp_interop_metadata: ClassVar[bool] = False

    def _is_single_file_target(self) -> bool:
        name = self.path.lower()
        return any(name.endswith(ext) for ext in self.extensions)

    def save(
        self,
        *,
        ct: Optional[CT] = None,
        cst: Optional[StructureSet] = None,
        dose: Optional[sitk.Image] = None,
        **extra,
    ) -> None:
        if ct is None and cst is None and dose is None:
            raise ValueError("Nothing to export: provide at least one of ct, cst, dose.")

        # A label map needs the CT geometry; keep a folder self-contained.
        if cst is not None and ct is None:
            ct = cst.ct_image

        ext = self.extensions[0]

        if self._is_single_file_target():
            if cst is not None:
                raise ValueError(
                    "A single-file target cannot hold a StructureSet; use a directory."
                )
            if ct is not None and dose is not None:
                raise ValueError(
                    "A single-file target holds one image, but both a ct and a dose were "
                    "provided; use a directory to write both."
                )
            image = ct.cube_hu if ct is not None else dose
            _check_3d(image)
            write_image(image, self.path)
            return

        os.makedirs(self.path, exist_ok=True)

        if ct is not None:
            _check_3d(ct.cube_hu)
            write_image(ct.cube_hu, os.path.join(self.path, f"ct{ext}"))

        if dose is not None:
            _check_3d(dose)
            write_image(dose, os.path.join(self.path, f"dose{ext}"))

        if cst is not None:
            label_image, extents = cst_to_label_image(cst)
            if label_image is not None:
                if self.stamp_interop_metadata:
                    apply_interop_metadata(label_image, cst, extents)
                write_image(label_image, os.path.join(self.path, f"cst{ext}"))
                write_sidecar(os.path.join(self.path, "cst.json"), build_sidecar(cst))
