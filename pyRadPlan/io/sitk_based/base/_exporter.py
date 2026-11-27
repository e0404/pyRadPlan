"""Shared exporter base for SimpleITK-based backends (NIfTI, NRRD, MetaImage)."""

import os
import re
import warnings
from typing import ClassVar, Literal, Optional

import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.result import PlanResult

from ...base import BaseExporter
from ._serialize import (
    SEGMENTATION_MODES,
    apply_interop_metadata,
    build_sidecar,
    interop_entries,
    masks_to_label_image,
    masks_to_layers,
    masks_to_onehot,
    num_phases,
    overlapping_vois,
    phase_masks,
    phase_reference,
    write_image,
    write_sidecar,
    write_slicer_segmentation,
)


class SitkExporterBase(BaseExporter):
    """Exporter writing CT/dose/cst as SimpleITK images.

    If the target path ends with a known extension (e.g. ``ct.nii.gz``) a single
    image is written (the CT, or the dose). Otherwise the target is a directory
    holding ``ct.<ext>``, ``dose.<ext>`` and the structure set (``cst.<ext>`` + ``cst.json``).
    A 4D CT or dose is written as one 4D image (phases along the 4th axis). Concrete
    subclasses set ``format``/``extensions``.

    Parameters
    ----------
    path : str or os.PathLike
        Target file or directory.
    segmentation : {"labelmap", "auto", "layers"}, default "labelmap"
        How a structure set is stored. ``"labelmap"`` writes one label per voxel, so
        overlapping structures lose voxels (a warning names the affected VOIs).
        ``"layers"`` keeps all overlaps: a 3D Slicer segmentation ``cst.seg.nrrd`` with one
        layer per set of non-overlapping segments for NRRD, a 4D one-hot image with one
        channel per VOI for NIfTI and MetaImage. ``"auto"`` writes a label map unless
        VOIs overlap. The image and ``cst.json`` hold the VOI names, types and colors; the
        importers read all variants. Exporting a structure set removes the structure set
        files (``cst*``) of earlier exports from the folder.

    Notes
    -----
    A 4D structure set (VOI masks with a phase axis) is written as one file per phase,
    ``cst_phase1.<ext>``, ``cst_phase2.<ext>``, ... (each with its own ``.json``), and
    re-imported as 4D masks. VOIs that are empty in every phase of a label map are lost.
    """

    format: ClassVar[Optional[str]] = None
    name: ClassVar[str] = "SimpleITK Exporter"
    extensions: ClassVar[tuple[str, ...]] = ()
    container: ClassVar[bool] = False
    supports: ClassVar[frozenset[str]] = frozenset({"ct", "cst", "dose"})
    #: Stamp pyradplan/Slicer keys into the label image (NRRD/MetaImage only).
    stamp_interop_metadata: ClassVar[bool] = False
    #: Write ``segmentation="layers"`` as a 3D Slicer ``.seg.nrrd`` (else a 4D one-hot image).
    slicer_layers: ClassVar[bool] = False

    #: Default used when the exporter is reached via the combined handler (whose
    #: cooperative __init__ chain does not pass through SitkExporterBase.__init__).
    segmentation: str = "labelmap"

    def __init__(
        self,
        path,
        segmentation: Literal["labelmap", "auto", "layers"] = "labelmap",
    ):
        super().__init__(path)
        if segmentation not in SEGMENTATION_MODES:
            raise ValueError(f"segmentation must be one of {SEGMENTATION_MODES}.")
        self.segmentation = segmentation

    def _is_single_file_target(self) -> bool:
        name = self.path.lower()
        return any(name.endswith(ext) for ext in self.extensions)

    def save(
        self,
        *,
        ct: Optional[CT] = None,
        cst: Optional[StructureSet] = None,
        dose: Optional[sitk.Image] = None,
        result: Optional[PlanResult] = None,
        **extra,
    ) -> None:
        # These formats store single images, not a full result.
        dose = self.resolve_dose(dose, result)
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
            write_image(ct.cube_hu if ct is not None else dose, self.path)
            return

        os.makedirs(self.path, exist_ok=True)

        if ct is not None:
            write_image(ct.cube_hu, os.path.join(self.path, f"ct{ext}"))

        if dose is not None:
            write_image(dose, os.path.join(self.path, f"dose{ext}"))

        if cst is not None:
            self._remove_stale_cst()
            if cst.vois:
                self._save_cst(cst, ext)

    def _remove_stale_cst(self) -> None:
        """Delete structure set files of earlier exports (all variants and sidecars)."""
        exts = "|".join(re.escape(e) for e in self.extensions)
        pattern = re.compile(
            rf"cst(_phase\d+)?(\.seg)?({exts})|cst(_phase\d+)?\.json", re.IGNORECASE
        )
        for name in os.listdir(self.path):
            full = os.path.join(self.path, name)
            if pattern.fullmatch(name) and os.path.isfile(full):
                os.remove(full)

    def _save_cst(self, cst: StructureSet, ext: str) -> None:
        phases = num_phases(cst)
        reference = phase_reference(cst)
        masks_per_phase = [phase_masks(cst, p) for p in range(phases)]

        mode = self.segmentation
        overlapping = overlapping_vois(masks_per_phase, cst)
        if mode == "auto":
            mode = "layers" if overlapping else "labelmap"

        lost: list[str] = []
        for phase, masks in enumerate(masks_per_phase):
            stem = "cst" if phases == 1 else f"cst_phase{phase + 1}"
            if mode == "labelmap":
                lost += self._write_label_map(cst, masks, reference, stem, ext)
            else:
                self._write_layers(cst, masks, reference, stem, ext)

        if lost:
            names = ", ".join(dict.fromkeys(lost))
            warnings.warn(
                f"The label map holds one label per voxel; overlapping voxels of {names} were "
                "assigned to other structures and are missing. Use segmentation='layers' (or "
                "'auto') to keep overlapping structures.",
                UserWarning,
                stacklevel=4,
            )

    def _write_label_map(self, cst, masks, reference, stem, ext) -> list[str]:
        image, extents, lost = masks_to_label_image(cst, masks, reference)
        if self.stamp_interop_metadata:
            apply_interop_metadata(image, cst, extents)
        write_image(image, os.path.join(self.path, f"{stem}{ext}"))
        write_sidecar(os.path.join(self.path, f"{stem}.json"), build_sidecar(cst))
        return lost

    def _write_layers(self, cst, masks, reference, stem, ext) -> None:
        sidecar = build_sidecar(cst, "layers" if self.slicer_layers else "onehot")
        if self.slicer_layers:
            arrays, layers, extents = masks_to_layers(masks, reference)
            entries = interop_entries(cst, extents, layers, encoding="layers")
            write_slicer_segmentation(
                os.path.join(self.path, f"{stem}.seg{ext}"), arrays, reference, entries
            )
        else:
            image = masks_to_onehot(masks, reference)
            if self.stamp_interop_metadata:
                for key, value in interop_entries(cst, encoding="onehot").items():
                    image.SetMetaData(key, value)
            write_image(image, os.path.join(self.path, f"{stem}{ext}"))
        write_sidecar(os.path.join(self.path, f"{stem}.json"), sidecar)
