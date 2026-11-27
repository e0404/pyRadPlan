"""Importer for the NumPy ``.npz`` backend."""

import json
from typing import ClassVar, Optional

import numpy as np
import SimpleITK as sitk

from pyRadPlan.ct import validate_ct, CT
from pyRadPlan.cst import validate_cst, validate_voi, StructureSet

from ..base import BaseImporter
from ._serialize import array_geom_to_image, geom_to_ct_kwargs


def _indices_to_mask(indices: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Rebuild a (possibly 4D) uint8 mask from C-order linear indices into ``reference``."""
    array = np.zeros(sitk.GetArrayViewFromImage(reference).shape, dtype=np.uint8)
    array.flat[indices] = 1
    mask = sitk.GetImageFromArray(array, isVector=False)
    mask.CopyInformation(reference)
    return mask


class NpzImporter(BaseImporter):
    """Importer for pyRadPlan ``.npz`` archives (CT, StructureSet, dose)."""

    format: ClassVar[str] = "npz"
    name: ClassVar[str] = "NumPy Importer"
    extensions: ClassVar[tuple[str, ...]] = (".npz",)

    def __init__(self, path):
        super().__init__(path)
        self._npz = None
        self._meta = None

    def _load(self):
        """Lazily open the archive and parse its JSON metadata (cached)."""
        if self._npz is None:
            self._npz = np.load(self._require_path(), allow_pickle=False)
            self._meta = json.loads(self._npz["meta"].item())
        return self._npz, self._meta

    def _has(self, key: str) -> bool:
        _, meta = self._load()
        return key in meta.get("contents", [])

    def load_ct(self) -> CT:
        npz, meta = self._load()
        if not self._has("ct"):
            raise ValueError(f"No 'ct' found in {self.path}.")
        block = meta["ct"]
        return validate_ct(cube_hu=npz[block["array"]], **geom_to_ct_kwargs(block))

    def load_cst(self, ct: Optional[CT] = None) -> Optional[StructureSet]:
        npz, meta = self._load()
        if not self._has("cst"):
            return None
        if ct is None:
            ct = self.load_ct()

        vois = []
        for vmeta in meta["cst"]["vois"]:
            mask = _indices_to_mask(npz[vmeta["indices_array"]], ct.cube_hu)
            kwargs = {
                "name": vmeta["name"],
                "voi_type": vmeta["voi_type"],
                "mask": mask,
                "ct_image": ct,
                "overlap_priority": vmeta["overlap_priority"],
                "visible": vmeta["visible"],
            }
            # Archives written before tissue properties were stored only carry alpha/beta.
            if vmeta.get("tissue_properties") is not None:
                kwargs["tissue_properties"] = vmeta["tissue_properties"]
            else:
                kwargs.update(alpha_x=vmeta["alpha_x"], beta_x=vmeta["beta_x"])
            # Files written before the rename carry the color under "visible_color",
            # where it may also be null; either way a VOI without one falls back to
            # its type's default.
            color = vmeta.get("color", vmeta.get("visible_color"))
            if color is not None:
                kwargs["color"] = color
            if "overlap_handling" in vmeta:
                kwargs["overlap_handling"] = vmeta["overlap_handling"]
            vois.append(validate_voi(**kwargs))
        return validate_cst(vois, ct)

    def load_dose(self) -> Optional[sitk.Image]:
        npz, meta = self._load()
        if not self._has("dose"):
            return None
        block = meta["dose"]
        return array_geom_to_image(npz[block["array"]], block)
