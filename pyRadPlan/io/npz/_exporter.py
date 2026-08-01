"""Exporter for the NumPy ``.npz`` backend."""

import json
from typing import ClassVar, Optional

import numpy as np
import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet

from ..base import BaseExporter
from ._serialize import FORMAT_VERSION, image_to_array_geom


class NpzExporter(BaseExporter):
    """Exporter writing CT, StructureSet and dose into a single ``.npz`` archive."""

    format: ClassVar[str] = "npz"
    name: ClassVar[str] = "NumPy Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".npz",)
    container: ClassVar[bool] = True

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

        # A StructureSet needs its CT to be reconstructed, so keep the file self-contained.
        if cst is not None and ct is None:
            ct = cst.ct_image

        arrays: dict = {}
        meta: dict = {"format_version": FORMAT_VERSION, "contents": []}

        if ct is not None:
            array, geom = image_to_array_geom(ct.cube_hu)
            arrays["ct_cube"] = array
            meta["ct"] = {"array": "ct_cube", **geom}
            meta["contents"].append("ct")

        if dose is not None:
            array, geom = image_to_array_geom(dose)
            arrays["dose_cube"] = array
            meta["dose"] = {"array": "dose_cube", **geom}
            meta["contents"].append("dose")

        if cst is not None:
            voi_meta = []
            for i, voi in enumerate(cst.vois):
                key = f"cst_voi{i}_indices"
                arrays[key] = np.asarray(voi.indices_numpy, dtype=np.int64)
                voi_meta.append(
                    {
                        "name": voi.name,
                        "voi_type": voi.voi_type,
                        "alpha_x": voi.alpha_x,
                        "beta_x": voi.beta_x,
                        "overlap_priority": voi.overlap_priority,
                        "visible": voi.visible,
                        "visible_color": (
                            list(voi.visible_color) if voi.visible_color is not None else None
                        ),
                        "indices_array": key,
                    }
                )
            meta["cst"] = {"vois": voi_meta}
            meta["contents"].append("cst")

        np.savez_compressed(self.path, meta=np.array(json.dumps(meta)), **arrays)
