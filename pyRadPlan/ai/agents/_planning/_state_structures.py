"""Structure tools of the planning state: mapping, geometry and helper structures."""

from typing import Any, Literal, Optional

import numpy as np

from pyRadPlan.cst._voi import create_voi

from .._plan_agents import cst_geometry_summary
from . import _geometry as geo
from ._recipes import HelperRecipe
from ._state_base import PlanningError, UnknownVoiError, _NO_DOSE, sitk_array


class StructuresMixin:
    """Structure mapping, exclusion, geometry and helper structures."""

    # ---- structure tools ---------------------------------------------------------------
    def inspect_structures(self) -> dict:
        """List VOIs (type, volume, % inside targets) and the unmapped protocol names."""
        idx = {v.name: np.asarray(v.indices_numpy) for v in self.cst.vois}
        spacing = np.prod(self.ct.cube_hu.GetSpacing()) / 1000.0  # cc per voxel
        targets = [v.name for v in self.cst.vois if v.voi_type == "TARGET"]

        rows = []
        for v in self.cst.vois:
            ix = idx[v.name]
            row: dict[str, Any] = {
                "name": v.name,
                "type": v.voi_type,
                "volume_cc": round(float(ix.size * spacing), 2),
            }
            if ix.size == 0:
                row["empty"] = True
            if ix.size and v.name not in targets:
                ov = {
                    t: round(100 * np.intersect1d(ix, idx[t]).size / ix.size, 1) for t in targets
                }
                if any(ov.values()):
                    row["pct_inside_target"] = {t: p for t, p in ov.items() if p}
            rows.append(row)

        unmapped = sorted(
            {g.voi for g in self.protocol.goals if self.aliases.get(g.voi, g.voi) not in idx}
        )
        self._log("inspect_structures")
        return {"structures": rows, "aliases": self.aliases, "unmapped_protocol_names": unmapped}

    def map_structures(
        self,
        aliases: Optional[dict[str, str]] = None,
        voi_types: Optional[dict[str, str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> dict:
        """Map protocol names to VOIs, fix VOI types, drop helper / duplicate VOIs.

        aliases: protocol name -> VOI. voi_types: VOI -> type. exclude: VOIs to drop.
        """
        for voi in (aliases or {}).values():
            if voi not in self.voi_names:
                raise UnknownVoiError(voi, self.voi_names)
            if voi in self.created_helpers or self._voi(voi).voi_type == "HELPER":
                raise PlanningError(f"'{voi}' is a helper; goals must use clinical structures.")
        for name, t in (voi_types or {}).items():
            if name not in self.voi_names:
                raise UnknownVoiError(name, self.voi_names)
            if t.upper() not in ("TARGET", "OAR", "HELPER", "EXTERNAL"):
                raise PlanningError(f"Invalid type '{t}'. Use TARGET, OAR, HELPER or EXTERNAL.")
        for name in exclude or []:
            if name not in self.voi_names:
                raise UnknownVoiError(name, self.voi_names)
            if name in self.aliases.values() or name in (aliases or {}).values():
                raise PlanningError(f"'{name}' is mapped to a protocol goal; cannot exclude it.")

        targets_before = self._target_union()
        vois, excluded = [], []
        for v in self.cst.vois:
            if v.name in (exclude or []):
                excluded.append(v.name)
                continue
            if v.name in (voi_types or {}) and voi_types[v.name].upper() != v.voi_type:
                data = {
                    k: getattr(v, k)
                    for k in type(v).model_fields
                    if k not in ("voi_type", "overlap_priority", "color")
                }
                vois.append(create_voi(voi_type=voi_types[v.name].upper(), **data))
                continue
            vois.append(v)
        if not any(v.voi_type == "TARGET" for v in vois):
            raise PlanningError("At least one TARGET VOI must remain.")
        self.aliases.update(aliases or {})
        self.excluded.extend(excluded)
        self.cst = self.cst.model_copy(update={"vois": vois})
        if exclude or voi_types:
            self.dose = self.last_evaluation = None
            # The beamlets cover the union of the targets: while it does not grow (e.g. a
            # CTV dropped), they still cover every target, so the dose influence stays
            after = self._target_union()
            if targets_before is None or (after & ~targets_before).any():
                self.stf = self.dij = None
                self.beam_pool = None
        self._log("map_structures", aliases=aliases, voi_types=voi_types, exclude=exclude)
        return {
            "vois": [(v.name, v.voi_type) for v in self.cst.vois],
            "aliases": self.aliases,
            "excluded": self.excluded,
        }

    def _target_union(self) -> Optional[np.ndarray]:
        masks = [geo.as_bool(v.mask) for v in self.cst.vois if v.voi_type == "TARGET"]
        return np.logical_or.reduce(masks) if masks else None

    def _target_mask(self) -> np.ndarray:
        """Union of all targets; a PlanningError without one."""
        return np.logical_or.reduce([geo.as_bool(t.mask) for t in self._targets()])

    # ---- geometry tools --------------------------------------------------------------
    def _clinical_vois(self) -> list:
        return [
            v
            for v in self.cst.vois
            if v.voi_type != "HELPER" and v.name not in self.created_helpers
        ]

    def _voi(self, name: str):
        for v in self.cst.vois:
            if v.name == name:
                return v
        raise UnknownVoiError(name, self.voi_names)

    def _resolve(self, name: str) -> str:
        """Protocol name or VOI name -> VOI name."""
        name = self.aliases.get(name, name)
        self._voi(name)
        return name

    def _targets(self, target: Optional[str] = None) -> list:
        if target is not None:
            return [self._voi(self._resolve(target))]
        targets = [v for v in self.cst.vois if v.voi_type == "TARGET"]
        if not targets:
            raise PlanningError("No TARGET VOI in the structure set.")
        return targets

    def _patient_region(self) -> np.ndarray:
        """EXTERNAL VOI if present, else the union of all clinical VOIs."""
        ext = [v for v in self.cst.vois if v.voi_type == "EXTERNAL"]
        vois = (
            ext
            or [v for v in self._clinical_vois() if v.voi_type != "TARGET"]
            or (self._clinical_vois())
        )
        region = np.zeros_like(geo.as_bool(vois[0].mask))
        for v in vois + (self._targets() if not ext else []):
            region |= geo.as_bool(v.mask)
        return region

    def _normal_tissue_region(self) -> np.ndarray:
        """Patient more than normal_tissue_margin_mm outside all targets.

        Right next to a target the dose cannot fall below the prescription with photons,
        so a max goal on tissue touching the targets would fail by construction.
        """
        margin = self.normal_tissue_margin_mm
        key = (tuple((v.name, v.voi_type) for v in self._clinical_vois()), margin)
        if self._normal_tissue is None or self._normal_tissue[0] != key:
            region = self._patient_region()
            for t in self._targets():
                region &= ~geo.expand(t.mask, margin)
            self._normal_tissue = (key, region)
        return self._normal_tissue[1]

    def analyze_target_shape(self, target: Optional[str] = None) -> dict:
        """Describe each target's shape and OAR neighbourhood, with helper hints.

        Solidity, components, concavity and the OARs in it, distance / overlap to OARs.
        """
        out = {}
        for t in self._targets(target):
            others = {
                v.name: v.mask
                for v in self._clinical_vois()
                if v.voi_type not in ("TARGET", "EXTERNAL")
            }
            out[t.name] = geo.target_shape(t.mask, others, t.shape_parameters)
        self._log("analyze_target_shape", target=target)
        return out

    def create_helper(self, recipe: HelperRecipe) -> dict:
        """Create an optimization helper VOI from a recipe.

        Ring, margin (PRV / shrink), subtract / intersect / union of two VOIs, or isodose
        (current dose >= threshold, e.g. on a hot spot). Helpers carry objectives only;
        goals are evaluated on clinical VOIs. Overlaps are given both ways: % of the helper
        inside each clinical VOI, and % of each clinical VOI inside the helper.
        """
        name = recipe.name if recipe.name.startswith("opt_") else f"opt_{recipe.name}"
        if name in self.voi_names:
            raise PlanningError(f"VOI '{name}' already exists; choose another name.")
        patient = self._patient_region()

        def m(voi_name: str, margin: float = 0.0) -> np.ndarray:
            return geo.expand(self._voi(self._resolve(voi_name)).mask, margin)

        if recipe.kind == "ring":
            tgt = self._voi(self._resolve(recipe.target))
            remove = self._target_mask()
            for ex in recipe.exclude:
                remove |= m(ex.voi, ex.margin_mm)
            try:
                arr = geo.ring(
                    tgt.mask,
                    recipe.inner_mm,
                    recipe.outer_mm,
                    body=patient,
                    remove=remove,
                    skip_concavity=recipe.skip_concavity,
                )
            except ValueError as e:
                raise PlanningError(str(e)) from e
        elif recipe.kind == "margin":
            arr = m(recipe.source, recipe.margin_mm) & patient
        elif recipe.kind in ("subtract", "intersect", "union"):
            a, b = m(recipe.a), m(recipe.b, recipe.b_margin_mm)
            arr = {"subtract": a & ~b, "intersect": a & b, "union": a | b}[recipe.kind]
            arr &= patient
        elif recipe.kind == "isodose":
            if self.dose is None:
                raise PlanningError(_NO_DOSE)
            dose = sitk_array(self.dose)
            arr = (dose >= recipe.threshold) & patient
            if recipe.outside_targets:
                for t in self._targets():
                    arr &= ~geo.expand(t.mask, recipe.target_margin_mm)
        else:  # pragma: no cover - guarded by the schema
            raise PlanningError(f"Unknown recipe '{recipe.kind}'.")

        ref = self._targets()[0]
        vol = geo.volume_cc(arr, ref.mask)
        if vol < recipe.min_cc:
            raise PlanningError(
                f"Helper '{name}' would be {vol} cc (< {recipe.min_cc} cc); adjust the recipe."
            )
        data = {k: getattr(ref, k) for k in ("grid", "visible", "color")}
        # The optimizer gives overlapping voxels to the VOI with the lower overlap priority
        # number; with the HELPER default a helper would lose all of them, even to the body
        # outline. Taking the weakest of the per-voxel winners keeps the voxels where no
        # stronger VOI wins and takes none away from the VOIs that win them.
        winner = np.full(arr.shape, np.inf)
        for v in self._clinical_vois():
            mask = geo.as_bool(v.mask)
            winner[mask] = np.minimum(winner[mask], v.overlap_priority)
        covered = winner[arr & np.isfinite(winner)]
        if covered.size:
            data["overlap_priority"] = int(covered.max())
        helper = create_voi(voi_type="HELPER", name=name, mask=geo.to_mask(arr, ref.mask), **data)
        self.cst = self.cst.model_copy(update={"vois": [*self.cst.vois, helper]})
        self.created_helpers.append(name)

        dij_valid = self.dij is not None and bool(
            self._dij_region is not None and not (arr & ~self._dij_region).any()
        )
        if self.dij is not None and not dij_valid:
            self.stf = self.dij = self.dose = self.last_evaluation = None
        helper_in_voi, voi_in_helper = {}, {}
        for v in self._clinical_vois():
            vm = geo.as_bool(v.mask)
            if n := float((arr & vm).sum()):
                helper_in_voi[v.name] = round(100 * n / float(arr.sum()), 1)
                voi_in_helper[v.name] = round(100 * n / float(vm.sum()), 1)
        self._log("create_helper", name=name, recipe=recipe.model_dump())
        return {
            "name": name,
            "volume_cc": vol,
            "pct_of_helper_in_voi": helper_in_voi,
            "pct_of_voi_in_helper": voi_in_helper,
            "dose_influence": (
                "not computed yet"
                if self.dij is None and not self.num_dij_calcs
                else "kept"
                if dij_valid
                else "recomputed at next optimize"
            ),
            "note": "Helpers carry objectives only; goals are evaluated on clinical VOIs.",
        }

    def delete_helper(self, name: str) -> dict:
        """Delete a helper you created."""
        if name not in self.created_helpers:
            raise PlanningError(f"'{name}' is not a helper created by you: {self.created_helpers}")
        self.cst = self.cst.model_copy(
            update={"vois": [v for v in self.cst.vois if v.name != name]}
        )
        self.created_helpers.remove(name)
        self._log("delete_helper", name=name)
        return {"deleted": name, "helpers": self.created_helpers}

    def localize_dose_spots(
        self,
        kind: Literal["hot", "cold"],
        threshold: float,
        voi: Optional[str] = None,
        top_n: int = 3,
    ) -> dict:
        """Find the largest regions above (hot) / below (cold) a dose threshold.

        By default outside (hot) / inside (cold) the targets. Volume, extreme dose,
        location relative to the target center, VOIs hit.
        """
        if self.dose is None:
            raise PlanningError(_NO_DOSE)
        dose = sitk_array(self.dose)
        targets = self._target_mask()
        if voi is not None:
            region = geo.as_bool(self._voi(self._resolve(voi)).mask)
        elif kind == "hot":
            region = self._patient_region() & ~targets
        else:
            region = targets
        labels = {
            v.name: geo.as_bool(v.mask) for v in self._clinical_vois() if v.voi_type != "EXTERNAL"
        }
        out = geo.dose_spots(
            dose, region, threshold, kind, self._targets()[0].mask, labels, targets, top_n
        )
        out["region"] = voi or ("patient outside targets" if kind == "hot" else "targets")
        self._log("localize_dose_spots", kind=kind, threshold=threshold, voi=voi)
        return out

    def structure_geometry(self) -> dict:
        """Per-VOI center of mass, principal axes and shape in LPS mm."""
        return cst_geometry_summary(self.cst)
