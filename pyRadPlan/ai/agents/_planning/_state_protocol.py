"""Protocol tools of the planning state: goals, evaluation, feasibility and tables."""

from typing import Optional

import numpy as np

from pyRadPlan.analysis import QICollection

from . import _geometry as geo
from ._constraints import (
    D_RE,
    V_RE,
    Goal,
    Protocol,
    available_constraint_tables,
    load_constraint_table,
    site_goals,
)
from ._evaluation import evaluate_protocol
from ._state_base import NORMAL_TISSUE, PlanningError, UnknownVoiError, _NO_DOSE

# Dose fall-off (% of the target floor per mm) rated ok / steep by check_feasibility
_FALLOFF_OK = 5.0
_FALLOFF_STEEP = 10.0
# D<x> goals with x <= this are treated like max-dose goals by check_feasibility
_NEAR_MAX_PCT = 10.0
# A mean goal is rated tight when the part of the VOI outside the targets has to stay
# below this fraction of the goal value
_MEAN_TIGHT = 0.5


class ProtocolMixin:
    """Protocol goals, their evaluation, feasibility checks and constraint tables."""

    def set_protocol(self, protocol: Protocol) -> list[str]:
        """Replace the goals, re-evaluate every plan and saved version; return the changes."""
        old = {g.tagged_label for g in self.protocol.goals}
        new = {g.tagged_label for g in protocol.goals}
        self.protocol = protocol
        current = (self.cst, self.dose)
        try:
            for p in self.plans:
                self.cst, self.dose = p.cst, p.dose
                p.evaluation = {**p.evaluation, **self._evaluate()}
            for snap in self.snapshots.values():
                if snap["dose"] is not None and snap["evaluation"] is not None:
                    self.cst, self.dose = snap["cst"], snap["dose"]
                    snap["evaluation"] = {**snap["evaluation"], **self._evaluate()}
        finally:
            self.cst, self.dose = current
        if self.dose is not None and self.last_evaluation is not None:
            self.last_evaluation = {**self.last_evaluation, **self._evaluate()}
        return [f"removed {g}" for g in sorted(old - new)] + [
            f"added {g}" for g in sorted(new - old)
        ]

    def _full_protocol(self) -> Protocol:
        """Return the protocol with the implicit goals (NormalTissue)."""
        return self.protocol.model_copy(
            update={"goals": [*self.protocol.goals, *self._implicit_goals()]}
        )

    def _evaluate(self, qis: Optional[QICollection] = None) -> dict:
        qis = self._qis() if qis is None else qis
        return evaluate_protocol(self._full_protocol(), qis, self.aliases)

    def _evaluate_dose(self) -> dict:
        """Evaluate the goals on the current dose, with the NormalTissue QIs."""
        qis = self._qis()
        evaluation = self._evaluate(qis)
        evaluation["normal_tissue"] = {
            k: round(float(qis[NORMAL_TISSUE][k].value), 2)
            for k in ("mean", "D50", "D5", "D2", "max")
        }
        return evaluation

    def _implicit_goals(self) -> list[Goal]:
        rx = self._fractionation()
        rx = (
            rx["prescribed_total_gy"]
            if self.pln.dose_convention == "total"
            else (rx["dose_per_fraction_gy"])
        )
        # The plan's prescription can be a default unrelated to the protocol (e.g. 60 Gy
        # for a 70 Gy boost), so the highest target floor counts as a dose level too
        level = max([rx or 0.0, *self._dose_floors(targets_only=True).values()])
        if not level:
            return []
        return [
            Goal(
                voi=NORMAL_TISSUE,
                metric="max",
                op="<=",
                value=round(level, 4),
                hard=False,
                source=(
                    "implicit: no dose above the highest target dose level more than "
                    f"{self.normal_tissue_margin_mm:g} mm outside the targets"
                ),
            )
        ]

    def analyze_qis(
        self,
        vois: Optional[list[str]] = None,
        ref_vols: Optional[list[float]] = None,
        ref_doses: Optional[list[float]] = None,
    ) -> dict:
        """Return QIs of the current dose per VOI (all if omitted), incl. NormalTissue.

        Mean, max, min, std, D<x> and V<x>Gy.
        """
        if self.dose is None:
            raise PlanningError(_NO_DOSE)
        names = [*self.voi_names, NORMAL_TISSUE]
        unknown = [v for v in vois or [] if v not in names]
        if unknown:
            raise UnknownVoiError(unknown[0], names)
        qis = self._qis(ref_vols=ref_vols, ref_doses=ref_doses)
        return {
            n: {k: round(float(q.value), 3) for k, q in qis[n].items() if np.isfinite(q.value)}
            for n in vois or names
        }

    def _qis(self, ref_vols=None, ref_doses=None) -> QICollection:
        masks = {v.name: v.mask for v in self.cst.vois}
        masks[NORMAL_TISSUE] = geo.to_mask(self._normal_tissue_region(), self._targets()[0].mask)
        return QICollection.from_masks(
            masks,
            self.dose,
            ref_vols=sorted({*(ref_vols or [50, 5, 2, 98]), *self.protocol.ref_vols()}),
            ref_doses=sorted({*(ref_doses or []), *self.protocol.ref_doses()}) or None,
        )

    def _dose_floors(self, targets_only: bool = False) -> dict[str, float]:
        """VOI name -> highest dose floor (min / D>= goal) of the protocol."""
        floors: dict[str, float] = {}
        for g in self.protocol.goals:
            if g.op == ">=" and (g.metric == "min" or g.metric.startswith("D")):
                try:
                    name = self._resolve(g.voi)
                except PlanningError:
                    continue
                if targets_only and self._voi(name).voi_type != "TARGET":
                    continue
                floors[name] = max(floors.get(name, 0.0), g.value)
        return floors

    def check_feasibility(self) -> dict:
        """Check target dose floors against OAR limits.

        Conflicts (overlap with an OAR limit below the floor) and the dose fall-off
        (%/mm) each OAR goal needs. Overlaps are given both ways: % of the OAR inside the
        target, and % of the target inside the OAR. optimizer_assigns_overlap_to: the VOI
        whose objectives see the overlap voxels (lower overlap priority number wins, "both" on
        a tie); goals are still evaluated on the full VOIs. mean_budgets: for mean goals, the
        mean the target overlap alone forces and the mean left for the rest of the VOI.
        """
        floors = self._dose_floors()
        if not floors:
            return {"checks": [], "note": "No target dose floors (min / D>= goals) in protocol."}

        checks = []
        for tname, floor in floors.items():
            tmask = self._voi(tname).mask
            t = geo.as_bool(tmask)
            dist = geo.signed_distance_mm(tmask)
            vox = geo.voxel_cc(tmask)
            spacing = min(tmask.GetSpacing())
            for g in self.protocol.goals:
                if g.op != "<=":
                    continue
                try:
                    oname = self._resolve(g.voi)
                except PlanningError:
                    continue
                if oname == tname:
                    continue
                o = geo.as_bool(self._voi(oname).mask)
                if not o.any():
                    continue
                ov = o & t
                pct_oar_in_target = 100 * float(ov.sum()) / o.sum()
                pct_target_in_oar = 100 * float(ov.sum()) / t.sum()
                min_dist = float(max(dist[o].min(), 0.0))
                entry = {
                    "target": tname,
                    "goal": g.label,
                    "target_floor": floor,
                    "overlap_cc": round(float(ov.sum()) * vox, 2),
                    "pct_of_oar_in_target": round(pct_oar_in_target, 1),
                    "pct_of_target_in_oar": round(pct_target_in_oar, 1),
                    "min_distance_mm": round(min_dist, 1),
                }
                if ov.any():
                    tp = self._voi(tname).overlap_priority
                    op = self._voi(oname).overlap_priority
                    entry["optimizer_assigns_overlap_to"] = (
                        tname if tp < op else oname if op < tp else "both"
                    )
                if m := V_RE.match(g.metric):
                    thr = float(m.group(1))
                    if thr < floor and pct_oar_in_target > g.value:
                        entry["verdict"] = "conflict"
                        entry["why"] = (
                            f"{pct_oar_in_target:.1f}% of {oname} lies in {tname}, which must get "
                            f">= {floor:g}; V{thr:g}Gy <= {g.value:g}% cannot hold."
                        )
                    else:
                        entry["verdict"] = "ok"
                elif g.metric in ("max",) or (
                    (dm := D_RE.match(g.metric)) and float(dm.group(1)) <= _NEAR_MAX_PCT
                ):
                    if ov.any() and g.value < floor:
                        entry["verdict"] = "conflict"
                        entry["why"] = (
                            f"{oname} overlaps {tname}: the overlap must be >= {floor:g} "
                            f"and <= {g.value:g} at the same time."
                        )
                    else:
                        gap = floor - g.value
                        grad = 100 * gap / floor / max(min_dist, spacing)
                        entry["required_falloff_pct_per_mm"] = round(grad, 1)
                        entry["verdict"] = (
                            "ok"
                            if grad <= _FALLOFF_OK
                            else "steep"
                            if grad <= _FALLOFF_STEEP
                            else "very steep"
                        )
                elif g.metric == "mean":
                    entry["verdict"] = "see mean_budgets"
                else:
                    entry["verdict"] = "not assessed (large-volume goal)"
                checks.append(entry)
        budgets = self._mean_budgets(floors)
        conflicts = [c for c in checks + budgets if c["verdict"] == "conflict"]
        self._log("check_feasibility", conflicts=len(conflicts))
        return {
            "checks": checks,
            "mean_budgets": budgets,
            "summary": {
                "conflicts": len(conflicts),
                "steep": sum(c["verdict"] in ("steep", "very steep") for c in checks),
                "tight_mean_budgets": sum(b["verdict"] == "tight" for b in budgets),
            },
            "advice": (
                "Conflicts cannot be solved by optimization: split the target (subtract / "
                "intersect helper with a lower goal in the overlap) or report the trade-off. "
                f"Photon dose fall-off of ~{_FALLOFF_OK:g}-{_FALLOFF_STEEP:g} %/mm is "
                "achievable only with good beam "
                "arrangements; beyond that expect the OAR goal to fail. Mean goals are "
                "evaluated on the whole VOI, target overlap included, while its objectives see "
                "only the voxels it keeps after overlap priorities (objectives_see_cc): aim "
                "the part outside the targets at max_mean_outside_targets, which still has "
                "to take the dose fall-off around the targets."
            ),
        }

    def _mean_budgets(self, floors: dict[str, float]) -> list[dict]:
        """Per mean goal: the mean forced by the target floors, the mean left outside them."""
        goals = [g for g in self.protocol.goals if g.metric == "mean" and g.op == "<="]
        if not goals:
            return []
        ref = self._voi(next(iter(floors))).mask
        floor_map = np.zeros(geo.as_bool(ref).shape)
        for tname, floor in floors.items():
            t = geo.as_bool(self._voi(tname).mask)
            floor_map[t] = np.maximum(floor_map[t], floor)
        vox = geo.voxel_cc(ref)
        apply_overlap = bool((self.pln.prop_opt or {}).get("apply_overlap", True))
        opt_cst = self.cst.apply_overlap_priorities() if apply_overlap else self.cst
        seen = {v.name: int(geo.as_bool(v.mask).sum()) for v in opt_cst.vois}

        out = []
        for g in goals:
            try:
                name = self._resolve(g.voi)
            except PlanningError:
                continue
            if name in floors:
                continue
            o = geo.as_bool(self._voi(name).mask)
            n = int(o.sum())
            if not n:
                continue
            forced = float(floor_map[o].sum())
            n_overlap = int((floor_map[o] > 0).sum())
            floor_mean = forced / n
            entry = {
                "goal": g.label,
                "voi_cc": round(n * vox, 1),
                "target_overlap_cc": round(n_overlap * vox, 1),
                "mean_from_target_floors": round(floor_mean, 2),
            }
            if floor_mean > g.value:
                entry["verdict"] = "conflict"
                entry["why"] = (
                    f"The target overlap alone gives {name} a mean of at least "
                    f"{floor_mean:.3g} > {g.value:g}."
                )
            elif n_overlap < n:
                rest = (g.value * n - forced) / (n - n_overlap)
                entry["max_mean_outside_targets"] = round(rest, 2)
                entry["verdict"] = "tight" if rest < _MEAN_TIGHT * g.value else "ok"
            else:
                entry["verdict"] = "ok"
            if seen.get(name, n) < n:
                entry["objectives_see_cc"] = round(seen[name] * vox, 1)
            out.append(entry)
        return out

    # ---- clinical knowledge tools ----------------------------------------------------
    def list_constraint_tables(self) -> dict:
        """List constraint tables, their sites and the plan fractionation."""
        out = {}
        for tid in available_constraint_tables():
            t = load_constraint_table(tid)
            out[tid] = {
                "name": t.name,
                "dose_reference": t.dose_reference.type,
                "sites": {
                    s: s_.description
                    + (
                        f" [{s_.prescription.total_dose_gy:g} Gy / "
                        f"{s_.prescription.num_fractions} fx]"
                        if s_.prescription
                        else ""
                    )
                    for s, s_ in t.sites.items()
                },
            }
        return {"tables": out, "plan": self._fractionation()}

    def _fractionation(self) -> dict:
        n = self.pln.num_of_fractions
        rx = getattr(self.pln, "prescribed_dose", None)
        return {
            "num_fractions": n,
            "prescribed_total_gy": rx,
            "dose_per_fraction_gy": round(rx / n, 3) if rx else None,
            "dose_convention": self.pln.dose_convention,
        }

    def get_site_constraints(self, table: str, site: str, apply: bool = False) -> dict:
        """Look up a site's constraints, converted (EQD2/LQ) to the plan's fractionation.

        apply=True replaces the protocol unless it is locked (given by a human; then a
        proposal only). A single target maps automatically; map other names first with
        map_structures, goals on unmapped structures are dropped.
        """
        try:
            t = load_constraint_table(table)
            goals, notes = site_goals(
                t,
                site,
                self.pln.num_of_fractions,
                getattr(self.pln, "prescribed_dose", None),
                per_fraction=self.pln.dose_convention == "per_fraction",
            )
        except ValueError as e:
            raise PlanningError(str(e)) from e

        auto = {}
        clinical = [v.name for v in self._clinical_vois()]
        targets = [v.name for v in self._clinical_vois() if v.voi_type == "TARGET"]
        for name in {g.voi for g in goals}:
            if name in self.aliases or name in clinical:
                continue
            if name in t.sites[site].targets and len(targets) == 1:
                auto[name] = targets[0]
        unmatched = sorted({g.voi for g in goals} - set(clinical) - set(self.aliases) - set(auto))
        result = {
            "table": f"{t.name} ({t.version})",
            "plan_fractionation": self._fractionation(),
            "goals": [g.tagged_label for g in goals],
            "auto_mapped": auto,
            "unmatched_structures": unmatched,
            "notes": notes,
        }
        if apply and self.protocol.locked:
            result["applied"] = False
            result["why"] = "Protocol is locked (given by a human); goals are a proposal only."
        elif apply:
            self.aliases.update(auto)
            kept = [g for g in goals if g.voi not in unmatched]
            self.set_protocol(
                Protocol(name=f"{t.id}/{site}", goals=kept, locked=False, notes=notes)
            )
            result["applied"] = True
            result["dropped_goals_for_unmatched_structures"] = len(goals) - len(kept)
        self._log("get_site_constraints", table=table, site=site, apply=apply)
        return result
