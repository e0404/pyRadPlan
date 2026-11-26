"""The planning state: everything the agent works on, with one method per tool.

All tools are plain methods on :class:`PlanningState`, so they can be tested and used
without any LLM; :mod:`._agent` registers them with pydantic-ai.
"""

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Literal, Optional

import numpy as np
import SimpleITK as sitk

from pyRadPlan._settings import get_settings
from pyRadPlan.analysis import QICollection
from pyRadPlan.core import ComputeControl, StatusReport, observe_control, observe_reports
from pyRadPlan.cst import StructureSet
from pyRadPlan.cst._voi import create_voi
from pyRadPlan.ct import CT
from pyRadPlan.dij import Dij
from pyRadPlan.dose._calc_dose import calc_dose_influence
from pyRadPlan.optimization._fluence_optimization import fluence_optimization
from pyRadPlan.optimization.objectives import MeanDose
from pyRadPlan.optimization.solvers import SolverBase, get_available_solvers
from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.result import PlanResult
from pyRadPlan.stf._generate_stf import generate_stf

from .._cst_agents import _objectives_by_voi
from .._plan_agents import BeamSetup, cst_geometry_summary
from . import _geometry as geo
from . import _pareto as pareto
from ._constraints import (
    D_RE,
    V_RE,
    Goal,
    Protocol,
    available_constraint_tables,
    load_constraint_table,
    qi_key,
    site_goals,
)
from ._evaluation import PlanRecord, evaluate_protocol, voxel_metric
from ._recipes import HelperRecipe
from ._structure_names import canonical_structure_name, name_candidates

logger = logging.getLogger(__name__)

_NO_DOSE = "No dose available yet. Call optimize first."
# Evaluated like a VOI but never part of the structure set: the patient outside all targets
NORMAL_TISSUE = "NormalTissue"
# Dose fall-off (% of the target floor per mm) rated ok / steep by check_feasibility
_FALLOFF_OK = 5.0
_FALLOFF_STEEP = 10.0
# D<x> goals with x <= this are treated like max-dose goals by check_feasibility
_NEAR_MAX_PCT = 10.0
# A mean goal is rated tight when the part of the VOI outside the targets has to stay
# below this fraction of the goal value
_MEAN_TIGHT = 0.5
# optimize stops once the best objective improved too little over this many iterations
_STALL_WINDOW = 50
# A beamlet passes through an OAR when its max dose there reaches this share of its
# max dose in the targets (entrance or exit side; scatter beside the beamlet stays below)
_CROSSING_FRACTION = 0.5


class PlanningError(Exception):
    """Errors meant to be shown to the model so it can correct itself."""


class UnknownVoiError(PlanningError):
    def __init__(self, name: str, valid: list[str]):
        super().__init__(f"Unknown VOI '{name}'. Valid VOIs: {valid}.")


def sitk_array(img: Any) -> np.ndarray:
    """Return a dose image (sitk) or array as numpy array (z, y, x)."""
    if isinstance(img, sitk.Image):
        return sitk.GetArrayFromImage(img)
    return np.asarray(img)


def _setup_text(key: tuple) -> str:
    """Gantry angles of a beam setup key: '0/72/144'."""
    return "/".join(f"{g:g}" for g, _ in key)


def _solver_name(solver: Any) -> Optional[str]:
    """Registry name of a prop_opt solver entry (name, config dict or instance)."""
    if isinstance(solver, dict):
        return solver.get("name")
    if isinstance(solver, SolverBase):
        return solver.short_name
    return solver


@contextmanager
def _cpu_backend(enabled: bool = True) -> Iterator[None]:
    """Run the block with numpy on the CPU (the backend is a global setting).

    GPU reductions are not deterministic, so the same objectives could give different
    plans; the agent needs reproducible results to judge its changes.
    """
    xp = get_settings().xp
    before = xp.prefer_gpu, xp.preferred_cpu_array_backend
    if enabled:
        xp.prefer_gpu, xp.preferred_cpu_array_backend = False, "numpy"
    try:
        yield
    finally:
        xp.prefer_gpu, xp.preferred_cpu_array_backend = before


def _beam_subset(dij: Dij, beams: list[int]) -> Dij:
    """Return the dose influence of some beams of *dij*, in the given order."""
    beam_num = np.asarray(dij.beam_num).ravel()
    per_beam = [np.flatnonzero(beam_num.astype(int) == b) for b in beams]
    cols = np.concatenate(per_beam)
    update: dict[str, Any] = {
        "beam_num": np.concatenate(
            [np.full(c.size, k, dtype=beam_num.dtype) for k, c in enumerate(per_beam)]
        ),
        "ray_num": np.asarray(dij.ray_num).ravel()[cols],
        "bixel_num": np.asarray(dij.bixel_num).ravel()[cols],
        "num_of_beams": len(beams),
    }
    for q in dij.quantities:
        container = getattr(dij, q)
        sliced: dict[int, Any] = {}  # slots sharing a matrix keep sharing the slice
        new = np.empty_like(container, dtype=object)
        for i, m in enumerate(container.flat):
            if m is not None:
                if id(m) not in sliced:
                    sliced[id(m)] = m[:, cols]
                new.flat[i] = sliced[id(m)]
        update[q] = new
    return dij.model_copy(update=update)


@dataclass
class PlanningState:
    """Everything the agent works on. Lives in Python, never in the chat."""

    ct: CT
    cst: StructureSet
    pln: Plan
    protocol: Protocol
    max_dij_calcs: int = 3
    # Limits of set_beams and optimize, set by the user
    min_beams: int = 1
    max_beams: int = 30
    allow_couch: bool = True
    max_iter: int = 5000
    max_optimizations: int = 8
    # set_beams is refused once this many optimizations are used (None: never)
    lock_beams_after: Optional[int] = None
    # Comparing beam setups: at most max_beam_setups get optimizations, each up to
    # trial_optimizations_per_setup; one more makes that setup final (None: no limit)
    max_beam_setups: Optional[int] = None
    trial_optimizations_per_setup: Optional[int] = None
    # A setup becomes final only once this many setups had all their trial optimizations
    min_beam_setups: Optional[int] = None
    # A new setup must differ from every compared one in at least this many beams
    min_setup_difference: Optional[int] = None
    # Planner reviews (see review_reason): every this many optimizations, 0 never
    review_every: int = 0
    # NormalTissue (implicit goal, QIs) is the patient this far outside the targets
    normal_tissue_margin_mm: float = 5.0
    # Solver of every optimize, over the plan's prop_opt (None: the plan's or the default)
    solver: Optional[str] = "scipy"
    # Optimize with numpy on the CPU: the same inputs give the same plan
    cpu_only: bool = True

    stf: Any = None
    dij: Optional[Dij] = None
    dose: Any = None
    last_evaluation: Optional[dict] = None
    history: list[dict] = field(default_factory=list)
    num_dij_calcs: int = 0
    num_optimizations: int = 0
    aliases: dict[str, str] = field(default_factory=dict)  # protocol name -> VOI name
    excluded: list[str] = field(default_factory=list)
    snapshots: dict[str, dict] = field(default_factory=dict)
    created_helpers: list[str] = field(default_factory=list)
    plans: list[PlanRecord] = field(default_factory=list)  # every optimization and blend
    last_optimizer: Optional[dict] = None  # see optimizer_report
    last_pareto: Optional[dict] = None  # see pareto_front
    # set by a host (e.g. the GUI): a stop request ends the run at the next tool call
    control: Optional[ComputeControl] = None
    # called as listener(event, **data) on tool calls, dij/optimize phases and new plans
    listener: Optional[Callable[..., None]] = None
    num_compactions: int = 0  # see compact_after of create_planning_agent
    # Set by a host: pause before the next optimize / set_beams
    pause_requested: bool = False
    finish_requested: bool = False  # the planner ended the run: only the report is left
    feedback: Optional[str] = None  # planner feedback for the next optimize result
    _reviewed_at: int = 0  # num_optimizations at the last review
    _reviewed_budget: int = 0  # max_optimizations at the last review
    _reviewed_hard_pass: bool = False
    _dij_region: Optional[np.ndarray] = None  # CT-grid voxels covered by the dij
    _normal_tissue: Optional[tuple] = None  # (cache key, mask), see _normal_tissue_region
    # Candidate beams with one dose influence (set_beam_pool): gantry_angles,
    # couch_angles, stf, dij, region
    beam_pool: Optional[dict] = None
    _influence: Optional[tuple] = None  # (dij, VOI names, rows), see beam_influence

    # ---- helpers ---------------------------------------------------------------------
    @property
    def voi_names(self) -> list[str]:
        return [v.name for v in self.cst.vois]

    def _log(self, action: str, **info: Any) -> None:
        self.history.append({"step": len(self.history) + 1, "action": action, **info})

    def _notify(self, event: str, **data: Any) -> None:
        if self.listener is None:
            return
        try:
            self.listener(event, **data)
        except Exception:  # a broken listener must not end the run
            logger.exception("Planning listener failed on %s", event)

    def _ensure_dij(self) -> Dij:
        if (
            self.dij is None
            and self.beam_pool is not None
            and not self.current_beams()["gantry_angles"]
        ):
            raise PlanningError(
                "No beams set yet: pick candidate beams with set_beams (case_overview)."
            )
        if self.dij is None:
            if self.num_dij_calcs >= self.max_dij_calcs:
                raise PlanningError(
                    f"Dose influence budget of {self.max_dij_calcs} calculations used up."
                )
            self._notify("phase", phase="dij")
            self.stf = generate_stf(self.ct, self.cst, self.pln)
            self.dij = calc_dose_influence(self.ct, self.cst, self.stf, self.pln)
            self.num_dij_calcs += 1
            region = np.zeros_like(geo.as_bool(self.cst.vois[0].mask))
            for v in self.cst.vois:
                region |= geo.as_bool(v.mask)
            self._dij_region = region
        return self.dij

    # ---- planner reviews -------------------------------------------------------------
    def review_reason(self, tool: str) -> Optional[str]:
        """Why the planner should review before *tool* runs, None to go on.

        'pause' (requested), and before an optimize with ``review_every`` set: 'budget'
        (used up), 'hard_goals' (all passed the first time), 'review' (every N plans).
        """
        if self.pause_requested:
            return "pause"
        if tool != "optimize" or not self.review_every:
            return None
        n = self.num_optimizations
        if n >= self.max_optimizations and self._reviewed_budget != self.max_optimizations:
            return "budget"
        if not self._reviewed_hard_pass and self._any_plan_passes():
            return "hard_goals"
        if n - self._reviewed_at >= self.review_every:
            return "review"
        return None

    def mark_reviewed(self) -> None:
        """Record a review: the reasons that applied now are not raised again."""
        self.pause_requested = False
        self._reviewed_at = self.num_optimizations
        self._reviewed_budget = self.max_optimizations
        self._reviewed_hard_pass = self._any_plan_passes()

    def _any_plan_passes(self) -> bool:
        return any(p.evaluation["all_hard_passed"] for p in self.plans)

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

    # ---- structure tools ---------------------------------------------------------------
    def inspect_structures(self) -> dict:
        """List VOIs: type, volume, canonical name, % inside targets.

        Also candidate VOIs for unmapped protocol names.
        """
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
            if c := canonical_structure_name(v.name):
                row["canonical"] = c
            if ix.size and v.name not in targets:
                ov = {
                    t: round(100 * np.intersect1d(ix, idx[t]).size / ix.size, 1) for t in targets
                }
                if any(ov.values()):
                    row["pct_inside_target"] = {t: p for t, p in ov.items() if p}
            rows.append(row)

        unmapped = {
            g.voi: name_candidates(g.voi, list(idx))
            for g in self.protocol.goals
            if self.aliases.get(g.voi, g.voi) not in idx
        }
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

    # ---- plan versions -----------------------------------------------------------------
    def _current_plan_index(self) -> Optional[int]:
        """Index of the optimization the current dose comes from, None without a dose."""
        if self.dose is None:
            return None
        return next((p.index for p in reversed(self.plans) if p.dose is self.dose), None)

    def snapshot(self, label: str) -> dict:
        """Save the current objectives, helpers, dose and evaluation under *label*."""
        plan = self._current_plan_index()
        self.snapshots[label] = {
            "cst": self.cst,
            "helpers": list(self.created_helpers),
            "beams": self.current_beams(),
            "evaluation": self.last_evaluation,
            "qis": self._clinical_qi_summary(),
            "dose": self.dose,
            "optimizer": self.last_optimizer,
            "plan": plan,
        }
        self._log("snapshot", label=label, plan=plan)
        return {"saved": label, "plan": plan, "snapshots": list(self.snapshots)}

    def restore(self, label: str) -> dict:
        """Return to the objectives, helpers and dose of snapshot *label*.

        Beams must be unchanged. A snapshot saved with a dose needs no new optimize.
        """
        if label not in self.snapshots:
            raise PlanningError(f"No snapshot '{label}'. Available: {list(self.snapshots)}")
        snap = self.snapshots[label]
        if snap["beams"] != self.current_beams():
            raise PlanningError("Beams changed since this snapshot; set_beams first.")
        self.cst = snap["cst"]
        self.created_helpers = list(snap["helpers"])
        self.dose, self.last_evaluation = snap["dose"], snap["evaluation"]
        self.last_optimizer = snap["optimizer"]
        self._log("restore", label=label, plan=snap["plan"])
        out = {"restored": label, "plan": snap["plan"], "evaluation": snap["evaluation"]}
        if self.dose is None:
            out["note"] = "Saved without a dose: call optimize to compute one."
        else:
            of_plan = f" of plan {snap['plan']}" if snap["plan"] is not None else ""
            out["note"] = f"Dose{of_plan} restored; no optimize needed."
        return out

    def _clinical_qi_summary(self) -> Optional[dict]:
        if self.dose is None:
            return None
        qis = self._qis()
        return {
            v.name: {k: round(float(qis[v.name][k].value), 3) for k in ("mean", "max")}
            for v in self._clinical_vois()
            if v.name in qis
        }

    def compare_snapshots(self, a: str, b: str = "current") -> dict:
        """Compare two versions: goal values and mean/max dose per clinical VOI, b - a."""

        def get(label):
            if label == "current":
                return {"evaluation": self.last_evaluation, "qis": self._clinical_qi_summary()}
            if label not in self.snapshots:
                raise PlanningError(f"No snapshot '{label}'. Available: {list(self.snapshots)}")
            return self.snapshots[label]

        sa, sb = get(a), get(b)
        if not sa["evaluation"] or not sb["evaluation"]:
            raise PlanningError("Both versions need an optimized dose (call optimize first).")
        ga_by = {g["goal"]: g for g in sa["evaluation"]["goals"]}
        goals = []
        for gb in sb["evaluation"]["goals"]:
            ga = ga_by.get(gb["goal"])
            if ga is None:  # goal not present in version a (protocol changed)
                goals.append({"goal": gb["goal"], "passed": [None, gb["passed"]]})
                continue
            row = {"goal": gb["goal"], "passed": [ga["passed"], gb["passed"]]}
            if "value" in ga and "value" in gb:
                row["value"] = [ga["value"], gb["value"]]
                row["delta"] = round(gb["value"] - ga["value"], 3)
            goals.append(row)
        vois = {}
        for name in sorted(set(sa["qis"] or {}) & set(sb["qis"] or {})):
            vois[name] = {
                k: round(sb["qis"][name][k] - sa["qis"][name][k], 3) for k in ("mean", "max")
            }
        self._log("compare_snapshots", a=a, b=b)
        return {
            "versions": [a, b],
            "hard_passed": [
                sa["evaluation"]["all_hard_passed"],
                sb["evaluation"]["all_hard_passed"],
            ],
            "goals": goals,
            "dose_delta_per_voi": vois,
        }

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
        proposal only). Canonical name matches and a single target map automatically.
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
            matches = [v for v in clinical if canonical_structure_name(v) == name]
            if name in {tn for tn in t.sites[site].targets} and len(targets) == 1:
                matches = targets
            if len(matches) == 1:
                auto[name] = matches[0]
        unmatched = sorted({g.voi for g in goals} - set(clinical) - set(self.aliases) - set(auto))
        result = {
            "table": f"{t.name} ({t.version})",
            "plan_fractionation": self._fractionation(),
            "goals": [g.tagged_label for g in goals],
            "auto_mapped": auto,
            "unmatched_structures": {n: name_candidates(n, clinical) for n in unmatched},
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

    # ---- planning tools ----------------------------------------------------------------
    def case_overview(self) -> dict:
        """Summarize plan settings, protocol, aliases, helpers, beams, objectives, budget."""
        return {
            "radiation_mode": self.pln.radiation_mode,
            "fractionation": self._fractionation(),
            "protocol": [g.tagged_label for g in self.protocol.goals],
            "protocol_locked": self.protocol.locked,
            "structure_aliases": self.aliases,
            "normal_tissue": f"patient more than {self.normal_tissue_margin_mm:g} mm outside "
            "the targets",
            "helpers": self.created_helpers,
            "overlap_priorities": {
                "note": "Overlapping voxels count only for the objectives of the VOI with the "
                "lower number (ties: both); goals are evaluated on the full VOIs.",
                **{v.name: v.overlap_priority for v in self.cst.vois},
            },
            "current_beams": self.current_beams(),
            "current_objectives": self.current_objectives(),
            "budget": {
                "dij_calcs_left": self.max_dij_calcs - self.num_dij_calcs,
                "optimizations_left": self.max_optimizations - self.num_optimizations,
                "beams": f"{self.min_beams}-{self.max_beams}"
                + ("" if self.allow_couch else ", coplanar only (couch 0)"),
                "max_iter": self.max_iter,
                **self._beam_budget(),
            },
        }

    def _beam_budget(self) -> dict:
        out = {}
        if self.beam_pool is not None:
            pool = self.beam_pool
            out["beam_candidates"] = (
                f"{len(pool['gantry_angles'])} beams (gantry "
                + ", ".join(
                    f"{g:g}" + (f"/couch {c:g}" if c else "")
                    for g, c in zip(pool["gantry_angles"], pool["couch_angles"])
                )
                + "): set_beams picks any of them without a dose influence calculation"
            )
        if self.trial_optimizations_per_setup is not None:
            tried = self._optimizations_per_setup()
            final = self._final_setup()
            out["beam_setups"] = {
                "optimizations_per_setup": {_setup_text(k): n for k, n in tried.items()},
                "rule": self._setup_rule(),
                "final": _setup_text(final) if final else None,
            }
        if self.lock_beams_after is not None:
            left = self.lock_beams_after - self.num_optimizations
            out["beams_final_after_optimization"] = self.lock_beams_after
            out["beam_changes"] = (
                f"possible for {left} more optimization(s)" if left > 0 else "locked"
            )
        return out

    def _setup_rule(self) -> str:
        n = self.trial_optimizations_per_setup
        if self.min_beam_setups:
            rule = (
                f"compare {self.min_beam_setups} setups with {n} optimizations each "
                f"before one more makes one of them final"
            )
        else:
            rule = (
                f"at most {self.max_beam_setups or 'any number of'} setups with up to "
                f"{n} optimizations each; one more makes that setup final"
            )
        if self.min_setup_difference:
            rule += (
                f"; each setup differs from the others in at least {self.min_setup_difference} "
                "beams"
            )
        return rule

    def _check_setup_trials(self) -> None:
        """Refuse the optimization that would make the current setup final too early."""
        n = self.trial_optimizations_per_setup
        if n is None or not self.min_beam_setups or self._final_setup() is not None:
            return
        tried = self._optimizations_per_setup()
        if tried.get(self._setup_key(self.current_beams()), 0) < n:
            return
        done = [k for k, c in tried.items() if c >= n]
        if len(done) < self.min_beam_setups:
            raise PlanningError(
                f"This would be optimization {n + 1} with these beams and make them final, "
                f"but {self.min_beam_setups} setups need {n} optimizations each first "
                "(optimizations so far: "
                + "; ".join(f"{_setup_text(k)}: {c}" for k, c in tried.items())
                + "). set_beams to another setup."
            )

    def structure_geometry(self) -> dict:
        """Per-VOI center of mass, principal axes and shape in LPS mm."""
        return cst_geometry_summary(self.cst)

    def current_beams(self) -> dict:
        prop = self.pln.prop_stf or {}
        return {
            "gantry_angles": list(prop.get("gantry_angles", [])),
            "couch_angles": list(prop.get("couch_angles", [])),
        }

    def current_objectives(self) -> dict[str, list[dict]]:
        return _objectives_by_voi(self.cst)

    def set_beams(self, gantry_angles: list[float], couch_angles: Optional[list[float]]) -> dict:
        """Set gantry / couch angles in degrees.

        Angles among the candidate beams (case_overview) cost no dose influence calculation;
        others make the next beam_influence / optimize compute one. Plans of other beams
        cannot be blended with, compared on a Pareto front or started from.
        """
        if self.lock_beams_after is not None and self.num_optimizations >= self.lock_beams_after:
            raise PlanningError(
                f"The beams are final after {self.lock_beams_after} optimizations: "
                f"{self.current_beams()}. Plan with them."
            )
        if self._final_setup() is not None:
            raise PlanningError(
                f"The beams are final ({self.trial_optimizations_per_setup + 1} optimizations "
                f"with them): {self.current_beams()}. Plan with them."
            )
        if not self.min_beams <= len(gantry_angles) <= self.max_beams:
            raise PlanningError(
                f"{len(gantry_angles)} beams: use {self.min_beams} to {self.max_beams}."
            )
        if not self.allow_couch and any(couch_angles or []):
            raise PlanningError("Only coplanar beams are allowed: leave the couch at 0.")
        setup = BeamSetup(gantry_angles=gantry_angles, couch_angles=couch_angles or [])
        tried = self._optimizations_per_setup()
        key = self._setup_key(setup.model_dump())
        if (
            self.max_beam_setups is not None
            and key not in tried
            and len(tried) >= self.max_beam_setups
        ):
            raise PlanningError(
                f"{len(tried)} beam setups have optimizations already (at most "
                f"{self.max_beam_setups}): pick one of them, gantry "
                + "; ".join(map(_setup_text, tried))
            )
        if self.min_setup_difference and key not in tried:
            new = set(key)
            close = [k for k in tried if len(new - set(k)) < self.min_setup_difference]
            if close:
                raise PlanningError(
                    "Too close to a compared setup (gantry "
                    + "; ".join(map(_setup_text, close))
                    + f"): a new setup must differ in at least {self.min_setup_difference} beams."
                )
        picked = self._pool_beams(setup.gantry_angles, setup.couch_angles)
        if picked is None and self.num_dij_calcs >= self.max_dij_calcs:
            if self.beam_pool is not None:
                raise PlanningError(
                    "No dose influence calculation left: pick angles among the candidate "
                    "beams (case_overview, beam_influence)."
                )
            raise PlanningError(
                "Dose influence budget used up: new beams could not be calculated, so the "
                "beams cannot change any more."
            )
        self._write_beams(setup)
        if picked is not None:
            pool = self.beam_pool
            self.stf = pool["stf"].model_copy(
                update={"beams": [pool["stf"].beams[i] for i in picked]}
            )
            self.dij = _beam_subset(pool["dij"], picked)
            self._dij_region = pool["region"]
        self._log("set_beams", **self.current_beams(), from_candidates=picked is not None)
        return {
            "beams": self.current_beams(),
            "dij_calcs_left": self.max_dij_calcs - self.num_dij_calcs,
            **({"from_candidates": "no dose influence calculation needed"} if picked else {}),
        }

    def set_beam_pool(
        self, gantry_angles: list[float], couch_angles: Optional[list[float]] = None
    ) -> dict:
        """Compute one dose influence for candidate beams (a host step, not a tool).

        set_beams then picks any subset of them without a calculation, and beam_influence
        reports every candidate. Uses one dose influence calculation; afterwards no beams
        are set until set_beams picks some.
        """
        setup = BeamSetup(gantry_angles=gantry_angles, couch_angles=couch_angles or [])
        self._write_beams(setup)
        self.beam_pool = None
        dij = self._ensure_dij()
        self.beam_pool = {
            "gantry_angles": list(setup.gantry_angles),
            "couch_angles": list(setup.couch_angles),
            "stf": self.stf,
            "dij": dij,
            "region": self._dij_region,
        }
        self.pln.prop_stf["gantry_angles"] = []
        self.pln.prop_stf["couch_angles"] = []
        self.stf = self.dij = None
        self._log("set_beam_pool", num_beams=len(setup.gantry_angles))
        return {"num_candidates": len(setup.gantry_angles), "bixels": dij.total_num_of_bixels}

    def _write_beams(self, setup: BeamSetup) -> None:
        """Put *setup* into the plan; the dose influence and the dose become stale."""
        if self.pln.prop_stf is None:
            self.pln.prop_stf = {}
        self.pln.prop_stf["gantry_angles"] = setup.gantry_angles
        self.pln.prop_stf["couch_angles"] = setup.couch_angles
        self.pln = validate_pln(self.pln)
        self.stf = self.dij = self.dose = self.last_evaluation = None

    @staticmethod
    def _setup_key(beams: dict) -> tuple:
        return tuple(
            (round(float(g), 2), round(float(c), 2))
            for g, c in zip(beams["gantry_angles"], beams["couch_angles"])
        )

    def _optimizations_per_setup(self) -> dict[tuple, int]:
        counts: dict[tuple, int] = {}
        for p in self.plans:
            if not p.blend:
                key = self._setup_key(p.beams)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _final_setup(self) -> Optional[tuple]:
        """Return the setup made final by an optimization beyond its trials, None if none."""
        if self.trial_optimizations_per_setup is None:
            return None
        for key, n in self._optimizations_per_setup().items():
            if n > self.trial_optimizations_per_setup:
                return key
        return None

    def _pool_beams(self, gantry: list[float], couch: list[float]) -> Optional[list[int]]:
        """Pool index of every requested beam, None unless all are candidates."""
        if self.beam_pool is None:
            return None
        pool = list(zip(self.beam_pool["gantry_angles"], self.beam_pool["couch_angles"]))

        def same(a: float, b: float) -> bool:
            return abs((a - b + 180.0) % 360.0 - 180.0) < 0.01

        picked = []
        for g, c in zip(gantry, couch):
            i = next((k for k, (pg, pc) in enumerate(pool) if same(g, pg) and same(c, pc)), None)
            if i is None:
                return None
            if i in picked:
                raise PlanningError(
                    f"Gantry {g:g} / couch {c:g} is given twice: use each beam once."
                )
            picked.append(i)
        return picked

    def _dose_grid_indices(self, dij: Dij, apply_overlap: bool = False) -> dict[str, np.ndarray]:
        cst = self.cst.apply_overlap_priorities() if apply_overlap else self.cst
        with warnings.catch_warnings():  # empty structures are reported by the callers
            warnings.filterwarnings("ignore", "Resampling created an empty structure")
            cst_dg = cst.resample_on_new_ct(self.ct.resample_to_grid(dij.dose_grid))
        return {v.name: np.asarray(v.indices_numpy) for v in cst_dg.vois}

    def beam_influence(self) -> dict:
        """Per beam, the unit-fluence mean dose of each VOI relative to the target.

        From the actual beam path incl. scatter; < 1 means the beam spares that VOI.
        With candidate beams (case_overview), every candidate is reported, the current
        ones marked. Per OAR also the share of the beam's beamlets that pass through it
        (in front of or behind the target, as in the beam's eye view).
        """
        if self.beam_pool is not None:
            pool = self.beam_pool
            dij = pool["dij"]
            beams = {"gantry_angles": pool["gantry_angles"], "couch_angles": pool["couch_angles"]}
        else:
            dij = self._ensure_dij()
            beams = self.current_beams()
        names = tuple((v.name, v.voi_type) for v in self.cst.vois)
        if self._influence is None or self._influence[0] is not dij or self._influence[1] != names:
            self._influence = (dij, names, self._beam_rows(dij, beams))
        current = set(zip(*self.current_beams().values()))
        rows = [
            {**r, **({"current": (r["gantry"], r["couch"]) in current} if self.beam_pool else {})}
            for r in self._influence[2]
        ]
        self._log("beam_influence")
        return {
            "note": "rel_mean_dose_vs_target: unit-fluence mean dose of the VOI / of the "
            "targets, lower is better for OARs. crossing_pct: % of the beam's beamlets "
            f"that reach the target and pass through the OAR (OAR max >= "
            f"{_CROSSING_FRACTION:.0%} of the beamlet's target max); clean_pct: % passing "
            "through no OAR. The optimizer turns crossing beamlets down, so a beam with "
            "little clean share adds little target dose without OAR dose.",
            "beams": rows,
        }

    def _beam_rows(self, dij: Dij, beams: dict) -> list[dict]:
        mat = dij.physical_dose.flat[0]
        beam_num = np.asarray(dij.beam_num).astype(int).ravel()
        idx = self._dose_grid_indices(dij)
        targets = [v.name for v in self.cst.vois if v.voi_type == "TARGET"]
        target_idx = np.unique(np.concatenate([idx[t] for t in targets]))
        # Body outlines typed OAR hold the targets: every beamlet passes through them
        oars = [
            v.name
            for v in self._clinical_vois()
            if v.voi_type == "OAR"
            and idx[v.name].size
            and np.isin(target_idx, idx[v.name]).mean() < 0.5
        ]
        rows = []
        for b in range(int(dij.num_of_beams)):
            block = mat[:, np.flatnonzero(beam_num == b)].tocsr()
            dose_b = np.asarray(block.sum(axis=1)).ravel()
            target_mean = float(dose_b[target_idx].mean()) or np.finfo(float).tiny
            rel = {
                name: round(float(dose_b[ix].mean()) / target_mean, 3) if ix.size else None
                for name, ix in idx.items()
                if name not in targets
            }
            t_max = block[target_idx].max(axis=0).toarray().ravel()
            hits = t_max > 0
            crossing, any_crossing = {}, np.zeros(int(hits.sum()), dtype=bool)
            for name in oars:
                o_max = block[idx[name]].max(axis=0).toarray().ravel()[hits]
                c = o_max >= _CROSSING_FRACTION * t_max[hits]
                any_crossing |= c
                if c.any():
                    crossing[name] = round(100 * float(c.mean()), 1)
            rows.append(
                {
                    "beam": b,
                    "gantry": beams["gantry_angles"][b],
                    "couch": beams["couch_angles"][b],
                    "rel_mean_dose_vs_target": rel,
                    "crossing_pct": crossing,
                    "clean_pct": round(100 * float(1 - any_crossing.mean()), 1)
                    if hits.any()
                    else 0.0,
                }
            )
        return rows

    def set_objectives(self, voi_name: str, objectives: list[Any]) -> dict:
        """Replace all objectives of one VOI. Penalties are relative weights.

        Mean Dose pulls the mean towards d_ref from both sides; for an upper limit use
        Mean Overdosing (d_max) or Mean Dose with d_ref 0.
        """
        if voi_name not in self.voi_names:
            raise UnknownVoiError(voi_name, self.voi_names)
        vois = [
            v.model_copy(update={"objectives": list(objectives)}) if v.name == voi_name else v
            for v in self.cst.vois
        ]
        self.cst = self.cst.model_copy(update={"vois": vois})
        self.dose = self.last_evaluation = None
        self._log("set_objectives", voi=voi_name, n=len(objectives))
        out = {"voi": voi_name, "objectives": self.current_objectives()[voi_name]}
        if self._voi(voi_name).voi_type != "TARGET" and any(
            isinstance(o, MeanDose) and o.d_ref > 0 for o in objectives
        ):
            out["note"] = (
                "Mean Dose with d_ref > 0 also pushes a lower mean up to d_ref. For an upper "
                "limit use Mean Overdosing (d_max) or d_ref 0."
            )
        return out

    def optimize(
        self,
        max_iter: int = 500,
        min_improvement: float = 1e-2,
        start_from: Optional[int] = None,
    ) -> dict:
        """Optimize fluence and evaluate every goal (margin >= 0: passed).

        Includes a soft NormalTissue goal: max of the patient beyond a margin around the
        targets (case_overview) below the highest target dose level.
        max_iter: ~500 to compare settings, 1000-2000 for promising plans. Stops early once
        the objective improved by less than min_improvement (relative) over 50 iterations;
        0 disables.
        start_from: plan number (same beams) whose fluence the solver starts from instead of
        zero. Worth it after small changes, above all to DVH objectives (Min / Max DVH), whose
        result depends on the start: the plan changes only where the objectives ask for it
        and needs fewer iterations. Start from zero after large changes.
        An exact repeat of an earlier optimization (same objectives, start and settings, or
        continuing a converged plan with its own objectives) is refused without using one.
        """
        if not any(v.objectives for v in self.cst.vois):
            raise PlanningError("No objectives defined. Call set_objectives first.")
        max_iter = min(max_iter, self.max_iter)
        run = {
            "start_from": start_from,
            "max_iter": max_iter,
            "min_improvement": min_improvement,
            "setup": (self.solver, self.cpu_only, repr(self.pln.prop_opt)),
        }
        self._refuse_repeat(run)
        self._check_setup_trials()
        dij = self._ensure_dij()
        initial = None if start_from is None else self._plan_fluence(start_from, dij)
        # An objective over zero voxels is NaN, and the solver stops at iteration 0. The
        # optimizer sees the VOIs after overlap priorities, on the dose grid.
        apply_overlap = bool((self.pln.prop_opt or {}).get("apply_overlap", True))
        idx = self._dose_grid_indices(dij, apply_overlap=apply_overlap)
        empty = [v.name for v in self.cst.vois if v.objectives and not idx[v.name].size]
        if empty:
            res = "x".join(f"{r:g}" for r in dij.dose_grid.resolution.values())
            raise PlanningError(
                f"{empty} have no voxels left on the {res} mm dose grid once overlapping "
                "voxels go to structures with a higher overlap priority (lower number), so "
                "their objectives cannot be evaluated. Remove them (set_objectives with []) "
                "or put the objectives on a margin helper (PRV) around the structure."
            )

        prop_opt = dict(self.pln.prop_opt or {})
        solver = prop_opt.get("solver")
        if self.solver is not None and _solver_name(solver) != self.solver:
            solver = self.solver
        if solver is None:
            # Without a configured solver the optimizer falls back the same way
            available = list(get_available_solvers())
            solver = "ipopt" if "ipopt" in available else available[0]
        if isinstance(solver, (str, dict)):
            solver = {"name": solver} if isinstance(solver, str) else dict(solver)
            prop_opt["solver"] = {**solver, "max_iter": max_iter}
        opt_info: dict = {}
        self._notify("phase", phase="optimize")
        # A stop keeps the last iterate, so stopping on a stalled objective ends the run
        # like convergence would. This control replaces self.control while the solver
        # runs, so a stop request on that is passed on.
        stall = ComputeControl()
        best: list[float] = []

        def watch(report: Any) -> None:
            if self.control is not None and self.control.stop_requested:
                stall.request_stop()
            f = report.data.get("objective") if isinstance(report, StatusReport) else None
            if f is None or not np.isfinite(f):
                return
            best.append(min(f, best[-1]) if best else f)
            if min_improvement > 0 and len(best) > _STALL_WINDOW:
                if best[-_STALL_WINDOW - 1] - best[-1] < min_improvement * abs(best[-1]):
                    stall.request_stop()

        with observe_reports(watch), observe_control(stall), _cpu_backend(self.cpu_only):
            result = fluence_optimization(
                self.ct,
                self.cst,
                self.stf,
                dij,
                self.pln.model_copy(update={"prop_opt": prop_opt}),
                opt_info=opt_info,
                initial_fluence=initial,
            )
        aborted = self.control is not None and self.control.stop_requested
        converged = stall.stop_requested and not aborted
        result, quantity = self._use_result(result)
        self.num_optimizations += 1

        self.last_evaluation = self._evaluate_dose()
        self.last_optimizer = self._optimizer_summary(opt_info, max_iter, apply_overlap)
        self.last_optimizer["converged"] = converged
        self.last_optimizer["start_from"] = start_from
        self.last_evaluation["optimizer"] = {
            k: self.last_optimizer[k]
            for k in ("iterations", "max_iter", "converged", "improvement_last_20pct")
        }
        if start_from is not None:
            self.last_evaluation["optimizer"]["start_from"] = start_from
        self._add_plan(
            result, quantity, cst=self.cst, objectives=self.current_objectives(), run=run
        )
        self._log(
            "optimize",
            max_iter=max_iter,
            start_from=start_from,
            all_hard_passed=self.last_evaluation["all_hard_passed"],
            num_failed=self.last_evaluation["num_failed"],
        )
        return self.last_evaluation

    def _refuse_repeat(self, run: dict) -> None:
        """Refuse an optimization that can only return an earlier plan, before it costs one.

        On the CPU the solver is deterministic: the same objectives (on the same structures)
        from the same start with the same settings give the same plan, and continuing a plan
        that converged with its own objectives returns it again unless min_improvement is
        now stricter.
        """
        if not self.cpu_only:
            return
        objectives = self.current_objectives()
        masks = {v.name: v.mask for v in self.cst.vois}
        for p in self.plans:
            prev = p.run
            if prev is None or prev["setup"] != run["setup"]:
                continue
            if p.objectives != objectives or p.beams != self.current_beams():
                continue
            if any(masks.get(v.name) is not v.mask for v in p.cst.vois if v.objectives):
                continue
            iterations = (p.evaluation.get("optimizer") or {}).get("iterations")
            stopped_early = iterations is not None and iterations < prev["max_iter"]
            if (
                prev["start_from"] == run["start_from"]
                and prev["min_improvement"] == run["min_improvement"]
                and (
                    prev["max_iter"] == run["max_iter"]
                    or (stopped_early and run["max_iter"] >= iterations)
                )
            ):
                start = "zero" if run["start_from"] is None else f"plan {run['start_from']}"
                raise PlanningError(
                    f"Plan {p.index} was optimized with exactly these objectives and "
                    f"settings from {start}; the solver is deterministic, so this would "
                    "return the same plan. Not run, no optimization used: change objectives "
                    "or settings, start from another plan, or blend plans."
                )
            if (
                run["start_from"] == p.index
                and stopped_early
                and run["min_improvement"] >= prev["min_improvement"]
            ):
                raise PlanningError(
                    f"Plan {p.index} was optimized with exactly these objectives until it "
                    "converged; continuing it returns the same plan. Not run, no optimization "
                    "used: change objectives, or lower min_improvement to converge further."
                )

    def blend_plans(self, plans: list[int], weights: list[float]) -> dict:
        """Mix the fluences of earlier plans (same beams) and evaluate the mix like a plan.

        Costs no optimization: the dose is computed from the mixed fluence, so the blend is
        an exact, deliverable plan between its sources. Blend plans that pass the hard goals
        but differ in their trade-offs (e.g. one sparing the cord, one the parotids) to find
        plans in between; blending near-identical plans gains nothing. Physical (and
        constant-RBE) dose is linear in the fluence, so max, min and mean goals that all
        sources pass also pass in the blend; D<x> and V<x> goals can change either way.
        Weights (>= 0) are normalized to sum 1. The blend becomes the current
        plan: save it with plan_versions or use it as start_from.
        """
        if len(plans) < 2 or len(set(plans)) != len(plans):
            raise PlanningError("Give at least two different plan numbers.")
        if len(weights) != len(plans):
            raise PlanningError(f"{len(plans)} plans need {len(plans)} weights.")
        w = np.asarray(weights, dtype=float)
        if (w < 0).any() or not w.sum() > 0:
            raise PlanningError("Weights must be >= 0 and not all 0.")
        w = w / w.sum()
        if self.dij is None:
            raise PlanningError(
                "No dose influence loaded (new beams, or a helper outside its region): "
                "optimize first."
            )
        fluences = [self._plan_fluence(i, self.dij) for i in plans]
        fluence = sum(wi * f for wi, f in zip(w, fluences))
        result, quantity = self._use_result(
            self.dij.compute_result_ct_grid(fluence, num_of_fractions=self.pln.result_dose_factor)
        )
        weights_out = [round(float(wi), 4) for wi in w]

        self.last_evaluation = self._evaluate_dose()
        self.last_evaluation["blend"] = {"plans": list(plans), "weights": weights_out}
        self.last_optimizer = None
        main = self._plan(plans[int(np.argmax(w))])
        self._add_plan(
            result,
            quantity,
            cst=main.cst,
            objectives=main.objectives,
            blend=self.last_evaluation["blend"],
        )
        self._log("blend_plans", plans=list(plans), weights=weights_out)
        return self.last_evaluation

    def pareto_front(
        self,
        plans: list[int],
        axes: Optional[list[str]] = None,
        n_samples: int = 300,
        top_n: int = 6,
    ) -> dict:
        """Sample blends of plans (same beams) and return their Pareto front.

        Costs no optimization and changes no plan. Only blends passing all hard goals count.
        Trade-off axes are the target goals, soft goals and mean goals whose value differs
        between the plans by >= 1 % of its limit (or the goal labels in axes); other hard
        goals (OAR max and the like) only need to pass (constraint_goals). Axis values are
        raw goal values, not the clipped margins of the score, so sparing far below a limit
        counts; differences below 1 % of a limit do not count (epsilon dominance), so the
        front holds blends that differ noticeably. Returns the axes (best / worst on the
        front, value per plan), up to top_n
        spread-out front blends (weights in the order of plans: blend_plans(plans,
        weights) makes one a plan), the compromise blend (closest to the best on every
        axis at once) with its weakest axes, the plans a blend beats on every axis
        (dominated_plans: they extend nothing) and the goals no plan changes (flat_goals).
        Grow the front where the compromise is weakest: blend it, optimize start_from that
        blend with its weakest axes tightened, and pass the result here with the others.
        """
        if len(plans) < 2 or len(set(plans)) != len(plans):
            raise PlanningError("Give at least two different plan numbers.")
        sources = [self._plan(i) for i in plans]
        if any(p.beams != sources[0].beams for p in sources):
            raise PlanningError("All plans need the same beams.")
        protocol = self._full_protocol()
        masks = {v.name: geo.as_bool(v.mask) for v in self.cst.vois}
        masks[NORMAL_TISSUE] = self._normal_tissue_region()
        keys: dict[str, set[str]] = {}
        for g in protocol.goals:
            voi = self.aliases.get(g.voi, g.voi)
            if voi in masks:
                keys.setdefault(voi, set()).add(qi_key(g.metric))
        if not keys:
            raise PlanningError("No goal can be evaluated: map the protocol structures first.")
        doses = [sitk_array(p.dose) for p in sources]
        # Doses are linear in the fluence, so a blend's dose is the blend of the doses
        voxels = {voi: np.stack([d[masks[voi]] for d in doses]) for voi in keys}

        def evaluate(w: np.ndarray) -> dict:
            qis = {}
            for voi, metric_keys in keys.items():
                v = w @ voxels[voi]
                qis[voi] = {k: SimpleNamespace(value=voxel_metric(v, k)) for k in metric_keys}
            return evaluate_protocol(protocol, qis, self.aliases)

        n = len(plans)
        weights = pareto.sample_weights(n, n_samples)
        evals = [evaluate(w / w.sum()) for w in weights]

        labels = [r["goal"] for r in evals[0]["goals"]]
        goal_of = {g.label: g for g in protocol.goals}
        values = np.array([[r["value"] for r in e["goals"]] for e in evals], dtype=float)
        sign = np.array([1.0 if goal_of[lbl].op == "<=" else -1.0 for lbl in labels])
        cost = sign * values
        limit = np.array([max(abs(goal_of[lbl].value), 1e-6) for lbl in labels])

        def trade_off(label: str) -> bool:
            g = goal_of[label]
            voi = self.aliases.get(g.voi, g.voi)
            target = voi in self.voi_names and self._voi(voi).voi_type == "TARGET"
            return target or not g.hard or g.metric == "mean"

        constraints = [lbl for lbl in labels if not trade_off(lbl)]
        if axes is None:
            spread = np.ptp(values[:n], axis=0)
            ax = [
                g
                for g, lbl in enumerate(labels)
                if trade_off(lbl) and spread[g] >= pareto.FLAT * limit[g]
            ]
        else:
            unknown = [a for a in axes if a not in labels]
            if unknown:
                raise PlanningError(f"Unknown goals {unknown}. Goals: {labels}")
            ax = [labels.index(a) for a in dict.fromkeys(axes)]
        if not ax:
            raise PlanningError(
                "The plans differ on no goal by >= 1 % of its limit: optimize plans with "
                "different trade-offs first."
            )

        def entry(s: int) -> dict:
            e = evals[s]
            return {
                # As sampled: blend_plans normalizes them the same way
                "weights": [round(float(w), 2) for w in weights[s]],
                "score": e["score"],
                "all_hard_passed": e["all_hard_passed"],
                "num_failed": e["num_failed"],
                "values": {labels[g]: round(float(values[s, g]), 3) for g in ax},
            }

        out: dict = {
            "plans": list(plans),
            "num_blends": len(weights),
            "constraint_goals": [] if axes is not None else constraints,
            "flat_goals": []
            if axes is not None
            else [lbl for g, lbl in enumerate(labels) if g not in ax and lbl not in constraints],
        }
        feasible = np.flatnonzero([e["all_hard_passed"] for e in evals])
        out["num_passing_hard"] = len(feasible)
        if not feasible.size:
            best = sorted(range(len(evals)), key=lambda s: evals[s]["score"], reverse=True)
            out["front"] = [entry(s) for s in best[:top_n]]
            out["note"] = "No blend passes all hard goals; front holds the best blends by score."
            self._log("pareto_front", plans=list(plans), front=0)
            return out

        c = cost[feasible][:, ax]
        front = feasible[pareto.epsilon_front(c, pareto.EPS * limit[ax])]
        fc = cost[front][:, ax]
        best, worst = fc.min(axis=0), fc.max(axis=0)
        regret, picked = pareto.spread_picks(fc, top_n)
        compromise = picked[0]

        out["axes"] = [
            {
                "goal": labels[g],
                "best": round(float(sign[g] * best[j]), 3),
                "worst": round(float(sign[g] * worst[j]), 3),
                "plans": {p: round(float(values[i, g]), 3) for i, p in enumerate(plans)},
            }
            for j, g in enumerate(ax)
        ]
        out["front_size"] = len(front)
        out["front"] = [
            {
                **entry(int(front[k])),
                "best_on": [labels[g] for j, g in enumerate(ax) if regret[k, j] == 0],
            }
            for k in picked
        ]
        order = np.argsort(-regret[compromise])
        out["compromise"] = {
            **out["front"][0],
            "weakest_axes": [
                {
                    "goal": labels[ax[j]],
                    "pct_of_range_from_best": round(100 * regret[compromise, j], 1),
                }
                for j in order[:2]
                if regret[compromise, j] > 0
            ],
        }
        # Plain Pareto dominance, no epsilon: one blend at least as good on every axis and
        # better on one is enough, however small the difference
        beaten = pareto.dominated(cost[:n][:, ax], c)
        out["dominated_plans"] = [p for i, p in enumerate(plans) if beaten[i]]
        out["plans_failing_hard"] = [
            p for i, p in enumerate(plans) if not evals[i]["all_hard_passed"]
        ]
        self.last_pareto = out
        self._log("pareto_front", plans=list(plans), front=len(front))
        self._notify("pareto", result=out)
        return out

    def _plan(self, index: int) -> PlanRecord:
        plan = next((p for p in self.plans if p.index == index), None)
        if plan is None:
            raise PlanningError(f"No plan {index}. Plans: {[p.index for p in self.plans]}")
        return plan

    def _plan_fluence(self, index: int, dij: Dij) -> np.ndarray:
        """Return the fluence of plan *index*, checked against the current beams and dij."""
        plan = self._plan(index)
        if plan.beams != self.current_beams():
            raise PlanningError(f"Plan {index} has other beams than the current ones.")
        fluence = plan.fluence
        if fluence is None or fluence.size != dij.total_num_of_bixels:
            raise PlanningError(f"Plan {index} does not fit the current dose influence.")
        return fluence

    def _use_result(self, result: PlanResult) -> tuple[PlanResult, str]:
        """Make the dose of a CT-grid *result* the current dose.

        Returns the result without its per-beam and per-scenario quantities, which every
        kept plan would otherwise hold in memory, and the quantity the goals are judged on.
        """
        result = result.model_copy(update={"beam_quantities": {}, "scenario_quantities": {}})
        quantity = "rbe_x_dose" if "rbe_x_dose" in result else "physical_dose"
        self.dose = result[quantity].values
        return result, quantity

    def _evaluate_dose(self) -> dict:
        """Evaluate the goals on the current dose, with the NormalTissue QIs."""
        qis = self._qis()
        evaluation = self._evaluate(qis)
        evaluation["normal_tissue"] = {
            k: round(float(qis[NORMAL_TISSUE][k].value), 2)
            for k in ("mean", "D50", "D5", "D2", "max")
        }
        return evaluation

    def _add_plan(self, result: PlanResult, quantity: str, **data: Any) -> None:
        """Record *result*, the current dose, as a new plan (see :attr:`plans`)."""
        plan = PlanRecord(
            index=len(self.plans) + 1,
            beams=self.current_beams(),
            evaluation=self.last_evaluation,
            result=result,
            quantity=quantity,
            **data,
        )
        self.plans.append(plan)
        self._notify("plan", plan=plan)

    def optimizer_report(self) -> dict:
        """Report on the last optimization: convergence and objective contributions.

        Solver status, iterations, objective start / end, improvement over the last 20 %
        (large: not converged, raise max_iter), each objective's share of the total.
        """
        if self.last_optimizer is None:
            raise PlanningError(
                "No optimization behind the current dose (none yet, or a blend). "
                "Call optimize first."
            )
        self._log("optimizer_report")
        return self.last_optimizer

    def _optimizer_summary(self, opt_info: dict, max_iter: int, apply_overlap: bool) -> dict:
        hist = [float(f) for f in opt_info.get("obj_history", [])]
        info = opt_info.get("result_info", {})
        iterations = opt_info.get("num_iter", info.get("num_iter"))
        improvement = None
        if len(hist) > 1 and hist[-1]:
            improvement = round((hist[int(0.8 * (len(hist) - 1))] - hist[-1]) / abs(hist[-1]), 4)

        dose = sitk_array(self.dose)
        cst = self.cst.apply_overlap_priorities() if apply_overlap else self.cst
        rows = []
        for v in cst.vois:
            values = dose[geo.as_bool(v.mask)]
            for o in v.objectives:
                f = float(o.priority * o.compute_objective(values)) if values.size else 0.0
                rows.append({"voi": v.name, "objective": type(o).__name__, "value": f})
        total = sum(r["value"] for r in rows) or 1.0
        for r in rows:
            r["share_pct"] = round(100 * r["value"] / total, 1)
            r["value"] = round(r["value"], 5)
        return {
            "status": info.get("message", info.get("status")),
            "iterations": iterations,
            "max_iter": max_iter,
            "hit_iteration_limit": iterations is not None and iterations >= max_iter,
            "objective_start": round(hist[0], 5) if hist else None,
            "objective_final": round(hist[-1], 5) if hist else None,
            "improvement_last_20pct": improvement,
            "contributions": sorted(rows, key=lambda r: r["share_pct"], reverse=True),
        }

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
