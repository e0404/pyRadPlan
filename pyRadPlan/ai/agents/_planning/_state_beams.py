"""Beam tools of the planning state: beam setups, the beam pool and dose influence."""

import warnings
from typing import Any, Optional

import numpy as np

from pyRadPlan.dij import Dij
from pyRadPlan.dose._calc_dose import calc_dose_influence
from pyRadPlan.plan import validate_pln
from pyRadPlan.stf._generate_stf import generate_stf

from .._plan_agents import BeamSetup
from . import _geometry as geo
from ._state_base import PlanningError

# A beamlet passes through an OAR when its max dose there reaches this share of its
# max dose in the targets (entrance or exit side; scatter beside the beamlet stays below)
_CROSSING_FRACTION = 0.5


def _setup_text(key: tuple) -> str:
    """Gantry angles of a beam setup key: '0/72/144'."""
    return "/".join(f"{g:g}" for g, _ in key)


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


class BeamsMixin:
    """Beam setups and their budget, the beam pool and the dose influence."""

    def _ensure_dij(self) -> Dij:
        if self.dij is None and not self.current_beams()["gantry_angles"]:
            raise PlanningError(
                "No beams set yet: pick candidate beams with set_beams (case_overview)."
                if self.beam_pool is not None
                else "No beams set yet: call set_beams first; the next beam_influence or "
                "optimize uses a dose influence calculation for those beams."
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

    def current_beams(self) -> dict:
        prop = self.pln.prop_stf or {}
        return {
            "gantry_angles": list(prop.get("gantry_angles", [])),
            "couch_angles": list(prop.get("couch_angles", [])),
        }

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
        setup = self._beam_setup(gantry_angles, couch_angles)
        tried = self._optimizations_per_setup()
        key = self._setup_key(setup.model_dump())
        if self.dij is not None and key == self._setup_key(self.current_beams()):
            return {
                "beams": self.current_beams(),
                "dij_calcs_left": self.max_dij_calcs - self.num_dij_calcs,
                "note": "These are the current beams: nothing changed, dose influence kept.",
            }
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
        setup = self._beam_setup(gantry_angles, couch_angles)
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

    @staticmethod
    def _beam_setup(gantry_angles: list[float], couch_angles: Optional[list[float]]) -> BeamSetup:
        """Validate the angles; a beam (gantry and couch angle) may be given only once."""
        setup = BeamSetup(gantry_angles=gantry_angles, couch_angles=couch_angles or [])
        seen = set()
        for g, c in zip(setup.gantry_angles, setup.couch_angles):
            beam = (round(float(g) % 360.0, 2), round(float(c) % 360.0, 2))
            if beam in seen:
                raise PlanningError(
                    f"Gantry {g:g} / couch {c:g} is given twice: use each beam once."
                )
            seen.add(beam)
        return setup

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
        Without candidate beams, it reports the beams set with set_beams and computes their
        dose influence if needed (one of dij_calcs_left). With candidate beams
        (case_overview), every candidate is reported, the current ones marked. Per OAR also
        the share of the beam's beamlets that pass through it (in front of or behind the
        target, as in the beam's eye view).
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
