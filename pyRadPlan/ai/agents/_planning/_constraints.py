"""Clinical goals, constraint tables and EQD2 conversion.

Constraint tables (JSON, see :class:`ConstraintTable`) are looked up and converted to
the plan's fractionation by deterministic code, so the numbers an agent works with come
from a table entry, never from LLM memory. The bundled ``demo`` table holds dummy values
only; give the agent your own validated table as a file.
"""

import json
import math
import re
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Goals
# --------------------------------------------------------------------------------------

D_RE = re.compile(r"^D(\d+(?:\.\d+)?)$")
V_RE = re.compile(r"^V(\d+(?:\.\d+)?)Gy$")


class Goal(BaseModel):
    """One clinical goal, e.g. ``PTV D95 >= 1.9`` or ``SpinalCord max <= 1.5``.

    Dose values use the plan's ``dose_convention`` (per fraction by default),
    volumes are in percent.
    """

    voi: str
    metric: str = Field(
        description="'mean', 'max', 'min', 'D<vol%>' (e.g. 'D95') or 'V<dose>Gy' (e.g. 'V20Gy')."
    )
    op: Literal["<=", ">="]
    value: float
    hard: bool = Field(True, description="Hard constraint (must pass) or soft goal.")
    source: Optional[str] = Field(None, description="Where the goal comes from.")

    def model_post_init(self, _ctx: Any) -> None:
        """Validate the metric string."""
        if self.metric not in ("mean", "max", "min") and not (
            D_RE.match(self.metric) or V_RE.match(self.metric)
        ):
            raise ValueError(f"Unsupported metric '{self.metric}'.")

    @property
    def label(self) -> str:
        """Short human-readable form."""
        return f"{self.voi} {self.metric} {self.op} {self.value:g}"

    @property
    def tagged_label(self) -> str:
        """Label with '(hard)' or '(soft)'."""
        return f"{self.label} ({'hard' if self.hard else 'soft'})"


class Protocol(BaseModel):
    """A set of clinical goals the plan is evaluated against.

    ``locked`` protocols (given by a human) cannot be replaced by the agent; it can
    only propose additional goals.
    """

    name: str = "custom"
    goals: list[Goal] = Field(default_factory=list)
    locked: bool = True
    notes: list[str] = Field(default_factory=list)

    def ref_vols(self) -> list[float]:
        """DX reference volumes needed to evaluate the goals."""
        return sorted({float(m.group(1)) for g in self.goals if (m := D_RE.match(g.metric))})

    def ref_doses(self) -> list[float]:
        """VX reference doses needed to evaluate the goals."""
        return sorted({float(m.group(1)) for g in self.goals if (m := V_RE.match(g.metric))})


DoseConvention = Literal["total", "per_fraction"]


class GoalFile(BaseModel):
    """A JSON file of goals (see :func:`save_goals` and :func:`load_goals`)."""

    format: Literal["pyRadPlan-goals"] = "pyRadPlan-goals"
    version: Literal[1] = 1
    name: str = ""
    dose: DoseConvention = Field(description="Whether the doses are per fraction or total.")
    fractions: Optional[int] = Field(None, ge=1)
    goals: list[Goal]


def save_goals(
    path: Union[str, Path],
    goals: list[Goal],
    dose: DoseConvention,
    fractions: Optional[int] = None,
    name: str = "",
) -> None:
    """Write *goals* to a JSON file; *dose* and *fractions* describe their values."""
    data = GoalFile(name=name, dose=dose, fractions=fractions, goals=goals)
    # One goal per line keeps the file readable and easy to edit by hand
    head = json.dumps(data.model_dump(exclude={"goals"}, exclude_none=True), indent=2)[:-2]
    rows = ",\n".join("    " + json.dumps(g.model_dump(exclude_defaults=True)) for g in data.goals)
    goals = f"[\n{rows}\n  ]" if rows else "[]"
    Path(path).write_text(f'{head},\n  "goals": {goals}\n}}\n', encoding="utf-8")


def load_goals(
    path: Union[str, Path], dose: DoseConvention, fractions: int
) -> tuple[list[Goal], list[str]]:
    """Read a goal file and convert its doses to the plan's *dose* convention.

    Total and per-fraction doses are converted with the plan's number of *fractions*.
    A different fractionation in the file is reported in the returned notes, not
    converted (that would need an EQD2 conversion with an alpha/beta per structure).
    """
    data = GoalFile.model_validate_json(Path(path).read_text(encoding="utf-8"))
    notes = []
    if data.fractions is not None and data.fractions != fractions:
        notes.append(
            f"The goals are for {data.fractions} fractions, the plan has {fractions}; "
            "the doses are not converted to the plan's fractionation."
        )
    if data.dose == dose:
        return data.goals, notes
    factor = 1.0 / fractions if dose == "per_fraction" else float(fractions)
    return [_scale_goal(g, factor) for g in data.goals], notes


def _scale_goal(goal: Goal, factor: float) -> Goal:
    """Multiply the doses of *goal* by *factor* (the dose in ``V<dose>Gy`` included)."""
    if m := V_RE.match(goal.metric):
        return goal.model_copy(update={"metric": f"V{_round(float(m.group(1)) * factor):g}Gy"})
    return goal.model_copy(update={"value": _round(goal.value * factor)})


def qi_key(metric: str) -> str:
    """Map a goal metric to the key QICollection uses (``D95``, ``V20Gy``, ``mean``)."""
    if m := D_RE.match(metric):
        return f"D{float(m.group(1)):g}"
    if m := V_RE.match(metric):
        return f"V{float(m.group(1)):g}Gy"
    return metric


# --------------------------------------------------------------------------------------
# LQ / EQD2
# --------------------------------------------------------------------------------------

LQ_WARNING_DOSE_PER_FRACTION = 6.0  # Gy; LQ/EQD2 becomes questionable above this


def eqd2(total_dose: float, num_fractions: int, alpha_beta: float) -> float:
    """Equivalent dose in 2 Gy fractions: D (d + a/b) / (2 + a/b)."""
    d = total_dose / num_fractions
    return total_dose * (d + alpha_beta) / (2.0 + alpha_beta)


def physical_from_eqd2(eqd2_dose: float, num_fractions: int, alpha_beta: float) -> float:
    """Total physical dose in *num_fractions* fractions with the given EQD2.

    Solves D^2/n + (a/b) D - EQD2 (2 + a/b) = 0 for D > 0.
    """
    n, ab = num_fractions, alpha_beta
    return n * (-ab + math.sqrt(ab * ab + 4.0 * eqd2_dose * (2.0 + ab) / n)) / 2.0


def convert_fractionation(
    total_dose: float, from_fractions: int, to_fractions: int, alpha_beta: float
) -> float:
    """Isoeffective total dose when changing the number of fractions."""
    if from_fractions == to_fractions:
        return total_dose
    return physical_from_eqd2(
        eqd2(total_dose, from_fractions, alpha_beta), to_fractions, alpha_beta
    )


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------


class TableConstraint(BaseModel):
    metric: str
    op: Literal["<=", ">="]
    value: float
    hard: bool = True
    unit: Optional[Literal["Gy", "%", "%Rx"]] = None
    endpoint: Optional[str] = None
    note: Optional[str] = None


class TableStructure(BaseModel):
    alpha_beta: Optional[float] = None
    constraints: list[TableConstraint]


class Prescription(BaseModel):
    total_dose_gy: float
    num_fractions: int


class TableSite(BaseModel):
    description: str = ""
    prescription: Optional[Prescription] = None
    targets: dict[str, TableStructure] = Field(default_factory=dict)
    structures: dict[str, TableStructure] = Field(default_factory=dict)


class DoseReference(BaseModel):
    type: Literal["EQD2", "fractionation"]
    note: str = ""


class ConstraintTable(BaseModel):
    """A table of organ and target constraints per treatment site.

    ``dose_reference.type`` is ``"EQD2"`` (dose values are EQD2, converted to the plan's
    fractionation with each structure's ``alpha_beta``) or ``"fractionation"`` (physical
    doses for the site's ``prescription``, converted via EQD2 if the plan differs).
    ``sites.<site>.structures`` and ``sites.<site>.targets`` map structure names to an
    ``alpha_beta`` and their constraints: ``metric`` (``mean``, ``max``, ``min``,
    ``D<vol%>``, ``V<dose>Gy``), ``op``, ``value`` (Gy for dose metrics, % for V metrics),
    ``hard`` and an optional ``unit`` (``"%Rx"``: percent of the plan's prescription, not
    converted).
    """

    id: str
    name: str
    version: str
    disclaimer: str = ""
    source: str = ""
    dose_reference: DoseReference
    sites: dict[str, TableSite]


def _table_dir():
    return resources.files("pyRadPlan") / "data" / "constraint_tables"


def available_constraint_tables() -> dict[str, str]:
    """Bundled tables as {id: name}."""
    out = {}
    for f in _table_dir().iterdir():
        if f.name.endswith(".json"):
            t = ConstraintTable.model_validate_json(f.read_text())
            out[t.id] = t.name
    return out


def load_constraint_table(table: Union[str, Path]) -> ConstraintTable:
    """Load a bundled table by id (e.g. ``"demo"``) or a JSON file path."""
    p = Path(table)
    if p.suffix == ".json" and p.exists():
        return ConstraintTable.model_validate_json(p.read_text())
    f = _table_dir() / f"{table}.json"
    if not f.is_file():
        raise ValueError(
            f"Unknown constraint table '{table}'. Available: {list(available_constraint_tables())}"
        )
    return ConstraintTable.model_validate_json(f.read_text())


def _round(x: float, digits: int = 6) -> float:
    return float(f"{x:.{digits}g}")


def site_goals(
    table: ConstraintTable,
    site: str,
    num_fractions: int,
    prescription_total_gy: Optional[float] = None,
    per_fraction: bool = True,
    default_alpha_beta: float = 3.0,
) -> tuple[list[Goal], list[str]]:
    """Convert a table site to goals for a plan with *num_fractions* fractions.

    Dose values are converted from the table's reference (EQD2 or the site's
    prescription fractionation) with the structure's alpha/beta, and expressed per
    fraction if *per_fraction* (pyRadPlan's default dose convention).

    Returns
    -------
    goals, notes
        The goals (with provenance in ``source``) and warnings/notes for the planner.
    """
    if site not in table.sites:
        raise ValueError(f"Site '{site}' not in table '{table.id}'. Sites: {list(table.sites)}")
    s = table.sites[site]
    notes: list[str] = []
    if table.disclaimer:
        notes.append(table.disclaimer)

    rx = prescription_total_gy
    if rx is None and s.prescription is not None:
        rx = s.prescription.total_dose_gy
        notes.append(f"No plan prescription given; using the table's {rx:g} Gy.")

    if table.dose_reference.type == "EQD2":
        ref_fx: Optional[int] = None  # values are EQD2
    else:
        if s.prescription is None:
            raise ValueError(f"Site '{site}' needs a prescription for fractionation reference.")
        ref_fx = s.prescription.num_fractions
        if ref_fx != num_fractions:
            notes.append(
                f"Table values are for {ref_fx} fractions; converted to {num_fractions} "
                "fractions via EQD2 (LQ model)."
            )
    if rx and rx / num_fractions > LQ_WARNING_DOSE_PER_FRACTION:
        notes.append(
            f"{rx / num_fractions:.1f} Gy per fraction: LQ/EQD2 conversion is unreliable at "
            "this fraction size; use dedicated SBRT constraint tables."
        )

    scale = 1.0 / num_fractions if per_fraction else 1.0

    def to_plan_dose(value_gy: float, ab: float) -> float:
        if ref_fx is None:
            total = physical_from_eqd2(value_gy, num_fractions, ab)
        else:
            total = convert_fractionation(value_gy, ref_fx, num_fractions, ab)
        return total * scale

    goals: list[Goal] = []
    groups = [(n, st, True) for n, st in s.targets.items()] + [
        (n, st, False) for n, st in s.structures.items()
    ]
    for name, st, is_target in groups:
        ab = st.alpha_beta or (10.0 if is_target else default_alpha_beta)
        for c in st.constraints:
            src = f"{table.id} v{table.version} / {site} / {name} {c.metric}"
            if c.endpoint:
                src += f" ({c.endpoint})"
            metric, value = c.metric, c.value
            if c.unit == "%Rx":
                if rx is None:
                    notes.append(f"Skipped {name} {c.metric}: relative goal without prescription.")
                    continue
                value = rx * c.value / 100.0 * scale  # relative to the plan Rx, no conversion
            elif m := V_RE.match(c.metric):
                thr = to_plan_dose(float(m.group(1)), ab)
                metric = f"V{_round(thr, 4):g}Gy"
            elif c.unit != "%":  # dose metric in Gy
                value = to_plan_dose(c.value, ab)
                unconverted = c.value * scale
                if c.metric == "mean" and not math.isclose(value, unconverted, rel_tol=1e-3):
                    notes.append(f"{name} mean: EQD2 conversion of a mean dose is approximate.")
            goals.append(
                Goal(
                    voi=name, metric=metric, op=c.op, value=_round(value), hard=c.hard, source=src
                )
            )
    return goals, notes
