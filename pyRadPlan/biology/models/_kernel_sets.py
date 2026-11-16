"""
Kernel sets: families of per-energy biological kernel data in a machine.

A machine may tabulate several families of biological kernel data, e.g. α/β kernels, z*
kernels or a LET. Each family is a *kernel set* with a kind, an optional variant, the
provenance (radiobiological model) that produced it and, for tissue-dependent kinds, one
column per tissue class. The sets are declared in ``machine.meta.kernelSets`` and the
tissue classes in ``machine.meta.tissueParameters``; machines without these declarations
get one inferred set per kind present in their kernels (legacy data). See
``docs/development/bio_model_formalisms.md`` for the metadata specification.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from pyRadPlan.biology.tissues import (
    MKMTissueProperties,
    TissueProperties,
    UNDECLARED_PROVENANCE,
    get_provenance,
    provenance_label,
    tissue_provenance,
)

logger = logging.getLogger(__name__)

#: Default kernel fields of each kernel set kind.
KERNEL_SET_KINDS: dict[str, tuple[str, ...]] = {
    "alpha_beta": ("alpha", "beta"),
    "zs": ("zs",),
    "let": ("let",),
}

#: Kinds whose kernels hold one column per tissue class (``rbe_table``: RBE tables, which
#: a tabulated formalism reads instead of machine kernels).
TISSUE_DEPENDENT_KINDS: frozenset[str] = frozenset({"alpha_beta", "zs", "rbe_table"})

#: Display name of each kernel set kind.
KERNEL_SET_KIND_LABELS: dict[str, str] = {
    "alpha_beta": "α/β kernels",
    "zs": "z* kernels",
    "let": "LET",
    "rbe_table": "α/β tables",
}

#: Machine quantities a kernel set kind provides (see ``Machine.provided_quantities``).
KERNEL_SET_QUANTITIES: dict[str, tuple[str, ...]] = {
    "alpha_beta": ("alpha", "beta"),
    "zs": ("zs",),
    "let": ("let",),
}

# Kernel fields stored under another name in the kernel model than in the .mat file.
_MAT_KERNEL_FIELDS: dict[str, str] = {"LET": "let", "Z": "idd"}

_MISSING = object()


@dataclass(frozen=True, eq=False)
class TissueClassInfo:
    """
    One tissue class of a machine: its parameters and where its base data lives.

    Attributes
    ----------
    tissue : TissueProperties
        The parameters of the class (one entry of ``machine.tissue_properties``).
    kernel_set : str or None
        Id of the kernel set holding its base data; None if it belongs to no set.
    column : int or None
        Zero-based column of that kernel set; None if unknown or not tissue dependent.
    """

    tissue: TissueProperties
    kernel_set: Optional[str] = None
    column: Optional[int] = None

    @property
    def name(self) -> str:
        """Display name: the cell line, else the provenance and column."""
        if self.tissue.cell_line:
            return str(self.tissue.cell_line)
        suffix = f" (column {self.column + 1})" if self.column is not None else ""
        return f"{provenance_label(self.provenance)}{suffix}"

    @property
    def provenance(self) -> str:
        """Canonical provenance of the class' parameters."""
        return tissue_provenance(self.tissue)


@dataclass(frozen=True, eq=False)
class KernelSetInfo:
    """
    Description of one kernel set of a machine, without the kernel arrays.

    Attributes
    ----------
    id : str
        Unique name within the machine, referenced by tissue classes (``ref_Kernel``).
    kind : str
        ``"alpha_beta"``, ``"zs"``, ``"let"`` or a future kind.
    fields : tuple[str, ...]
        Kernel fields holding the data, e.g. ``("alpha", "beta")``.
    variant : str or None
        Kind-specific definition, e.g. ``"LETd"`` or ``"z1D"``.
    provenance : str
        Canonical provenance, ``"undeclared"`` if the metadata names none.
    reference : str
        Free-text reference (publication, code, version) of the data.
    default : bool
        Whether the set is used when several sets of its kind exist and none is chosen.
    declared : bool
        Whether the set is declared in ``kernelSets`` (False: inferred from legacy data).
    tissue_classes : tuple[TissueClassInfo, ...]
        The tissue classes of the set, in declaration order.
    problems : tuple[str, ...]
        Inconsistencies of the base data that prevent using the set.
    complete : bool
        Whether every kernel carries every field of the set.
    """

    id: str
    kind: str
    fields: tuple[str, ...]
    variant: Optional[str] = None
    provenance: str = UNDECLARED_PROVENANCE
    reference: str = ""
    default: bool = False
    declared: bool = False
    tissue_classes: tuple[TissueClassInfo, ...] = ()
    problems: tuple[str, ...] = ()
    complete: bool = True

    @property
    def tissue_dependent(self) -> bool:
        """Whether the kernels hold one column per tissue class."""
        return self.kind in TISSUE_DEPENDENT_KINDS

    @property
    def provenance_label(self) -> str:
        """Display name of the provenance, e.g. ``"LEM I"`` or ``"model not declared"``."""
        return provenance_label(self.provenance)

    @property
    def kind_label(self) -> str:
        """Display name of the kind, e.g. ``"α/β kernels"``."""
        return KERNEL_SET_KIND_LABELS.get(self.kind, self.kind)

    @property
    def label(self) -> str:
        """Display label, e.g. ``"α/β kernels · LEM I"`` or ``"LET · LETd"``."""
        if self.tissue_dependent:
            return f"{self.kind_label} · {self.provenance_label}"
        return f"{self.kind_label} · {self.variant}" if self.variant else self.kind_label

    @property
    def tissues(self) -> tuple[TissueProperties, ...]:
        """Parameters of the set's tissue classes."""
        return tuple(info.tissue for info in self.tissue_classes)


# ----------------------------------------------------------------- .mat parsing
def mat_kernel_field(name: str) -> str:
    """Kernel model attribute of a ``.mat`` kernel field name (``LET`` -> ``let``)."""
    return _MAT_KERNEL_FIELDS.get(name, name)


def struct_rows(struct: Any, count_keys: Sequence[str] = ()) -> list[dict[str, Any]]:
    """
    Split a matRad struct (array) as ``read_mat`` delivers it into one dict per element.

    ``read_mat`` returns a struct array as one list per field and squeezes a single element
    to scalars and strings; a cell array of structs arrives as a list of dicts. A field
    holding a cell array per element (e.g. ``fields``) is a list as well, so the element
    count is taken from ``count_keys`` (fields holding one string or number per element)
    when given, otherwise from the first field.

    Raises
    ------
    ValueError
        The struct is not a mapping or a list of mappings, or its fields describe differing
        element counts.
    """
    if isinstance(struct, Sequence) and not isinstance(struct, (str, bytes)):
        rows = list(struct)
        if not all(isinstance(row, Mapping) for row in rows):
            raise ValueError(f"Expected a struct array, got {struct!r}.")
        return [dict(row) for row in rows]
    if not isinstance(struct, Mapping) or not struct:
        raise ValueError(f"Expected a non-empty struct, got {struct!r}.")

    def is_column(values: Any) -> bool:
        return isinstance(values, (list, tuple)) or (
            isinstance(values, np.ndarray) and values.ndim > 0
        )

    def column(values: Any) -> list:
        if isinstance(values, np.ndarray):
            return list(np.reshape(values, (-1,)).tolist())
        return list(values)

    key = next((k for k in count_keys if k in struct), next(iter(struct)))
    if not is_column(struct[key]):
        return [dict(struct)]
    n_rows = len(column(struct[key]))
    rows: list[dict[str, Any]] = [{} for _ in range(n_rows)]
    for name, values in struct.items():
        entries = column(values) if is_column(values) else [values] * n_rows
        if len(entries) != n_rows:
            raise ValueError(
                f"Struct field '{name}' describes {len(entries)} elements, '{key}' {n_rows}."
            )
        for row, value in zip(rows, entries):
            row[name] = value
    return rows


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    try:
        return bool(np.asarray(value).reshape(-1)[0]) if np.size(value) else False
    except (TypeError, ValueError):
        return False


def _field_names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        names = re.split(r"[\s,;]+", value.strip())
    elif isinstance(value, np.ndarray):
        names = [str(v) for v in value.reshape(-1).tolist()]
    elif isinstance(value, (list, tuple)):
        names = [str(v) for v in value]
    else:
        names = [str(value)]
    return tuple(mat_kernel_field(name.strip()) for name in names if name and name.strip())


def parse_kernel_set_declarations(raw: Any) -> list[dict[str, Any]]:
    """
    Normalize a ``machine.meta.kernelSets`` entry into one dict per kernel set.

    Each dict holds ``id``, ``kind``, ``fields`` (kernel attribute names), ``variant``,
    ``model`` (canonical provenance), ``reference`` and ``default``. ``fields`` may be given
    as a cell array of names or one comma-separated string; it defaults to the kind's
    standard fields. A missing ``id`` defaults to the kind, suffixed by the position if
    the kind occurs several times.

    Raises
    ------
    ValueError
        The struct cannot be split into sets, a set names no kind, or ids repeat.
    """
    if raw is None:
        return []
    if isinstance(raw, Mapping) and not raw:
        return []
    rows = struct_rows(raw, count_keys=("id", "kind"))
    declarations = []
    for position, row in enumerate(rows):
        kind = _text(row.get("kind"))
        if kind is None:
            raise ValueError(f"Kernel set {position + 1} in 'kernelSets' declares no kind.")
        fields = _field_names(row.get("fields")) or KERNEL_SET_KINDS.get(kind, ())
        if not fields:
            raise ValueError(
                f"Kernel set {position + 1} of unknown kind '{kind}' declares no fields."
            )
        provenance = get_provenance(_text(row.get("model")), _text(row.get("version")))
        declarations.append(
            {
                "id": _text(row.get("id")),
                "kind": kind,
                "fields": fields,
                "variant": _text(row.get("variant")),
                "model": provenance.name,
                "reference": _text(row.get("reference")) or "",
                "default": _flag(row.get("default", False)),
            }
        )

    kinds = [d["kind"] for d in declarations]
    for position, declaration in enumerate(declarations):
        if declaration["id"] is None:
            kind = declaration["kind"]
            repeated = kinds.count(kind) > 1
            declaration["id"] = f"{kind}_{kinds[: position + 1].count(kind)}" if repeated else kind
    ids = [d["id"] for d in declarations]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Kernel set ids in 'kernelSets' must be unique, repeated: {duplicates}.")
    return declarations


def tissue_class_reference(row: Mapping[str, Any]) -> dict[str, Any]:
    """
    Kernel set and column a ``tissueParameters`` row assigns its class to.

    ``ref_Kernel`` names the kernel set id; ``column`` is one-based in the ``.mat`` file
    (MATLAB convention) and returned zero-based.
    """
    ref = None
    for key in ("ref_Kernel", "refKernel", "ref_kernel", "kernelSet", "kernel_set"):
        if key in row:
            ref = _text(row[key])
            break
    column = row.get("column")
    try:
        value = float(np.asarray(column).reshape(-1)[0]) if column is not None else math.nan
    except (TypeError, ValueError, IndexError):
        value = math.nan
    return {
        "kernel_set": ref,
        "column": int(value) - 1 if math.isfinite(value) and value >= 1 else None,
    }


def extra_kernel_fields(raw_kernel_sets: Any, known_fields: Any) -> tuple[str, ...]:
    """Kernel fields named by declared kernel sets that the kernel model does not define."""
    try:
        declarations = parse_kernel_set_declarations(raw_kernel_sets)
    except ValueError:
        return ()
    names: list[str] = []
    for declaration in declarations:
        names.extend(f for f in declaration["fields"] if f not in known_fields and f not in names)
    return tuple(names)


# ----------------------------------------------------------------- kernel access
def kernel_field(kernel: Any, name: str, default: Any = None) -> Any:
    """
    Kernel array ``name`` of one kernel, from its own fields or its additional data.

    Works for kernel models and for kernel dicts as dose engines see them.
    """
    if isinstance(kernel, Mapping):
        return kernel.get(name, default)
    value = getattr(kernel, name, _MISSING)
    if value is not _MISSING:
        return value
    additional = getattr(kernel, "additional_data", None) or {}
    return additional.get(name, default)


def _field_state(kernel: Any, name: str) -> Optional[bool]:
    """Tell whether the kernel holds ``name`` (None: a stand-in that cannot tell)."""
    value = getattr(kernel, name, _MISSING)
    if value is _MISSING:
        additional = getattr(kernel, "additional_data", _MISSING)
        if additional is _MISSING:
            return None
        return additional.get(name) is not None
    return value is not None


def kernel_reference_classes(kernels: Mapping[Any, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the ``(alpha_x, beta_x)`` tissue classes of the α/β kernel columns.

    The per-bixel α/β kernels are ``(n_classes, n_depths)`` arrays whose rows are addressed
    by a single class index computed once for the whole dose grid. That index is only
    meaningful when every energy tabulates the same tissue classes in the same order, so a
    differing energy is rejected rather than silently selecting another tissue's α/β.

    Raises
    ------
    ValueError
        No kernels, an energy without its tissue class metadata, or classes differing
        between energies.
    """
    if not kernels:
        raise ValueError(
            "α/β kernels need pencil-beam kernels with per-tissue-class alpha / beta data, "
            "but the machine carries none."
        )
    reference = None
    for energy, kernel in kernels.items():
        alpha_x, beta_x = getattr(kernel, "alpha_x", None), getattr(kernel, "beta_x", None)
        if alpha_x is None or beta_x is None:
            raise ValueError(
                "α/β kernels need the reference photon alpha_x / beta_x of every kernel "
                f"tissue class, but the kernel of energy {energy} declares none."
            )
        alpha_x = np.reshape(np.asarray(alpha_x, dtype=float), (-1,))
        beta_x = np.reshape(np.asarray(beta_x, dtype=float), (-1,))
        if alpha_x.shape != beta_x.shape:
            raise ValueError(
                f"Kernel of energy {energy} declares {alpha_x.size} reference alpha_x but "
                f"{beta_x.size} reference beta_x values."
            )
        if reference is None:
            reference = (energy, alpha_x, beta_x)
            continue
        ref_energy, ref_alpha_x, ref_beta_x = reference
        if not (np.array_equal(alpha_x, ref_alpha_x) and np.array_equal(beta_x, ref_beta_x)):
            raise ValueError(
                "α/β kernels require identical tissue classes in the same order for every "
                f"machine energy. Energy {energy} declares alpha_x={alpha_x.tolist()}, "
                f"beta_x={beta_x.tolist()}, while energy {ref_energy} declares "
                f"alpha_x={ref_alpha_x.tolist()}, beta_x={ref_beta_x.tolist()}."
            )
    return reference[1], reference[2]


def _n_columns(kernels: Mapping[Any, Any], fields: Sequence[str]) -> Optional[int]:
    for kernel in kernels.values():
        for name in fields:
            values = kernel_field(kernel, name)
            if values is not None and np.ndim(values) == 2:
                return int(np.shape(values)[0])
        return None
    return None


# ----------------------------------------------------------------- inference
def _detect_kinds(kernels: Optional[Mapping[Any, Any]]) -> list[str]:
    """Kernel set kinds legacy kernels provide, in the order of :data:`KERNEL_SET_KINDS`."""
    if not kernels:
        return []
    values = list(kernels.values())
    kinds = []
    if any(
        getattr(k, "alpha", None) is not None or getattr(k, "alpha_x", None) is not None
        for k in values
    ):
        kinds.append("alpha_beta")
    if all(getattr(k, "zs", None) is not None for k in values):
        kinds.append("zs")
    if all(getattr(k, "let", None) is not None for k in values):
        kinds.append("let")
    return kinds


def _kind_of_tissue(tissue: TissueProperties) -> str:
    return "zs" if isinstance(tissue, MKMTissueProperties) else "alpha_beta"


def build_kernel_sets(
    *,
    declarations: Sequence[Mapping[str, Any]] = (),
    tissue_classes: Sequence[TissueProperties] = (),
    tissue_refs: Optional[Sequence[Mapping[str, Any]]] = None,
    kernels: Optional[Mapping[Any, Any]] = None,
    classes_declared: bool = True,
    available_kinds: Optional[Sequence[str]] = None,
) -> dict[str, KernelSetInfo]:
    """
    Combine kernel set declarations, tissue classes and kernels into :class:`KernelSetInfo`.

    Parameters
    ----------
    declarations : sequence of dict
        Output of :func:`parse_kernel_set_declarations`; empty for legacy data, which gets
        one set per kind present in the kernels.
    tissue_classes : sequence of TissueProperties
        The machine's tissue classes.
    tissue_refs : sequence of dict, optional
        Per tissue class ``{"kernel_set": id, "column": int}`` (see
        :func:`tissue_class_reference`); ignored unless aligned with ``tissue_classes``.
    kernels : mapping, optional
        Pencil-beam kernels per energy. ``None`` for objects not carrying kernel data, whose
        sets are then inferred from the tissue classes alone.
    classes_declared : bool
        False if the machine declares no tissue classes, so that they are derived from the
        α/β kernels' reference parameters (one class per column).
    available_kinds : sequence of str, optional
        Restricts the kinds inferred without kernel data (e.g. from provided quantities).
    """
    refs = list(tissue_refs) if tissue_refs is not None else []
    if len(refs) != len(tissue_classes):
        refs = [{} for _ in tissue_classes]

    specs = _set_specs(declarations, kernels, tissue_classes, refs, available_kinds)
    legacy = [s["id"] for s in specs if _reads_reference_columns(s) and kernels is not None]
    reference, reference_problem = None, None
    if legacy:
        try:
            reference = kernel_reference_classes(kernels)
        except ValueError as exc:
            reference_problem = str(exc)

    classes = list(tissue_classes)
    if not classes_declared:
        classes, refs = [], []
        if reference is not None:
            classes = [
                TissueProperties.from_base_data(alpha_x=float(a), beta_x=float(b))
                for a, b in zip(reference[0], reference[1])
            ]
            refs = [{"kernel_set": legacy[0], "column": i} for i in range(len(classes))]

    members = _assign_members(specs, classes, refs)
    sets: dict[str, KernelSetInfo] = {}
    for spec in specs:
        problems, complete = _field_problems(spec, kernels)
        tissues = members[spec["id"]]
        columns: list[Optional[int]] = [column for _, column in tissues]
        if spec["kind"] in TISSUE_DEPENDENT_KINDS:
            if spec["id"] in legacy and reference_problem is not None:
                problems.append(reference_problem)
            elif spec["id"] in legacy:
                columns, matching_problems = _match_reference_columns(
                    tissues, reference, explicit=any(c is not None for c in columns)
                )
                problems.extend(matching_problems)
            else:
                columns = [c if c is not None else i for i, c in enumerate(columns)]
            problems.extend(_column_problems(spec, kernels, columns))
        sets[spec["id"]] = _set_info(spec, tissues, columns, problems, complete)
    return sets


def _legacy_spec(kind: str) -> dict[str, Any]:
    return {
        "id": kind,
        "kind": kind,
        "fields": KERNEL_SET_KINDS[kind],
        "variant": None,
        "model": UNDECLARED_PROVENANCE,
        "reference": "",
        "default": False,
        "declared": False,
    }


def _set_specs(
    declarations: Sequence[Mapping[str, Any]],
    kernels: Optional[Mapping[Any, Any]],
    tissue_classes: Sequence[TissueProperties],
    refs: Sequence[Mapping[str, Any]],
    available_kinds: Optional[Sequence[str]],
) -> list[dict[str, Any]]:
    """Return the declared kernel sets, or one legacy set per kind the data provides."""
    if declarations:
        return [{**declaration, "declared": True} for declaration in declarations]
    if kernels is not None:
        kinds = _detect_kinds(kernels)
    else:
        kinds = []
        for tissue, ref in zip(tissue_classes, refs):
            kind = ref.get("kernel_set") or _kind_of_tissue(tissue)
            if kind in KERNEL_SET_KINDS and kind not in kinds:
                kinds.append(kind)
        if available_kinds is not None:
            kinds = [k for k in kinds if k in available_kinds]
    return [_legacy_spec(kind) for kind in KERNEL_SET_KINDS if kind in kinds]


def _reads_reference_columns(spec: Mapping[str, Any]) -> bool:
    """Whether a set's columns are identified by the kernels' reference alpha_x / beta_x."""
    return spec["kind"] == "alpha_beta" and tuple(spec["fields"]) == ("alpha", "beta")


def _assign_members(
    specs: Sequence[Mapping[str, Any]],
    classes: Sequence[TissueProperties],
    refs: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[TissueProperties, Optional[int]]]]:
    """
    Tissue classes (with their explicit column) of every kernel set.

    A class names its set by ``ref_Kernel``; without one it belongs to the set of its kind
    (MKM parameters: z*, others: α/β), chosen by provenance or the default flag if there are
    several. A class referencing an unknown set belongs to none.
    """
    members: dict[str, list[tuple[TissueProperties, Optional[int]]]] = {s["id"]: [] for s in specs}
    for tissue, ref in zip(classes, refs or [{}] * len(classes)):
        set_id = ref.get("kernel_set")
        if set_id is not None:
            if set_id in members:
                members[set_id].append((tissue, ref.get("column")))
            else:
                logger.warning(
                    "Machine tissue class %r references unknown kernel set '%s'.", tissue, set_id
                )
            continue
        candidates = [s for s in specs if s["kind"] == _kind_of_tissue(tissue)]
        if len(candidates) > 1:
            provenance = tissue_provenance(tissue)
            candidates = [s for s in candidates if s["model"] == provenance] or [
                s for s in candidates if s["default"]
            ]
        if len(candidates) == 1:
            members[candidates[0]["id"]].append((tissue, ref.get("column")))
    return members


def _field_problems(
    spec: Mapping[str, Any], kernels: Optional[Mapping[Any, Any]]
) -> tuple[list[str], bool]:
    """Problems of kernel fields the set reads but the kernels lack, and completeness."""
    if kernels is None:
        return [], True
    set_id, fields = spec["id"], tuple(spec["fields"])
    if not kernels:
        return [
            f"Kernel set '{set_id}' needs pencil-beam kernels with {fields} data, but the "
            "machine carries none."
        ], False
    problems = []
    for name in fields:
        lacking = [energy for energy, k in kernels.items() if _field_state(k, name) is False]
        if lacking:
            problems.append(
                f"Kernel set '{set_id}' reads kernel field '{name}', which the kernels of "
                f"{len(lacking)} energies (e.g. {lacking[0]}) do not carry."
            )
    return problems, not problems


def _column_problems(
    spec: Mapping[str, Any], kernels: Optional[Mapping[Any, Any]], columns: Sequence[Any]
) -> list[str]:
    n_columns = _n_columns(kernels, tuple(spec["fields"])) if kernels else None
    if n_columns is None:
        return []
    outside = [c for c in columns if c is not None and c >= n_columns]
    if not outside:
        return []
    return [
        f"Kernel set '{spec['id']}' tabulates {n_columns} tissue classes, but its tissue "
        f"classes reference columns {[c + 1 for c in outside]}."
    ]


def _set_info(
    spec: Mapping[str, Any],
    tissues: Sequence[tuple[TissueProperties, Optional[int]]],
    columns: Sequence[Optional[int]],
    problems: Sequence[str],
    complete: bool,
) -> KernelSetInfo:
    provenance = spec["model"]
    if not spec["declared"]:
        # Legacy sets take the provenance their tissue classes declare unanimously.
        named = {tissue_provenance(t) for t, _ in tissues}
        if len(named) == 1 and UNDECLARED_PROVENANCE not in named:
            provenance = named.pop()
    dependent = spec["kind"] in TISSUE_DEPENDENT_KINDS
    return KernelSetInfo(
        id=spec["id"],
        kind=spec["kind"],
        fields=tuple(spec["fields"]),
        variant=spec["variant"],
        provenance=provenance,
        reference=spec["reference"],
        default=spec["default"],
        declared=spec["declared"],
        tissue_classes=tuple(
            TissueClassInfo(tissue=t, kernel_set=spec["id"], column=c if dependent else None)
            for (t, _), c in zip(tissues, columns)
        ),
        problems=tuple(problems),
        complete=complete,
    )


def _match_reference_columns(
    tissues: Sequence[tuple[TissueProperties, Optional[int]]],
    reference: tuple[np.ndarray, np.ndarray],
    explicit: bool,
) -> tuple[list[Optional[int]], list[str]]:
    """
    Columns of α/β tissue classes, matched onto the kernels' reference alpha_x / beta_x.

    Classes with an explicit column keep it. Without any explicit column the classes must
    correspond one-to-one to the kernel columns, as legacy data has no other way of telling
    which column belongs to which class.
    """
    kernel_rows = list(zip(reference[0].tolist(), reference[1].tolist()))
    problems: list[str] = []
    if not explicit and tissues and len(tissues) != len(kernel_rows):
        problems.append(
            f"The machine declares {len(tissues)} tissue classes for its α/β kernels "
            f"({[t for t, _ in tissues]}), but the kernels tabulate {len(kernel_rows)} "
            f"classes with (alpha_x, beta_x) = {kernel_rows}."
        )
        return [None] * len(tissues), problems

    columns: list[Optional[int]] = []
    taken: set[int] = set()
    for tissue, column in tissues:
        if column is not None:
            columns.append(column)
            continue
        rows = [
            row
            for row, (alpha_x, beta_x) in enumerate(kernel_rows)
            if np.isclose(tissue.alpha_x, alpha_x, rtol=1e-6, atol=0.0)
            and np.isclose(tissue.beta_x, beta_x, rtol=1e-6, atol=0.0)
        ]
        if len(rows) != 1 or rows[0] in taken:
            problems.append(
                f"Machine tissue class {tissue} does not identify exactly one alpha/beta "
                f"kernel row; the kernels tabulate (alpha_x, beta_x) = {kernel_rows}."
            )
            columns.append(None)
            continue
        taken.add(rows[0])
        columns.append(rows[0])
    return columns, problems


def machine_kernel_sets(machine: Any) -> dict[str, KernelSetInfo]:
    """
    Kernel sets of a machine, a :class:`BioMachineSummary` or a duck-typed stand-in.

    Machines and summaries report their own ``kernel_sets``; other objects get them
    inferred from their ``pb_kernels`` and ``tissue_properties`` (and, lacking kernel data,
    from the tissue classes and the quantities they provide).
    """
    if machine is None:
        return {}
    sets = getattr(machine, "kernel_sets", None)
    if isinstance(sets, Mapping):
        return dict(sets)

    tissues = getattr(machine, "tissue_properties", None)
    tissues = [] if tissues is None else list(tissues)
    fields_set = getattr(machine, "model_fields_set", None)
    declared = fields_set is None or "tissue_properties" in fields_set
    kernels = getattr(machine, "pb_kernels", None)
    available = None
    if kernels is None:
        provided = getattr(machine, "provided_quantities", None)
        if callable(provided):
            quantities = set(provided())
            available = [
                kind
                for kind, names in KERNEL_SET_QUANTITIES.items()
                if all(name in quantities for name in names)
            ]
    try:
        declarations = parse_kernel_set_declarations(
            getattr(machine, "kernel_set_declarations", None)
        )
    except ValueError:
        declarations = []
    return build_kernel_sets(
        declarations=declarations,
        tissue_classes=tissues,
        tissue_refs=getattr(machine, "tissue_class_refs", None),
        kernels=kernels,
        classes_declared=declared,
        available_kinds=available,
    )


def kernel_set_quantities(sets: Mapping[str, KernelSetInfo]) -> list[str]:
    """
    Quantities the kernel sets provide.

    These are the quantities of their kinds and, per kind with several sets,
    ``"<quantity>:<variant>"`` for each variant.
    """
    quantities: list[str] = []
    for kind, names in KERNEL_SET_QUANTITIES.items():
        of_kind = [s for s in sets.values() if s.kind == kind]
        if not of_kind:
            continue
        quantities.extend(n for n in names if n not in quantities)
        if len(of_kind) > 1:
            for s in of_kind:
                if s.variant:
                    quantities.extend(
                        f"{n}:{s.variant}" for n in names if f"{n}:{s.variant}" not in quantities
                    )
    return quantities


@dataclass(frozen=True, eq=False)
class BioMachineSummary:
    """
    What biological model options and tissue checks read off a machine, without its kernels.

    Base data files run to hundreds of megabytes; a summary keeps only the metadata and can
    be cached and passed wherever :func:`~pyRadPlan.biology.models.bio_model_options` or
    :meth:`~pyRadPlan.biology.models.BiologicalModel.check_tissues` take a machine.

    Attributes
    ----------
    name : str
        Machine name.
    radiation_mode : str
        Radiation mode of the machine.
    kernel_sets : Mapping[str, KernelSetInfo]
        The machine's kernel sets.
    tissue_properties : tuple[TissueProperties, ...]
        Every tissue class of the machine.
    quantities : tuple[str, ...]
        Quantities the machine provides (``Machine.provided_quantities()``).
    has_fluence_spectrum : bool
        Whether the kernels carry fragment fluence spectra.
    """

    name: str = ""
    radiation_mode: str = ""
    kernel_sets: Mapping[str, KernelSetInfo] = field(default_factory=dict)
    tissue_properties: tuple[TissueProperties, ...] = ()
    quantities: tuple[str, ...] = ()
    has_fluence_spectrum: bool = False

    @classmethod
    def from_machine(cls, machine: Any) -> "BioMachineSummary":
        """Summarize a machine (or return a summary unchanged)."""
        if isinstance(machine, cls):
            return machine
        provided = getattr(machine, "provided_quantities", None)
        quantities = tuple(provided()) if callable(provided) else ()
        tissues = getattr(machine, "tissue_properties", None)
        fluence = getattr(machine, "has_fluence_spectrum", None)
        return cls(
            name=str(getattr(machine, "name", "") or ""),
            radiation_mode=str(getattr(machine, "radiation_mode", "") or ""),
            kernel_sets=machine_kernel_sets(machine),
            tissue_properties=tuple(tissues) if tissues is not None else (),
            quantities=quantities,
            has_fluence_spectrum=bool(fluence) if fluence is not None else "fluence" in quantities,
        )

    def provided_quantities(self) -> list[str]:
        """Quantities the base data provides, as :meth:`Machine.provided_quantities` does."""
        return list(self.quantities)
