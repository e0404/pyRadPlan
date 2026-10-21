"""
Converting tissue properties between provenances and applying tissue classes of base data.

A cell-line preset or a structure's tissue is often available for another provenance than
the base data a biological model reads (e.g. an MKM preset for mMKM z* kernels).
:func:`adopt_provenance` converts it, keeping the parameters both provenances share.
:func:`apply_tissue_class` copies a tissue class of base data onto a structure's tissue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from ._base import REFERENCE_FIELDS, TissueProperties
from ._provenance import Provenance, get_provenance, provenance_label, tissue_provenance

# Fields naming or describing a tissue rather than parameterizing it, and the reference
# radiation, which a tissue keeps as long as its new type knows it.
_DESCRIPTIVE_FIELDS = (
    "cell_line",
    "cell_line_source",
    "model",
    "version",
    "reference_sources",
    "reference_radiation",
)
_KEPT_FIELDS = ("cell_line", "cell_line_source", "reference_sources", "reference_radiation")

# Reference sources of values a tissue class of base data declares.
_DECLARED_SOURCES = ("base_data", "explicit", "published")


def _parameter_fields(tissue_type: type[TissueProperties]) -> tuple[str, ...]:
    return tuple(name for name in tissue_type.model_fields if name not in _DESCRIPTIVE_FIELDS)


def _is_set(value: Any) -> bool:
    return not (isinstance(value, float) and math.isnan(value))


def _target_provenance(target: Any) -> Provenance:
    if isinstance(target, Provenance):
        return target
    if isinstance(target, TissueProperties):
        return get_provenance(tissue_provenance(target))
    if isinstance(target, type) and issubclass(target, TissueProperties):
        default = target.model_fields["model"].default
        version = (
            target.model_fields["version"].default if "version" in target.model_fields else None
        )
        provenance = get_provenance(default, version)
        if not issubclass(provenance.tissue_type, target):
            raise ValueError(f"No provenance is described by {target.__name__}.")
        return provenance
    return get_provenance(target)


@dataclass(frozen=True, eq=False)
class ProvenanceAdoption:
    """
    Result of :func:`adopt_provenance`.

    Attributes
    ----------
    tissue : TissueProperties
        The converted tissue.
    source : str
        Canonical provenance of the original tissue, e.g. ``"MKM"``.
    target : str
        Canonical provenance adopted, e.g. ``"mMKM"``.
    carried : tuple[str, ...]
        Parameters taken over from the original tissue (always `alpha_x` / `beta_x`).
    defaulted : tuple[str, ...]
        Parameters of the target the original did not provide, left at their defaults.
    cell_line : str or None
        Cell line of the original tissue, kept on the converted one.
    """

    tissue: TissueProperties
    source: str
    target: str
    carried: tuple[str, ...]
    defaulted: tuple[str, ...]
    cell_line: Optional[str] = None

    @property
    def changed(self) -> bool:
        """Whether the provenance actually changed."""
        return self.source != self.target

    def describe(self) -> str:
        """
        Where the values came from, e.g. ``"values from HSG · MKM; adopted as mMKM"``.

        Without a change of provenance only the origin is named.
        """
        origin = provenance_label(self.source)
        if self.cell_line:
            origin = f"{self.cell_line} · {origin}"
        if not self.changed:
            return f"values from {origin}"
        return f"values from {origin}; adopted as {provenance_label(self.target)}"


def adopt_provenance(tissue: TissueProperties, target: Any) -> ProvenanceAdoption:
    """
    Convert a tissue to another provenance, keeping the parameters both share.

    Meant for "load a preset of another provenance, then edit": e.g. an MKM cell-line
    preset adopted as mMKM keeps `r_nucleus`, `r_domain`, `alpha_0` and `beta_0`, a LEM I
    preset adopted as LEM IV keeps `d_t` and `r_nucleus`. Parameters are shared only between
    provenances described by the same tissue properties class (the MKM variants, the LEM
    versions); across classes (LEM to MKM) only the reference parameters `alpha_x` /
    `beta_x` are, since equally named parameters of different models are fitted
    differently. The reference parameters keep their `reference_sources` (derived ones are
    derived anew by the target, e.g. saturation-corrected for mMKM), the cell line, its
    source and the reference radiation are kept, and every other parameter of the target gets
    its default.

    Parameters
    ----------
    tissue : TissueProperties
        The tissue to convert; not modified.
    target : str, Provenance, TissueProperties or type[TissueProperties]
        The provenance to adopt: a name as :func:`get_provenance` accepts it (``"mMKM"``,
        ``"LEM_IV"``), a provenance, a tissue whose provenance to adopt, or a tissue
        properties class (selecting its default model, e.g. ``MKM`` or ``LEM I``).

    Returns
    -------
    ProvenanceAdoption
        The converted tissue and which parameters were carried over or defaulted.
    """
    provenance = _target_provenance(target)
    target_type = provenance.tissue_type
    shared = (
        _parameter_fields(target_type)
        if type(tissue) is target_type
        else tuple(name for name in REFERENCE_FIELDS if name in target_type.model_fields)
    )
    carried = tuple(
        name
        for name in shared
        if name in type(tissue).model_fields and _is_set(getattr(tissue, name))
    )
    params: dict[str, Any] = {name: getattr(tissue, name) for name in carried}
    params.update(provenance.parameters)
    if not provenance.registered and provenance.name:
        params["model"] = provenance.name
    for name in _KEPT_FIELDS:
        if name in type(tissue).model_fields and name in target_type.model_fields:
            params[name] = getattr(tissue, name)
    adopted = target_type.model_validate(params)
    defaulted = tuple(name for name in _parameter_fields(target_type) if name not in carried)
    return ProvenanceAdoption(
        tissue=adopted,
        source=tissue_provenance(tissue),
        target=tissue_provenance(adopted),
        carried=carried,
        defaulted=defaulted,
        cell_line=tissue.cell_line,
    )


def declared_reference_parameters(tissue_class: TissueProperties) -> dict[str, float]:
    """
    Return the reference parameters a tissue class of base data declares.

    Those that are finite and were given by the base data (or explicitly, or as
    published), not defaulted or estimated; empty for a class leaving them NaN, such as the
    mMKM class of the bundled oxygen machine.
    """
    return {
        name: float(getattr(tissue_class, name))
        for name in REFERENCE_FIELDS
        if tissue_class.reference_sources[name] in _DECLARED_SOURCES
        and math.isfinite(getattr(tissue_class, name))
    }


def apply_tissue_class(
    tissue: TissueProperties, tissue_class: TissueProperties
) -> TissueProperties:
    """
    Copy a tissue class of base data onto a structure's tissue.

    The tissue adopts the class's provenance (see :func:`adopt_provenance`; a
    reference-only class of undeclared provenance keeps the tissue's), then takes over
    the parameters identifying the class (:attr:`~TissueProperties.identifying_fields`),
    except those the class leaves undeclared (NaN). Everything else is kept, e.g. the
    `alpha_0` / `beta_0` MKM reads per voxel and the cell line.

    Reference parameters `alpha_x` / `beta_x` follow the resolution order of
    :class:`TissueProperties`: identifying ones (LEM, reference-only classes) are copied
    since they must match; otherwise the values the class declares
    (:func:`declared_reference_parameters`) replace only defaulted or base-data ones, never
    explicit, published or estimated ones. A copied reference value gets the source
    ``"base_data"`` in the tissue's `reference_sources`; the other keeps its own.

    Parameters
    ----------
    tissue : TissueProperties
        The structure's tissue; not modified.
    tissue_class : TissueProperties
        The tissue class, e.g. one entry of ``machine.tissue_properties``.

    Returns
    -------
    TissueProperties
        A new tissue that matches the class (``tissue.matches(tissue_class,
        ignore_undeclared=True)``) unless the class declares no identifying parameter.
    """
    class_provenance = get_provenance(tissue_provenance(tissue_class))
    if type(tissue_class) is TissueProperties and not class_provenance.declared:
        result = tissue
    else:
        result = adopt_provenance(tissue, class_provenance).tissue

    changes: dict[str, Any] = {}
    for name in type(tissue_class).identifying_fields:
        if name not in type(result).model_fields:
            continue
        value = getattr(tissue_class, name)
        if _is_set(value):
            changes[name] = value

    for name, value in declared_reference_parameters(tissue_class).items():
        if name not in changes and result.reference_sources[name] in ("default", "base_data"):
            changes[name] = value

    copied = [name for name in REFERENCE_FIELDS if name in changes]
    if copied:
        changes["reference_sources"] = {
            **result.reference_sources,
            **dict.fromkeys(copied, "base_data"),
        }
    if not changes and result is tissue:
        return tissue.model_copy()
    return type(result).model_validate({**result.model_dump(), **changes})
