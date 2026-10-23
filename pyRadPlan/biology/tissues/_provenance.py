"""
Provenance registry: the radiobiological model that produced a set of base data.

Base data (α/β or z* kernels, RBE tables) is computed with one radiobiological model and
parameterization, its *provenance* (e.g. LEM I, MKM, mMKM). The tissue classes of such
data carry the parameters of that model, so the provenance decides which
:class:`~pyRadPlan.biology.tissues.TissueProperties` subclass describes them. New
provenances register here with :func:`register_provenance`;
:meth:`TissueProperties.from_dict` dispatches through this registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

from ._base import (
    LEM_VERSIONS,
    LEMTissueProperties,
    MKMTissueProperties,
    TissueProperties,
    _parse_lem_version,
)

#: Provenance of base data whose metadata does not name the model that produced it.
UNDECLARED_PROVENANCE = "undeclared"


@dataclass(frozen=True)
class Provenance:
    """
    One radiobiological model (and version) base data can be produced with.

    Attributes
    ----------
    name : str
        Canonical identifier, e.g. ``"LEM_I"``, ``"MKM"`` or ``"undeclared"``.
    label : str
        Display name, e.g. ``"LEM I"`` or ``"model not declared"``.
    tissue_type : type[TissueProperties]
        Tissue properties class holding the parameters of this provenance.
    parameters : Mapping[str, Any]
        Constructor values fixed by the provenance, e.g. ``{"model": "LEM", "version": "I"}``.
    family : str
        Model family, e.g. ``"LEM"`` for every LEM version.
    registered : bool
        Whether the provenance is known to the registry; unknown names are still accepted
        and described by plain :class:`TissueProperties`.
    """

    name: str
    label: str
    tissue_type: type[TissueProperties]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    family: str = ""
    registered: bool = True

    @property
    def declared(self) -> bool:
        """Whether the base data names its model at all."""
        return self.name != UNDECLARED_PROVENANCE


_PROVENANCES: dict[str, Provenance] = {}
_ALIASES: dict[str, str] = {}


def _key(name: str) -> str:
    return name.strip().casefold()


def register_provenance(
    name: str,
    tissue_type: type[TissueProperties],
    *,
    label: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    family: Optional[str] = None,
    aliases: tuple[str, ...] = (),
) -> Provenance:
    """
    Register a provenance and the tissue properties class describing its tissue classes.

    Parameters
    ----------
    name : str
        Canonical identifier as written into base-data metadata, e.g. ``"mMKM"``.
    tissue_type : type[TissueProperties]
        Class holding the parameters of this provenance.
    label : str, optional
        Display name; defaults to ``name``.
    parameters : mapping, optional
        Constructor values every tissue class of this provenance gets, e.g. its ``model``.
    family : str, optional
        Model family; defaults to ``name``.
    aliases : tuple[str, ...]
        Further names (case-insensitive) resolving to this provenance.

    Returns
    -------
    Provenance
        The registered provenance.
    """
    if not isinstance(tissue_type, type) or not issubclass(tissue_type, TissueProperties):
        raise TypeError("A provenance needs a TissueProperties subclass as tissue type.")
    provenance = Provenance(
        name=name,
        label=label or name,
        tissue_type=tissue_type,
        parameters=MappingProxyType(dict(parameters or {})),
        family=family or name,
    )
    _PROVENANCES[name] = provenance
    for alias in (name, *aliases):
        _ALIASES[_key(alias)] = name
    return provenance


def registered_provenances() -> tuple[Provenance, ...]:
    """Every registered provenance, in registration order."""
    return tuple(_PROVENANCES.values())


def get_provenance(name: Any = None, version: Any = None) -> Provenance:
    """
    Resolve a provenance from the model name (and version) given in base-data metadata.

    Accepted are the canonical names, their aliases (case-insensitive), the LEM version
    spellings (``"LEM_II"``, ``"LEM 2"``, ``"LEMII"``) and ``model: LEM`` with a separate
    ``version`` (``"LEM_I"`` or ``"I"``); a bare ``"LEM"`` names LEM I. A missing or empty
    name, ``"Generic"`` and ``"not declared"`` resolve to the undeclared provenance. An unknown
    name resolves to an unregistered provenance described by plain :class:`TissueProperties`.
    """
    model = "" if name is None else str(name).strip()
    version_text = "" if version is None else str(version).strip()

    lem_version = _parse_lem_version(version_text) if version_text else None
    if lem_version is not None and (not model or _key(model) == "lem"):
        return _PROVENANCES[f"LEM_{lem_version}"]

    canonical = _ALIASES.get(_key(model))
    if canonical is not None:
        return _PROVENANCES[canonical]

    model_version = _parse_lem_version(model) if model else None
    if model_version is not None:
        return _PROVENANCES[f"LEM_{model_version}"]

    return Provenance(
        name=model,
        label=model,
        tissue_type=TissueProperties,
        family=model,
        registered=False,
    )


def provenance_label(name: Any) -> str:
    """Display name of a provenance, e.g. ``"LEM I"`` or ``"model not declared"``."""
    return get_provenance(name).label


def tissue_provenance(tissue: TissueProperties) -> str:
    """
    Canonical provenance a tissue's parameters belong to.

    A LEM tissue names its version (``"LEM_II"``); a plain :class:`TissueProperties` carrying
    only reference parameters is ``"undeclared"`` unless its ``model`` names one.
    """
    if isinstance(tissue, LEMTissueProperties):
        return f"LEM_{tissue.version}"
    return get_provenance(getattr(tissue, "model", None)).name


register_provenance(
    UNDECLARED_PROVENANCE,
    TissueProperties,
    label="model not declared",
    family=UNDECLARED_PROVENANCE,
    aliases=("Generic", "not declared", "none", ""),
)
for _numeral in LEM_VERSIONS:
    register_provenance(
        f"LEM_{_numeral}",
        LEMTissueProperties,
        label=f"LEM {_numeral}",
        parameters={"model": "LEM", "version": _numeral},
        family="LEM",
        aliases=("LEM",) if _numeral == "I" else (),
    )
register_provenance(
    "MKM",
    MKMTissueProperties,
    parameters={"model": "MKM"},
    family="MKM",
)
register_provenance(
    "mMKM",
    MKMTissueProperties,
    parameters={"model": "mMKM"},
    family="MKM",
)
