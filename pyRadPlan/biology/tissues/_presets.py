"""
Cell line presets: published (or otherwise sourced) tissue parameters of a cell line.

Each preset holds the parameters of one cell line for one provenance (e.g. HSG for LEM II)
together with where they come from (:class:`ParameterSource`). A cell line and provenance
may have several presets, one per source (e.g. HSG · LEM III of Manganaro et al. 2018 and of
Elsässer et al. 2008), told apart by the source's label; the first registered is the
default. A tissue built from a preset (:meth:`TissueProperties.from_preset`) records the
cell line and the source's citation, which identifies the preset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union

from ._base import REFERENCE_FIELDS, TissueProperties
from ._provenance import get_provenance
from ._reference import DEFAULT_REFERENCE_RADIATION, OKAMOTO_2011
from ._sources import UNVERIFIED_SOURCE, ParameterSource


@dataclass(frozen=True)
class CellLinePreset:
    """
    Parameters of one cell line for one provenance, with their source.

    Attributes
    ----------
    cell_line : str
        Name of the cell line, e.g. ``"HSG"``.
    provenance : str
        Canonical provenance, e.g. ``"MKM"`` or ``"LEM_II"``.
    parameters : Mapping[str, Any]
        Model and reference parameters as tissue field names. A model deriving its
        reference parameters (MKM) gives its model parameters and the reference radiation
        they were published for (`reference_radiation`), so that the reference follows from
        the conversion, unless the preset also gives the published reference parameters
        (`alpha_x` / `beta_x`), which are then kept as published.
    source : ParameterSource
        Where the values come from; its label tells the preset apart from others of the
        same cell line and provenance, its citation identifies it on a tissue.
    reference : ParameterSource, optional
        Where published reference parameters come from, if not from `source` (e.g. the
        X-ray measurement a model fit refers to).
    """

    cell_line: str
    provenance: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    source: ParameterSource = field(default_factory=ParameterSource)
    reference: Optional[ParameterSource] = None

    @property
    def label(self) -> str:
        """Label of the preset among those of its cell line and provenance: its source's."""
        return self.source.label

    @property
    def reference_radiation(self) -> Optional[str]:
        """
        Registered reference radiation the preset's values refer to, None if it names none.

        Set for presets of a model deriving its reference parameters (MKM / mMKM), see
        :func:`~pyRadPlan.biology.tissues.get_reference_radiation`.
        """
        return self.parameters.get("reference_radiation")

    @property
    def published_reference(self) -> dict[str, float]:
        """Reference parameters the preset gives as published, by name."""
        return {
            name: self.parameters[name] for name in REFERENCE_FIELDS if name in self.parameters
        }

    @property
    def reference_label(self) -> str:
        """Short source of the published reference parameters, e.g. ``"Furusawa 2000"``."""
        return (self.reference or self.source).label

    @property
    def group(self) -> tuple[str, str]:
        """Cell line (upper case) and canonical provenance the preset is one of."""
        return _group(self.cell_line, self.provenance)

    @property
    def key(self) -> tuple[str, str, str]:
        """Key of the preset in :data:`CELL_LINE_PRESETS`: :attr:`group` and :attr:`label`."""
        return (*self.group, self.label)

    def tissue(
        self, tissue_type: Optional[type[TissueProperties]] = None, **overrides
    ) -> TissueProperties:
        """
        Build the tissue the preset describes, recording its cell line and source.

        Reference parameters the preset gives are ``"published"``, those of `overrides`
        ``"explicit"`` (see :data:`~pyRadPlan.biology.tissues.REFERENCE_SOURCES`).

        Parameters
        ----------
        tissue_type : type[TissueProperties], optional
            Class to build; the provenance's tissue class if None.
        **overrides
            Field values replacing those of the preset.
        """
        provenance = get_provenance(self.provenance)
        sources = {
            **dict.fromkeys(self.published_reference, "published"),
            **{name: "explicit" for name in REFERENCE_FIELDS if name in overrides},
        }
        params = {
            **provenance.parameters,
            "cell_line": self.cell_line,
            "cell_line_source": self.source.citation,
            **self.parameters,
            "reference_sources": sources,
            **overrides,
        }
        return (tissue_type or provenance.tissue_type)(**params)


def _group(cell_line: str, model: Any) -> tuple[str, str]:
    return cell_line.strip().upper(), get_provenance(model).name


#: The registered cell line presets, keyed by (cell line in upper case, canonical
#: provenance, label of the source), e.g. ``("HSG", "LEM_III", "Elsässer 2008")``, in
#: registration order: the first of a cell line and provenance is its default. Add presets
#: with :func:`register_cell_line_preset`, list them with
#: :func:`registered_cell_line_presets`.
CELL_LINE_PRESETS: dict[tuple[str, str, str], CellLinePreset] = {}


def register_cell_line_preset(
    cell_line: str,
    model: str,
    parameters: Optional[Mapping[str, Any]] = None,
    *,
    source: Union[str, ParameterSource, None] = UNVERIFIED_SOURCE,
    reference: Union[str, ParameterSource, None] = None,
    default: bool = False,
    validate: bool = True,
    **params: Any,
) -> CellLinePreset:
    """
    Register a cell line preset, replacing one of the same cell line, provenance and source.

    Presets of the same cell line and provenance are told apart by the label of their
    source (:attr:`ParameterSource.label`: its `short` form); the first registered is the
    default of :func:`get_cell_line_preset`.

    Parameters
    ----------
    cell_line : str
        Name of the cell line, e.g. ``"HSG"``; looked up case-insensitively.
    model : str
        Provenance as :func:`~pyRadPlan.biology.tissues.get_provenance` accepts it, e.g.
        ``"mMKM"`` or ``"LEM_II"``; it must be registered.
    parameters : mapping, optional
        Tissue parameters by field name; merged with `params`. Reference parameters
        (`alpha_x` / `beta_x`) given for a model deriving them (MKM) are kept as published
        rather than derived.
    source : str or ParameterSource
        Where the values come from: any provenance string (paper, toolkit, dataset,
        personal communication, ...), or :data:`UNVERIFIED_SOURCE` (the default) /
        :data:`UNPUBLISHED_SOURCE`. Its citation must differ from those of the other
        presets of the cell line and provenance, since a tissue records the preset by it.
    reference : str or ParameterSource, optional
        Where published reference parameters come from, if not from `source`.
    default : bool
        Make the preset the default of its cell line and provenance.
    validate : bool
        Build the tissue once, so invalid parameters raise here rather than on use.
    **params
        Further tissue parameters, e.g. ``r_domain=0.32``.

    Returns
    -------
    CellLinePreset
        The registered preset.
    """
    provenance = get_provenance(model)
    if not provenance.registered:
        raise ValueError(f"Unknown provenance {model!r} for cell line preset {cell_line!r}.")
    preset = CellLinePreset(
        cell_line=cell_line.strip(),
        provenance=provenance.name,
        parameters=MappingProxyType({**(parameters or {}), **params}),
        source=ParameterSource.coerce(source),
        reference=None if reference is None else ParameterSource.coerce(reference),
    )
    for other in registered_cell_line_presets(preset.cell_line, preset.provenance):
        if other.key != preset.key and other.source.citation == preset.source.citation:
            raise ValueError(
                f"Preset {preset.key} has the citation of preset {other.key}; presets of a "
                "cell line and provenance need distinct citations."
            )
    if validate:
        preset.tissue()
    CELL_LINE_PRESETS[preset.key] = preset
    if default:
        _make_default(preset)
    return preset


def _make_default(preset: CellLinePreset) -> None:
    """Move a registered preset before the others of its cell line and provenance."""
    items = [(key, other) for key, other in CELL_LINE_PRESETS.items() if key != preset.key]
    position = next(
        (i for i, (_key, other) in enumerate(items) if other.group == preset.group), len(items)
    )
    items.insert(position, (preset.key, preset))
    CELL_LINE_PRESETS.clear()
    CELL_LINE_PRESETS.update(items)


def registered_cell_line_presets(
    cell_line: Optional[str] = None, model: Optional[str] = None
) -> list[CellLinePreset]:
    """
    Return the registered cell line presets, in registration order.

    Parameters
    ----------
    cell_line : str, optional
        Only the presets of this cell line (case-insensitive).
    model : str, optional
        Only the presets of this provenance, as
        :func:`~pyRadPlan.biology.tissues.get_provenance` accepts it.
    """
    presets = list(CELL_LINE_PRESETS.values())
    if cell_line is not None:
        presets = [p for p in presets if p.group[0] == cell_line.strip().upper()]
    if model is not None:
        provenance = get_provenance(model).name
        presets = [p for p in presets if p.provenance == provenance]
    return presets


def get_cell_line_preset(
    cell_line: str, model: str, source: Optional[str] = None
) -> CellLinePreset:
    """
    Return a preset of a cell line for a provenance.

    Parameters
    ----------
    cell_line : str
        Cell line, case-insensitive.
    model : str
        Provenance as :func:`~pyRadPlan.biology.tissues.get_provenance` accepts it, e.g.
        ``"LEM"`` (LEM I), ``"LEM_II"``, ``"mMKM"``.
    source : str, optional
        Which preset of several: its label (case-insensitive, e.g. ``"Elsässer 2008"``) or
        the full citation of its source, as a tissue built from it records it
        (`cell_line_source`). None selects the default, the first registered.

    Raises
    ------
    ValueError
        If there is no such preset.
    """
    group = _group(cell_line, model)
    presets = registered_cell_line_presets(*group)
    if not presets:
        available = sorted({preset.group for preset in CELL_LINE_PRESETS.values()})
        raise ValueError(f"Unknown preset {group}. Available: {available}")
    if source is None:
        return presets[0]
    wanted = source.strip()
    for preset in presets:
        if preset.label.casefold() == wanted.casefold() or preset.source.citation == wanted:
            return preset
    labels = [preset.label for preset in presets]
    raise ValueError(f"Unknown preset {group} of source {source!r}. Available: {labels}")


MANGANARO_2018 = ParameterSource(
    citation=(
        "Manganaro et al., 'Survival': a simulation toolkit introducing a modular approach "
        "for radiobiological evaluations in ion beam therapy, Phys. Med. Biol. 63 (2018) "
        "08NT01, Table 1 (p. 5), https://doi.org/10.1088/1361-6560/aab697"
    ),
    short="Manganaro 2018",
)

_ELSASSER_2007 = (
    "Elsässer and Scholz, Radiat. Res. 167 (2007) 319, https://doi.org/10.1667/RR0467.1"
)

_FURUSAWA_2000 = (
    "Furusawa et al., Radiat. Res. 154 (2000) 485, Table 1 (p. 487), "
    "https://doi.org/10.1667/0033-7587(2000)154[0485:IOAAHC]2.0.CO;2"
)

#: Where the presets' X-ray LQ parameters were measured (Furusawa et al. 2000, p. 486).
_FURUSAWA_XRAYS = (
    "200 kVp X-rays (20 mA, W target, 0.5 mm Al + 0.5 mm Cu), aerobic cells in plastic "
    "dishes, LQ fit to the mean of five (V79, HSG) or two (T1) experiments"
)

#: Source of the published X-ray LQ parameters of aerobic HSG, V79 and T1 cells.
FURUSAWA_2000 = ParameterSource(
    citation=(
        "Furusawa et al., Inactivation of aerobic and hypoxic cells from three different cell "
        "lines by accelerated 3He-, 12C- and 20Ne-ion beams, Radiat. Res. 154 (2000) 485-496, "
        "Table 1 (p. 487), https://doi.org/10.1667/0033-7587(2000)154[0485:IOAAHC]2.0.CO;2"
    ),
    short="Furusawa 2000",
    note=f"X-ray LQ parameters of aerobic cells: {_FURUSAWA_XRAYS} (p. 486).",
)

INANIWA_2010 = ParameterSource(
    citation=(
        "Inaniwa et al., Treatment planning for a scanned carbon beam with a modified "
        "microdosimetric kinetic model, Phys. Med. Biol. 55 (2010) 6721-6737, Sect. 2.2 "
        "(pp. 6724-6725) and Fig. 3, https://doi.org/10.1088/0031-9155/55/22/008"
    ),
    short="Inaniwa 2010",
    note=(
        "R_n = 3.9 µm, r_d = 0.32 µm and alpha_0 = 0.172 Gy⁻¹ fitted (weighted least squares) "
        "to the D10 of aerobic HSG cells for 3He, 12C and 20Ne ions of Furusawa et al., "
        "Radiat. Res. 154 (2000) 485, with cylindrical domain and nucleus and the "
        "Kiefer-Chatterjee track structure; beta = 0.0615 Gy⁻² taken over from Kase et al., "
        "Phys. Med. Biol. 53 (2008) 37 (p. 6724). Saturation correction: Eqs. (3)-(5). The "
        "paper gives no y_D or z_1D of X-rays: alpha_0 is the alpha in the limit LET = 0 and "
        "the reference radiation of its RBE (Eq. 6) is not specified. With 200 kV X-rays "
        "(y_D = 4.2 keV/µm) the conversion gives alpha_x = 0.301 Gy⁻¹; the paper quotes "
        "alpha_x = 0.331 Gy⁻¹, beta_x = 0.0593 Gy⁻² (D10 = 4.04 Gy) for 200 kVp X-rays measured "
        "by Furusawa et al. 2000 (p. 6728), whose Table 1 (p. 487) however lists alpha = "
        "0.313 Gy⁻¹, beta = 0.0615 Gy⁻² (D10 = 4.08 Gy) for aerobic HSG cells. For the HSG "
        "cells of 2010 it corrects alpha_0 to 0.0708 Gy⁻¹ "
        "(Sect. 2.5, p. 6729: alpha = 0.282 Gy⁻¹ at z* = 3.43 Gy of 290 MeV/u carbon ions), "
        "the value of its treatment planning. Inaniwa et al., Phys. Med. Biol. 60 (2015) "
        "3271 (p. 3276) reuse this set for the clinical NIRS system with a carbon-ion "
        "reference (alpha_r = 0.764 Gy⁻¹, abstract) instead of X-rays. It is the geometry "
        "of the mMKM class of the generic oxygen machine."
    ),
)

_INANIWA_DERIVED = ParameterSource(
    citation=INANIWA_2010.citation,
    short="Inaniwa 2010, derived reference",
    note=(
        f"{INANIWA_2010.note} This preset derives the reference parameters by the conversion "
        "(default 200 kV X-ray reference); the preset 'Inaniwa 2010 / Furusawa 2000 "
        "reference' gives the measured X-ray values instead."
    ),
)

_INANIWA_FURUSAWA = ParameterSource(
    citation=(f"{INANIWA_2010.citation}; reference alpha_X / beta_X: {FURUSAWA_2000.citation}"),
    short="Inaniwa 2010 / Furusawa 2000 reference",
    note=(
        "The mMKM parameters of Inaniwa et al. 2010 (see the preset 'Inaniwa 2010, derived "
        "reference') with the X-ray reference parameters as published: alpha_X = 0.313 "
        "Gy⁻¹, beta_X = 0.0615 Gy⁻² of aerobic HSG cells of Furusawa et al. 2000, Table 1 "
        f"(p. 487; {_FURUSAWA_XRAYS}). Inaniwa et al. 2010 (p. 6728) quote alpha_x = 0.331 "
        "Gy⁻¹, beta_x = 0.0593 Gy⁻² for these measurements, but Table 1 of Furusawa et al. "
        "gives 0.313 / 0.0615. The conversion for 200 kV X-rays (y_D = 4.2 keV/µm) would "
        "derive alpha_x = 0.301 Gy⁻¹."
    ),
)

_OKAMOTO_HSG = ParameterSource(
    citation=(
        "Okamoto et al., Relation between lineal energy distribution and relative biological "
        "effectiveness for photon beams according to the microdosimetric kinetic model, "
        "J. Radiat. Res. 52 (2011) 75-81, Table 2 and Fig. 4 (p. 79), Discussion (p. 80), "
        "https://doi.org/10.1269/jrr.10073"
    ),
    short=OKAMOTO_2011.short,
    note=(
        "Plain MKM fitted to HSG survival after 200 kV X-rays, Co-60 and 6 MV X-rays against "
        "the TEPC y_D (1 µm site, Table 1): the alpha of LQ fits with beta fixed to beta_MK = "
        "0.036 Gy⁻², the mean of the three free fits (Table 2), is linear in y_D, alpha = "
        "0.0338 y_D + 0.0885 (Fig. 4), giving r_d = 0.23 ± 0.03 µm and alpha_0 = 0.088 ± 0.023 "
        "Gy⁻¹ (p. 79, restated p. 80); no nucleus radius, which the plain MKM does not need. "
        "With the measured y_D of 4.51 keV/µm the conversion reproduces the fitted alpha of "
        "0.241 Gy⁻¹ for 200 kV X-rays (Table 2) within 2 %; the free LQ fit for these X-rays "
        "gives alpha = 0.205 Gy⁻¹, beta = 0.043 Gy⁻²."
    ),
)

_KASE_2006 = ParameterSource(
    citation=(
        "Kase et al., Microdosimetric measurements and estimation of human cell survival for "
        "heavy-ion beams, Radiat. Res. 166 (2006) 629-638, pp. 632-633 and Eqs. (11)-(12), "
        "https://doi.org/10.1667/RR0536.1"
    ),
    short="Kase 2006",
    note=(
        "HSG · MKM: r_d = 0.42 ± 0.04 µm and alpha_0 = 0.13 ± 0.03 Gy⁻¹ (Eqs. 11-12, p. 633) "
        "from the alpha of HSG cells for 200 kVp X-rays (alpha_X = 0.19 Gy⁻¹, y_D = 4.3 "
        "keV/µm measured by Coppola et al.) and for monoenergetic 290 MeV/u carbon ions "
        "(alpha = 0.35 Gy⁻¹, y_D = 15.6 keV/µm); beta = 0.05 Gy⁻², the quadratic coefficient "
        "of the LQ fit to the 200 kVp X-ray survival (p. 632, Fig. 3), taken as independent "
        "of the radiation; R_n = 4.1 µm, the G1 nucleus radius measured for HSG cells (p. "
        "633). Kase et al. correct the saturation by the saturation-corrected lineal energy "
        "y* (y_0 = 150 keV/µm, Eqs. 7-9), not by the mMKM. The preset gives the published "
        "X-ray alpha_X = 0.19 Gy⁻¹ / beta_X = 0.05 Gy⁻² (p. 632); the conversion for 200 kV "
        "X-rays (y_D = 4.2 keV/µm) would derive alpha_x = 0.191 Gy⁻¹."
    ),
)


def _kase_2008(location: str, note: str, reference: str, alpha_x: float) -> ParameterSource:
    return ParameterSource(
        citation=(
            "Kase et al., Biophysical calculation of cell survival probabilities using "
            "amorphous track structure models for heavy-ion irradiation, Phys. Med. Biol. 53 "
            f"(2008) 37-59, {location}, https://doi.org/10.1088/0031-9155/53/1/003"
        ),
        short="Kase 2008",
        note=(
            f"{note} Kase et al. use cylindrical domains, the Kiefer-Chatterjee track "
            "structure and Hawkins' non-Poisson correction, not the mMKM saturation. They take "
            "alpha_0 / beta equal to the X-ray alpha / beta, the z_1D of X-rays being "
            "negligible against that of heavy ions (p. 45); Manganaro et al., Phys. Med. Biol. "
            "63 (2018) 08NT01, Table 1, list the same values as alpha_X / beta_X. The preset "
            f"gives these X-ray values as the published reference ({reference}). The "
            "conversion does not neglect the z_1D of X-rays: for 200 kV X-rays (y_D = 4.2 "
            f"keV/µm) it would derive alpha_x = {alpha_x:.3f} Gy⁻¹ instead of alpha_X = "
            "alpha_0."
        ),
    )


def _manganaro_2018(note: str) -> ParameterSource:
    return ParameterSource(MANGANARO_2018.citation, short=MANGANARO_2018.short, note=note)


def _elsasser_2007(location: str, note: str) -> ParameterSource:
    return ParameterSource(
        citation=(
            "Elsässer and Scholz, Cluster effects within the local effect model, Radiat. Res. "
            f"167 (2007) 319-329, {location}, https://doi.org/10.1667/RR0467.1; also "
            f"{MANGANARO_2018.citation}"
        ),
        short="Elsässer 2007",
        note=note,
    )


def _register_builtin(cell_line: str, model: str, source: ParameterSource, **params) -> None:
    register_cell_line_preset(cell_line, model, source=source, validate=False, **params)


# ------------------------------------------------------------------------------- HSG
_register_builtin(
    "HSG",
    "MKM",
    _OKAMOTO_HSG,
    r_domain=0.23,
    alpha_0=0.088,
    beta_0=0.036,
    reference_radiation="xray_200kV_tepc",
)
register_cell_line_preset(
    "HSG",
    "MKM",
    source=_kase_2008(
        "Table 1 (p. 45)",
        "HSG · MKM: alpha_0 = 0.313 Gy⁻¹, beta = 0.0615 Gy⁻², R_d = 0.34 µm, R_n = 4.1 µm, "
        "fitted to the D10 of aerobic HSG cells for 3He, 12C and 20Ne ions of Furusawa et al., "
        "Radiat. Res. 154 (2000) 485; alpha_0 / beta are the X-ray alpha / beta of aerobic "
        "HSG cells of its Table 1 (p. 487, 200 kVp X-rays). Manganaro et al. 2018, Table 1, "
        "list the set as the HSG · MKM row.",
        "Furusawa et al. 2000, Table 1",
        alpha_x=0.427,
    ),
    reference=FURUSAWA_2000,
    validate=False,
    r_nucleus=4.1,
    r_domain=0.34,
    alpha_0=0.313,
    beta_0=0.0615,
    alpha_x=0.313,
    beta_x=0.0615,
    reference_radiation=DEFAULT_REFERENCE_RADIATION,
)
_register_builtin(
    "HSG",
    "MKM",
    _KASE_2006,
    r_nucleus=4.1,
    r_domain=0.42,
    alpha_0=0.13,
    beta_0=0.05,
    alpha_x=0.19,
    beta_x=0.05,
    reference_radiation=DEFAULT_REFERENCE_RADIATION,
)
_HSG_MMKM = dict(r_nucleus=3.9, r_domain=0.32, alpha_0=0.172, beta_0=0.0615)
_register_builtin(
    "HSG", "mMKM", _INANIWA_DERIVED, **_HSG_MMKM, reference_radiation=DEFAULT_REFERENCE_RADIATION
)
register_cell_line_preset(
    "HSG",
    "mMKM",
    source=_INANIWA_FURUSAWA,
    reference=FURUSAWA_2000,
    validate=False,
    **_HSG_MMKM,
    alpha_x=0.313,
    beta_x=0.0615,
    reference_radiation=DEFAULT_REFERENCE_RADIATION,
)
_HSG_LEM_I_II = _elsasser_2007(
    "Fig. 11 (p. 327) and p. 326",
    "HSG · LEM I / II: alpha_X = 0.313 Gy⁻¹, beta_X = 0.0615 Gy⁻², D_t = 30 Gy for the "
    "original LEM (LEM I) and 6.0 Gy for the cluster extension (LEM II) (Fig. 11), "
    "r_nucleus = 5.0 µm with D_t fitted at 30.6 and 137 keV/µm (p. 326). alpha_X / beta_X "
    f"are the X-ray LQ parameters of aerobic HSG cells of {_FURUSAWA_2000} ({_FURUSAWA_XRAYS}). "
    "The same set in Elsässer et al., New J. Phys. 10 (2008) 075005, Figs. 5-6 (pp. 13-14: "
    "D_t = 6 Gy cluster version, 30 Gy original), and Kase et al., Phys. Med. Biol. 53 (2008) "
    "37 (pp. 49-50: R_n = 5.0 µm, D_t = 30 Gy); Manganaro et al. 2018, Table 1, list it as "
    "the HSG · LEM row.",
)
_HSG_LEM_III = _manganaro_2018(
    "HSG · LEM III: alpha_X, beta_X and R_N as for LEM I / II (Elsässer and Scholz 2007, "
    "Fig. 11 and p. 326). The threshold dose of 19 Gy is given by Manganaro et al. only; no "
    "original publication of it was found. LEM III, the velocity-dependent track core "
    "r_min = beta * 40 nm, was introduced by Elsässer, Krämer and Scholz, Int. J. Radiat. "
    "Oncol. Biol. Phys. 71 (2008) 866, https://doi.org/10.1016/j.ijrobp.2008.02.037, which "
    "fit D_t for 15 other cell lines (Table 2, p. 869) but not HSG. Elsässer et al., New J. "
    "Phys. 10 (2008) 075005, fit D_t = 23 Gy (not 19 Gy) for HSG with that track structure, "
    "see the preset 'Elsässer 2008'.",
)
_HSG_LEM_III_ELSASSER_2008 = ParameterSource(
    citation=(
        "Elsässer, Cunrath, Krämer and Scholz, Impact of track structure calculations on "
        "biological treatment planning in ion radiotherapy, New J. Phys. 10 (2008) 075005, "
        "Fig. 6 and Table 1 (p. 14), https://doi.org/10.1088/1367-2630/10/7/075005"
    ),
    short="Elsässer 2008",
    note=(
        "HSG · LEM III: D_t = 23 Gy fitted to the initial slopes of HSG cells after carbon "
        "ions of Furusawa et al. with the energy-dependent track core r_min = beta * 40 nm "
        "(EDR40), the radial dose of the latest LEM (LEM III, p. 8), and alpha = 0.313 Gy⁻¹, "
        "beta = 0.0615 Gy⁻², r_n = 5 µm (Fig. 6, p. 14; RBE uncertainties for D_t ± 10 % in "
        "Table 1). alpha / beta are the X-ray LQ parameters of aerobic HSG cells of "
        f"{_FURUSAWA_2000} ({_FURUSAWA_XRAYS}). Manganaro et al. 2018, Table 1, give D_t = "
        "19 Gy instead, see the preset 'Manganaro 2018'."
    ),
)
for _version, _d_t, _source in (
    ("I", 30, _HSG_LEM_I_II),
    ("II", 6, _HSG_LEM_I_II),
    ("III", 19, _HSG_LEM_III),
    ("III", 23, _HSG_LEM_III_ELSASSER_2008),
):
    _register_builtin(
        "HSG",
        f"LEM_{_version}",
        _source,
        alpha_x=0.313,
        beta_x=0.0615,
        d_t=_d_t,
        r_nucleus=5.0,
    )

# ------------------------------------------------------------------------------- CHO
register_cell_line_preset(
    "CHO",
    "MKM",
    source=_kase_2008(
        "Sect. 3.3 (p. 52)",
        "CHO · MKM: alpha_0 = 0.228 Gy⁻¹, beta = 0.02 Gy⁻², R_d = 0.2 µm, R_n = 4.1 µm, used "
        "for the 1 %, 10 % and 50 % survival doses of CHO-K1 cells after carbon ions "
        "(Weyrather et al. 1999, Fig. 13); alpha_0 / beta are the X-ray alpha_X / beta_X of "
        "the CHO LEM parameters (Elsässer and Scholz 2007, Fig. 8), which Krämer and Scholz, "
        "Phys. Med. Biol. 45 (2000) 3319, Table 1, attribute to Kraft-Weyrather et al. "
        "(1999).",
        "Elsässer and Scholz 2007, Fig. 8",
        alpha_x=0.335,
    ),
    reference=ParameterSource(
        "Elsässer and Scholz, Cluster effects within the local effect model, Radiat. Res. 167 "
        "(2007) 319-329, Fig. 8 (p. 325), https://doi.org/10.1667/RR0467.1",
        short="Elsässer 2007",
        note="X-ray LQ parameters of CHO cells, alpha = 0.228 Gy⁻¹, beta = 0.02 Gy⁻² (Fig. 8).",
    ),
    validate=False,
    r_nucleus=4.1,
    r_domain=0.2,
    alpha_0=0.228,
    beta_0=0.02,
    alpha_x=0.228,
    beta_x=0.02,
    reference_radiation=DEFAULT_REFERENCE_RADIATION,
)
_CHO_LEM_I_II = _elsasser_2007(
    "Fig. 8 and p. 325",
    "CHO · LEM I / II: alpha_X = 0.228 Gy⁻¹, beta_X = 0.02 Gy⁻², D_t = 40 Gy for the "
    "original LEM (LEM I) and 9.5 Gy for the cluster extension (LEM II) (Fig. 8), "
    "r_nucleus = 4.7 µm with D_t fitted at 32 and 275 keV/µm (p. 325); also Kase et al., "
    "Phys. Med. Biol. 53 (2008) 37 (p. 52), and Manganaro et al. 2018, Table 1 (CHO · LEM "
    "row). Krämer and Scholz, Phys. Med. Biol. 45 (2000) 3319, Table 1 (p. 3322), "
    "https://doi.org/10.1088/0031-9155/45/11/314, attribute alpha_X / beta_X = 0.228 / 0.02 "
    "to Kraft-Weyrather et al. (1999) (Weyrather et al., Int. J. Radiat. Biol. 75 (1999) "
    "1357, CHO-K1 against 250 kV X-rays; not checked), with another LEM I set: D_cut = "
    "30 Gy, R_nucl = 5.5 µm, see the preset 'Krämer 2000'.",
)
_CHO_LEM_III = _manganaro_2018(
    "CHO · LEM III: alpha_X, beta_X and R_N as for LEM I / II (Elsässer and Scholz 2007, "
    "Fig. 8 and p. 325). The threshold dose of 55 Gy is given by Manganaro et al. only; no "
    "original publication of it was found. The paper introducing LEM III (Elsässer, Krämer "
    "and Scholz, Int. J. Radiat. Oncol. Biol. Phys. 71 (2008) 866) has no CHO set.",
)
_CHO_LEM_I_KRAMER_2000 = ParameterSource(
    citation=(
        "Krämer and Scholz, Treatment planning for heavy-ion radiotherapy: calculation and "
        "optimization of biologically effective dose, Phys. Med. Biol. 45 (2000) 3319-3330, "
        "Table 1 (p. 3322), https://doi.org/10.1088/0031-9155/45/11/314"
    ),
    short="Krämer 2000",
    note=(
        "CHO · LEM I: alpha_X = 0.228 Gy⁻¹, beta_X = 0.02 Gy⁻², D_cut = 30 Gy (the threshold "
        "dose D_t), s_max = 1.34 Gy⁻¹, R_nucl = 5.5 µm (Table 1, p. 3322), attributed to "
        "Kraft-Weyrather et al. (1999); the original LEM of the GSI treatment planning "
        "(Scholz and Kraft 1996, Scholz et al. 1997). Elsässer and Scholz 2007 fit D_t = "
        "40 Gy with r_nucleus = 4.7 µm instead, see the preset 'Elsässer 2007'."
    ),
)
for _version, _d_t, _r_nucleus, _source in (
    ("I", 40, 4.7, _CHO_LEM_I_II),
    ("I", 30, 5.5, _CHO_LEM_I_KRAMER_2000),
    ("II", 9.5, 4.7, _CHO_LEM_I_II),
    ("III", 55, 4.7, _CHO_LEM_III),
):
    _register_builtin(
        "CHO",
        f"LEM_{_version}",
        _source,
        alpha_x=0.228,
        beta_x=0.02,
        d_t=_d_t,
        r_nucleus=_r_nucleus,
    )

# -------------------------------------------------------------------------------- T1
register_cell_line_preset(
    "T1",
    "MKM",
    source=_kase_2008(
        "Table 1 (p. 45)",
        "T1 · MKM: alpha_0 = 0.0305 Gy⁻¹, beta = 0.0585 Gy⁻², R_d = 0.35 µm, R_n = 3.5 µm, "
        "fitted to the D10 of aerobic T1 cells for 3He, 12C and 20Ne ions of Furusawa et al., "
        "Radiat. Res. 154 (2000) 485; alpha_0 / beta are the X-ray alpha / beta of aerobic T1 "
        "cells of its Table 1 (p. 487, 200 kVp X-rays).",
        "Furusawa et al. 2000, Table 1",
        alpha_x=0.133,
    ),
    reference=FURUSAWA_2000,
    validate=False,
    r_nucleus=3.5,
    r_domain=0.35,
    alpha_0=0.0305,
    beta_0=0.0585,
    alpha_x=0.0305,
    beta_x=0.0585,
    reference_radiation=DEFAULT_REFERENCE_RADIATION,
)

# ------------------------------------------------------------------------------- V79
register_cell_line_preset(
    "V79",
    "MKM",
    source=_kase_2008(
        "Table 1 (p. 45)",
        "V79 · MKM: alpha_0 = 0.184 Gy⁻¹, beta = 0.02 Gy⁻², R_d = 0.26 µm, R_n = 4.1 µm, "
        "fitted to the D10 of aerobic V79 cells for 3He, 12C and 20Ne ions of Furusawa et al., "
        "Radiat. Res. 154 (2000) 485; alpha_0 / beta are the X-ray alpha / beta of aerobic "
        "V79 cells of its Table 1 (p. 487, 200 kVp X-rays).",
        "Furusawa et al. 2000, Table 1",
        alpha_x=0.247,
    ),
    reference=FURUSAWA_2000,
    validate=False,
    r_nucleus=4.1,
    r_domain=0.26,
    alpha_0=0.184,
    beta_0=0.02,
    alpha_x=0.184,
    beta_x=0.02,
    reference_radiation=DEFAULT_REFERENCE_RADIATION,
)
_V79_SECOND_SET = (
    "Manganaro et al. 2018, Table 1, have a second V79 · LEM row, alpha_X = 0.093 Gy⁻¹, "
    "beta_X = 0.026 Gy⁻², R_N = 3.8 µm, D_t = 30 / 7.5 / 30 Gy (LEM I and II as in "
    f"{_ELSASSER_2007}, Fig. 8 and p. 323), see the presets 'Elsässer 2007' and "
    "'Manganaro 2018, 2nd V79 set'."
)
_V79_LEM_II = ParameterSource(
    citation=(
        "Kase et al., Biophysical calculation of cell survival probabilities using amorphous "
        "track structure models for heavy-ion irradiation, Phys. Med. Biol. 53 (2008) 37-59, "
        "Sect. 3.4 (pp. 52-53) and Table 1 (p. 45), "
        "https://doi.org/10.1088/0031-9155/53/1/003; also "
        f"{MANGANARO_2018.citation}"
    ),
    short="Kase 2008",
    note=(
        "V79 · LEM II: the modified LEM uses the photon LQ parameters of Table 1, alpha_X = "
        "0.184 Gy⁻¹, beta_X = 0.02 Gy⁻², with D_t = 5.5 Gy and R_n = 4 µm (pp. 52-53), for the "
        "RBE10 of V79 cells after H and He ions (Fig. 14). alpha_X / beta_X are the X-ray LQ "
        f"parameters of aerobic V79 cells of {_FURUSAWA_2000} ({_FURUSAWA_XRAYS}). Kase et "
        "al. note that Elsässer and Scholz (2006) found D_t = 15 Gy best for the carbon data "
        "of Furusawa et al. Manganaro et al. 2018, Table 1, list the set as the first V79 · "
        f"LEM row. {_V79_SECOND_SET}"
    ),
)
_V79_LEM_I_III = _manganaro_2018(
    "V79 · LEM I / III: alpha_X, beta_X and R_N as for LEM II (Kase et al., Phys. Med. Biol. "
    "53 (2008) 37, pp. 52-53; alpha_X / beta_X from Furusawa et al. 2000, Table 1). The "
    "threshold doses of 60 Gy for LEM I and LEM III are given by Manganaro et al. only; no "
    "original publication of them was found (searched: Scholz and Kraft 1994 and 1996, "
    "Scholz et al. 1997, Krämer and Scholz 2000 and 2006, Elsässer and Scholz 2007, "
    "Elsässer et al. 2008 (both), Kase et al. 2008). R_N = 4 µm matches the effective "
    "nuclear cross section of 50 µm² of V79 cells of Scholz and Kraft, Radiat. Prot. Dosim. "
    "52 (1994) 29 (p. 32), used there with other X-ray parameters. 60 Gy is also the LEM "
    "III threshold dose of the rat spinal cord (alpha = 0.1 Gy⁻¹, beta = 0.05 Gy⁻², R = "
    "5 µm) of Elsässer, Krämer and Scholz, Int. J. Radiat. Oncol. Biol. Phys. 71 (2008) "
    f"866, Fig. 1 (p. 870). {_V79_SECOND_SET}"
)
for _version, _d_t, _source in (
    ("I", 60, _V79_LEM_I_III),
    ("II", 5.5, _V79_LEM_II),
    ("III", 60, _V79_LEM_I_III),
):
    _register_builtin(
        "V79",
        f"LEM_{_version}",
        _source,
        alpha_x=0.184,
        beta_x=0.02,
        d_t=_d_t,
        r_nucleus=4.0,
    )

# The second V79 set, of the cluster extension paper of the LEM
_V79_ELSASSER_2007 = _elsasser_2007(
    "Figs. 6 and 8 (pp. 324-325) and p. 323",
    "V79 · LEM I / II: alpha_X = 0.093 Gy⁻¹, beta_X = 0.026 Gy⁻², D_t = 30 Gy for the "
    "original LEM (LEM I) and 7.5 Gy for the cluster extension (LEM II) (Figs. 6 and 8), "
    "r_nucleus = 3.8 µm (p. 323), with D_t fitted to the initial RBE at 32 and 153 keV/µm "
    "(p. 325); the experimental data are of Weyrather et al. (1999). Manganaro et al. "
    "2018, Table 1, list the set as the second V79 · LEM row, with D_t = 30 Gy for LEM III "
    "(see the preset 'Manganaro 2018, 2nd V79 set'). The first V79 · LEM row (alpha_X = "
    "0.184 Gy⁻¹, beta_X = 0.02 Gy⁻², R_N = 4 µm) is the default preset.",
)
_V79_LEM_III_SECOND = ParameterSource(
    citation=MANGANARO_2018.citation.replace(
        "Table 1 (p. 5)", "Table 1 (p. 5), second V79 · LEM row"
    ),
    short="Manganaro 2018, 2nd V79 set",
    note=(
        "V79 · LEM III: alpha_X = 0.093 Gy⁻¹, beta_X = 0.026 Gy⁻², R_N = 3.8 µm as for LEM I / "
        f"II ({_ELSASSER_2007}, Figs. 6 and 8 and p. 323). The threshold dose of 30 Gy is "
        "given by Manganaro et al. only, in their second V79 · LEM row; no original "
        "publication of it was found."
    ),
)
for _version, _d_t, _source in (
    ("I", 30, _V79_ELSASSER_2007),
    ("II", 7.5, _V79_ELSASSER_2007),
    ("III", 30, _V79_LEM_III_SECOND),
):
    _register_builtin(
        "V79",
        f"LEM_{_version}",
        _source,
        alpha_x=0.093,
        beta_x=0.026,
        d_t=_d_t,
        r_nucleus=3.8,
    )
