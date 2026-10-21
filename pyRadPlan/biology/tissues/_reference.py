"""
Reference radiations and the MKM conversion to reference LQ parameters.

The RBE of a biological model is taken relative to a reference radiation (photons or
electrons), whose LQ parameters `alpha_x` / `beta_x` every tissue carries. The
Microdosimetric Kinetic Model does not take them as input: it describes a tissue by `alpha_0`,
`beta_0` and its domain geometry, and the response to any radiation, the reference included,
follows from the dose-mean specific energy z_1D that radiation deposits in a domain:
``alpha = alpha_0 + beta_0 * z_1D``, ``beta = beta_0`` (Hawkins, Radiat. Res. 140 (1994) 366,
Eq. 26; Kase et al., Radiat. Res. 166 (2006) 629, Eq. 2). Evaluated for the reference radiation
this gives `alpha_x` and `beta_x`. The modified MKM (mMKM) uses the saturation-corrected z*_1D
instead (Kase et al. 2006, Eqs. 7-9; Inaniwa et al., Phys. Med. Biol. 55 (2010) 6721,
Eqs. 2-5). A reference radiation is described by its dose-mean lineal energy y_D (keV/µm),
from which z_1D follows for any domain radius, or by z_1D itself for one known domain size.

y_D depends on the site it is measured (or simulated) in: for 200 kV X-rays, 4.2 keV/µm in a
site of 0.97 µm diameter against 3.3 keV/µm in one of 2.06 µm (Coppola et al. 1976, quoted by
Okamoto et al., J. Radiat. Res. 52 (2011) 75). The MKM literature uses y_D measured or
simulated for a site of about 1 µm diameter whatever the domain radius it converts it for
(Kase et al. 2006, pp. 630 and 633: a TEPC simulating 1.0 µm for domains of 0.5-1.0 µm
diameter; Kase et al. 2011; Okamoto et al. 2011), and so does z_1D = y_D / (rho pi r_d**2)
here; `ReferenceRadiation.site_diameter` records the site a value refers to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

from ._sources import UNVERIFIED_SOURCE, ParameterSource

#: Specific energy in Gy of 1 keV deposited in 1 µm³ of unit density (water):
#: 1.602e-16 J / 1e-15 kg.
GY_PER_KEV_PER_UM3: float = 0.1602


def specific_energy_from_lineal_energy(y: float, r_domain: float) -> float:
    """
    Dose-mean specific energy z_1D [Gy] in a domain, from the dose-mean lineal energy.

    ``z_1D = y_D * l / m = y_D / (rho * pi * r_domain**2)`` (Kase et al., Radiat. Res. 166
    (2006) 629, Eq. 3), for a spherical domain of radius `r_domain` [µm] (mean chord length
    l = 4/3 r_domain, mass m) of unit density, with `y` in keV/µm.

    Parameters
    ----------
    y : float
        Dose-mean lineal energy y_D [keV/µm].
    r_domain : float
        Domain radius [µm], positive.

    Returns
    -------
    float
        z_1D [Gy].
    """
    return GY_PER_KEV_PER_UM3 * y / (math.pi * r_domain**2)


def saturation_specific_energy(r_domain: float, r_nucleus: float, beta_0: float) -> float:
    """
    Saturation parameter z_0 [Gy] of the modified MKM.

    ``z_0 = r_nucleus**2 / (r_domain * sqrt(beta_0 * (r_domain**2 + r_nucleus**2)))``
    (Inaniwa et al., Phys. Med. Biol. 55 (2010) 6721, Eq. 5, written there as
    ``(R_n/r_d)**2 / sqrt(beta (1 + (R_n/r_d)**2))``; equivalently ``z_0 = y_0 / (rho pi
    r_domain**2)`` with ``y_0 = rho pi r_domain r_nucleus**2 / sqrt(beta_0 (r_domain**2 +
    r_nucleus**2))`` of Kase et al., Radiat. Res. 166 (2006) 629, Eq. 9, which gives their
    y_0 = 150 keV/µm for HSG). Infinite for ``beta_0 == 0``, i.e. no saturation.

    Parameters
    ----------
    r_domain, r_nucleus : float
        Domain and nucleus radius [µm], positive.
    beta_0 : float
        Quadratic coefficient beta_0 [Gy⁻²], non-negative.
    """
    if beta_0 == 0:
        return math.inf
    return r_nucleus**2 / (r_domain * math.sqrt(beta_0 * (r_domain**2 + r_nucleus**2)))


def saturation_corrected_specific_energy(z: float, z_0: float) -> float:
    """
    Saturation-corrected specific energy z* [Gy] of one specific energy z.

    ``z* = z_0**2 * (1 - exp(-z**2 / z_0**2)) / z`` (Inaniwa et al. 2010, Eq. 4; in lineal
    energy, Kase et al. 2006, Eq. 8). The mMKM averages this over the single-event spectrum
    of the radiation (Inaniwa et al. 2010, Eq. 3); for a reference radiation known only by its
    dose-mean value the spectrum is taken as concentrated there. Both agree to first order in
    ``(z / z_0)**2``, which for photons and electrons (z_1D of a few Gy against a z_0 of
    tens of Gy) is far below a percent.

    Parameters
    ----------
    z : float
        Specific energy [Gy], non-negative.
    z_0 : float
        Saturation parameter [Gy], see :func:`saturation_specific_energy`.
    """
    if z == 0 or math.isinf(z_0):
        return z
    return z_0**2 * -math.expm1(-((z / z_0) ** 2)) / z


@dataclass(frozen=True)
class ReferenceRadiation:
    """
    A reference radiation the RBE is taken relative to, for the MKM conversion.

    Exactly one of `y_d` and `z_1d` describes it.

    Attributes
    ----------
    name : str
        Identifier stored with a tissue (`MKMTissueProperties.reference_radiation`).
    label : str
        Display name, e.g. ``"200 kV X-rays"``.
    y_d : float, optional
        Dose-mean lineal energy [keV/µm]; z_1D then follows from the domain radius
        (:func:`specific_energy_from_lineal_energy`).
    z_1d : float, optional
        Dose-mean specific energy per domain [Gy], used as it is whatever the domain
        radius; only meaningful for the domain size it was determined for.
    site_diameter : float, optional
        Diameter [µm] of the (simulated) site `y_d` was measured or computed for, e.g.
        ``1.0`` for a TEPC simulating a 1 µm tissue sphere. Descriptive only: y_D falls with
        the site size, and the conversion uses `y_d` as it is.
    source : ParameterSource
        Where the values come from (publication, measurement, ...); a string is taken as
        the citation (:meth:`ParameterSource.coerce`), ``"Unverified"`` if not given.
    """

    name: str
    label: str
    y_d: Optional[float] = None
    z_1d: Optional[float] = None
    source: ParameterSource = ParameterSource()
    site_diameter: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", ParameterSource.coerce(self.source))
        if (self.y_d is None) == (self.z_1d is None):
            raise ValueError("A reference radiation needs exactly one of y_d and z_1d.")
        value = self.y_d if self.y_d is not None else self.z_1d
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(f"Reference radiation {self.name!r}: invalid value {value!r}.")

    def specific_energy(self, r_domain: float) -> float:
        """
        Dose-mean specific energy z_1D [Gy] in a domain of radius `r_domain` [µm].

        NaN if `y_d` describes the radiation and `r_domain` is not a positive number.
        """
        if self.z_1d is not None:
            return self.z_1d
        if not (math.isfinite(r_domain) and r_domain > 0):
            return math.nan
        return specific_energy_from_lineal_energy(self.y_d, r_domain)

    def describe(self) -> str:
        """Short description, e.g. ``"200 kV X-rays (y_D = 4.2 keV/µm)"``."""
        if self.y_d is not None:
            return f"{self.label} (y_D = {self.y_d:g} keV/µm)"
        return f"{self.label} (z_1D = {self.z_1d:g} Gy)"

    def details(self) -> str:
        """Value with the site it refers to, e.g. ``"y_D = 4.2 keV/µm in a 0.97 µm site"``."""
        if self.y_d is None:
            return f"z_1D = {self.z_1d:g} Gy"
        site = f" in a {self.site_diameter:g} µm site" if self.site_diameter else ""
        return f"y_D = {self.y_d:g} keV/µm{site}"


_REFERENCE_RADIATIONS: dict[str, ReferenceRadiation] = {}

#: Name of the reference radiation an MKM tissue refers to unless it names another.
DEFAULT_REFERENCE_RADIATION = "xray_200kV"


def register_reference_radiation(
    name: str,
    label: Optional[str] = None,
    *,
    y_d: Optional[float] = None,
    z_1d: Optional[float] = None,
    source: Union[str, ParameterSource, None] = UNVERIFIED_SOURCE,
    site_diameter: Optional[float] = None,
) -> ReferenceRadiation:
    """
    Register a reference radiation, replacing one of the same name.

    Parameters
    ----------
    name : str
        Identifier, e.g. ``"co60"``.
    label : str, optional
        Display name; defaults to `name`.
    y_d : float, optional
        Dose-mean lineal energy [keV/µm].
    z_1d : float, optional
        Dose-mean specific energy per domain [Gy], instead of `y_d`.
    source : str or ParameterSource
        Where the values come from; ``"Unverified"`` if not given.
    site_diameter : float, optional
        Diameter [µm] of the site `y_d` refers to, see :class:`ReferenceRadiation`.

    Returns
    -------
    ReferenceRadiation
        The registered radiation.
    """
    radiation = ReferenceRadiation(
        name=name,
        label=label or name,
        y_d=y_d,
        z_1d=z_1d,
        source=source,
        site_diameter=site_diameter,
    )
    _REFERENCE_RADIATIONS[name] = radiation
    return radiation


def get_reference_radiation(name: Optional[str] = None) -> ReferenceRadiation:
    """
    Return a registered reference radiation, the default one for None.

    Raises
    ------
    KeyError
        If no reference radiation of that name is registered.
    """
    key = DEFAULT_REFERENCE_RADIATION if name is None else name
    try:
        return _REFERENCE_RADIATIONS[key]
    except KeyError:
        raise KeyError(
            f"Unknown reference radiation {key!r}. Registered: {list(_REFERENCE_RADIATIONS)}"
        ) from None


def registered_reference_radiations() -> tuple[ReferenceRadiation, ...]:
    """Every registered reference radiation, in registration order."""
    return tuple(_REFERENCE_RADIATIONS.values())


OKAMOTO_2011 = ParameterSource(
    citation=(
        "Okamoto et al., Relation between lineal energy distribution and relative biological "
        "effectiveness for photon beams according to the microdosimetric kinetic model, "
        "J. Radiat. Res. 52 (2011) 75-81, Table 1 (p. 79), https://doi.org/10.1269/jrr.10073"
    ),
    short="Okamoto 2011",
)

register_reference_radiation(
    DEFAULT_REFERENCE_RADIATION,
    "200 kV X-rays",
    y_d=4.2,
    site_diameter=0.97,
    source=ParameterSource(
        citation=(
            "Coppola et al., Experimental evaluation of the spectral energy deposition in "
            "small volumes by low LET radiations, Proc. 5th Symp. on Microdosimetry, EUR 5452 "
            "(1976) 377; quoted as the published y_D of 200 kV X-rays by Okamoto et al., "
            "J. Radiat. Res. 52 (2011) 75-81, Table 1 (p. 79) and Discussion (p. 80), "
            "https://doi.org/10.1269/jrr.10073"
        ),
        short="Coppola 1976",
        note=(
            "Measured with a walled spherical proportional counter simulating a 0.97 µm site; "
            "the same work gives 3.3 keV/µm for a 2.06 µm site. Read in Okamoto et al. 2011; "
            "the proceedings themselves were not accessible. Kase et al., Radiat. Res. 166 "
            "(2006) 629, p. 633, take y_D = 4.3 ± 0.2 keV/µm, measured by Coppola et al. with "
            "a TEPC at 0.76 and 0.97 µm diameter, for their 200 kVp X-rays (0.5 mm Al + "
            "0.5 mm Cu filter), from which they derive their HSG alpha_0 and r_d."
        ),
    ),
)
register_reference_radiation(
    "xray_200kV_tepc",
    "200 kV X-rays (TEPC)",
    y_d=4.51,
    site_diameter=1.0,
    source=ParameterSource(
        citation=OKAMOTO_2011.citation,
        short=OKAMOTO_2011.short,
        note=(
            "TEPC measurement in a 1 µm water-equivalent site of a 200 kV X-ray unit (YXLON "
            "MG226/4.5) behind 0.5 mm Cu + 0.5 mm Al (HVL 11.9 mm Al), 4.51 ± 0.05 keV/µm; "
            "GEANT4 gives 4.41 ± 0.03 keV/µm (Table 1; set-up p. 76). The reference of the "
            "HSG · MKM preset of the same paper. Not the X-rays of Furusawa et al., Radiat. "
            "Res. 154 (2000) 485 (Shimadzu Shinai-7, 200 kVp, 0.5 mm Al + 0.5 mm Cu)."
        ),
    ),
)
register_reference_radiation(
    "co60",
    "Co-60 γ-rays",
    y_d=2.34,
    site_diameter=1.0,
    source=ParameterSource(
        citation=OKAMOTO_2011.citation,
        short=OKAMOTO_2011.short,
        note=(
            "TEPC measurement in a 1 µm site, 2.34 ± 0.03 keV/µm (GEANT4: 2.24 ± 0.01), "
            "equal to the published 2.34 keV/µm for a 0.95 µm site (Lindborg, Proc. 5th "
            "Symp. on Microdosimetry, EUR 5452 (1976) 347)."
        ),
    ),
)
register_reference_radiation(
    "xray_6MV",
    "6 MV X-rays",
    y_d=2.36,
    site_diameter=1.0,
    source=ParameterSource(
        citation=OKAMOTO_2011.citation,
        short=OKAMOTO_2011.short,
        note=(
            "TEPC measurement in a 1 µm site at 100 mm depth in water, 2.36 ± 0.04 keV/µm "
            "(GEANT4: 2.27 ± 0.08)."
        ),
    ),
)
