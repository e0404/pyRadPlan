"""Format-neutral helpers shared across import/export backends."""

import colorsys
import re
from typing import Optional

# Keyword heuristics mapping structure names to pyRadPlan VOI types.
_TARGET_KEYWORDS = ("ptv", "ctv", "gtv", "itv", "tumor", "tumour", "target", "boost")
_EXTERNAL_KEYWORDS = ("body", "external", "skin", "outline", "patient", "contour_ext")

#: Defined Terms of RT ROI Interpreted Type (3006,00A4), DICOM PS3.3 C.8.8.8.
RT_ROI_INTERPRETED_TYPES = frozenset(
    {
        "EXTERNAL",
        "PTV",
        "CTV",
        "GTV",
        "TREATED_VOLUME",
        "IRRAD_VOLUME",
        "OAR",
        "BOLUS",
        "AVOIDANCE",
        "ORGAN",
        "MARKER",
        "REGISTRATION",
        "ISOCENTER",
        "CONTRAST_AGENT",
        "CAVITY",
        "BRACHY_CHANNEL",
        "BRACHY_ACCESSORY",
        "BRACHY_SRC_APP",
        "BRACHY_CHNL_SHLD",
        "SUPPORT",
        "FIXATION",
        "DOSE_REGION",
        "CONTROL",
        "DOSE_MEASUREMENT",
    }
)

_TARGET_TOKEN_TYPES = {"gtv": "GTV", "ctv": "CTV", "itv": "CTV", "ptv": "PTV"}
_TARGET_TOKEN_PATTERN = re.compile(r"(?<![a-z])(gtv|ctv|itv|ptv)(?![a-z])")

_INTERPRETED_TO_VOI_TYPE = {
    "EXTERNAL": "EXTERNAL",
    "PTV": "TARGET",
    "CTV": "TARGET",
    "GTV": "TARGET",
    "ITV": "TARGET",
    "TREATED_VOLUME": "TARGET",
    "IRRAD_VOLUME": "TARGET",
    "CONTROL": "HELPER",
    "DOSE_REGION": "HELPER",
    "ORGAN": "OAR",
    "OAR": "OAR",
    "AVOIDANCE": "OAR",
    "CAVITY": "OAR",
}


def determine_structure_type(name: str) -> str:
    """
    Heuristically determine the pyRadPlan VOI type from a structure name.

    Parameters
    ----------
    name : str
        The structure / ROI name.

    Returns
    -------
    str
        One of ``"TARGET"``, ``"EXTERNAL"`` or ``"OAR"``.
    """
    lowered = str(name).lower()

    if any(keyword in lowered for keyword in _TARGET_KEYWORDS):
        return "TARGET"
    if any(keyword in lowered for keyword in _EXTERNAL_KEYWORDS):
        return "EXTERNAL"
    return "OAR"


def roi_interpreted_type(name: str, voi_type: str) -> str:
    """
    Determine the DICOM RT ROI Interpreted Type for a pyRadPlan VOI.

    Parameters
    ----------
    name : str
        The structure / ROI name; target names are matched case-insensitively
        for the tokens ``GTV``, ``CTV``, ``ITV`` and ``PTV`` (the first one in
        the name wins). The current DICOM standard has no ``ITV`` term, which is
        therefore exported as ``"CTV"``.
    voi_type : str
        The VOI type: ``"TARGET"``, ``"OAR"``, ``"HELPER"`` or ``"EXTERNAL"``.

    Returns
    -------
    str
        A Defined Term of RT ROI Interpreted Type (3006,00A4): ``"EXTERNAL"``,
        ``"GTV"``, ``"CTV"``, ``"PTV"``, ``"ORGAN"``, ``"AVOIDANCE"`` or ``"CONTROL"``.
    """
    voi_type = str(voi_type).upper()
    if voi_type == "EXTERNAL":
        return "EXTERNAL"
    if voi_type == "TARGET":
        match = _TARGET_TOKEN_PATTERN.search(str(name).lower())
        return _TARGET_TOKEN_TYPES[match.group(1)] if match else "PTV"
    if voi_type == "HELPER":
        return "CONTROL"
    return "AVOIDANCE" if "avoid" in str(name).lower() else "ORGAN"


def voi_type_from_interpreted_type(interpreted_type: Optional[str]) -> Optional[str]:
    """
    Map a DICOM RT ROI Interpreted Type to a pyRadPlan VOI type.

    Parameters
    ----------
    interpreted_type : str, optional
        Value of RT ROI Interpreted Type (3006,00A4).

    Returns
    -------
    str or None
        ``"TARGET"``, ``"OAR"``, ``"HELPER"`` or ``"EXTERNAL"``, or ``None`` if the
        type is missing, empty or has no pyRadPlan counterpart (e.g. ``MARKER``).
    """
    if not interpreted_type:
        return None
    return _INTERPRETED_TO_VOI_TYPE.get(str(interpreted_type).strip().upper())


def generate_colors(n: int) -> list[tuple[int, int, int]]:
    """
    Generate ``n`` visually distinct RGB colors (0-255).

    Parameters
    ----------
    n : int
        Number of colors to generate.

    Returns
    -------
    list[tuple[int, int, int]]
        A list of RGB tuples.
    """
    colors = []
    for i in range(max(n, 1)):
        hue = (i / max(n, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        colors.append((int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
    return colors
