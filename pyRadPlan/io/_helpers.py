"""Format-neutral helpers shared across import/export backends."""

import colorsys

# Keyword heuristics mapping structure names to pyRadPlan VOI types.
_TARGET_KEYWORDS = ("ptv", "ctv", "gtv", "itv", "tumor", "tumour", "target", "boost")
_EXTERNAL_KEYWORDS = ("body", "external", "skin", "outline", "patient", "contour_ext")


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
