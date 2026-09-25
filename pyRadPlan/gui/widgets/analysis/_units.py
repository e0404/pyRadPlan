"""Unit helpers for the analysis widgets.

The analysis viewer carries quantity units as display labels rather than pint
expressions. These helpers resolve such labels to ``pint`` units and decide
whether two quantities can share a plot axis.
"""

from __future__ import annotations

import logging

import pint

from pyRadPlan.analysis._dvh import ureg

logger = logging.getLogger(__name__)

UNIT_ALIASES: dict[str, str] = {"Gy (RBE)": "gray", "Gy½": "gray**0.5"}


def safe_unit(unit_str: str) -> pint.Unit:
    """Coerce a display unit string into a ``pint.Unit``.

    Known labels pint cannot parse (``"Gy (RBE)"``, ``"Gy½"``) are mapped
    through ``UNIT_ALIASES``. Empty or otherwise unparsable strings degrade to
    dimensionless instead of raising.

    Parameters
    ----------
    unit_str : str
        Display unit string.

    Returns
    -------
    pint.Unit
        The resolved unit, or ``ureg.dimensionless`` if it cannot be resolved.
    """
    if not unit_str:
        return ureg.dimensionless
    try:
        return ureg.Unit(UNIT_ALIASES.get(unit_str, unit_str))
    except (pint.UndefinedUnitError, TypeError, ValueError):
        logger.debug("Unit %r is not pint-parseable; falling back to dimensionless", unit_str)
        return ureg.dimensionless


def compare_units(primary: str, secondary: str) -> tuple[bool, float]:
    """Decide whether a secondary quantity can share the primary quantity's axis.

    Identical unit strings always share an axis. Otherwise both strings are
    resolved with :func:`safe_unit`; a dimensionless (or unresolvable) unit on
    either side prevents sharing, equal units share without rescaling, and
    units of the same dimensionality share after conversion.

    Parameters
    ----------
    primary : str
        Display unit string of the primary quantity.
    secondary : str
        Display unit string of the secondary quantity.

    Returns
    -------
    tuple[bool, float]
        ``(shared_axis, scale)`` where ``scale`` multiplies the secondary
        quantity's values to express them in the primary unit. ``scale`` is
        ``1.0`` whenever no conversion applies.
    """
    if primary == secondary:
        return True, 1.0
    primary_unit = safe_unit(primary)
    secondary_unit = safe_unit(secondary)
    if ureg.dimensionless in (primary_unit, secondary_unit):
        return False, 1.0
    if primary_unit == secondary_unit:
        return True, 1.0
    if primary_unit.dimensionality == secondary_unit.dimensionality:
        return True, float((1 * secondary_unit).to(primary_unit).magnitude)
    return False, 1.0
