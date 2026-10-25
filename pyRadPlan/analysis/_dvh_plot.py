"""Backend-neutral description of a DVH plot.

:func:`build_dvh_plot` decides what a DVH plot shows -- curves, colors, line
styles, axis labels, whether a compared quantity shares the x axis, the legend
and the D_x / V_x points of the quality indicators -- without drawing anything, so that different renderers (the Qt GUI,
matplotlib) show the same plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import numpy as np
import pint

from ._dvh import DVH
from ._qi import format_unit_symbol
from ._units import compare_units, safe_unit

if TYPE_CHECKING:
    from ._qi import QICollection

LineStyle = Literal["solid", "dashed"]
RGB = tuple[float, float, float]

#: Color of structures without an assigned color.
DEFAULT_CURVE_COLOR: RGB = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class DVHCurve:
    """One DVH curve, colored in ``[0, 1]`` RGB."""

    name: str
    x: np.ndarray
    y: np.ndarray
    color: RGB
    line_style: LineStyle
    #: Whether the curve belongs to the secondary x axis.
    secondary_axis: bool = False


@dataclass(frozen=True)
class LegendEntry:
    """One legend row; ``color=None`` means the renderer's text color."""

    label: str
    color: Optional[RGB]
    line_style: LineStyle


@dataclass(frozen=True)
class QIMarker:
    """A D_x or V_x quality indicator as a point on its structure's primary curve.

    For ``"dx"`` the parameter is the reference volume in % and the value the dose
    covering it; for ``"vx"`` the parameter is the reference dose (in the unit of the
    x axis) and the value the volume in % receiving it.
    """

    structure: str
    metric: str
    kind: Literal["dx", "vx"]
    parameter: float
    value: float
    color: RGB
    #: Unit symbols of the value and the parameter, ``"%"`` for volumes.
    value_unit: str
    parameter_unit: str

    @property
    def value_text(self) -> str:
        """The value with its unit, e.g. ``"1.52 Gy"`` or ``"93.2 %"``."""
        return format_marker_value(self.value, self.value_unit)

    def parameter_text(self, parameter: Optional[float] = None) -> str:
        """Return the metric name for *parameter* (default: its own), e.g. ``"D98"``."""
        return format_marker_parameter(
            self.kind, self.parameter if parameter is None else parameter, self.parameter_unit
        )

    @property
    def x(self) -> float:
        """Position on the dose (x) axis."""
        return self.value if self.kind == "dx" else self.parameter

    @property
    def y(self) -> float:
        """Position on the volume (y) axis."""
        return self.parameter if self.kind == "dx" else self.value


@dataclass(frozen=True)
class DVHPlot:
    """Everything a renderer needs to draw a DVH plot."""

    curves: tuple[DVHCurve, ...]
    x_label: str
    y_label: str
    #: Label of the secondary x axis, ``None`` when all curves share the primary one.
    secondary_x_label: Optional[str]
    legend: tuple[LegendEntry, ...]
    markers: tuple[QIMarker, ...] = ()

    def primary_curve(self, structure: str) -> Optional[DVHCurve]:
        """Return the solid curve of *structure*, or ``None`` if it is not plotted."""
        for curve in self.curves:
            if curve.name == structure and curve.line_style == "solid":
                return curve
        return None


def _axis_label(label: str, unit: str) -> str:
    name = label or "Dose"
    return f"{name} [{unit}]" if unit else name


def _unit_symbol(unit: pint.Unit) -> str:
    if str(unit) == "percent":
        return "%"
    return "" if unit.dimensionless else format_unit_symbol(unit)


def format_marker_value(value: float, unit: str) -> str:
    """Format a D_x / V_x value, e.g. ``"1.52 Gy"`` or ``"93.2 %"``."""
    text = f"{value:.1f}" if unit == "%" else f"{value:.2f}"
    return f"{text} {unit}" if unit else text


def format_marker_parameter(kind: str, parameter: float, unit: str) -> str:
    """Format a D_x / V_x metric name like its metric id, e.g. ``"D98"`` or ``"V20Gy"``."""
    if kind == "dx":
        return f"D{round(parameter, 1):g}"
    return f"V{round(parameter, 2):g}{unit}"


def _qi_markers(
    qis: QICollection, structures: Sequence[str], colors: dict[str, RGB], unit: str
) -> list[QIMarker]:
    """D_x / V_x points of *qis* for the plotted *structures*, in the x-axis unit."""
    axis_unit = safe_unit(unit)
    markers = []
    for name in structures:
        if name not in qis:
            continue
        for metric, qi in qis[name].metrics.items():
            if qi.qi_type not in ("dx", "vx") or not np.isfinite(qi.value):
                continue
            if qi.qi_type == "dx":
                parameter = qi.ref_vol
            else:
                try:
                    parameter = float((qi.ref_dose * qi.ref_unit).to(axis_unit).magnitude)
                except pint.DimensionalityError:
                    parameter = qi.ref_dose
            markers.append(
                QIMarker(
                    name,
                    metric,
                    qi.qi_type,
                    parameter,
                    qi.value,
                    colors[name],
                    value_unit=_unit_symbol(qi.unit),
                    parameter_unit="%" if qi.qi_type == "dx" else _unit_symbol(axis_unit),
                )
            )
    return markers


def build_dvh_plot(
    dvhs: Sequence[DVH],
    voi_colors: Optional[dict[str, tuple[int, int, int]]] = None,
    unit: str = "",
    label: str = "",
    secondary_dvhs: Optional[Sequence[DVH]] = None,
    secondary_unit: str = "",
    secondary_label: str = "",
    primary_name: str = "",
    secondary_name: str = "",
    qis: Optional[QICollection] = None,
) -> DVHPlot:
    """Describe a DVH plot of primary curves and optional dashed secondary curves.

    Secondary curves use the same color per structure as the primary ones. If the
    secondary unit is compatible with the primary unit (see
    :func:`~pyRadPlan.analysis._units.compare_units`), they share the primary x axis
    and are rescaled to the primary unit; otherwise they get a secondary x axis.

    The legend lists the plotted structures in plot order. With secondary curves, two
    rows in the text color follow that identify the solid and dashed line styles.

    Parameters
    ----------
    dvhs : Sequence[DVH]
        Primary DVHs, drawn solid.
    voi_colors : dict[str, tuple[int, int, int]] | None, optional
        RGB colors (0-255) keyed by structure name; others use
        :data:`DEFAULT_CURVE_COLOR`.
    unit, label : str, optional
        Unit and name of the primary quantity for the x-axis label; the name
        defaults to "Dose".
    secondary_dvhs : Sequence[DVH] | None, optional
        DVHs of a compared quantity, drawn dashed.
    secondary_unit, secondary_label : str, optional
        Unit and name of the compared quantity; the name is only used for a
        secondary x axis.
    primary_name, secondary_name : str, optional
        Legend labels of the solid and dashed line styles; default to "Primary"
        and "Secondary".
    qis : QICollection | None, optional
        Quality indicators of the primary quantity; their D_x / V_x metrics of
        the plotted structures become :attr:`DVHPlot.markers`.

    Returns
    -------
    DVHPlot
        The plot description.
    """

    def _color(name: str) -> RGB:
        if voi_colors and name in voi_colors:
            r, g, b = voi_colors[name]
            return (r / 255, g / 255, b / 255)
        return DEFAULT_CURVE_COLOR

    curves = [
        DVHCurve(
            dvh.name, np.asarray(dvh.bins), np.asarray(dvh.cum_volume), _color(dvh.name), "solid"
        )
        for dvh in dvhs
    ]
    secondary_x_label = None
    if secondary_dvhs:
        shared, scale = compare_units(unit, secondary_unit)
        if not shared:
            secondary_x_label = _axis_label(secondary_label, secondary_unit)
        curves += [
            DVHCurve(
                dvh.name,
                np.asarray(dvh.bins) * scale if shared else np.asarray(dvh.bins),
                np.asarray(dvh.cum_volume),
                _color(dvh.name),
                "dashed",
                secondary_axis=not shared,
            )
            for dvh in secondary_dvhs
        ]

    names = list(dict.fromkeys(curve.name for curve in curves))
    legend = [LegendEntry(name, _color(name), "solid") for name in names]
    if secondary_dvhs and names:
        legend += [
            LegendEntry(primary_name or "Primary", None, "solid"),
            LegendEntry(secondary_name or "Secondary", None, "dashed"),
        ]

    return DVHPlot(
        curves=tuple(curves),
        x_label=_axis_label(label, unit),
        y_label="Volume [%]",
        secondary_x_label=secondary_x_label,
        legend=tuple(legend),
        markers=tuple(
            _qi_markers(qis, [dvh.name for dvh in dvhs], {n: _color(n) for n in names}, unit)
            if qis is not None
            else ()
        ),
    )
