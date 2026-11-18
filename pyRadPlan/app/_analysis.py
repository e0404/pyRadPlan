"""DVH sets, quality indicators and unit comparison of the analysis window (protocol v1).

Mirrors ``gui/widgets/_analysis_widget.py``: DVHs per structure (``DVH.compute``), the QI
table (``QICollection.from_masks`` with the Qt parameter fallbacks) and the shared-axis
decision for a compared quantity (``pyRadPlan.analysis._units.compare_units``). All
arrays come from the session's :class:`~pyRadPlan.app._views.ViewCache`, so they are the
CT-grid volumes the viewer shows.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ._views import component_info, result_entry, unit_info


def _require_entry(session: Any, key: str) -> None:
    session.views._require_ct()
    if result_entry(session.workspace.result, key) is None:
        raise KeyError(f"No volume quantity {key!r} in the result")


def dvh_sets(
    session: Any,
    keys: Sequence[str],
    names: Optional[Sequence[str]] = None,
    num_points: int = 500,
    range_mode: str = "shared",
) -> dict[str, Any]:
    """One DVH set per component key; structures without voxels are omitted."""
    if isinstance(keys, str) or not keys:
        raise TypeError("keys must be a non-empty list of component keys")
    result = session.workspace.result
    for key in keys:
        _require_entry(session, key)
    vois = session.views._vois(names)
    sets = []
    for key in keys:
        info = component_info(result, key)
        curves = session.views.dvh_curves(key, vois, num_points, range_mode)
        sets.append(
            {
                "key": key,
                "label": info["meta_label"],
                "name": info["label"],
                "unit": info["unit"],
                "unit_info": unit_info(info["unit"]),
                "curves": [
                    {"name": voi.name, "color": list(voi.color), **curve}
                    for voi, curve in zip(vois, curves)
                    if curve["stats"]["mean"] is not None
                ],
            }
        )
    return {"sets": sets}


def _valid_list(values: Any, check) -> Optional[list[float]]:
    """Return the numbers of *values* if all are finite and pass *check*, else ``None``."""
    if values is None or isinstance(values, (str, bytes)):
        return None
    try:
        numbers = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if not numbers or not all(np.isfinite(v) and check(v) for v in numbers):
        return None
    return numbers


def _value_unit(qi: Any) -> str:
    from pyRadPlan.analysis import format_unit_symbol  # noqa: PLC0415

    if str(qi.unit) == "percent":
        return "%"
    if qi.unit.dimensionless:
        return ""
    return format_unit_symbol(qi.unit)


def quality_indicators(
    session: Any,
    quantity: str,
    names: Optional[Sequence[str]] = None,
    ref_vols: Optional[Sequence[float]] = None,
    ref_doses: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """Compute the Qt QI table of *quantity* for the structures (all without *names*).

    Invalid or empty ``ref_vols`` (finite, 0..100 %) and ``ref_doses`` (finite, >= 0)
    fall back to :data:`~pyRadPlan.analysis.DEFAULT_REF_VOLS` and
    :meth:`~pyRadPlan.analysis.QICollection.default_ref_doses` over the whole quantity,
    as ``_analysis_widget.py:_qi_params`` / ``_update_qi_table`` do. The values used are
    returned.
    """
    from pyRadPlan.analysis import (  # noqa: PLC0415
        DEFAULT_REF_VOLS,
        QICollection,
        format_metric_label,
    )
    from pyRadPlan.analysis._units import safe_unit  # noqa: PLC0415

    _require_entry(session, quantity)
    info = component_info(session.workspace.result, quantity)
    vois = session.views._vois(names)
    values = session.views.volume_array(quantity)
    used_vols = _valid_list(ref_vols, lambda v: 0.0 <= v <= 100.0) or list(DEFAULT_REF_VOLS)
    used_doses = _valid_list(ref_doses, lambda v: v >= 0.0)
    if used_doses is None:
        used_doses = QICollection.default_ref_doses(values)
    masks = {voi.name: session.views.mask_array(voi) for voi in vois}
    qis = QICollection.from_masks(
        masks,
        values,
        ref_vols=used_vols,
        ref_doses=used_doses,
        dose_unit=safe_unit(info["unit"]),
    )
    metric_ids = qis.metric_ids([voi.name for voi in vois])
    metrics = []
    for metric in metric_ids:
        qi = next(s.metrics[metric] for s in qis if metric in s.metrics)
        entry = {
            "id": metric,
            "label": format_metric_label(metric, qi),
            "kind": qi.qi_type,
            "unit": _value_unit(qi),
        }
        if qi.qi_type == "dx":
            entry["parameter"] = float(qi.ref_vol)
        elif qi.qi_type == "vx":
            entry["parameter"] = float(qi.ref_dose)
        metrics.append(entry)
    rows = []
    for voi in vois:
        structure = qis[voi.name]
        row_values = {}
        for metric in metric_ids:
            qi = structure.metrics.get(metric)
            row_values[metric] = (
                None if qi is None or not np.isfinite(qi.value) else float(qi.value)
            )
        rows.append({"name": voi.name, "color": list(voi.color), "values": row_values})
    return {
        "quantity": quantity,
        "label": info["meta_label"],
        "name": info["label"],
        "unit": info["unit"],
        "unit_symbol": unit_info(info["unit"])["symbol"],
        "ref_vols": used_vols,
        "ref_doses": [float(v) for v in used_doses],
        "metrics": metrics,
        "rows": rows,
    }


def compare_quantity_units(session: Any, a: str, b: str) -> dict[str, Any]:
    """Whether *b* can share *a*'s DVH axis, and the factor taking *b* into *a*'s unit."""
    from pyRadPlan.analysis._units import compare_units  # noqa: PLC0415

    result = session.workspace.result
    for key in (a, b):
        _require_entry(session, key)
    shareable, scale = compare_units(
        component_info(result, a)["unit"], component_info(result, b)["unit"]
    )
    return {"shareable": bool(shareable), "scale": float(scale) if shareable else None}
