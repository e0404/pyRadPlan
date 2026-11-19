"""Pure workflow rules: stage staleness, action availability, status text and load merging.

These are the rules of the Qt :class:`~pyRadPlan.gui.widgets.workflow.WorkflowWidget`,
lifted out of the widget so every frontend applies the same ones. Nothing here holds
state or imports Qt; :class:`StageTracker` is the only stateful piece and it is a plain
value object.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import Any, Iterable, Literal, Mapping, Optional

import numpy as np

StageStatus = Literal["missing", "done", "stale"]

#: Stage names of :func:`stage_status`, in workflow order.
STAGES: tuple[str, ...] = ("data", "plan", "stf", "dij", "optimization")

#: Action names of :func:`available_actions`.
ACTIONS: tuple[str, ...] = ("generate_stf", "calc_dose_influence", "optimize", "recalc")

#: Result entry naming the scenarios of the result's ``"<key>_scenario"`` lists.
SCENARIO_LABELS_KEY = "scenario_labels"
#: Suffix of the per-scenario result lists (see ``Dij.compute_result_ct_grid``).
SCENARIO_SUFFIX = "_scenario"


def tissue_signature(cst: Any) -> Optional[tuple]:
    """Per-structure tissue properties, as far as they influence the dose influence.

    Keyed by the structure's position rather than its name, so renaming a structure leaves
    the signature unchanged. ``None`` without a structure set; a structure without
    serializable tissue properties contributes ``None``.
    """
    vois = getattr(cst, "vois", None)
    if vois is None:
        return None
    return tuple(_tissue_json(voi) for voi in vois)


def mask_identity(cst: Any) -> Optional[tuple]:
    """Return the mask objects of the structures by position; ``None`` without a structure set.

    The dose engines compute on the union of the structures' voxels, so a replaced mask
    outdates the dose influence even if the tissues are unchanged. Display-only edits
    (color, visibility, name) keep the mask objects.
    """
    vois = getattr(cst, "vois", None)
    if vois is None:
        return None
    return tuple(getattr(voi, "mask", None) for voi in vois)


def _same_objects(a: Optional[tuple], b: Optional[tuple]) -> bool:
    return (
        a is not None and b is not None and len(a) == len(b) and all(x is y for x, y in zip(a, b))
    )


def _tissue_json(voi: Any) -> Optional[str]:
    dump = getattr(getattr(voi, "tissue_properties", None), "model_dump_json", None)
    if not callable(dump):
        return None
    try:
        return dump()
    except Exception:  # noqa: BLE001 - an unserializable tissue cannot be compared
        return None


@dataclass(frozen=True)
class StageTracker:
    """Staleness flags of the computed products, updated by :meth:`update`.

    A product is *stale* once an upstream object it was computed from changes. The rules
    are those of the Qt workflow widget, with ``stf`` tracked as a stage of its own
    and the CT counted as an upstream object (see :meth:`update`).
    """

    stf_stale: bool = False
    dij_stale: bool = False
    result_stale: bool = False
    tissue_signature: Optional[tuple] = None
    mask_identity: Optional[tuple] = dataclass_field(default=None, compare=False)

    def update(
        self, changed_keys: Iterable[str], present: Iterable[str], cst: Any
    ) -> StageTracker:
        """Return the flags after a workspace change.

        Parameters
        ----------
        changed_keys
            Keys written in this change. A product written in the same change as its
            upstream (e.g. a ``.mat`` carrying both ``pln`` and ``dij``) is fresh; only an
            upstream-only change marks it stale.
        present
            Keys holding a value after the change.
        cst
            The structure set after the change. Only a change of the structures' tissue
            properties or masks outdates the dose influence; objectives, colors, names and
            visibility only outdate the result.

        Notes
        -----
        Deviation from the Qt widget: a CT written without the products also outdates
        them. The widget never sees this in practice (a new CT clears the products), but
        a CT replaced on the same grid would otherwise leave a dose influence of the old
        HU values marked current.
        """
        keys = set(changed_keys)
        have = set(present)
        signature = tissue_signature(cst)
        masks = mask_identity(cst)
        tissue_changed = self.tissue_signature is not None and signature != self.tissue_signature
        masks_changed = self.mask_identity is not None and not _same_objects(
            self.mask_identity, masks
        )

        stf_stale = self.stf_stale
        if "stf" not in have or "stf" in keys:
            stf_stale = False
        elif keys & {"ct", "pln"}:
            stf_stale = True

        dij_stale = self.dij_stale
        if "dij" not in have or "dij" in keys:
            dij_stale = False
        elif keys & {"ct", "pln", "stf"} or ("cst" in keys and (tissue_changed or masks_changed)):
            dij_stale = True

        result_stale = self.result_stale
        if "result" not in have or "result" in keys:
            result_stale = False
        elif keys & {"ct", "pln", "cst", "stf", "dij"}:
            result_stale = True

        return replace(
            self,
            stf_stale=stf_stale,
            dij_stale=dij_stale,
            result_stale=result_stale,
            tissue_signature=signature,
            mask_identity=masks,
        )


def _status(done: bool, stale: bool = False) -> StageStatus:
    if not done:
        return "missing"
    return "stale" if stale else "done"


def stage_status(present: Iterable[str], tracker: StageTracker) -> dict[str, StageStatus]:
    """Status of every stage in :data:`STAGES`."""
    have = set(present)
    return {
        "data": _status({"ct", "cst"} <= have),
        "plan": _status("pln" in have),
        "stf": _status("stf" in have, tracker.stf_stale),
        "dij": _status("dij" in have, tracker.dij_stale),
        "optimization": _status("result" in have, tracker.result_stale),
    }


def available_actions(present: Iterable[str]) -> dict[str, bool]:
    """Which workflow actions can run, as the Qt widget enables its buttons.

    The dose influence calculation generates the steering information itself, so it only
    needs the patient and a plan, like ``generate_stf``.
    Stale inputs do not disable an action, they only change the status text.
    """
    have = set(present)
    has_patient_and_plan = {"ct", "cst", "pln"} <= have
    has_stf_dij = {"stf", "dij"} <= have
    return {
        "generate_stf": has_patient_and_plan,
        "calc_dose_influence": has_patient_and_plan,
        "optimize": has_stf_dij,
        "recalc": has_stf_dij and "result" in have,
    }


def status_text(present: Iterable[str], tracker: StageTracker) -> str:
    """One-line workflow status, as the Qt widget shows it."""
    have = set(present)
    if not {"ct", "cst"} <= have:
        return "No data loaded"
    if "pln" not in have:
        return "CT and structures loaded – configure a plan to continue"
    if not {"stf", "dij"} <= have:
        return "Ready for dose influence calculation"
    if tracker.dij_stale:
        return "Dose influence outdated — recalculate"
    if "result" not in have:
        return "Ready for optimization"
    if tracker.result_stale:
        return "Result outdated — re-optimize"
    return "Optimization result available"


def merge_loaded_data(
    data: Mapping[str, Any], has_ct: bool, keys: Iterable[str]
) -> tuple[dict[str, Any], list[str]]:
    """Decide which loaded objects go into the workspace (Qt ``_merge_loaded_data``).

    A supplied CT starts a new patient: every key absent from the import is cleared, so
    observers never see a new CT with the previous patient's structures or products.
    Imports without a CT update the current patient. A structure set needs a CT, in the
    file or already loaded; otherwise it is skipped with a warning. A bare ``dose`` image
    (formats without a matRad ``result``) becomes ``{"physical_dose": dose}``.

    Parameters
    ----------
    data
        What the importer returned (see :func:`pyRadPlan.io.load_data`).
    has_ct
        Whether the workspace already holds a CT.
    keys
        The workspace keys.

    Returns
    -------
    payload, warnings
        Keyword arguments for :meth:`Workspace.set_many` and user-facing warnings.
    """
    keys = tuple(keys)
    warnings: list[str] = []
    payload = {k: data[k] for k in keys if data.get(k) is not None}
    if "cst" in payload and "ct" not in payload and not has_ct:
        warnings.append("Load a CT before loading a structure set; the structures were skipped.")
        payload.pop("cst")
    if "result" not in payload and data.get("dose") is not None:
        payload["result"] = {"physical_dose": data["dose"]}
    if "ct" in payload:
        payload = {**dict.fromkeys(keys), **payload}
    return payload, warnings


def recalculation_angles_match(pln: Any, stf: Any) -> bool:
    """Whether forward dose can reuse the stored beams for the plan's angles.

    Plans imported without angles use the existing steering information. A single couch
    angle applies to every beam; angles are compared modulo 360.
    """
    if pln is None or stf is None or not stf.beams:
        return False
    props = pln.prop_stf or {}
    for field, attribute in (("gantry_angles", "gantry_angle"), ("couch_angles", "couch_angle")):
        value = props.get(field)
        if value is None:
            if field == "couch_angles" and props.get("gantry_angles") is not None:
                value = [0.0]
            else:
                continue
        try:
            angles = np.asarray(value, dtype=float).ravel()
        except (TypeError, ValueError):
            return False
        if field == "couch_angles":
            if angles.size == 0:
                angles = np.zeros(len(stf.beams))
            elif angles.size == 1:
                angles = np.repeat(angles, len(stf.beams))
        if angles.size != len(stf.beams) or not np.all(np.isfinite(angles)):
            return False
        stored = np.array([getattr(beam, attribute) for beam in stf.beams])
        difference = (angles - stored + 180.0) % 360.0 - 180.0
        if not np.allclose(difference, 0.0, atol=1e-6, rtol=0.0):
            return False
    return True


def nested_fraction(levels: Iterable[Any]) -> Optional[float]:
    """Combine nested progress levels (outer to inner) into one fraction.

    Each level contributes its progress weighted by the share of the enclosing level it
    represents, e.g. beam 1/2 with ray 50/100 gives ``0.5 + 0.5 * 0.5 = 0.75``. ``None``
    when the outermost level is indeterminate.
    """
    fraction = 0.0
    weight = 1.0
    seen = False
    for level in levels:
        if not level.total:
            break
        fraction += weight * level.fraction
        weight /= level.total
        seen = True
    return fraction if seen else None


def progress_detail(levels: Iterable[Any]) -> str:
    """``"Beam 1/3 · Ray 20/121"`` for the levels of a progress report."""
    return " · ".join(
        f"{lvl.name} {lvl.current}/{lvl.total}" if lvl.total else f"{lvl.name} {lvl.current}"
        for lvl in levels
    )
