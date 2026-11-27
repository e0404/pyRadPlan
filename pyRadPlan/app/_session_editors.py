"""Session features around the workflow core: profiles, revisions, results, history, editors.

Mixed into :class:`pyRadPlan.app.Session`; kept in its own module so that the session
core stays readable.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from . import _objectives, _planform, _results, _tissue
from ._history import SessionHistory
from ._profiles import resolve_profile, running_in_pyodide
from ._revisions import Revisions
from ._views import resolve_structure

#: Structure changes that only affect how a structure is shown (``update_structure``).
DISPLAY_CHANGES = frozenset({"color", "visible"})


def _precondition(
    message: str, code: str, details: Optional[list] = None, data: Optional[Mapping] = None
) -> Exception:
    from ._session import PreconditionError  # noqa: PLC0415

    return PreconditionError(message, code, details=details, data=data)


def _tissue_errors(errors: list) -> Exception:
    return _precondition(
        "The biological model cannot be evaluated for these tissues; pass apply_anyway=true "
        "to write them nonetheless.",
        "tissue_errors",
        details=[_planform.issue_dict(issue) for issue in errors],
        data={"issues": list(errors)},
    )


class SessionEditorsMixin:
    """Profiles, revisions, history and the structure, tissue, objective and result editors."""

    # ------------------------------------------------------------------
    # Setup and state
    # ------------------------------------------------------------------

    def _init_editors(
        self,
        profile: Optional[str],
        options: Optional[Mapping[str, Any]],
        explicit_default_plan: bool,
    ) -> None:
        name, resolved = resolve_profile(profile, options)
        self.profile = name
        if not explicit_default_plan:
            self.default_plan = dict(resolved["default_plan"])
        self.locked_fields: list[str] = list(resolved.get("locked_fields") or [])
        self.history = SessionHistory()
        self.revisions = Revisions(self.workspace.get)
        self._voxel_counts: dict[int, tuple[Any, int]] = {}

    def _editors_workspace_changed(self, changed_keys: list[str]) -> None:
        self.revisions.update(changed_keys)

    def _num_voxels(self, index: int, voi: Any) -> int:
        cached = self._voxel_counts.get(index)
        if cached is not None and cached[0] is voi.mask:
            return cached[1]
        import numpy as np  # noqa: PLC0415
        import SimpleITK as sitk  # noqa: PLC0415

        try:
            count = int(np.count_nonzero(sitk.GetArrayViewFromImage(voi._nominal_mask_3d())))
        except Exception:  # noqa: BLE001 - a mask that cannot be read counts as empty
            count = 0
        self._voxel_counts[index] = (voi.mask, count)
        return count

    def _editors_state(self, state: dict[str, Any]) -> dict[str, Any]:
        ws = self.workspace
        present = ws.present()
        idle = self._busy is None
        state["revisions"] = self.revisions.as_dict()
        try:
            state["results"] = _results.result_catalog(
                ws.result_entries(), ws.dij, ws.pln, list(ws.results), ws.active_result
            )
        except Exception:  # noqa: BLE001 - placeholders a host stored have no catalog
            state["results"] = _results.result_catalog()
        state["options"] = {
            "profile": self.profile,
            "default_plan": dict(self.default_plan),
            "locked_fields": list(self.locked_fields),
        }
        state["actions"].update(
            {
                "snapshot": idle and "result" in present,
                "load_dij": idle,
                "save_pln": idle and "pln" in present,
                "save_dij": idle and "dij" in present,
                "save_cst": idle and "cst" in present,
                "save_result": idle and bool(ws.results),
                "save_workspace": idle and "ct" in present,
            }
        )
        if state["structures"]:
            vois = ws.cst.vois
            for entry in state["structures"]:
                index = entry["index"]
                voi = vois[index]
                entry["overlap_priority"] = int(voi.overlap_priority)
                entry["num_objectives"] = len(voi.objectives or [])
                entry["num_voxels"] = self._num_voxels(index, voi)
        return state

    # ------------------------------------------------------------------
    # History and control
    # ------------------------------------------------------------------

    def _editors_begin(self, operation: Any) -> None:
        if operation.name == "optimize":
            self.history.start_optimization(operation.label)

    def _editors_status(self, operation: Any, data: dict[str, Any]) -> None:
        if operation.name == "optimize":
            self.history.add_point(data)

    def _editors_end(self, operation: Any, outcome: str) -> None:
        if operation.name == "optimize":
            self.history.finish_optimization(outcome)

    def history_since(self, since: int = 0) -> dict[str, Any]:
        """Return the ``session.history`` result."""
        return self.history.as_dict(int(since or 0))

    def _control_or_raise(self) -> Any:
        if running_in_pyodide():
            raise _precondition(
                "Pause and resume need a second thread, which the browser worker does not "
                "have; use Stop instead.",
                "not_available",
            )
        if self._busy is None:
            raise _precondition("Nothing is running", "not_available")
        return self._busy.control

    def pause(self) -> None:
        """Pause the running operation at its next checkpoint (native / server only)."""
        self._control_or_raise().pause()

    def resume(self) -> None:
        """Resume a paused operation (native / server only)."""
        self._control_or_raise().resume()

    # ------------------------------------------------------------------
    # Plan editor
    # ------------------------------------------------------------------

    def _require_idle(self) -> None:
        if self._busy is not None:
            raise _precondition(f"{self._busy.label} is still running", "busy")

    def plan_detail(self) -> Optional[dict[str, Any]]:
        """``plan.get {detail: true}``."""
        return _planform.plan_detail(self.workspace.pln, self.workspace.cst)

    def plan_options(self, **params: Any) -> dict[str, Any]:
        """``plan.options``."""
        return _planform.plan_options(
            self.workspace.pln, self.default_plan, set(self.locked_fields), **params
        )

    def plan_form(self, kind: str, name: str, **params: Any) -> dict[str, Any]:
        """``plan.form``."""
        return _planform.plan_form(self.workspace.pln, kind, name, **params)

    def plan_form_choices(
        self, kind: str, name: str, radiation_mode: Optional[str] = None
    ) -> dict[str, Any]:
        """``plan.form_choices``: suggested values of an algorithm form's fields."""
        pln = self.workspace.pln
        mode = radiation_mode or (pln.radiation_mode if pln is not None else None)
        return _planform.form_choices(kind, name, mode)

    def plan_tissue_check(self, draft: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """``plan.tissue_check``."""
        return _planform.tissue_check(self.workspace.pln, self.workspace.cst, draft)

    def _build_plan_checked(self, current: Any, changes: Mapping[str, Any]) -> Any:
        from ._plan import build_plan, plan_summary  # noqa: PLC0415

        locked = [name for name in changes if name in self.locked_fields]
        if locked:
            baseline = plan_summary(current) or self.default_plan
            really = [n for n in locked if changes[n] != baseline.get(n)]
            if really:
                raise _precondition(
                    f"Plan field(s) {really} are fixed in this deployment", "locked"
                )
        if current is None:
            changes = {**self.default_plan, **changes}
        summary_changes, overrides = _planform.resolve_changes(current, changes)
        return build_plan(current, summary_changes, overrides)

    # ------------------------------------------------------------------
    # Structures, tissues, objectives
    # ------------------------------------------------------------------

    def cst_get(self, names: Optional[Sequence[Any]] = None) -> dict[str, Any]:
        """``cst.get``; *names* are positions or unique names, the result keeps cst order."""
        cst = self.workspace.cst
        if cst is None:
            raise LookupError("No structure set loaded")
        vois = list(cst.vois)
        wanted = None if names is None else {resolve_structure(vois, ref) for ref in names}
        details = []
        for index, voi in enumerate(vois):
            if wanted is None or index in wanted:
                detail = _tissue.structure_detail(voi, index, self.workspace.result_entries())
                detail["num_voxels"] = self._num_voxels(index, voi)
                details.append(detail)
        return {"structures": details}

    def _structure(self, ref: Any) -> dict[str, Any]:
        return self.cst_get([ref])["structures"][0]

    def update_structure(
        self,
        name: Any,
        changes: Mapping[str, Any],
        apply_anyway: bool = False,
        detail: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Change a structure's type, colour, visibility, overlap priority or tissue.

        *name* is the structure's position or its name, which must then be unique.

        Colour and visibility only change how the structure is shown. On their own they are
        set on the structure in place, outdate nothing and are accepted while an operation
        runs; listeners get a state event (once idle) and ``revisions.workspace.cst``
        counts the change. The other changes recreate the structure and write the structure
        set, which is refused while an operation runs; a changed tissue the biological
        model cannot be evaluated for is refused unless *apply_anyway*. A tissue is a
        ``TissueDraft`` or (in-process) tissue properties.

        Returns the ``StructureDetail`` after the change (``cst.update_structure``), or
        ``None`` without *detail*.
        """
        unknown = set(changes) - {"type", "color", "visible", "overlap_priority", "tissue"}
        if unknown:
            raise ValueError(f"Unknown structure change(s): {sorted(unknown)}")
        display_only = set(changes) <= DISPLAY_CHANGES
        if not display_only:
            self._require_idle()
        ws = self.workspace
        cst = ws.cst
        if cst is None:
            raise LookupError("No structure set loaded")
        vois = list(cst.vois)
        index = resolve_structure(vois, name)
        voi = vois[index]
        update: dict[str, Any] = {}
        if "type" in changes:
            update["voi_type"] = str(changes["type"]).upper()
        if "color" in changes:
            update["color"] = tuple(int(c) for c in changes["color"])
        if "visible" in changes:
            update["visible"] = bool(changes["visible"])
        if display_only:
            if update:
                for field, value in update.items():
                    setattr(voi, field, value)
                self.revisions.update(["cst"])
                if self._busy is None:
                    self._emit_state()
            return self._structure(index) if detail else None
        if "overlap_priority" in changes:
            update["overlap_priority"] = int(changes["overlap_priority"])
        if changes.get("tissue") is not None:
            tissue = _tissue.tissue_of(changes["tissue"], voi.tissue_properties)
            if tissue != voi.tissue_properties:
                if not apply_anyway:
                    errors = _tissue.check_errors(ws.pln, cst, {index: tissue})
                    if errors:
                        raise _tissue_errors(errors)
                update["tissue_properties"] = tissue
        if update:
            vois[index] = _tissue.recreate_voi(voi, **update)
            cst.vois = vois
            ws.cst = cst
        return self._structure(index) if detail else None

    def tissue_schema(self, **context: Any) -> dict[str, Any]:
        """``tissue.schema``."""
        return _tissue.tissue_schema(self.workspace.pln, self.workspace.cst, **context)

    def tissue_evaluate(self, rows: Sequence[Mapping[str, Any]], **params: Any) -> dict:
        """``tissue.evaluate``."""
        if self.workspace.cst is None:
            raise LookupError("No structure set loaded")
        return _tissue.evaluate_tissues(self.workspace.pln, self.workspace.cst, rows, **params)

    def tissue_apply(self, rows: Sequence[Mapping[str, Any]], apply_anyway: bool = False) -> None:
        """``tissue.apply``: write edited tissues (VOIs recreated as in the Qt dialog).

        Rows are ``{"structure": name or position, "tissue": TissueDraft}`` (a name must be
        unique); an in-process frontend may pass tissue properties and ``voi_index``, the
        structure's position, which also tells apart structures of the same name.
        """
        self._require_idle()
        ws = self.workspace
        cst = ws.cst
        if cst is None:
            raise LookupError("No structure set loaded")
        vois = list(cst.vois)
        replaced: dict[int, Any] = {}
        for row in rows:
            name = row["structure"]
            index = row.get("voi_index")
            if index is None:
                index = resolve_structure(vois, name)
            elif not (0 <= index < len(vois) and vois[index].name == name):
                raise KeyError(f"No structure {name!r} at position {index}")
            voi = vois[index]
            tissue = _tissue.tissue_of(row["tissue"], voi.tissue_properties)
            if tissue != voi.tissue_properties:
                replaced[index] = tissue
        if not replaced:
            return
        if not apply_anyway:
            errors = _tissue.check_errors(ws.pln, cst, replaced)
            if errors:
                raise _tissue_errors(errors)
        _tissue.replace_tissues(cst, replaced)
        ws.cst = cst

    def objectives_catalog(self) -> dict[str, Any]:
        """``objectives.catalog``."""
        return _objectives.catalog(self.workspace.result_entries())

    def objectives_edit(
        self, op: str, structure: Any = None, detail: bool = True, **params: Any
    ) -> Optional[dict]:
        """Run one objectives operation (``add``, ``update``, ``change_type``, ``remove``).

        *structure* is the structure's position or its name, which must then be unique; an
        in-process frontend may instead pass the name with ``voi_index``, the position.
        Returns the ``StructureDetail`` of the structure afterwards (``None`` without
        *detail*).
        """
        self._require_idle()
        if self.workspace.cst is None:
            raise LookupError("No structure set loaded")
        if op not in ("add", "update", "change_type", "remove"):
            raise ValueError(f"Unknown objectives operation {op!r}")
        vois = list(self.workspace.cst.vois)
        index = params.pop("voi_index", None)
        if index is None:
            index = resolve_structure(vois, structure)
        name = vois[index].name if isinstance(structure, int) else structure
        getattr(_objectives, op)(self.workspace, name, voi_index=index, **params)
        return self._structure(index) if detail else None

    def objectives_set_quantity_all(self, quantity: str) -> None:
        """``objectives.set_quantity_all``."""
        self._require_idle()
        _objectives.set_quantity_all(self.workspace, quantity)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def snapshot(self, tag: Optional[str] = None) -> str:
        """Keep the active result as a result of its own named *tag*.

        The snapshot shares the images of the active result; a result of that name is
        overwritten. Returns the name used (sanitized, the plan's default tag when
        omitted).

        Raises
        ------
        ValueError
            If the tag is empty after sanitizing or names the active result.
        """
        self._require_idle()
        ws = self.workspace
        result = ws.result
        if result is None:
            raise _precondition("There is no result to snapshot", "not_available")
        tag = _results.sanitize_tag(tag or _results.default_tag(ws.pln))
        if tag == ws.active_result:
            raise ValueError(
                f"'{tag}' names the result being worked on; choose another snapshot name."
            )
        ws.add_result(tag, result.model_copy())
        return tag
