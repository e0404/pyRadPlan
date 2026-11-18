"""Protocol v1 methods, part A (registered on import by :mod:`pyRadPlan.app._dispatch`)."""

from __future__ import annotations

from typing import Any, Optional

from ._dispatch import method
from ._session import Session


@method("session.history")
def _history(session: Session, since: int = 0) -> dict:
    return session.history_since(since)


@method("session.pause")
def _pause(session: Session) -> dict:
    session.pause()
    return {}


@method("session.resume")
def _resume(session: Session) -> dict:
    session.resume()
    return {}


@method("session.cancel")
def _cancel(session: Session) -> dict:
    session.cancel()
    return {}


@method("plan.options")
def _plan_options(
    session: Session,
    radiation_mode: Optional[str] = None,
    machine: Optional[str] = None,
    engine: Optional[str] = None,
) -> dict:
    return session.plan_options(radiation_mode=radiation_mode, machine=machine, engine=engine)


@method("plan.form")
def _plan_form(
    session: Session,
    kind: str,
    name: str,
    radiation_mode: Optional[str] = None,
    values: Optional[dict] = None,
) -> dict:
    return session.plan_form(kind, name, radiation_mode=radiation_mode, values=values)


@method("plan.tissue_check")
def _plan_tissue_check(session: Session, draft: Optional[dict] = None) -> dict:
    return session.plan_tissue_check(draft)


@method("cst.get")
def _cst_get(session: Session, names: Optional[list] = None) -> dict:
    return session.cst_get(names)


@method("cst.update_structure")
def _cst_update_structure(
    session: Session, name: str, changes: dict, apply_anyway: bool = False
) -> dict:
    return session.update_structure(name, changes, apply_anyway=apply_anyway)


@method("tissue.schema")
def _tissue_schema(
    session: Session,
    bio_model: Any = None,
    machine: Any = None,
    engine: Any = None,
    draft: Optional[dict] = None,
) -> dict:
    return session.tissue_schema(bio_model=bio_model, machine=machine, engine=engine, draft=draft)


@method("tissue.evaluate")
def _tissue_evaluate(
    session: Session,
    rows: list,
    op: str = "none",
    pick: Optional[str] = None,
    targets: Optional[list] = None,
    edited: Optional[dict] = None,
    bio_model: Any = None,
    machine: Any = None,
    engine: Any = None,
    draft: Optional[dict] = None,
) -> dict:
    return session.tissue_evaluate(
        rows,
        op=op,
        pick=pick,
        targets=targets,
        edited=edited,
        bio_model=bio_model,
        machine=machine,
        engine=engine,
        draft=draft,
    )


@method("tissue.apply")
def _tissue_apply(session: Session, rows: list, apply_anyway: bool = False) -> dict:
    return session.tissue_apply(rows, apply_anyway=apply_anyway)


@method("objectives.catalog")
def _objectives_catalog(session: Session) -> dict:
    return session.objectives_catalog()


@method("objectives.add")
def _objectives_add(session: Session, structure: str) -> dict:
    return session.objectives_edit("add", structure)


@method("objectives.update")
def _objectives_update(session: Session, structure: str, index: int, changes: dict) -> dict:
    return session.objectives_edit("update", structure, index=index, changes=changes)


@method("objectives.change_type")
def _objectives_change_type(session: Session, structure: str, index: int, name: str) -> dict:
    return session.objectives_edit("change_type", structure, index=index, name=name)


@method("objectives.remove")
def _objectives_remove(session: Session, structure: str, index: int) -> dict:
    return session.objectives_edit("remove", structure, index=index)


@method("objectives.set_quantity_all")
def _objectives_set_quantity_all(session: Session, quantity: str) -> dict:
    return session.objectives_set_quantity_all(quantity)


@method("workflow.snapshot")
def _snapshot(session: Session, tag: Optional[str] = None) -> dict:
    return session.snapshot(tag)
