"""Qt-free application layer between frontends and the pyRadPlan library.

Holds what every frontend of a planning workflow needs -- the observable workspace, the
workflow rules (stage staleness, action availability), the workflow operations with
progress reporting, view products on the CT grid, and a transport-agnostic request
dispatch -- so the PySide6 GUI, a browser frontend running pyRadPlan under Pyodide and a
future server share one implementation. See ``web/PROTOCOL.md`` for the request
protocol and ``docs/development/app_layer.md`` for the design.

Nothing here imports Qt or starts threads.
"""

from ._codec import decode, encode
from ._dispatch import (
    SessionManager,
    UnknownMethodError,
    UnknownSessionError,
    app_info,
    available_methods,
    default_manager,
    dispatch,
)
from ._plan import DEFAULT_PLAN, build_plan, plan_summary
from ._session import Operation, PreconditionError, Session
from ._stages import (
    ACTIONS,
    STAGES,
    StageTracker,
    available_actions,
    merge_loaded_data,
    nested_fraction,
    recalculation_angles_match,
    stage_status,
    status_text,
    tissue_signature,
)
from ._views import ViewCache, default_quantity, grid_info, result_quantities
from ._workspace import WORKSPACE_KEYS, Workspace

__all__ = [
    "ACTIONS",
    "DEFAULT_PLAN",
    "STAGES",
    "WORKSPACE_KEYS",
    "Operation",
    "PreconditionError",
    "Session",
    "SessionManager",
    "StageTracker",
    "UnknownMethodError",
    "UnknownSessionError",
    "ViewCache",
    "Workspace",
    "app_info",
    "available_actions",
    "available_methods",
    "build_plan",
    "decode",
    "default_manager",
    "default_quantity",
    "dispatch",
    "encode",
    "grid_info",
    "merge_loaded_data",
    "nested_fraction",
    "plan_summary",
    "recalculation_angles_match",
    "result_quantities",
    "stage_status",
    "status_text",
    "tissue_signature",
]
