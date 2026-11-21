"""Generate web/src/api/mock/fixtures/*.json: recorded responses of ``pyRadPlan.app``.

The mock backend answers the static catalogue requests (registries, plan options, configuration
forms, tissue schema, objectives catalogue) with these recordings instead of re-implementing the
app layer. The recordings simulate the browser runtime (Pyodide, ``web`` profile, numpy only), so
they do not depend on the machine that runs this script: environment-dependent features are
pinned and the user data folder is an empty temporary directory.

The files are not committed: ``npm run fixtures`` (run automatically before every npm script that
type-checks, tests, builds or serves the mock) regenerates them from the pyRadPlan source of this
checkout; its dependencies come from the active Python environment.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "src" / "api" / "mock" / "fixtures"

sys.path.insert(0, str(ROOT))

BROWSER_IPOPT = "ipyopt is not installed"


def _pin_environment() -> None:
    os.environ["PYRADPLAN_DATA_DIR"] = tempfile.mkdtemp(prefix="pyrp-fixtures-")
    for name in list(os.environ):
        if name.startswith("PYRADPLAN_") and name != "PYRADPLAN_DATA_DIR":
            del os.environ[name]


def _simulate_pyodide() -> None:
    from pyRadPlan.app import (  # noqa: PLC0415
        _io,
        _profiles,
        _session,
        _session_editors,
        _settings,
    )
    from pyRadPlan.optimization import solvers  # noqa: PLC0415

    def in_pyodide() -> bool:
        return True

    for module in (_profiles, _io, _session, _session_editors, _settings):
        module.running_in_pyodide = in_pyodide  # type: ignore[attr-defined]

    solvers.IPOPT_DISABLED_REASON = BROWSER_IPOPT
    registered = solvers.get_available_solvers

    def without_ipopt() -> dict:
        return {k: v for k, v in registered().items() if k != "ipopt"}

    solvers.get_available_solvers = without_ipopt


class Recorder:
    """Send requests to the application layer and return their results."""

    def __init__(self) -> None:
        from pyRadPlan.app import SessionManager, dispatch  # noqa: PLC0415

        self._dispatch = dispatch
        self._manager = SessionManager()
        self._n = 0

    def call(self, method: str, session: str | None = None, **params: Any) -> Any:
        """Return the result of one request; an error response raises."""
        self._n += 1
        request = {"id": self._n, "method": method, "params": params}
        if session is not None:
            request["session"] = session
        response = self._dispatch(request, manager=self._manager, with_traceback=False)
        if not response["ok"]:
            raise RuntimeError(f"{method}{params}: {response['error']['message']}")
        if response["buffers"]:
            raise RuntimeError(f"{method} returned binary buffers")
        return response["result"]


def record() -> dict[str, Any]:
    """Record the responses, by fixture name."""
    _pin_environment()
    _simulate_pyodide()
    rec = Recorder()

    info = rec.call("app.info")
    app_info = {k: info[k] for k in ("phantoms", "machines", "solvers", "engines", "quantities")}

    session = rec.call("session.create", profile="web")["session"]
    locked = rec.call(
        "session.create", profile="web", options={"locked_fields": ["dose_grid_resolution"]}
    )["session"]

    options: dict[str, Any] = {}
    for mode, machines in info["machines"].items():
        for machine in machines:
            for engine in info["engines"][mode]:
                key = f"{mode}|{machine}|{engine}"
                result = rec.call(
                    "plan.options", session, radiation_mode=mode, machine=machine, engine=engine
                )
                result.pop("bio_selection")
                options[key] = result
    fields_locked = rec.call("plan.options", locked)["fields"]
    contexts = list(options.values())
    shared = {
        key: contexts[0][key]
        for key in contexts[0]
        if all(context[key] == contexts[0][key] for context in contexts)
    }
    options = {
        key: {k: v for k, v in context.items() if k not in shared}
        for key, context in options.items()
    }

    forms: dict[str, Any] = {}
    kinds = {
        "engine": {(m, e) for m, es in info["engines"].items() for e in es},
        "scenario": {(None, s["value"]) for s in shared["scenario_models"]},
        "problem": {(None, p["value"]) for p in shared["problems"]},
        "solver": {(None, s["value"]) for s in shared["solvers"]},
    }
    models = {
        (mode, json.loads(b["value"])["model"])
        for key, opt in options.items()
        for mode in (key.split("|")[0],)
        for b in opt["bio_models"]
    }
    kinds["bio_model"] = models
    kinds["sequencer"] = {
        (key.split("|")[0], s["value"]) for key, opt in options.items() for s in opt["sequencers"]
    }
    for kind, entries in kinds.items():
        for mode, name in sorted(entries, key=lambda e: (str(e[0]), e[1])):
            if kind in {"bio_model", "engine", "sequencer"}:
                result = rec.call("plan.form", session, kind=kind, name=name, radiation_mode=mode)
                forms[f"{kind}:{mode}:{name}"] = result
            else:
                forms[f"{kind}:{name}"] = rec.call("plan.form", session, kind=kind, name=name)

    rec.call("data.load_phantom", session, name="TG119")
    tissue_schema = rec.call("tissue.schema", session)
    catalog = rec.call("objectives.catalog", session)
    phantom = {
        "TG119": {
            s["name"]: {
                "overlap_priority": s["overlap_priority"],
                "tissue": s["tissue"],
                "objectives": [
                    {k: v for k, v in o.items() if k != "index"} for o in s["objectives"]
                ],
            }
            for s in rec.call("cst.get", session)["structures"]
        }
    }

    return {
        "app_info": app_info,
        "plan_options": {"shared": shared, "contexts": options},
        "plan_fields_locked_grid": fields_locked,
        "plan_forms": forms,
        "tissue_schema": tissue_schema,
        "objectives_catalog": catalog,
        "phantom_structures": phantom,
    }


def main() -> int:
    """Write the fixtures."""
    files = {
        f"{name}.json": json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        for name, value in record().items()
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (OUT / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {len(files)} fixtures to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
