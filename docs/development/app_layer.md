---
orphan: true
---

# Application layer (`pyRadPlan.app`)

Status: draft prototype on `dev/web-pyodide`. It is tested natively (`test/app`); running it
under Pyodide depends on the parallel porting work and has not been verified yet. The Qt GUI does
not use it yet.

## Why

The PySide6 GUI holds workflow logic that is not about widgets: which pipeline objects exist,
when a dose influence is outdated, which steps may run, how a result dict is assembled after an
optimization, how a loaded file is merged into the workspace, which checks run before a
calculation. A browser frontend (React UI, pyRadPlan in Pyodide inside a web worker) needs the
same logic. Reimplementing it in TypeScript would fork the rules; keeping them in Qt widgets
makes them unreachable. `pyRadPlan.app` is the Qt-free layer between frontends and the library
that holds this shared functionality.

```
React UI  -- typed client -->  Web Worker  -- dispatch(dict) -->  pyRadPlan.app  -->  pyRadPlan
Qt GUI    -- direct Python calls (later) ---------------------->  pyRadPlan.app  -->  pyRadPlan
Server    -- WebSocket/HTTP -->  FastAPI  -- dispatch(dict) -->  pyRadPlan.app  -->  pyRadPlan
```

Hard constraints, all checked by tests: no Qt import (`import pyRadPlan.app` does not load
PySide6 or `pyRadPlan.gui`), no threads or processes (no `threading`, `multiprocessing` or
`concurrent` import in the package; everything runs synchronously in the calling thread), solver
and engine choices are parameters.

## Object model

| Object | Module | Responsibility |
|---|---|---|
| `Workspace` | `_workspace.py` | Observable store of `ct, cst, pln, stf, dij, result`; same keys and change semantics as `gui.workspace.WorkspaceManager` (`set`, `set_many` with one notification, `clear`, `has`, `refresh`), plus `subscribe(cb) -> unsubscribe`, `present()`, `snapshot()`. Attribute access (`ws.ct = ...`) as in the Qt manager. |
| `StageTracker` + pure functions | `_stages.py` | Staleness flags (`update(changed_keys, present, cst)` returns a new tracker), `stage_status`, `available_actions`, `status_text`, `merge_loaded_data`, `recalculation_angles_match`, `nested_fraction`, `tissue_signature`. No state besides the frozen tracker, no I/O. |
| `build_plan`, `plan_summary` | `_plan.py` | JSON-able `PlanSummary` <-> validated `Plan`, partial updates carrying everything the summary does not cover, as the Qt plan form's `_build_plan` does. `DEFAULT_PLAN` = the plan form defaults. |
| `ViewCache` | `_views.py` | Volumes (`ct` and result quantities, float32 on the CT grid in `(z, y, x)`), uint8 structure masks, cumulative DVHs. Cached by identity of the source objects, revalidated on every workspace change. |
| `Session` | `_session.py` | Owns or wraps a `Workspace`, keeps the tracker and the view cache in sync with it, runs workflow operations, emits events, answers `state()`. |
| `Operation` | `_session.py` | A prepared step: `compute()` (expensive, never touches the workspace) and `commit(value)` (writes the workspace), plus a `ComputeControl` and a label. |
| `SessionManager`, `dispatch` | `_dispatch.py` | Session registry and the request/response entry point of `web/PROTOCOL.md`; protocol methods are registered with `@method(name)`. |
| `encode`, `decode` | `_codec.py` | JSON + binary buffers codec (`{"$array": i, dtype, shape}`). |

Sessions do the work through three paths, all ending in the same code:

```python
session.calc_dose_influence()                      # synchronous convenience method
session.execute(session.prepare("calc_dose_influence"))   # the same, spelled out
op = session.prepare("calc_dose_influence"); value = op.compute(); op.commit(value)  # split
```

`execute` wraps `compute` in `observe_reports` (progress/status/warning reports of the
algorithms become `progress` / `log` events), `observe_control` (cooperative stop via
`Session.cancel()`), and a temporary `logging` handler on the `pyRadPlan` logger (records become
`log` events, deduplicated per operation because algorithms both log and report a warning).

Workflow operations: `load_phantom`, `load_path`, `load_file(filename, bytes)` (written to a
temporary file, then `pyRadPlan.io.load_data`; works in Pyodide's MEMFS), `set_plan`, `get_plan`,
`generate_stf`, `calc_dose_influence(physical_only=False)` (generates the beams first if missing
or stale), `optimize(solver=None, max_iter=None)`, `recalc(physical_only=False)`,
`run_all(...)` (only missing or stale steps). Views: `volume(name, phase=0)`,
`structure_masks(names=None)`, `dvh(quantity=None, names=None, num_points=500)`.

## Protocol summary

The full contract, with every amendment made while implementing it, is
[`web/PROTOCOL.md`](../../web/PROTOCOL.md). In short: a request is
`{"id", "session", "method", "params", "buffers"?}`; a response is
`{"id", "ok": true, "result", "buffers"}` or `{"id", "ok": false, "error": {type, message, code?,
traceback?}}`; arrays travel as buffers referenced from JSON by `{"$array": i, dtype, shape}`
(C-contiguous, little endian, axis order `(z, y, x)`; spacing/origin `(x, y, z)` mm, LPS). While a
request runs, `emit(event)` receives `state`, `progress` (`label, fraction | null, detail, data?`)
and `log` events. `dispatch` never raises; every exception, including `KeyboardInterrupt`, becomes
an error response.

## What was extracted from the Qt widgets

| Qt source | Now in `pyRadPlan.app` | Notes |
|---|---|---|
| `WorkspaceManager` | `Workspace` | Same keys and semantics, no QObject. |
| `WorkflowWidget._update_staleness`, `_tissue_classes_signature` | `StageTracker.update`, `tissue_signature` | Faithful, plus two extensions: `stf` is a stage of its own (stale on `ct`/`pln` changes) and a CT written without its products outdates them. `test_tracker_agrees_with_the_qt_workflow_widget` replays random change sequences on both and compares the flags. |
| `_update_button_states` | `available_actions` | `generate_stf` is new (the Qt widget generates beams inside "Calc. Dose Influence"). |
| `_update_status` | `status_text` | Without the snapshot-count suffix. |
| `_merge_loaded_data` | `merge_loaded_data` + `Session._commit_loaded` | Warnings are returned instead of shown. |
| `_recalculation_angles_match`, `_nested_fraction` | same names | Copied, the GUI still has its own. |
| `_on_calc_dose`, `_on_optimize`, `_on_recalc_dose` compute/success closures | `Session._prepare_*` | Same result assembly (`compute_result_ct_grid` with the plan's dose factor, scenario labels, `w`, carry-forward of `import_*` keys, iso-center move on recalc). |
| `_check_bio_model`, `_check_objective_quantities` | `Session._check_bio_model`, `_check_objective_quantities` | The question dialog becomes an up-front `physical_only` flag and a `PreconditionError` with `code="bio_model_unsupported"` a frontend can turn into the same question. |
| `PlanWidget._build_plan`, `_apply_default_plan` | `build_plan`, `DEFAULT_PLAN`, auto-plan on load | Only the fields of `PlanSummary` are editable. |
| `ViewingWidget` data derivation, `default_quantity_key` | `ViewCache`, `default_quantity` | `(z, y, x)` instead of the viewer's `(z, x, y)`; identity caching as in `_ct_array` / `_mask_array`. |
| Analysis window DVHs | `ViewCache.dvh` | Uses `pyRadPlan.analysis.DVH.compute` per structure on one shared dose axis. Needs `pyRadPlan.analysis` to import without `matplotlib.pyplot` in a worker (the porting work defers that import). |

What deliberately stays in the frontends: everything interactive (file dialogs, the
"apply pending changes?" question, which becomes "the frontend applies its form before calling a
step"), widget state (window/level, colormaps, slice position, selected structures, isolines),
rendering (slices, contours, beam overlays, DVH plots), result snapshots by tag (a GUI workflow
feature; the layer only carries `import_*` keys forward), exports to disk, the optimization
status window, the AI assistant.

## How sessions map to deployments

**(a) In-browser, Pyodide.** One web worker = one Python interpreter = one `SessionManager`
(the module-level `default_manager()`). The worker's message handler calls
`dispatch(request, emit)` with `emit` posting events back; the response's `buffers` are
transferred as `ArrayBuffer`s. Several sessions (e.g. tabs of plans) can live in one worker, but a
request blocks the worker until it finishes, so a second request of any session waits. Cancelling
cannot be a request (none is processed while one runs); instead the main thread writes to
Pyodide's interrupt buffer (`pyodide.setInterruptBuffer`, needs a SharedArrayBuffer, i.e.
cross-origin isolation), Python raises `KeyboardInterrupt` at the next bytecode boundary and
`dispatch` turns it into an error response; the session state stays consistent because nothing is
committed before `compute` returns. Parallel sessions would need one worker each.

**(b) Server.** A FastAPI app holds one `SessionManager` per process. Each WebSocket connection
creates (or resumes) a session and forwards frames to the same `dispatch`, sending events as JSON
frames and buffers as binary frames following the response; HTTP-only clients can call
`dispatch` per POST and fetch buffers separately. Because `dispatch` is synchronous, the server
runs it in a thread pool (`run_in_threadpool`), one request per session at a time; a session lock
and `Session.cancel()` (real cooperative stop via `ComputeControl`) come for free in this setting.
Memory per session (a dij can be hundreds of MB) is the main sizing constraint; an idle-timeout
`close()` is needed. Nothing of this is implemented.

**(c) Qt GUI in-process.** The GUI would call `Session` directly, no `dispatch`, no codec. Sketch
(not applied; the GUI is unchanged):

```python
class WorkspaceManager(QObject):
    workspace_changed = Signal(list)

    def __init__(self, parent=None, store: Workspace | None = None):
        super().__init__(parent)
        self.store = store or Workspace()
        self.store.subscribe(self.workspace_changed.emit)

    def set_many(self, **kwargs):
        self.store.set_many(**kwargs)

    def clear(self, keys=None):
        self.store.clear(keys)

    def has(self, *keys):
        return self.store.has(*keys)

    ct = property(lambda self: self.store.ct, lambda self, v: self.store.set("ct", v))
    # ... same for cst, pln, stf, dij, result


class WorkflowWidget(WorkspaceWidget):
    def __init__(self, workspace=None, parent=None):
        super().__init__(workspace, parent)
        self._session = Session(workspace=self._ws.store, auto_plan=False)  # PlanWidget applies it
        self._session.add_listener(self._on_session_event)  # state -> dots, status, buttons

    def _on_calc_dose(self):
        if not self._resolve_pending_changes("Calculate Dose Influence"):
            return
        try:
            op = self._session.prepare("calc_dose_influence")
        except PreconditionError as exc:
            if exc.code != "bio_model_unsupported" or not self._ask_physical_dose_only(...):
                return self._show_error(exc)
            op = self._session.prepare("calc_dose_influence", physical_only=True)
        self._run_in_thread(op.compute, on_success=op.commit, control=op.control,
                            busy_text=op.label)
```

`compute` runs on the existing `QThread` `Worker` (which already installs `observe_reports` and
`observe_control`), `commit` runs on the GUI thread in `_on_thread_finished`, exactly where the
widget writes the workspace today, so Qt signals keep firing on the GUI thread. The widget would
read its dots, status text and button states from `session.state()` instead of its own flags.
Migration can go one operation at a time; the staleness parity test protects the rules meanwhile.

## Design choices and alternatives

**Command dispatch vs. exposing Python objects through Pyodide proxies.** Pyodide can hand the
JS side `PyProxy` objects (`session.workspace.ct.cube_hu...`), which is tempting for a prototype.
Rejected: proxies leak Python lifetimes into JS (every proxy must be `destroy()`ed or it pins
memory), the React code would bind to pyRadPlan's internal object model (any refactor breaks the
UI), each attribute access is a cross-language call, and the same API would not work over a
network. A command protocol of plain dicts is transport-independent (worker today, WebSocket
later, in-process for tests), has one documented surface, and errors cross it as data. Cost: every
new UI need is a new method; we accept that as the place where the contract is reviewed.

**JSON + separate buffers vs. msgpack / Arrow / base64.** JSON with out-of-band buffers is native
on both sides (`postMessage` structured clone plus a transfer list, WebSocket text + binary
frames), needs no extra dependency in Pyodide or the bundle, and lets large arrays be transferred
zero-copy instead of serialized. msgpack would put metadata and binary into one message but adds a
dependency on both sides and still copies; Arrow/IPC fits tabular data, not 3-D volumes, and is a
large wasm/JS dependency; base64 in JSON inflates by a third and costs encoding time on 15 MB CT
volumes. The `{"$array": i, dtype, shape}` descriptor keeps the door open: a future transport can
map buffers to whatever it likes without touching the methods.

**State snapshot events vs. fine-grained diffs.** The `state` event carries the full
`SessionState` (a few kB: no arrays, only flags, the plan summary, structure metadata). Diffs
(JSON Patch, or per-key change events like `workspace_changed`) would be smaller but force every
frontend to maintain a replica and handle ordering and missed events. Snapshots are idempotent,
make reconnects trivial (`session.state`), and fit React (`setState(event.state)`). Large data is
never in the state; frontends re-request views when `quantities` / `structures` change. Snapshots
are emitted once per workspace change when idle and once at the start and end of an operation, not
per intermediate write.

**Where the staleness logic lives.** Options: (1) keep it in the Qt widget and duplicate it in
TypeScript, (2) put it into `Workspace` itself, (3) pure functions plus a small value object in the
app layer, driven by the session. (1) forks the rules. (2) makes the store opinionated and every
consumer of a workspace pay for workflow semantics it may not want (e.g. scripts). (3) keeps the
store dumb, the rules testable without any objects (tests pass `object()` sentinels), and the
tracker reusable by the Qt widget unchanged; the session is the only place that combines them.
Chosen: (3).

**Synchronous operations with a prepare/compute/commit split vs. an executor in the session.** An
executor hook (`Session(executor=...)`) would make the session schedule work itself, but then it
also owns result delivery, cancellation and thread affinity of the workspace writes, all of which
differ between Qt (QThread + signals), a server (thread pool + event loop) and Pyodide (none).
Splitting each step into `compute` (pure with respect to the workspace) and `commit` lets each
host schedule `compute` its own way and call `commit` where the workspace lives, with no
threading code in the layer.

**Identity-based view caching.** Views are cached keyed by what they show and validated by the
identity of their source objects (CT image, result entry, structure mask), the same strategy as
`ViewingWidget`. This survives the frequent workspace writes that do not change the images
(objective edits, plan changes) and needs no version counters in the data model. It does not
notice in-place mutation of an image, which pyRadPlan code does not do to stored objects.

## Known gaps and open problems (draft quality)

- **Not run under Pyodide yet.** Everything is tested natively on CPython 3.11/Windows only.
  `tempfile`, `logging` and the synchronous design should be fine in Pyodide; the library
  underneath (SimpleITK, numba, h5py, scipy under wasm) is the porting work's concern.
- `ComputeControl` uses `threading.Event` internally (in `pyRadPlan.core`), which exists in
  Pyodide but cannot be signalled while Python runs; cancellation there is the interrupt buffer.
- tqdm console bars still print during operations (`ProgressReporter.console_progress`); a worker
  should turn them off globally. IPOPT prints its banner/iterations to stdout natively.
- The biological-model check reimplements the GUI's "quietly" helpers from
  `gui/widgets/_tissue_params.py` (a Qt module) with the biology API directly, without the
  machine-summary cache: for ion plans it loads the machine base data before each calculation.
  Those helpers should move to a Qt-free module (e.g. `pyRadPlan.biology`), then both use them.
- `recalculation_angles_match` and `nested_fraction` are duplicated between the GUI and the app
  layer until the GUI imports them from here.
- Loading bare image files (`.nii`, `.nrrd`, ...) goes through `load_data` without the Qt dialog's
  "is this a CT, structures or a dose?" choice; DICOM folders are not supported through
  `data.load_file` (single file upload). Saving/exporting is not in the protocol.
- `PlanSummary` covers only the fields the protocol names; objectives, scenario models, tissue
  parameters, engine and solver options have no methods yet (`cst` editing is the obvious next
  one).
- No result snapshots, no `Session` persistence (pickle of the workspace), no per-session locking
  (a second thread calling the same session while busy gets `PreconditionError(code="busy")`
  only for steps; views are not guarded).
- Commit does not re-check that the inputs captured at prepare time are still current (the Qt
  widget does not either).
- Volumes are float32; a CT could go as int16 to halve transfers. Masks are one uint8 volume per
  structure; a bit-packed or label-map encoding would be much smaller for many structures.
- `view.volume` recomputes the value range per call (cheap but not cached).
