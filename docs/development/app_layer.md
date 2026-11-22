---
orphan: true
---

# Application layer (`pyRadPlan.app`)

Status: draft prototype on `dev/web-pyodide`. It is tested natively (`test/app`) and under
Pyodide (`web/pyodide`). The Qt GUI runs on it: its `WorkspaceManager` wraps a `Workspace` and a
`Session`; the workflow widget takes its stage flags, status text, workflow steps, loads, imports
and saves from that session, and the plan form, the objectives table, the VOI editor and colour
picks and the tissue dialog write through it (see
[The Qt GUI on the session](#the-qt-gui-on-the-session)). The Qt modules also import the pure
helpers listed in [Shared with the Qt GUI](#shared-with-the-qt-gui).

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
Qt GUI    -- direct Python calls (Session, Workspace) --------->  pyRadPlan.app  -->  pyRadPlan
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
| `Session` | `_session.py` | Owns or wraps a `Workspace`, keeps the tracker and the view cache in sync with it, runs workflow operations, emits events, answers `state()`. Host options: `solver_preference` (empty: the problem's default solver), `capture_logs`, `auto_plan`, `profile`. |
| `Operation` | `_session.py` | A prepared step: `compute()` (expensive, never touches the workspace) and `commit(value)` (writes the workspace), plus a `ComputeControl`, a label and the non-blocking `warnings` found while preparing. |
| `PreconditionError` | `_session.py` | A step cannot run: `code` to branch on, `details` (JSON-able, sent as `error.details`), `data` (Python objects for in-process frontends, e.g. the `TissueIssue`s of `bio_model_unsupported`; never serialized). |
| `SessionManager`, `dispatch` | `_dispatch.py` | Session registry and the request/response entry point of `web/PROTOCOL.md`; protocol methods are registered with `@method(name)`. |
| `encode`, `decode` | `_codec.py` | JSON + binary buffers codec (`{"$array": i, dtype, shape}`). |

Sessions do the work through three paths, all ending in the same code:

```python
session.calc_dose_influence()                      # synchronous convenience method
session.execute(session.prepare("calc_dose_influence"))   # the same, spelled out
op = session.prepare("calc_dose_influence")        # host-scheduled (the Qt GUI):
session.begin(op)                                  #   owning thread: busy, history, state event
value = session.run(op)                            #   any thread: compute with observation
session.finish(op, value)                          #   owning thread: commit, idle, state event
```

`execute` is `begin` + `run` + `finish` in the calling thread (`abort(op, exc)` when `run`
raises), so the web path is a special case of the host-scheduled one. `run` wraps `compute` in
`observe_reports` (progress/status/warning reports of the algorithms become `progress` / `log`
events), `observe_control` (cooperative stop via `Session.cancel()`, pause/resume through the
operation's `ComputeControl`) and, with `capture_logs`, a temporary `logging` handler on the
`pyRadPlan` logger (records become `log` events, deduplicated per operation because algorithms
both log and report a warning). Report observers and the control are context variables, so they
must be installed in the thread running `compute`; that is why `run` is separate. `run` never
writes the workspace; a failing `commit` in `finish` aborts the operation and re-raises.

Events are only built while there are listeners (a state snapshot costs a pass over every
structure), and the guard against re-entrant events is per context, so an event emitted by a
worker thread is not dropped while the owning thread emits one. `state()` reports parts it
cannot derive (placeholder objects a host or a test stored) as missing instead of raising.

Workflow operations: `load_phantom`, `load_path`, `load_file(filename, bytes)` (written to a
temporary file, then `pyRadPlan.io.load_data`; works in Pyodide's MEMFS), `set_plan`, `get_plan`,
`generate_stf`, `calc_dose_influence(physical_only=False)` (always generates the beams first, as
the Qt button does: the automatic isocenter follows the structures, whose edits do not outdate the
stf), `optimize(solver=None, max_iter=None)`,
`recalc(physical_only=False)`, `snapshot(tag=None)` (returns the stored tag),
`merge_loaded(data)` (returns the warnings), prepare-only `load_dij(path)`,
`run_all(...)` (only missing or stale steps). Views: `volume(name, phase=0)`,
`structure_masks(names=None)`, `dvh(quantity=None, names=None, num_points=500)`.

Native file operations (prepare-only, paths instead of bytes, defined in `_io.py`; not protocol
methods): `save_path(object, format, path, quantities=None, structure_format=None)` writes with
the same `write_export` as the `export.save` download (a file, or a folder for label maps, DICOM
and several result quantities; a path without an ending gets the format's, another format's ending
is refused) and returns the paths written; `import_image(path, mode, name=None)` (`ct_new`,
`ct_replace`, `structures`, `dose`, the Qt Import Image choices), `import_binary(ct_file,
selections)`, `scan_dicom(directory)` (returns the importer and the catalog) and
`import_dicom(importer, selection)`. They need SimpleITK file IO and refuse with
`PreconditionError(code="not_available")` without it, e.g. under the Pyodide stand-in, where the
capability flags report them unavailable. `load_path` reads a dose-only `.npz` as a bare dose, as
the upload does. The format tables (`EXPORT_FORMATS`, `import_formats()`) also give the Qt file
dialog filters (`load_filter`, `export_filter`, `entry_for_filter`, `path_label`).

`call_when_idle(callback)` runs a callback now or once the running operation has ended; the Qt
viewer keeps a VOI edit made during a calculation with it. An `Operation` with
`keeps_partial_result` (the optimization) ends with its result so far on a stop request; under
Pyodide the first interrupt (Stop) becomes such a request, a second one raises
`KeyboardInterrupt` as before, and the history outcome of a stopped optimization is `"stopped"`.

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
| `WorkspaceManager` | `Workspace` | Same keys and semantics, no QObject; the manager now wraps one. |
| `WorkflowWidget._update_staleness`, `_tissue_classes_signature` | `StageTracker.update`, `tissue_signature` | Removed from the widget, which reads `session.tracker`. Two extensions over the old widget: `stf` is a stage of its own (stale on `ct`/`pln` changes) and a CT written without its products outdates them. `test_tracker_agrees_with_input_versions_over_random_changes` checks the tracker against a version-counter oracle. |
| `_update_button_states` | `available_actions` | `generate_stf` is new (the Qt widget generates beams inside "Calc. Dose Influence"). Qt keeps "Load Dij" always enabled. |
| `_update_status` | `status_text` | Qt appends its "· n saved snapshots" suffix. |
| (former `_merge_loaded_data`) | `merge_loaded_data` + `Session.merge_loaded` / `_commit_loaded` | Warnings are returned (and logged as events); Qt shows them after the write. |
| `_recalculation_angles_match`, `_nested_fraction` | same names | The GUI imports them from here. |
| `_on_calc_dose`, `_on_optimize`, `_on_recalc_dose`, loads, Load Dij, Snapshot Result | `Session._prepare_*`, `Session.snapshot` | The widget runs the session's operations (see below). Same result assembly (`compute_result_ct_grid` with the plan's dose factor, scenario labels, `w`, carry-forward of snapshots and `import_*` keys, iso-center move on recalc). |
| `_check_bio_model`, `_check_objective_quantities` | `Session._check_bio_model`, `_check_objective_quantities` | The question dialog becomes an up-front `physical_only` flag and a `PreconditionError` with `code="bio_model_unsupported"`; Qt asks with its own dialog from `error.data` and prepares again with `physical_only=True`. The tissue check uses the Qt-free helpers of `pyRadPlan.biology._editing` (machine summary cache) for both frontends. |
| Viewer `QUANTITY_META`, `default_quantity_key`, `_plan_iso_centers`, `_plan_beam_angles`, `_beam_geometry`, `_apply_scenario_labels`; Save Result dialog and `analysis_viewer` labels | `quantity_meta`, `default_quantity`, `plan_iso_centers`, `plan_beam_angles`, `beam_geometry` (LPS mm), `scenario_labels` | One table without the `*_beam` duplicates: `quantity_meta` strips `_beam`, `_scenario`, `_beam` in turn; the Save Result dialog and `analysis_viewer` pass `suffixes=("_beam",)` to label exactly what they labelled before. Qt keeps only its LPS-to-voxel transform (`_viewer_beams`). |
| `PlanWidget._build_plan`, `_apply_default_plan`, `_props_key`, `_form_prop_opt`, problem naming | `Session.set_plan` (`resolve_changes` + `build_plan`), `changes_from_form`, `plan_form_values`, `comparison_key`, `DEFAULT_PLAN` / profile `default_plan` | The form sends its fields as `plan.set` params (`PlanWidget._form_changes`); `changes_from_form` leaves out those equal to the plan's, so what the form cannot represent survives (see below). The form's initial values are the session's `default_plan`. |
| Plan form option lists and sync (`_refresh_machines`, `_refresh_engines`, `_refresh_problems`, scenario combo, angle presets, `_restore_engine_from_pln`, `_restore_problem_from_pln`, `_restore_iso_center`) | `machine_choices`, `engine_choices`, `problem_choices`, `problem_classes`, `engine_class`, `scenario_choices`, `beam_presets`, `plan_machine_label`, `plan_engine_config`, `plan_problem`, `plan_summary` (`_planform.py`; `plan.options` and `plan.get {detail}` are built from the same functions) | Bio-model options stay `BioModelOption` objects from `biology._editing` (the helper `plan.options` lists too). |
| `PlanWidget._refresh_bio_models` selection, `_replacement_note`, `_add_unsupported_bio_model` texts, `_first_option_index` | `select_bio_model` -> `BioSelection`, `first_bio_option`, `UNSUPPORTED_SUFFIX` (`_planform.py`); `plan.options {current_bio}` -> `bio_selection` | Qt fills its combo from the `BioSelection` (entries, selected option, the kept unsupported entry with its label and tooltip, whether to take over the option's parameters, the note); the web applies `bio_selection` (`planModel.applyBioSelection`) and has no rules of its own. `test_plan_form_read.py` checks the Qt selection and note against `plan.options` for 13 plans and three target modes. |
| Optimization dialog `_default_solver`, config form `_merged_mapping`, `_label_for` | `default_solver`, `merged_mapping`, `field_label` (`_forms.py`, `_planform.py`) | `merged_mapping` keeps field values (no `model_dump()` / `asdict()`) and leaves out non-init dataclass fields; what the forms show is unchanged. |
| `ViewingWidget` data derivation, `default_quantity_key` | `ViewCache`, `default_quantity` | `(z, y, x)` instead of the viewer's `(z, x, y)`; identity caching as in `_ct_array` / `_mask_array`. |
| Analysis window DVHs | `ViewCache.dvh` | Uses `pyRadPlan.analysis.DVH.compute` per structure on one shared dose axis. Needs `pyRadPlan.analysis` to import without `matplotlib.pyplot` in a worker (the porting work defers that import). |
| Load File / Save ... file filters, `_cst_export_options`, `_image_formats`, `_image_format_from_filter`, `_build_*_filter` | `export_entries`, `path_label`, `path_extensions`, `writes_folder`, `load_filter`, `export_filter`, `entry_for_filter` (`_io.py`) | One table for both frontends. Qt changes: labels from the table ("MATLAB (*.mat)", "NIfTI (*.nii.gz *.nii)", label maps "(folder)"), DICOM no longer in the Load File filter (Load Folder imports it; "All files" still opens one), result image formats in table order (NIfTI first), formats whose writer is missing (e.g. RTSTRUCT without `contourpy`) not offered. |
| `_on_save_workspace`, `_save_single_object`, `_on_save_cst`, `_on_save_result_to_disk` writes | `save_path` operation (`prepare_save_path`, `write_export`) | Qt keeps the dialogs, the objectives warning and the "Saved:" box; the session writes. A file name with another format's ending is refused with "Use .x or .y to save ..." for every object (Qt refused it for the workspace only; results got the ending appended). Several result quantities are written as `<key>.<ext>` with the key sanitized. |
| Image / binary / DICOM import computes, `_unique_name`, `_folder_has_images`, binary `_scan_folder`, DICOM `scan_folder`, image dialog inspection | `import_image`, `import_binary`, `scan_dicom`, `import_dicom`, `unique_name`, `folder_has_images`, `folder_kind`, `binary_folder_files`, `dicom_catalog`, `inspect_image` (`_io.py`) | A new-patient CT import is one workspace write (was clear + set); "CT replace" decides on the grid itself (header read). |
| `OptimizationWidget` edits (`_on_*_changed`, `_write_objectives`, `_set_all_quantities`, `_iter_objectives`, `_reference_dose_options`, `_DEFAULT_OBJECTIVE_BY_TYPE`) | `Session.objectives_edit`, `objectives_set_quantity_all`; `objectives_of`, `reference_values` (`_objectives.py`) | Rows address the VOI by position (`voi_index`), so structures of the same name stay apart (names alone were the app's key). Edits are validated as a whole before anything is written; a rejection shows its reason in the table's status line. Both skip empty and invalid objective entries. |
| `ViewingWidget._on_voi_replaced`, VOI colour picks, `PlanWidget._on_tissue_parameters`, the tissue dialog's write | `Session.update_structure`, `tissue_apply` (`voi_index` rows, tissue properties accepted), `replace_tissues` (`_tissue.py`) | Colour and visibility are display edits (in place, allowed while busy, outdating nothing); a VOI edit made during a calculation is applied when it ends. |

### Shared with the Qt GUI

The Qt modules import these instead of keeping copies (import direction gui -> app -> library):

| Qt module | Imports |
|---|---|
| `menus/_help.py` | `DistributionInfo`, `installed_distributions`, `license_text` (`_settings.py`); `DESCRIPTION`, `LICENSE_NOTE`, `DISCLAIMER_TITLE`, `DISCLAIMER` (`_about.py`, read from `about.json`, which the web build embeds as well) |
| `menus/_settings.py`, `menus/_file.py` | `apply_settings_values`, `SECTION_TITLES` (`_settings.py`); the web writes `str(jsonable(value))` through `render`, Qt `str(value)` |
| `widgets/ai` | `AI_MISSING`, `AI_NO_MODEL` (`_settings.py`) as `AI_MISSING_TIP`, `AI_NO_MODEL_TIP` |
| `menus/_base_data.py` | `machine_identity`, `rbe_table_summary`, `mat_file_name` (`_basedata.py`) |
| `gui/workspace.py` | `Workspace`, `Session` (the manager's store and session) |
| `widgets/workflow/_workflow_widget.py` | `Session` operations and `PreconditionError`; `available_actions`, `status_text`, `nested_fraction`, `progress_detail` (`_stages.py`); `default_tag`, `is_tagged` (`_results.py`); `load_filter`, `export_entries`, `export_filter`, `entry_for_filter`, `path_label`, `writes_folder`, `folder_kind`, `is_image_file`, `image_keys` (`_io.py`) |
| `widgets/workflow/_image_import_dialog.py`, `_binary_import_dialog.py`, `_dicom_import_dialog.py` | `inspect_image`, `binary_folder_files`, `dicom_catalog` (`_io.py`) |
| `widgets/optimization/_optimization_widget.py` | `Session.objectives_edit`, `objectives_set_quantity_all`; `objectives_of`, `reference_values` (`_objectives.py`) |
| `widgets/_result_widget.py` | `quantity_meta`, `scenario_labels` (`_results.py`); `beam_geometry`, `plan_beam_angles`, `plan_iso_centers`, `default_quantity` (`_views.py`); `Session.update_structure`, `call_when_idle` |
| `widgets/plan/_tissue_dialog.py` | `replace_tissues` (`_tissue.py`); writes through `Session.tissue_apply` when the plan form passes it |
| `widgets/result/_save_result_dialog.py`, `apps.py` | `quantity_meta` (`_results.py`), `default_quantity` (`_views.py`) |
| `widgets/plan/_plan_widget.py` | `Session.set_plan`, `default_plan`, `call_when_idle`, `PreconditionError`; `plan_class`, `plan_summary` (`_plan.py`); `RADIATION_MODES`, `bio_change`, `optimization_change`, `changes_from_form`, the choice helpers and plan readers above, `scenario_overrides`, `select_bio_model`, `BioSelection`, `first_bio_option`, `ENGINE_FORM_EXCLUDE` (engine dialog exclude set), `engine_form_values` (`_planform.py`); `comparison_key` (`_forms.py`) |
| `widgets/plan/_optimization_dialog.py` | `PROBLEM_FORM_EXCLUDE`, `split_solver_spec`, `solver_defaults`, `default_solver` (`_planform.py`) |
| `widgets/_config_form.py` | `unwrap_optional`, `nested_fields`, `valid_entries`, `merged_mapping`, `field_label` (`_forms.py`) |
| `widgets/result/vois_widget.py` | `pyRadPlan.cst.recreate_voi` (library; `_tissue.recreate_voi` delegates to it) |

What deliberately stays in the frontends: everything interactive (file dialogs, option dialogs,
the "apply pending changes?" question, which becomes "the frontend applies its form before calling
a step"), widget state (window/level, colormaps, slice position, selected structures, isolines;
the viewer's structure checkboxes are view state, not `voi.visible`, in both frontends), rendering
(slices, contours, beam overlays, DVH plots), threads, the optimization status window, the AI
assistant.

### The Qt GUI on the session

`WorkspaceManager` (a `QObject` singleton, API unchanged: properties, `set_many`, `clear`, `has`,
`refresh`, `keys`) holds `store: Workspace` and `session: Session(workspace=store,
profile="native", auto_plan=False, solver_preference=(), capture_logs=False)`. The session
subscribes to the store first and the manager then re-emits `workspace_changed`, so every widget
sees the session's flags already updated (the store calls its listeners in subscription order).
The Qt options keep what the GUI did before: the plan form applies the default plan, the
optimization problem's default solver applies and the log panel has its own handler.

The workflow widget no longer keeps staleness, tags or status of its own. Its dots show
`session.stages()`, its status line `status_text(present, session.tracker)` plus the snapshot
count, its buttons `available_actions(present)` (the busy cycle disables them while a run is
active, so they do not use the busy-gated `session.actions()`). The staleness and snapshot
tags are read from `session.tracker` and `session.saved_tags`.

Steps go through `_prepare(step, action)` and `_run_operation(op)`:

```python
op = session.prepare("calc_dose_influence")     # PreconditionError -> the Qt dialogs
session.begin(op)                               # GUI thread
self._run_in_thread(lambda: session.run(op),    # QThread Worker, pause/resume/stop via op.control
                    on_success=lambda v: session.finish(op, v),   # GUI thread: commit
                    on_error=lambda exc: session.abort(op, exc),
                    control=op.control, busy_text=f"{op.label}…")
```

`QThread`, `Worker`, the progress bar and status window (fed by the raw `ComputeReport`s the
worker forwards), pause/resume, stop and GPU use stay in Qt. `PreconditionError` codes map to
the dialogs the widget showed before, with unchanged texts: `bio_model_unsupported` asks for the
physical dose only and prepares again with `physical_only=True`, `invalid_reference` and
`biological_objectives` are explained from `error.data`, `no_weights` /
`beam_geometry_changed` / `no_dij` as before. Tissue warnings of a step come as
`Operation.warnings` and are logged and shown in the status line before the run starts.
Migrated: Calc. Dose Influence, Optimize, Recalculate Dose, phantom and file loads, Load Dij,
Snapshot Result, the image / binary / DICOM imports (scan and load), every save and export
(`save_path`, reported by the same "Saved:" box), the objectives table, VOI editor and colour
edits, the tissue dialog. A save refused at prepare time (missing object, wrong file ending) shows
the session's message in a warning box. The VOI editor is reachable while a calculation runs (the
viewer stays enabled); its edit is kept with `call_when_idle` and written when the run ends,
instead of into the structure set the run is using. The status line is refreshed when any
operation ends, also one that writes nothing (a save used to leave "Saving workspace…" until
the next workspace change).

**Plan form.** `PlanWidget` keeps its widgets, option bookkeeping (the combo entries of the
`BioSelection` that `select_bio_model` returns after a mode, machine or engine change or for a
plan, the explicit scenario entry), per-field orange borders and dialogs. Apply and the default
plan go through the session:

```python
form = widget._form_changes()                # every field as plan.set params; Qt specs
                                             # translated (bio_change, optimization_change)
changes = changes_from_form(ws.pln, form)    # what differs from the plan (all of it without a
                                             # plan or for another radiation mode)
session.set_plan(changes)                    # resolve_changes + build_plan, under hold_updates()
```

Sending only what differs keeps plan settings the form shows but cannot represent: a live solver
instance (`build_plan` no longer deep-copies it), `prop_opt` entries outside the problem form, a
plan without an explicit dose engine (shown as the mode's default), the plan's scenario model
instance, per-beam isocenters, `prop_opt` and engine entries the forms do not show (the engine
dialog leaves out `ENGINE_FORM_EXCLUDE`; `plan.set` keeps such entries of the same problem or
engine when they are not sent, or sent back unchanged). A radiation mode change keeps the
scenario model and the problem settings when they are not sent (`resolve_changes`), so the web,
which sends only its modified fields, keeps them as Qt does by sending every field. The comparison uses `comparison_key` (`_forms.py`), which the dirty
tracking (`_capture_state`) and `resolve_changes`' keep-the-scenario-instance decision use too:
tuples and lists, numpy and Python numbers, enums and their values compare alike, `NaN` stays
distinct from `None`. The old Qt keys (`json.dumps(default=str)`, plain `!=` for the engine and
problem settings) told a tuple from a list and compared numpy scalars as text. Which widget is
bordered stays Qt presentation.

The form's initial values come from `session.default_plan` (the profile's). When CT and
structures are present without a plan, the form applies itself through `set_plan` (on the
session's default plan) via `call_when_idle`, so data committed by a running load gets its plan
once the operation has ended. `auto_plan` stays off for Qt: the form applies what it shows (edits
made before the data arrived included) and is the only trigger, also for data written directly to
the workspace. An Apply while an operation runs (the form is disabled then, and the pending-changes
question is asked before a step starts) is refused by the session and shown in the status line
("Plan not applied: Calculating dose influence is still running."); nothing is written.
`test/gui/widgets/plan/test_plan_form_parity.py` compares the plan Apply writes with a frozen copy
of the form's former builder for 58 form states (known difference: a plan leaving the engine to
its mode keeps doing so); `test_plan_form_read.py` checks for 13 plans that the synchronised form
describes the plan unchanged (nothing to send, an unedited Apply keeps it) and that its
selections and option lists are those of `plan_detail` / `plan_options`.

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

**(c) Qt GUI in-process.** The GUI calls `Session` directly, no `dispatch`, no codec, and no
listeners (so no state snapshots are built). `run` executes on the GUI's `QThread` `Worker`,
`begin` / `finish` / `abort` on the GUI thread, where the workspace is written and Qt signals
fire; see [The Qt GUI on the session](#the-qt-gui-on-the-session).

## Trust model and host flags

The dispatch surface assumes **one trusted user per process**. The log capture of a running
operation is a handler on the process-wide `pyRadPlan` logger, and the settings
(`app.settings_set`), the user base data folders and the `PYRADPLAN_*` environment variables are
process-wide, so concurrent sessions of different users would see and change each other's state.
A server host therefore needs one process (or worker) per user until this changes.

`SessionManager` takes two opt-in guards, set by the host and never by a request (the `options` of
`session.create` cannot change them); the defaults keep the native behaviour:

- `allow_pickle=False` makes every session refuse to load pickles (`data.load_file`,
  `data.load_dij`, `Session.load_path`, `base_data.add` of a pickle phantom) with
  `PreconditionError(code="not_available")`; `app.info` reports it as the unavailable
  `pickle_load` feature and `pickle` import format.
- `allow_global_settings=False` refuses `app.settings_set`, `base_data.add` and
  `base_data.remove` (and with them the environment variable changes of the settings dialog).

**Pickles in the browser.** Unpickling runs arbitrary code, which in the page could read the
patient data and send it out despite the "nothing is uploaded" notice. Under Pyodide pickle
*loading* is therefore always refused (`pyRadPlan.app._io.pickle_load_reason`), reported in
`capabilities.features.pickle_load`; pickle *saving* stays available. The flags above are
guards for a single-user host, not isolation between users.

**Structures are addressed by position.** Structure names are not unique. View products (masks,
DVHs), state entries, voxel counts and the editors identify a structure by its position in the
structure set; requests may still name one while the name is unique (`resolve_structure` in
`_views.py`).

**Prepare has no effects.** `Session.prepare` validates and captures inputs; only `commit` writes
the workspace. A biological model that cannot be evaluated and is switched off with
`physical_only` yields the physical-only plan at prepare, but it is written by the commit together
with the result, so dropping the operation leaves the plan alone. The history is locked, as the Qt
host calls `history_since` from the GUI thread while the worker records.

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
threading code in the layer. `begin` / `run` / `finish` / `abort` wrap the split so that busy
state, history, report observation and events stay the session's even when the host schedules.

**Identity-based view caching.** Views are cached keyed by what they show and validated by the
identity of their source objects (CT image, result entry, structure mask), the same strategy as
`ViewingWidget`. This survives the frequent workspace writes that do not change the images
(objective edits, plan changes) and needs no version counters in the data model. It does not
notice in-place mutation of an image, which pyRadPlan code does not do to stored objects.

## Known gaps and open problems (draft quality)

- **Mock fixtures are generated.** The web frontend's mock backend replays recordings of this
  layer's catalogues (`web/src/api/mock/fixtures/`, written by `web/scripts/gen_mock_fixtures.py`).
  While the session API still changes they are generated on the fly (`npm run fixtures`, run by
  npm pre-hooks) and not committed, so building the frontend needs Python with pyRadPlan's
  dependencies. Once the API is stable, committing them together with a drift test (regenerate
  and compare in the Python test suite) lets the frontend build without Python.
- **Not run under Pyodide yet.** Everything is tested natively on CPython 3.11/Windows only.
  `tempfile`, `logging` and the synchronous design should be fine in Pyodide; the library
  underneath (SimpleITK, numba, h5py, scipy under wasm) is the porting work's concern.
- `ComputeControl` uses `threading.Event` internally (in `pyRadPlan.core`), which exists in
  Pyodide but cannot be signalled while Python runs; cancellation there is the interrupt buffer.
- tqdm console bars still print during operations (`ProgressReporter.console_progress`); a worker
  should turn them off globally. IPOPT prints its banner/iterations to stdout natively.
- The session refuses its own edits (`set_plan`, structure, tissue and objective edits,
  snapshots) while an operation runs, except the display edits (structure colour and
  visibility).
- The native imports (`import_image`, `import_binary`, `scan_dicom` / `import_dicom`) are session
  operations but not protocol methods: `data.load_file` still refuses the image kinds and there is
  no DICOM folder upload. Under Pyodide they would need SimpleITK file IO anyway.
- Qt-side logic left in `pyRadPlan/gui`: in the plan form the per-field dirty snapshot (`_capture_state`; the web's
  `captureState` is its twin), the couch / beam-count coupling, the widget-setting part of
  `_sync_from_workspace`; the Qt config forms are not rendered from `build_form` descriptors (both
  engine forms leave out `ENGINE_FORM_EXCLUDE`); the machine / engine re-selection on a mode change
  (`_refresh_machines`, `_refresh_engines`; the web's `pickMachine` / `pickIdentifier`); the AI tasks; the analysis window's QI/DVH computations; the viewer's array derivation and LPS
  to voxel mapping (`_viewer_beams`, mask/quantity caches); `_tissue_params.py` widgets (helpers
  are in `biology/_editing.py`); the VOI editor's own recreate for validation before it closes;
  the base data menu's file handling (shares `_basedata.py` summaries).
- `PlanSummary` covers only the fields the protocol names; objectives, scenario models, tissue
  parameters, engine and solver options have no methods yet (`cst` editing is the obvious next
  one).
- No `Session` persistence beyond the workspace pickle, no per-session locking beyond the
  history (a second thread calling the same session while busy gets
  `PreconditionError(code="busy")` only for steps; views are not guarded).
- Commit does not re-check that the inputs captured at prepare time are still current (the Qt
  widget does not either).
- Volumes are float32; a CT could go as int16 to halve transfers. Masks are one uint8 volume per
  structure; a bit-packed or label-map encoding would be much smaller for many structures.
- `view.volume` recomputes the value range per call (cheap but not cached).
