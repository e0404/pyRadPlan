# pyRadPlan application protocol (draft v0)

Contract between a frontend (web UI, later also the Qt GUI) and the Qt-free application
layer `pyRadPlan.app`. This file is the single source of truth while the prototype is built;
if the implementation has to deviate, change this file in the same edit and list the change under
"Amendments" at the bottom.

## Layers

```
React UI  <-- typed client (web/src/api) -->  Web Worker  <-- dispatch(dict) -->  pyRadPlan.app  -->  pyRadPlan
Qt GUI    <-- direct Python calls ------------------------------------------->  pyRadPlan.app  -->  pyRadPlan
```

`pyRadPlan.app` must not import Qt, must not require threads, and must run under Pyodide.

## Python side (`pyRadPlan/app/`)

- `Workspace`: observable store of the six pipeline objects `ct, cst, pln, stf, dij, result`
  (same keys and semantics as `pyRadPlan.gui.workspace.WorkspaceManager`, but plain Python with
  `subscribe(callback) -> unsubscribe`). `result` is the dict the Qt GUI already stores.
- `Session`: one planning session. Owns a `Workspace`, runs workflow steps, tracks stage status
  (missing / done / stale), exposes JSON-able state and view data, emits events.
- `SessionManager`: creates / looks up / closes sessions by id.
- `dispatch(request, emit=None) -> response`: transport-agnostic entry point (below).

## Request / response

Request (plain dict / JSON):

```json
{"id": 7, "session": "s1", "method": "workflow.optimize", "params": {}}
```

Response:

```json
{"id": 7, "ok": true, "result": {...}, "buffers": [<bytes-like>, ...]}
{"id": 7, "ok": false, "error": {"type": "ValueError", "message": "...", "traceback": "..."}}
```

Binary data never travels inside JSON. An array is represented in `result` as

```json
{"$array": 0, "dtype": "float32", "shape": [nz, ny, nx]}
```

where `$array` indexes `buffers`. Buffers are C-contiguous, little-endian. In the browser they are
transferred as `ArrayBuffer`s; a later server transport can send them as binary frames.

Events are pushed through the `emit` callback while a request runs (and are also what a server
transport would push over a socket):

```json
{"event": "state", "session": "s1", "state": {...SessionState}}
{"event": "progress", "session": "s1", "label": "Dose calculation", "fraction": 0.42, "detail": "beam 2/5"}
{"event": "log", "session": "s1", "level": "info", "message": "..."}
```

`fraction` is `null` for indeterminate progress.

## SessionState

```json
{
  "session": "s1",
  "has": {"ct": true, "cst": true, "pln": true, "stf": false, "dij": false, "result": false},
  "stages": {"data": "done", "plan": "done", "stf": "missing", "dij": "missing", "optimization": "missing"},
  "actions": {"generate_stf": true, "calc_dose_influence": false, "optimize": false, "recalc": false},
  "ct": {"shape": [nz, ny, nx], "spacing": [sx, sy, sz], "origin": [x, y, z], "num_scenarios": 1},
  "plan": {...PlanSummary} ,
  "quantities": ["physical_dose"],
  "structures": [{"name": "Core", "type": "OAR", "color": [r, g, b], "visible": true, "index": 0}],
  "status": "Ready"
}
```

Stage values: `"missing" | "done" | "stale"`. A stage is stale when an upstream object changed
after it was computed (same rule as the Qt `WorkflowWidget`).

## Methods (v0)

| Method | Params | Result |
|---|---|---|
| `app.info` | – | `{version, python, backend, phantoms: [...], machines: {mode: [names]}, solvers: [...], engines: {...}}` |
| `session.create` | `{}` | `{session}` |
| `session.close` | – | `{}` |
| `session.state` | – | `SessionState` |
| `data.load_phantom` | `{name}` | `SessionState` |
| `data.load_file` | `{filename, data: {$array}}` (raw file bytes, uint8) | `SessionState` |
| `plan.get` | – | `PlanSummary` |
| `plan.set` | partial `PlanSummary` | `SessionState` |
| `workflow.generate_stf` | – | `SessionState` |
| `workflow.calc_dose_influence` | – | `SessionState` |
| `workflow.optimize` | `{solver?, max_iter?}` | `SessionState` |
| `workflow.run_all` | – | `SessionState` |
| `view.volume` | `{name}` with `name = "ct"` or a quantity from `quantities` | `Volume` |
| `view.structure_masks` | `{names?}` | `{grid, masks: [{name, color, data: {$array uint8}}]}` |
| `analysis.dvh` | `{quantity?}` | `{quantity, unit, curves: [{name, color, dose: {$array}, volume: {$array}}]}` |

`PlanSummary`: `{radiation_mode, machine, num_of_fractions, gantry_angles: [..], couch_angles: [..],
bixel_width, iso_center: [x,y,z] | null, dose_grid_resolution: [x,y,z] | null, engine: str | null}`.

`Volume`:

```json
{
  "name": "physical_dose", "unit": "Gy",
  "data": {"$array": 0, "dtype": "float32", "shape": [nz, ny, nx]},
  "grid": {"shape": [nz, ny, nx], "spacing": [sx, sy, sz], "origin": [x, y, z],
           "direction": [9 floats, row-major]},
  "range": [min, max]
}
```

Array axis order is `(z, y, x)` (the order of `sitk.GetArrayFromImage`); `spacing` / `origin` are
`(x, y, z)` in mm, LPS. Result quantities are delivered resampled on the CT grid so every volume of
a session shares one grid; the frontend never resamples.

## Amendments

All additive unless marked **changed**. Implemented in `pyRadPlan/app/` (see
`docs/development/app_layer.md`).

1. Requests may carry `"buffers": [...]` like responses; `{$array}` descriptors in `params` index
   them (needed by `data.load_file`). A descriptor without `shape` decodes to a flat array.
2. New method `workflow.recalc` `{physical_only?}` -> `SessionState` (forward dose of the current
   weights with the current plan; `actions.recalc` already existed).
3. `workflow.calc_dose_influence`, `workflow.recalc` and `workflow.run_all` accept
   `physical_only?: bool` (default `false`). If the plan's biological model cannot be evaluated
   for the structures, the request fails with `PreconditionError`; with `physical_only: true` the
   plan's model is switched to `none` (as the Qt dialog offers) and the step runs.
4. `workflow.run_all` takes `{solver?, max_iter?, physical_only?}`, applies the default plan if
   there is none, and runs only the steps that are missing or stale.
5. **changed** `actions.calc_dose_influence` is `true` whenever CT, structures and plan exist
   (as the Qt button): it generates the steering information itself (since v1 amendment 20
   always, before that when it was missing or stale). The SessionState example above showing
   `false` there is outdated.
6. `SessionState` adds `busy: string | null` (label of the running operation; `status` then equals
   it) and `ct.direction` (9 floats). `ct` / `plan` are `null` without CT / plan.
7. `PlanSummary` adds `bio_model: string | null`. `iso_center` is `null` (automatic: target center
   of mass), `[x, y, z]`, or a list of `[x, y, z]` per beam. `plan.set`: omitted `couch_angles`
   with a new gantry list become `0` per beam, a single couch angle applies to every beam;
   changing `radiation_mode` resets `machine`, `bio_model` and `engine` to that mode's defaults
   unless given; unknown fields fail with `ValueError`. `plan.get` returns `null` without a plan.
8. Loading CT + structures without a plan applies the default plan (photons, `Generic`, 30 fx,
   gantry `[0]`, bixel width 5 mm, dose grid 3 mm: the Qt plan form defaults).
9. `workflow.optimize` returns `SessionState` plus `optimization: {solver, num_iter, objective}`.
   Without a `solver` param or a solver in the plan's `prop_opt`, the first registered of
   `ipopt`, `scipy` is used (under Pyodide: `scipy`).
10. `progress` events may carry `data` (JSON numbers from the optimizer, e.g. `iteration`,
    `objective`); during optimization `fraction` is `iteration / max_iter` when `max_iter` is known,
    else `null`. A successful operation ends with a `fraction: 1.0` event.
11. `log` events add `logger`. Warnings of the algorithms and `pyRadPlan` log records (warning
    and above unless the session lowers the level) become `log` events.
12. `view.volume` accepts `phase?` (4D CT phase, default 0); `range` entries are `null` when the
    volume has no finite values. CT volumes are float32 HU, `unit: "HU"`.
13. `analysis.dvh` takes `{quantity?, names?, num_points? = 500}`; `dose` is the left bin edges and
    `volume` the cumulative volume in % (both float32), on one dose axis `0 .. 1.05 * max` shared
    by all curves; each curve adds `stats: {mean, min, max}` (`null` for an empty structure).
14. `app.info` adds `platform` and `methods`; `engines` is `{mode: [short names]}`.
15. `error.traceback` is optional. Error `type`s in use: `PreconditionError` (step not possible in
    this state), `UnknownMethodError`, `UnknownSessionError`, `TypeError` (bad params),
    `KeyError` (unknown quantity / structure), `LookupError` (no CT / structures loaded),
    `ValueError`, `FileNotFoundError`, `KeyboardInterrupt` (Pyodide interrupt buffer),
    `ComputeCancelledError`. A `PreconditionError` adds `error.code`: `not_available`, `busy`,
    `no_data`, `bio_model_unsupported` (retry with `physical_only: true`), `invalid_reference`,
    `biological_objectives`, `no_weights`, `beam_geometry_changed`.
16. `session.create` emits an initial `state` event. In Python, `buffers` are `memoryview`s
    (buffer protocol); a worker copies or transfers them (e.g. `PyProxy.getBuffer()` / `toJs()`).

# v1 (full Qt parity round; draft, not implemented yet)

Scope and per-element targets: `web/docs/parity.md`. v1 only **adds** methods, optional params and
fields; every v0 request keeps its v0 result. A v0 client works unchanged against a v1 backend.

## Conventions

**Cost classes.** Every method below is one of:

- **read**: cheap, no workspace write, no progress events. May run whenever the worker is free.
- **write**: cheap, writes the workspace (emits `state`), refused with
  `PreconditionError(code="busy")` while a long operation runs (natively/server; in Pyodide requests
  are serialized anyway).
- **long**: sets `state.busy`, emits `progress` events, is stoppable (Pyodide interrupt buffer,
  natively `session.cancel`), and refuses to start while busy (`code="busy"`). A stopped
  `workflow.optimize` ends after its current iteration and succeeds with its last iterate
  (`session.history` outcome `"stopped"`); under Pyodide a second interrupt discards it with
  `KeyboardInterrupt`. Other long methods fail when stopped.

"May load base data" marks reads whose first call per machine reads machine base data (seconds);
the result is cached per machine file.

**Validation errors.** A rejected value fails with `error.type = "ValidationError"`,
`error.code = "invalid"` and `error.details = [{field, message}]`, where `field` is the dot path of
the offending descriptor (`"options.max_iter"`, `"parameters.d_ref"`). Nothing is written when a
request fails.

**Display names.** Wherever a `Choice` is sent, `label` is the Qt display text (`cls.name`,
`RTQuantity.name`, bio option "formalism · provenance") and `identifier` the Python identifier the
Qt GUI shows faintly in open popups.

**Locked fields.** Plan fields listed in `state.options.locked_fields` are sent with
`editable: false` and rejected by `plan.set` with `PreconditionError(code="locked")`.

## Shared types

```ts
type Choice = {
  value: string | number | boolean | null;   // what is sent back
  label: string;                             // Qt display text
  identifier?: string;                       // Python identifier (popup hint)
  enabled?: boolean;                         // default true
  reason?: string;                           // why disabled
  group?: string;                            // section header in the list
  description?: string;                      // tooltip
};

type FieldDescriptor = {
  name: string;            // key in `values`; dot path inside groups ("setup.sd")
  label: string;           // AlgorithmParameterMetadata.label, else name.replace("_"," ").capitalize()
  type: "bool" | "int" | "float" | "str" | "choice" | "number_list" | "json" | "group"
      | "color" | "image_ref";
  default?: unknown;       // JSON value of the model default
  nullable?: boolean;      // Optional[...]: null allowed ("None" / indeterminate checkbox)
  editable?: boolean;      // default true
  reason?: string;         // why not editable (locked, runtime, Qt shows it read-only)
  advanced?: boolean;      // locked in the educational UI mode
  description?: string;    // pydantic Field description (tooltip)
  unit?: string;           // display unit, e.g. "mm", "Gy", "%"
  unit_from?: "quantity";  // unit is that of the objective's quantity (objective parameters)
  min?: number; max?: number;                     // ge / le
  exclusive_min?: number; exclusive_max?: number; // gt / lt
  step?: number; decimals?: number;
  choices?: Choice[];      // type "choice" and "image_ref"
  editable_choice?: boolean; // choice that also accepts free text (Union[Literal, str])
  item_type?: "int" | "float"; length?: number;  // type "number_list" (length for tuples)
  fields?: FieldDescriptor[]; // type "group": nested model/dataclass, shown collapsed
};

type FormSchema = {
  id: string;              // e.g. "engine:SVDPB", "solver:scipy", "scenario:wcScen"
  title: string;           // Qt dialog title, e.g. "Configure SciPy minimize"
  fields: FieldDescriptor[];
  values: Record<string, unknown>; // current values; groups as nested override dicts
  explicit: string[];      // names whose values are set (reported back even if unchanged)
  notes?: string[];        // e.g. "Solver 'ipopt' is unavailable: …"
};

type FileResult = {        // every save returns one file; folders and several files are zipped
  filename: string;        // suggested download name
  mime: string;            // "application/x-matlab-data" | "application/octet-stream" | "application/zip"
  data: ArrayRef;          // {$array, dtype: "uint8"}
  files?: string[];        // archive members when mime is application/zip
};

type Issue = {             // pyRadPlan.biology.models.TissueIssue
  structure: string | null; kind: string; severity: "error" | "warning";
  message: string; closest_class?: string | null;
};
```

Type mapping from the pydantic / config models (as `gui/widgets/_config_form.py` does, with the Qt
gaps closed): `bool` → `bool`; `int` / `float` → `int` / `float` with bounds from `ge/le/gt/lt`;
`Literal[...]` → `choice`; `Union[Literal[...], str]` → `choice` + `editable_choice`;
`Union[Literal[...], bool]` → `choice` with the literals plus `true` / `false` (Qt cannot set
these); `str` → `str`; `list` / `tuple` of numbers → `number_list`; nested dataclass / pydantic
value → `group`; `Optional[X]` → `X` + `nullable`; everything else (dict, TypedDict, Path, Any, lists
of models) → `json`. `AlgorithmParameterMetadata(configurable=False)` fields and `hidden_entries`
are omitted. The engine form omits the inherited `DoseEngineBase` fields the plan sets elsewhere
(`mult_scen`, `bio_model`, `dose_grid`, `voxel_sub_ix`, `select_voxels_in_scenarios`).

Values sent back: only names in `explicit` or edited, groups as nested override dicts, exactly the
semantics of `ConfigFormWidget.values()`.

## SessionState v1 additions

```ts
revisions: {
  workspace: {ct: number, cst: number, pln: number, stf: number, dij: number, result: number}; // +1 per write of the key
  volumes: Record<string, number>; // per volume key ("ct" and every component key): +1 when its source object changes
  masks: number;      // +1 when a mask, the structure name list or the CT changes
  beams: number;      // +1 when pln, stf or cst (automatic isocenter) changes
  base_data: number;  // process-wide, +1 per base_data.add / remove
};
results: {
  groups: [{
    base: string; tag: string;           // tag "" = current result
    label: string;                       // selector label: RTQuantity.name (+ " (<tag>)")
    meta_label: string; unit: string;    // colorbar / save-dialog label and unit (QUANTITY_META)
    components: [{key: string; label: string; kind: "full" | "beam" | "scenario" | "other"}];
  }];
  scenario_labels: string[];   // result["scenario_labels"], else dij.scenario_labels()
  default: string | null;      // rbe_x_dose > physical_dose > first (Qt default_quantity_key)
  tags: string[];              // saved snapshot tags
  default_tag: string;         // "<radiation_mode>_<n>fx"
};
options: {profile: "native" | "web"; default_plan: Partial<PlanSummary>; locked_fields: string[]};
actions: {…v0, snapshot: boolean, load_dij: boolean, save_pln: boolean, save_dij: boolean,
          save_cst: boolean, save_result: boolean, save_workspace: boolean};
```

Component keys (accepted by `view.volume` and `analysis.*`): `"<identifier>"`,
`"<identifier>_beam <i>"`, `"<identifier>_scenario <i>"` (flat scenario index), each optionally
tagged before the index: `"physical_dose_<tag>"`, `"physical_dose_beam_<tag> 1"`. Grouping follows
`gui/widgets/_display_names.py:group_result_keys`. v0 `quantities` stays the list of full keys.

`structures[]` (v0) adds `overlap_priority`, `num_objectives`, `num_voxels`.

## Events v1

- `log` events add `seq` (per-session counter) so `session.history` can resume.
- `progress` is unchanged; optimizer metrics stay in `progress.data` (`iteration`, `objective`,
  and for ipax `constraint_violation`, `dual_inf`, `step`, `kkt_error`, `mu`). The optimization
  status window is fed from these events.

## Methods v1

### App

| Method | Cost | Params | Result |
|---|---|---|---|
| `app.info` (extended) | read | – | v0 fields + `runtime: "pyodide" \| "native"`, `capabilities` (below), `quantities: [{identifier, name, meta_label, unit}]`, `default_profile` |
| `app.licenses` | read | `{name?}` | without `name`: `[{name, version, license}]` of the installed distributions; with `name`: `{name, version, license, text \| null}` |
| `app.settings` | read | – | `{sections: [{id: "general" \| "xp" \| "ai" \| "gui", title, form: FormSchema}]}` (fields of `pyRadPlan.settings`; non-applicable ones `editable: false` with reason) |
| `app.settings_set` | write | `{section, values}` | as `app.settings`; writes the settings singleton and the `PYRADPLAN_*` env vars like the Qt dialog; refused for non-editable fields |

`capabilities`:

```ts
{
  features: Record<FeatureId, {available: boolean; reason: string | null}>;
  // FeatureId: "image_file_io" | "dicom_import" | "dicom_export" | "ai" | "ipopt" | "gpu" | "numba"
  //          | "monte_carlo" | "pause" | "open_folder" | "host_paths" | "base_data_persistence"
  formats: {
    import: [{format, label, extensions: string[], objects: string[], available, reason}];
    export: [{object: "pln" | "dij" | "cst" | "result" | "workspace", format, label,
              extension, mime, available, reason, keeps_objectives?: boolean,
              archive: boolean, note?: string}];
  };
}
```

Detection: `runtime` from `sys.platform == "emscripten"`; image IO and DICOM import from
`SimpleITK.IS_PYRADPLAN_SHIM`; `ai` from `importlib.util.find_spec("pydantic_ai")` plus the
`ai_disabled_reason` logic; `ipopt` from `optimization.solvers.IPOPT_DISABLED_REASON`;
`monte_carlo` and `pause` false under Pyodide; `dicom_export` from `find_spec("pydicom")` and
`find_spec("contourpy")`. The web runtime's export table (verified natively with the shim):

| object | formats available in the browser | unavailable (reason) |
|---|---|---|
| pln | mat, pickle | – |
| dij | mat, pickle | – |
| cst | mat (needs a CT on reload), pickle, npz (with CT), DICOM RTSTRUCT / SEG (zip, with CT series) | NIfTI / NRRD / MetaImage label maps (SimpleITK WriteImage) |
| result | npz, mat (`resultGUI`), DICOM RTDOSE (zip); several quantities zipped | NIfTI / NRRD / MetaImage (SimpleITK WriteImage) |
| workspace | pickle (`note`: not interchangeable with native pickles until the shim pickles like SimpleITK) | – |

### Session

| Method | Cost | Params | Result |
|---|---|---|---|
| `session.create` (extended) | write | `{profile?: "native" \| "web", options?: {default_plan?, locked_fields?, auto_plan?, solver_preference?, max_iter?, log_level?}}` | `{session, options}` |
| `session.history` | read | `{since?: number}` | `{log: LogEvent[] (seq > since, at most 2000), optimization: {label, points: [{iteration, objective, …}], finished: boolean, outcome: "done" \| "stopped" \| "failed" \| null} \| null}` |
| `session.log_level` | write | `{level: "debug" \| "info" \| "warning" \| "error"}` | `{level}`: level of the `pyRadPlan` logger while operations run (the Qt log combo) |
| `session.pause` / `session.resume` | read | – | `{}`; native/server only (needs a second thread); Pyodide: `PreconditionError(code="not_available")` |
| `session.cancel` | read | – | `{}`; native/server cooperative stop (`ComputeControl.request_stop`); Pyodide uses the interrupt buffer instead |

Profiles: `"native"` uses `DEFAULT_PLAN` (3 mm dose grid); `"web"` uses `DEFAULT_PLAN` with
`dose_grid_resolution: [5, 5, 5]` (owner decision) and `locked_fields: []` (the owner can lock
`dose_grid_resolution` later by changing the profile; the frontend follows the descriptors). The
default profile is `"web"` under Pyodide, else `"native"`. Open: the Qt form's `dose_convention`
default is `"total"` (`parity.md`, ambiguity 1); the web profile adopts it once the owner confirms.

### Data and files

| Method | Cost | Params | Result |
|---|---|---|---|
| `data.load_file` (extended) | long | `{filename, data, kind?: "auto"}` | `SessionState`. Accepts `.mat`, `.npz`, `.pkl`, `.pickle`; restores snapshot tags from a workspace pickle (`saved_result_tags`). The `kind`s `"ct_new"`, `"ct_replace"`, `"structures"`, `"dose"` are reserved for image files and fail with `PreconditionError(code="not_available")` without image IO |
| `data.load_dij` | long | `{filename, data}` | `SessionState`; merges `stf` + `dij` from the file; `PreconditionError(code="no_dij")` when it holds none |
| `export.save` | long | `{object, format, quantities?: string[], structure_format?: "rtstruct" \| "seg", basename?}` | `FileResult`. `quantities` (component keys) for `object: "result"`, default all full quantities. Workspace pickles include `saved_result_tags` |

Pending plan-form edits are resolved by the frontend before calling `export.save` (Qt asks for
Save Plan and Save Workspace).

### Base data

| Method | Cost | Params | Result |
|---|---|---|---|
| `base_data.list` | read | `{kind: "machines" \| "rbe_tables" \| "stopping_power_tables" \| "phantoms"}` | `{kind, folders: [{source: "bundled" \| "user" \| "shared", label, path}], rows: [{name, file, source, removable, radiation_mode?, model?, tissue_classes?, reference?, problem?}]}` (columns of the Qt Base Data dialog; `problem` = why a file is unreadable) |
| `base_data.add` | write (validation may take seconds) | `{kind, filename, data, name?, radiation_mode?, overwrite?: boolean, dry_run?: boolean}` | `{row, suggested_name, radiation_mode, clash: {file, source} \| null}`; `dry_run` validates and reports without writing; a clash without `overwrite` fails with `PreconditionError(code="name_clash")` |
| `base_data.remove` | write | `{kind, file}` | `{}`; user files only, else `PreconditionError(code="not_removable")` |

### Plan

| Method | Cost | Params | Result |
|---|---|---|---|
| `plan.get` (extended) | read | `{detail?: boolean}` | without `detail`: v0 `PlanSummary`; with: `PlanDetail` |
| `plan.options` | read, may load base data | `{radiation_mode?, machine?, engine?, current_bio?}` (default: the plan's, else the default plan's; `current_bio`: amendment 22) | `PlanOptions` (+ `bio_selection`, amendment 22) |
| `plan.form` | read | `{kind: "engine" \| "bio_model" \| "scenario" \| "problem" \| "solver", name, radiation_mode?, values?}` | `FormSchema`; `values` default to the plan's (engine: `prop_dose_calc` minus `engine` / `dose_grid`; problem: `prop_opt` minus `problem` / `solver`; `max_iter` / `display` are forwarded into the solver form as the Qt dialog does) |
| `plan.set` (extended) | write | partial `PlanSummary` plus the v1 fields below | `SessionState` |
| `plan.tissue_check` | read, may load base data | `{draft?: {radiation_mode, machine, engine, bio_model}}` (the unapplied form; default: the plan) | `{marker: "error" \| "warning" \| "", issues: Issue[]}` (the ⚠ / ℹ marker) |

```ts
type PlanDetail = PlanSummary & {
  dose_convention: "per_fraction" | "total";
  prescribed_dose: number;                       // read-only (not shown in Qt, carried over)
  machine_source: "name" | "plan";               // "plan": the plan carries loaded base data ("(from plan)")
  resolved_iso_center: Vec3 | null;              // target centre of mass when automatic
  engine_config: {name: string | null; values: Record<string, unknown>};
  bio: {model: string; option: string | null; values: Record<string, unknown>};
  scenario: {model: string; values: Record<string, unknown>};
  problem: {name: string; values: Record<string, unknown>};
  solver: {name: string | null; values: Record<string, unknown>};
};

type PlanOptions = {
  radiation_modes: Choice[];
  machines: Choice[];                            // + "(from plan)" entry when machine_source is "plan"
  engines: Choice[];                             // MC engines disabled with reason in the browser
  bio_models: (Choice & {model: string; parameters: Record<string, unknown>})[]; // BioModelOption; value = option id
  default_bio_model: string;                     // option id
  scenario_models: Choice[];                     // "Explicit Scenarios" only while the plan holds one
  problems: Choice[];
  solvers: (Choice & {gpu_compatible: boolean})[]; // an unavailable plan solver as disabled entry
  default_solver: string;
  beam_presets: {name: string; label: string; gantry_angles: number[]; couch_angles: number[];
                 description: string; typical: boolean}[];
  fields: FieldDescriptor[];                     // the fixed plan fields: bounds, defaults, advanced, editable
};
```

`plan.set` v1 fields (optional, combinable with the v0 fields): `dose_convention`; `machine`;
`engine_config: {values}` (replaces the engine-specific entries of `prop_dose_calc`);
`bio_model: string | {model, option?, values?}` (the v0 string is still accepted; `option`
selects the base-data option, `values` the model parameters); `scenarios: {model, values}`;
`problem: {name, values}`; `solver: string | {name, values}`. Semantics follow
`PlanWidget._build_plan` (see `parity.md`): `prop_opt` is rebuilt from problem + solver,
`prop_seq` and the other `prop_stf` keys survive only within the same radiation mode, an unchanged
scenario model keeps the plan's instance.

`plan.options.fields` covers, with the Qt bounds and defaults: `num_of_fractions` (1..1000,
advanced), `dose_convention` (choice, advanced), `bixel_width` (0.1..100 mm),
`dose_grid_resolution` (`number_list` length 3, 0.1..50 mm, `editable` per profile),
`gantry_angles` and `couch_angles` (`number_list`), `iso_center` (`number_list` length 3,
`nullable` = Auto), and the disabled Qt fields `optimized_quantity` and the four sequencing fields
(`editable: false`, `reason: "not yet implemented"`).

### Structures, tissues and objectives

| Method | Cost | Params | Result |
|---|---|---|---|
| `cst.get` | read | `{names?}` | `{structures: StructureDetail[]}` |
| `cst.update_structure` | write | `{name, changes: {type?, color?, visible?, overlap_priority?, tissue?: TissueDraft}, apply_anyway?: boolean}` | `{state, structure: StructureDetail}`; the VOI is recreated with `create_voi` as the Qt Edit VOI dialog does; tissue errors fail with `PreconditionError(code="tissue_errors")` and the issues in `error.details` unless `apply_anyway`. Changes of only `color` / `visible` are display settings: set in place, accepted while busy, outdating nothing (`revisions.workspace.cst` still counts them) |
| `tissue.schema` | read, may load base data | `{bio_model?, machine?, engine?}` (default: the plan's) | `{models: [{label: "Generic" \| "LEM" \| "MKM" \| "mMKM", fields: FieldDescriptor[]}], reference_radiations: Choice[], picks: [{id, kind: "base_data" \| "preset", label, group, exact: boolean, model, values, description}], option_label}` |
| `tissue.evaluate` | read, may load base data | `{rows: [{structure, tissue: TissueDraft}], op?: "none" \| "derive" \| "match" \| "relabel" \| "apply_pick", pick?: string, targets?: string[], edited?: {structure, field}}` | `{rows: [{structure, tissue: TissueDraft, issues: Issue[], comparison: {closest_class, matched, provenance_fix, parameters: [{name, declared, value, matches}]} \| null, sources: Record<string, "explicit" \| "published" \| "estimated" \| "base_data" \| "default" \| "derived">}], notes: string[], can_match: boolean, can_relabel: string[], can_derive: boolean}`; nothing is written; `edited` triggers the live re-derivation of α_x / β_x for MKM rows |
| `tissue.apply` | write | `{rows: [{structure, tissue}], apply_anyway?: boolean}` | `{state}`; same error gate as `cst.update_structure` |
| `objectives.catalog` | read | – | `{objectives: [{name, scenario_aware, parameters: FieldDescriptor[], robustness: Choice[]}], quantities: Choice[], robustness: Choice[], default_by_type: {TARGET: "Squared Deviation", "*": "Squared Overdosing"}, reference_images: Choice[]}`; parameter descriptors carry bounds, `unit` ("%" for relative volumes) or `unit_from: "quantity"` for dose references, `f_diff` as a choice, `d_ref` of Squared Mimicking as `image_ref` |
| `objectives.add` | write | `{structure}` | `{state, structure: StructureDetail}`; default type by VOI type, quantity shared by the existing objectives if they agree |
| `objectives.update` | write | `{structure, index, changes: {priority?, quantity?, robustness?, parameters?: Record<string, unknown>}}` | `{state, structure}` |
| `objectives.change_type` | write | `{structure, index, name}` | `{state, structure}`; keeps priority and an allowed robustness, resets quantity and parameters (Qt rule) |
| `objectives.remove` | write | `{structure, index}` | `{state, structure}` |
| `objectives.set_quantity_all` | write | `{quantity}` | `{state}` |

```ts
type TissueDraft = {model: "Generic" | "LEM" | "MKM" | "mMKM";
                    values: Record<string, number | string | null>; // NaN-default parameters as null
                    sources?: Record<string, string>};              // reference_sources
type ObjectiveData = {index: number; name: string; priority: number; quantity: string;
                      robustness: string; parameters: Record<string, unknown>};
// image references: {"$result": "<component key>"} or {"$custom": "<summary>"} (kept as is)
type StructureDetail = {
  name: string; index: number; type: string; color: [number, number, number]; visible: boolean;
  overlap_priority: number; num_voxels: number; num_scenarios: number;
  center_of_mass: Vec3 | null; alpha_x: number; beta_x: number;
  tissue: TissueDraft; tissue_summary: string;   // Qt VOI row tooltip text
  objectives: ObjectiveData[];
};
```

### Workflow and results

| Method | Cost | Params | Result |
|---|---|---|---|
| `workflow.snapshot` | write | `{tag?}` (default `results.default_tag`) | `{state, tag}`; tag sanitized `\W+` → `_`; `"beam"` / `"scenario"` fail with `ValueError`; reusing a tag overwrites; copies every untagged key `K` to `K_<tag>` (Qt `_on_save_result`) |

Optimization after a snapshot carries tagged keys forward and recalculation keeps them (Qt rules).

### Views

| Method | Cost | Params | Result |
|---|---|---|---|
| `view.volume` (extended) | read | `{name, phase?, encoding?: "float32" \| "auto"}` | v0 `Volume`; `name` may be any component key; `"auto"` sends the CT as int16 when all values are integral and within int16 (else float32); default `"float32"` (v0) |
| `view.structure_masks` (extended) | read | `{names?, encoding?: "uint8" \| "bits" \| "crops"}` | `"uint8"`: v0. `"bits"`: `{grid, encoding: "bits", planes: ArrayRef[], masks: [{name, color, plane, bit, num_voxels, bbox}]}`, each plane a `(z, y, x)` uint8 / uint16 / uint32 volume holding up to 32 structures, bit `bit` of plane `plane` set inside the structure. `"crops"`: `{grid, encoding: "crops", masks: [{name, color, num_voxels, bbox: [z0, y0, x0, z1, y1, x1] \| null, data: ArrayRef \| null}]}`, the uint8 crop of shape `(z1-z0, y1-y0, x1-x0)` (ends exclusive) |
| `view.beams` | read | `{draft?: {gantry_angles, couch_angles, iso_center?}}` | `{source: "stf" \| "plan" \| null, iso_centers: Vec3[], beams: [{index, name, label, gantry, couch, iso_center, source_direction, orbit_u?, orbit_w?, orbit_axis?, pending}]}` |

`view.beams` mirrors `gui/widgets/_result_widget.py:_beam_geometry` in LPS millimetres instead of
viewer voxels: `source_direction` is the unit vector isocenter → source; `orbit_u` / `orbit_w` are
the source directions at gantry 0° and 90° for the beam's couch angle and `orbit_axis = u × w`
(only for beams with plan angles, i.e. draggable ones). The stf beams are used while their angles
equal the plan's (or the draft's), else the plan (or draft) angles; with plan angles the plan
isocenters take precedence. `pending` marks draft beams whose angles differ from the applied plan.
The source distance on screen is a renderer choice (Qt: the largest circle inside the slice). The
renderer maps LPS to voxels with the CT direction matrix.

### Analysis

| Method | Cost | Params | Result |
|---|---|---|---|
| `analysis.dvh` (extended) | read | v0 params + `{keys?: string[], range?: "shared" \| "per_structure"}` | without `keys`: v0 shape + `label`, `unit_info: {symbol, dimensionless}`. With `keys`: `{sets: [{key, label, unit, unit_info, curves: DvhCurve[]}]}` (one set per key; empty structures omitted). `range: "per_structure"` bins each structure from 0 to 1.05 × its own maximum (`DVH.compute`, Qt); default `"shared"` (v0). The analysis window sends `num_points: 1000, range: "per_structure"`; "All scenarios" sends every scenario key of the primary group |
| `analysis.qi` | read | `{quantity, names?, ref_vols?: number[], ref_doses?: number[]}` | `{quantity, label, unit, unit_symbol, ref_vols, ref_doses, metrics: [{id, label, kind: "mean" \| "std" \| "max" \| "min" \| "dx" \| "vx", parameter?, unit}], rows: [{name, color, values: Record<string, number \| null>}]}`. Defaults: `DEFAULT_REF_VOLS` (2, 5, 50, 95, 98); `ref_doses` from `QICollection.default_ref_doses` over the full quantity; invalid lists fall back to the defaults (Qt); metric labels from `format_metric_label` ("D98 [Gy]") |
| `analysis.compare_units` | read | `{a, b}` (component keys) | `{shareable: boolean, scale: number \| null}` (`pyRadPlan/analysis/_units.compare_units`): shareable units share one axis scaled by `scale`, else a second x axis |

## Not in v1 (stubs; see `web/docs/parity.md`)

Image file import (Import Image, Import Binary Patient), label-map and image exports, DICOM folder
import, Monte Carlo engines, GPU / backend settings, IPOPT, AI features, pause / resume under
Pyodide, opening host folders. Their entries stay visible and disabled with the reason from
`app.info.capabilities`.

## v1 amendments

All additive unless marked **changed**; `(B)` = app, files, base data, views, analysis.

1. 2026-11-18 (B): Capability detection probes the runtime's SimpleITK: a function that exists
   but raises `NotImplementedError` (the Pyodide stand-in defines `ReadImage` / `WriteImage` /
   `ImageSeriesReader` that way) counts as missing. Natively with real SimpleITK every image
   format is `available`.
2. 2026-11-18 (B): `capabilities.formats.export` entries of the two cst DICOM formats share
   `format: "dcm"` and add `structure_format: "rtstruct" | "seg"`, which the frontend passes back
   to `export.save`. Import and export entries may carry `note`.
3. 2026-11-18 (B): `export.save` file layout. `object: "result"`: `mat` writes every selected
   quantity into one `resultGUI` (matRad names `physicalDose`, `RBExDose`, `LET`, other keys
   with non-word characters replaced by `_`), never a zip; `npz` and the image formats write one
   file per quantity, zipped when several; `dcm` is always a zip, one `RTDOSE.dcm` per quantity
   (in a folder per key when several). Default `filename`: `<object><ext>`, a single result
   quantity `<key><ext>`, several `result.zip`; `basename` replaces the stem. Missing objects
   fail with `PreconditionError(code="no_data")`, unavailable formats with `not_available`.
4. 2026-11-18 (B): `data.load_file`: a dose-only `.npz` (Save Result as npz) loads as
   `result.physical_dose` of the current patient (the npz importer alone needs a CT). With
   `kind: "auto"`, a bare image file fails with `not_available` when the runtime has no image
   IO; the reserved `kind`s fail with `not_available` in every runtime in v1.
5. 2026-11-18 (B): `analysis.dvh` sets and `analysis.qi` add `name` (the selector label,
   `"dose, beam 0"`, as in `state.results`); `label` is the axis label (`meta_label`, Qt
   `QUANTITY_META`) and `unit` its unit. The v0-shaped `analysis.dvh` adds the same `label`.
   `view.volume` sends the `QUANTITY_META` unit for component keys that are no registered
   quantity identifier.
6. 2026-11-18 (B): `view.structure_masks` `"bits"` and `"crops"` masks both carry `num_voxels`
   and `bbox`; a bit plane is uint8 for up to 8 structures, uint16 up to 16, else uint32.
7. 2026-11-18 (B): `view.beams` beams always carry `index`, `gantry`, `couch`; the orbit
   fields only with plan (or draft) angles. `iso_centers` are the plan's resolved isocenters
   (the target centre of mass when automatic), also without stf, as the Qt isocenter marker.
8. 2026-11-18 (B): `base_data.list` rows add `path`; `name` is the machine name (machines),
   the file stem (phantoms) or the file name (tables). `base_data.add` accepts
   `kind: "phantoms"` (the file must hold a CT); its `row` is `null` for `dry_run`; a clash with
   a shared file is reported but not refused (the added file shadows it); an unreadable file
   fails with `ValueError`.
9. 2026-11-18 (B): `app.settings_set` refuses non-editable fields with
   `PreconditionError(code="locked")`; rejected values fail with `code: "invalid"` and
   `details[].field` naming the field of the section form. `app.licenses {name}` with an
   unknown name fails with `KeyError`.
10. 2026-11-18 (A): Any pydantic validation error (e.g. `plan.set {num_of_fractions: 0}`) now
    also carries `code: "invalid"` and `details`; `details[].field` of algorithm forms is
    prefixed with the form kind (`"problem.p_norm"`, `"engine.kernel_cutoff"`), of objective
    edits with `parameters.` (`"parameters.v_max"`).
11. 2026-11-18 (A): Bio-model option ids (`plan.options.bio_models[].value`,
    `default_bio_model`, `PlanDetail.bio.option`) are opaque strings (JSON of the formalism and
    its base-data selection); pass them back unchanged. `plan.options` adds `radiation_mode`
    (the mode it describes). The "(from plan)" machine entry has `value: "__plan__"`; sending it
    in `plan.set` keeps the plan's loaded machine. Beam presets without couch angles get one `0`
    per beam.
12. 2026-11-18 (A): `plan.form` of a solver that is not registered returns `fields: []`,
    `values: {}` and the Qt note in `notes`; its configuration stays in the plan.
13. 2026-11-18 (A): `tissue.schema.picks[]` add `preferred` (Qt: presets of the base data's
    provenance) and have ids `"class:<i>"` / `"preset:<i>"`. `tissue.evaluate` rows return only
    `structure`, `tissue`, `issues`, `comparison`; reference-value sources are in
    `tissue.sources` (TissueDraft). `cst.update_structure` accepts `type` in any case.
14. 2026-11-18 (A): `revisions.volumes`: a key whose source object is the same as another key's
    (a snapshot copies the arrays under `<key>_<tag>`) gets the same revision, so a frontend can
    reuse what it already fetched. `revisions.base_data` is bumped by `base_data.add/remove`.
15. 2026-11-18 (A): `session.pause` / `session.resume` also fail with `not_available` when
    nothing is running. An objective parameter sent back as `{"$custom": …}` keeps its current
    value.
16. 2026-11-18 (integration): `tissue.schema` and `tissue.evaluate` accept
    `draft?: {radiation_mode?, machine?, engine?, bio_model?}` (the `plan.tissue_check` draft:
    the unapplied plan form; `bio_model` a model name or `{model, option, values}`), so the
    tissue dialog evaluates against the form's selection like Qt; explicit `bio_model` /
    `machine` / `engine` params take precedence. Draft fields missing for another radiation mode
    fall back to that mode's defaults (also in `plan.tissue_check`). `plan.options.bio_models[]`
    add `configurable: boolean` (whether the model's `plan.form` has fields; Qt's "…" button).
17. 2026-11-18 (integration): `log` events of algorithm warning reports (`WarningReport`, what
    the Qt workflow collects and shows once after a run) carry `report: true`; when the
    algorithm already logged the same message as a record during the operation, the report is
    still sent, with `repeat: true` (the log console skips it). Frontends show the warnings
    dialog from `report` events only, not from every warning record.
18. 2026-11-18 (integration): `plan.set` rejects an `engine` that is not available for the
    (new) radiation mode with `ValidationError` (`details: [{field: "engine", …}]`) instead of
    storing it; the dose calculation would otherwise fail later.
19. 2026-11-18 (scenarios): `workflow.optimize`'s `optimization.solver` (v0 amendment 9) is always
    the solver name or `null`, also when the plan configures the solver with
    `plan.set {solver: {name, values}}` (stored as `prop_opt.solver = {name, …}`); it was the
    configuration dict before. Clarification, no new field: after an optimization with a
    multi-scenario dose influence, `state.results.groups[].components` list one
    `"<identifier>_scenario <i>"` component per scenario (`kind: "scenario"`, flat index into
    `results.scenario_labels`), served by `view.volume`, `analysis.dvh {keys}` and `analysis.qi`
    like any other component key and carried into snapshots (`"<identifier>_scenario_<tag> <i>"`);
    they are not in v0 `quantities`. Verified natively and under Pyodide (`test/app/test_app_scenarios.py`,
    `e2e:pyodide` stage `robust`).
20. 2026-11-18 (stf): **changed** `workflow.calc_dose_influence` (and `workflow.run_all` when it
    calculates the dose influence) always generates the steering information again, also when
    `stages.stf` is `"done"`, as the Qt "Calc. Dose Influence" does: the automatic isocenter
    follows the structures, whose edits do not mark the stf stale. The result is unchanged
    (`SessionState`, with `stf` and `dij` written together; `revisions.workspace.stf` and
    `revisions.beams` advance). `workflow.generate_stf` stays a method of its own (beam preview
    before the dose calculation, tests). The session option `always_regenerate_stf` is gone.
21. 2026-11-18 (plan form): `plan.set` refuses a `bio_model` object with keys other than
    `model`, `option`, `values` (`ValidationError`, `details: [{field: "bio_model"}]`); before,
    a plain model specification lost its parameters silently. `problem.values` replaces the
    fields of the problem form; `prop_opt` entries the form does not show (`convert_dose_objectives`,
    entries of no problem field) are kept unless sent. A solver instance in `prop_opt` (pickled
    plans) is kept as it is. Requests of the web frontend are unaffected: it sends `{model,
    option, values}` and the problem values of `PlanDetail` (whose unshown entries are still
    validated when sent).
22. 2026-11-18 (plan form, one set of rules): `plan.options` takes
    `current_bio?: {radiation_mode, model, option: string | null, values?, label?}` (the form's
    biological model selection before a radiation mode, machine or engine change: the mode it was
    made in, the option id or `null` for the plan's model kept as unsupported entry, the
    non-selection parameters, the entry label) and returns
    `bio_selection: {option: string | null, model, replaced: boolean, unsupported: {model, label,
    description} | null, note: {text, reason} | null}`: the entry the form selects
    (`pyRadPlan.app._planform.select_bio_model`, the rules of the Qt form, which uses the same
    function). Without `current_bio` the plan's model is restored for the plan's radiation mode
    (`unsupported` is the disabled "… (not supported by …)" entry keeping a model the context
    cannot run), else the mode's default is selected. `replaced`: take over the selected option's
    base-data parameters; `note`: the status note and its tooltip when a selection the context
    cannot run was replaced. Frontends apply it and decide nothing themselves. `machine:
    "__plan__"` now lists the models of the plan's own base data (it listed the plain formalism
    list before). **changed** `plan.set`: a radiation mode change keeps the scenario model and
    the problem settings (problem, its `prop_opt` entries, solver) when they are not sent, as the
    Qt form does by sending them: the scenario model is rebuilt from its name and parameters for
    the new mode, an explicit scenario set is kept as it is (also when sent as `{model:
    "explicit"}`, which failed before); the problem is kept when the new mode offers it, otherwise
    the mode's default problem starts from its defaults. They were reset to the mode's defaults
    before. **changed** `PlanDetail.problem.values` lists only what the problem form (and, for
    `max_iter` / `display`, the solver form) shows; `PlanDetail.engine_config.values` and the
    engine form leave out `ENGINE_FORM_EXCLUDE` (`mult_scen`, `bio_model`, `dose_grid`,
    `voxel_sub_ix`, `select_voxels_in_scenarios`). `plan.set` keeps such unshown problem and engine
    entries of the plan (same problem / same engine) when they are not sent, and accepts them
    when sent back with the plan's value; a changed value is still refused as an unknown field.
    A plan carrying `convert_dose_objectives` can therefore be edited from the web.
