# Qt GUI parity inventory (web GUI round 2)

Status: specification for the "full mimic" round. Every user-visible element of the PySide6 GUI,
its web status today and the target for this round. Backend contract: `web/PROTOCOL.md` section
"v1". File references are relative to `pyRadPlan/gui/` unless they start with `pyRadPlan/`; line
numbers are from the `dev/web-pyodide` worktree (HEAD 66763c82 plus the staged prototype).

Columns:

- **Web** (state after the integration pass, 2026-11-18): `verified on Pyodide` (exercised in the
  browser against the real backend by `npm run e2e:pyodide`, or its backend call by
  `npm run app-smoke -- --strict`), `done (mock e2e)` (implemented; exercised only against the mock
  backend by `e2e`, `e2e:plan`, `e2e:viewer`, `e2e:menus` or unit tests, or frontend-only
  behaviour that does not depend on the backend; not clicked against Pyodide), `stub (as Qt; …)` (shown disabled like Qt or with the capability reason), `n/a`.
- **Target**: `impl` (implement this round), `mirror-off` (Qt itself shows it disabled or as a
  placeholder; the web shows the same), `stub: <reason>` (entry shown disabled with a tooltip giving
  the reason, which comes from `app.info.capabilities`), `n/a` (no user-visible function to mirror),
  `frontend` (no backend needed).
- **Methods**: protocol methods it needs (v0 names or v1 names from `web/PROTOCOL.md`).

## Code shared with the Qt GUI

The Qt GUI runs on the same `pyRadPlan.app` session (see `docs/development/app_layer.md`, "The Qt
GUI on the session"), so the rules behind most rows below are one implementation for both
frontends rather than a web copy of Qt code: the workflow steps, loads and the merge of loaded
data, the native imports (image file, binary folder, DICOM folder; Qt only, the web reports them
unavailable), every save (the Qt dialogs write to paths, the web downloads the same writer's
output), the format tables behind the Qt file filters and `capabilities.formats`, the objectives
table edits, structure (VOI editor, colour) and tissue writes, the default plan, and the plan form's
Apply: the Qt form sends its fields as `plan.set` params through `Session.set_plan` (only those
differing from the plan, `changes_from_form`), shows its option lists from the helpers behind
`plan.options` and compares values with the app's `comparison_key`. Which biological model entry
the plan form selects after a radiation mode, machine or engine change, or for a plan (the
"(not supported …)" entry, the replacement note), is `select_bio_model` in
`pyRadPlan/app/_planform.py` for both: Qt calls it, the web applies `plan.options.bio_selection`
(v1 amendment 22). A radiation mode change keeps the scenario model and problem settings on the
backend for both. Both engine "…" forms leave out `ENGINE_FORM_EXCLUDE` (`mult_scen`,
`bio_model`, `dose_grid`, `voxel_sub_ix`, `select_voxels_in_scenarios`); values a loaded plan
carries for them stay in the plan. What is still Qt-only logic is listed in
`docs/development/app_layer.md` ("Known gaps"): mainly the per-field dirty snapshot of the plan
form, which the web ports in `planModel.ts`.

## Runtime checks behind the stubs

What was checked, natively with the Pyodide SimpleITK shim on `sys.path`
(`web/pyodide/shims`, `IS_PYRADPLAN_SHIM`), plus the Pyodide lock file and `web/pyodide/FINDINGS.md`:

| Capability | Check | Result |
|---|---|---|
| SimpleITK image file IO | `grep sitk.ReadImage/WriteImage/ImageSeriesReader pyRadPlan/` (outside gui): `io/sitk_based/_binary_import.py:68,98,179`, `io/sitk_based/base/_serialize.py:33,38`, `io/dicom/_import_ct.py:44`, `io/dicom/_import_dose.py:114`, `ct/_ct.py:547`; shim `__init__.py` raises for anything not implemented | unavailable: NIfTI/NRRD/MHA read and write, DICOM CT/RTDOSE read. Verified: `save_data(format="nifti", dose=...)` → `NotImplementedError: SimpleITK.WriteImage`; loading an exported DICOM folder → `NotImplementedError: SimpleITK.ImageSeriesReader` |
| `.mat` (matRad) | save + `load_data` round trip with the shim | pln, dij, stf, full workspace (ct, cst, pln, stf, dij): ok. A cst-only `.mat` saves but does not load (`KeyError: 'ct'`, a library limitation, also native). A dose saved as `resultGUI` loads to nothing without its CT |
| `.npz` | same | container for ct, cst, dose only: cst (with ct) and workspace (ct, cst) ok; pln/dij/stf refused by the exporter ("Nothing to export") |
| pickle | same | everything round-trips within one runtime (ct, cst, pln, stf, dij, result, dose). **Not across runtimes**: a native pickle fails in the browser (`No module named 'SimpleITK.SimpleITK'`) and vice versa (`No module named 'SimpleITK._image'`). Needs a shim fix (see "Dependencies") |
| DICOM export | `save_data(format="dcm", cst=..., dose=...)` with the shim | writes RTSTRUCT (+ CT series) and RTDOSE natively with the shim (pydicom only). RTSTRUCT contours need `contourpy`, which is in the Pyodide lock but not in the bootstrap package list; not verified in Pyodide |
| AI (`pydantic_ai`) | Pyodide lock file and `bootstrap.js` package list; FINDINGS #2 | not installed (not in the Pyodide distribution; installation is `deps=False` from a pinned list without it); would also need provider API keys and cross-origin network calls from the browser |
| IPOPT | `optimization/solvers/__init__.py:19-37` registers it only if `ipyopt` imports; FINDINGS #4 | not registered in Pyodide; ipax and scipy are |
| GPU / array backends / numba | FINDINGS #5, #7 | numpy only; numba missing; `xp.*` settings have no effect |
| Monte Carlo engines | FRED/TOPAS (`dose/engines/_fredmc.py`, `_topasmc.py`) run external executables through `subprocess` | Emscripten has no processes; the engines are registered but cannot run |
| Pause/resume | `core/_progress.py:152-196`: pause blocks in `wait_if_paused` until another thread resumes | Pyodide runs requests synchronously in one thread; no request can arrive to resume. Stop works through the interrupt buffer (`web/docs/frontend.md`) |
| Host file browser / data folder paths | `menus/_base_data.py:114-117` (`QDesktopServices.openUrl`), settings `base_data_path` | no host file system in the browser (MEMFS); base data added in the browser lives until the page is closed |

## Main window and layout (`windows/_main_win.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Window title "pyRadPlan (NOT FOR CLINICAL USE!)" | `_main_win.py:77` | verified on Pyodide | frontend | – |
| Window icon (skull SVG) | `_main_win.py:78` | done (mock e2e) | frontend | – |
| Startup size / maximized | `_main_win.py:287-290` | n/a | n/a | – |
| Compact font ×0.8 | `_main_win.py:124-139` | done (mock e2e) | frontend | – |
| Three-column splitter (50/37/13, min widths 330/240) | `_main_win.py:210-266` | verified on Pyodide | frontend | – |
| Left: Workflow group | `_main_win.py:240` | verified on Pyodide | impl (see Workflow) | – |
| Left: Plan group | `_main_win.py:241` | verified on Pyodide | impl (see Plan) | – |
| Left: "Objectives & constraints" group | `_main_win.py:242` | verified on Pyodide | impl (see Objectives) | – |
| Left: Log group | `_main_win.py:244` | verified on Pyodide | impl (see Log) | – |
| Centre: Logo banner | `_main_win.py:251` | verified on Pyodide | frontend | – |
| Centre: Slice Viewer group | `_main_win.py:254` | verified on Pyodide | impl (see Viewer) | – |
| Centre: Visualization group | `_main_win.py:255` | verified on Pyodide | impl | – |
| Right: Viewer Options | `_main_win.py:261` | verified on Pyodide | impl | – |
| Right: Structure Visibility (VOIs widget) | `_main_win.py:262` | verified on Pyodide | impl | – |
| Right: Info | `_main_win.py:263` | verified on Pyodide | frontend | `app.info` |
| Status bar: "<Widget>: <message>" for 10 s on a widget update failure | `_main_win.py:110-119` | done (mock e2e) | frontend | – |
| Busy lock: computation disables Plan, Objectives and the whole File menu (Base Data / Settings stay enabled) | `_main_win.py:107,146-150` | done (mock e2e) | frontend | `state.busy` |
| Close while busy: "Computation running … Quit anyway?" | `_main_win.py:152-174` | done (mock e2e) | frontend (`beforeunload` prompt while busy) | `state.busy` |
| Beam drag routed into the plan form (beam editor), edited angles drawn as pending | `_main_win.py:99-101` | done (mock e2e) | impl | `view.beams` |
| Pending plan edits offered before Save Plan / Save Workspace / steps (`add_pending_changes_source`) | `_main_win.py:102-104`, `widgets/workflow/_workflow_widget.py:1043-1101` | done (mock e2e) | frontend (dialog "Apply and continue / Continue without applying / Cancel", then `plan.set`) | `plan.set` |
| Toolbar | none exists (stale docstring `_main_win.py:10-11`) | n/a | n/a | – |
| Keyboard shortcuts on actions | none (mnemonics only) | n/a | n/a | – |

## File menu (`menus/_file.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Load File… (any supported file; container formats load directly) | `_file.py:66-67` → `_workflow_widget.py:805-819` | verified on Pyodide | impl: accept `.mat`, `.npz`, `.pkl/.pickle`; file filter from capabilities | `data.load_file`, `app.info.capabilities.formats` |
| Load File… on a bare image (`.nii/.nrrd/.mha/.mhd`) → Import Image dialog | `_workflow_widget.py:816-818` | stub (as Qt; reason from the real capabilities) | stub: needs SimpleITK file IO | – |
| Load Folder… (DICOM or binary CT + masks) | `_file.py:69-74` → `_workflow_widget.py:908-925` | stub (as Qt; reason from the real capabilities) | stub: DICOM CT/RTDOSE and image files need SimpleITK file IO | – |
| Open Example Phantom ▸ (grouped by source, tooltips "Shipped with pyRadPlan" / path) | `_file.py:76-80,130-146` | verified on Pyodide | impl: group by source from `base_data.list {kind:"phantoms"}` | `base_data.list`, `data.load_phantom` |
| Open Example Phantom ▸ Open Phantom Folder | `_file.py:148-151` | stub (as Qt; reason from the real capabilities) | stub: no host file browser | – |
| Save Plan… (.mat; pending-edit check) | `_file.py:83-84` → `_workflow_widget.py:1515-1534` | verified on Pyodide | impl (mat, pickle) | `export.save {object:"pln"}` |
| Save Dij… (.mat) | `_file.py:85-86` → `_workflow_widget.py:1536-1537` | verified on Pyodide | impl (mat, pickle) | `export.save {object:"dij"}` |
| Save CST… format choice + "cannot store objectives" warning | `_file.py:87-89` → `_workflow_widget.py:1540-1616` | verified on Pyodide | impl: mat, pickle, npz, DICOM RTSTRUCT / SEG (zip); label maps NIfTI/NRRD/MetaImage: stub (SimpleITK WriteImage) | `export.save {object:"cst"}`, `app.info.capabilities.formats` |
| Save Result… (Save Result dialog, then image files) | `_file.py:90-91` → `_workflow_widget.py:1618-1701` | verified on Pyodide | impl with the browser-capable formats (npz, mat `resultGUI`, DICOM RTDOSE; one file or zip); NIfTI/NRRD/MetaImage: stub (WriteImage) | `export.save {object:"result"}` |
| Save Workspace… (pickle incl. all results and the active name; requires CT; pending-edit check) | `_file.py:94-95` → `_workflow_widget.py:1489-1513` | verified on Pyodide | impl (pickle; cross-runtime compatibility depends on the shim fix) | `export.save {object:"workspace"}` |
| UI Mode ▸ Default / Educational (exclusive, tooltips) | `_file.py:98-110,159-161` | verified on Pyodide | impl (frontend setting; plan form locks `advanced` fields) | `app.settings` (optional), descriptors `advanced` |
| Exit | `_file.py:113-114` | n/a | n/a (closing the tab) | – |
| Enable rules (pln / dij / cst / result / ct present) | `_file.py:122-128` | done (mock e2e) | impl | `state.actions` (v1 entries `save_*`) |

## View menu (`menus/_view.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Log Panel (checkable, on) | `_main_win.py:183-187` | verified on Pyodide | done | – |
| Screenshot, Zoom In, Zoom Out, Pan, Data Cursor, Toggle Legend, Toggle Colorbar (all disabled) | `_view.py:18-25,51-54` | stub (as Qt; reason from the real capabilities) | mirror-off | – |
| Dark Mode (checkable, follows platform until clicked, not persisted) | `_view.py:56-64,174-240` | verified on Pyodide | frontend | – |

## Base Data menu (`menus/_base_data.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Show Available Base Data… → dialog "Base Data" with tabs | `_base_data.py:398-403,339-376` | verified on Pyodide | impl | `base_data.list` |
| Tab Machines: Radiation mode / Name / Source / File | `_base_data.py:95-111` | verified on Pyodide | impl | `base_data.list {kind:"machines"}` |
| Tab RBE Tables: File / Model / Tissue classes / Source / Reference (unreadable rows orange) | `_base_data.py:60-78` | done (mock e2e) | impl | `base_data.list {kind:"rbe_tables"}` |
| Tab Stopping Power Tables: File / Source | `_base_data.py:95-111` | done (mock e2e) | impl | `base_data.list {kind:"stopping_power_tables"}` |
| Tab folder lines (shipped / added by you / shared) | `_base_data.py:244-254` | verified on Pyodide | impl (shows MEMFS paths; note "added files last until the page is closed") | `base_data.list` |
| Tab Add… | `_base_data.py:268-269,310-314` | verified on Pyodide | impl (session-lifetime) | `base_data.add` |
| Tab Remove (user files only, confirm) | `_base_data.py:306-308,316-336` | verified on Pyodide | impl | `base_data.remove` |
| Tab Open folder | `_base_data.py:276-280` | stub (as Qt; reason from the real capabilities) | stub: no host file browser | – |
| Add Machine File… (validate, name, clash dialogs, copy as `<mode>_<name>.mat`) | `_base_data.py:406-410,120-211` | verified on Pyodide | impl (`dry_run` for the name/clash dialog) | `base_data.add {kind:"machines"}` |
| Add RBE Table… | `_base_data.py:411-415,213-231` | done (mock e2e) | impl | `base_data.add {kind:"rbe_tables"}` |
| Open Data Folder | `_base_data.py:418-422` | stub (as Qt; reason from the real capabilities) | stub: no host file browser | – |
| `base_data_changed` → plan widget refreshes machines | `_main_win.py:189` | done (mock e2e) | impl (refetch `plan.options` after add/remove) | `plan.options` |

## Settings menu (`menus/_settings.py`, `pyRadPlan/_settings.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| XP (Backend)… (prefer_gpu, preferred_cpu/gpu_array_backend, jit_backends) | `_settings.py:226-230`, `pyRadPlan/_settings.py:249-270` | stub (as Qt; reason from the real capabilities) | stub: numpy only, no GPU, no numba in Pyodide (show values read-only) | `app.settings` |
| AI… (agents_model, display usage, modelhub_*) | `pyRadPlan/_settings.py:184-201` | stub (as Qt; reason from the real capabilities) | stub: AI not installed | `app.settings` |
| GUI… (ui_mode) | `pyRadPlan/_settings.py:273-317` | verified on Pyodide | impl (frontend-held; same descriptor form) | `app.settings` |
| Preferences… (tabs General / XP / AI / GUI; OK applies, red validation status) | `_settings.py:232-233,169-207` | verified on Pyodide | impl: General `base_data_path` stub (host paths), others as above | `app.settings`, `app.settings_set` |

## Help menu (`menus/_help.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Documentation (readthedocs) | `_help.py:331-332` | done (mock e2e) | frontend (link) | – |
| Source Code (GitHub) | `_help.py:333-334` | done (mock e2e) | frontend (link) | – |
| Third-party Licenses… (filter, Package / Version / License, license text) | `_help.py:154-243` | verified on Pyodide | impl (the packages of the Pyodide runtime) | `app.licenses` |
| (web only) links to `THIRD_PARTY_NOTICES.txt` (JavaScript bundle, Pyodide runtime) and `LICENSE.txt` in the licenses view | – | done (mock e2e; files checked by `e2e/pages.mjs`) | frontend (build plugin) | – |
| About pyRadPlan… (logo, version, Python, description, links, BSD note, NOT FOR CLINICAL USE box, Third-party Licenses button) | `_help.py:246-317` | verified on Pyodide | impl; texts from `pyRadPlan/app/about.json` in both | `app.info` |
| (web only) About: "Privacy…" button, privacy line, web build (channel, version, commit) | – | done (mock e2e) | frontend | – |
| (web only) First-visit disclaimer and privacy notice (must be acknowledged, backend starts afterwards), status-bar link, splash line | – | done (mock e2e, `e2e/pages.mjs`) | frontend | – |

## Splash (`windows/_splash_win.py`, `apps.py:33-168`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Frameless splash: logo, status line, n/N counter, progress bar | `_splash_win.py:37-156` | verified on Pyodide | done | worker boot events |
| Phase messages ("Loading core…" … "Loading main window…") | `apps.py:33-46,118-120` | verified on Pyodide | frontend (map boot stages to the Qt wording where they correspond) | – |
| "Detecting array backends…" | `apps.py:122,150-168` | n/a | n/a | – |
| "Loading patient <x>…" (command-line patient) | `apps.py:126-128` | done (mock e2e) | frontend: URL parameter `?phantom=TG119` | `data.load_phantom` |

## Result window (`windows/_result_win.py`) and launchers (`apps.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Standalone "Plan Result Viewer" (`launch_viewer`) | `_result_win.py:29-122` | n/a | n/a: the main window's viewer is the same widget | – |
| Its File ▸ Open NPY/NPZ… (validates, then discards the array) | `_result_win.py:50-93` | n/a | n/a (no effect in Qt) | – |
| `apps.analysis_viewer` standalone DVH window | `apps.py:232-312` | n/a | n/a (the analysis window is opened from the viewer) | – |

## Log console (`widgets/_log_console.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Level combo Debug / Info (default) / Warning / Error; sets the `pyRadPlan` logger level | `_log_console.py:29-36,90-100,129-136` | verified on Pyodide | impl | `session.log_level` |
| Clear | `_log_console.py:102-104` | done (mock e2e) | frontend | – |
| Log view: "HH:MM:SS [LEVEL] logger: message", colours by level, 2000 lines | `_log_console.py:39-64,141-155` | verified on Pyodide | frontend | `log` events |
| History after reload / reconnect | – | done (mock e2e) | impl | `session.history` |

## Info and logo (`widgets/_info_widget.py`, `widgets/_logo_widget.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| "pyRadPlan v<version> (Python x.y.z)" | `_info_widget.py:16-35` | verified on Pyodide | done | `app.info` |
| GitHub link | `_info_widget.py:37-42` | verified on Pyodide | done | – |
| DKFZ logo (blue/white by palette), "NOT FOR CLINICAL USE!", pyRadPlan logo | `_logo_widget.py:27-104` | verified on Pyodide | frontend (dark-mode variant) | – |

## UI mode (`widgets/_ui_mode.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Default / Educational: educational greys out the plan's advanced controls (machine, engine + "…", fractions + doses, scenario + "…", optimized quantity, problem + "…", sequencing), values kept | `_ui_mode.py:8-22`, `widgets/plan/_plan_widget.py:616-639,704-712` | verified on Pyodide | impl (descriptor flag `advanced`) | `plan.options` descriptors |

## Workflow widget (`widgets/workflow/_workflow_widget.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Status label (texts of `_update_status`) | `:479-497` | verified on Pyodide | impl: add the " · N further results" suffix (results besides the active one) | `state.status`, `state.results.tags` |
| Refresh | `:262-264,1040-1041` | verified on Pyodide | done | `session.state` |
| Stop (cooperative; optimization keeps last iterate: under Pyodide the first interrupt requests the cooperative stop of an optimization, a second one discards it) | `:265-272,613-619` | verified on Pyodide | done | interrupt buffer / `session.cancel` (native) |
| Busy bar (indeterminate → determinate nested fraction) | `:276-282,636-684` | verified on Pyodide | done | `progress` events |
| Progress text in the status label while busy (Qt: the progress levels; web: label · detail · fraction, no separate detail line) | `:657-664,697` | done (mock e2e) | frontend | `progress` events |
| Button grid exactly as Qt: Loading = Load File / Load Folder / Import from Binary, Dose Influence = Calc. Dose Influence / Load Dij, Result Computation = Optimize / Recalculate Dose, Export / Save = Snapshot Result; no Run All, Generate STF or phantom selector (phantoms: File > Open Example Phantom) | `:284-330` | verified on Pyodide | impl | – |
| Status dots Loading / Dose Influence / Result Computation (tooltips) | `:302-316,456-477` | verified on Pyodide | done | `state.stages` |
| Load File | `:286` | verified on Pyodide | impl (formats as in File menu) | `data.load_file` |
| Load Folder | `:287` | stub (as Qt; reason from the real capabilities) | stub: SimpleITK file IO | – |
| Import from Binary (Qt: permanently disabled "Not yet implemented") | `:288-290` | stub (as Qt; reason from the real capabilities) | mirror-off | – |
| Calc. Dose Influence (pending-edit check, bio-model check dialog, always generates the stf again; v1 amendment 20) | `:291,1199-1220` | verified on Pyodide | done | `workflow.calc_dose_influence` |
| Load Dij (stf + dij from .mat; warns when the file has no dij) | `:292,1014-1035` | verified on Pyodide | impl | `data.load_dij` |
| Optimize (pending-edit check, biological-objective check, opens the status window) | `:293,1268-1316` | verified on Pyodide | impl (status window) | `workflow.optimize`, `progress` events |
| Run Sequencing (photon plan with sequencing enabled, dose influence and a current result; tooltip gives the reason while disabled; replaces the active result, adding the sequence and the `_sequenced` doses) | `_on_sequence`, `_update_button_states` | **gap**: no button | impl (backend ready, frontend open) | `workflow.sequence`, `state.sequencing` |
| Recalculate Dose (weights, angle match, bio check, moves beams to iso) | `:294,1333-1402` | done (mock e2e) | done | `workflow.recalc` |
| Snapshot Result (name prompt, default "<mode>_<n>fx", sanitized; a result of its own, the name of the active result is refused) | `:295-300,1408-1442` | verified on Pyodide | impl | `workflow.snapshot` |
| Warnings collected during a run, shown once afterwards | `:699-702` | verified on Pyodide | frontend: dialog with the run's `log` warnings | `log` events |
| Error dialog "<Type>: <message>" | `:704-705` | verified on Pyodide | done | error responses |
| "Physical dose only?" question for a failing bio model; "invalid α_x/β_x" blocker | `:1107-1182` | done (mock e2e) | done | `PreconditionError` codes |
| Biological objectives without biological dij (blocks Optimize) | `:1226-1266` | verified on Pyodide | done | `PreconditionError biological_objectives` |
| Tissue warnings in status line | `:1184-1197` | done (mock e2e) | frontend | `log` events |

### Import dialogs (`widgets/workflow/`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Import Binary Patient (CT file, masks table File/Name/Type, Add file(s)/Add folder/Remove) | `_binary_import_dialog.py:61-193` | stub (as Qt; reason from the real capabilities) | stub: CT and masks are read with `sitk.ReadImage` | – |
| Import DICOM (CT series, structure set, dose combos) | `_dicom_import_dialog.py:48-127` | stub (as Qt; reason from the real capabilities) | stub: CT needs `sitk.ImageSeriesReader`, RTDOSE `sitk.ReadImage` (the pydicom-only catalog alone is no use) | – |
| Import Image (CT new / CT replace / structures / dose + result name, kind inference) | `_image_import_dialog.py:64-172` | stub (as Qt; reason from the real capabilities) | stub: needs `sitk.ReadImage` | – |

## Plan widget (`widgets/plan/_plan_widget.py`)

`[adv]` = locked in educational mode.

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Radiation mode (photons, protons, helium, carbon, oxygen, VHEE); re-lists machines, engines, problems, bio models, presets | `:330-332,775-782` | verified on Pyodide | impl: re-list from backend | `plan.options` |
| Machine (editable combo, "(from plan)" entry, default Generic) `[adv]` | `:334-339,726-768` | verified on Pyodide | impl | `plan.options`, `plan.set {machine}` |
| Dose engine (display name, identifier in popup) `[adv]` | `:649-656,1211-1220` | verified on Pyodide | impl (MC engines disabled with reason) | `plan.options`, `plan.set {engine}` |
| Dose engine "…" → generic config form | `:657-660,1222-1236` | verified on Pyodide | impl | `plan.form {kind:"engine"}`, `plan.set {engine_config}` |
| Fractions 1..1000 (30) `[adv]` | `:343-345` | verified on Pyodide | done | `plan.set` |
| Doses: per fraction / total (form default total) `[adv]` | `:347-358` | done (mock e2e) | impl | `plan.set {dose_convention}` |
| Gantry angles [deg] text | `:376-381` | verified on Pyodide | done | `plan.set` |
| Couch angles [deg] text (auto-resized to the beam count on edit) | `:383-389,1582-1600` | verified on Pyodide | frontend (resize rule) | `plan.set` |
| Number of beams spin (sets equidistant gantry angles) | `:390-397,1558-1580` | verified on Pyodide | frontend | – |
| Choose preset… (beam angle presets by mode) | `:399-410,1538-1556` | done (mock e2e) | impl | `plan.options.beam_presets` |
| ✨ AI beam angles | `:412-417,1709-1774` | stub (as Qt; reason from the real capabilities) | stub: AI not installed in the browser | `app.info.capabilities` |
| Bixel width / spot spacing [mm] 0.1..100 (5) | `:431-439` | verified on Pyodide | done | `plan.set` |
| Iso center [mm] + Auto. (shows target centre of mass; per-beam isos "multiple iso centers") | `:441-462,1489-1536` | verified on Pyodide | impl | `plan.get {detail}` (`resolved_iso_center`) |
| Biological model combo (options "formalism · provenance", unavailable entries disabled with reason, unsupported plan model kept; the selection after a mode / machine / engine change is the backend's `bio_selection`, the proton default verified on Pyodide) | `:465-479,866-1000`, `pyRadPlan/app/_planform.py:select_bio_model` | verified on Pyodide | impl (shared rules) | `plan.options {radiation_mode, machine, engine, current_bio}`, `plan.set {bio_model}` |
| Bio "…" → config form (model parameters) | `:1042-1109` | done (mock e2e) | impl | `plan.form {kind:"bio_model"}` |
| Replacement note when the bio model changed with mode/machine (status, tooltip) | `:813-819`, `select_bio_model` | done (mock e2e; backend note checked against Qt in `test_plan_form_read.py`) | backend (`bio_selection.note`) | `plan.options {current_bio}` |
| Scenario model (Nominal, Worst-case Grid, Importance Grid, Random Samples; Explicit only when the plan holds one; kept on a radiation mode change, verified on Pyodide photons → protons) `[adv]` | `:486-499,1242-1302` | verified on Pyodide | impl | `plan.options`, `plan.set {scenarios}` |
| Scenario "…" → config form (setup sd/shift, range abs/rel/combined, ct_phase, wc_sigma, distribution, grid/sample counts) `[adv]` | `:1263-1278` | verified on Pyodide | impl | `plan.form {kind:"scenario"}` |
| Optimized quantity ("auto", disabled "not yet implemented") `[adv]` | `:508-517` | stub (as Qt; reason from the real capabilities) | mirror-off | – |
| Set tissue parameters (enabled with cst) → Tissue dialog | `:519-525,1111-1123` | verified on Pyodide | impl (see Tissue dialog) | `tissue.*` |
| Tissue marker ⚠ (error) / ℹ (warning) with issue tooltips | `:528-531,1165-1205` | done (mock e2e) | impl (evaluated for the unapplied form model) | `plan.tissue_check` |
| Dose grid res. x/y/z 0.1..50 (Qt 3.0; web default 5.0, possibly locked) | `:594-614,669-674` | verified on Pyodide | impl: default from the session profile, `editable:false` when locked | `state.options`, `plan.options` descriptors |
| Use CT grid | `:598-599,1602-1611` | done (mock e2e) | done (disabled when locked) | `state.ct` |
| Sequencing: Run sequencing + sequencer combo + "…" → config form (photon plans; the settings do not outdate the dose influence or the result) `[adv]` | `_build_sequencing_row`, `_form_sequencing`, `_on_sequencer_config` | **gap**: shown disabled ("Run sequencing" / "Levels"), not wired | impl (backend ready, frontend open) | `plan.options.sequencers`, `plan.form {kind:"sequencer"}`, `plan.set {sequencing}` |
| Run DAO / 3D conformal (disabled) `[adv]` | `_build_sequencing_row` | stub (as Qt; reason from the real capabilities) | mirror-off | – |
| Optimization problem combo `[adv]` | `:548-550,1312-1325` | verified on Pyodide | impl | `plan.options`, `plan.set {problem}` |
| Problem "…" → Optimization dialog `[adv]` | `:1327-1338` | verified on Pyodide | impl (see below) | `plan.form {kind:"problem"}`, `plan.form {kind:"solver"}` |
| Status label (applied / default applied / discarded / CT grid / errors in red / "Modified — not applied") | `:297,1871-2008` | verified on Pyodide | frontend | – |
| Discard (enabled when modified and a plan exists) | `:300-303,1641-1665` | done (mock e2e) | done | – |
| Apply / "Apply *" bold when modified | `:305-307,1613-1630` | verified on Pyodide | impl (all v1 fields) | `plan.set` |
| Orange modified frame per field | `:107-142,1956-1994` | done (mock e2e) | frontend | – |
| Default plan auto-applied on load (Qt: the form, initialised from the session's `default_plan`, applies itself through `set_plan`; deferred until a running load has ended) | `:1388-1406` | verified on Pyodide | done | `session.create` profile |

### Optimization dialog (`widgets/plan/_optimization_dialog.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Problem settings form (apply_overlap, evaluate_all_scenarios, bypass_objective_jacobian, max_approximation, p_norm; hides solver, convert_dose_objectives, max_iter, display) | `:35,144-149` | verified on Pyodide | impl | `plan.form {kind:"problem"}` |
| Solver combo (registered solvers; plan solver not registered → "(not available)") | `:152-176,255-260` | verified on Pyodide | impl (IPOPT unavailable with reason in the browser) | `plan.options.solvers` |
| Solver form (max_iter, abs_obj_tol, max_time, method, options, …; values kept per solver) | `:281-293,262-265` | verified on Pyodide | impl | `plan.form {kind:"solver"}` |
| Forwarded max_iter / display from the problem into the solver form | `:121-130,286-288` | done (mock e2e) | impl (backend does the forwarding) | `plan.form` |
| Unavailable solver text | `:273-276` | done (mock e2e) | impl | `plan.form` |
| OK / Cancel, red validation status | `:188-201,295-299` | done (mock e2e) | frontend | descriptor validation, `ValidationError.details` |

### Generic configuration form (`widgets/_config_form.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| bool → checkbox; Optional[bool] None shown unchecked | `:329-330,356-360` | verified on Pyodide | impl (null shown as indeterminate) | descriptors `type:"bool"`, `nullable` |
| int / float spins with bounds; Optional None → "None" text field | `:331-404` | verified on Pyodide | impl | `type:"int"/"float"`, `min/max/exclusive_*`, `nullable` |
| Literal → combo; Union with str → editable combo | `:338-341,406-419` | verified on Pyodide | impl (Union[Literal,"bool"] becomes a choice incl. true/false; see ambiguities) | `type:"choice"`, `choices`, `editable_choice` |
| str → text | `:342-343,421-429` | verified on Pyodide | impl | `type:"str"` |
| list/tuple of numbers → "a b c" text | `:344-351,431-451` | verified on Pyodide | impl | `type:"number_list"` |
| Nested dataclass / model → collapsible section, dot-path keys, hidden entries | `:478-526` | verified on Pyodide | impl | `type:"group"`, `fields` |
| Everything else → JSON text | `:352-354,453-472` | done (mock e2e) | impl | `type:"json"` |
| Label from metadata or name, tooltip from description, configurable=False hidden | `:285-314` | verified on Pyodide | impl | `label`, `description` |
| Live validation (red border + tooltip) | `:575-592` | done (mock e2e) | impl: client-side bounds; server-side on Apply | `ValidationError.details` |
| "No configurable parameters." | `:223-224` | done (mock e2e) | frontend | – |

## Tissue dialog (`widgets/plan/_tissue_dialog.py`, helpers `widgets/_tissue_params.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Note label | `_tissue_dialog.py:120-126,185-188` | verified on Pyodide | frontend | – |
| Tissue picker (base-data classes ✓, cell-line presets, other provenances; Apply to selected / all) | `_tissue_dialog.py:196-198,348-372,456-494` | done (mock e2e) | impl | `tissue.schema` (picks), `tissue.evaluate {op:"apply_pick"}` |
| Table: Structure (tinted by issue), Type | `_tissue_dialog.py:288-293,707-793` | verified on Pyodide | impl | `tissue.evaluate` (issues per row) |
| Model combo per row Generic / LEM / MKM / mMKM | `_tissue_dialog.py:296-301,624-673` | verified on Pyodide | impl | `tissue.schema.models` |
| LEM version I–IV | `_tissue_dialog.py:304-311` | done (mock e2e) | impl | `tissue.schema` |
| Numeric parameter cells D_t, r_nucleus, r_domain, α_0, β_0, α_x, β_x (units in headers, "unset" for NaN) | `_tissue_dialog.py:324-335`, `_tissue_params.py:1348-1384` | verified on Pyodide | impl (no artificial 0..100 clamp; bounds from the model) | `tissue.schema` |
| Reference radiation combo (MKM/mMKM) | `_tissue_dialog.py:210-231,313-322` | done (mock e2e) | impl | `tissue.schema.reference_radiations` |
| Set parameter (bulk: parameter, value, Set) | `_tissue_dialog.py:374-454` | done (mock e2e) | frontend (applies to the draft rows) | – |
| Status label | `_tissue_dialog.py:249-253` | done (mock e2e) | frontend | `tissue.evaluate.notes` |
| Match base data | `_tissue_dialog.py:255-260,496-519` | done (mock e2e) | impl | `tissue.evaluate {op:"match"}` |
| Relabel as … (provenance fix) | `_tissue_dialog.py:262-267,795-819` | done (mock e2e) | impl | `tissue.evaluate {op:"relabel"}` |
| Derive from model (MKM/mMKM; "Nothing to derive") | `_tissue_dialog.py:269-271,584-602` | done (mock e2e) | impl | `tissue.evaluate {op:"derive"}` |
| Live derivation of α_x/β_x and live comparison colouring (debounced) | `_tissue_dialog.py:579-582,675-705`, `_tissue_params.py:1219-1223,1479-1574` | done (mock e2e) | impl (debounced `tissue.evaluate`) | `tissue.evaluate` |
| OK: errors → "Apply anyway / Keep editing"; writes cst (VOIs recreated) | `_tissue_dialog.py:821-863`, `_tissue_params.py:881-902` | verified on Pyodide | impl | `tissue.apply {apply_anyway}` |

## Objectives & constraints (`widgets/optimization/_optimization_widget.py`)

No constraints editor exists in Qt (`:9`); only objectives.

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| VOI selector "name (TYPE)" | `:126,216-228` | verified on Pyodide | impl | `cst.get` |
| + Add objective (Squared Deviation for TARGET, Squared Overdosing otherwise; shared quantity) | `:129-130,453-471` | verified on Pyodide | impl | `objectives.add` |
| ✨ AI (objectives, QI-adaptation prompt) | `:131-136,473-561` | stub (as Qt; reason from the real capabilities) | stub: AI not installed in the browser | `app.info.capabilities` |
| Change all quantities… menu | `:140-158,636-657` | done (mock e2e) | impl | `objectives.set_quantity_all`, `objectives.catalog` |
| "N objective(s)" | `:160,253-257` | verified on Pyodide | frontend | `cst.get` |
| Table columns VOI / Objective / Penalty / Quantity / Robustness / Parameters / Remove | `:164-182` | verified on Pyodide | impl | `cst.get`, `objectives.catalog` |
| Objective type combo (type change keeps priority and allowed robustness, resets quantity and parameters) | `:284-290,572-588` | verified on Pyodide | impl | `objectives.change_type` |
| Penalty spin 0..1e9 | `:292-298,590-602` | verified on Pyodide | impl | `objectives.update` |
| Quantity combo (RTQuantity names, identifiers) | `:300-312,604-618` | done (mock e2e) | impl | `objectives.update` |
| Robustness combo (per class `available_robustness`) | `:314-333,620-634` | done (mock e2e) | impl | `objectives.catalog`, `objectives.update` |
| Parameter editors (numeric spins; `f_diff` read-only label in Qt; "—" without parameters) | `:345-374` | verified on Pyodide | impl (bounds and units from descriptors; `f_diff` as a choice, see ambiguities) | `objectives.catalog`, `objectives.update` |
| Image reference dropdown (Squared Mimicking d_ref: result images or "Custom image (…)") | `:376-447,673-685` | done (mock e2e) | impl | `objectives.catalog.reference_images`, `objectives.update` |
| Remove (no confirmation) | `:339-343,563-570` | verified on Pyodide | impl | `objectives.remove` |
| Status label (rejected changes) | `:185-186,714-715` | done (mock e2e) | frontend (error messages) | error responses |

### Optimization status window (`widgets/optimization/_status_window.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Window "Optimization Status" (opened by Optimize, reused) | `:53-120`, `_workflow_widget.py:1318-1331` | verified on Pyodide | impl | `progress` events (`data`) |
| Summary "iter N · f=… · Δf=…" | `:81,234-251` | verified on Pyodide | frontend | `progress.data` |
| Objective plot (log y, "+" markers, x = iteration) | `:122-164` | verified on Pyodide | frontend | `progress.data` |
| Pause / Resume | `:88-90,188-195` | stub (as Qt; reason from the real capabilities) | stub in the browser (single-threaded worker); impl natively/server | `session.pause`, `session.resume` |
| Stop | `:91-92,197-201` | verified on Pyodide | impl (interrupt buffer in the browser) | interrupt / `session.cancel` |
| Finalize: buttons off, curves kept | `:180-186` | verified on Pyodide | frontend | – |
| History after reload | – | done (mock e2e) | impl | `session.history` |

## Viewer (`widgets/result/`, `widgets/_result_widget.py`)

### Quantity widget: rendering and interaction (`result/quantity_widget.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| CT background (grayscale, window or full range) | `:781-797` | verified on Pyodide | done | `view.volume ct` |
| Quantity overlay (zeros transparent, colormap jet, opacity 0.4) | `:799-834` | verified on Pyodide | done | `view.volume` |
| Colorbar with "<label> [<unit>]" / "CT [HU]" | `:836-923` | verified on Pyodide | impl: Qt label names (QUANTITY_META) | `state.results.groups[].meta_label` |
| Slice slider + spinbox + "Slice: i / N (<Plane>)" | `:381-390,686-755` | verified on Pyodide | done | – |
| Plane combo Axial / Sagittal / Coronal; home slice at isocenter | `:376-379,670-718` | verified on Pyodide | done | – |
| Follow isocenter until the user moves the slice | `:280,1525-1527,1604-1620` | done (mock e2e) | frontend | – |
| Wheel = slice ±1; Ctrl+wheel zoom; drag pan; right-drag zoom; pinch | `:224-230,1559-1565` | verified on Pyodide | done | – |
| pyqtgraph right-click menu (View All, axis options) | pyqtgraph default | done (mock e2e) | frontend: "View all" only | – |
| Isolines (default 0.5/0.8/0.95 × max, custom levels, colour from colormap) | `:964-1007` | verified on Pyodide | done | – |
| VOI contours (selected VOIs, colours) | `:1043-1064` | verified on Pyodide | impl: switch to compact masks | `view.structure_masks {encoding}` |
| Beams overlay (projected axes, wedge, source marker, ⊙/⊗, labels "#i", tooltips) | `:1202-1317` | verified on Pyodide | impl | `view.beams` |
| Beam drag (rotate gantry on the orbit, preview ellipse, Esc cancels, commits into the plan form) | `:207-216,1319-1430` | done (mock e2e) | impl | `view.beams` (orbit), plan form draft |
| Pending (unapplied) beam angles drawn dashed with "*" | `_result_widget.py:587-623,764-778` | done (mock e2e) | impl | `view.beams {draft}` |
| Isocenter cross (full on its slice, half within 5 slices) | `:1432-1462` | verified on Pyodide | done | `view.beams.iso_centers` |
| Orientation marker (mannequin with R/L A/P I/S, top-left) | `result/_orientation_marker.py:15-155`, `:1071-1077` | verified on Pyodide | frontend (mannequin) | `state.ct.direction` |
| Cursor readout: Pos (mm), Index sitk/numpy, CT HU, one line per full quantity | `:527-580` | verified on Pyodide | impl: list all full quantities with RTQuantity names | `state.results.groups` |
| Empty state | `:465-503` | verified on Pyodide | done | – |

### Viewer options (`result/viewer_options_widget.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Mode toggle Quantity / CT | `:47-52` | done (mock e2e) | done | – |
| Colormap combo (jet, viridis, magma, plasma, inferno, gray, bone, seismic, coolwarm, RdBu_r, bwr) | `:54-72` | done (mock e2e) | done | – |
| CT window preset Custom / Abdomen / Lung / Bone / Brain | `:74-80` | done (mock e2e) | done | – |
| Window centre / width spins + sliders | `:82-110,204-309` | done (mock e2e) | done | – |
| Range min / max | `:112-129,311-331` | done (mock e2e) | done | – |
| Use local min/max | `:131-134` | done (mock e2e) | done | – |
| Overlay opacity slider | `:136-143` | done (mock e2e) | done | – |
| Reset options | `:145-148` | done (mock e2e) | done | – |

### Visualization widget (`result/visualization_widget.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| CT / CST / Quantity / Isolines / Isocenter checkboxes | `:63-87` | verified on Pyodide | done | – |
| Beams checkbox (disabled until beam angles exist) | `:89-92,151-157` | verified on Pyodide | impl | `view.beams` |
| Orientation checkbox | `:94-102` | verified on Pyodide | done | – |
| Overlay quantity selector (two combos: quantity group incl. the groups of further results, component full / beam i / scenario i: label of the group's own result; scenario components after a robust optimization verified on Pyodide with a worst-case grid plan) | `:117-126`, `widgets/_quantity_selector.py:20-188` | verified on Pyodide | impl | `state.results`, `view.volume` (component keys) |
| Set isolines dialog (OK / Cancel / Reset; says "(Gy)") | `:130-132,167-199` | done (mock e2e) | done | – |
| Recenter | `:134-136` | done (mock e2e) | done | – |
| Show Sequence → sequence window (per beam: MLC shapes of the delivered segments, playback, fluence delivered / optimized / difference, overlays) | `result/visualization_widget.py`, `result/aperture_widget.py`, `windows/_sequence_win.py` | **gap**: not available (no view method for sequences yet) | impl (backend and frontend open) | – |
| Show DVH / QI → analysis window | `:145-149`, `_result_widget.py:861-887` | verified on Pyodide | impl (see Analysis) | `analysis.*` |
| Default quantity after each result change (rbe_x_dose > physical_dose > first) | `_result_widget.py:62-74,553-556` | verified on Pyodide | done | `state.results.default` |

### VOIs widget (`result/vois_widget.py`)

Qt has no context menu here; editing goes through the click on the name.

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Group mode By Type / By Overlap | `:785-789,853-883` | done (mock e2e) | impl | `cst.get` (`overlap_priority`) |
| Groups Targets / OARs / External / Helpers / Other; names sorted; 2 columns | `:102-108,823-870,970-986` | verified on Pyodide | done | `state.structures` |
| Row: colour swatch + checkbox with 12-char truncated name | `:988-1017` | verified on Pyodide | done | – |
| Rich tooltip: name, type, tissue model, α_x, β_x, α/β, model params, overlap priority | `:1019-1044` | done (mock e2e) | impl | `cst.get` |
| Default checked set (VOIs with objectives, else first EXTERNAL/TARGET/OAR, …; empty masks never) | `:893-968` | done (mock e2e) | impl | `cst.get` (`num_objectives`, `num_voxels`) |
| All / None | `:773-782,1086-1098` | verified on Pyodide | done | – |
| Colour swatch click → colour picker → contour colour (Qt mutates `voi.color` in place) | `:1066-1080` | verified on Pyodide | impl (persisted in the cst) | `cst.update_structure {color}` |
| Click on the name → "Edit VOI: <name>" dialog | `:125-144` | verified on Pyodide | impl | – |
| Edit VOI: Name (read-only) | `:216-219` | verified on Pyodide | impl | `cst.get` |
| Edit VOI: Type TARGET / OAR / EXTERNAL / HELPER | `:221-226` | verified on Pyodide | impl | `cst.update_structure {type}` |
| Edit VOI: Tissue model, parameter groups (model and reference parameters in one row), reference radiation, live derivation of α_x / β_x | `:228-305,332-684` | verified on Pyodide | impl (same backend as the tissue dialog, one row) | `tissue.schema`, `tissue.evaluate` |
| Edit VOI: tissue picker + Apply, preselecting the pick the structure's tissue corresponds to (matching base data class or applied preset, neutral "No matching tissue" otherwise; follows Apply / Match base data / Relabel) | `:239-249,334-377` | done (mock e2e); not yet verified on Pyodide. The row held no picker, Match base data, Relabel or Derive before 2026-11-20 although this table said so | impl (`tissue.evaluate` rows return `pick`, amendment 24; drafts carry the cell line, amendment 25) | `tissue.schema`, `tissue.evaluate {op: "apply_pick", pick}` |
| Edit VOI: base data row (Match base data / Relabel as …), Derive from model, status | `:332-684` | done (mock e2e, mock does not simulate them); not yet verified on Pyodide | impl | `tissue.evaluate {op: "match" \| "relabel" \| "derive"}` |
| Edit VOI: Overlap priority 0..9999 | `:307-311` | verified on Pyodide | impl | `cst.update_structure {overlap_priority}` |
| Edit VOI: OK (tissue errors ask "apply anyway") / Cancel | `:313-318,686-733` | verified on Pyodide | impl | `cst.update_structure`, `tissue.apply` |
| Edit VOI: OK with a picker selection that was not applied asks "Apply and continue / Discard selection and continue / Keep editing" | `:688-710` | done (mock e2e); not yet verified on Pyodide | impl | `tissue.evaluate {op: "apply_pick"}` |

### Save result dialog (`result/_save_result_dialog.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Checkbox per exportable quantity "<label> (<key>)", All / None, "Select at least one quantity" | `:37-94` | verified on Pyodide | impl | `state.results`, `export.save {object:"result", quantities}` |
| Format / file choice (Qt: image formats, folder for several) | `_workflow_widget.py:1658-1701` | verified on Pyodide | impl with npz / mat / DICOM RTDOSE (zip for several); image formats stub | `export.save` |

## Analysis window (`widgets/_analysis_widget.py`, `widgets/analysis/`, `windows/_analysis_win.py`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| Window "DVH / QI Analysis" (non-modal, closes with its viewer) | `_analysis_win.py:17-86` | verified on Pyodide | impl | – |
| Tabs DVH / QI and Gamma Analysis | `_analysis_widget.py:120-162` | verified on Pyodide | impl | – |
| Primary quantity selector (group + component; opens on the viewer's quantity, incl. a scenario; the quantity combo stretches and the component combo fits its text without squeezing it, long scenario labels elided with the full text as tooltip) | `_analysis_widget.py:182-186`, `widgets/_quantity_selector.py:52-62` | verified on Pyodide | impl | `state.results` |
| Compare quantity selector ("— None —"; dashed curves; second x axis for incompatible units) | `_analysis_widget.py:188-193,766-779` | verified on Pyodide | impl | `analysis.dvh` (`unit_info`), `analysis.compare_units` |
| All scenarios checkbox (thin band curves of every scenario of the primary group; shown only when the group has scenarios) | `_analysis_widget.py:195-202,788-794` | verified on Pyodide | impl | `analysis.dvh {keys}` |
| Structures groups by type, colour swatch + checkbox, All / None | `_analysis_widget.py:456-552,752-764` | verified on Pyodide | impl | `state.structures` |
| Structure colour pick (analysis-local in Qt) | `_analysis_widget.py:540-552` | done (mock e2e) | frontend (local) | – |
| QI parameters D_x [%] (default "2 5 50 95 98", shared) | `_analysis_widget.py:262-267` | verified on Pyodide | impl | `analysis.qi {ref_vols}` |
| QI parameters V_x [unit] (per quantity; default five values from the max) | `_analysis_widget.py:269-276,582-611` | verified on Pyodide | impl | `analysis.qi {ref_doses}` |
| "Show in DVH" toggles for D_x / V_x markers | `_analysis_widget.py:308-322` | verified on Pyodide | frontend | – |
| Metric checkboxes mean / std / max / min, All / None | `_analysis_widget.py:279-382` | verified on Pyodide | frontend | – |
| DVH plot: per-structure curves (1000 bins, range per structure), legend, grid, axis "<label> [<unit>]", "Volume [%]" | `analysis/_dvh.py:93-288`, `pyRadPlan/analysis/_dvh_plot.py:116-287` | verified on Pyodide | impl | `analysis.dvh {range:"per_structure", num_points:1000}` |
| DVH plot and analysis window follow light / dark mode, also when switched while open (figure `--window`, axes `--base`, text / ticks / axes / cursor `--text`, grid blended; legend, hover box, marker labels on the axes colour; QI table, tabs, title bar from the theme variables) | `widgets/_plot_theme.py`, `analysis/_dvh.py:507-528` | done (mock e2e) | frontend | – |
| Hover cursor: dotted crosshair (vertical and horizontal line) + readout of the pointer's dose and volume and of every curve's volume | `analysis/_dvh.py:452-509` | done (mock e2e screenshot) | frontend | – |
| Highlight (from table, curve click, marker click; empty click clears) | `analysis/_dvh.py:305-314,425-447` | done (mock e2e) | frontend | – |
| QI markers (D_x circles, V_x squares), guide lines, draggable parameter line, release → QI point dialog | `analysis/_dvh.py:318-414` | done (mock e2e) | frontend (math on the returned curve arrays) | `analysis.qi` |
| Double-click → add QI point | `analysis/_dvh.py:441-443` | done (mock e2e) | frontend | `analysis.qi` |
| pyqtgraph export (PNG / SVG / CSV of curves) | pyqtgraph context menu | done (mock e2e) | frontend (PNG and CSV) | – |
| QI table: Structure + metric columns ("mean [Gy]", "D98 [Gy]", "V20Gy [%]"), "%.2f", "-" for missing; selection syncs with the plot | `analysis/_qi.py:34-334` | verified on Pyodide | impl | `analysis.qi` |
| Copy (TSV to clipboard) / Export CSV… | `analysis/_qi.py:95-103,222-263` | done (mock e2e) | frontend | – |
| QI point dialog (D_x / V_x radios, spins, Replace / Add as new / Cancel) | `analysis/_qi_point_dialog.py:56-112` | done (mock e2e) | frontend | – |
| Gamma Analysis tab ("not implemented") | `analysis/_gamma.py:8-17` | stub (as Qt; reason from the real capabilities) | mirror-off (same placeholder text) | – |

## AI widgets (`widgets/ai/`)

| Element | Qt | Web | Target | Methods |
|---|---|---|---|---|
| AI task dialog (model, treatment site, user prompt, system prompt, data context, Run / Close, usage line) | `ai/_ai_task_dialog.py:74-239` | stub (as Qt; reason from the real capabilities) | stub: `pydantic_ai` not installed in Pyodide; needs provider keys and network | `app.info.capabilities.features.ai` |
| Availability tooltips (missing dependency / no model key) | `ai/__init__.py:13-40` | done (mock e2e) | impl (reason text from capabilities) | `app.info.capabilities` |

## Dependencies outside `pyRadPlan/app` and `web/` (for the orchestrator)

1. **Pickle compatibility between native SimpleITK and the shim** (shim agent): native pickles
   reference `SimpleITK.SimpleITK.Image` (reduce args `(size, pixel_id, components)` plus state);
   the shim pickles as `SimpleITK._image.Image`. For "Save Workspace" in one runtime to load in the
   other, the shim needs a `SimpleITK.SimpleITK` submodule whose `Image` accepts the native reduce
   format, and should itself reduce to that name and format. Without it, workspace pickles are
   runtime-specific (the app reports this in `capabilities.formats`).
2. Done: the Qt-free tissue helpers (`available_tissue_classes`, `ranked_tissue_picks`,
   `compare_tissues_quietly`, `tissue_issues_quietly`, `build_tissue_properties`,
   `load_machine_quietly`, `bio_options_quietly`, field/label helpers) live in
   `pyRadPlan/biology/_editing.py`, which both the app layer and `gui/widgets/_tissue_params.py`
   (re-exporting them) import; nothing was duplicated.
3. `contourpy` must be added to the Pyodide bootstrap package list for DICOM RTSTRUCT export.
4. Done: `pyRadPlan.app.DEFAULT_PLAN` (native and web profile) carries the Qt form's
   `dose_convention: "total"` (`_plan_widget.py:347-352`); see ambiguity 1.

## Ambiguities found in the Qt GUI

1. Dose convention: the Qt plan form defaults to "total" (`_plan_widget.py:352`) while `Plan`
   defaults to "per_fraction" (`pyRadPlan/plan/_plans.py:76-81`), so the Qt default plan reports
   total-course doses. Resolved: both profiles of the app layer default to `"total"` as the Qt
   form does.
2. DVH binning: Qt uses 1000 bins with a range per structure (`pyRadPlan/analysis/_dvh.py:319-327`);
   protocol v0 `analysis.dvh` uses 500 bins on one shared axis. v1 adds `range` and keeps v0's
   defaults; the analysis window requests Qt's settings.
3. D_x has two definitions on one screen: the table uses `np.percentile` over voxels
   (`pyRadPlan/analysis/_qi.py:260`), markers and dragging use the histogram (`analysis/_dvh.py:388-395`).
   The web mirrors both.
4. The default V_x ladder uses the maximum of the whole quantity array, including voxels outside all
   structures (`pyRadPlan/analysis/_qi.py:514-536`).
5. Two naming systems for quantities: the selector and readout use `RTQuantity.name` ("dose",
   "RBExDose"); colorbar, overlay and save dialog use `QUANTITY_META` ("Dose", "RBE-weighted Dose")
   (`_result_widget.py:41-56`). v1 sends both.
6. The viewer's quantity choice is reset to the default after every result change
   (`_result_widget.py:553-556`), overriding the user's selection.
7. VOI colour picks mutate `voi.color` in place without a workspace notification. Both
   frontends now go through `cst.update_structure` / `Session.update_structure`, which treats
   colour and visibility as display settings: set in place, accepted while a run is busy,
   outdating nothing, counted in `revisions.workspace.cst`.
8. The viewer ignores the CT direction matrix for beams, isocenter and the "Pos" readout
   (`_result_widget.py:187-196`, `quantity_widget.py:562`); only the orientation marker uses it.
   v1 sends LPS geometry and leaves the mapping to the renderer.
9. `Union[Literal["auto"], bool]` fields (`calc_let`, `calc_bio_dose`) cannot be set to a bool in
   the Qt form (`_config_form.py:340`); `Optional[bool]` None looks like False. v1 represents them
   as choices `auto / true / false` and `nullable` booleans.
10. Objectives: changing the type drops quantity and parameters (`_optimization_widget.py:583-587`);
    parameter spins ignore the model bounds (`:364`); `f_diff` is read-only (`:359-372`);
    unparsable objectives vanish on the next write (`:259-270,702`). v1 keeps the type-change rule,
    sends bounds and makes `f_diff` a choice (an improvement, flagged here).
11. Tissue spins clamp the display to 0..100 while the stored value is unclamped
    (`_tissue_params.py:1372-1425`).
12. The busy lock leaves Base Data and Settings enabled during a computation
    (`_main_win.py:146-150`). The web follows the backend: any long request blocks the worker, so
    base-data edits wait (they are cheap methods, but the worker is single-threaded).
13. A cst-only `.mat` cannot be loaded at all (`load_data` raises `KeyError: 'ct'`), although the
    Qt merge rule handles "structures without CT in the file" (`_workflow_widget.py:783-789`).
14. Resolved by the result collection (PROTOCOL.md v1 amendment 26): snapshots are results of
    their own with keys `"<result>/<key>"`; no tags are guessed from key endings any more.
15. The isoline dialog always says "(Gy)" (`result/visualization_widget.py:172`); the Save Result
    and VOI labels are truncated to 12 characters.
16. The AI tooltip calls `pydantic-ai` optional while it is a core dependency in `pyproject.toml`.
17. Pending-edit questions are asked for Save Plan and Save Workspace but not for Save Dij / CST /
    Result (`_workflow_widget.py:1494,1531-1537,1563,1618`).

## Deliberate deviations

Where the Qt behaviour is plainly a bug, the app layer does the sane thing (orchestrator
decision, 2026-11-18). Everything else mimics Qt, including the quirks listed above.

Backend part A (plan editor, structures, tissues, objectives, results):

1. `Union[Literal[...], bool]` engine fields (`calc_let`, `calc_bio_dose`, …) are choices of the
   literals plus `true` / `false`; Qt can only keep "auto" (`_config_form.py:340`).
2. `Optional[bool]` fields are `nullable` (null shown distinctly); Qt shows `None` as unchecked.
3. A VOI colour change goes through `cst.update_structure` in both frontends (the Qt viewer
   through `Session.update_structure`); as in Qt it does not outdate the result.
4. Objective parameters carry the model bounds and units and `f_diff` is an editable choice; Qt's
   spins allow ±1e9 and show `f_diff` read-only (`_optimization_widget.py:345-374`). Edits of both
   frontends go through the session's validated objectives editor, so a rejected edit writes
   nothing in either (the Qt table shows the reason in its status line).
5. Tissue parameter values are not clamped to 0..100 for display (`_tissue_params.py:1373`);
   the model validates them.
6. (Phase 0) A CT written without its products marks stf, dij and result stale
   (`pyRadPlan/app/_stages.py`, `StageTracker.update`).
7. (Plan form on the session, 2026-11-18) Behaviour of both frontends now: `plan.set` keeps a live
   solver instance of `prop_opt` instead of deep-copying it, keeps `prop_opt` entries the problem
   form does not show (`convert_dose_objectives`, entries of no problem field) when
   `problem.values` does not send them or sends them back unchanged, and rejects a `bio_model`
   mapping with keys other than `model`, `option`, `values` (a plain specification
   `{model, **parameters}` lost its parameters silently before). The Qt Apply sends only what
   differs from the plan, so a plan without an explicit dose engine keeps leaving it to the
   radiation mode (Qt named the engine shown on every Apply). The web sends its modified fields
   (`planSetParams`); a radiation mode change keeps the scenario model and the problem settings on
   the backend (v1 amendment 22), so both frontends keep them without the web resending them. Qt
   editing a plan with `convert_dose_objectives` (problem dialog, or a mode change) was refused
   with "Unknown field(s)" before and now works; a mode change with an explicit scenario set
   keeps the set (Qt failed to build `{model: "explicit"}` before).

## Frontend work packages

Three packages with disjoint component files, to run in parallel. Each package keeps its types,
store slice, mock handlers and components in feature-local modules.

### Shared files and the rule for touching them

- `src/api/protocol.ts`: v1 types live in feature modules `src/api/protocol/<feature>.ts`
  (`plan.ts`, `viewer.ts`, `io.ts`), each exporting an interface `PlanMethods` / `ViewerMethods` /
  `IoMethods`. `protocol.ts` only gets one line per package, extending
  `Methods extends V0Methods, PlanMethods, ViewerMethods, IoMethods`, plus the shared v1 types that
  all packages use (`FieldDescriptor`, `FormSchema`, `FileResult`, `Revisions`, the SessionState v1
  fields, `Capabilities`). **Package A owns the shared v1 type block** and lands it first (small,
  copied from `web/PROTOCOL.md`); B and C import it.
- `src/store/appStore.ts`: each package adds a slice `src/store/<feature>Slice.ts` (zustand slice
  pattern) and registers it with one spread line in `appStore.ts`. Revision-driven refetching lives
  in one helper `src/store/revisions.ts` owned by B (viewer data is the main consumer); A and C call
  it and do not edit it.
- Mock backend `src/api/mock/mockBackend.ts`: each package adds handlers in
  `src/api/mock/<feature>.ts` exporting `{[method]: handler}`, registered with one line in the
  handler table.
- Generic form renderer `src/components/forms/ConfigForm.tsx` (descriptor → editors) is owned by A;
  C uses it for Preferences.
- `App.tsx` / `panels.tsx` layout: each package mounts its components in existing panel slots with
  one-line edits; nobody restructures the layout this round.
- Conflict rule: edits to a shared file are additive (new lines), never reformat or reorder existing
  code, and are kept to registration lines.

### Package A: plan, configuration forms, objectives, optimization, tissue

Components: `PlanWidget.tsx` (rewrite), `forms/ConfigForm.tsx`, `plan/EngineDialog.tsx`,
`plan/OptimizationDialog.tsx`, `plan/ScenarioDialog.tsx`, `plan/BioModelDialog.tsx`,
`plan/TissueDialog.tsx` (incl. a single-row mode used by B's VOI editor), `objectives/ObjectivesPanel.tsx`
(replaces the placeholder in `panels.tsx`), `optimization/StatusWindow.tsx`.
Methods: `plan.get {detail}`, `plan.options`, `plan.form`, `plan.set` (v1), `plan.tissue_check`,
`tissue.schema`, `tissue.evaluate`, `tissue.apply`, `cst.get`, `objectives.catalog`,
`objectives.add`, `objectives.update`, `objectives.change_type`, `objectives.remove`,
`objectives.set_quantity_all`, `workflow.optimize` (status window), `progress` events,
`session.history` (optimization part), `session.pause` / `session.resume` (native only),
`app.info.capabilities`.

### Package B: viewer parity, VOIs widget, analysis window, result snapshots

Components: `SliceViewer.tsx` + renderer (beams overlay and drag, pending beams, orientation
mannequin, compact masks), `VisualizationWidget.tsx` (beams toggle, quantity selector),
`QuantitySelector.tsx` (new), `StructureVisibility.tsx` (overlap grouping, tooltips, colour pick,
Edit VOI dialog `vois/EditVoiDialog.tsx` embedding A's tissue single-row mode), `analysis/AnalysisWindow.tsx`
(replaces `DvhDialog.tsx`: DVH plot, QI table, QI point dialog, gamma placeholder),
`store/revisions.ts`.
Methods: `view.volume` (v1 encodings and component keys), `view.structure_masks` (v1 encodings),
`view.beams`, `cst.get`, `cst.update_structure`, `analysis.dvh` (v1), `analysis.qi`,
`analysis.compare_units`, `state.results`, `state.revisions`, `workflow.snapshot` (the Snapshot
Result button lives in the workflow widget but its data model is B's; B adds the button handler in
`results/snapshot.ts`, C only wires the button).

### Package C: workflow import/export, menus, settings, help, info, log, UI mode, splash

Components: `MenuBar.tsx` (File / View / Base Data / Settings / Help), `WorkflowWidget.tsx` (Load
Dij, Snapshot button wiring, warnings-after-run dialog, disabled stubs with capability tooltips),
`io/SaveDialogs.tsx` (Save Plan / Dij / CST / Result / Workspace incl. Save Result quantity
checklist and pending-edit question), `basedata/BaseDataDialog.tsx`, `settings/PreferencesDialog.tsx`,
`help/LicensesDialog.tsx`, `help/AboutDialog.tsx`, log level and history in `panels.tsx`
`LogConsole`, UI-mode setting and dark mode (`theme.ts`), splash wording, close-while-busy prompt.
Methods: `data.load_file` (v1), `data.load_dij`, `export.save`, `base_data.list`, `base_data.add`,
`base_data.remove`, `app.settings`, `app.settings_set`, `app.licenses`, `app.info` (v1
capabilities, formats), `session.log_level`, `session.history` (log part), `workflow.snapshot`
(button only), `state.actions` (v1 `save_*` entries).
