# Frontend architecture

```
React components (src/components)          Qt widgets they mirror
  └─ zustand store (src/store/appStore.ts)  WorkspaceManager + widget state
       └─ Backend (src/api/backend.ts)       typed request / events / boot status, array codec
            ├─ MockBackend   (src/api/mock)      pure TS, synthetic phantoms + fake dose
            └─ PyodideBackend (src/api/pyodide)  module worker → Pyodide → pyRadPlan.app.dispatch
  └─ SliceRenderer (src/renderer)            framework-free WebGL2 + Canvas2D, see renderer.md
```

## Backend interface

`Backend.request(method, params, session)` is typed by the method table in `src/api/protocol.ts`
(`Methods`), mirroring `web/PROTOCOL.md` including its amendments. `BaseBackend` owns what both
implementations share: request ids, the `{$array}` codec (`src/api/codec.ts`), event fan-out
(`subscribe`) and boot status (`onBoot`). Implementations only move wire messages (`transmit`).
Errors come back as `BackendError` with the Python exception `type`, `message`, `traceback` and
`code` (e.g. `PreconditionError` / `bio_model_unsupported`).

The codec decodes `{"$array": i, "dtype", "shape"}` into `{dtype, shape, data: TypedArray}` views on
the transferred buffers (zero-copy unless a buffer is misaligned for its dtype) and encodes request
arrays the same way. A request's buffers are *transferred* to the worker, so the caller's array is
detached afterwards (only `data.load_file` sends arrays).

The mock produces real wire messages (results are encoded, then decoded by `BaseBackend`), so the
codec path is exercised in tests and in the UI with `?backend=mock`.

The mock's static catalogues (registries, plan options, configuration forms, tissue schema,
objectives catalogue) are recordings of `pyRadPlan.app` in `src/api/mock/fixtures/`, which
`scripts/gen_mock_fixtures.py` writes (`npm run fixtures`, run by npm pre-hooks before `dev`,
`build`, `typecheck`, `test` and the `e2e*` scripts). They are generated, not committed, while the
session API is still changing; building the frontend therefore needs a Python with pyRadPlan's
dependencies (`PYTHON` selects the interpreter). Once the API is stable, committing them with a
drift test in the Python test suite (regenerate and compare) lets the frontend build without
Python.

## Pyodide worker

`src/api/pyodide/pyodide.worker.ts` (module worker) boots in this order:

1. `import(<indexURL>/pyodide.mjs)` and `loadPyodide({indexURL, packageBaseUrl, stdout, stderr})`.
2. `setInterruptBuffer(Int32Array(SharedArrayBuffer))` when the page is cross-origin isolated.
3. `fetch(<pyrp>/manifest.json)`, then `wheelsFromManifest(manifest, <pyrp>/)` and
   `installPyRadPlan(pyodide, {wheel, shimWheel, pureWheels, onProgress})` from
   `web/pyodide/bootstrap.js` (bundled into the worker at build time via `import.meta.glob`).
   `onProgress({stage, fraction, detail?})` drives the splash.
4. Runs `glue.py` in a fresh namespace: imports `pyRadPlan.app`, disables tqdm console bars
   (`ProgressReporter.console_progress = False`) and defines `web_dispatch`.

Each request is handled by `web_dispatch(request_json, buffers, emit)`: the JSON part is parsed in
Python, `request["buffers"]` becomes a list of `bytes`, `pyRadPlan.app.dispatch(request, emit)` runs
synchronously (an awaitable result would be awaited), and events go out through `emit` as JSON
strings while the request runs (the main thread stays responsive and receives them immediately).
The response comes back as a JS array `[json, *Uint8Array]`: one copy out of the wasm heap per
buffer (unavoidable, wasm memory cannot be transferred), then the `ArrayBuffer`s are transferred to
the main thread without a further copy. Requests are queued and handled one at a time.

**Stop** writes `2` (SIGINT) into the interrupt buffer: Python raises `KeyboardInterrupt` at the next
bytecode check and `dispatch` turns it into an error response, which the UI shows as "… stopped"
in the status bar and log (no error dialog). Verified in the Pyodide e2e run: the dose influence
calculation stopped ~0.1 s after the click and the session stayed consistent (dij stale). Long
numpy/scipy calls in C are not interrupted until they return. Without cross-origin isolation the Stop
button is disabled.

## How Pyodide is loaded

Chosen: **npm package `pyodide@314.0.8`, runtime self-hosted, package wheels from the jsDelivr CDN.**
The Vite plugin in `vite.config.ts` serves `pyodide.mjs`, `pyodide.asm.mjs`, `pyodide.asm.wasm`,
`python_stdlib.zip` and `pyodide-lock.json` from `node_modules/pyodide` at `/pyodide-runtime/` in dev
and copies them into `dist/pyodide-runtime/` on build. The npm package does not contain the package
wheels (numpy, scipy, h5py, pydantic, …), so `packageBaseUrl` points at
`https://cdn.jsdelivr.net/pyodide/v314.0.8/full/` (jsDelivr sends CORS and CORP headers, so it works
under `Cross-Origin-Embedder-Policy: require-corp`; verified in the Pyodide e2e run). The npm package
also provides the TypeScript types and pins runtime and lock file to one version.

Alternatives, selectable by URL parameter:

- `?pyodide=cdn`: runtime from the CDN too (no self-hosting at all).
- `?pyodide=<url>` / `?packages=<url>`: any mirror, e.g. a fully self-hosted copy of the Pyodide
  "full" distribution (≈ 300 MB unpacked, only the used wheels are fetched) for offline / intranet
  installs. That is the route for a production deployment that must not depend on jsDelivr.
- `?pyrp=<url>`: other location of the pyRadPlan wheels + `manifest.json` (default `/pyrp/`).

Outside the dev server (`import.meta.env.DEV`), `pyodide`, `packages` and `pyrp` are honoured only
for URLs below the app's own directory (`BASE_URL`, relative ones included) and for `pyodide=cdn`
(`trustedSource` / `backendSources` in `src/api/index.ts`): on a public deployment a crafted link must
not make the app run code from another site, nor from another project on a shared origin (GitHub
Pages) or the other channel. A deployment using a mirror serves it inside the app's directory.

`?backend=mock` is honoured only in the dev server and in builds made with `VITE_ENABLE_MOCK=1`; the
mock is loaded with a dynamic import that a production build drops, and an active mock shows a banner.

## COOP / COEP

Dev and preview servers send `Cross-Origin-Opener-Policy: same-origin`,
`Cross-Origin-Embedder-Policy: require-corp` (and `Cross-Origin-Resource-Policy: same-origin`), so
`crossOriginIsolated` is true and `SharedArrayBuffer` is available (status bar shows it). A
production host must send the same headers (or use a service-worker shim such as coi-serviceworker).

## State flow

- Events: `state` replaces `SessionState`; `progress` drives the workflow progress bar (determinate
  when `fraction` is a number); `log` goes to the Log panel (with the logger name when given).
- Workflow requests run through one `run()` helper: busy flag (disables inputs), progress, status
  bar text, error dialog + log entry on failure. After `data.*` the CT and masks are refetched
  (`view.volume ct`, `view.structure_masks`) and the viewer jumps to the isocenter / target centre
  slices; after `workflow.optimize|recalc|run_all` every name in `quantities` is fetched with
  `view.volume`. SessionState carries no revision numbers, so refetching is driven by which
  method ran.
- `PreconditionError` with `code: "bio_model_unsupported"` opens a confirm dialog and retries with
  `physical_only: true`, like the Qt dialog.

## Protocol interpretations

- Requests carry `buffers` like responses (amendment 1); the worker hands them to Python as `bytes`.
- The workflow grid follows the Qt widget's columns; the protocol's five stages map to its three
  dots: Loading = `data`, Dose Influence = worst of `stf`/`dij` (missing if `dij` is missing),
  Result Computation = `optimization`. The buttons are the Qt ones in the Qt places (Load File /
  Load Folder / Import from Binary, Calc. Dose Influence / Load Dij, Optimize / Recalculate Dose /
  Run Sequencing, Snapshot Result); there is no Run All, Generate STF or phantom selector, phantoms
  are opened from File > Open Example Phantom as in Qt. "Load Folder" (capability reason) and "Import from Binary"
  ("Not yet implemented", as Qt) are disabled. While busy the status label shows the progress label,
  detail and fraction (Qt shows the progress levels there); there is no separate detail line.
  `workflow.run_all` and `workflow.generate_stf` stay protocol methods (tests, app smoke).
  Calc. Dose Influence generates the stf again every time (v1 amendment 20), so its dot turns
  green for both `stf` and `dij`; a structure edit leaves the stf "done" (as in Qt) and the next
  calculation still places the beams at the new automatic isocenter.
- Run Sequencing (amendment 29) is enabled by `state.sequencing.available`, not by
  `state.actions`; while disabled its tooltip is `state.sequencing.reason`. It runs
  `workflow.sequence` like the other steps (busy bar, Stop, unapplied plan edits offered first)
  and the views refetch by revision: the active result gains the `"<identifier>_sequenced"`
  quantities, which `results/resultGroups.ts` names "<name> (sequenced)" for backends without
  `state.results`. A backend without `state.sequencing` or `workflow.sequence` leaves the button
  disabled.
- Show Sequence (amendment 30; Qt `SequenceWindow` with `ApertureViewerWidget`) is enabled by
  `state.sequencing.sequenced` (and `view.sequence` among `app.info.methods`).
  `components/sequence/SequenceWindow.tsx` is a non-modal window like the analysis window
  (`sequenceOpen` in the viewer slice). `showSequence()` (`store/appStore.ts`) opens it on the
  beam of the shown per-beam quantity (`"…_beam <n>"`); `actions.sequence` calls it after a
  successful run, as Qt opens the window when sequencing finishes. The window fetches
  `view.sequence` whenever `revisions.workspace.result` changes, resets its view options and
  players with each fetched sequence, and closes when the active result has none (`sequenced`
  false, or `PreconditionError` / `no_sequence`). Every beam tab keeps its own mounted player
  (shape, Play, speed, Loop), so a hidden tab keeps playing as in Qt. What is drawn is computed
  by pure functions in `sequenceModel.ts`, one per helper of
  `pyRadPlan/visualization/_plot_aperture_info.py` (leaf rows, blocked rectangles, weight
  colour, MLC outlines, bixel grid, colour limits, dwell time, titles incl. Python's `%.2g`);
  `SequencePlot.tsx` draws both panels on one canvas with the plot theme. The vertical axis
  increases upwards: row 0 of a fluence map and the first leaf row are the lowest stack
  position, bixels without a ray (NaN) stay empty, and the target outline is the 0.5
  marching-squares contour of `target.mask` (bin centres at `origin + index * spacing`). The
  aperture panel fills its half (matplotlib `aspect="auto"`), the fluence panel keeps the
  aspect ratio. Mock: `api/mock/sequence.ts` derives shapes, fluence maps and target from the
  phantom's target and the plan's beams.
- Structure visibility starts from `structures[].visible` and stays a viewer setting (not sent back).
- Volume `range` entries that are `null` fall back to `[0, 1]` (quantity) / `[-1000, 1000]` (CT).
- `iso_center` as a list of per-beam points: the viewer and the plan form show the first one; the plan
  form only sends fields that were edited, so per-beam isocenters survive unrelated edits.
- The Visualization panel sits below the slice viewer (as in `_main_win.py`), not in the left column.

## Revision-driven refetching (v1)

`src/store/revisions.ts` owns all viewer data fetching. Each `state` event schedules
`syncViews()`, which compares `state.revisions` with the revisions the held data was fetched at
(`appliedRevisions` in `store/viewerSlice.ts`) and refetches only what changed: the CT
(`revisions.volumes.ct`, `encoding: "auto"`), masks (`revisions.masks`, `encoding: "crops"`),
structure details (`revisions.workspace.cst` → `cst.get`), beams (`revisions.beams` → `view.beams`
with the plan form's draft), and the shown quantity plus every full quantity
(`revisions.volumes[key]`, also the `"<result>/<key>"` keys of snapshots and imported doses;
snapshot keys sharing a revision reuse the held array). After a result
change the shown quantity is reset to `results.default` (Qt ambiguity 6). Without `revisions`
(v0 backend) the caller's hint decides, as before (`run(..., "data" | "result")`). Other packages
call `syncViews` / `scheduleViewSync` / `selectQuantity` / `setDraftBeams` /
`registerBeamEditor`; the plan form registers the beam editor and reports its draft angles.

## Files, menus, settings and help (package C)

Modules: `src/api/protocol/io.ts` (types), `src/api/mock/io.ts` (mock handlers, capabilities of both
runtimes: `?runtime=native` flips the mock to a native runtime), `src/store/ioActions.tsx` (flows),
`src/store/{ioStore,dialogStore,pendingChanges,uiMode}.ts`, `src/theme.ts` + `theme.css`,
`src/components/{MenuBar,Splash}.tsx` and `components/{io,basedata,settings,help,log}/`.

- **Menus** are a pure model (`components/io/menuModel.ts`: entries, enabled flags, tooltips) that
  `MenuBar.tsx` renders. Enabling follows `state.actions.save_*` and `app.info.capabilities`;
  stubs are `aria-disabled` (so the tooltip with the reason still shows) and carry
  "Not available: <reason>" from the capabilities. Qt disables the whole File menu while busy.
- **Dialogs** are promise based: `showDialog(render)` (`store/dialogStore.ts`) stacks them in the
  `DialogHost` mounted by the menu bar; `components/io/messageBox.tsx` has the QMessageBox /
  QInputDialog look-alikes. Escape only closes the topmost one.
- **Saving** calls `export.save`, then `components/io/download.ts` offers the bytes through an
  anchor with the `download` attribute. The browser has no save dialog, so `askSaveFile` asks for
  format and file name (stand-in for QFileDialog); formats come from `capabilities.formats.export`
  (unavailable ones are listed disabled with their reason). Plan and workspace saves resolve pending
  plan edits first (`store/pendingChanges.ts`: editors register with `registerPendingSource`).
- **Loading** uses a file chooser (`pickFile`, needs the user gesture, so the first call of a flow is
  the chooser). No drag and drop: the Qt GUI has none.
- **Warnings after a run**: `log` events with `report: true` (the backend's `WarningReport`s, what
  Qt collects) received while busy are shown once when the operation ends (`installWindowHooks`);
  plain warning records only go to the log console, and `repeat: true` reports are not logged twice.
  The same hook asks before the page is closed while busy (`beforeunload`).
- **UI mode** (`store/uiMode.ts`) and **dark mode** (`theme.ts`) are frontend state; the dark colours
  are CSS variables on `:root[data-theme="dark"]` plus a few overrides of literal colours in
  `theme.css`. Dark mode follows `prefers-color-scheme` until the View menu is used. Canvas plots
  read the variables at draw time (`components/analysis/plotTheme.ts`, as Qt `_plot_theme.py`:
  `--window` around the axes, `--base` inside, `--text` for text, ticks, axes, cursor and marker
  outlines, grid = 30 % text over the base colour; hover box, legend and marker labels on the base
  colour with the grid colour as border) and redraw when `useTheme` changes.
- **Log**: the console filters by level on the client and sets the backend's `pyRadPlan` level with
  `session.log_level`; records carry `logger` and `seq`, `session.history` fills in what was missed.

## Package A: plan form, configuration forms, objectives, tissues, optimization status

Files: `src/api/protocol/plan.ts` (v1 types; the shared v1 block lives in `protocol.ts`),
`src/api/mock/{plan,planData}.ts`, `src/store/planSlice.ts` (own zustand store `usePlanStore`, no
`appStore` registration), `components/forms/` (`ConfigForm`, `Combo`, `ModalDialog`, `formModel.ts`),
`components/plan/` (`PlanWidget`, `ConfigDialogs`, `OptimizationDialog`, `TissueDialog` incl.
`TissueSingleRow`, `planModel.ts`), `components/objectives/`, `components/optimization/StatusWindow.tsx`.

- **Forms** are rendered from the backend `FormSchema` only (`formModel.ts`: descriptor → editor,
  text ↔ value, bounds / list-length checks, values = names in `explicit` or edited, groups as nested
  override dicts). Unlike Qt, OK is refused while an entry is invalid (Qt keeps the last valid value).
- **Combo boxes** are a Qt-like component (`forms/Combo.tsx`): identifiers faint in the open popup,
  disabled entries with their reason as tooltip, section headers, an editable variant.
- **Plan form**: the draft keeps the Qt per-algorithm entries (engine / scenario / problem / bio
  parameters); "modified" compares a snapshot like `_capture_state`; Apply sends only changed fields
  (`planModel.planSetParams`), a new radiation mode also sends machine, engine and model (the
  backend keeps scenario model and problem settings itself). Options are `plan.options` for the
  draft's mode / machine / engine (cached; without `plan.options` they are derived from
  `app.info`); the bio-model entry selected is the backend's `bio_selection` (the form sends its
  previous selection as `current_bio` and applies the answer, `planModel.applyBioSelection`).
  Changing mode / machine / engine re-lists asynchronously; Apply awaits
  the pending re-list so a click right after a mode change cannot send the old mode's engine. The
  tissue dialog and its "…" checks send the unapplied form as `draft`. The form registers as
  pending-changes source and as beam editor and reports edited angles with `setDraftBeams`.
  The sequencing row (Run sequencing, sequencer, "…") lists `plan.options.sequencers` and is
  disabled for a radiation mode without any (`planModel.applySequencers`, Qt
  `_update_sequencing_availability`); its three fields are tracked like the others and sent
  together as `plan.set {sequencing}`, which outdates no stage.
- **Objectives** read `structureDetails` (filled from `cst.get` by `revisions.ts`) and merge the
  `structure` returned by every `objectives.*` write.
- **Status window** opens when an optimization starts (`busy` label or the first `progress.data`
  metric), plots `progress.data.objective` per `iteration`, and is restored from `session.history`
  when a session is (re)connected.
- **Mock** feature handlers form a chain (`FEATURE_HANDLERS` in `mockBackend.ts`, `ctx.fallback()`
  calls the next one, finally the v0 switch); `STATE_EXTENSIONS` add v1 state fields.

## Integration pass: the whole GUI on real Pyodide

`npm run e2e:pyodide` (`e2e/pyodide.mjs`) clicks through the GUI on TG119 with the web profile
(5 × 5 × 5 mm dose grid, `total` doses) against the real worker: boot, phantom, plan widget and its
engine / scenario / optimization dialogs, objectives (add, edit, change type, remove), Calc. Dose
Influence + Optimize with the status window, sequencing of the default plan (sequencer form, Run
Sequencing, the sequenced dose, the sequence window checked against the `view.sequence` data),
5 beams, viewer (beams, quantity selector incl.
per-beam, isolines, planes), snapshot + analysis compare, VOI edit, every available File-menu save,
reload of workspace pickle / plan .mat / Load Dij, base data add + remove, Preferences, licenses,
About, log level, dark mode, educational mode, a robust 1-beam plan (worst-case grid, voxel-wise
worst-case objective, 15 iterations) with a scenario shown in the viewer and the analysis window
("All scenarios"), a proton plan with tissue dialog and bio model list, and Stop during a dose
calculation. Stages can be selected with `--only=a,b` (boot and load always run); browser console
errors fail the run; metrics go to `e2e/screenshots/pyodide-metrics.json`, downloads to
`e2e/downloads/` (git-ignored). `npm run app-smoke -- --strict` (in `pyodide/`) covers the v1
write calls (plan.set, objectives.*, cst.update_structure, base_data.*, export.save, load_dij …)
on Node Pyodide without the GUI.

Measured on the development machine (Edge, SwiftShader, Pyodide 314.0.8, 2026-11-18):

| Case | Time | wasm heap afterwards |
| --- | --- | --- |
| Boot (worker + wheels, until the GUI is ready) | 10–12 s | 160–190 MB |
| Run All, default plan (1 beam, photons, 5 mm) | 6.5–9 s | 460–560 MB |
| Run All, 5 beams, photons, 5 mm | 32–38 s | 860–910 MB |
| Run All, 5 beams, protons (HongPB, constant RBE), 5 mm | ~43 s | ~1.9 GB (heap never shrinks) |
| Run All, 1 beam, protons, 5 mm (fresh session) | ~15 s | ~660 MB |
| Dose influence + 15 iterations, 1 beam, photons, 5 mm, worst-case grid (9 scenarios), voxel-wise worst-case objective (fresh session) | ~33 s | ~1.0 GB |
| the same after the 5-beam plan (stage `robust` of the full run) | ~36 s | ~2.0 GB |

Saves take 1–3.5 s each; `dij.mat` is ~127 MB and the workspace pickle ~335 MB at 5 beams. The
wasm heap only grows, so a long session approaches the 4 GB wasm32 limit;
`performance.measureUserAgentSpecificMemory` reports only 170–280 MB and undercounts the worker
heap.

Known core issue found by the run (not web specific, Qt behaves the same): a structure with
objectives that its overlap priority empties (e.g. TG119 Core at priority 4, under BODY at 3)
makes the objective NaN at iteration 0; the optimizer stops and the result is zero (protons) or the
initial weights (photons), although the log says its objectives "will not be considered". The e2e
restores the priority after testing the VOI edit.
