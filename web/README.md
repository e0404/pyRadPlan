# pyRadPlan web frontend (prototype)

A browser GUI for pyRadPlan that mimics the PySide6 main window and runs pyRadPlan itself in the
browser via [Pyodide](https://pyodide.org) in a web worker. The UI talks to the Qt-free application
layer `pyRadPlan.app` through the protocol in [PROTOCOL.md](PROTOCOL.md). **Not for clinical use.**

## Setup

Requires Node ≥ 20 (developed with Node 24) and npm.

```sh
cd web
npm install
```

On this machine Node is portable: `export PATH="/d/Python/pyRadPlan-web-tools/node:$PATH"`.

## Run

```sh
npm run dev            # http://localhost:5173/  (COOP/COEP headers set)
```

- `http://localhost:5173/?backend=mock` — mock backend (synthetic phantoms, fake dose, simulated
  progress), no Python needed. `&speed=0` makes computations instant, `&speed=3` slower.
- `http://localhost:5173/` or `?backend=pyodide` — the real backend: Pyodide + pyRadPlan in a worker.
  Needs the wheels and `manifest.json` in `public/pyrp/` (built by `web/pyodide/build.py`) and network
  access to the Pyodide CDN for numpy/scipy. The first start downloads ~40 MB.

Other URL parameters: `pyodide=cdn|<url>` (Pyodide runtime location), `packages=<url>` (package wheel
mirror), `pyrp=<url>` (pyRadPlan wheel directory), `sliceMode` (force the renderer's per-slice texture
fallback). `VITE_BACKEND=mock` changes the default backend at build time.

```sh
npm run build          # type check + production build into dist/ (includes dist/pyodide-runtime/)
npm run preview        # serve dist/ with the same headers on http://localhost:4173/
```

## Tests

```sh
npm test               # vitest: codec, mock backend, renderer math (geometry, LUTs, marching squares)
npx tsc --noEmit       # type check
npm run e2e            # Playwright (Edge, SwiftShader WebGL) against ?backend=mock; screenshots in e2e/screenshots/
npm run e2e:viewer     # viewer parity (beams, drag, snapshots, VOIs, analysis window) against the mock, port 5202
npm run e2e:plan       # plan widget, config forms, tissue dialog, objectives, status window against the mock
npm run e2e:menus      # menus, dialogs, saves (downloads), loads, base data, settings, help, log, dark mode (port 5203, screenshots menus-*.png)
npm run e2e:pyodide    # the whole GUI against the real Pyodide backend (~3 min, see below); `-- --preview` uses the production build
```

`e2e:pyodide` clicks through TG119 with the web profile defaults in stages (boot, phantom, default
Run All, plan form + engine / scenario / optimization dialogs, objectives, 5-beam Run All with the
status window, viewer, snapshot + analysis window, VOI edit, every File-menu save the capabilities
report available (downloads kept in `e2e/downloads/`), reload of the workspace pickle and the plan
`.mat`, Load Dij, base data add/remove, Preferences, licenses, About, log level, dark mode,
educational mode, a proton plan with its bio models and the tissue dialog, Stop during the dose
calculation). Each stage screenshots `pyodide-*.png`; a failing stage is recorded
(`pyodide-fail-<stage>.png`) and the run continues; `--only=<stage,...>` runs a subset. Timings
and memory go to `e2e/screenshots/pyodide-metrics.json`. It needs the wheels in `public/pyrp/`
(`npm run build -- --skip-pure` in `web/pyodide`) and network access to the Pyodide CDN.

The backend calls themselves are covered without a browser by `npm run app-smoke -- --strict` in
`web/pyodide` (rebuilds the wheels, then runs `app_smoke.table.js` under Pyodide in Node; needs
`PYTHON=<python with the build deps>`).

The e2e scripts use the installed Microsoft Edge (`channel: "msedge"`); `npm run e2e -- --chromium`
uses Playwright's own Chromium instead (`npx playwright install chromium` first), `-- --headed` shows
the browser.

## Layout

```
web/
  PROTOCOL.md              UI <-> pyRadPlan.app protocol (shared contract)
  index.html, vite.config.ts, tsconfig.json, package.json
  src/
    main.tsx, App.tsx      entry, main window layout (menu, three columns, status bar, dialogs)
    styles.css             Qt-like look (group boxes, dense controls)
    assets/                logos copied from pyRadPlan/gui/assets/logos
    api/                   typed protocol client
      protocol.ts          message / method types (incl. PROTOCOL.md amendments)
      codec.ts             {$array} <-> typed arrays
      backend.ts           Backend interface, BaseBackend, BackendError
      mock/                MockBackend + synthetic phantoms / fake dose
      pyodide/             PyodideBackend, module worker, glue.py (runs inside Pyodide)
    store/appStore.ts      zustand store: session state, volumes, logs, viewer settings, actions
    renderer/              framework-free slice renderer (WebGL2 + Canvas2D overlay)
    components/            React widgets mirroring the Qt widgets
  test/                    vitest unit tests
  e2e/                     Playwright scripts + screenshots
  docs/renderer.md         renderer options, decision, verified limits
  docs/frontend.md         architecture, worker, Pyodide loading, protocol interpretations
  scripts/gen_colormaps.py regenerates src/renderer/colormapData.ts from matplotlib
  pyodide/                 (porting agent) bootstrap.js, wheel build, headless harness
  public/pyrp/             (generated) pyRadPlan wheel, SimpleITK shim, pure wheels, manifest.json
```

## Viewer controls

Wheel: next/previous slice · Ctrl+wheel: zoom at cursor · left/middle drag: pan · right drag: zoom ·
double-click: reset view · ↑/↓, PgUp/PgDn: slices (canvas focused) · hover: position, HU and dose.
