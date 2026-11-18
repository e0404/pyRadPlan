# pyRadPlan on Pyodide

Headless harness and browser bootstrap for running pyRadPlan in
[Pyodide](https://pyodide.org) (CPython 3.14 on WebAssembly) with the numpy backend.
Findings, measurements and open problems: [FINDINGS.md](FINDINGS.md).

## Quick start

```bash
cd web/pyodide
npm install                      # pyodide 314.0.8 (Python 3.14.2)
PYTHON=/path/to/python npm run build   # wheels + manifest.json into web/public/pyrp/
npm run pipeline -- --mode photons --preset fast
```

`npm run build` needs a native Python with `pip` and network access (the build backend,
setuptools-scm, and the pinned pure-Python wheels are fetched from PyPI). `PYTHON` defaults to
`python`. The first `npm run pipeline` downloads the Pyodide distribution packages from the
jsDelivr CDN into `.cache/packages/`; later runs are offline.

| Command | What it does |
|---|---|
| `npm run build` | `build.py`: pyRadPlan wheel, SimpleITK shim wheel, vendored pure wheels, `manifest.json` in `web/public/pyrp/` (`-- --skip-pure` keeps the vendored wheels) |
| `npm run pipeline -- [options]` | `run_pipeline.mjs`: boot Pyodide in Node, install via `bootstrap.js`, run `pipeline.py`; writes `results/pyodide_<mode>_<preset><tag>.{npz,json}` |
| `npm run native -- [options] --out results/x.npz` | the same `pipeline.py` with the native Python (checkout on `PYTHONPATH`); `--sitk-shim` swaps in the shim |
| `npm run compare -- a.npz b.npz` | relative differences of dose cubes, unit-weight dose, weights, dij column sums, stf geometry |
| `npm run test:shim` | `tests/`: the shim against real SimpleITK, the native pipeline with shim vs. real SimpleITK, and pickles across both runtimes (needs SimpleITK and pytest in `PYTHON`) |
| `npm run app-smoke` | rebuild the wheels (`--skip-pure`), pickle a TG119 workspace natively (`npm run native-pickle`), then `app_smoke.mjs`: drive `pyRadPlan.app.dispatch` under Pyodide through `app_smoke.table.js`, print ok/error and timing per entry, exit 1 if a v0 entry fails |
| `npm run app-smoke:run -- [--strict] [--only v0\|v1] [--verbose]` | the smoke run alone on the current wheels; v1 failures are reported as pending unless `--strict` |

Pipeline options (both `run_pipeline.mjs` and `pipeline.py`): `--mode photons|protons`,
`--preset default|demo|fast` (see `PRESETS` in `pipeline.py`), `--solver scipy|ipax`,
`--max-iter N` (default 200), `--no-dij-sampling` (photons; makes the dij deterministic up to
floating point, for comparisons), `--dij-seed N`. Harness-only: `--plotting` (also load
matplotlib), `--install-only`, `--via-http` (serve `web/public/pyrp` on localhost and install
from URLs, as the browser does), `--cache-dir DIR`, `--tag SUFFIX`, `--verbose`.

Example comparison:

```bash
npm run native -- --mode photons --preset fast --no-dij-sampling --out results/native_photons_fast_nosamp.npz
npm run pipeline -- --mode photons --preset fast --no-dij-sampling --tag _nosamp
npm run compare -- results/native_photons_fast_nosamp.npz results/pyodide_photons_fast_nosamp.npz
```

## Bootstrap API (`bootstrap.js`)

Plain ESM without Node imports, shared by the Node harness and the browser worker.

```js
export const PYODIDE_VERSION      // "314.0.8"
export const PYODIDE_CDN          // jsDelivr indexURL of that version
export const PYODIDE_PACKAGES     // packages from the Pyodide distribution
export const PYODIDE_PLOT_PACKAGES // ["matplotlib"], only with { plotting: true }
export const PURE_WHEELS          // pinned pure-Python deps ("name==version"), vendored by build.py
export const PYRADPLAN_ENV        // env vars set before import (numpy backend, no GPU, no numba)
export async function installPyRadPlan(pyodide, { wheel, shimWheel, pureWheels?, plotting?, onProgress? })
  // -> { version, timings }; each wheel is a URL string or { name, bytes }
export function wheelsFromManifest(manifest, baseUrl)
  // -> { wheel, shimWheel, pureWheels } as absolute URLs for installPyRadPlan
```

Browser worker sketch (Vite serves `web/public/` at the site root):

```js
import { loadPyodide } from "pyodide";   // or import(`${PYODIDE_CDN}pyodide.mjs`)
import { installPyRadPlan, PYODIDE_CDN, wheelsFromManifest } from "../pyodide/bootstrap.js";

const pyodide = await loadPyodide({ indexURL: PYODIDE_CDN });
const base = new URL("pyrp/", self.location.origin + import.meta.env.BASE_URL).href;
const manifest = await (await fetch(base + "manifest.json")).json();
await installPyRadPlan(pyodide, {
  ...wheelsFromManifest(manifest, base),
  onProgress: ({ stage, fraction }) => postMessage({ event: "progress", stage, fraction }),
});
const { dispatch } = pyodide.pyimport("pyRadPlan.app");
```

Pass absolute URLs (micropip fetches them). Without `pureWheels`, micropip resolves
`PURE_WHEELS` from PyPI instead of the vendored copies. If Vite pre-bundles the `pyodide`
npm package, exclude it (`optimizeDeps: { exclude: ["pyodide"] }`) or load it from the CDN.

## Layout

- `shims/SimpleITK/`: numpy/scipy stand-in for the SimpleITK subset pyRadPlan uses (SimpleITK
  has no WebAssembly build). Installed only into Pyodide, as the `pyradplan-sitk-shim` wheel that
  provides the import name `SimpleITK`; anything outside the subset raises
  `NotImplementedError` / `AttributeError` naming the shim. Images pickle exactly like
  SimpleITK 2.x (class path `SimpleITK.SimpleITK.Image`, same reduce tuple and state version 0,
  `shims/SimpleITK/SimpleITK.py` mirrors the module), so workspace pickles move between the
  browser and native pyRadPlan in both directions.
- `tests/`: shim vs. real SimpleITK (pixel types, geometry, slicing, operators, casts, masks,
  morphology, label statistics, resampling on aligned / shifted / rotated grids, TG119 masks)
  and the end-to-end native pipeline with shim vs. SimpleITK (bit-identical).
- `app_smoke.table.js`: the smoke method table (`method`, `params`, `v`, `expect`, `setup`,
  `needs`, or `python` for a code step); extend it as `pyRadPlan.app` grows.
- `tools/trace_sitk.py`: records which SimpleITK functions and methods a native run calls.
- `tools/run_python.mjs`: runs `$PYTHON` with the checkout on `PYTHONPATH`.
- `results/`, `.cache/`, `node_modules/` and `web/public/pyrp/*` are build output (ignored).
