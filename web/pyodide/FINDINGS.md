# pyRadPlan on Pyodide: findings

Prototype status, 2026-11-18, branch `dev/web-pyodide` (from `develop`, pyRadPlan 0.8.0). Measured
on Windows 11, Node 24.22.3, Pyodide 314.0.8 (CPython 3.14.2, Emscripten 5.0.3, wasm32). Reproduce
with the commands in [README.md](README.md).

## Summary

- **The standard pipeline runs end to end under Pyodide, for photons and protons.** That covers
  TG119 load, plan, STF, dij, fluence optimization (SciPy L-BFGS-B and ipax) and the dose on the CT
  grid, with three presets each, from 1 beam at defaults to 5 beams. `pyRadPlan.app` also runs:
  `app.info`, `data.load_phantom`, `plan.set`, `workflow.run_all`, `view.volume`, `analysis.dvh`
  and `view.structure_masks` all return ok.
- **The only blocker was SimpleITK.** There is no WebAssembly build. A numpy/scipy stand-in
  (`shims/SimpleITK`, about 1100 lines, outside pyRadPlan) provides the subset in use.
  - Natively, the full pipeline is bit-identical with the shim and with SimpleITK, for photons and
    protons (`tests/test_pipeline_shim.py`).
  - 115 focused tests compare the shim with SimpleITK 2.5.6. All resampling cases agree to 1e-12
    relative; nearest-neighbour cases agree exactly.
- **Numerical faithfulness:**
  - With photon dij sampling off (it is random), the Pyodide and native dij agree to 1e-7 to 2e-6
    relative L2. The cause is measured 1–3 ulp libm/SIMD differences in transcendentals.
  - With sampling on (the default), the ulp differences flip sampling decisions, and the optimized
    dose differs about as much as two native runs with different seeds.
- **Speed:** 1.5–2.3× native (native also with numba off). The fast preset (5 beams) takes 5–6 s;
  5-beam default-resolution plans take 21–24 s. Boot plus install is about 5 s with warm caches.
- **Size:** about 71 MB per cold start: runtime 13.5, distribution packages 21.0, pure wheels 4.0,
  pyRadPlan wheel 32.4. 31.3 MB of the pyRadPlan wheel is bundled `.mat` data (TG119 23, machines
  8.3); the code is under 1 MB.
- **Peak WebAssembly memory is 0.57–1.79 GB,** close to native RSS. 5-beam protons at 5 mm use
  1.79 GB, near the wasm32 4 GB ceiling.

## Incompatibilities and resolutions

| # | Problem | Resolution | Where |
|---|---|---|---|
| 1 | SimpleITK: not in the Pyodide 314.0.8 distribution (357 packages); no pyodide/emscripten/wasm32 wheel on PyPI (newest 2.5.7). `sitk.Image` stores CT, masks and dose (53 modules). | numpy/scipy shim, installed only in Pyodide as the separate wheel `pyradplan-sitk-shim` (import name `SimpleITK`). | `shims/SimpleITK`, `tests/` |
| 2 | micropip dependency resolution of pyRadPlan fails (`Can't find a pure Python 3 wheel for: simpleitk`) and would pull in `pydantic_ai`, `huggingface_hub` and `safetensors`. | Install with `deps=False` from an explicit pinned list (`PURE_WHEELS`, vendored by `build.py`) plus Pyodide-built `PYODIDE_PACKAGES`. | `bootstrap.js`, `build.py` |
| 3 | matplotlib was imported eagerly by `cst/_voi.py` (only color names), `analysis/_dvh.py` and `_qi.py` (only plotting); contourpy by `io/dicom/_export_cst.py` (only RTSTRUCT export). | Imports moved into the functions that use them (4 small edits). The pipeline, io, analysis and app import without matplotlib, saving 10.3 MB. Plotting still works with `plotting: true` (Agg, `plot_slice` to PNG verified). | 4 pyRadPlan files |
| 4 | IPOPT is unavailable. The default `solver = "ipopt"` falls back, with a warning, to the first registered solver (ipax). | No change. ipax 0.13.0 (pure wheel) and SciPy work. The app should choose the solver explicitly. | — |
| 5 | numba is unavailable. | The import was already guarded; `jit_backends = ""` is set anyway. | `bootstrap.js` |
| 6 | Threads: the tqdm monitor thread cannot start (a warning). | `tqdm.tqdm.monitor_interval = 0`. `KeyboardListener` does not start a thread (stdin is not a tty). The `native_output_to_logger` thread pump is used only by IPOPT. `threading.Event` works. No multiprocessing on the path. | `bootstrap.js` |
| 7 | No GPU backends. | `PYRADPLAN_XP_PREFER_GPU=false` and `PREFERRED_CPU_ARRAY_BACKEND=numpy` before import, and settings set again after. `app.info` reports `backend: numpy`. | `bootstrap.js` |
| 8 | No bytecode: `sys.dont_write_bytecode=True` and micropip wheels have no `.pyc`. Compiling all of pyRadPlan takes 0.9 s and the pure deps 2.4 s per session. | Not fixed (see recommendations). | — |
| 9 | Image file IO (`ReadImage`, `WriteImage`, `ImageSeriesReader`) is not in the shim. | Unsupported. `.mat` import works (pymatreader, h5py, scipy.io). | — |
| 10 | Pickles did not cross runtimes: a native pickle failed in the shim with `No module named 'SimpleITK.SimpleITK'`, and a shim pickle failed natively with `No module named 'SimpleITK._image'`. | The shim now writes and reads SimpleITK 2.x's own pickle format: class `SimpleITK.SimpleITK.Image` (a mirror module `shims/SimpleITK/SimpleITK.py`), constructor args `(size, pixel_id, components)`, state `(0, buffer, origin, spacing, direction, metadata)`, `PickleBuffer` for protocol 5. Image metadata (`Get/SetMetaData`) is kept. Both directions are byte-compatible and tested for protocols 2, 4 and 5, for every pixel kind used (int16, uint8, float32 vector, float64 with rotation and metadata, 4D, 2D int64). A TG119 `ct`/`cst` round trip through `pyRadPlan.io` pickle save/load also passes in both directions, and a native workspace pickle loads under Pyodide (`npm run app-smoke`). | `shims/SimpleITK`, `tests/test_pickle_compat.py` |
| 11 | DICOM RTSTRUCT export needs `contourpy`. | Added to `PYODIDE_PACKAGES` (0.12 MB wheel, depends only on numpy). RTSTRUCT export itself was not run under Pyodide. | `bootstrap.js` |

The AI dependencies are not installed. `pyRadPlan.ai` imports lazily; `pyRadPlan.gui` imports but
needs Qt.

## SimpleITK shim

**Scope**, sized by `tools/trace_sitk.py`, which records the SimpleITK calls of native photon and
proton runs:

- **Image** (2D–4D, scalar and vector, all integer and float pixel types): size, spacing, origin
  and direction, pixel-ID API, `CopyInformation`, `MakeUnique`, index/physical-point transforms,
  `Get`/`SetPixel`. Slicing follows SimpleITK's origin and spacing rules (`img[:, :, :, i]`, steps,
  collapsed axes). Operators follow SimpleITK's type rules: constants are cast to the pixel type,
  `/` gives float64, comparisons give uint8. Deepcopy works; pickles are SimpleITK-compatible (#10); images are unhashable.
- **Array conversion:** `GetArrayFromImage` (copy), `GetArrayViewFromImage` (read-only view),
  `GetImageFromArray` (bool rejected).
- **Pixel-wise and structural:** `Cast` (static_cast: truncation and wrap-around),
  `Mask`/`MaskImageFilter`, `And`, `Or`, `Xor`, `Not`, `Equal`, `NotEqual`, `Greater`, `Less`,
  `Add`, `Subtract`, `Multiply`, `JoinSeries`, `Extract`.
- **Resampling:** `ResampleImageFilter` and `Resample` (reference and geometry forms), identity
  `Transform` only. Interpolators: nearest neighbour, linear, B-spline orders 1–5. Also the default
  pixel value (plain static_cast), the nearest-neighbour extrapolator, the output pixel type
  (bounds-checked cast) and vector images.
- **Morphology:** `BinaryDilate` (box, made separable; and cross), `BinaryThreshold`,
  `ConnectedComponent`, `BinaryFillhole`.
- **LabelShapeStatistics:** count, centroid, bounding box, physical size, principal moments and
  axes, elongation, flatness, equivalent ellipsoid diameter, spherical radius.
- **Constants:** values from SimpleITK 2.5.

Everything else raises `NotImplementedError`, or an `AttributeError` that names the shim, so
`hasattr` keeps working.

**Accuracy.** Resampling reproduces ITK's arithmetic exactly:

- the continuous index is interpolated linearly along each output scanline (the order pyRadPlan's
  `ArrayBackendGridResampler` uses);
- the buffer test is `[-0.5, N-0.5)`, with edge clamping inside it;
- rounding is `floor(x + 0.5)`;
- zero-weight neighbours never propagate NaN;
- linear blending runs x, then y, then z.

For axis-aligned grids, a separable path does the same floating-point operations on whole planes.
Testing found two ITK details: label statistics add no voxel moment, and the default pixel value is
cast without clamping.

**Performance.** Natively the shim is slower than ITK for full-volume resampling (fast preset
`compute_result_ct_grid` 0.10 → 0.49 s). The other steps are on par after making box dilation
separable (1.3 s → 0.01 s). Under Pyodide, the shim's resampling takes about 1 s of a 38 s cProfile
run.

## Numerical agreement

**Reference.** Native is CPython 3.11, numpy 2.3.5, scipy 1.17.1, real SimpleITK, numba off. A
control with CPython 3.14.6, numpy 2.4.6 and scipy 1.18.0 plus the shim is bit-identical to it, so
versions and shim contribute nothing; every difference comes from the wasm runtime.

**Root cause.** On 200 001 points, transcendentals differ by 1–3 ulp:

- float32 `exp` on 39 % of inputs (≤ 2 ulp), `log` on 7 % (≤ 3 ulp), `sin`/`cos` on 16 %, `tanh`
  on 20 % (≤ 2 ulp);
- float64 on 0.03–19 % of inputs (≤ 2 ulp);
- `sqrt` identical.

**Photon dij sampling off** (relative L2 / max-abs-diff relative to the max):

| Preset | Unit-weight dose | dij column sums (L2) | Optimized dose | Target D95 native / Pyodide [Gy] |
|---|---|---|---|---|
| default | 1.1e-7 / 6.0e-7 | 9.6e-8 | 7.9e-5 / 2.8e-4 | 37.590 / 37.590 |
| fast | 2.0e-6 / 1.1e-4 | 9.6e-7 | 1.5e-3 / 3.9e-3 | 41.294 / 41.297 |
| demo | 5.3e-7 / 3.4e-5 | 2.4e-7 | 9.1e-3 / 3.0e-2 | 45.291 / 45.309 |

200 L-BFGS-B iterations on the flat TG119 objective amplify the 1e-7 gradient differences; D95, D5
and mean agree to within 0.05 %. ipax gives the same picture (fast, no sampling: dose 5.0e-3 L2,
D95 41.373 / 41.355).

**Default settings (sampling on):**

| Case | Unit-weight dose | Optimized dose | Target mean native / Pyodide [Gy] |
|---|---|---|---|
| photons default | 5.1e-6 / 3.8e-4 | 7.1e-2 / 0.33 | 47.27 / 47.27 |
| photons fast | 3.5e-3 / 1.3e-2 | 0.12 / 0.46 | 48.53 / 48.54 |
| photons demo | 1.1e-3 / 3.4e-3 | 2.1e-2 / 0.16 | 49.21 / 49.21 |
| protons default | 2.4e-7 / 9.0e-6 | 1.7e-3 / 7.3e-3 | 44.76 / 44.76 |
| protons fast | 1.1e-7 / 1.7e-7 | 7.0e-3 / 1.6e-2 | 43.49 / 43.50 |
| protons demo | 1.9e-7 / 4.8e-6 | 2.7e-3 / 4.7e-3 | 44.75 / 44.75 |

- Photon `_sample_dij` (stochastic tail sparsification) compares a fixed random draw with dose
  ratios. The ulp differences flip some accept/reject decisions (nnz 1 512 384 vs 1 512 254 for
  fast).
- The effect equals the method's own variability: native seed 0 vs seed 1 differ by 0.15 L2 / 0.48
  max, against 0.12 / 0.46 for native vs Pyodide.
- Side observation, not Pyodide-specific: the optimized TG119 photon dose varies 15 % L2 between
  sampling seeds while the target statistics barely move.

## Timings

Steps were timed in `pipeline.py` and include first-use imports. Single runs, roughly ±10 % noise.
Native has numba off (pyRadPlan's native default is on).

| Case | Bixels | Dose grid | Native [s] | Pyodide [s] | Ratio | Pyodide stf / dij / opt / resample [s] | Native peak RSS [MB] | wasm heap [MB] |
|---|---|---|---|---|---|---|---|---|
| photons default | 340 | 101×101×65 | 2.8 | 5.4 | 1.9 | 0.6 / 3.1 / 1.2 / 0.1 | 535 | 565 |
| photons fast | 594 | 63×63×41 | 4.0 | 5.9 | 1.5 | 0.7 / 3.3 / 1.4 / 0.3 | 723 | 782 |
| photons demo | 1567 | 101×101×65 | 13.1 | 24.4 | 1.9 | 0.7 / 16.9 / 6.1 / 0.5 | 874 | 904 |
| protons default | 3592 | 101×101×65 | 3.6 | 8.1 | 2.3 | 0.7 / 1.6 / 4.9 / 0.8 | 802 | 901 |
| protons fast | 4672 | 63×63×41 | 3.0 | 5.2 | 1.8 | 0.8 / 1.8 / 1.7 / 0.8 | 1420 | 1448 |
| protons demo | 17777 | 101×101×65 | 13.8 | 21.3 | 1.6 | 2.0 / 9.6 / 7.9 / 1.4 | 1699 | 1793 |

- Loading TG119 takes 0.2–0.4 s in both. The wasm heap never shrinks, so its final size is the
  peak; Node RSS is 0.35–0.45 GB higher.
- **Boot and install (warm cache):** `loadPyodide` 1.3 s; `installPyRadPlan` 3.5–4.4 s
  (distribution packages 2.1–2.3 s, pure wheels 0.5 s, pyRadPlan and shim 0.3 s, `import pyRadPlan`
  1.1 s).
- **Cold cache:** install 4.2 s; the jsDelivr download added almost nothing on this connection.
  Installing over HTTP (`--via-http`) took 5.3 s.
- **Hotspots under Pyodide** (cProfile, photons demo): `ndarray.take` 6.0 s (kernel interpolation),
  sparse matvecs 6.2 s, `compile` 3.2 s (no `.pyc`; inflated by the profiler), `searchsorted`
  2.9 s, FFT 1.8 s.

## Download size (per cold start)

| Part | MB |
|---|---|
| Pyodide runtime (`pyodide.asm.wasm`, `python_stdlib.zip`, `pyodide.asm.mjs`, lock) | 13.5 |
| Distribution: scipy 14.0, numpy 3.0, h5py 1.6, pydantic_core 1.3, pydantic 0.5, regex 0.2, jinja2 0.13, contourpy 0.12, micropip 0.12, tqdm 0.08, rest 0.12 | 21.1 |
| Pure wheels: pydicom 2.4, types-array-api 0.37, ipax 0.36, pint 0.31, array-api-* 0.28, numpydantic 0.10, rest 0.15 | 4.0 |
| Shim wheel | 0.02 |
| pyRadPlan wheel (TG119 23.0, machines 8.3, code < 1) | 32.4 |
| **Total** | **≈ 71** |
| Optional plotting: matplotlib 7.0, fonttools 1.2, pillow 1.0, pytz 0.5, dateutil 0.2, other 0.4 | +10.3 |

## Alternatives for SimpleITK

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| (a) numpy/scipy shim outside pyRadPlan (chosen) | Fast to build; pyRadPlan unchanged; bit-identical natively and test-verified; 16 kB; browser-only. | Second implementation to keep in sync: new SimpleITK calls fail in the browser until added (CI should run `tests/` plus a Pyodide smoke run). No file IO. Full-volume resampling about 5× slower than ITK natively. | Use for the prototype and probably beyond. |
| (b) Internal image abstraction (numpy array + `Grid`, SimpleITK only at IO) | One code path; Array-API and GPU friendly; drops a heavy native dependency. | About 57 files, pydantic `sitk.Image` fields, GUI, matRad/DICOM IO, serialization; breaks the API (`ct.cube_hu`). | Long-term project; the shim's tested semantics can serve as its reference. |
| (c) Build SimpleITK for Emscripten | Real ITK semantics including IO. | ITK + SimpleITK + SWIG on Emscripten has no public Pyodide package; tens of MB; rebuild per Pyodide ABI; not attempted. | Not worth it. |
| (d) itk-wasm | Maintained by ITK; wasm IO and filters. | Different API (JS/WASI pipelines, not `SimpleITK.Image`), so it needs an adapter the size of (a); not evaluated in depth. | Possibly later, for browser file IO only. |

## Unsupported or untested

- **SimpleITK image IO:** DICOM CT (`ImageSeriesReader`), RTDOSE, NRRD/MHA (`io.sitk_based`).
  pydicom imports, but the CT path needs `ImageSeriesReader`. The `.npz` handler was not
  tested; pickle works across runtimes (#10).
- **Monte Carlo engines** (TOPAS, FRED, pyRadMC): they need external executables and processes.
  Not applicable.
- **Also unavailable:** IPOPT, numba kernels, cupy/torch/jax, the Qt GUI, the AI features.
- **Shim gaps:** non-identity transforms; Gaussian and windowed-sinc interpolators; ball kernels;
  perimeter, Feret diameter and oriented bounding box; negative slice steps; slice assignment.
- **Untested under Pyodide:** carbon, helium and VHEE; robust/scenario optimization; 4D CT;
  sequencing; `calc_dose_forward`; FSPB.
- **wasm32 limits:** 4 GB address space; `np.intp` is int32 (no overflow seen); single thread; no
  SIMD transcendentals.

## Recommendations

1. **Keep the shim.** Add CI jobs for `npm run test:shim` and a Pyodide smoke run
   (`--preset fast --max-iter 20`, about 15 s plus about 75 MB of cached downloads). Consider (b)
   long term.
2. **Split the data out of the wheel.** Ship a code-only wheel (< 1 MB) and fetch `TG119.mat` and
   the needed machines (photons 2.3 MB, protons 1.2 MB) on demand. First load drops from about
   71 MB to about 43 MB plus the data used.
3. **Ship `.pyc`** built with CPython 3.14 for pyRadPlan and the pure deps, saving about 1–3 s per
   load (gain unverified). Load pydicom (2.4 MB) lazily.
4. **Browser defaults:** use the fast preset for interactive runs (5–6 s). 5 mm, 5-beam plans take
   21–24 s, so they need the worker and progress events. Set the solver explicitly.
5. **Memory guard:** check bixels × dose voxels before dose calculation (17 777 proton bixels on
   101×101×65 use 1.8 GB). Real patients need coarser grids, float32 dij, or a server.
6. **Cross-platform comparisons** need `enable_dij_sampling=False` for photons, or a statistical
   comparison. Bit-equality is impossible because of the libm differences.

## Unverified or caveated

- The measurements in this file come from the Node harness. The browser run is covered separately
  by the frontend's `npm run e2e:pyodide` (see `web/README.md`); browser memory limits and mobile
  devices are untested.
- Timings come from single runs on one machine, with numba off natively. The CDN timing reflects a
  fast connection.
- Shim coverage is verified for the traced TG119 photon and proton (nominal) paths and the app
  calls above. Other paths may hit missing functions; they fail loudly.
- B-spline matches SimpleITK to 1e-5 of the range, not bit-exactly, and is not on the traced path.
- Pins follow the native environment (`numpydantic==1.10.0`, `pint==0.25.3`). `ipax==0.13.0`
  matches pyproject's `>=0.13,<0.14`, whereas the native venv has 0.12.0. Pyodide's pydantic 2.12.5
  and scipy 1.18.0 are within pyproject's bounds.
