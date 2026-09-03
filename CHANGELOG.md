# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Tests pinning the RTSTRUCT contour-to-mask fill rule on a grid small enough to read the expected masks off the page: the boundary tie-break, that a contour narrower than one voxel still produces voxels rather than vanishing, concave outlines, clipping at the grid edge, degenerate (zero-area) contours, and that several contours on one slice are combined. The conversion was additionally cross-checked against label maps exported from MITK Workbench for a 512x512x297 CT (3 differing voxels out of 17.6 million across four structures, two of them voxel-identical); that data is too large to vendor, so the miniature cases stand in for it
- `xp_utils.warn_on_unreliable_openblas()`, called on `pyRadPlan` import: emits a `RuntimeWarning` when NumPy is linked against an OpenBLAS older than 0.3.28, whose multithreaded GEMM can silently return corrupted, nondeterministic results (observed with the scipy-openblas 0.3.27 bundled in the NumPy 2.0 wheels)
- `xp_utils.stream_wait_event()`: enqueue a device-side ordering constraint on a stream via a recorded event instead of blocking the host (no-op on backends without stream support), and `xp_utils.is_host_device()`: check whether a device specification refers to a host (CPU) device via its DLPack device tuple
- `xp_utils.scatter()`: backend-agnostic indexed assignment (`arr[key] = values`), out-of-place on JAX (`.at[key].set`), in place everywhere else
- `xp_utils.jittable`: decorator wrapping a generic array-API kernel with jit-compiled execution paths — per-backend registered implementations (e.g. numba for NumPy, with a `NotImplemented` fallback protocol) and jit compilation of the generic code (`jax.jit`, `torch.compile`) for backends the kernel declares jit-capable. Which backends run jit-compiled paths is steered globally by the new `settings.xp.jit_backends`  (`PYRADPLAN_XP_JIT_BACKENDS`, default `"numpy,jax"`); failed compilations fall back to the generic code with a one-time warning
- Documentation: "Jit-compiled kernels" section under Dose calculation covering `xp_utils.jittable` and `settings.xp.jit_backends`, `[numba]` extra description updated to point to it, and a troubleshooting entry for the `torch.compile`/Triton warning; new `xp_utils` API entries (`scatter`, `jittable`, `JittableKernel`, `is_host_device`, `stream_wait_event`, `choose_device`, `is_on_gpu`, `openblas_has_gemm_race`, `warn_on_unreliable_openblas`) added to the API reference
- The ray tracer hot paths (plane alphas, alpha merge, index computation, radiological-depth segment selection) are extracted into jittable kernels in `raytracer._kernels`, with fused `numba` `prange` implementations for the NumPy backend in `raytracer._kernels_numba` (registered only when numba is installed, exact for index computation, boundary-tie-level differences for the float32 selection)
- `pyRadPlan.ai.modelhub` module: HuggingFace-based AI model handling. `load_model()` loads a model and its preprocessor from a local directory, an explicit `repo_id`, or a name — bare (resolved to `<hf_org>/<name>`) or a full `<org>/<repo>` id used as-is — with version dedup, offline support and logging. A pinned `revision` is reused from disk without touching the network; an unpinned request asks the Hub whether the local copy is current and falls back to it when the Hub is unreachable. Downloads are laid out as `<local_models_dir>/<org>/<repo>`, so a private fork never shares a directory with its upstream namesake. `list_local_models()` lists the models available on disk and in the HuggingFace cache (no network) as full `<org>/<repo>` ids, optionally filtered by `ModelTask` (`dose_calc`/`outcome`), read from `metadata.task` in a model's `model_config.json` and falling back to the repository-name prefix. Includes a `BasePreprocessor` contract, a `model_config.template.json` and a reference model folder under `test/data/ai_models/dummy_model`. Settings are the `modelhub_*` fields of the `ai` sub-configuration of `pyRadPlan.settings` (`settings.ai`), read from `PYRADPLAN_AI_MODELHUB_*` environment variables. `huggingface_hub`/`safetensors` are installed with pyRadPlan; only a `torch` build matching your platform must be installed separately.
- Security: loading a model resolved from the HuggingFace Hub executes the `model.py` and `preprocessor.py` it ships, so it requires an explicit opt-in — `trust_remote_code=True` or `PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE=1`; the setting defaults to `False`. A directory passed explicitly as `load_model(local_dir=...)` is exempt. Each model folder's code is imported under a package name unique to that folder, so several models can be loaded side by side and the classes they define stay picklable.
- `pyRadPlan.core.get_data_dir()` / `get_data_subdir()`: a single writable data root for pyRadPlan (default `~/.pyradplan`, relocatable via `PYRADPLAN_DATA_DIR`). Downloaded AI models default to `<data_dir>/ai_models`; the location is reserved for future downloaded datasets (patients, phantoms, machines).
- Global pydantic-settings configuration `pyRadPlan.settings` (`PyRadPlanSettings`), read from `PYRADPLAN_*` environment variables / a `.env` file, with sub-configurations under extended prefixes (currently `PYRADPLAN_XP_*`, `PYRADPLAN_AI_*`)
- GUI: the Settings menu offers quick links per sub-configuration ("XP (Backend)", "AI") opening a single-section editor, plus "Preferences" opening a tabbed editor for the full `PyRadPlanSettings` hierarchy (a General tab for top-level fields when present, one tab per sub-configuration); accepted edits update the runtime settings and the process environment
- `[profiling]` extra (`line_profiler`) for the line-profiling harnesses in `benchmark/`, plus a "Benchmarks and profiling" section in the installation guide describing how to run both
- `interpnd` accepts `bounds_error=True` to raise on query points outside the grid instead of silently clipping them; used by the photon SVD pencil-beam engine, where an out-of-grid query indicates a mis-sized convolution grid rather than intended extrapolation
- CI: `tests-backends` job exercising the optional array-API backends on their CPU builds
- Tests covering DLPack device parsing and detection (`test/core/xp_utils/helpers/test_device_parsing.py`)
- Regression test ensuring `trace_cubes` traverses the full cube at oblique beam angles on an anisotropic geometry (a cubic phantom cannot catch index-layout mix-ups)
- `PYRADPLAN_GUI_DISABLED` environment variable: reports the GUI as unavailable (`pyRadPlan.gui.GUI_AVAILABLE` is `False`) so scripts fall back to static plots, used when executing the examples for the documentation
- Documentation: the example scripts in `examples/` are rendered as notebooks in a new "Tutorials" section (myst-nb + jupytext). Notebooks are not executed during the docs build; outputs come from executed notebooks committed in `docs/tutorials/examples/`, refreshed locally with `python docs/execute_examples.py`
- Preferred array backends are now the `xp` sub-configuration of the settings (`settings.xp.prefer_gpu`, `settings.xp.preferred_cpu_array_backend`, `settings.xp.preferred_gpu_array_backend`; `None` auto-selects the best available GPU backend), configurable via `PYRADPLAN_XP_PREFER_GPU`, `PYRADPLAN_XP_PREFERRED_CPU_ARRAY_BACKEND` and `PYRADPLAN_XP_PREFERRED_GPU_ARRAY_BACKEND`
- GUI: File menu collecting all data I/O (load/import/export)
- GUI: resizable, collapsible main-window panels via splitters
- GUI: workflow staleness indicators that flag outdated dose influence / results
- GUI: objective count in the objectives widget header
- GUI: AI buttons to suggest VOI objectives and beam angles via a reusable AI task dialog
- GUI: when a result dose exists, the objectives AI button offers to adapt the current objectives using quality indicators computed from a selectable result dose
- `VOI.center_of_mass`, `VOI.principal_axes` and `VOI.shape_parameters` computed fields describing the structure's geometry (nominal scenario, world LPS coordinates)
- `ai.agents.generate_beam_angles` accepts an optional structure set; its per-VOI geometry (via new `cst_geometry_summary()`) is sent to the model, so beam directions can respect the patient anatomy (used by the GUI when a structure set is loaded)
- `ai.agents.generate_voi_objectives` accepts an optional `QICollection` (`qis=`) to adapt the existing objectives based on quality indicators from a previous optimization run instead of suggesting fresh ones
- GUI: persistent log console panel (bottom-left) fed by Python's logging system, with a level filter and colored warnings/errors — messages are visible even without a terminal
- GUI: "Log Panel" toggle in the View menu
- GUI: VOI list tooltips showing the structure's metadata (type, α_x, β_x, α/β, overlap priority)
- GUI: clicking a VOI name opens a popup to edit type, α_x, β_x and overlap priority (validated against the VOI model); toggling visibility remains on the checkbox
- GUI: VOI list can be grouped by overlap priority (one group per priority level) instead of by type via a new dropdown
- GUI: per-objective quantity selection in the objectives table (dropdown over the registered quantities)
- `ai.agents.available_models()` to discover usable models from configured API keys
- global variable GUI_AVAILABLE, checking for pyside6 and pyqtgraph
- pre-commit hook checking dependency license compliance via `licensecheck` (configured in `[tool.licensecheck]` in pyproject.toml); compatibility is derived from the project's own license instead of a hand-maintained allowlist
- VOI `objectives` are now validated into `Objective` instances (from names or dicts) instead of being stored as raw values
- IO: New extensible import/export framework in `pyRadPlan.io` with a layered design: `base/` (`BaseImporter`, `BaseExporter`) and per-format backends `matlab/`, `dicom/`, `npz/`, `pickle/` and `sitk_based/` (NIfTI/NRRD/MetaImage)
- IO: Top-level I/O API: `load_data(path)` (loads everything found into a dict: ct, cst, dose, ...) and `save_data(ct=..., cst=..., dose=..., file_name=..., format=...)` with a smart default format (`.mat`) and per-object file naming when no `file_name` is given
- IO: Per-format low-level handlers (`MatlabHandler`, `DicomHandler`, `NpzHandler`) exposing `load_ct`/`load_cst`/`load_dose`/`load_patient`/`load_data` and `save`; the individual `*Importer`/`*Exporter` classes live in the backend submodules (e.g. `pyRadPlan.io.dicom`)
- IO: DICOM import for CT series, RTSTRUCT, SEG and RTDOSE (orientation-aware, SimpleITK-backed), and DICOM export for CT series, RTSTRUCT, SEG and RTDOSE. Structures export as RTSTRUCT by default; pass `DicomExporter(path, structure_format="seg")` to export as SEG instead
- IO: NumPy `.npz` backend (`NpzImporter`/`NpzExporter`/`NpzHandler`) for fast single-file import/export of ct, cst and dose (VOIs stored as linear indices; geometry/metadata as JSON)
- IO: SimpleITK-based backends under `pyRadPlan.io.sitk_based` — NIfTI (`NiftiHandler`, `.nii`/`.nii.gz`), NRRD (`NrrdHandler`, `.nrrd`) and MetaImage (`MetaImageHandler`, `.mha`/`.mhd`) — sharing common base classes. A patient maps to a folder (`ct`, `dose`, and a label-map `cst` + JSON sidecar; NRRD/MetaImage also embed 3D-Slicer/`pyradplan_*` metadata). A single image file is read as a CT.
- IO: Pickle backend (`PickleHandler`, `.pkl`/`.pickle`) for fast, full-fidelity single-file import/export of ct, cst, dose and arbitrary extras. (Unpickling executes code; load only trusted files.)
- IO: `load_binary_patient(ct_file, structure_paths, selections=...)` imports a *foreign* folder: a CT from an arbitrarily named image file (values taken as HU) plus one binary mask file per structure (mixed formats allowed); masks on a different grid are nearest-neighbor resampled onto the CT. VOI names come from file stems, types from a name heuristic; per-file `selections` can rename/re-type/ignore masks. Helpers `list_image_files`, `masks_to_cst`, `mask_file_to_voi`, `read_ct_image` in `pyRadPlan.io.sitk_based`
- IO: `DicomImporter` can enumerate its source (`list_ct_series()`, `list_structure_sets()`, `list_doses()`) and load selectively (`load_ct(series_uid=...)`, `load_cst(struct_file=...)`, `load_dose(dose_file=...)`)
- GUI: "Load Folder" now routes to an import dialog: DICOM folders open a series/structure/dose selection dialog; folders of image files open a binary import dialog with a CT file field and an editable structure review table (name + TARGET/OAR/EXTERNAL/IGNORED type per mask)
- GUI: loading a single bare image file (NIfTI/NRRD/MetaImage) asks what it represents: CT as a new patient (clears the workspace), CT replacing only the current one (mismatched grids clear the dependent structures/dose influence/results after a warning), structure(s) added to the structure set (binary mask or multi-label label map incl. sidecar/embedded metadata; name clashes get a numeric suffix), or a dose added to the result collection under a chosen name (resampled onto the CT grid if needed). The dialog preselects the likely option from the pixel data (`infer_image_kind`: unsigned integers -> structures, negative values -> CT in HU, non-negative floats -> dose)
- IO: `image_file_to_vois(ct, path)` reads a structure image file into VOIs — a single binary mask becomes one VOI, a multi-label image becomes one VOI per label (using a JSON sidecar or embedded `pyradplan_*` metadata for names/types when present)
- `ParticleFredMCEngine.execution_timeout` (seconds, default `None` = wait indefinitely): aborts a FRED run that exceeds the timeout, killing the whole FRED process tree (e.g. when the GPU is occupied by another process)
- Importers report progress: `BaseImporter` mixes in `ProgressReporter`, so an import can be followed through `observe_reports` (or a console `tqdm` bar) like a dose calculation. The DICOM importer reports one nested, determinate level per step — scanning the folder's headers, reading the CT slices (driven by ITK's own per-slice progress), converting each RTSTRUCT ROI / SEG segment, and reading the dose cube — and logs what it found and what it is loading
- GUI: importing a DICOM folder shows these steps in the progress bar and status line, and logs them to the output panel, instead of an untitled busy bar. Enumerating the folder now also runs in the worker thread, so picking a large folder no longer freezes the window before the import dialog appears
- `pyRadPlan.util.openmp`: detection of clashing OpenMP runtimes in the running process. Several wheels vendor their own copy of the Intel/LLVM runtime, which calls `abort()` (`OMP: Error #15`) when a second copy initializes — killing the interpreter with no catchable exception, and only at the first parallel region rather than at import. `blocked_by_openmp()` reports such a clash for a given package, built on `loaded_runtimes()`, `duplicate_loaded_runtimes()`, `runtimes_shipped_by()` (which inspects a package on disk without importing it) and `duplicate_runtimes_allowed()`. The latter also honours `KMP_DUPLICATE_LIB_OK` set only in pyRadPlan's `.env` file, copying it into `os.environ` (as pydantic-settings' own `.env` handling never does) since that is where the native OpenMP runtime actually reads it from

### Changed

- IPOPT is no longer registered as a solver when `ipyopt`'s vendored OpenMP runtime clashes with one already loaded in the process (e.g. PyTorch's), since starting a solve would abort the interpreter. `pyRadPlan.optimization.solvers.IPOPT_DISABLED_REASON` explains why, a warning is logged, and `PlanningProblem` falls back to the next available solver as it already did when `ipyopt` was not installed. Set `KMP_DUPLICATE_LIB_OK=TRUE` before starting Python to use IPOPT anyway (unsafe, per Intel's documentation); `OptimizerIpopt` also re-checks before each solve, so a package imported after the solver raises a `RuntimeError` instead of crashing the process
- Siddon ray tracing orders its per-axis plane-alpha streams with device-side event waits (`stream_wait_event`) instead of a host-blocking device synchronization, and the alpha-limit computation runs sequentially on the main stream (its arrays are too small for stream parallelism to pay off, and the streams required two further device synchronizations)
- Siddon ray tracing caches constant geometry arrays (voxel planes, resolution, cube dimensions) and the uploaded cube buffers per array backend/device/precision, instead of re-converting and re-uploading them on every trace call
- `trace_cubes` derives its BEV ray-matrix extent from the 8 cube corners (the maximum of an affine map over a box is attained at a corner) and evaluates the radiological-depth ray selection only on the valid segments through a single composed voxel-index-to-BEV affine map in working precision — previously every voxel coordinate was rotated per call in float64. Boundary-tie voxels of the selection may differ at the sub-micrometer level
- `trace_cubes` runs its radiological-depth segment selection and cube filling on the active compute backend and consumes the traced arrays directly on the device through a new private `_trace_rays_device` hook (overridden by the Siddon tracer to skip the host round trip and the conversion of the unused alphas/d12; the numpy output contract of the public `trace_rays` is unchanged). Only the finished depth cubes are transferred back, removing the host-bound numpy post-processing that dominated GPU runtimes. Cube filling now scatters into a flat device buffer via the new `xp_utils.scatter` helper; the out-of-range index recovery path is gone since the selection mask only ever marks bounds-validated indices
- The candidate ray matrix is computed with a loop-free interval-stabbing formulation (sort + searchsorted over per-row disc intervals) in Array API code, replacing both the per-ray Python loop and the optional numba kernel (`raytracer._numba_perf` is removed); results are identical, large ray sets reach the former numba speed, and GPU backends no longer launch per-ray kernels
- Siddon ray tracing gathers through a C-contiguous representation of the SimpleITK image while preserving its public Fortran-order voxel indices, avoiding the previous Fortran-order buffer rearrangement
- **Breaking:** the AI subpackages are nested under a common `pyRadPlan.ai` parent: `pyRadPlan.ai_agents` is now `pyRadPlan.ai.agents`, and `pyRadPlan.ai_models` is now `pyRadPlan.ai.modelhub`. No compatibility shims are provided; update imports to `from pyRadPlan.ai import agents` / `from pyRadPlan.ai.modelhub import load_model`. Importing `pyRadPlan.ai` pulls in neither optional dependency stack.
- All AI configuration is unified in a single `AiSettings` class, the `ai` sub-configuration of `pyRadPlan.settings` (`settings.ai`), with the two subsystems kept apart by the field-name prefix: the agents' `agents_model` / `agents_display_usage` (`PYRADPLAN_AI_AGENTS_MODEL`, `PYRADPLAN_AI_AGENTS_DISPLAY_USAGE`; the 0.4.1 names `PYRADPLAN_AI_MODEL` / `PYRADPLAN_AI_DISPLAY_USAGE` are still read as legacy aliases, the canonical name wins when both are set) and the model hub's `modelhub_*` fields (`PYRADPLAN_AI_MODELHUB_*`). The earlier `PYRADPLAN_AI_MODELS_*` and `PYRADPLAN_HUGGINGFACE_PATH` names are no longer read; one GUI settings section "AI" covers both subsystems.
- The AI agents now consult the runtime settings singleton (`settings.ai`) instead of re-reading the environment on every call, matching `settings.xp` and the model hub: set `PYRADPLAN_AI_*` variables before importing pyRadPlan (or in a `.env` file), or mutate `pyRadPlan.settings.ai` at runtime.
- `xp_utils.choose_device` now returns backend device objects (`torch.device`, `cupy.cuda.Device`, or `None` for NumPy/array-api-strict, which expose no device concept) instead of strings such as `"cpu"` or `"0"`. The returned objects round-trip back into `to_namespace(device=...)` and `from_numpy(device=...)`
- Optional compute backends are now declared as extras: `[torch]`, `[cupy]` and `[jax]`. For a specific CUDA build, prefer the vendor install commands documented in the installation guide
- Performance work is separated from the test suite: `[tool.pytest.ini_options]` sets `testpaths = ["test"]`, the line-profiling harness moved from `test/` to `benchmark/`, and benchmarks share a common `benchmark_*.py` prefix (profiling scripts use `profile_*.py`)
- GUI: objectives table is now scoped to the VOI selected above it (dropped redundant VOI columns)
- launch_viewer calls to multiple examples with fallback to plot_slice
- grids now have a 4D representation (x,y,z,t)
- VOIs now connect to grids (3D or 4D)
- VOIs validate from more formats formats (given the context)
- GUI: shared worker-thread, adaptive spin box and number-list helpers consolidated in `pyRadPlan.gui.widgets._base`
- GUI: viewer caches derived CT/quantity/mask arrays, optimization plots redraw rate-limited, and spin boxes commit on focus-out instead of per keystroke (performance)
- GUI: colors picked in the VOI list are written back to `voi.visible_color`, so custom colors survive list rebuilds (e.g. grouping changes)
- `ai.agents.generate_voi_objectives` no longer mutates the passed structure set; it returns an updated copy
- `pyRadPlanGUI` console script now uses a dedicated `pyRadPlan.gui.main()` CLI entry point; `gui()` no longer reads `sys.argv`
- `ViewingWidget.set_data/set_vois/set_masks` are restored as deprecated shims (populate the `WorkspaceManager` instead)
- GUI: visualization controls moved from the lower-left corner to the center column below the slice viewer, using the vertical slack under the square CT view; the log panel takes their former place
- docs: `pip install "pyRadPlan[gui]"` is now the recommended install command (README and installation guide); the plain install is documented as the headless variant
- IO: Refactored the `pyRadPlan.io` package around the new framework. `load_patient`, `load_tg119` and `validate_matrad_patient` remain available; the legacy `MatlabFileHandler` and top-level `matfile` module were removed (low-level `.mat` read/write lives in `pyRadPlan.io.matlab`)
- examples: `proton_mc_topas.py` replaced by `mc_topas.py` with added result viewer
- examples: `utils_matrad.py` uses `pyRadPlan.io` (`MatlabHandler`, `save_data`) instead of `pymatreader` / `scipy.io.savemat`

### Deprecated

- `xp_utils.PREFER_GPU`, `xp_utils.PREFERRED_CPU_ARRAY_BACKEND` and `xp_utils.PREFERRED_GPU_ARRAY_BACKEND`: reads and writes still work but emit a `DeprecationWarning`; use `pyRadPlan.settings.xp` instead

### Fixed

- `DVH.compute` rejected a float32 quantity with a pydantic `ValidationError` on `bin_edges`. `np.histogram` takes the bin dtype from the quantity, so the float32 cube the DICOM importer produces yielded float32 edges against a model field declared as float64; the histogram output is now cast explicitly. Widening float32 is exact, so no bin edge moves
- `DVH.get_dy` returned the same value (the last bin edge) for every volume percentage, so D2, D50 and D95 were all identical and above the maximum dose in the structure. It interpolated a fraction (`y / 100`) against a cumulative volume held in percent, and did so with `np.interp`, which requires an increasing x array while the cumulative DVH decreases. Dy is now the largest quantity value still covering at least y percent of the volume, matching the "at least" reading documented on `DVH.cumulative` and agreeing with the independent percentile that `QICollection`'s `DX` computes. It is deliberately not interpolated across the cumulative curve, because a uniformly irradiated region forms a plateau there and interpolation reports the dose at the wrong end of it. `get_dy` now also accepts an array of volume percentages. `QICollection` was never affected, as it computes `DX` from the voxels directly
- DICOM: an imported RTDOSE is now resampled onto the CT grid. RTDOSE is stored on its own grid (for a CIRS phantom export, 124x145x158 at 2 mm against a 512x512x297 CT at ~1 mm), and the cube was handed on unchanged. Since the viewer overlays the dose slice with the same index as the CT slice, the dose was drawn at the wrong scale in a corner of the image, and scrolling past the dose cube's extent raised an `IndexError` that left the viewer stuck. Dose outside the RTDOSE cube is zero rather than extrapolated, so no dose is invented where none was computed; a dose already on the CT grid (as in a matRad export) is passed through untouched. `DicomImporter.load_dose` takes an optional `ct` for the target grid, defaulting to the CT the importer loaded most recently
- DICOM RTSTRUCT import is roughly 20x faster (a 512x512x297 CT with four structures went from ~50 s to ~2 s): each contour was rasterized by testing *every* voxel of the slice against it (262144 point-in-polygon tests per contour on a 512x512 grid, 92% of import time). Contours are now filled by scanline, evaluating one intersection per edge per voxel row instead of testing every voxel against every edge, and only within the contour's bounding box; the world-to-voxel interpolators are also built once per structure set rather than twice per contour. The resulting masks are unchanged, voxel for voxel, on both reference datasets. `matplotlib` is no longer used for structure import
- `DoseWeightedLET` & `LETxDose` quantities: fix inverted physical unit fractions (`keV/µm` instead of `µm/keV`), properly compute `DoseWeightedLET` as quotient of `let_dose` and `physical_dose`, and implement quotient rule chain derivative.
- Radiological depth cubes could differ between identical runs at scattered boundary voxels: NumPy's bundled threaded OpenBLAS (scipy-openblas 0.3.27, observed on Zen) returns wrong, run-to-run varying elements for the tall-skinny `(N, 3) @ (3, 3)` matmuls used to rotate coordinates; when such an OpenBLAS (< 0.3.28) is detected, `trace_cubes` falls back to applying these affine transforms elementwise (BLAS keeps handling them otherwise)
- `xp_utils.elapsed_time` returned negated durations for torch CUDA events (event arguments were passed in reversed order)
- `Beam` derives unset `source_point_bev` / `source_point` from `sad` and the beam angles (matching the stf generators) instead of static defaults that were mutually inconsistent with `sad`; explicitly provided values are kept unchanged
- `RayTracerSiddon.trace_ray` rejected every input under the torch backend: `Tensor.size` is a method, so the shape guard compared a bound method and was always true (now uses `array_api_compat.size`)
- Siddon ray tracing preserves stored plane coordinates during alpha generation so boundary-plane intersections deduplicate exactly and copies read-only SimpleITK views before backend conversion
- Siddon ray tracing now runs with JAX namespaces that do not expose the previously hard-coded Array API specification version, and its voxel-index calculation no longer uses in-place updates
- `xp_utils.to_namespace` without an explicit `device` moved arrays to the GPU whenever the target backend had one, ignoring `settings.xp.prefer_gpu`; it now keeps the source array's device when the target namespace supports it and otherwise uses the namespace default, which honors the setting
- `xp_utils.choose_device` raised `RuntimeError` on CPU-only JAX installs when a GPU was preferred (`jax.devices("gpu")` raises instead of returning an empty list)
- Ray tracer: voxel indices past the end of the coordinate array are now treated as invalid instead of raising during radiological depth lookup
- `xp_utils.choose_device` raised `RuntimeError` instead of falling back to the CPU when a GPU was preferred (the default) but unavailable, breaking dose engines and the Siddon ray tracer on CPU-only PyTorch, CuPy and JAX installs
- `interp1d` rejected 1-D `y` with an `IndexError` on backends without a native `xp.interp` (array-api-strict, PyTorch), and mis-broadcast the out-of-bounds `left`/`right` values for 2-D `y`. The `left=None`/`right=None` defaults are again rank-agnostic and match `numpy.interp`; N-D query points are supported
- `_fft2`/`_ifft2` handed non-NumPy arrays to `scipy.fft`; the generic path now uses the array API `fft` extension
- Photon SVD pencil-beam engine: a NumPy scalar leaked into backend array expressions, and `np.exp`/`np.real` on backend arrays silently returned to NumPy; the primary-fluence grid also lost its explicit `float32` dtype
- Sparse conversion helpers: a CuPy index array was passed to `torch.sparse_coo_tensor` instead of the converted tensor, `_is_torch_sparse_tensor` raised `AttributeError` when PyTorch was absent, and a CuPy-to-CuPy conversion ignored the requested device index
- `xp_utils.synchronize` swallowed stream synchronization errors via a `return` inside `finally`
- `xp_utils.choose_device` raised `ValueError: CuPy does not support CPU` for the CuPy namespace when `prefer_gpu` was disabled; CuPy is GPU-only, so the namespace already implies a CUDA device
- Pencil-beam `calc_geo_dists` mixed the globally configured device into array constructors whose namespace came from the input data, so a device from one backend could reach another backend's `asarray`. The device is now taken from the input arrays
- Photon SVD pencil-beam engine: the per-ray custom fluence path (used by field-based dose calculation) was hard-wired to NumPy and `scipy.fft`, so it failed on every other backend; it also multiplied beamlet masks in place, mutating the stored masks
- examples: `pencilbeam_photon.py` forced the JAX backend on import, so it failed on any install without the optional `[jax]` extra
- Device detection no longer misclassifies devices: any device object whose `repr` merely contained "cpu" was treated as CPU (array-api-strict devices are now matched by type), an array whose device could not be determined was silently reported as CPU (it now warns), and JAX's global device id was used to index the per-platform device list, selecting the wrong GPU or raising `IndexError` on hosts with more than one platform
- `ai_agents`: importing the module no longer floods stderr with `BeartypeClawDecorWarning`s (pydantic-ai pulls in `key_value.aio`, whose beartype import hook trips over numpydantic's vendored nptyping); the package's `PY_KEY_VALUE_DISABLE_BEARTYPE` opt-out is now set before the import
- IO: `load_data` on a DICOM folder picked an arbitrary RTDOSE file (often a per-beam or LET cube); it now selects the plan-level physical dose via `DoseSummationType`/descriptor filtering
- IO: exporting a ct *and* a dose to a single-file SimpleITK target silently dropped the dose; it now raises (a single image file holds one image — use a directory for both)
- IO: `DicomHandler` ignored its `structure_format` argument (SEG export via the handler)
- IO: exported RTSTRUCT files now reference the CT series/slices (`RTReferencedStudy` → `RTReferencedSeries` → `ContourImageSequence`), so third-party viewers associate the structures with the CT instead of relying on the frame-of-reference UID alone
- IO: exported DICOM SEG files are now conformant `BINARY` segmentations (1-bit packed frames instead of 8-bit pixels)
- GUI: exporting a result quantity stored as a raw matRad array wrote it mis-oriented ((y,x,z) was not transposed to (z,y,x)); saving a single quantity now also honors the chosen image format instead of falling back to `.mat` for extension-less file names
- GUI: DKFZ logo pinned to the top-left of the banner in wide windows
- global variable GUI_AVAILABLE, checking for pyside6 and pyqtgraph
- GUI: "Save / Keep Result" silently skipped every snake_case quantity (e.g. `physical_dose`), so snapshots lost the dose
- GUI: re-optimizing no longer overwrites the fresh per-beam dose with the previous run's (`physical_dose_beam` matched the snapshot carry-forward prefix)
- GUI: closing the main window during a computation aborted the whole process (running QThread destroyed); now asks and stops the worker
- GUI: a failed AI objectives request wiped all existing objectives on the live structure set
- GUI: closing the AI task dialog while a request was running froze the GUI until the request finished
- GUI: the workflow "Import Dose" button was only enabled once a result existed; it now only requires a CT (like the File menu)
- GUI: plan widget crashed its workspace sync (silently) when `pln.prop_dose_calc["dose_grid"]` held a `Grid` instance or `None`
- GUI: the viewer kept the previous patient's VOIs/masks/images when the workspace was cleared or lost its structure set
- GUI: widget refresh failures are now shown in the main-window status bar instead of only being logged
- GUI: config forms show a generic JSON editor for unsupported field types instead of hiding them
- GUI: the plan widget's iso-center field stayed at "0 0 0" in "Auto." mode; it now previews the automatic isocenter (target center of mass), matching the viewer
- GUI: the optimization status window could open so far down that the Pause/Stop buttons were off-screen; it now opens centered over the main window, clamped to the screen
- GUI: loading patient data left the workspace without a plan while the plan widget looked applied; the form defaults are now auto-applied as the initial plan on data load (falling back to the "modified" highlight if they don't validate)
- Ipopt intermediate-callback guard admitted arities it then indexed past (`>= 3` vs. index 9)
- VHEE now has target points for matRad export
- Cleaned up in-repo deprecation warnings for forward compatibility (Pydantic V3, NumPy, Python)
- examples: crashing plot block in `pencilbeam_carbon.py` and broken `.mat` unpacking in `utils_matrad.py`, deprecated `plot_slice(ct=...)` calls, missing jupytext cell markers in `pencilbeam_vhee.py`, plus stale docstrings, engine names and file references

## [0.4.1] - 2026-06-16

### Added

- multiple corresponding units for given quantities in the GUI
- AI agents now log token usage and estimated cost (USD) after each run; toggle via `AiSettings.display_usage` (`PYRADPLAN_AI_DISPLAY_USAGE`)
- `get_objectives_union()` exposing all registered objectives as a discriminated union, used to give the AI agent the exact objective schema

### Fixed

- missing keys in init_beam not being skipped
- hotfix for deprecated numpy.matrix in to_namespace()

## [0.4.0] - 2026-05-17

### Added

- `pyRadPlan.ai_agents` module: LLM-powered treatment planning helpers built on `pydantic-ai`
- `generate_beam_angles(pln, treatment_site)` — queries an LLM to suggest gantry and couch angles for a given treatment site and radiation mode, and writes them into `pln.prop_stf`
- `generate_voi_objectives(pln, cst, treatment_site)` — queries an LLM to propose optimization objectives for each VOI in a `StructureSet` and attaches validated objective instances directly to the VOIs
- `AiSettings` — pydantic-settings class for global configuration; reads the default model from the `PYRADPLAN_AI_MODEL` environment variable (default: `claude-sonnet-4-5`); any provider supported by pydantic-ai can be selected via the model string
- New example `examples/utils_ai_agents.py` demonstrating an end-to-end proton prostate plan with AI-generated beam angles and objectives
- AGENTS.md and CLAUDE.md for AI-assisted development
- GPU-accelerated dose calculation via Array API using CuPy and PyTorch backends (alongside NumPy/`array_api_strict`), including memory management, streaming, and per-beam cleanup
- `to_namespace()` helper to convert arrays (and scipy sparse matrices) between Array API namespaces, with explicit `device=` and `keep_sparse_compat` options
- `choose_device()` to select a sensible default device for a given namespace, with multi-GPU index support (`gpu:N` / `cuda:N`)
- DLPack-based device handling (`get_device_info`, `is_on_gpu`, `DLPACK_CPU`/`DLPACK_CUDA` constants) for seamless backend interop
- GPU lifecycle helpers: `free_gpu_memory()`, `create_stream()`, `get_current_stream()`, `synchronize()`, `record_event()`, `elapsed_time()`
- `from_numpy()` / `to_numpy()` helpers with device targeting
- Backend availability checks and a preferred-backend wishlist (`cupy_available`, `pytorch_gpu_available`, `jax_available`, `jax_gpu_available`, `numba_cuda_available`, `PREFERRED_GPU_ARRAY_BACKEND`)
- Native CUDA kernel implementation for geometric distance calculation: `_calc_geo_dists_cupy_kernel` (CuPy `ElementwiseKernel`), `_calc_geo_dists_cupy_raw_kernel` (CuPy `RawKernel`), and `_calc_geo_dists_torch_kernel`, replacing the previous Numba CUDA path
- Array API conform N-D interpolation `interpnd()` on rectilinear grids (generic 2D/3D fallback, with dedicated `RegularGridInterpolator` paths for NumPy/SciPy, JAX, and CuPy)
- Improved `interp1d`: fast paths using `xp.interp` for NumPy/JAX/CuPy, JAX `jit` and PyTorch `torch.compile` backends (with `torch.jit.script` fallback if Triton is not installed), support for lists/tuples/dicts of arrays with optional stacking
- Array API compatibility for beam initialization and ray geometry computation (`get_gantry_rotation_matrix`, `get_couch_rotation_matrix`)
- More efficient sparse matrix conversion using direct CSR/CSC construction, avoiding unnecessary deep copies
- Kernel-data caching and `ParticlePencilBeamKernel.to_namespace()` to move kernel arrays onto the active backend/device
- Device propagation through fluence optimization (`NonLinearFluencePlanningProblem`, `OptimizerIpopt`, scipy solver, `SolverBase.device`)
- New example `examples/utils_backends.py` demonstrating how to query backends and run dose calculation on different array backends
- Benchmarks `benchmark/benchmark_interp1d.py` and `benchmark/benchmark_interpnd.py` for interpolation across backends
- quantity resolver that checks for presence of quantities and instantiates the required ones
- biological RBE calculation from alpha and beta kernels
- biological based optimization
- alpha and beta parameters to dij with function `get_reference_lq_params` to get them for a given ct and cst
- quantity resolver that checks for presence of quantities and instantiates the required ones
- a TOPAS monte carlo interface. Implemented are protons and ions with basic physical and let based scoring. Material conversions are water and a pre defined schneider converter. The interface is structure into input files so that different beam models or scorers can be added easily. The interface also as template files that are used to create the simulation files. Which also give a nice overwiev on the structure of the simulation files.
- jinja2 as project dependency
- Documentation: extended user guide
- Documentation: extended installation instructions
- First implementation of an interactive result viewer widget (`gui` extra) for visualizing dose/quantity distributions slice-by-slice with scroll/zoom, VOI contour overlay, isoline rendering, and colormap selection
- DVH viewer and DVH comparison tools in the GUI analysis widget
- Quality indicator (QI) panel in the GUI analysis widget
- `visible` and `visible_color` fields on `VOI` for storing per-structure display properties
- `DEFAULT_VOI_COLORS` palette (per VOI type: TARGET, OAR, EXTERNAL, HELPER) exported from `pyRadPlan.cst`
- `StructureSet.set_colors()` method that auto-assigns colors from the predefined palette, skipping colors already in use, preserving any explicitly set `visible_color`
- `visible_color` field validator on `VOI` accepting named color strings, float 0–1 arrays, and int 0–255 tuples

### Changed

- Pencil beam dose calculation now applies the lateral cutoff mask before computation rather than after, and keeps Dij assembly on the CPU (size limited) while the rest of the calculation runs on GPU
- Per-beam dose/LET/effect computation in `Dij.compute_beam_dose()` now slices intensities by beam (faster matmul) instead of multiplying by a beam mask, and is fully Array API namespace aware
- Siddon raytracer now picks up the engine's device and allocates plane/coordinate arrays directly on that device
- SVD photon pencil beam engine refactored to use Array API arrays for ray-position aggregation and kernel weighting (still calls SciPy interpolators on host arrays)
- resampling in BLD now uses interpolation to return mask on the grid provided by the dose engine
- in SVDPB field_grid is now built before the resampling of the beamlet mask to guarantee matching grids
- `_draw_contours()` in `plot_slice` now uses `voi.visible_color` instead of a colormap cycle, so contour colors are consistent with the GUI viewer
- GUI optional dependency changed from `PyQt5` to `pyside6>=6.0.1` and `pyqtgraph>=0.12.0`
- Proton pencil-beam example updated to use the new dose viewer widget

### Fixed

- `resample_image` now falls back to linear interpolation when BSpline is requested but the image has fewer than 4 voxels in any dimension, preventing intermittent NaN values from the BSpline prefilter on small grids
- CuPy issue in LPS coordinate handling (gantry/couch rotation matrices now built via `xp.stack` with the correct device/dtype)
- SVD pencil beam engine updated to match changes in the base pencil beam engine
- Device handling and type checks in optimization solvers (IPOPT and SciPy) so the optimization runs on the same device as the quantities
- Preliminary workaround for CUDA / cuBLAS DLL conflicts when PyTorch and CuPy are imported in the same environment
- `free_gpu_memory()` now skips NumPy/`array_api_strict` namespaces silently
- `Beam.validate_nparray_dtype` handles non-list array inputs via `to_numpy()` so non-NumPy arrays validate correctly on import
- `to_namespace()` raises `TypeError` for scalar / list / tuple inputs instead of failing on the sparse-array check
- `StructureSet` now calls `set_colors()` during validation so every VOI always has a color assigned
- property .size raising an error in dij.py when torch is used. Switching to array_api_compat.size()

## [0.3.5] - 2026-05-12

### Changed

- transmission mask in beam limiting devices now uses edge smoothing to account for finite grid size

### Fixed

- Bugfix in FRED MC engine, where temporary files were not deleted

## [0.3.4] - 2026-05-05

### Added

- dose engines now have private flags `_dij_guarantee_canonical` and `_dij_guarantee_nonzero` to guarantee sparse dij structure before finalization
- Issue and Pull/Merge Request templates
- default pre-commit hooks for line endings, case conflicts, merge conflicts, etc. Also added codespell, CITATION.cff verifyer and toml check
- `plot_slice()` and `plot_multiple_slices()` now accept a generic `image_volume` (CT, dict, or `sitk.Image`) instead of only a `CT` via the `ct` argument
- `plot_slice()` gained `image_window` and `window_mode` (`"minmax"` or `"centerwidth"`) options for grayscale windowing, replacing the previous `ct_window`
- `overlay_unit` in `plot_slice()` now also accepts a `pint.Quantity`
- Jennifer Josephine Hardt added as author (CITATION.cff and pyproject.toml)
- Added an `extrapolate` option to `resample_image` to choose the extrapolator. Default is to use a nearest neighbor extrapolator, but individual values can beprovided as well.

### Changed

- New version of photons_Generic.mat basedata file can now be provided, allowing a "version" field alongside "meta" and "data" files within the machine struct. Version 2 requires correct kernel normalization (without implying a spacing in the convolution integral). photons_Generic.mat has been updated to version 2 with correct kernel normalization.
- Photon dose calculation now does not rely on hardcoded convolution resolution integral normalization of machine kernels. Assumes that old kernels use hardcoded factor of 4 for 0.5 mm resolution (1/0.5^2).
- Sparse structure now uses one shared index structure across dijs if possible
- `plot_slice()` internally refactored into modular helpers (input validation, axis resolution, image/contour/overlay drawing, scale bar, finalization) for readability and reuse
- Slice titles now follow the format `Slice ID = N at X mm` (or `Slice ID = N` when no image volume is provided)
- `plot_slice()`/`plot_multiple_slices()` parameter `ct` renamed to `image_volume` and `ct_window` renamed to `image_window`

### Deprecated

- `ct` and `ct_window` arguments to `plot_slice()`/`plot_multiple_slices()` still work as aliases for `image_volume` and `image_window` but emit a `DeprecationWarning`

### Fixed

- CI Release Workflow now tests correctly on release commits without an [Unreleased] section in the Changelog.
- CI Release Workflow now fetches tags correctly and supplies the correct release body from the tag message

## [0.3.3] - 2026-04-14

### Added

- Automatic Release workflow on GitHub reading CHANGELOG.md and tag message
- Benchmark Folder with initial Raytracer benchmark that can be run with pytest-benchmark

### Changed

- Changelog now follows thee Keep A Changelog conventions

### Fixed

- Fixed raytracer issue where certain geometrical configurations could lead to individual rays starting with invalid indices, inserting misplaced NaN's into the radiological depth cube.

## [0.3.2] - 2026-04-07

### Added

- Mimicking objective for dose-mimicking optimization (`SquaredMimicking`)
- Prototype for field-based dose calculation using Beam Limiting Device & FieldShape hierarchy
- CITATION.cff with authors and ORCIDs

### Fixed

- Raytracer candidate matrix now uses lateral cutoff by default (overridable)
- Fixed binary mask interpolation in VOI
- Fixed import of empty voxel index lists from matRad
- Fixed crash when optimization problem has no objectives
- Fixed export of `None` values and type check in `dij`/grids
- Fixed missing x-divergence of ray in FRED MC engine
- Fixed single-field STF generator to properly inherit from IMRT generator

## [0.3.1] - 2026-01-28

### Added

- Performant and Array API compatible candidate ray matrix setup alternative for cube raytracing
- Convenience plotting function to display multiple slices (`plot_distributions`)

### Changed

- `numba` is no longer a mandatory dependency

### Fixed

- Fix readthedocs YAML to correct Python version
- Small code quality fixes

## [0.3.0] - 2026-01-12

### Added

- Partial Array API compatibility with GPU support for CuPy and Torch (drops Python 3.9)
- FRED interface (Monte Carlo tool)
- VHEE planning with a generic (unfocused) beam and a focused beam
- `dij.compute_result_ct_grid()` now returns quantities per beam
- `create_body_segmentation()` method for the CST object
- Option to cancel solver at any iteration via keyboard input
- Comprehensive Sphinx documentation
- Various examples conforming to jupytext norm

### Changed

- Increased memory efficiency in dose calculation
- Tuned initial Scipy solver parameters
- Refactored cst, ct, machine, and stf test data
- Added Python version matrix to CI tests

### Fixed

- Fixed overlap priorities when similar levels exist
- Fixed `np.floating` deprecation
- Fixed pydantic >= 2.11 compatibility
- Fixed issues with single bixel calculations in raytracer
- Elevated minimum required version of numpydantic

## [0.2.8] - 2025-06-28

### Added

- `DVHCollection` and `DVH` for plan analysis
- Maps to associate bixel/beamlet indices with beams/rays in stf

### Changed

- Performance improvements for raytracer and dij filling

### Fixed

- Ray tracer recovery in case of numerical issues
- CT validates given x/y/z vectors correctly
- Fixed validation of VOIs with single voxels imported from matRad

## [0.2.7] - 2025-06-27

## [0.2.6] - 2025-06-26

## [0.2.5] - 2025-06-20

## [0.2.4] - 2025-05-22

## [0.2.3] - 2025-05-09

### Added

- Slice visualization function
- LET calculation for protons

### Changed

- Scenarios are now pydantic models
- Docstring and code quality improvements

### Fixed

- Performance fix for raytracer
- Fixed issues with ray validation
- Various validation fixes

## [0.2.2] - 2025-02-27

### Added

- matRad-compatible data structures with stable validation and serialization using pydantic
- Native reimplementation of matRad's pencil beam dose calculation for photons, protons & ions
- Generic machine data
- Native optimization framework using scipy or IPOPT (via ipyopt)

## [0.2.1] - 2025-02-20

## [0.2.0] - 2025-02-13

## [0.1.0] - 2025-02-05

## [0.0.2] - 2025-01-10

## [0.0.1] - 2025-01-10

[Unreleased]: https://github.com/e0404/pyRadPlan/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/e0404/pyRadPlan/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/e0404/pyRadPlan/compare/v0.3.5...v0.4.0
[0.3.5]: https://github.com/e0404/pyRadPlan/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/e0404/pyRadPlan/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/e0404/pyRadPlan/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/e0404/pyRadPlan/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/e0404/pyRadPlan/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/e0404/pyRadPlan/compare/v0.2.8...v0.3.0
[0.2.8]: https://github.com/e0404/pyRadPlan/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/e0404/pyRadPlan/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/e0404/pyRadPlan/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/e0404/pyRadPlan/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/e0404/pyRadPlan/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/e0404/pyRadPlan/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/e0404/pyRadPlan/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/e0404/pyRadPlan/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/e0404/pyRadPlan/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/e0404/pyRadPlan/compare/v0.0.2...v0.1.0
[0.0.2]: https://github.com/e0404/pyRadPlan/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/e0404/pyRadPlan/releases/tag/v0.0.1
