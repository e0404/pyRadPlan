# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- multiple corresponding units for given quantities in the GUI

### Fixed
- missing keys in init_beam not being skipped

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
-
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

[Unreleased]: https://github.com/e0404/pyRadPlan/compare/v0.4.0...HEAD
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
