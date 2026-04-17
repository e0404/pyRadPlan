# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- dose engines now have private flags `_dij_guarantee_canonical` and `_dij_guarantee_nonzero` to guarantee sparse dij structure before finalization

### Changed
- Sparse structure now uses one shared index structure across dijs if possible

### Fixed
- CI Release Workflow now tests correctly on release commits without an [Unreleased] section in the Changelog.
- CI Release Workflow now fetches tags correctly and supplies the correct release body from the tag message

## [0.3.3] - 2026-04-14

### Added
- Automatic Release workflow on GitHub reading CHANGELOG.md and tag message
- Benchmark Folder with inital Raytracer benchmark that can be run with pytest-benchmark

### Changed
- Changelog now follows thee Keep A Changelog conventions

### Fixed

- Fixed raytracer issue where certain geometrical configurations could lead to individual rays starting with invalid indices, inserting misplaced NaN's into the radiological depth cube.

## [0.3.2] - 2026-04-07

Small patch release with a new optimization objective, a structure for Beam Limiting Devices for field/shape-based dose calculation, several bug fixes, and a CITATION.cff.

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

Small update to drop `numba` as a mandatory dependency. `numba` accelerations are still available as an optional complement to Array API conform base implementations.

### Added

- Performant and Array API compatible candidate ray matrix setup alternative for cube raytracing
- Convenience plotting function to display multiple slices (`plot_distributions`)

### Changed

- `numba` is no longer a mandatory dependency

### Fixed

- Fix readthedocs YAML to correct Python version
- Small code quality fixes

## [0.3.0] - 2026-01-12

Array API compatibility, FRED Monte Carlo interface, VHEE model, Sphinx documentation, jupytext-style examples, refactoring, and bugfixes. **Python 3.9 support has been removed.**

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

Performance improvements, plan analysis via DVHs, and data validation fixes. Corresponds to the version used in the [SynthRad2025 challenge](https://github.com/SynthRad2025).

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

First official pre-release of pyRadPlan — an open-source radiotherapy treatment planning toolkit designed for interoperability with matRad.

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

[Unreleased]: https://github.com/e0404/pyRadPlan/compare/v0.3.3...HEAD
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
