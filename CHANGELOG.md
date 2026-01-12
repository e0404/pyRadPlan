# Changelog

## Version 0.3.0 - Pre-Release
This update features Array API compatibility, a FRED interface, VHEE model, documentation, examples in jupytext style, some refactoring and bugfixes.
We further are removing python 3.9 support due to array api compatibility.

### New Features
- Partly integrated Array API compatibility. Coming with GPU support for Cupy and Torch. Note: Python 3.9 support has been removed.
- FRED interface (Monte Carlo tool)
- VHEE planning with a Generic (unfocused) beam and a focused beam.
- `dij.compute_result_ct_grid()` now returns quantities per beam too.
- `create_body_segmentation()` method for the CST object
- Option to cancel solver at any iteration via keyboard input

### Bug Fixes & Performance
- Increased memory efficiency in dose calc
- Fixed overlap priorities when similar levels exist
- Fixed `np.floating` deprecation
- Fixed pydantic>=2.11 compatibility
- Fixed issues with single bixel calculations in Raytracer
- Elevated minimum required version of numpydantic
- Tuning of initial Scipy parameters

### User Experience
- Comprehensive documentation using Sphinx
- Added various examples and conform to jupytext norm

### Development and CI
- Refactoring cst, ct, machine and stf test data
- Adding python versions tests

## Version 0.2.8 - Patch Release
This patch release incorporates performance improvements, plan analysis in form of DVHs, and some data validation fixes.
It corresponds to the version used in the SynthRad2025 challenge, see also https://github.com/SynthRad2025

### New Features
- DVHCollection and DVH for plan analysis
- Maps to associate bixel/beamlet indices with beams / rays in stf

### Bug Fixes & Performance
- Ray Tracer recovery in case of numerical issues
- CT validates given x/y/z vectors correctly
- Fix for validating VOIs with single voxels from matRad
- Performance improvements for raytracer and dij filling

## Version 0.2.3 - Patch Release

### New Features
- Added slice visualization function
- Added LET for protons

### Bug Fixes & Performance
- Performance fix for raytracer
- Fixes issues with ray validation
- Various validation fixes

### Development
- Scenarios are now pydantic models
- Docstring / code quality improvements

## Version 0.2.2 - First Official Pre-release
First official pre-release of pyRadPlan. pyRadPlan is an open-source radiotherapy treatment planning toolkit designed for interoperability with matRad.

### Major Changes and New Features
- matRad compatible data structures with stable validation and serialization with pydantic
- native reimplementation of matRad's pencil beam dose calculation for photons, protons & ions
- generic machine data
- native optimization framework using scipy or IPOPT (via ipyopt)

### Disclaimers & License
- pyRadPlan is still a work in progress, thus we decided to not assign a major version number yet. Everything is still subject to change, so handle with care.
- DO NOT USE PYRADPLAN CLINICALLY - Check the LICENSE file and README.md for more infos.
