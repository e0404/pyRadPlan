# Agent Guide — pyRadPlan

pyRadPlan is a Python radiotherapy treatment planning toolkit, interoperable with matRad. It supports photon and ion (proton/carbon) planning via a Python Array API-compliant backend (numpy, cupy, pytorch).

## Setup

```bash
pip install -e .[dev]
pre-commit install
```

## Branch Naming

| Prefix | Purpose |
|---|---|
| `main` | stable releases only |
| `develop` | integration branch |
| `feature/*` | new functionality |
| `bugfix/*` | non-critical fixes |
| `hotfix/*` | critical fixes targeting main |
| `refactor/*` | internal restructuring |
| `interface/*` | API/interface changes |
| `devops/*` | CI, tooling, docs |
| `dev/*` | experimental |

Always branch from `develop` (not `main`) for regular work.

## Code Style

- **Formatter/linter**: `ruff` (enforced via pre-commit)
- **Line length**: 99 characters
- **Docstrings**: NumPy convention

## Testing

```bash
pytest                        # run all tests
pytest --cov=pyRadPlan        # with coverage
```

CI runs on Python 3.10–3.13. Do not reduce existing coverage.

## Changelog

Update [CHANGELOG.md](CHANGELOG.md) under `## [Unreleased]` following [Keep a Changelog](https://keepachangelog.com/) conventions. Categories: `Added`, `Changed`, `Fixed`, `Removed`, `Deprecated`, `Security`.

## PR / MR Checklist

- [ ] All existing tests pass; new behaviour has tests
- [ ] Coverage not reduced
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] Branch targets `develop` (not `main`)
- [ ] Changes to data structures or algorithm calling APIs checked for GUI compatibility (`pyRadPlan/gui/` usages, `pytest test/gui`)

## Documentation
If the changes are complex, consider adding or suggesting an addition or change to the user guide (in docs/user_guide).
If you add documentation, you can run multiple builds of the documentation and iterate through the sphinx-build errors to make clean edits.

## Key Architecture Notes

- Data models use **pydantic** for validation.
- Geometry / image handling via **SimpleITK**.
- Dose engines are modality-specific; photon and proton engines live under `pyRadPlan/engines/`.
- The Array API abstraction (`array_api_compat`) lets algorithms run on numpy, cupy, or torch without code changes — keep new numerical code array-backend-agnostic where possible.
- DICOM I/O via **pydicom**; matRad interop via HDF5 (**h5py**).

## Project-Specific Coding Rules

- Keep numerical code Array API-compatible where practical; avoid NumPy-only APIs in backend-agnostic paths.
- Preserve matRad interoperability when changing serialized data models or HDF5 import/export.
- Be careful with physical units, coordinate systems, voxel ordering, and SimpleITK image metadata.
- Prefer focused tests for geometry, dose, and serialization changes because regressions can be subtle.
- Treat dose engine changes as high-risk and validate against existing fixtures where possible.
- The GUI (`pyRadPlan/gui/`) binds directly to the pydantic data models and to algorithm calling APIs: widgets read/write model fields (e.g. `VOI` metadata, objectives, plan properties) and invoke dose engines, stf generators, and optimizers through their public interfaces. When changing a data structure or an algorithm's calling API, search `pyRadPlan/gui/` for usages, update the affected widgets, and run the GUI tests (`pytest test/gui`).
