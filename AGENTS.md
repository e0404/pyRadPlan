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
| `rc/X.Y.Z` | release candidate, branched from `develop`, merged into `main` |

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

GitHub CI runs on Python 3.10–3.14. GitLab CI runs on 3.13 for every push, and also on 3.10 and 3.14 for open merge requests, `main`, `develop`, `rc/*`, `devops/*` and tags. Do not reduce existing coverage.

Performance work lives in `benchmark/`, never in `test/`. `testpaths` is set to `test`,
so a bare `pytest` run never collects it — invoke it explicitly:

```bash
pytest benchmark/benchmark_interp1d.py                 # one harness ([dev])
pytest benchmark/ -o python_files="benchmark_*.py"     # all of them
python benchmark/profile_pencilbeam_photon_forward_lineprof.py  # line_profiler ([profiling])
```

Keep the `benchmark_*.py` / `profile_*.py` prefixes when adding new performance work.

## Changelog

Update [CHANGELOG.md](CHANGELOG.md) under `## [Unreleased]` following [Keep a Changelog](https://keepachangelog.com/) conventions. Categories: `Added`, `Changed`, `Fixed`, `Removed`, `Deprecated`, `Security`.

## PR / MR Checklist

- [ ] All existing tests pass; new behaviour has tests
- [ ] Coverage not reduced
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] Branch targets `develop` (not `main`)
- [ ] Changes to data structures or algorithm calling APIs checked for GUI compatibility (`pyRadPlan/gui/` usages, `pytest test/gui`)

## Releases

Releases are cut from a release candidate branch, never from `develop` directly, so `develop`
stays open for regular work while the candidate is stabilised:

1. Branch `rc/X.Y.Z` from `develop` and open the release MR/PR from `rc/X.Y.Z` into `main`.
2. Review `rc/X.Y.Z` and address the remaining concerns on it.
3. Once the review is addressed and approved and CI is green, commit the executed example
   notebooks: `python docs/execute_examples.py --release` runs the examples and stages them in
   `docs/tutorials/examples/`.
4. Bump the version only now that the release code is final: `pyproject.toml`, `CHANGELOG.md`
   (`[Unreleased]` becomes `[X.Y.Z] - YYYY-MM-DD`), `CITATION.cff`. This is the last commit
   before merging.
5. Merge `rc/X.Y.Z` into `main`.
6. After merging, tag `vX.Y.Z` on `main`, verify the published release and the PyPI upload, merge
   `main` back into `develop` while untracking the executed notebooks (`git merge --no-ff
   --no-commit main`, `git rm -r --cached --quiet --ignore-unmatch docs/tutorials/examples`,
   `git commit`), and delete `rc/X.Y.Z`.

When proposing or preparing a release, follow
[.gitlab/merge_request_templates/Release.md](.gitlab/merge_request_templates/Release.md) — it is
the authoritative checklist for both the pre-merge and post-merge steps, and it applies whichever
platform hosts the release. Do not tag, publish, or push a release without explicit confirmation.

## Documentation
If the changes are complex, consider adding or suggesting an addition or change to the user guide (in docs/user_guide).
If you add documentation, you can run multiple builds of the documentation and iterate through the sphinx-build errors to make clean edits.

The scripts in `examples/` are jupytext percent notebooks rendered in the docs "Tutorials" section. On `develop` and feature branches they are the only committed form of the examples: never commit `.ipynb` files there. The docs build converts them into `docs/tutorials/examples/` (git-ignored) without outputs; neither CI nor Read the Docs ever executes them. Only release commits (`rc/X.Y.Z`, hence `main` and the tags) carry executed notebooks there, written and staged by `python docs/execute_examples.py --release`, so the docs show outputs for releases only. The docs build renders an executed notebook as is while its script is unchanged (it records the script hash) and otherwise falls back to a conversion without outputs with a warning. Examples needing external tools (`SKIP` in `docs/execute_examples.py`) are never executed. Check that changed examples run with `python docs/execute_examples.py [name ...]` (writes to `docs/_build/executed_examples`); `PYRADPLAN_DOCS_NB_EXECUTION=cache` is a local opt-in to execute them during a docs build.

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
