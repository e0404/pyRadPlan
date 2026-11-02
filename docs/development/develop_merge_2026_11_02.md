# Develop integration, 2026-11-02

Integration completed 2026-11-03.

Merge `develop` / `origin/develop` at `078ce0f0` into
`refactor/dose-calc-performance` at `51a7f1d4`. Origin was fetched before merging;
the local and remote develop tips agreed. Develop itself is not modified.
The [outstanding performance backlog](dose_performance_backlog.md) is preserved
separately; this merge does not implement the next optimization.

## Conflict resolution and semantic compatibility

Seven files had text conflicts: CHANGELOG.md, cst/_cst.py, dose/engines/_base.py,
dose/engines/_base_pencilbeam.py, optimization/objectives/_objective.py,
test/cst/test_cst.py, and test/test_raytracersiddon.py. Both branches' tests are
retained. Released changelog history comes from develop; the performance branch's
entries remain under Unreleased rather than being attributed to an upstream release.

| Area | Conflict or risk | Integrated behavior |
| --- | --- | --- |
| Scenario execution | Legacy multidimensional masks and scenario numbers have been replaced. | Keep develop's explicit ScenarioSet, setup geometry groups, phase selection, derived scenarios and public output contracts. |
| Photon sampling | The old override would no longer run; old scenario-number bounds could be wrong or unavailable. | Attach bounds in `_apply_scenario_to_ray`, keyed by `(absolute, relative)` range values, with one conservative count per distinct computed range shift. Prepare once per beam/setup group; preserve RNG consumption. |
| Tissue preparation | Repeating a single label map across CT phases loses moving structures and tissue parameters. | Resolve overlaps once, build one label column per phase, resample labels once per phase, and map to the ScenarioSet's voxel-parameter columns. Keep the combined alpha/beta/tissue preparation. |
| Voxel selection | The old multiscenario fallback used flattened 4D indices and could return tuples from nonzero. | Retain the nominal mask-union fast path, but use develop's phase-local VOI indices for 4D cases; nearest-neighbor masks use flatnonzero on every phase. |
| SSD | Develop needs one SSD per phase; keeping the old scalar path would break particle focus/range offsets. | Keep per-phase SSDs and missing-ray handling, downloading SSD/validity vectors once per phase. Replace array-method squeeze with Array API reshape. |
| Sparse output | Develop allocates only computed flat slots; naive resolution would restore int64 everywhere or allocate derived slots. | Keep computed-only allocation/finalization, safe int32 indices and overflow promotion, and upstream derived/alias handling. |
| Machine and kernel state | Moving base initialization could bypass develop's loaded-machine resolver and phase mappings. | Keep `_resolve_machine`, per-calculation settings restoration, phase-specific biological gathers, and particle cutoff-cache invalidation per geometry group/beam. |
| Radiological depths | Develop's scenario tests intercepted the image API, but production now uses backend arrays. | Keep `trace_cubes_array` and backend resampling for both tracing-grid modes; adapt the offset reference test to intercept the array API. |
| Lazy imports | Upstream settings initialization imports core paths; eager resampler exports then re-entered settings through xp_utils. | Resolve core resampling exports lazily using the existing package helper; preserve public names and the compatibility module. Fresh-interpreter tests cover both import orders. |
| Benchmark metadata | Upstream restores engine settings after each call, including a dose-grid dictionary supplied by the plan. | The main audit harness captures effective grid/model metadata during finalization, before restoration, for both forward and influence calculations. |
| Shared interpolation/resampling | Choosing upstream imports wholesale would restore deprecated resampling calls. | Keep `resample_array`, prepared interpolators, shared photon component evaluation and existing backend dispatch. |

## Performance implications

- Prepared ray transforms and compact beam geometry remain local to the beam in
  its setup group. They cannot be reused across changed setup shifts.
- A 4D CT requires phase-specific SSDs, depth data and tissue labels. These costs
  scale with the phases processed; dropping that work would change the result.
- Photon sampling bounds conservatively cover all CT phases and are shared by
  scenarios with the same range shift. Derived scenarios do not require bounds.
- Particle beam cutoff recalibration and cache invalidation from develop are
  retained. Extending kernel cache lifetime without accounting for geometry would
  reintroduce stale cutoffs. Reference-phase cutoff policy remains upstream's.
- Flat scenario containers and derived/alias outputs follow develop. Historical
  benchmark scripts that restore frozen pre-scenario allocation, initialization
  or collection methods may need adapters before they can compare multi-scenario
  runs. Frozen reference snapshots have not been silently rewritten.
- No new speedup is claimed by this merge. Earlier CPU/GPU measurements remain
  workload- and revision-specific, especially the variable proton recheck.

## Validation

Added combined-tissue comparisons against the independent getters for moving
4D structures, with and without resampling. Added sampled photon comparisons
against per-ray count discovery on NumPy/CuPy with both tracing grids, reordered
CT phases, setup/range changes, derived aliases, exact sparse structure and RNG
state checks. Extended phase-specific proton biological dose comparisons to both
backends and both tracing grids. Updated sparse and voxel-selection fixtures to
the explicit scenario contract; retained strict Array API SSD coverage.

Executed checks:

- Full `pytest -q --disable-warnings --maxfail=10`: **3,824 passed, 14 skipped,
  2 failed** in 482.90 s. Both failures were fresh-interpreter imports exposing
  the core/settings cycle described above; they were fixed afterward.
- Post-fix `pytest test/test_lazy_imports.py test/ai/agents/test_lazy_import.py
  test/core test/test_settings.py -q --disable-warnings --maxfail=5`:
  **623 passed, 7 skipped** in 43.31 s, including both previously failing cases
  and two additional import-order regressions. The entire suite was not rerun
  after this import-only fix; there are no known unresolved test failures.
- Targeted scenario/cache/backend integration run before the full suite:
  **19 passed**. The full suite also includes these checks and both new tissue tests.
- Final end-to-end TG119 influence smoke calculations: photons (sampled), protons
  and VHEE, each on NumPy and CuPy, all completed with int32 sparse indices.
  These are single cold runs, not a performance comparison; metadata lives in
  `benchmark/pb_audit_results/develop_merge_2026_11_02/`.
- Pre-commit checks and staged Git whitespace checks passed.

Reproduction of the smoke checks:

```powershell
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend cupy --mode protons --grid 5 --bixel 10 --beams 1 --sampling --runs 0 --no-matrices --out benchmark/pb_audit_results/develop_merge_recheck
```

Repeat for numpy and the photons/VHEE modes. The harness only enables sampling
for photons. Existing frozen reference methods retain their historical semantics;
use scenario-aware tests rather than legacy benchmark adapters for 4D correctness.
