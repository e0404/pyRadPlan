# Photon sampling bounds and shared output selection

The [develop integration](develop_merge_2026_11_02.md) adapts this cache to explicit scenario objects and range-value keys. Measurements below describe the original implementation.

Follow-up to [photon bixel evaluation](photon_bixel_preparation.md), comparing
against `5d7f6266` on `refactor/dose-calc-performance`.

## Implementation

The photon engine prepares a depth-cluster count for each range scenario once
per beam/shift, on first ray initialization after the complete subclass beam
setup. It reduces the finite, beam-valid radiological depths to extrema for each
CT scenario, evaluates both endpoints under the range shift using the same
backend scalar promotion as ray extraction, clips negative depths, and applies
the configured bin width. Both endpoints are evaluated to cover negative range
factors as well as the normal positive factors.

The count conservatively covers every CT scenario on that beam for the selected
range scenario. It is the maximum bin over that union, rather than a ray-specific
maximum. This can allocate a few unused bins for a shallower ray/CT scenario but
avoids a scan and scalar synchronization on every bixel. Empty/invalid CT supports
contribute only one unused bin. Positive finite bin width is checked during
preparation. Only the small vector of counts is downloaded, once per beam.

Each extracted scenario ray receives the corresponding host integer count.
The sampler still computes voxel group IDs on the backend. Its optional
`num_clusters` argument retains the old per-ray count calculation for standalone
calls without prepared beam state. `grouped_max` still validates group ranges;
its validation synchronization and the all-core early-exit check are unchanged.

Final selection now compacts the keep mask once and gathers both outputs:

```python
selected = xp.nonzero(keep)[0]
values = xp.where(ix_tail, threshold_dose, bixel_dose)
return xp.take(ix, selected, axis=0), xp.take(values, selected, axis=0)
```

This uses Array API operations and preserves ordered rows. Like Boolean indexing,
`nonzero` produces an input-dependent output size; this does not make the sampler
statically shaped. RNG generation, draw order, early exits, dose thresholds and
sampling probabilities are unchanged. No backend-specific sampling branch is
introduced.

## Validation and benchmark isolation

`profile_photon_sampling_reference.py` freezes the previous sampler, adding only
an optional supplied count to support isolated comparisons. The full-dose harness
provides these variants against the current implementation (`baseline`):

- `legacy_photon_sampling`: previous Boolean selection and per-ray counts;
  beam count preparation disabled.
- `sampling_legacy_count`: current selection, but per-ray counts and no preparation.
- `sampling_boolean_selection`: prepared counts, but previous Boolean selection.

The new `--paired-warmup` option runs both paths before timed comparisons. It
warms backend kernels without omitting beam preparation from any measured dose
calculation. Initial runs without that option remain in the raw results.

Focused tests exercise NumPy, CuPy, strict Array API and Torch, positive/negative/
zero range factors, absolute shifts, float32/float64 depth arrays, noninteger bin
widths, finite-mask holes, empty supports and invalid bin widths. They compare
bounds with binning all valid depths, check one count-vector transfer, verify
cache lifetime and scenario-number attachment, and preserve exact RNG consumption
for empty, all-core, zero-tail and ordinary-tail rays. Existing sampling-policy
tests compare seeded dose/indices with and without a conservative supplied count.

The isolated `profile_photon_sampling.py` measures all four sampler combinations
at 5k/50k/200k voxels and separately compares just Boolean selection versus shared
`nonzero`/`take` at 10%, 50% and 90% retention. It excludes beam-bound preparation;
full-dose timings include it. Every isolated comparison checks identical outputs.

## Complete-dose measurements

TG119, five-field IMRT, 3 mm dose grid, 5 mm bixels, 1,567 bixels, sampled
influence matrices. Backend versions: NumPy 2.3.5 and CuPy 13.3.0; GPU: RTX 4070
Laptop. Each measured calculation includes beam preparation. The final comparison
warms both implementations and alternates execution order over two pairs:

| Backend | Previous runs (s) | Current runs (s) | Previous median | Current median |
| --- | --- | --- | ---: | ---: |
| NumPy | 24.874, 24.239 | 24.437, 24.474 | 24.557 | 24.455 |
| CuPy | 9.225, 9.375 | 8.700, 9.837 | 9.300 | 9.268 |

**Neither backend has a demonstrated end-to-end speedup here.** Median differences
of 0.4% CPU / 0.3% GPU are smaller than run-to-run variation; the winner reverses
between pairs on both backends. Every comparison has exactly equal sparse dose
values, row indices and column pointers within its backend. Matrix storage is
unchanged (516,589,184 bytes NumPy; 516,617,568 bytes CuPy). Cross-backend numerical
identity is not implied.

Earlier runs are retained rather than discarded. Without symmetrical warm-up,
combined NumPy medians were 27.702 -> 25.403 s and CuPy 10.359 -> 10.365 s. Single
CPU isolation pairs gave 23.612 -> 25.984 s for prepared counts and 37.083 ->
29.452 s for shared selection. Those unstable full-dose figures do not establish
individual effects; use the isolated sampler measurements below for attribution.
In those initial full-dose runs, bounds were prepared at the end of the photon
base beam initializer. The final implementation and warmed measurements prepare
on the first ray, after the entire subclass beam initialization.

The change removes redundant count discovery and mask compaction, retains Array
API compatibility, and helps the isolated sampler. It should be regarded as an
incremental cleanup with a workload-dependent performance benefit, rather than
a major dose-calculation acceleration. Remaining synchronization in `grouped_max`,
the all-core check, host RNG and interpolation/geometry costs are unchanged.

A separate synchronized GPU stage profile measured **5.48 ms total** for bound
preparation across all five beams (about 1.10 ms per beam), including the count
vector transfer. This instrumented run is used for cost attribution only, not
for the speedup table. Preparation is a negligible fraction of the measured
whole calculation.

Additional exact full-matrix comparisons cover CT-grid tracing with a 5 mm dose
grid and one beam, for both standard photon bixels and field-shaped photon dose
on NumPy/CuPy. Across initial, warmed and CT-grid cases, all 14 paired comparisons
have zero dose difference and identical sparse structure.

## Isolated measurements

Median milliseconds per sampler call (five alternating rounds of ten calls,
float64 depths/doses, fixed seeded radius-policy data):

| Backend | Voxels | Previous | Count only | Selection only | Both |
| --- | ---: | ---: | ---: | ---: | ---: |
| numpy | 5,000 | 0.133 | 0.133 | 0.115 | 0.112 |
| numpy | 50,000 | 1.078 | 1.110 | 0.775 | 0.808 |
| numpy | 200,000 | 4.620 | 4.546 | 3.310 | 3.152 |
| cupy | 5,000 | 1.672 | 1.646 | 1.664 | 1.635 |
| cupy | 50,000 | 1.860 | 1.728 | 1.845 | 1.802 |
| cupy | 200,000 | 2.240 | 2.110 | 2.037 | 2.061 |

The count-only change has little measurable CPU benefit; it primarily removes a
GPU scalar read/reduction from each sampled bixel. The shared-selection change
accounts for most of the synthetic CPU benefit. These figures exclude beam-bound
preparation and are not full-dose speedups.

Selection alone depends on retention. At 50k voxels and 50% retention, shared
`nonzero`/`take` takes 0.130 ms versus 0.426 ms for two Boolean gathers on NumPy;
on CuPy, 0.150 ms versus 0.216 ms. At 90% retention, NumPy Boolean selection wins
(0.160 ms versus 0.243 ms). At 5k voxels and 90% retention, shared selection is
also slower on NumPy (0.025 versus 0.008 ms), though the absolute cost is small.
CuPy shared selection wins across the nine measured size/retention combinations.
There is no new backend or retention-based dispatch: a per-ray decision would
itself add work, and the complete workload is the acceptance criterion.

## Regression checks

- Selected dose-engine, sampling, ray-geometry and sparse-output tests: **318 passed**
  (384 unrelated tests deselected).
- Dij and GUI suites: **580 passed**.
- Ruff check/format and Git whitespace checks passed.
- The frozen executable sampler matches `5d7f6266`, except its optional prepared
  count argument; executable AST comparison ignores only docstring indentation.

## Reproduce

```powershell
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend cupy --mode photons --grid 3 --bixel 5 --beams 5 --sampling --runs 0 --variant legacy_photon_sampling --paired --paired-warmup --pairs 2 --no-matrices --out benchmark/pb_audit_results/photon_sampling/recheck
.venv/Scripts/python.exe benchmark/profile_photon_sampling.py --backend cupy --out benchmark/pb_audit_results/photon_sampling/isolated_recheck.json
.venv/Scripts/python.exe -m pytest test/dose -q -k "svdpb or ray_geometry_preparation or sparse_output or dose_engine_base or hongpb"
.venv/Scripts/python.exe -m pytest test/test_dij.py test/gui -q
```

Repeat with NumPy and the two isolated full-dose variants. Raw measurements and
full sparse comparison summaries live in `benchmark/pb_audit_results/photon_sampling/`.
