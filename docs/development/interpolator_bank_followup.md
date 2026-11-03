# Reusable interpolation bank and follow-up checks

2026-11-03, following [pipeline batching](particle_pipeline_batching.md).
Work remains on `refactor/dose-calc-performance`. The full develop baseline is
still deferred. Existing dose engines do not automatically enable batching or
use this new bank.

## Completed changes

`ArrayBackend1DInterpolatorBank` now lives in `core.xp_utils`. It prepares
unequal-length one-dimensional grids and compatible scalar/row-valued fields,
then accepts a table index and optional grid translation for every query. Queries
can be interleaved, duplicated or empty. Inputs stay on their backend/device.
Grid and index validation, dtype handling and backend dispatch belong to the
interpolation utility, rather than the engine.

The bank offers:

- NumPy grouping through the existing prepared interpolator.
- A portable Array API binary search shared by all fields, followed by gathers.
- An **opt-in** CuPy float64 specialization: one search kernel per query batch,
  then one evaluation kernel per field, including all its rows. Unsupported
  dtypes fall back to the portable path. CUDA multiply-add contraction is disabled
  to preserve the portable arithmetic order.

`prefer_native=None` selects NumPy grouping and portable execution elsewhere;
`False` forces portable execution; `True` permits the CuPy specialization.
`bank.implementation` reports the selected path. Generic/fused bank arithmetic
follows generic prepared interpolation; NumPy native preference retains its
existing dispatch and precision. Singleton/NaN semantics are covered explicitly.
There is one host-visible index/offset validity decision per call, not a claim of
zero GPU synchronization. No query or table arrays are downloaded for evaluation.

See the [API documentation](../api/core.rst) for an example. The pipeline harness
adds `bank` (explicit specialization preference) and `bank_generic` variants.
The original experimental `binary` implementation remains available for comparison.

The earlier PyTorch CT-grid tracing blocker is also fixed: linear image
interpolation now clips the upper neighbor index through Array API `clip`, rather
than passing a Python scalar to PyTorch `minimum`. Commit `d42db2ab` includes the
fix and expanded resampling tests: **220 passed**. A complete PyTorch phased
CT-grid calculation subsequently matched the scalar reference.

## Performance evidence and an important measurement limit

The isolated harness is
[`benchmark_interpolator_bank.py`](../../benchmark/benchmark_interpolator_bank.py).
It uses 18 variable-length tables, three fields and 64 query segments, with float64
tables and float32 queries. Construction and first-call costs are separate from
15 warmed, alternating evaluation samples. GPU timing synchronizes complete calls.
The segment reference includes concatenating its outputs, as needed by a packed
physics evaluator; it is not an isolated timing of one scalar interpolation.

| Backend | Queries | Prepared segments (ms) | Portable bank (ms) | Specialized bank (ms) |
|---|---:|---:|---:|---:|
| NumPy | 1,024 | 0.446 | 0.256 | 0.661 |
| NumPy | 16,384 | 1.437 | 2.089 | 2.113 |
| NumPy | 65,536 | 4.918 | 8.514 | 7.010 |
| PyTorch CPU | 1,024 | 6.234 | 0.915 | same portable implementation |
| PyTorch CPU | 16,384 | 19.915 | 8.129 | same portable implementation |
| PyTorch CPU | 65,536 | 18.823 | 19.265 | same portable implementation |
| CuPy, shared GPU load | 1,024 | 59.134 | 18.082 | 2.327 |
| CuPy, shared GPU load | 16,384 | 58.463 | 19.627 | 2.328 |
| CuPy, shared GPU load | 65,536 | 57.965 | 19.883 | 2.360 |

**The GPU measurements are exploratory, not a clean dispatch baseline.** After
the first smoke run, the same scalar proton workload changed from about 7.56 s to
1.58-1.65 s. `nvidia-smi` reported 99% GPU utilization while these benchmarks were
idle and listed another Python process. That work was left alone. The observed
device was an RTX 4070 Laptop GPU, driver 610.74, P0 at 2520 MHz SM / 8001 MHz
memory. The cause of the timing shift was not isolated.

The three-pair repeat observed portable-bank pipeline time 0.686 s against a
1.652 s scalar reference, and specialized-bank time 0.562 s against 1.580 s.
The initial one-pair smoke instead measured 1.231 s specialized and 0.654 s
portable, with scalar references about 7.56 s. Both are retained. Do not compare
the 0.562 s result directly with the earlier 1.632 s prototype to claim a new
speedup, or select a universal GPU default from these data.

NumPy still favors its existing scalar/segmented path for the larger workloads.
PyTorch also has a batch-size crossover. The fused implementation is therefore
available explicitly, while automatic CuPy bank dispatch stays portable until
idle-device measurements across sizes and modalities justify a policy.

Raw measurements and complete comparison errors are in
[`interpolator_bank/`](../../benchmark/pb_audit_results/interpolator_bank/).
First construction/call numbers include cold backend compilation effects and are
not balanced estimates of steady-state preparation cost. Validation runs with
one pair are correctness checks, not performance claims.

## Validation

The bank checks cover NumPy, PyTorch CPU, strict Array API and CuPy; float32/64
grids and queries, mixed field dtypes, row-valued fields, arbitrary table order,
translated grids, singleton tables, empty and strided queries, exact knots,
NaN/Inf samples/queries, extrapolation and rejection of unsafe indices. The CuPy
specialization is checked against portable interpolation, including nonfinite
sample behavior. The tests assert which implementation is actually selected.

The combined regression run passed **1,133 tests**: 304 bank tests, 208 existing
prepared-interpolation/particle-cache tests and 621 GUI tests. GUI source has no
consumer of the new utility; its dose calling interfaces remain unchanged.

Complete pipeline comparisons have zero maximum absolute difference for:

- CuPy protons, physical dose and LET.
- NumPy and PyTorch protons with Wedenberg biology, two CT phases, setup/range
  changes, a derived alias and CT-grid tracing.
- CuPy weighted forward dose with the same kinds of scenarios and biology.
- CuPy carbon with kernel-based LQ biology, including row-valued kernel fields.
- CuPy Focused VHEE with its anisotropic lateral model.

Sparse structure and delivery IDs match; the untimed warm-up compares selected
support fields exactly. Existing single-grid interpolation/cache tests also pass.
The specialization currently covers only float64 table grids and fields. Other
devices, arbitrary engine extensions and broader scheduler lifecycle validation
remain work for production batching integration.

## Next integration boundary

Keep packet scheduling separate from this backend utility. Candidate selection
and selected-dose evaluation still need independent bounds, with scalar fallback
and explicit scenario/delivery identities. The next engine change should introduce
those optional hooks and adopt the bank through its public interface; it should
not copy these kernels or table-search code into an engine. A measured execution
policy can then choose scalar, physics-only or full-pipeline batching.

```powershell
.venv/Scripts/python.exe benchmark/benchmark_interpolator_bank.py --backend numpy
.venv/Scripts/python.exe benchmark/benchmark_interpolator_bank.py --backend torch
# Repeat on an idle GPU before choosing automatic specialization:
.venv/Scripts/python.exe benchmark/benchmark_interpolator_bank.py --backend cupy
.venv/Scripts/python.exe benchmark/benchmark_particle_pipeline.py --backend cupy --size 64 --two-stage --variants bank_generic bank --pairs 3 --out benchmark/pb_audit_results/interpolator_bank/recheck
```
