# Opt-in production particle batching

2026-11-03, `refactor/dose-calc-performance`, following the
[hook overhead investigation](batching_hook_overhead.md). The full develop
baseline remains deferred.

## Implementation and use

The Hong particle engine now has a production `ParticleBatchProcessor` using the
shared scheduling hooks. Enable it explicitly through the normal plan options:

```python
pln.prop_dose_calc.update(
    batch_size=64,
    batch_selection_max_entries=65536,
    batch_evaluation_max_entries=65536,
)
```

`batch_size=0` remains the default; 1 also uses scalar execution. The selected
backend is respected. CuPy is the main motivation; measured NumPy workloads do
not justify enabling batching by default. The generated GUI configuration form
exposes the same properties, including optional `batch_max_bytes`.

The adapter separates:

- Scalar metadata preparation, including focus, SSD and range-shifter inputs.
- Exact packet support selection with independent candidate-entry limits.
- Interpolation through the public `ArrayBackend1DInterpolatorBank` interface.
- Shared lateral/physical/LET computation on selected work, bounded independently.
- Existing biological evaluator calls per beamlet, retaining their current
  context shapes and extension behavior.
- One host transfer per packed quantity and row array, followed by existing
  sparse or weighted-forward assembly using original delivery identities.

The scalar engine and bank now obtain their grids/fields from the same
`_depth_interpolation_data` helper. The processor does not inspect interpolator
private attributes or implement library-specific interpolation kernels. CuPy
fusion is requested through the existing `prefer_native=True` API; dispatch
remains inside `xp_utils`. Scalar fallback interpolators are prepared lazily.

A fresh processor prepares tables for each initialized beam/geometry group.
The scheduler drains before geometry cleanup. Custom numerical overrides retain
the scalar path with a warning; an engine extension can explicitly provide its
own processor. This prevents bypassing custom support, interpolation, modifier
or physical-dose implementations.

## Reusing selected packets

Selection creates a packet with selected arrays, owner IDs, host segment offsets,
table rows and its interpolation bank. Each beamlet keeps a packet reference and
its original position. The packet has no reference back to beamlets.

When all members arrive together at evaluation, the processor returns that packet
directly. It avoids concatenating the selected arrays and rebuilding owner/table
metadata. If a budget or compatibility boundary splits/regroups members, it packs
the new group instead. Empty members are retained and ordering is unchanged.
Singleton or oversized work uses the corresponding scalar stage routine.

Byte costs account for full packet bases retained by slices, conservatively
counted per queued member, plus a per-entry scratch estimate. They are estimates
for each queue, not a hard allocator/device peak: both queues, shared geometry,
tables and temporary storage can coexist. The default remains entry-count bounds
without an additional byte cap.

## Measurements

Sequential runs on the RTX 4070 Laptop GPU, CuPy 13.3.0. Complete calculation
timings synchronize GPU boundaries and include setup and collection. Comparison
work is untimed. Absolute times remain variable; use each paired comparison on
its own. No cross-session or full-develop speedup is claimed.

| TG119 workload | Paired repeats | Scalar median | Production batch median |
|---|---:|---:|---:|
| Protons, one beam, 5 mm grid, physical dose + LET, cap 64 | 3 | 7.637 s | 2.430 s |
| Focused VHEE, one beam, 200 MeV, 5 mm grid, cap 16 | 3 | 2.925 s | 2.806 s |

These final paired measurements show a 3.14x proton speedup, but only about 4%
less time for Focused VHEE. Packet count alone does not predict runtime benefit.

The proton calculation evaluates 908 beamlets in 15 packets. Focused VHEE uses
18 packets because its selected supports reach the entry cap before the
16-beamlet cap. Generic VHEE retains 121 singleton selections and eight evaluation
packets (seven of 16, one of 9), checked separately. All quantities and sparse
structures in these comparisons match the scalar reference exactly.

An alternating seven-pair carbon calculation with kernel-based LQ biology
compares **the same production processor**, forcing repacking versus allowing
reuse. All 19 packets are reusable. Medians are **5.476 → 5.267 s**, about 4%
less time. Paired differences average −0.263 s with SD 0.412 s; this timing spread
does not establish a reliable universal 4% gain. The removal of 19 repacking
operations is confirmed, and physical/biological matrices match exactly.

Final timings and pre-run source hashes are retained in
[production_batches/final](../../benchmark/pb_audit_results/production_batches/final/).
The other result folders retain preliminary measurements and correctness checks.
Some preliminary runs preceded lazy fallback preparation and recorded hashes at
the end of the run; use the final folder for the completed implementation's timing
evidence. Comparisons within each run used the same loaded code.

## Correctness and compatibility

The combined dose, scheduler, interpolation, configuration and GUI regression
selection passed **1,339 tests**. After the final budget-validation and custom
biology-override safeguards, the focused production/GUI checks passed **94 tests**.
Ruff lint and formatting checks also passed.

The numerical checks cover:

- Proton, carbon, helium and oxygen fixtures, including single/double/multiple
  Gaussian models, every computed quantity, sparse indices and delivery columns.
- Focused and Generic VHEE on CuPy.
- Float32/float64 packet physics on NumPy, PyTorch CPU, strict Array API and CuPy,
  for all four lateral models, ragged/empty support and invalid dose rejection.
- Direct packet identity reuse and safe repacking after a split.
- CPU influence matrices with two beams, two CT phases, setup/range changes,
  a derived alias, Wedenberg biology and dose-grid tracing.
- CuPy weighted-forward results with the same kinds of scenarios and biology,
  using CT-grid tracing, including output image metadata.
- Tight entry/byte budgets, engine reuse, injected failures, custom numerical
  override fallback, invalid settings and GUI configuration round trips.

Complete recorded scenario outputs have zero maximum absolute error. One-pair
scenario/budget measurements are correctness checks, not reliable performance
estimates. In particular, NumPy batching remains slower in the checked phased
case; the scalar default is intentional. Strict Array API testing caught and
fixed a missing explicit trailing slice dimension before integration.

## Remaining work

The particle adapter is opt-in; automatic backend/workload policy remains open.
Per-beamlet biological evaluation is still a substantial opportunity for models
whose contexts can safely be combined. The shared scheduler can support photon
processors and future fine-sample aggregation independently; neither is enabled
by this particle change. Candidate acceleration and further synchronization/output
work remain in the [backlog](dose_performance_backlog.md).

```powershell
.venv/Scripts/python.exe benchmark/benchmark_particle_pipeline.py --backend cupy --size 64 --pairs 3 --variants production --out benchmark/pb_audit_results/production_batches/final
.venv/Scripts/python.exe benchmark/benchmark_particle_pipeline.py --backend cupy --mode VHEE --machine Focused --size 16 --pairs 3 --variants production --out benchmark/pb_audit_results/production_batches/final
.venv/Scripts/python.exe benchmark/benchmark_particle_pipeline.py --backend cupy --mode carbon --bio-model kernel_based_lq --size 64 --pairs 7 --variants production_repack production --paired-variants --out benchmark/pb_audit_results/production_batches/final
.venv/Scripts/python.exe -m pytest -q test/dose/test_particle_batching.py test/gui/widgets/test_particle_cutoff_config.py
```
