# Idle-GPU fusion measurements and shared batching hooks

2026-11-03, `refactor/dose-calc-performance`, following the
[interpolator bank](interpolator_bank_followup.md). The full baseline against
develop remains deferred. These are computational equivalence and performance
checks, not a new validation of the underlying dose models.

Follow-up: [hook overhead investigation](batching_hook_overhead.md) isolates
about 3 ms of host scheduling cost and reduces it with a small validation shortcut.
The earlier 50 ms whole-calculation gap is not established as intrinsic hook cost.

## CuPy fusion measurements

The RTX 4070 Laptop GPU was available for this run; `nvidia-smi` reported 0%
utilization before starting. Measurements were run sequentially. CuPy 13.3.0,
NumPy 2.3.5. Timings synchronize the complete call/calculation, include output
collection, and exclude untimed correctness comparisons.

The isolated interpolation benchmark uses 18 unequal-length tables, three
float64 fields, float32 queries and 64 segments. Medians of 15 warmed calls:

| Queries | Prepared segments | Portable bank | Fused bank |
|---:|---:|---:|---:|
| 1,024 | 10.767 ms | 2.882 ms | 0.308 ms |
| 16,384 | 10.944 ms | 2.835 ms | 0.290 ms |
| 65,536 | 10.713 ms | 2.704 ms | 0.291 ms |

Fusion reduces this interpolation call by about **9–10×** versus the portable
bank. Construction and first-call samples are retained separately; they include
backend warm-up/compilation and are not balanced preparation-cost estimates.

For full-dose measurements, both variants are warmed, then their order alternates
within each of seven pairs. These compare the **same experimental batched
pipeline**, changing only the bank's native preference:

| TG119, one beam, 5 mm dose grid | Beamlets / evaluation cap | Portable | Fused | Time reduction |
|---|---:|---:|---:|---:|
| Protons, physical dose and LET | 908 / 64 | 0.620 s | 0.528 s | 14.9% |
| Carbon, physical dose | 1,182 / 64 | 1.152 s | 0.939 s | 18.5% |
| VHEE, Generic, 200 MeV | 121 / 16 | 0.437 s | 0.401 s | 8.3% |

All full outputs match the scalar reference exactly in these runs, including
sparse support and delivery IDs. Warm-ups also compare selected support fields.
Raw samples are in [fusion_idle](../../benchmark/pb_audit_results/fusion_idle/);
the `direct` subdirectory is the basis for this table.

The earlier scalar-versus-variant blocks are also retained, but scalar timings
still change substantially between blocks, including carbon 7.04 versus 1.68 s.
The cause has not been established. They must not be combined to claim a fusion
speedup. Alternating the variants directly reduces that confounding factor; this
is still one laptop GPU and a small set of workloads, not a universal speed ratio.

**Decision:** the existing float64 CuPy specialization is justified for the
particle batching adapter. Keep specialization inside `xp_utils` and use its
public `prefer_native=True` interface. This change leaves the general bank's
automatic policy and current scalar engine defaults unchanged. Float32 tables
and unsupported field types retain portable execution.

## Shared engine hooks

`PencilBeamEngineAbstract._create_bixel_batch_processor(dij, stf)` now optionally
returns a `BixelBatchProcessor` after beam preparation. Returning `None` retains
the original scalar calculation. All existing engines currently return `None`.
No plan property, GUI control, serialized model or public dose-call signature
changes.

The processor supplies five engine-specific operations:

1. `prepare(ray, index)` snapshots beamlet metadata into an owned payload.
2. `cost(work, stage)` and `compatibility_key(work, stage)` describe host-known
   work size and which payloads can safely execute together.
3. `select(batch)` finds exact support and releases candidate references.
4. `evaluate(batch)` computes dose quantities on selected support.
5. `collect(batch)` delivers results; the existing `_collect_bixel_batch` helper
   routes them through sparse or weighted-forward assembly.

The internal scheduler owns **independent** candidate and selected-work queues.
Each has a beamlet cap, an entry cap and an optional estimated-byte cap. Physics,
array packing and interpolation stay in the processor/backend. Thus a common
scheduler does not impose a particle-specific table layout on photon or future
AAA engines.

`BixelDelivery` preserves scenario, beam, ray, beamlet and original output column
as immutable metadata. Hooks mutate payloads in place and cannot replace or
reorder the work tuple. Empty supports are still delivered. Normal beam exit
drains both queues before geometry cleanup; failures discard pending work
without further evaluation/collection. A fresh processor is created for each
beam/geometry group, including subsequent engine calculations.

Single-item hooks (`select_one`, `evaluate_one`) allow the engine to use its
scalar implementation when only one item fits or an item exceeds the budget.
The scheduler cannot safely split an arbitrary engine's numerical problem.
Fine-sampled children should be aggregated within the processor into their
parent delivery payload; generic child-reduction semantics are not implemented.

The caps bound **queued estimates per stage**, not allocator peak memory. Both
queues, an incoming payload, shared geometry, views retaining full base arrays,
and execution scratch may coexist. Processors must estimate retained/scratch
storage conservatively and release candidate references. A single oversized
beamlet may exceed a stage cap even with scalar fallback. These limits are not
a device-wide hard memory guarantee.

Minimal integration shape (engine-specific numerical operations omitted):

```python
class MyEngine(PencilBeamEngineAbstract):
    def _create_bixel_batch_processor(self, dij, stf):
        return MyProcessor(self, dij, stf)


class MyProcessor(BixelBatchProcessor):
    def __init__(self, engine, dij, stf):
        super().__init__(
            selection_limits=BixelBatchLimits(max_bixels=64, max_entries=65536),
            evaluation_limits=BixelBatchLimits(max_bixels=64, max_entries=65536),
        )
        self.engine, self.dij, self.stf = engine, dij, stf

    # Implement prepare, cost, select and evaluate for this engine.
    # Override singleton hooks where the scalar routines avoid packing.

    def collect(self, batch):
        self.engine._collect_bixel_batch(batch, self.dij, self.stf)
```

## Numerical adapter and integration boundary

`benchmark_particle_pipeline.py --hooks` exercises the real particle numerical
prototype through these production hooks. It no longer intercepts `_fill_dij`
to schedule pending work. Use `bank_hooks`/`bank_generic_hooks` with
`--paired-variants` for direct comparisons with the earlier prototype scheduler.
The numerical adapter remains in `benchmark/`; this is not yet an automatically
batched production Hong engine.

For VHEE, selection can remain 121 singleton operations while evaluation forms
eight packets (seven of 16, one of 9). This is the reason for separate budgets:
large candidate support does not force small selected work to execute singly.

The first five-pair comparison of old versus shared scheduling measured protons
0.489 → 0.541 s and VHEE 0.422 → 0.394 s. Packet counts/support were unchanged.
This does **not** establish a scheduling speedup; in particular, the proton
adapter has overhead to investigate before production opt-in. The purpose of
these hooks is a common, tested execution lifecycle, not a performance claim
for wrapping existing code in more Python objects.

## Scalar-path performance and correctness checks

The scalar traversal comparison loads only `_calc_dose` from `51353010`, holding
all other current preparation/numerical code fixed. Fresh engines execute the
same one-beam TG119 case in alternating order, with an untimed warm-up. This is
an isolated hook-regression check, not a comparison with develop or the GUI's
five-field defaults. Photon sampling is disabled to make outputs deterministic.

| NumPy calculation | Pairs | Before hooks | Scalar default with hooks available |
|---|---:|---:|---:|
| Protons | 11 | 0.377 s | 0.382 s |
| Photons | 7 | 0.648 s | 0.639 s |

These differences are small relative to the run-to-run spread; neither establishes
a meaningful speedup or regression. All compared dose outputs are exact.

Additional numerical hook checks match the scalar reference exactly:

- NumPy proton influence matrices with two beams, two CT phases, range/setup
  changes, a derived alias, Wedenberg biology and CT-grid tracing.
- PyTorch CPU proton influence matrices with phased CT, range/setup changes,
  a derived alias, Wedenberg biology and CT-grid tracing.
- CuPy weighted forward results with two beams and the same scenario/biology
  features, including image metadata.
- CuPy influence matrices with 4,096-entry selection/evaluation caps and a
  65,536-byte estimated-storage cap, exercising smaller packets and scalar
  fallback. This check establishes correctness, not a tight-budget speed gain.
  The prototype estimates retained voxel arrays, including full packet bases
  retained by slices; its byte estimate excludes execution scratch.

Raw timings/errors are in
[batching_hooks](../../benchmark/pb_audit_results/batching_hooks/). The one-pair
scenario/budget runs are correctness checks, not reliable performance estimates.
NumPy and PyTorch full-pipeline batching remain slower on the checked CPU cases.

Scheduler tests cover independent stage limits, empty work, compatibility groups,
byte/entry limits, oversized scalar callbacks, immutable identities, ordering,
normal draining, hook failures and external interruption. Photon/particle
integration tests exercise influence and weighted-forward collection, computed
and derived scenarios, beam lifetimes, and engine reuse after an injected error.
The GUI still invokes the unchanged public dose APIs.

The combined regression run passed **1,101 tests**: 176 dose/lifecycle tests,
304 interpolation-bank tests (including CuPy), and 621 GUI tests. Ruff checks and
format checks passed for all changed Python files. Existing fixture/configuration,
ray-tracing, nonfinite-interpolation and deprecation warnings remain.

## Reproduction

```powershell
.venv/Scripts/python.exe benchmark/benchmark_interpolator_bank.py --backend cupy --out benchmark/pb_audit_results/fusion_idle
.venv/Scripts/python.exe benchmark/benchmark_particle_pipeline.py --backend cupy --size 64 --pairs 7 --two-stage --variants bank_generic bank --paired-variants --out benchmark/pb_audit_results/fusion_idle/direct
# Repeat the preceding command with --mode carbon, or --mode VHEE --size 16.
.venv/Scripts/python.exe benchmark/benchmark_particle_pipeline.py --backend cupy --size 64 --pairs 5 --two-stage --variants bank bank_hooks --paired-variants --out benchmark/pb_audit_results/batching_hooks/direct
.venv/Scripts/python.exe benchmark/benchmark_batch_hooks_scalar.py --mode protons
```

Next: promote the particle adapter incrementally after removing avoidable
packing/dispatch overhead and defining measured backend policies. NumPy should
retain its fast scalar path where batching loses. Photon processors can adopt
the same hooks independently, preserving sampling RNG order and exact support.
