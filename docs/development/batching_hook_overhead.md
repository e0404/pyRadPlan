# Batching hook overhead investigation

2026-11-03, following [fusion measurements and hook integration](cupy_fusion_and_batching_hooks.md).
Starting revision: `582bb77a`, on `refactor/dose-calc-performance`.
The full develop baseline remains deferred.

The earlier approximately 50 ms proton difference is **not established as an
intrinsic scheduling cost**. A host-only replay of the actual work finds about
3 ms of additional scheduling overhead. A small validation shortcut reduces it
further. There is no evidence of extra GPU operations in this proton adapter
comparison, and no reason from these results to redesign the shared hooks.

## Full-calculation recheck

TG119, one proton beam, 10 mm beamlet spacing, 5 mm dose grid, physical dose and
LET, 908 beamlets. Both variants use fused CuPy interpolation, 65,536-entry stage
limits and a 64-beamlet evaluation cap. Warm-ups are excluded and variant order
alternates within pairs. Complete calculation boundaries are synchronized;
correctness comparisons are outside the timers. Benchmarks ran sequentially.

The RTX 4070 Laptop GPU reported 0% utilization before the first run. Absolute
times nevertheless changed substantially from the preceding session. The cause
has not been isolated; idle utilization alone does not establish stable timing.

| Backend | Pairs | Old prototype | Shared hooks | Mean paired difference, hooks minus old | SD of paired differences |
|---|---:|---:|---:|---:|---:|
| CuPy | 11 | 2.504 s | 2.559 s | +49 ms | 174 ms |
| NumPy | 7 | 0.484 s | 0.483 s | −2 ms | 29 ms |

Calculation times are medians. CuPy paired differences range from −339 to +319 ms.
These observations do not reliably isolate a 50 ms hook cost. They also do not
justify comparing absolute times from separate sessions to claim a speedup.

## What profiling rules out in this case

The warmed `cProfile` comparison records three calculations per variant. Both
have the same 23 selection packets, 15 evaluation packets, packet lengths,
1,447,803 candidate pairs and 933,649 selected pairs. Every one of the 44 retained
CuPy-related function rows has identical call counts. Examples per calculation:

| Counted operation | Old prototype | Shared hooks |
|---|---:|---:|
| CuPy array construction | 1,263 | 1,263 |
| CuPy `take` | 977 | 977 |
| CuPy concatenation | 272 | 272 |
| CuPy `get` | 385 | 385 |
| `evaluate_entries` (selected-work packing) | 15 | 15 |
| Existing per-column dose collector | 908 | 908 |

These are whole-calculation API call counts, not CUDA kernel traces or byte
counts. They rule out additional calls to these operations as the explanation
for the proton gap; they do not prove identical GPU scheduling or memory-pool
behavior. The numerical arrays and output matrices match exactly.

New host work includes 908 delivery/work wrappers, 1,816 stage-cost objects,
metadata validation and two queue traversals. `cProfile` amplifies Python call
overhead and includes native/GPU waits in wall times, so its timings are not used
to claim an end-to-end speedup. The report retains all project and CuPy rows,
plus expensive external routines; omitted small external rows are counted.

## Isolated host replay and the implemented improvement

The replay captures real NumPy TG119 candidate/selected lengths, field-schema
groups and delivery identities. It replaces arrays with shape metadata and
omits numerical work/transfers. The old path reuses the prototype's queue methods;
the new path uses the production scheduler and cost/delivery objects. Every
replay must reproduce the captured numerical calculation's complete evaluation
packet identities. It is a scheduling microbenchmark, not a simulated dose time.

Medians of 101 alternating repetitions after warm-up:

| Host-only replay | Time |
|---|---:|
| Old prototype queue | 0.873 ms |
| Shared hooks, original validation | 4.043 ms |
| Shared hooks, Python-integer shortcut | 2.771 ms |

The shortcut saves **1.27 ms**, about 31% of this shared-scheduler replay. The
remaining difference from the old prototype is about 1.90 ms. The initial replay
gave the same conclusion: 0.942 / 4.281 / 2.913 ms respectively.

Array shapes and Python counters normally already contain built-in integers.
`BixelBatchCost` previously dispatched through `numbers.Integral`'s abstract
class checks for each value. `_integer` now returns immediately for a valid
built-in integer and retains the old validation for every other input:

```python
if type(value) is int and value >= minimum:
    return
# Existing bool rejection, Integral check and lower-bound check follow.
```

NumPy integer types remain accepted. Booleans, floats, negative costs and invalid
limits remain rejected. No validation, error messages, scheduler budgets or
numerical operations were removed. No engine dispatch policy changed.

## Full-dose check of the shortcut

A separate alternating comparison runs **the same shared hooks** with either
the frozen old integer validator or the new shortcut. It does not compare the
old prototype scheduler against the new one. Eleven pairs per backend:

| Backend | Old validator median | New validator median | Mean paired difference | SD of differences |
|---|---:|---:|---:|---:|
| CuPy | 0.476 s | 0.457 s | −13 ms | 41 ms |
| NumPy | 0.499 s | 0.485 s | −12 ms | 34 ms |

Both outputs match the scalar reference exactly in every checked calculation.
The timing spread is too large to attribute these full-dose differences to the
roughly 1 ms host saving. **No reliable whole-dose speedup is claimed.**
In these files, `bank` labels the shared hooks with the frozen validator and
`bank_hooks` labels the shared hooks with the shortcut; accompanying
`comparison.json` files record that mapping.

Validation: **27 tests passed**, covering metadata types/bounds, independent
budgets, ordering, failures and photon/proton influence/forward lifecycle tests.
The public dose interface and GUI bindings are unchanged. Ruff and formatting
checks pass. The preceding 1,101-test integration run is recorded in the parent
report; this follow-up reruns the focused checks for the small private validator
change rather than claiming a new full-suite run.

## Next steps

Keep the shared hook architecture. In this workload, bookkeeping cost is small beside
selection, interpolation, ray/beam preparation and output work. The next useful
step is an opt-in production particle processor with backend-specific execution
policy, keeping NumPy scalar where batching loses and interpolation dispatch in
`xp_utils`.

During that integration, preserve already packed selected arrays when selection
and evaluation boundaries coincide. The earlier prototype has an immediate
evaluation shortcut; the current hook adapter always enters `evaluate_entries`.
It does not add packing calls in the measured proton case, but other workloads
can exercise that difference. This is an adapter-level follow-up, not evidence
that the shared scheduler needs engine-specific array logic.

Raw timings, capture records and profiles:
[hook_overhead](../../benchmark/pb_audit_results/hook_overhead/).

```powershell
# Original prototype versus shared hooks, without profiling:
.venv/Scripts/python.exe benchmark/benchmark_particle_pipeline.py --backend cupy --size 64 --pairs 11 --two-stage --variants bank bank_hooks --paired-variants --out benchmark/pb_audit_results/hook_overhead/baseline
# Warmed call-count/time diagnosis (instrumented timings are not speed evidence):
.venv/Scripts/python.exe benchmark/profile_batch_hook_overhead.py --backend cupy --size 64 --pairs 3 --two-stage --variants bank bank_hooks --paired-variants --out benchmark/pb_audit_results/hook_overhead/profile
# Real-work capture and CPU-only queue replay:
.venv/Scripts/python.exe benchmark/benchmark_batch_scheduler_overhead.py
# Isolate the validator in otherwise identical shared-hook dose calculations:
.venv/Scripts/python.exe benchmark/benchmark_batch_scheduler_overhead.py --full-backend cupy --pairs 11 --out benchmark/pb_audit_results/hook_overhead/validation_cupy
.venv/Scripts/python.exe -m pytest -q test/dose/test_batching.py test/dose/test_batching_engines.py
```
