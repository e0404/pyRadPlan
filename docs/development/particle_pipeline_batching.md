# Particle pipeline batching: CPU and GPU investigation

2026-11-03, following [physics-only batching](particle_batching_exploration.md).
Reference: scalar production code at `a780b35c`, which already includes prepared
particle interpolation and the develop merge. This is **not** the deferred full
comparison against develop. Production dispatch remains unchanged.

Subsequent implementation and validation:
[reusable interpolation bank](interpolator_bank_followup.md), including opt-in
CuPy fusion and the limits of the later GPU measurements.

## Findings

Bounded pipeline batching substantially reduces CuPy proton runtime. The larger
prototype cuts this small TG119 calculation from 7.52 s to 1.63 s (4.6x speedup,
78% less time). NumPy remains faster at about 0.44 s on this workload; the GPU
improvement does not establish a CPU/GPU crossover. The same generic batching
strategy regresses NumPy and should not become its default.

The main architectural finding is that **candidate selection and selected-dose
evaluation need separate packet limits**. VHEE has many candidate voxels but a
small surviving support. A single packet limit can inadvertently serialize its
physics, even when interpolation and physics functions accept batches.

## What is actually batched

The prototype is in
[`benchmark_particle_pipeline.py`](../../benchmark/benchmark_particle_pipeline.py).
Its numerical table-bank experiment is separate in
[`benchmark_grouped_interp.py`](../../benchmark/benchmark_grouped_interp.py).

| Variant | Support selection | Interpolation | Physics/checks | Host collection |
|---|---|---|---|---|
| Scalar reference | Per beamlet | Prepared, per beamlet | Per beamlet | Per beamlet |
| Previous `physics` | Per beamlet | Prepared, per beamlet | Packed packet | Per beamlet |
| `native` pipeline | Candidate packet | Group existing prepared interpolators by energy and offset | Selected packet | One download per quantity and row vector per packet |
| `binary` pipeline | Candidate packet | Owner-tagged queries against a prepared bank of different grids | Selected packet | Same as native pipeline |

`native` means reusing the existing backend interpolation dispatch, not forcing a
library implementation. `binary` is an experimental Array API multi-table binary
search. It is **not one fused GPU kernel**: its search iterations, gathers and
field arithmetic launch multiple operations. Table preparation stays outside the
engine arithmetic; an eventual public implementation belongs in `xp_utils`.

The packed layout concatenates surviving voxel entries and carries ownership and
segment offsets. Batching is the decision to execute several beamlets together;
packing is just the representation of those entries. Python still initializes
individual beamlets, and ray geometry is unchanged. Biology remains per beamlet.

With `--two-stage`, candidate packets feed a separately bounded selected-work
queue. Singleton selection uses the existing scalar selection path. Source-ray
references are dropped once selection finishes. Oversized beamlets use the scalar
fallback; queues flush before beam/geometry replacement and finalization.
Delivery arguments remain attached to each beamlet, including its scenario slot.

## Repeated whole-calculation timings

TG119, one beam, 10 mm spot spacing, 5 mm dose grid, Generic machines; physical
dose plus LET where available. Each row reports medians of three alternating
scalar/prototype pairs, after warm-up, with a fresh engine for each calculation.
The dose interval includes preparation, support, interpolation, physics, output
transfers and sparse assembly. STF generation and engine construction are outside
it. CuPy is synchronized before and after the complete calculation, without
artificial synchronization between timed stages.

Hardware: RTX 4070 Laptop GPU, Intel family 6/model 183 CPU; NumPy 2.3.5,
CuPy 13.3.0, PyTorch 2.13.0+cu126 (24 CPU threads).

| Backend / workload | Prototype | Beamlet limit | Scalar (s) | Prototype (s) |
|---|---|---:|---:|---:|
| CuPy protons | Physics only | 16 | 7.589 | 3.935 |
| CuPy protons | Native, single stage | 16 | 7.534 | 2.901 |
| CuPy protons | Binary, single stage | 16 | 7.586 | 3.077 |
| CuPy protons | Native, two stages | 64 | 7.590 | 2.425 |
| CuPy protons | Binary, two stages | 64 | 7.518 | **1.632** |
| NumPy protons | Native, two stages | 64 | 0.514 | 0.566 |
| NumPy protons | Binary, two stages | 64 | 0.443 | **0.718** |
| PyTorch CPU protons | Native, two stages | 64 | 1.982 | 1.723 |
| PyTorch CPU protons | Binary, two stages | 64 | 1.988 | 1.562 |
| CuPy VHEE | Physics only | 16 | 1.519 | **1.079** |
| CuPy VHEE | Native, single stage | 16 | 1.487 | 2.269 |
| CuPy VHEE | Binary, single stage | 16 | 1.518 | 5.728 |
| CuPy VHEE | Native, two stages | 16 | 1.503 | 1.174 |
| CuPy VHEE | Binary, two stages | 16 | 1.500 | 1.199 |
| NumPy VHEE | Native, two stages | 16 | 0.473 | 0.498 |
| NumPy VHEE | Binary, two stages | 16 | 0.471 | 0.552 |
| PyTorch CPU VHEE | Native, two stages | 16 | 1.099 | 1.062 |
| PyTorch CPU VHEE | Binary, two stages | 16 | 1.083 | 1.170 |

These are small workloads and three pairs, not confidence intervals or a universal
policy. NumPy scalar timings vary noticeably between runs. The large CuPy change
is repeatable within these runs; smaller changes need broader confirmation.
The size-16 and size-64 proton rows change both scheduling and maximum beamlet
count, so they do not isolate either change on its own.

All raw runs, including regressions, are retained under
[`pipeline_batches/`](../../benchmark/pb_audit_results/pipeline_batches/).
`*_submit_s` fields are host-side diagnostic intervals: they include submission
and incidental synchronization and **must not be treated as device-stage timings**.
Prepared-table submission was about 43 ms for the larger CuPy proton run; it is
included in the whole-calculation result, not an isolated synchronized creation
benchmark.

## Why the workloads differ

Protons: 908 beamlets, 18 energies, 1,447,803 candidate pairs and 933,649 selected
pairs. With the 65,536-pair cap and 64-beamlet cap, selection uses 23 packets and
interpolation/physics uses 15 packets. Selected packets contain 60-64 beamlets
except the last 30; their largest selected count is 65,445. The largest candidate
packet contains 65,500 pairs. Thus "64 beamlets at once" is an upper bound, not a
fixed rectangular allocation.

Native grouping has limited reuse: at size 16 it still makes 904 cutoff and 904
depth interpolation calls for 908 beamlets. At size 64 these become 855 and 833.
The binary table bank instead makes 23 cutoff-bank calls and 15 depth-bank calls.
That larger amount of work per invocation is where the generic bank becomes
useful on this GPU. Fewer Python calls alone do not imply fewer GPU kernels.

VHEE: 121 beamlets, 4,802,750 candidate pairs, but only 347,708 selected pairs.
A candidate packet cap of 65,536 fits just one beamlet (up to 44,500 candidates).
Single-stage scheduling therefore makes 121 physics packets and regresses badly.
Separate stages restore eight physics packets, up to 46,087 selected pairs each,
while retaining 121 scalar selections. Grouped native depth interpolation still
needs 78 calls because grid offsets differ. The binary bank needs eight calls,
but remains slower overall than the simpler physics-only prototype here.

The cap is a bound on entries **per stage**, not a measured total memory limit.
Two queues, temporary arrays, source geometry and prepared tables can coexist.
A production scheduler should estimate bytes from the active fields/component
counts and reserve room for temporaries, particularly for biology and future
fine-sampling or AAA.

## Correctness checks and limits

All recorded final dose comparisons have **zero maximum absolute difference**
against the scalar engine, including LET and Wedenberg biological quantities.
Influence comparisons check every scenario's sparse column pointers, row indices,
values and delivery IDs. Forward comparisons check every output image and its
physical metadata. The untimed prototype warm-up also compares batched support
selection with the original selection exactly, field by field.

Additional full-calculation checks cover:

- NumPy: two beams, two CT phases with a shifted outline, setup and range changes,
  a derived alias, Wedenberg biology, and CT-grid tracing.
- CuPy: nonuniform weighted forward dose for the same kinds of phase/setup/range
  changes and derived alias, with Wedenberg biology and dose-grid tracing.
- NumPy: Generic carbon and Focused VHEE (anisotropic lateral model).
- A 4096-pair limit and a limit of one. The latter returns all 908 proton beamlets
  to scalar evaluation, preserving output exactly.
- A 10 mm NumPy influence calculation explicitly checks that the derived alias
  retains the same sparse matrix object as its referenced scenario.

The independent table-bank check passes 12 cases on each of NumPy, PyTorch CPU,
strict Array API and CuPy. Each includes unequal-length and singleton float64
tables, float32/float64 queries, scalar and row-valued fields, translated grids,
knots, boundaries, NaN/Inf queries and empty query arrays, compared with prepared
backend interpolation (`rtol=3e-6`, `atol=1e-12`). One edge case matters: native
CuPy scalar-field interpolation propagates a NaN query even for a singleton grid;
the existing generic singleton path returns the constant sample. The prototype
preserves the selected backend path's behavior.

The existing prepared-interpolation and particle cache regression suites also
pass: **208 tests** (`test/core/xp_utils/test_interpolation1d.py` and
`test/dose/test_particle_interpolators.py`). The full repository suite was not
rerun for these benchmark-only changes.

These experiments do not establish arbitrary mixed table-dtype support, a full
GPU carbon validation, custom-engine compatibility, all biological models, or
production scheduler lifecycle coverage. Packet memory peaks were not measured.
There are no production API changes and no GUI call-site changes.

## Recommended integration sequence

1. Introduce optional prepare/select/evaluate/collect packet hooks with a scalar
   default. Preserve the current fast NumPy path. Keep candidate and selected-work
   scheduling independent, with explicit scenario/delivery identities and bounded
   memory. Particle/VHEE implementations can opt into the hooks; photon engines
   can adopt them without inheriting particle-specific fields or formulae.
2. Add prepared multi-table interpolation to the Array API interpolation backend,
   including owner IDs, per-table grid translations and grouped field evaluation.
   Dispatch there between existing prepared calls, generic table-bank execution
   and any later fused native implementation. Do not add interpolator fast paths
   to engine code. Preserve scalar-bound dtype promotion: PyTorch vector bounds
   otherwise change mixed-precision support decisions at table endpoints.
3. Enable a measured GPU execution policy after validation across resolutions,
   models and scenarios. Keep a simple physics-only option for cases such as this
   VHEE workload. Public configuration need not expose every experimental variant.
4. Then investigate conservative spatial candidates and fused table-bank kernels.
   VHEE still scans far more candidate entries than it keeps, and the generic
   binary search still issues many operations. Packet downloads reduce transfers;
   device-resident sparse assembly is a separate decision.

The common interfaces should describe query/support arrays, engine-owned fields
and output segments. Gaussian component counts, particle fine-sample children,
photon convolution/AAA components and biology belong in their engine evaluators.
This keeps batching extensible without imposing one physical model or one global
packet shape.

## Reproduction

```powershell
.venv/Scripts/python.exe benchmark/benchmark_particle_pipeline.py --backend cupy --variants physics native binary --pairs 3
.venv/Scripts/python.exe benchmark/benchmark_particle_pipeline.py --backend cupy --size 64 --two-stage --pairs 3
.venv/Scripts/python.exe benchmark/benchmark_particle_pipeline.py --backend cupy --mode VHEE --two-stage --pairs 3
.venv/Scripts/python.exe benchmark/benchmark_particle_pipeline.py --backend numpy --size 64 --two-stage --pairs 3
.venv/Scripts/python.exe benchmark/benchmark_particle_pipeline.py --backend torch --size 64 --two-stage --pairs 3
.venv/Scripts/python.exe benchmark/benchmark_grouped_interp.py --backend cupy
```

Run timed benchmarks sequentially. The older physics-only harness still hides
CUDA when run directly; importing its helpers no longer modifies GPU visibility.
