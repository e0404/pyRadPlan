# Particle interpolation preparation — CPU check

2026-11-03, `refactor/dose-calc-performance`, after merge `6ea49b74`.
This is a local comparison of the interpolation change against the immediately
preceding implementation. The full baseline against `develop` is deferred.
CUDA was hidden for all executions; another GPU task was running.

## Implementation

`core.xp_utils.ArrayBackend1DInterpolator` retains dispatch and field preparation
for repeated 1D queries. It reuses the existing interpolation kernels:

- NumPy keeps native `interp` for scalar-field dictionaries. No replacement with
  SciPy's regular-grid interpolator or a generic vectorized implementation.
- PyTorch packs compatible fields once, then evaluates the existing scripted
  kernel with one coordinate search shared across those fields.
- Different field dtypes stay separate. Biological multi-row fields retain their
  shapes. The generic Array API path follows the same packing strategy.
- `prefer_native=None` preserves existing native-dispatch choices; `True` prefers
  available native interpolation, including row-wise evaluation; `False` requests
  shared-search evaluation. The one-shot `interp1d` is unchanged.

The 1D object's field axis is last, matching `interp1d`; this differs from the
trailing-value layout of `ArrayBackendGridInterpolator`. Keeping these explicit
avoids silently changing the existing grid interpolator's photon dispatch.

```python
from pyRadPlan.core.xp_utils import ArrayBackend1DInterpolator

interpolator = ArrayBackend1DInterpolator(
    depths,
    {"idd": scaled_idd, "sigma": sigma, "let": let},
)
fields = interpolator(query_depths)
shifted_fields = interpolator(query_depths, grid_offset=-ray_depth_offset)
```

Particle/VHEE engines cache scaled IDD, selected lateral/LET/biological fields,
and the interpolator per converted kernel and biological-field presence. They
retain the original arithmetic `(kernel_depths + kernel_offset) - ray_offset`:
the backend translates the prepared grid; it does not move the offset into the
queries. Ray offsets include phase-specific SSD and range-shifter effects.

Converted kernels and prepared interpolation are invalidated together at beam
cutoff preparation and setup-group boundaries, and at calculation start. Cache
entries retain their source objects to prevent identity-key reuse. They do not
grow with the number of distinct ray offsets. Biological preparation is separate
from physical-only cutoff calibration.

Focus interpolation now retains the existing SciPy object per energy/focus for
one calculation, preserving linear extrapolation. Its cache resets before a
reused engine resolves the machine for the next calculation. Focus interpolation
remains a host operation; no new engine-specific numerical fast path was added.

## Measurements

Harness: [`benchmark_particle_interpolation.py`](../../benchmark/benchmark_particle_interpolation.py).
It freezes the two old methods from `6ea49b74`, alternates old/new order, warms
both paths, and creates a fresh engine on every full-dose run. Full timings include
interpolator construction and preparation. Captured microbenchmarks use actual
beamlet depth queries. Native NumPy dispatch and PyTorch's existing JIT are retained.

TG119, one beam, 5 mm dose grid, 10 mm bixel width, Generic machines, influence
matrices. Protons: 908 beamlets, single Gaussian, physical dose and LET. VHEE:
200 MeV, 121 beamlets, double Gaussian, physical dose (machine has no LET).
NumPy 2.3.5; PyTorch 2.13.0+cu126 **on CPU**, default 24 threads. CPU reports
Intel64 Family 6 Model 183 Stepping 1. Timings can vary with other activity on
this workstation, including CPU activity associated with the separate GPU job.

Median full-dose times:

| Workload | Backend | Previous | Prepared | Runtime change |
|---|---|---:|---:|---:|
| Protons, trace on dose grid | NumPy | 0.530 s | 0.513 s | -3.3% |
| Protons, trace on CT grid | NumPy | 0.620 s | 0.594 s | -4.2% |
| Protons, trace on dose grid | PyTorch CPU | 2.548 s | 2.062 s | -19.1% |
| VHEE, trace on dose grid | PyTorch CPU | 1.333 s | 1.166 s | -12.5% |
| VHEE, trace on dose grid, repeat | NumPy | 0.519 s | 0.523 s | +0.7% |

First four rows: five alternating pairs. Last row: fifteen alternating pairs.
The first five-pair NumPy VHEE batch was 0.492 → 0.526 s (+6.9%), which prompted
the longer repeat rather than accepting a regression. In the repeat, the median
*paired* percentage change was -1.4%, with an approximate 95% bootstrap interval
[-5.6%, +1.8%] (10,000 resamples, seed 41). Ratios of separate timing medians and
medians of paired ratios are different statistics. Treat NumPy as essentially
neutral to slightly faster: these measurements cannot establish a small gain or
exclude a small regression. The initial 7% slowdown was not reproduced.

Representative warm interpolation timings (largest captured beamlet):

| Workload | Backend | Queries / depth samples | Previous | Prepared | Object preparation |
|---|---|---:|---:|---:|---:|
| Protons | NumPy | 1189 / 184 | 39.0 µs | 32.2 µs | 46.5 µs |
| Protons | PyTorch CPU | 1189 / 184 | 886.8 µs | 367.6 µs | 275.4 µs |
| VHEE | NumPy | 2897 / 700 | 91.3 µs | 85.1 µs | 47.3 µs |
| VHEE | PyTorch CPU | 2897 / 700 | 1503.1 µs | 545.2 µs | 282.5 µs |

Micro timings include the engine interpolation method, cached lookup, and grid
translation; preparation is measured separately. Global PyTorch JIT is warm here.
These gains do not imply that PyTorch CPU becomes faster than NumPy overall.

Raw timings, including the initial unfavorable VHEE batch, are retained under
[`benchmark/pb_audit_results/interpolation_cpu/`](../../benchmark/pb_audit_results/interpolation_cpu/).
Every successful full-dose comparison had **zero maximum absolute difference**
in every output matrix, including proton LET. Captured interpolation comparisons
also had zero maximum absolute difference.

## Validation and limits

- 508 CPU tests passed in the interpolation, Hong/matRad dose, cutoff, biology,
  and scenario suites; 97 unrelated or GPU/JAX cases were deselected.
- The final focused suite passed 208 tests, including singleton grids and engine reuse.
  It checks NumPy, PyTorch CPU and strict Array API, float32/float64, mixed dtypes,
  biological row shapes, boundaries, NaN/Inf/empty queries, translated grids,
  extrapolating focus interpolation, cache invalidation and both tracing grids.
- Existing photon grid interpolation tests passed. Dose calling APIs and GUI
  model fields are unchanged; GUI source has no consumer of the new 1D utility.
- PyTorch **CT-grid tracing** could not be benchmarked: the frozen baseline fails
  before particle interpolation in `core/xp_utils/_image_interpolation.py:25` at
  `xp.minimum(lower + 1, size - 1)` because PyTorch requires a tensor as its second
  argument. That pre-existing image-resampling issue is outside this change.
- GPU execution and performance remain **unvalidated for this change**. NumPy and
  PyTorch CPU results must not be extrapolated to CuPy or PyTorch CUDA.
- This does not implement energy-lookup preparation, deferred validity checks,
  batching, new spatial candidate selection, or device output collection.

Reproduction (PowerShell; the harness also hides CUDA internally):

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYRADPLAN_XP_PREFER_GPU = 'false'
.venv/Scripts/python.exe benchmark/benchmark_particle_interpolation.py --backend numpy
.venv/Scripts/python.exe benchmark/benchmark_particle_interpolation.py --backend torch
.venv/Scripts/python.exe benchmark/benchmark_particle_interpolation.py --backend numpy --trace-on-ct-grid
.venv/Scripts/python.exe benchmark/benchmark_particle_interpolation.py --backend numpy --mode VHEE --pairs 15 --out benchmark/pb_audit_results/interpolation_cpu/repeat
.venv/Scripts/python.exe benchmark/benchmark_particle_interpolation.py --backend torch --mode VHEE
```

Next: once the GPU is free, measure prepared versus existing CuPy dispatch with
the same workloads and complete output comparisons. Then address per-beamlet
synchronization before broader batching. Keep a separate full `develop` baseline.
