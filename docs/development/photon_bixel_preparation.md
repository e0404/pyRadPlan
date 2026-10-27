# Photon bixel constants and component evaluation

Follow-up to [beam-level ray transforms](ray_transform_preparation.md), comparing
against `9d140807` on `refactor/dose-calc-performance`.

## Implementation

Each photon beam now owns a small cache of depth-kernel parameters: reshaped
`betas`, the attenuation parameter `m`, and `betas / (betas - m)`. Rays share this
cache, including their shallow scenario copies. Preparation occurs on first use
for the actual depth dtype/device, because scenario range adjustments can change
precision. The cache is discarded with the beam; changing beams/kernels or starting
a new calculation cannot reuse old parameters. Photon beams already reject mixed
kernel energies. Kernel data is fixed during beam evaluation.

Cache lookup groups by dtype and compares device metadata explicitly: Array API
device objects are not necessarily hashable, including CuPy's. No numerical host
copy or device synchronization is needed for a cache hit. For C components the
persistent arrays contain only 2*C+1 scalar values per dtype/device combination.

Component multiplication uses one broadcast operation instead of a Python loop
of indexed updates. Its result is explicitly cast back to the depth-component
dtype **before** the unchanged axis-zero sum:

```python
dose_component = xp.astype(
    dose_component * xp.permute_dims(interpolated_kernels, (1, 0)),
    dose_component.dtype,
    copy=False,
)
bixel_dose = xp.sum(dose_component, axis=0)
```

That cast preserves the previous assignment semantics when interpolation returns
higher precision than the depth arrays. It is not a dose-precision reduction.
The expression retains the existing exponentials, multiplication ordering and
component summation axis.

Geometry already supplies interpolation queries in the correct namespace/device.
The per-bixel `to_namespace` call is removed; it normally returned the same object,
so this removes dispatch/metadata checks rather than a host/device array copy.
The shared `ArrayBackendGridInterpolator` still performs evaluation and its own
input validation/dispatch. No interpolation fast path is added to the engine.
Sampling, RNG consumption, cutoff geometry and sparse output collection are unchanged.

## Validation and measurement method

The frozen bixel function in `benchmark/profile_photon_bixel_reference.py` is
installed only in benchmark processes. Paired runs alternate current code
(`baseline`) and the old bixel evaluation (`variant`). Both execute the same
current beam setup; the new empty cache dictionary is allocated in both cases,
while parameter creation on first bixel evaluation is included in candidate
timing. Full sparse values, indices and column pointers are compared.

Focused tests cover NumPy, CuPy, strict Array API and Torch; float32/float64 depths
and interpolation values; one, three and five components; signed component values;
empty rays; dtype/device retention; sharing between rays; and isolation between
beam kernels and depth precisions. A scalar-component reference explicitly applies
the former per-component assignment cast before summation.

Regression validation passed **242 dose/geometry tests**, including the 56 new
bixel arithmetic/cache cases, and **580 Dij/GUI tests**. Ruff checks and formatting
passed. No GUI code uses the changed private bixel methods or cache field; public
models and dose-engine calling interfaces are unchanged.

## Full-dose results

Sampled five-field TG119, 3 mm dose grid, 5 mm bixels, tracing on the dose grid
(1,567 rays/beamlets), with all preparation and SciPy output included:

| Backend / run | Previous | Current | Less elapsed time | Alternating pairs |
| --- | ---: | ---: | ---: | ---: |
| NumPy | 21.142 s | 20.770 s | 1.8% | 2 |
| CuPy, initial | 8.334 s | 7.809 s | 6.3% | 3 |
| CuPy, repeat | 8.608 s | 7.992 s | 7.2% | 3 |

The CPU difference is small relative to timing variation; this does not establish
a substantial NumPy speedup. The first GPU run included one slower candidate
pair (8.797 vs 8.334 s); all its results are retained in `cupy/`. The independent
repeat in `cupy_repeat/` has a similar median improvement. These are
workload-specific measurements, not a guarantee of improvement for every plan.

Hardware/software: Intel family 6 model 183 CPU, NVIDIA RTX 4070 Laptop GPU,
Python 3.11.9, NumPy 2.3.5 and CuPy 13.3.0 / CUDA 12.6, as in the preceding audit.

All **16 full sparse comparisons** have zero dose difference and identical row
indices/column pointers: eight sampled GUI-workload pairs and eight unsampled
pairs covering bixel-based and field-shaped dose on both backends. The smaller
unsampled cases use a 5 mm dose grid, 10 mm bixels, one beam and CT-grid tracing.
Both tracing modes are therefore covered. Field cases exercise per-ray custom
kernel interpolation; comparisons are within each backend.

## Isolated arithmetic and its limits

`profile_photon_bixel_arithmetic.py` precomputes interpolation outputs through the
real shared interpolator, preserving its output precision/layout, then measures
bixel evaluation without interpolation or sampling. Constants are warm in these
measurements; full-dose comparisons above include their creation.

For 50,000 voxels and float64 depths, median arithmetic time is approximately
**0.500 -> 0.180 ms on CuPy**, versus **1.005 -> 0.990 ms on NumPy**. Across the
six measured size/precision cases (5k, 50k and 200k voxels; float32/float64 depths),
GPU time decreases by 27-64%. This is only the changed arithmetic, not full dose.

CPU behavior is mixed: small arrays improve, but float32 depths with float64
interpolation at 50k/200k voxels take about 2%/6% longer. The broadcast expression
materializes a whole promoted component product before casting; the old loop
used smaller per-component temporaries. The complete measured NumPy workload
does not regress, but the microbenchmark identifies a limitation for other
precision/size combinations. No engine-specific CPU dispatch is introduced for
these small, synthetic differences. All arithmetic comparison outputs are exact.

## Reproduce

```powershell
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend cupy --mode photons --grid 3 --bixel 5 --beams 5 --sampling --runs 0 --variant legacy_photon_bixel --paired --pairs 3 --no-matrices --out benchmark/pb_audit_results/photon_bixel/recheck
.venv/Scripts/python.exe benchmark/profile_photon_bixel_arithmetic.py --backend cupy --out benchmark/pb_audit_results/photon_bixel/arithmetic_recheck.json
.venv/Scripts/python.exe -m pytest test/dose -q -k "svdpb or ray_geometry_preparation or sparse_output or dose_engine_base or hongpb"
```

Repeat with `--backend numpy`. Omit `--sampling` to check unsampled influence
matrices; `--mode field` exercises per-ray field-shaped interpolation. Raw timing
and complete-output comparison summaries live in
`benchmark/pb_audit_results/photon_bixel/`.
