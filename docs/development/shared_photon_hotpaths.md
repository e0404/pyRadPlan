# Shared photon sampling and interpolation reductions

Follow-up to [the GUI performance recheck](gui_photon_performance_recheck.md).
Measurements use TG119, the Generic photon SVD engine, five beams at 0/72/144/216/288
degrees, 3 mm dose grid, 5 mm bixels, sampling enabled, seed 1234, and tracing on
the dose grid unless stated otherwise. Timings exclude STF generation and Qt.

## Implementation

`core.xp_utils.grouped_max` reduces per-depth-cluster tail dose. NumPy and CuPy
use indexed maximum, replacing two stable sorts per ray. Other backends retain
a sorting fallback, but gather its output with Array API `take`/`searchsorted`
without the previous scatter helper's possible host-array copy. Input validation,
empty groups, initial values and NaN propagation are shared contracts. CuPy needs
a separate NaN reduction because CUDA atomic maximum ignores NaNs.

```python
cluster_dose = grouped_max(dose_tail, cluster, num_clusters, initial=0.0)
threshold_dose = xp.take(cluster_dose, cluster)
```

The sampling policy, random generator, number of draws, core voxels and canonical
row ordering are unchanged.

`ArrayBackendGridInterpolator` now accepts trailing value dimensions, like
SciPy's rectilinear interpolator. Coordinates are prepared once for all fields.
Both native dispatch and the generic 2D/3D kernels support this interface; scalar
interpolation keeps its existing dispatch. Multiple trailing dimensions are
flattened internally for native evaluation to work around CuPy's NaN-mask
broadcasting limitation, then restored on return.

```python
interpolator = ArrayBackendGridInterpolator(
    (grid, grid), xp.stack(convolved_kernels, axis=-1), bounds_error=True
)
# Shape: (number_of_voxels, number_of_kernel_components).
components = interpolator(iso_lat_dists)
```

The photon engine owns the physical convolution and component weighting. The
interpolation backend owns the searches, weights and library selection. The
same multi-value facility can serve future engines with fields on a shared grid;
it does not couple interpolation to photon physics. Scalar fields on different
grids continue to use separate objects. No global cache or implicit host/device
copies are introduced. Per-ray custom fluence uses the same shared-grid path.

## Measurement method

`profile_photon_hotpaths_reference.py` freezes the three changed engine methods
from `0c298044`. `profile_pb_audit.py` can install the old sampling only, the old
kernel queries only, or both. In paired JSON, **baseline is the new implementation**
and **variant is the selected reference/experiment**. Alternating order reduces
warm-up and ordering bias. Every pair compares the entire sparse influence matrix,
including indices and column pointers; it is not a sampled output check.

`profile_shared_grid_interpolation.py` separately compares three cached scalar
interpolators, a cached multi-value native interpolator and the generic multi-value
implementation. It covers float32/float64, 10k/50k/150k random queries on an 800x800
non-extrapolating grid with three fields. GPU timings synchronize the current
stream. These isolated random-query timings guide experiments; full-dose timings
determine the engine integration.


## Backend correctness checks

Focused tests cover NumPy, CuPy, strict Array API, Torch and JAX where installed.
They exercise unsorted and missing groups, negative values/depths, NaNs, empty
inputs, mixed trailing value shapes, descending/nonuniform grids, clipping,
query validation and device preservation. Sampling tests independently construct
per-group maxima and verify the retained rows, doses and RNG state, for both
radius/dose policies and boolean/integer voxel indices. Photon convolution tests
check repeated calls and isolation between different fluence maps.


The generic 2D multi-value path packs each field contiguously at construction.
Queries gather `(fields, queries)` arrays and share weights along the query axis;
results are restored to `(queries, *value_shape)`. Packing is paid once per
interpolator, not per ray. Native preference remains explicitly selectable:
`prefer_native=True` requests the library, `False` requests generic, and the
new default `None` selects generic for NumPy multi-value 2D grids and native
where available otherwise. Scalar, singleton-axis and higher-dimensional
native support is preserved.


## Experiments that informed the final layout

Simply stacking fields and calling native SciPy RGI was slower than the generic
combined evaluator in the full workload (29.66 vs 26.03 s in alternating pairs).
However, generic evaluation in `(queries, fields)` layout was essentially tied
with three scalar SciPy calls after the sampling fix (25.22 vs 25.02 s). The
random-query microbenchmark alone overstated its benefit for real photon rays.
The final generic path therefore packs fields along the outer axis once and
broadcasts query weights along contiguous rows.

An initial packing attempt using `asarray(transposed, copy=True)` preserved
Fortran strides in NumPy, causing `take` to copy the full field array repeatedly.
Flattening in logical row-major order before reshaping fixes this with Array API
operations. The rejected layout's raw measurements are retained as
`interp_numpy_packed.json`; `interp_numpy_contiguous.json` describes the corrected
layout. The earlier unpacked measurements are `interp_numpy.json` and the
`numpy_query_reuse/` paired results. None of the rejected layouts remains in
production dispatch.


## Timings

Final packed interpolation alone, keeping grouped maximum enabled on both sides:

| NumPy full GUI workload | Median of two alternating pairs |
| --- | ---: |
| Three cached scalar SciPy interpolators | 23.742 s |
| Shared generic interpolation, contiguous fields | 21.026 s |

This isolates **11.4% less elapsed time** (1.13x). Both full comparisons were
bit-identical in dose values, row indices and column pointers. The earlier native
multi-value and unpacked generic experiments also produced identical full-dose
outputs, but did not deliver the same speed improvement.


Both changes together, final direct comparisons against the frozen original code:

| Backend | Before | After | Less elapsed time | Speedup |
| --- | ---: | ---: | ---: | ---: |
| NumPy (CPU) | 35.027 s | 20.416 s | 41.7% | 1.72x |
| CuPy (GPU) | 15.082 s | 9.656 s | 36.0% | 1.56x |

These are medians of two alternating warm pairs, including interpolator
construction, packing, geometry preparation and sparse output assembly.
Measurements were sequential on Python 3.11.9 / NumPy 2.3.5, an Intel family 6
model 183 CPU (32 logical processors), and CuPy 13.3.0 / CUDA 12.6 on an NVIDIA
GeForce RTX 4070 Laptop GPU. They are workload-specific measurements, not a
universal speedup guarantee. Absolute timings varied during the session; the
final NumPy original-code median again reproduced the earlier approximately
35-second GUI measurement. These changes do not explain the historical 16-second
observation.

All four final full-workload comparisons had **zero dose difference and identical
sparse indices/column pointers**. Comparisons are within each backend: existing
NumPy/CuPy numerical and sampling differences are not claimed to disappear.
The CPU matrix contains 64,572,864 stored entries, the GPU matrix 64,576,412;
those counts are unchanged by these changes.

A supplementary one-beam workload (5 mm dose grid, 10 mm bixels, sampling on,
tracing on the CT grid) also produced identical outputs on both backends.
Its single warm pair measured NumPy 1.059 -> 0.921 s and CuPy 1.291 -> 0.912 s;
use these as mode-coverage checks rather than high-confidence speed estimates.

The final CuPy microbenchmark still favors native shared interpolation over the
packed generic implementation: for 50k float64 queries, three scalar native
calls take 4.156 ms, one native multi-value call 1.374 ms, and generic shared
interpolation 1.923 ms. Hence the engine uses generic packed evaluation on NumPy
and native multi-value evaluation on CuPy, selected inside the shared backend.

## Reproduce

```powershell
# Final full GUI workload; repeat with --backend cupy.
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend numpy --mode photons --grid 3 --bixel 5 --beams 5 --sampling --runs 0 --variant legacy_photon_hotpaths --paired --pairs 2 --no-matrices --out benchmark/pb_audit_results/shared_hotpaths/recheck

# Isolate query sharing with grouped maximum enabled on both sides.
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend numpy --mode photons --grid 3 --bixel 5 --beams 5 --sampling --runs 0 --variant legacy_kernel_queries --paired --pairs 2 --no-matrices --out benchmark/pb_audit_results/shared_hotpaths/query_recheck

# Compare cached interpolation paths; repeat with --backend cupy.
.venv/Scripts/python.exe benchmark/profile_shared_grid_interpolation.py --backend numpy --out benchmark/pb_audit_results/shared_hotpaths/interpolation_recheck.json
```

`--no-matrices` skips large NPZ files without skipping full sparse comparisons.
Raw timing and comparison JSON lives under `benchmark/pb_audit_results/shared_hotpaths/`.
`numpy_final/` and `cupy_final/` are the final combined comparisons;
`numpy_contiguous/` isolates the final interpolation improvement.
`numpy/` and `numpy_generic/` record the preceding native/unpacked experiments;
for those files, baseline means the production candidate **at measurement time**,
not the final implementation. The frozen reference includes only the changed
engine methods; their scalar interpolation uses the current class's unchanged
scalar dispatch.


## Regression validation

**1,313 passed, 6 skipped** (237.90 s), including all GUI tests and the existing
photon/particle fixture comparisons. Ruff and whitespace checks passed.

```powershell
.venv/Scripts/python.exe -m pytest test/core/xp_utils test/core/test_resample.py test/core/test_resample_array_api.py test/core/test_array_resample.py test/dose/test_svdpb.py test/dose/test_svdpb_interpolators.py test/dose/test_svdpb_sampling.py test/dose/test_svdpb_bio.py test/dose/test_hongpb.py test/dose/test_particle_cutoff.py test/dose/test_ray_geometry_preparation.py test/gui -q --disable-warnings
```

The public dose-engine calling interface and GUI model fields are unchanged.
The engine's private `kernel_interpolators` state now holds one multi-value
interpolator instead of a list of scalar interpolators; the custom-fluence and
field-based paths use the same factory and are covered by regression tests.

Follow-up: [compact sparse indices and device-collection measurements](sparse_output_performance.md).
