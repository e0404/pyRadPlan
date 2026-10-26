# Tissue preparation and backend-resident radiological depths

Measured 2026-10-26 on `refactor/dose-calc-performance`, relative to `164473bb`.
This follow-up changes shared patient tissue preparation and radiological-depth
consumption. CT/density construction and other SimpleITK geometry operations
remain as before.

## Implementation

`StructureSet.get_tissue_data()` resolves overlap priorities once, fills a VOI
label volume, resamples that volume once using nearest neighbors, and looks up
alpha, beta and tissue class. Label zero has zero alpha/beta and class index zero,
preserving the existing outside-structure convention. Equal tissue properties
still share a class; equal overlap priorities retain the original assignment
order. Existing separate getters and all public `Dij` fields remain available.
There is no persistent patient cache or new invalidation policy.

`RayTracerBase.trace_cubes_array()` returns `(z, y, x)` arrays in tracing precision
on the tracing backend. `trace_cubes()` remains an image-returning wrapper.
The dose engine consumes the arrays directly:

- `trace_on_dose_grid=True`: gather selected voxels directly from each depth array.
- `trace_on_dose_grid=False`: resample directly to the selected dose-grid voxels,
  on the same namespace/device. No full dose-grid depth volume is needed normally.
- `keep_rad_depth_cubes=True`: additionally prepare and download the full dose-grid
  image for the explicitly requested retained output, preserving image metadata.

The shared `ArrayBackendGridResampler` lives in `core`, with numerical evaluation
in `core.xp_utils._image_interpolation`. The engine contains no library-specific
interpolation fast path. Construction retains target indices and prepares small
scanline endpoint arrays; evaluation uses bounded chunks (262,144 queries by
default). The prepared mapping is reused across beams within one calculation
and reset for the next calculation. Only small grid metadata is handled in NumPy.

This is deliberately an image-resampling primitive, separate from general
rectilinear interpolation. It preserves nearest-pixel extrapolation outside the
half-voxel image extent and skips zero-weight NaN neighbors. Generic eight-corner
weighted sums can turn an exact finite voxel sample into NaN through `0 * NaN`.
Likewise, collapsing the coordinate transform algebraically can change tiny
boundary weights and therefore the NaN footprint. The implementation follows
[ITK's scanline coordinate calculation](https://github.com/InsightSoftwareConsortium/ITK/blob/master/Modules/Filtering/ImageGrid/include/itkResampleImageFilter.hxx)
and [linear interpolation behavior](https://github.com/InsightSoftwareConsortium/ITK/blob/master/Modules/Core/ImageFunction/include/itkLinearInterpolateImageFunction.h).
Coordinate and interpolation arithmetic uses float64, with output cast to the
input image dtype. Singleton axes, anisotropic/oblique geometry and arbitrary
selected-index order are supported. This API currently covers 3D floating scalar
images and linear interpolation with nearest extrapolation, not arbitrary ITK
transforms or all image interpolators.

## Measurements

TG119, one beam, Generic machines, 10 mm bixels and 5 mm dose grid; physical-dose
influence calculations without stochastic sampling. NumPy 2.3.5, CuPy 13.3.0,
Python 3.11.9, RTX 4070 Laptop GPU, CUDA runtime 12.6; Windows Intel family 6 model
183, 32 logical CPUs. Runs are sequential, with GPU synchronization included.
Imports, phantom loading and steering generation are excluded from dose timings.

### Tissue preparation

Five alternating measured rounds after warm-up. This is the same CPU setup for
both dose backends, including overlap handling, resampling and array assembly.

**150.04 ms -> 65.89 ms (2.28x)**. All tissue arrays matched exactly.

### Complete dose calculation

Three alternating current/legacy pairs per case. Times are medians; speedups are
medians of paired legacy/current ratios, not ratios of the displayed medians.
The legacy variant restores separate tissue getters and the depth image round
trip with SimpleITK resampling, leaving dose physics and ray tracing unchanged.

| Engine | Backend | Tracing grid | Before (s) | After (s) | Paired speedup |
|---|---|---|---:|---:|---:|
| VHEE | cupy | Dose | 0.819 | 0.771 | 1.06x |
| VHEE | cupy | CT | 0.936 | 0.818 | 1.14x |
| VHEE | numpy | Dose | 0.732 | 0.618 | 1.18x |
| VHEE | numpy | CT | 0.947 | 0.797 | 1.14x |
| photons | cupy | Dose | 1.401 | 1.268 | 1.11x |
| photons | cupy | CT | 1.455 | 1.328 | 1.08x |
| photons | numpy | Dose | 0.921 | 0.824 | 1.15x |
| photons | numpy | CT | 1.148 | 1.060 | 1.09x |
| protons | cupy | Dose | 1.669 | 1.528 | 1.10x |
| protons | cupy | CT | 1.907 | 1.713 | 1.13x |
| protons | numpy | Dose | 0.650 | 0.502 | 1.29x |
| protons | numpy | CT | 0.797 | 0.655 | 1.21x |

All **36 pairs had zero physical-dose difference and identical sparse matrix
indices/column pointers**, including CT-grid tracing. These are modest, noisy
laptop benchmarks; the timing ratios are workload-specific. GPU speedups include
the CPU tissue savings and must not be attributed entirely to device residency.
Exactness here is observed on these cases; general interpolation tests use tight
floating-point tolerances and separately check NaN locations.

### Isolated depth consumption

Captured radiological arrays from a real proton calculation, five alternating
measured rounds after warm-up, with synchronized transfers included. Tracing and
tissue setup are excluded. Cached evaluation reuses the prepared mapping; the
last column measures one additional call after discarding it, so includes map
creation but not cold library/kernel compilation.

| Backend | Tracing grid | Before (ms) | Cached after (ms) | After with map creation (ms) |
|---|---|---:|---:|---:|
| cupy | CT | 14.229 | 4.289 | 5.698 |
| cupy | Dose | 1.150 | 0.051 | 0.058 |
| numpy | CT | 14.192 | 9.622 | 12.240 |
| numpy | Dose | 0.505 | 0.140 | 0.136 |

The CuPy CT-grid depth step improves from about 14.2 ms to 4.3 ms (5.7 ms
including mapping preparation). On the dose grid, removing the round trip saves
about 1.1 ms per beam in this case. Tissue preparation supplies most of the
end-to-end improvement for this single-beam workload; the depth change removes a
per-beam transfer boundary and also avoids full-volume work for CT-grid tracing.

## Validation and reproduction

The numerical regression run passed 223 tests covering existing photon/ion and
biological fixtures, separate-versus-combined tissue outputs, both depth tracing
modes, retained image metadata, strict Array API compatibility, NumPy/CuPy,
oblique grids, singleton axes, boundary extrapolation and NaN footprints. Tests
also reject unexpected host-depth conversions when retained images are disabled.
Existing GUI usages of the unchanged entry points were checked; all 560 GUI tests passed.

```powershell
.venv/Scripts/python.exe benchmark/profile_patient_depth_preparation.py
.venv/Scripts/python.exe benchmark/profile_depth_sampling.py --backend cupy --trace-on-ct-grid
.venv/Scripts/python.exe -m pytest test/core/test_array_resample.py test/cst/test_cst.py test/test_raytracersiddon.py test/dose/test_depth_preparation.py test/dose/test_svdpb.py test/dose/test_hongpb.py test/dose/test_svdpb_bio.py test/dose/test_hongpb_bio.py test/dose/test_dose_engine_base.py test/dose/test_voxel_preparation.py -q
```

[Raw measurements](../../benchmark/pb_audit_results/patient_depth/) include timing
samples, environment metadata and numerical comparisons. In paired files,
`baseline` is current production and `variant` is the restored legacy path.
The audit also accepts `legacy_tissue` and `legacy_depth` to isolate the changes.
