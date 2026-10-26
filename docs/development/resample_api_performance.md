# Consolidated resampling API and full-volume dispatch

Follow-up to `dd06458d`, measured 2026-10-26. This changes the shared API and
execution policy for full-volume array resampling; the prepared, selected-voxel
depth path stays backend-resident for both tracing modes.

## API

```python
from pyRadPlan.core.resample import resample_array, ArrayBackendGridResampler

output = resample_array(
    input_array,
    reference_grid=source_grid,
    target_grid=target_grid,
    interpolator=sitk.sitkLinear,
    extrapolate="nearest",
    prefer_native=True,
)
```

The first eight positional arguments match `resample_numpy_array`; `prefer_native`
is keyword-only. Source geometry can be a reference image or grid. The target can
be an image, grid, or spacing. Array shape is `(z, y, x)` for 3D images. Output
retains input namespace, device and dtype for supported image types.

- `resample_image` retains its existing SimpleITK image interface.
- `resample_numpy_array` is a deprecated forwarding wrapper with a caller-facing
  `DeprecationWarning`; all internal callers now use `resample_array`.
- `ArrayBackendGridResampler` lives in `core.resample` and is still exported from
  `core`. The old private module is an import shim. Its cached mapping is reused
  by dose engines, with no implicit library choice or host-depth conversion.
- Prepared backend evaluation now accepts `sitkLinear` / `sitkNearestNeighbor`,
  nearest boundary extrapolation or constant padding. Nearest-neighbor labels
  do not pass through floating-point value arrays. Other enum requests are not
  silently substituted with linear interpolation.

## Dispatch decision

Default full-volume NumPy and CuPy calls use SimpleITK. For CuPy, the transfer in
both directions is included in the measurements below. This favors the measured
larger workloads and preserves the existing NumPy semantics. Small nearest-neighbor
volumes can favor generic execution; no hardware-specific crossover threshold is
inferred from one laptop.

`prefer_native=False` explicitly selects supported backend execution, including
for NumPy, and rejects unsupported methods/dtypes/dimensions. Linear backend
execution supports float32/float64; nearest-neighbor also supports integer labels
and representable constant padding. With native preference enabled, other
namespaces use supported generic operations and fall back to SimpleITK otherwise.
BSpline and other SimpleITK modes retain their existing native implementations,
including the existing small-image BSpline-to-linear fallback.

This is a dispatch choice, not an assertion that SimpleITK is always faster.
The earlier depth benchmark sampled only selected voxels; the benchmark here
resamples **every target voxel with every implementation**. The prepared depth
path therefore remains appropriate and does not acquire the full-volume API's
host round trip.

## Equivalent full-volume benchmark

Float32 scalar inputs, anisotropic spacing, nonzero origin, identical source and
target geometry for each comparison. Five alternating measured rounds after one
warm-up round; synchronization and output transfers included. Setup and target
index allocation are included in one-shot Array API measurements, excluded only
from the explicitly cached column. All results matched SimpleITK within
`rtol=atol=2e-6`. These timings are workload- and hardware-specific; small GPU
operations show considerable variation.

| Backend | Source / target voxels | Mode | SimpleITK (ms) | Array API one-shot (ms) | Array API cached (ms) |
|---|---:|---|---:|---:|---:|
| numpy | 110,592 / 46,332 | nearest | 3.73 | 2.93 | 2.14 |
| numpy | 110,592 / 46,332 | linear | 3.70 | 7.14 | 6.14 |
| cupy | 110,592 / 46,332 | nearest | 5.73 | 4.47 | 2.98 |
| cupy | 110,592 / 46,332 | linear | 5.04 | 7.11 | 5.70 |
| numpy | 884,736 / 365,508 | nearest | 6.19 | 26.08 | 20.39 |
| numpy | 884,736 / 365,508 | linear | 6.05 | 65.40 | 63.08 |
| cupy | 884,736 / 365,508 | nearest | 10.33 | 8.57 | 6.72 |
| cupy | 884,736 / 365,508 | linear | 10.46 | 14.72 | 11.95 |
| numpy | 7,077,888 / 2,903,472 | nearest | 21.24 | 200.29 | 180.41 |
| numpy | 7,077,888 / 2,903,472 | linear | 25.30 | 614.80 | 600.21 |
| cupy | 7,077,888 / 2,903,472 | nearest | 35.82 | 68.40 | 68.25 |
| cupy | 7,077,888 / 2,903,472 | linear | 34.11 | 80.88 | 75.11 |

Environment: NumPy 2.3.5, CuPy 13.3.0, RTX 4070 Laptop GPU, Windows x86-64,
Python 3.11.9. SimpleITK version is recorded in the raw JSON.

## Validation and reproduction

Tests cover NumPy, CuPy and strict Array API inputs; linear/nearest/BSpline/Gaussian
enums; constant and nearest extrapolation; integer and floating dtypes;
namespace/device retention; native fallback; explicit backend rejection of
unsupported requests; the old positional API and warning; and geometry alternatives.
The main regression run passed 337 cases, followed by 15 additional nearest-neighbor
half-voxel checks. This includes existing depth, structure, dose/biological and
objective-caller regressions. All 560 GUI tests also passed.
The prepared resampler's existing NaN, oblique-grid and singleton-axis checks remain.

```powershell
.venv/Scripts/python.exe benchmark/profile_resample_dispatch.py
.venv/Scripts/python.exe -m pytest test/core/test_resample_array_api.py test/core/test_array_resample.py test/core/test_resample.py -q
```

[Raw timing samples](../../benchmark/pb_audit_results/resample_dispatch/full_volume.json)
and the [reproducible harness](../../benchmark/profile_resample_dispatch.py) are committed.
