# Beam-level ray transform preparation

Follow-up to [sparse output measurements](sparse_output_performance.md), comparing
against `84ebc92b` on `refactor/dose-calc-performance`.

## Implementation

The common pencil-beam engine prepares rotations for all rays together when it
first prepares a beam's valid coordinates. This happens after modality-specific
beam initialization and after missing target points have been generated. The
cache belongs to that beam/shift; it is not shared across calculations.

Preparation normalizes the common source direction once, normalizes target
directions in a batch, and evaluates the same Rodrigues formula with a leading
ray dimension. Source and rotation arrays use the voxel-coordinate dtype and
remain on its backend/device. One vector of alignment flags is copied to the
host per beam, replacing a host-visible alignment decision for every ray.
Aligned rays retain the original componentwise `1e-7` tolerance and exact identity
path. A safe denominator prevents evaluating division by zero for these rays.
The existing voxel-distance calculation, cutoff rule and ordered selection are
unchanged, including the existing fused GPU distance kernel.

This changes preparation shared by photons, particles and VHEE. Geometry was
already shared by energy layers on a particle ray. Each ray still scans the
beam-valid coordinates; this change does not introduce voxel/ray batching or a
spatial candidate index. It also does not change output collection, photon
sampling, interpolation dispatch or the public `calc_geo_dists` mask interface.
Standalone geometry calls prepare one transform through the same helper.

The per-ray matrix casts are removed because prepared matrices already have the
coordinate dtype. Cache storage is one 3x3 matrix and one host Boolean per ray,
plus a common source vector: approximately 58 kB for 1,567 float32 ray matrices
and flags across the five beams (excluding transient target/direction arrays).
Only one beam's cache is needed at a time.

## Validation and measurements

The frozen reference in `benchmark/profile_ray_transform_reference.py` contains
the previous preparation, ray extraction and geometry methods. The full-dose
harness alternates current code (`baseline`) and the reference (`variant`),
including all setup costs. Sampling uses the same seed. Full sparse dose values,
row indices and column pointers are compared within each backend.

Geometry tests cover NumPy, CuPy, strict Array API and Torch, including empty
supports, multiple CT-scenario validity masks, empty ray batches, general source
positions, aligned and off-axis rays, both sides of the alignment tolerance,
float32/float64 preparation and preservation of input arrays. They also check
that a batch transfers one alignment vector and evaluation does not rebuild it.

Regression runs passed **186 dose/geometry tests** and **580 Dij/GUI tests**.
The dose run includes the existing photon/particle fixtures and sparse-output
regressions. Ruff checks and formatting passed. No GUI code uses the changed
private geometry methods; public model fields and dose-engine calls are unchanged.

## Full-dose results

Five-field TG119, 3 mm dose grid, 5 mm bixels, sampling enabled, tracing on the
dose grid (1,567 rays/beamlets):

| Backend | Previous | Batched transforms | Less elapsed time | Alternating pairs |
| --- | ---: | ---: | ---: | ---: |
| NumPy | 22.375 s | 21.696 s | 3.0% | 2 |
| CuPy | 9.873 s | 8.517 s | 13.7% | 3 |

All timings include preparation and final SciPy output. The NumPy improvement is
small relative to run-to-run variation: individual pairs improved by about 1.7%
and 4.5%. The three final GPU pairs consistently favor batching. These are
workload-specific measurements, not general CPU/GPU speed guarantees.

All 13 full sparse comparisons are exact, including indices and column pointers.
This includes six supplementary CT-grid-tracing cases: photons, protons (with
LET), and VHEE, each on NumPy and CuPy at 5 mm dose resolution. Their single
pairs primarily verify modality and tracing coverage. Comparisons are within
backends; pre-existing NumPy/CuPy differences are not changed.

`cupy_final/` contains the final GPU measurements. The initial `cupy/` experiment
also produced exact outputs and about 12% less elapsed time, but retained two
redundant per-ray matrix copies removed before the final measurement. The NumPy
measurements already use the final implementation. Hardware is the same Intel
family 6 model 183 CPU / NVIDIA RTX 4070 Laptop GPU used in the preceding audit,
with NumPy 2.3.5 and CuPy 13.3.0 / CUDA 12.6 on Python 3.11.9.

## Preparation cost and stage attribution

The synchronized GPU profile measures five transform-preparation calls totaling
**5.64 ms** for all 1,567 rays. This includes rotation computation and the five
alignment-vector downloads. Target stacking and coordinate compaction sit outside
that helper; complete beam-geometry preparation totals **13.02 ms** across the
five beams. Per-ray geometry/selection including that preparation totals **0.765 s**,
compared with approximately 2.02 s in the preceding audit's profile of the same
workload. These stage profiles are diagnostic, not paired timing estimates.
The unchanged voxel-distance core now accounts for about 0.586 s.

A smaller NumPy profile (121 rays, one beam, 5 mm dose grid) measures **0.278 ms**
inside transform preparation and **0.768 ms** for complete beam-geometry
preparation. Its workload differs from the GPU profile, so these values should
not be used for a CPU/GPU speed comparison.

## Reproduce

```powershell
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend cupy --mode photons --grid 3 --bixel 5 --beams 5 --sampling --runs 0 --variant legacy_ray_transforms --paired --pairs 2 --no-matrices --out benchmark/pb_audit_results/ray_transforms/recheck
.venv/Scripts/python.exe -m pytest test/dose -q -k "ray_geometry_preparation or sparse_output or svdpb or hongpb or dose_engine_base"
```

Repeat the dose benchmark with `--backend numpy`. The smaller coverage cases
use a 5 mm dose grid, 10 mm bixels and one beam; `--trace-on-ct-grid` exercises
CT-grid tracing, and `--mode protons --let` additionally checks LET output.
Raw results are under `benchmark/pb_audit_results/ray_transforms/`.

Follow-up: [photon bixel constants and component evaluation](photon_bixel_preparation.md).
