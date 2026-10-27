# Shared pencil-beam ray geometry preparation

Measured 2026-10-27 on `refactor/dose-calc-performance`, against `4c5bef97`.
The comparison includes the earlier interpolation, particle cutoff, voxel,
tissue and depth-resampling changes on **both** sides.

These one-beam timings are not representative of GUI-default sampled five-field
IMRT. See the [GUI photon recheck](gui_photon_performance_recheck.md) for that
workload, historical comparisons and the remaining photon bottlenecks.

## Implementation

The shared engine now prepares an ordered list of beam-valid voxel positions
and their compact BEV coordinates once. Preparation is lazy on the first ray,
after the complete modality-specific beam initializer has run. This matters for
particles, whose initializer further limits validity by maximum particle range.
The cache belongs to the beam dictionary and is discarded with that beam/shift.

Each ray returns local integer selections into these compact coordinates.
Selections map back to the engine-selected voxel positions, then to flattened
dose-grid rows. All gathers use Array API `take`. For multiple CT scenarios,
scenario membership is gathered within the selected ray support; the nominal
single-scenario path needs no additional mask or compaction. The engine no
longer builds the unused per-ray `valid_coords`/`valid_coords_all` full masks.

Conceptually:

```python
# Once per beam, after modality-specific validity limits:
valid_positions = xp.nonzero(beam_valid)[0]
coords = xp.take(beam_coords, valid_positions, axis=0)

# Per ray, using the existing cone and rotation calculations:
local = xp.nonzero(cone_mask)[0]
positions = xp.take(valid_positions, local, axis=0)
rows = xp.take(selected_dose_grid_rows, positions, axis=0)
depths = xp.take(beam_depths, positions, axis=0)
```

The existing CuPy and Torch kernels and numerical operation order are retained.
The CuPy path no longer stacks all three rotated-coordinate outputs just to
read the longitudinal component. CPU calls check the namespace before probing
GPU libraries. Bixel counts remain host metadata; SSD values and missing-ray
flags are downloaded once per vector, instead of scalar-by-scalar per ray.
SSD distance reshaping also uses the namespace API rather than an array
`.squeeze()` method, which strict Array API arrays do not provide. The existing
missing-SSD fallback is unchanged.

`calc_geo_dists` retains its existing signature and boolean-mask result for
external callers. It delegates numerical work to
`_calc_geo_dists_on_valid_coords` and reconstructs the full mask using sorted
indices and a sentinel, without backend-specific scatter assignment. The dose
loop calls the compact helper directly. Subclasses needing custom ray geometry
can override `_get_ray_geometry_from_beam` or the compact numerical helper;
modality cutoff hooks and downstream ray dose arrays keep their contracts.
No GUI usages of these internal geometry hooks or masks were found.

## Measurements

TG119, one beam, 10 mm bixel spacing, Generic machines (VHEE: 200 MeV).
The isolated benchmark uses actual prepared patient/beam data: 121 rays for
photons/VHEE and 72 rays / 908 beamlets for protons. It compares all ray geometry
outputs, including row indices, **exactly**, then times five alternating rounds
with device synchronization at the workload boundaries. Every current round
includes creating the beam cache. Downloads for validation are outside timing.

All 36 geometry workloads (three modalities, NumPy/CuPy, 3/5/8 mm grids,
both tracing modes) matched exactly. Median paired geometry speedups were
**1.70-3.10x on NumPy** and **1.95-2.60x on CuPy**. These are geometry-stage
speedups, not total dose speedups.

At 5 mm, median geometry time across all rays, in milliseconds:

| Modality | Backend | Tracing grid | Previous | Current |
|---|---|---|---:|---:|
| photons | numpy | Dose | 174.0 | 88.1 |
| photons | numpy | CT | 160.7 | 84.6 |
| photons | cupy | Dose | 365.1 | 167.1 |
| photons | cupy | CT | 338.5 | 170.0 |
| protons | numpy | Dose | 51.5 | 21.3 |
| protons | numpy | CT | 49.7 | 20.2 |
| protons | cupy | Dose | 215.1 | 101.9 |
| protons | cupy | CT | 200.8 | 99.5 |
| VHEE | numpy | Dose | 294.1 | 153.1 |
| VHEE | numpy | CT | 275.8 | 184.1 |
| VHEE | cupy | Dose | 338.8 | 161.2 |
| VHEE | cupy | CT | 332.5 | 167.1 |

The separate one-time preparation measurements ranged from 0.034 to 0.804 ms.
The cache holds float32 coordinates plus integer positions (20 bytes per valid
voxel with int64 indices), plus membership arrays for multiple CT scenarios.
Measured cache storage was at most 4.58 MB per beam. This deliberately trades a
small persistent beam allocation for repeated gathers and temporary masks; it
does not remove the original full beam arrays yet.

A fresh instrumented pre-change CuPy proton profile at 5 mm attributed 257 ms
to ray geometry (129 ms in `calc_geo_dists`, 128 ms in surrounding gathers/masks).
Instrumented timings are diagnostic and are not used for the speedup ratios.

## Full-dose validation and timings

The repeat campaign uses six alternating pairs for each of 12 workloads at
5 mm. All **72 pairs** have identical physical-dose values, CSC indices and
column pointers. The initial three-pair campaign also passed all 36 comparisons.
The table reports medians of complete influence-matrix calculation wall times
(seconds), including patient/beam setup, dose evaluation and sparse output:

| Modality | Backend | Tracing grid | Previous | Current | Speedup |
|---|---|---|---:|---:|---:|
| photons | cupy | CT | 1.473 | 1.324 | 1.11x |
| photons | cupy | Dose | 1.429 | 1.298 | 1.10x |
| photons | numpy | CT | 1.180 | 1.081 | 1.09x |
| photons | numpy | Dose | 0.986 | 0.890 | 1.11x |
| protons | cupy | CT | 1.709 | 1.562 | 1.09x |
| protons | cupy | Dose | 1.664 | 1.516 | 1.10x |
| protons | numpy | CT | 0.648 | 0.604 | 1.07x |
| protons | numpy | Dose | 0.568 | 0.507 | 1.12x |
| VHEE | cupy | CT | 0.829 | 0.677 | 1.23x |
| VHEE | cupy | Dose | 0.805 | 0.637 | 1.26x |
| VHEE | numpy | CT | 0.852 | 0.690 | 1.23x |
| VHEE | numpy | Dose | 0.631 | 0.506 | 1.25x |

Total speedups in this repeat were 1.07-1.26x. The initial short campaign had
large excursions (including an apparent photon/CuPy/CT-tracing regression), so
it was retained and followed by the longer repeat for **all** workloads. The
repeat runs full-dose cases in fresh processes without retaining the isolated
benchmark's GPU allocations. Both campaigns and individual samples are saved.
There is still timing noise; geometry-stage reductions are more reliable than
small differences in total dose runtime.

Validation: **383 focused tests passed**, covering the geometry API, prepared
ray selection, SSD metadata, depth/voxel preparation, particle indexing/cutoffs,
existing Hong/SVD dose fixtures and numerical kernels. The new tests exercise
NumPy, CuPy, strict Array API and Torch, including disjoint scenario supports,
empty supports and zero-length coordinate arrays, exact cone boundaries,
rotated rays, cache lifetime and ordered dose-row mapping. **560 GUI tests
passed**. Ruff checks and formatting passed.


## Scope and next opportunities

This preserves the existing physics, cone boundary decisions, selected row
ordering and modality-specific kernels. Synthetic multi-scenario selection is
tested, but this does not add full multi-CT support to the dose engine.
Tests use exactly representable boundary coordinates: the existing CPU and
fused GPU formulas can round a marginal cone boundary differently, and this
change intentionally does not redefine that behavior.

The algorithm still scans every beam-valid coordinate for every ray. It still
has one dynamic compaction per ray and the alignment scalar decision can still
synchronize the GPU. Subsequent architectural work can restrict candidate
voxels spatially or process bounded batches of rays. The explicit local-to-grid
mapping provides a common foundation for photons, particles, VHEE and future
fine sampling, while the dose formulas remain separate engine implementations.
Requested geometry fields and ownership of the full beam arrays can also be
streamlined later. No additional SimpleITK migration is part of this change.

## Reproduction

```powershell
.venv/Scripts/python.exe benchmark/profile_ray_geometry.py --full-dose
.venv/Scripts/python.exe benchmark/profile_ray_geometry.py --full-dose --skip-isolated --pairs 6 --out benchmark/pb_audit_results/geometry/repeat
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend cupy --mode protons --variant legacy_geometry --paired --pairs 3 --runs 1
```

`profile_ray_geometry_reference.py` freezes the changed methods from `4c5bef97`;
its installer patches only the benchmark process. In paired audit JSON,
`baseline` means **current production**, and `variant` means **previous geometry**.
Full-dose pairs alternate order and use identical CPU-generated steering.
Raw measurements are under `benchmark/pb_audit_results/geometry/` (initial campaign)
and its `repeat/` directory (six pairs per full-dose workload), with the
initial instrumented profile in `geometry_before/`. Hardware: Intel family 6,
model 183, 32 logical CPUs; NVIDIA RTX 4070 Laptop GPU; Python 3.11.9,
NumPy 2.3.5, CuPy 13.3.0, CUDA runtime 12.6. These laptop measurements include
thermal/runtime noise; consult individual samples rather than treating a small
total-runtime difference as a universal guarantee.
