# Pencil-beam dose calculation performance

Status (2026-11-03): the incremental optimizations of `refactor/dose-calc-performance`
are implemented. Architectural batching (shared work packets, an opt-in particle
adapter, a multi-table interpolation bank with optional CuPy fusion) lives on
`refactor/dose-calc-batching` and is reviewed separately.

This document summarizes what was tried, what was kept and why, and what remains.
Optimization and Monte Carlo engines are out of scope.

## 1. End-to-end result

TG119 influence matrices with the GUI defaults: 3 mm dose grid, 5 mm bixels, Generic
machines. Photons: five fields, dose sampling on (seed 1234). Protons: three fields,
physical dose. Warm medians, develop and branch alternating in fresh processes, on a
laptop (Intel family 6 model 183, RTX 4070 Laptop GPU, NumPy 2.3.5, CuPy 13.3.0).

| Workload | develop | branch | Speedup |
|---|---:|---:|---:|
| Photons, NumPy | 42.8 s | 19.6 s | 2.2x |
| Photons, CuPy | 25.7 s | 8.1 s | 3.2x |
| Protons, NumPy | 48.4 s | 35.4 s | 1.4x |
| Protons, CuPy | 21.0 s | 13.4 s | 1.6x |

Dose values, row indices and column pointers are identical to develop in all four
cases. Row indices and column pointers are stored as int32 when they fit, which saves
a third of the sparse buffers (about 260 MB for the photon case).

Reproduce with [`benchmark/benchmark_dose_revisions.py`](../../benchmark/benchmark_dose_revisions.py):

```powershell
.venv/Scripts/python.exe benchmark/benchmark_dose_revisions.py develop worktree --mode photons --backend cupy
```

Small workloads (one beam, 5 mm grid, 10 mm bixels, no sampling) are dominated by
fixed setup and hide the expensive photon sampling path. Use the GUI-default workload
above before making general speed claims; laptop timings vary by 10-20 % between
sessions, so only compare alternating runs from one session.

## 2. Where the time went

Profiling the photon workload with NumPy early on the branch (about 36 s, before the
sampling and interpolation changes) attributed:

| Work | Time |
|---|---:|
| Photon dose sampling, of which two stable sorts per ray | 13.7 s (10.8 s) |
| Three kernel interpolations per ray | 11.0 s |
| Ray geometry preparation and voxel selection | 5.1 s |
| Kernel convolution / interpolator preparation | 1.5 s |
| Ray tracing | < 0.9 s |
| Patient setup | 0.2 s |

The engines already vectorize over the voxels of one ray or beamlet. The remaining
cost is the number and organization of operations submitted per ray/beamlet:
repeated preparation, repeated mask compaction, sorting where a reduction suffices,
and per-beamlet host/device decisions. The GPU suffers most from the latter.

## 3. Adopted changes

Each change was validated by exact comparison of complete influence matrices against
its predecessor, on NumPy and CuPy, unless stated otherwise. Effects are
whole-calculation changes on the workload named; they do not add up across rows.

| Change | Where | Measured effect |
|---|---|---|
| Grouped maximum instead of two sorts per ray in photon sampling (`xp_utils.grouped_max`, native `maximum.at` on NumPy/CuPy, sorting fallback elsewhere) | `_svdpb` | Photons NumPy, GUI workload: 1.50x |
| One shared-grid interpolator for the convolved photon kernel components (`xp_utils.ArrayBackendGridInterpolator`, trailing value dimension, prepared once per beam) | `_svdpb`, `xp_utils` | 1.13x on top of the above; both together 1.72x NumPy, 1.56x CuPy |
| Compact beam-valid coordinates once per beam and carry integer selections through the ray geometry (`take` instead of repeated boolean masks) | `_base_pencilbeam` | 1.07-1.26x across photons/protons/VHEE, small workloads |
| Ray rotation matrices prepared in one batch per beam; alignment flags downloaded once | `_base_pencilbeam` | Photons CuPy 1.16x, NumPy 1.03x |
| Voxel selection: union VOI masks directly, skip redundant sorting/rounding for single-CT plans | `_base`, `_cst` | Setup 6-8x faster; 1.16-1.42x on small workloads |
| Tissue data from one overlap-resolved label volume; radiological depths stay on the compute backend (`trace_cubes_array`, `ArrayBackendGridResampler`) | `_cst`, `raytracer`, `core.resample` | Tissue setup 2.3x; 1.06-1.29x on small workloads |
| Photon depth-kernel constants cached per beam; broadcast multiply instead of per-component updates | `_svdpb` | Photons CuPy 1.07x, NumPy neutral |
| Particle beamlet mask compacted once, all fields gathered with `take` | `_base_pencilbeam_particle` | Protons CuPy 1.10-1.16x, NumPy neutral |
| Particle depth tables scaled and prepared once per converted kernel (`xp_utils.ArrayBackend1DInterpolator`); focus interpolators reused | `_base_pencilbeam_particle` | NumPy neutral; PyTorch CPU 1.14-1.24x |
| Particle lateral cutoff: kernels converted once per energy, depths evaluated in batches by a shared evaluator (`_particle_cutoff.py`) | `_base_pencilbeam_particle`, `_hongpb` | Cutoff preparation per energy 5.0 -> 0.85 ms (single Gaussian), 509 -> 235 ms (`singleXY`) |
| int32 sparse indices when they fit, promoted to int64 before overflow | `_sparse_output` | 33 % smaller sparse buffers; runtime neutral |
| Energy index lookups cached per calculation; source kernels fetched only on cache misses | `_base_pencilbeam_particle` | Too small to measure; kept because it is simple |

Reusable pieces went into `core.xp_utils` and `core.resample`, not into engine-specific
fast paths. `resample_array` replaces `resample_numpy_array` (deprecated) and accepts
any Array API array.

## 4. Tried and not adopted

| Approach | Outcome |
|---|---|
| Prepared upper bound on the photon sampling cluster count per beam/range shift, to avoid one device sync per ray | No measurable end-to-end gain on either backend; removed |
| Cutoff quadrature created and evaluated on the dose backend (GPU), with a `cutoff_force_numpy` override | Isotropic models are 2-3.5x slower on the GPU; only the anisotropic `singleXY` profits. Cutoff preparation stays on NumPy |
| Stacking photon kernel fields into one native SciPy `RegularGridInterpolator` | Slower than the generic Array API evaluation with packed fields; the default policy (`prefer_native=None`) selects generic for NumPy multi-value 2D grids |
| Full-volume resampling with Array API operations instead of SimpleITK | SimpleITK is faster for full NumPy/CuPy volumes, even including CuPy transfers; it stays the default. The prepared depth resampler only samples selected voxels |
| Uploading the whole kernel database to the GPU, or creating cutoff grids directly on the GPU | Differences within run-to-run noise |
| Device-resident output collection (collect sparse columns on the GPU, transfer once) | About 7 % for full collection, 1 % for bounded collection, with markedly different memory costs; no gain on a small proton case. Deferred: needs GUI/serialization compatibility for device `Dij` |
| A synthetic voxel-owned fused GPU kernel (Gaussian forward model) | Orders of magnitude faster than the per-beamlet loop on the GPU, but the dense CPU equivalent was 4.8x slower. Informs the batching architecture, not a drop-in change |

## 5. Architecture direction

The public facade (`calc_dose_influence`, `calc_dose_forward`, engine configuration,
`Dij`) stays. Below it, the next step is to move from one Python iteration per
ray/beamlet to bounded work packets:

- **Prepared data with explicit lifetimes:** patient (grid, densities, selection),
  beam (transforms, SSDs, depths), kernel (backend tables, interpolation state).
  Caches must reflect device/dtype, machine and energy, geometry-dependent cutoffs,
  focus and range shifters, explicit scenario sets (setup groups, CT phases, range
  values) and engine reuse.
- **Packets evaluated by modality-specific physics,** consumed by a shared output sink
  that keeps the CSC contract (exclusive column ranges, sorted rows). A per-beamlet
  adapter keeps existing engines working.
- **Conservative spatial support:** bin voxels in BEV tiles/depth slabs so a ray only
  scans plausible voxels, then apply the existing exact cutoffs. The envelope must
  cover divergence, anisotropy, range shifts and all CT phases.
- **Different traversals for influence and forward dose:** batched column production
  for influence matrices; voxel-owned reductions over nearby beamlets for forward dose.
- Future fine-sampling particle engines reduce child samples into parent columns;
  a photon AAA-type engine needs multi-component, density-scaled transport and must
  not be forced into "depth lookup times radial profile".

Never broadcast all beamlets against all voxels: a single float32 plane for 908
beamlets and 109k patient voxels is already about 400 MB. Tile by a memory budget,
and keep different strategies for CPU (compact first) and GPU (fewer, larger launches).

## 6. Backlog

1. **Batching** (`refactor/dose-calc-batching`): shared packet scheduler and the opt-in
   particle adapter, pending architectural review.
2. **Conservative spatial candidates** (section 5); measure the preprocessing
   crossover and memory overhead.
3. **Per-beamlet synchronization:** particle dose validity checks (`isnan`/negative
   per beamlet), `grouped_max` index validation and the cluster count in photon sampling
   each force a device sync per ray/beamlet. Batch them without losing error detection.
4. **Device output collection** (section 4), together with device-resident `Dij` support.
5. **Host geometry round trip:** beam preparation uploads ray coordinates, and the ray
   tracer rebuilds a host `Beam` from them; reuse the validated host geometry.

Constraints for all of them: keep the numerical model (cutoff integration, interpolation
precision, sampling policy and random draw order) unless a separate accuracy assessment
is done, keep Array API compatibility, and compare complete outputs, not samples.
