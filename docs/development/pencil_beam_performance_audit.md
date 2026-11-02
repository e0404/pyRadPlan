# Pencil-beam dose calculation: performance and architecture audit

Current outstanding work: [performance backlog](dose_performance_backlog.md).

2026-10-26. Reviewed revision `c1ee82ae868b92cadfe72adf09bcc030188533bb` on
`develop`. Experiments are isolated scripts under `benchmark/`; engine source
files were not edited. Optimization and Monte Carlo engines are outside scope.

This audit describes the original revision. The subsequent implementation of
photon interpolator reuse is recorded in the [step 1 follow-up](#step-1-photon-interpolator-reuse).
Its integration was subsequently moved into the reusable
[Array API interpolation backend](#reusable-array-api-interpolator).
The shared voxel-preparation follow-up, including measurements across modalities,
is documented in [voxel preparation performance](voxel_preparation_performance.md).
The next shared changes are recorded in
[tissue and backend depth preparation](patient_depth_performance.md).
The subsequent shared ray-loop changes are measured in
[ray geometry preparation](ray_geometry_performance.md).

## Main findings

The main limitation is the **size and organization of work submitted to the
backend**, rather than a missing GPU implementation of the Gaussian formula.
The engines already vectorize over the voxels of one ray/beamlet, and already
have several specialized geometry/ray-tracing kernels. Python still schedules
rays and beamlets individually, with data-dependent compaction, scalar decisions,
repeated interpolation setup, and per-beamlet host output.

There are three distinct opportunities:

1. **Improve the existing engines without changing their public interfaces.**
   Reuse interpolation objects, compact masks once, hoist invariant preparation,
   use direct reductions for photon sampling, and streamline patient setup.
2. **Introduce a shared prepared-data and batched-execution layer.** Keep geometry,
   support selection, physics, and output assembly independently replaceable.
   Both CPU and GPU benefit; the GPU particularly needs fewer small operations
   and host synchronization points.
3. **Use different traversal/output strategies for influence and forward dose.**
   Influence calculation benefits from batched column production. Forward dose
   can instead use voxel-owned reductions over nearby beamlets and avoid
   materializing individual columns. A universal Python loop over voxels would
   make matters worse; the voxel loop belongs in compiled/vectorized kernels.

The first implementations should preserve the numerical model. Changing cutoff
integration, interpolation precision, dose sampling, or transport discretization
requires a separate accuracy assessment, even when faster.

## Measurement method and coverage

Hardware: Windows, Intel64 Family 6 Model 183 Stepping 1, 32 logical CPUs;
NVIDIA GeForce RTX 4070 Laptop GPU, 8 GiB. Software: repository `.venv`, Python
3.11.9, NumPy 2.3.5, CuPy 13.3.0, CUDA runtime 12.6. Existing ray-tracer JIT
configuration was retained (`numpy,jax`, including the available NumPy/Numba
paths). No other array backends were benchmarked.

The main workload is the bundled TG119 phantom, one nominal beam at 0 degrees,
10 mm bixels, and a 5 mm dose grid: 101 × 101 × 65 total voxels, with 108,871
selected patient voxels. Particle steering is target-based; VHEE uses 200 MeV.
Generic and Focused VHEE were tested. The photon field case uses one jaw-shaped
field with 0.5 mm fluence resolution and the existing field-based calculation.
A finer proton case uses 3 mm voxels and 5 mm bixels.

Steering was generated with NumPy for both backends. Timings include a fresh
engine, an STF copy, patient/machine initialization, beam preparation, transport,
kernel evaluation, and output assembly. Forward timings also include the public
API's result conversion/resampling to the CT grid. They exclude initial script
imports, phantom loading, steering generation, and output-file serialization;
lazy backend imports triggered inside dose calculation remain included. The initial
CuPy context is touched before timing. The recorded `cold_seconds` is therefore
the **first dose call**, not full process startup.

Each ordinary case has one initial call and three warmed calls. Reported
speedups use six additional **interleaved baseline/prototype pairs**, alternating
execution order. CuPy wall times are synchronized at completion. Separate
`cProfile` and synchronized method-level runs identify bottlenecks; their timings
are not substituted for uninstrumented totals. Stage synchronization perturbs
execution and is not a CUDA kernel timeline. No Nsight occupancy or bandwidth
measurement was performed. These timing precautions follow the
[CuPy benchmarking guidance](https://docs.cupy.dev/en/stable/user_guide/performance.html).

Early exploratory proton timings showed large machine-state drift and were
rerun; the final baseline files and tables use the later runs. Remaining
variation is exposed in raw samples and paired ranges. This is a laptop/workstation
benchmark, not a claim about every CPU/GPU or production patient.

Physical dose is isolated using `bio_model="none"`, `calc_let=False`, and, unless
labelled sampled, `enable_dij_sampling=False`. Photon sampling is enabled in
separate cases with a fixed seed. Additional proton LET cases exercise the
existing additive output path. Biological influence kernels were inspected but
not timed. The multi-Gaussian branch was inspected; the selected bundled cases
exercise single, double, and anisotropic single-XY models. Robust scenarios,
range shifters, and multi-beam cache reuse are not validated by these timings.

### Full-engine baseline times

Warm median seconds. GPU/CPU is a time ratio: above 1 means the GPU was slower.
Rows are separate workloads; the one-field photon row is not an equivalent
replacement for the 121-column influence calculation.

| Influence workload | Rays / beamlets | NumPy s | CuPy s | GPU/CPU |
|---|---:|---:|---:|---:|
| Protons, single Gaussian | 72 / 908 | 1.010 | 2.360 | 2.34 |
| Helium, double Gaussian | 72 / 1,348 | 1.634 | 3.566 | 2.18 |
| Carbon, single Gaussian | 72 / 1,182 | 1.122 | 2.784 | 2.48 |
| Oxygen, single Gaussian | 72 / 1,332 | 1.260 | 3.053 | 2.42 |
| VHEE Generic, double Gaussian | 121 / 121 | 1.085 | 1.278 | 1.18 |
| VHEE Focused, single-XY Gaussian | 121 / 121 | 1.505 | 1.753 | 1.17 |
| Photon beamlets, unsampled | 121 / 121 | 1.383 | 2.422 | 1.75 |
| Photon beamlets, sampled | 121 / 121 | 1.571 | 2.651 | 1.69 |
| Photon single shaped field | 1 / 1 | 1.085 | 0.665 | 0.61 |
| Protons, 3 mm grid / 5 mm bixels | 272 / 3,592 | 3.267 | 7.520 | 2.30 |

The finer proton case has 502,192 selected voxels and approximately 16.54 million
stored influence entries. GPU execution does not become advantageous merely by
increasing the voxel and beamlet counts within the present schedule.

| Unsampled forward workload, 5 mm grid | NumPy s | CuPy s |
|---|---:|---:|
| Protons | 1.081 | 2.344 |
| Photon beamlets | 1.371 | 2.359 |
| VHEE Generic | 1.079 | 1.344 |

Forward calculation currently saves matrix storage but retains most of the
beamlet scheduling and output-transfer work, which is visible in these totals.

With proton LET output enabled, warm medians were 1.015 s on NumPy and 2.379 s
on CuPy. These overlap the physical-only timing variation, so they do not
establish a precise marginal LET cost. A separate storage check confirmed that
physical and LET matrices share row indices and column pointers; LET added
3,734,596 data bytes for 933,649 entries.

Selected synchronized stage observations, from separate instrumented calls:

| Workload / backend | Stage | Seconds |
|---|---|---:|
| Protons, 5 mm / NumPy | Shared dose initialization | 0.658 |
| Protons, 5 mm / CuPy | Shared dose initialization | 0.660 |
| Protons, 5 mm / CuPy | 908 bixel initializations / support selection | 0.763 |
| Protons, 5 mm / CuPy | Host output assembly/transfers | 0.086 |
| Protons, 3 mm / CuPy | 3,592 bixel initializations / support selection | 2.908 |
| Protons, 3 mm / CuPy | Patient bixel physics, excluding cutoff calibration | 2.029 |
| Protons, 3 mm / CuPy | Ray geometric-distance calculation | 0.447 |
| Protons, 3 mm / CuPy | Radiological depth tracing | 0.268 |
| Protons, 3 mm / CuPy | Host output assembly/transfers | 0.473 |
| Photon beamlets / CuPy | Bixel evaluation, dominated by interpolation | 1.227 |
| VHEE Focused / NumPy | Lateral cutoff preparation, including dummy physics | 0.581 |
| VHEE Focused / CuPy | Same CPU cutoff preparation | 0.570 |

These are not additive percentages of the earlier uninstrumented medians.
In particular, per-beamlet transfers are an architectural obstacle, but their
measured assembly stage is much smaller than support preparation and physics
dispatch. PCIe bandwidth alone does not explain the GPU slowdown.

### Full-engine prototype comparisons

Median of six interleaved runs of each version. The range is the minimum and
maximum of the six paired time ratios, not a confidence interval. All these
comparisons produced identical same-backend dose values; influence comparisons
also retained identical row indices and column pointers.

| Prototype / workload | Backend | Baseline s | Prototype s | Speedup | Paired ratio range |
|---|---|---:|---:|---:|---:|
| Persistent photon interpolators | CuPy | 2.277 | 1.697 | **1.34×** | 1.29–1.38 |
| Persistent photon interpolators | NumPy | 1.310 | 1.253 | 1.05× | 1.03–1.13 |
| Boolean VOI union + omit redundant nominal CT-index unique, photons | NumPy | 1.381 | 1.140 | **1.21×** | 1.17–1.25 |
| One bixel-mask compaction + hoisted cutoff conversion, protons | CuPy | 2.360 | 2.004 | **1.18×** | 1.12–1.22 |
| Same particle changes | NumPy | 1.060 | 1.009 | 1.05× | 0.99–1.16 |
| Direct cluster maximum, sampled photons | NumPy | 1.568 | 1.414 | **1.11×** | 1.09–1.21 |
| Direct cluster maximum, sampled photons | CuPy | 2.605 | 2.459 | **1.06×** | 1.04–1.13 |
| Hoist fluence FFT, single photon field | CuPy | 0.674 | 0.674 | 1.00× | 0.94–1.07 |
| Hoist cutoff dictionary conversion, Focused VHEE | NumPy | 1.501 | 1.491 | 1.01× | 0.93–1.03 |
| Device-only forward accumulation, protons | CuPy | 2.402 | 2.343 | 1.02× | 0.97–1.06 |
| Device-only forward accumulation, photon beamlets | CuPy | 2.360 | 2.315 | 1.02× | 0.96–1.08 |

The last four rows have **no convincing full-engine improvement in these runs**.
They illustrate why removing a redundant operation or copy is not sufficient
when it occupies little of the critical path, or when its replacement adds small
GPU scatter launches. The CPU particle improvement is also small/noisy. Other
than the explicitly combined particle prototype, these changes were tested
individually; do not multiply their speedups to predict a combined result.

### Numerical comparison and limits

Every tested full-engine prototype matched its **same-backend physical-dose
baseline exactly**, including forward results on the CT grid. The influence
prototypes also preserved CSC support and order, including the fixed-seed
sampling prototype. These checks cover the benchmark cases, not all possible
engine configurations.

NumPy and CuPy baseline outputs are not bitwise identical:

| Baseline influence case | max absolute difference / NumPy peak | Relative L2 difference | Support disagreements |
|---|---:|---:|---:|
| Protons, 5 mm | 3.86e-6 | 5.90e-7 | 0 |
| Helium | 5.05e-5 | 1.94e-6 | 6 entries |
| Carbon | 9.11e-6 | 1.22e-6 | 0 |
| Oxygen | 7.13e-6 | 1.23e-6 | 0 |
| VHEE Generic | 5.92e-7 | 2.45e-7 | 0 |
| VHEE Focused | 4.61e-7 | 1.62e-7 | 0 |
| Photon beamlets, unsampled | 7.81e-7 | 3.09e-7 | 0 |
| Photon shaped field | 3.15e-7 | 4.51e-8 | 0 |
| Protons, 3 mm / 5 mm bixels | **1.56e-3** | 3.04e-6 | 1 entry |

The fine proton maximum is a zero/nonzero disagreement at row 1,377,339,
column 3,098: NumPy stores zero and CuPy stores 2.3560e-6. It is 0.156% of the
maximum individual influence coefficient, despite the small global L2 error.
Helium similarly has six zero/nonzero disagreements. This is consistent with
hard support-selection sensitivity; the exact responsible predicate was not
isolated. Do not advertise cross-backend bitwise parity or hide support changes
behind an aggregate norm. These differences pre-exist the tested accelerations.

For the three 5 mm unsampled forward cases, peak-normalized differences are
4.77e-7 (protons), 8.94e-7 (photons) and 6.12e-7 (VHEE). The separately saved
proton LET influence matrix has peak-normalized difference 4.21e-6 and relative
L2 difference 1.22e-6. See the
[LET validation record](../../benchmark/pb_audit_results/let_validation/summary.json).

## Where the current implementation spends work

### Shared execution and geometry

[`PencilBeamEngineAbstract._calc_dose`](../../pyRadPlan/dose/engines/_base_pencilbeam.py)
loops over shift scenario → beam → ray → CT/range scenario → beamlet. Geometry
is shared across energies on a ray already; this is worth retaining. However,
the unit of physics work and output handling is still one beamlet.

`calc_geo_dists` scans the beam's valid voxel set for every ray. It repeatedly
derives `nonzero(rad_depth_mask)` and gathers the same coordinate subset, then
rotates it, selects a cone, reconstructs a full mask, and gathers outputs.
`_get_ray_geometry_from_beam` then builds further masks and compacts radial,
lateral, projected-lateral, geometric-depth, and radiological-depth arrays.
This is roughly **O(number of rays × beam-valid voxels)** before energy-specific
support selection. It becomes particularly wasteful for narrow particle spots
and fine lateral sampling.

The CuPy geometry kernel is already fused, but its surrounding operations are
not. Boolean/scalar conversions, `nonzero`, repeated boolean indexing, and the
small rotation setup remain. Merely replacing its ElementwiseKernel with a
RawKernel would not remove these costs.

Practical changes within this design:

- Precompute beam-valid indices and compact coordinates once per beam/scenario.
  Keep mappings from compact positions to global dose-grid rows.
- Produce one ordered integer selection and reuse it for every geometry field.
  Avoid expanding to a full mask only to compact it again.
- Make geometry fields demand-driven: isotropic particle models need squared
  radial distance and radiological depth; focused VHEE needs lateral components;
  photons also need projected lateral coordinates and geometric distance.
- Keep ray counts, energy identifiers, and tiny geometry metadata on the host,
  or upload them as a batch. `_init_beam` currently sends bixel counts to the GPU
  and `_init_ray` reads individual counts back with `int(...)`.
- Dispatch on the selected namespace before probing other backends. In
  `calc_geo_dists`, `cupy_available()` precedes the namespace check, and
  `pytorch_gpu_available()` precedes the Torch namespace check. These helpers
  import their backends. A NumPy dose call can therefore import CuPy and Torch
  unnecessarily on its first geometry call; warm calls still repeat probes.
  Move selection/capability resolution to preparation. No separate first-call
  speedup was measured for this proposed fix.
- Compute SSD as a batch and transfer the resulting scalar vector once if the
  existing Python ray dictionaries need host SSD values. The current per-ray
  `if ix_nan[j]` and `float(ssd[j])` create individual GPU synchronization points.

These are exact-representation changes if voxel ordering, finite-depth masks,
scenario selection, and cutoff boundaries are preserved.

### Patient setup is a CPU floor for every modality

[`DoseEngineBase._init_dose_calc`](../../pyRadPlan/dose/engines/_base.py)
constructs a union of VOI indices with concatenate/unique, repeats uniqueness
and membership work for scenarios, resamples masks/CT, generates coordinates,
and loads machine data. Nominal scenarios often reconstruct information that is
already unique or already available as masks.

It also builds alpha, beta, and tissue-class volumes even for physical-only
calculations. [`StructureSet`](../../pyRadPlan/cst/_cst.py) resolves overlap
priorities separately for the LQ arrays and tissue-class indices. This is a
real dose-initialization cost, independent of the optimizer. It explains why a
fast one-field photon kernel can have little effect on total latency.

An exact first step is a dense boolean union for bounded voxel indices, followed
by one `flatnonzero`; retain a sparse strategy for unusually huge grids with tiny
occupancy. Do not remove sorting/uniqueness for arbitrary user-supplied indices.
Resolve overlap once and reuse it. Longer term, cache immutable patient
preparation keyed by CT/density/structure revisions, grid, overlap rules, and
scenario. Make tissue arrays lazy where the `Dij` contract allows, or cache them
while preserving the current public fields.

### Ray tracing: useful existing optimizations, remaining boundaries

[`RayTracerSiddon`](../../pyRadPlan/raytracer/_siddon.py) already caches device
geometry and density arrays. [`raytracer/_kernels.py`](../../pyRadPlan/raytracer/_kernels.py)
provides specialized merge/index/selection work, and `trace_cubes` uses the
device-returning trace path internally. The ray-matrix extent is already
restricted using patient extent and ray positions. Reimplementing these solved
pieces is not the first priority on the measured workloads.

Two remaining round trips are explicit:

- `trace_cubes` finishes by copying its depth array to a SimpleITK image;
  `_init_beam` immediately uploads that image's array again for GPU dose work.
- `_compute_ssd` calls public `trace_rays`, whose contract returns host arrays,
  then converts the returned segment arrays back to the dose backend.

Add a private/public backend-array result with grid metadata and keep SimpleITK
conversion at image/output boundaries. SSD needs access to alpha/entry data as
well as the existing device trace outputs; simply replacing the call with the
current three-output `_trace_rays_device` is insufficient.

The padded ray × intersection arrays still scale with ray count and maximum
segment width. Tile tracing under a memory budget before considering a fused
per-ray traversal with prefix integration. A voxel-backprojection tracer is
another option, but it repeats path integration and changes the current
depth-assignment discretization. Compare accuracy as well as speed. The existing
data-dependent width reduction has a synchronization cost but can save much
more downstream work; removing every synchronization indiscriminately is not
an improvement.

### Particle and VHEE work

[`ParticlePencilBeamEngineAbstract`](../../pyRadPlan/dose/engines/_base_pencilbeam_particle.py)
serves protons, helium, carbon, oxygen, and VHEE through
[`ParticleHongPencilBeamEngine`](../../pyRadPlan/dose/engines/_hongpb.py).

**Cutoff calibration:** `_calc_lateral_particle_cut_off` always runs on the CPU.
For each used energy and roughly 35–38 depths it creates a dummy bixel and
evaluates a radial profile. Inside that depth loop it reconverts the same
kernel model to a NumPy dictionary. The conversion includes optional fields,
device dispatch, and repeated backend/module checks. Hoisting it once per
energy removes work without changing the integration. It does not eliminate
the remaining interpolation/profile/integration cost.

Focused VHEE's `singleXY` branch expands the 1,200 radial bins to 360 angles:
**432,000 sample points per depth**, approximately 16 million evaluations for
38 depths of one energy. This work remains on the CPU when dose uses CuPy.
Hoisting a small dictionary alone therefore has little impact for this case.
The valuable next step is to interpolate kernel parameters once per depth and
evaluate/integrate the lateral distribution in batches, without repeatedly
interpolating a depth vector whose every value is identical.

For an isotropic single Gaussian, the integral cutoff has a closed form:

```python
# For the continuous normalized 2D Gaussian and integral fraction p:
radius_squared = -2 * sigma_squared * log1p(-p)
```

For positive Gaussian mixtures, solve
`sum(w_j * exp(-R² / (2*sigma_j²))) = 1-p` with a bracketed vectorized solve.
These formulas need no IDD evaluation. Anisotropic distributions need angular
quadrature or a conservative major-axis envelope; the isotropic formula cannot
simply be reused. Continuous solutions will differ from today's discrete radial
quadrature and its compensation convention. Treat these as numerical-model
changes and validate retained dose, tail shape, and cutoff discontinuities.
The `relative` cutoff mode needs its own derivation.

**Per-beamlet preparation:** energy lookup and focus interpolation are repeated.
`_calc_sigma_ini_focus_ix` constructs a SciPy interpolator for a scalar SSD each
time. Prepare energy indices and reusable focus interpolators, or use a
carefully equivalent linear interpolation routine including extrapolation.

**Support:** `_get_bixel_indices_on_ray` computes a depth-dependent mask, then
applies that boolean mask separately to indices, radial distance, depth, lateral
distance, and optional tissue fields. A single integer compaction avoids repeated
GPU scans and result-size decisions:

```python
selected = xp.nonzero(keep)[0]
for name in required_fields:
    bixel[name] = xp.take(ray[name], selected, axis=0)
```

**Physics:** the NumPy/CuPy fast path in `interp1d` loops over dictionary fields,
calling `xp.interp` independently. IDD, sigma(s), mixture weights, LET and
biological kernels repeat the same depth search. Prepare one interpolation
stencil per beamlet/batch and reuse it across channels; compare with the very
efficient native NumPy interpolation before assuming it helps CPU performance.
Fuse interpolation, Gaussian mixture evaluation, cutoff, and output writing in
a backend specialization once batches are large enough. Mixed float32 geometry
and float64 base data/initial sigma promote intermediate work; a consistent
prepared precision could save bandwidth, but reducing precision is not an
unconditionally exact fix.

Cache the scaled IDD once per energy and upload only the kernel channels required
by the request; `kernel.to_namespace` currently visits optional LET/biological
fields even when they are not needed. The two `xp.any` dose-validity checks in Hong evaluate host conditionals per
beamlet. Preserve detection using a batch error flag and inspect it once per
batch, with diagnostic beamlet IDs. The current per-energy device kernel cache
is already useful; it should not be replaced with per-beamlet uploads.

**Cache semantics matter:** cutoff calibration mutates machine kernels for each
beam, whereas `_kernel_cache` is keyed only by energy and namespace. New caching
must include device, dtype, machine/kernel revision, cutoff method/level,
focus/SSD/range-shifter envelope and scenario-dependent inputs as appropriate.
Separate immutable physical tables from beam-specific cutoff preparation.
The existing multi-beam invalidation deserves a focused regression test before
extending cache lifetimes.

### Photon work

[`PhotonPencilBeamSVDEngine`](../../pyRadPlan/dose/engines/_svdpb.py) has two useful
paths already: ordinary beamlets share prepared convolution kernels when
primary fluence is uniform, and field-based dose can convolve a complete field.
Do not describe convolution as repeated per bixel in the shared-kernel path.

The expensive repetition is in evaluation. `_get_kernel_interpolators` returns
`partial(interpnd, ...)`, not a persistent interpolator.
[`interpnd`](../../pyRadPlan/core/xp_utils/compat.py) checks grid direction and
bounds, clips the queries, and constructs SciPy/CuPy `RegularGridInterpolator`
on **every call**. Three components × 121 rays means 363 constructions and
repeated checks/searches. Persisting the backend interpolator preserves its
bounds-error behavior while avoiding the redundant outer preparation. Sharing
the bilinear stencil across components is the next step; uniform convolution
grids also permit arithmetic indices instead of binary search if floating-point
boundary semantics are checked.

`_get_kernel_interpolators` recomputes the same fluence FFT for every SVD
component. Hoist that transform; cache component spectra and the Gaussian-filter
spectrum where their keys match. Batched FFTs and real transforms are additional
possibilities, but require padding/alignment/normalization checks. Do not shrink
the physical convolution support just to obtain faster FFT dimensions. Two
`conv_mx_*` meshgrids are allocated and then unused; their removal is straightforward.

`_compute_bixel` also repeatedly uploads beta/m constants, allocates a component
× voxel depth-response array, and multiplies each interpolated component into
it. These constants and shared depth response can live at beam scope; a fused
evaluation can avoid materializing every component. Per-beam response caching
trades memory for repeated exponential work and should be optional.

**Sampling:** `_sample_dij` computes maximum tail dose per depth cluster using
two stable sorts. NumPy and CuPy support a direct indexed maximum, preserving
the same finite-dose maximum and the existing host RNG stream:

```python
cluster_dose = xp.zeros(num_clusters, dtype=dose.dtype)
xp.maximum.at(cluster_dose, cluster, dose_tail)  # specialized NumPy/CuPy path
```

Keep a generic fallback because scatter-maximum is not in the base Array API.
An atomic GPU reduction can contend for few clusters, so measure at actual
cluster sizes. Sampling currently happens after full dose computation; its
main benefit is output sparsity, not avoiding the physics calculation. Host RNG
generation/upload and final compaction remain. Preserve deterministic sample
mapping if batching changes execution order; use a counter-based scheme only
with an explicitly defined reproducibility contract.

Forward dose has a dense final accumulator, so sparse-tail sampling provides
less benefit there. An explicit unsampled-forward policy is preferable to
silently inheriting influence sparsification, but changes existing stochastic
output semantics and should be an intentional API/default decision.

## Storage and host/device boundaries

`_allocate_quantity_matrices` deliberately allocates NumPy output. `_fill_dij`
copies each GPU beamlet's physical values and row indices to the host, and then
copies any additional quantities. Forward mode also accumulates on the CPU.
Its second `to_numpy(physical_dose)` is generally a dispatch/no-op on the already
converted array; it is **not a second device transfer** of the same values.

The sparse representation already shares indices and column pointers across
quantities during assembly and constructs CSC without an intended data copy.
Retain this benefit. The main opportunities are transfer granularity, index
width, and growth policy:

- **Forward:** accumulate on the device and copy one final dose array per
  beam/scenario or quantity. The tested prototype preserves the existing
  per-beamlet evaluation schedule; it therefore measures only the sink change.
- **Influence:** produce bounded chunks of columns/entries on the device and
  transfer them in larger contiguous blocks, optionally through pinned buffers.
  Budget chunk size from live quantities, support indices, FFT workspace and
  transport buffers. Keeping the entire influence matrix on an 8 GiB GPU is
  not required to obtain the batching benefit.
- **Index width:** current CSC arrays use int64 rows and pointers with float32
  values. When grid size, columns and total nnz fit the chosen signed integer
  limits, int32 rows/pointers reduce a physical-only matrix from roughly
  12 to 8 bytes/entry: about **33% less storage**, without dose rounding.
  If total nnz exceeds 2³¹−1, use int64 pointers, a compatible sparse library
  representation, or bounded blocks; never silently overflow. With Q values
  sharing support, bytes/entry are `8 + 4Q`, versus `4 + 4Q` for int32 rows.
- **Growth:** the current resize estimate extrapolates the current column's
  occupancy over all remaining columns. It can overallocate on heterogeneous
  beams. Use bounded geometric growth or chunks plus a final concatenate.
  A two-pass count/prefix/fill method is attractive when support predicates are
  cheap, but recomputing expensive physics just to count entries is undesirable.
  This was not a measured problem in the proton cases: recorded capacity was
  934,385 for 933,649 entries and 16,624,013 for 16,537,575 entries. Treat a growth
  policy change as a scalability safeguard, below the measured hotspots.

For the 5 mm proton case, 933,649 stored entries occupy 11,211,060 bytes including
CSC pointers; all-int32 indexing would use 7,472,828 bytes. These are calculated
storage sizes, not an end-to-end timing claim. Compact patient-row indices alone
do not shrink an index element unless its dtype/encoding changes. Preserve a
mapping to full image rows and materialize the current external `Dij` format
at its boundary.

The engine also calls `free_gpu_memory` at every beam boundary. This releases
cached free blocks, not all live arrays. Reusing a bounded pool across beams
can avoid allocation overhead. CuPy explicitly uses its pool to reduce allocation
and synchronization costs; use a budget/pressure policy rather than unconditional
purging. ([CuPy memory management](https://docs.cupy.dev/en/stable/user_guide/memory.html))

## Controlled primitive and architectural experiments

These are **microbenchmarks**, separate from the full-engine measurements above.
Raw repetitions are in
[`microbenchmarks.json`](../../benchmark/pb_audit_results/microbenchmarks.json).

| Primitive | Backend | Current pattern ms | Alternative ms | Ratio |
|---|---|---:|---:|---:|
| Four 100k-element arrays, 5% selection: repeated masks → one integer selection | NumPy | 0.369 | 0.131 | 2.83× |
| Same size/density | CuPy | 0.292 | 0.170 | 1.72× |
| Four 100k-element arrays, 50% selection | NumPy | 1.733 | 0.166 | 10.42× |
| Same size/density | CuPy | 0.306 | 0.147 | 2.09× |
| Maximum of 100k values in 300 clusters: two sorts → indexed maximum | NumPy | 13.414 | 0.103 | 130.23× |
| Same reduction | CuPy | 0.796 | 0.145 | 5.49× |
| Actual TG119 VOI union, 610,514 input indices: unique → boolean union | NumPy | 128.422 | 2.080 | 61.74× |

All primitive alternatives were checked for exact output equality. Large
primitive ratios become modest engine gains because the primitives account for
only part of runtime; the full-engine table is the decision-making evidence.

The architectural experiment uses **synthetic Gaussian forward scoring** with
65,536 voxel positions, 256 spots, a depth-dependent variance, a finite radial
cutoff and nonuniform weights. It includes neither pyRadPlan transport nor
machine-table interpolation. Inputs are prepared before timing; final host dose
transfer is included. The schedules compute the same synthetic model:

| Schedule | NumPy ms | CuPy ms |
|---|---:|---:|
| Python beamlet loop, compact support, host dose accumulation | 35.129 | 131.935 |
| Vectorized blocks of 16 beamlets, local/device accumulator | 170.102 | 5.664 |
| Fused kernel: one GPU thread owns a voxel and loops over spots | — | 0.276 |

The tiled GPU version is 23.3× faster than this synthetic GPU loop; the fused
voxel kernel is approximately 478× faster. This changes **fusion, launch count,
temporary allocation, compaction, and accumulation location together**, so it
does not isolate traversal order. It is not a prediction of a 478× pyRadPlan
speedup. The kernel still scans every spot and omits realistic interpolation,
multiple energies/components, transport, scenarios and sparse influence output.
Its final dose differs from the NumPy reference by at most 2.0e-7 of the peak;
the tiled versions differ by at most 4.0e-7, due to floating-point reduction order.

The CPU tiled version is **4.84× slower**: it materializes dense tiles and
evaluates exponentials even for rejected interactions, while the loop compacts
first. This supports a scheduler with CPU/GPU-specific strategies and bounded
candidate lists, rather than replacing every beamlet loop with dense broadcasting.

## A modular architecture that supports all engines

Keep `calc_dose_influence`, `calc_dose_forward`, plan/config models, engine
registration, progress reporting, and `Dij` as the public facade. Internally,
introduce the following contracts gradually:

| Component | Owns | Shared or modality-specific? |
|---|---|---|
| PreparedPatient | Grid transforms, density, selected-row map, resolved structures, optional tissue channels | Shared, with modality-specific density calibration |
| PreparedBeam/TransportContext | Beam transform, SSDs, radiological depth and optional transport fields | Shared interface; replaceable transport implementation |
| PreparedKernel | Immutable backend tables, interpolation metadata, constants, precision/device identity | Modality-specific preparation behind a common lifecycle |
| SupportProvider | Conservative bounds, candidate bins/tiles, ordered voxel/beamlet adjacency | Shared machinery; modality supplies bounds/dependencies |
| ExecutionStrategy | Work packet size/order, CPU/GPU dispatch, stream and memory budget | Shared |
| PhysicsEvaluator | Named additive contributions on a packet, optional richer transport operation | Modality-specific |
| DoseSink | Forward reduction or influence column/block assembly, quantity metadata | Shared |

An illustrative interface, not a proposed final API:

```python
patient = prepare_patient(ct, structures, grid, request)
for beam in steering.beams:
    context = transport.prepare(patient, beam, request)
    kernels = physics.prepare(beam, context, backend)
    support = physics.support_provider(context, kernels)
    for packet in scheduler.packets(context, support, memory_budget):
        contributions = physics.evaluate(packet, context, kernels, request.quantities)
        sink.consume(packet, contributions)
return sink.finalize(public_grid=grid)
```

Describe required inputs and produced additive quantities explicitly. Avoid a
Python callback per voxel or per pair: dispatch once per packet into an Array
API implementation or a registered backend specialization. Keep a reference
implementation and a legacy `_compute_bixel` adapter so adding a new engine
does not initially require writing CUDA. Backend specializations should implement
the same contracts, with capability tests and numeric parity tests.

### Traversal choices

| Strategy | Best use | Main tradeoff |
|---|---|---|
| Ray-major, batched energies | Low-risk migration for particles/VHEE and CPU sparse influence | Ragged supports; bucket similarly sized rays and avoid Python per-energy launches |
| Pair tiles / segmented sparse blocks | Shared GPU influence execution across modalities | Need compaction/prefix sums and stable external column identities |
| Voxel-major gather over nearby beamlets | Forward dose, many spots/energies, few final quantities | Must construct conservative candidate adjacency; otherwise scans all beamlets per voxel |
| Voxel × energy-layer work | Particle/VHEE reuse of depth-table interpolation | SSD, focus, range shifters and offsets can remain spot-specific; energy alone is not a universal cache key |
| Whole-field / component convolution | Photon fields where transport assumptions permit | Spatially varying heterogeneity and fluence/model distinctions limit convolution reuse |

For influence calculation, ultimately writing O(nnz × quantities) data is
unavoidable. Batched beamlet evaluation with exclusive output column ranges
avoids atomics and naturally preserves the current CSC contract. Generate sorted
row IDs or explicitly sort/canonicalize once. A voxel-major producer naturally
resembles CSR; converting it to CSC has a cost that belongs in the comparison.

For forward calculation, assigning each voxel/tile an owner avoids global
floating-point scatter atomics and keeps its sum in registers/local memory.
Parallelize over energy or components when this creates more work, then use a
bounded reduction. A photon field is not necessarily best handled by enumerating
all its tiny beamlets. A published proton implementation uses voxel × energy
threads and spot loops, demonstrating this scheduling family; its transport,
precision, and accuracy choices are not automatically interchangeable with
pyRadPlan. ([Primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12148800/))

The shared accelerator is **conservative spatial support**: bin projected voxel
positions or ray/spot centers in BEV tiles, optionally in depth slabs, and generate
only plausible interactions. Apply the existing exact cone and depth-dependent
cutoff after coarse selection. Divergence, anisotropic widths, halo components,
off-axis beams, range shifts, and all contributing energies must be covered by
the envelope. A radius inferred solely from the highest energy is not a general
proof of support for a new model. Index preparation has to be amortized; it may
not pay for a single field or very small case.

Do not globally broadcast `(all_beamlets, all_patient_voxels, all_components)`.
Even one float32 plane for 908 beamlets × 108,871 patient voxels is approximately
395 MB before geometry, masks, interpolation and output; using all 663,065 grid
voxels is about 2.41 GB. Tile by a measured memory budget. CPU execution should
also prefer cache-sized tiles and avoid oversubscribing BLAS, NumPy/Numba,
SimpleITK and outer worker threads simultaneously.

### Future particle fine-sampling

Represent delivery spots as stable **parent columns** and fine transport samples
as internal children with position, weight and transport state. Trace/transport
children in batches. Reduce child contributions into the parent's dose or
influence column before writing; do not expose every child as a new deliverable
spot. Preserve total particle/MU normalization and sum additive quantities
before constructing ratios or other derived output.

Fine samples can have distinct heterogeneity histories, splitting criteria and
anisotropic spreading. They must not all inherit a central-ray depth merely to
reuse a cache. Adaptive subdivision can be represented by ragged child ranges;
bucket work sizes, and construct parent support as a conservative union. This
fits the common packet/sink design without forcing photons to implement spot
splitting.

### Future photon AAA

AAA needs richer transport than plugging a new scalar Gaussian into Hong.
The published model combines primary/secondary photon and contamination-electron
contributions, with density-scaled three-dimensional scatter kernels. Spatially
varying density scaling prevents assuming a single shift-invariant whole-volume
FFT is generally equivalent.
([AAA evaluation paper](https://aapm.onlinelibrary.wiley.com/doi/10.1118/1.2358333))

Let the photon evaluator prepare multiple source components and access local
density/transport neighborhoods through `TransportContext`; it may use tiled
convolution/superposition rather than independent pointwise evaluations. The
common pieces remain packet scheduling, coordinates, conservative support,
precision/device management, named additive channels, and output reduction.
This preserves engine modularity without forcing every model to be separable
into one depth lookup times one radial profile.

### CPU parallelism and GPU specialization

CPU beam/ray blocks can be independent once prepared data are immutable and
output ranges are assigned. Prefer compiled parallel loops or coarse tasks over
one Python job per beamlet. Multiple processes can duplicate large CT/kernel
buffers; shared read-only arrays and explicit thread limits are needed. Keep
progress/cancellation at packet boundaries.

On GPU, one or a few fused kernels per packet should perform depth interpolation,
support testing, physics and scoring. Use structure-of-arrays beamlet metadata,
packed lookup tables, and fixed/bucketed sizes where useful. After launch counts
are reduced, investigate graph capture, asynchronous transfer overlap or multiple
streams. Dynamic compaction, allocations, and host branches currently make
graph capture a poor first intervention. Preserve a generic Array API route;
there is no requirement that the fastest GPU packet kernel be expressible only
as elementary Array API operations.

## Suggested implementation order

1. Land independently reviewable, exact changes: prepared photon interpolators;
   single mask compaction; cutoff-conversion and FFT hoisting; specialized
   sampling maximum; removal of unused grids. Keep finite/bounds checks and
   deterministic sampling behavior.
2. Streamline patient preparation, cache reusable results, and introduce a safe index-width policy.
   Retain external grid/serialization semantics. Add a device forward sink and
   bounded influence chunks; move transport array/image conversions to the edge.
3. Introduce immutable prepared beam/kernel data and packet evaluation behind
   existing engine classes. Start with ray-major particle batches and shared
   photon interpolation stencils. Add conservative spatial bins and measure
   their crossover at several resolutions/spot counts.
4. Add voxel-owned forward evaluation as a strategy, then parent/child fine
   sampling and richer photon transport. Keep independent physics validation
   for changes to approximation, cutoff, or transport.

For each phase validate all additive outputs, empty rays/columns, sorted unique
rows, nonuniform weights, beam/scenario ordering, units, anisotropic grids,
oblique beams, range shifters, focal modes, cutoff limits and matrix-free versus
matrix-based forward agreement. Include existing matRad fixtures and heterogeneous
cases; TG119 timing parity alone is insufficient to establish dose accuracy.
Use relative error with an absolute floor near zero and compare support as well
as values. Benchmark full calculations, memory capacity and cold preparation
alongside warmed kernel times.

The GUI invokes the public dose functions in
[`_workflow_widget.py`](../../pyRadPlan/gui/widgets/workflow/_workflow_widget.py)
and builds parameter forms from engine configuration. These proposals can begin
entirely below those interfaces. If configuration/data structures change, update
the GUI bindings and run `pytest test/gui`; no such production interface change
was made in this audit.

## Step 1: photon interpolator reuse

Initially implemented in `9a045eeb` on `refactor/dose-calc-performance` after audit commit
`f5becffd`. This integration is superseded by the reusable backend described below.
`PhotonPencilBeamSVDEngine._get_kernel_interpolators` initially constructed a SciPy or
CuPy `RegularGridInterpolator` once per convolved kernel. Its lifetime remains
the existing beam/ray lifetime: ordinary beamlets share beam interpolators, while
custom fluences and shaped fields receive their own. There is no global cache
or public API/model change. Other backends retain `partial(interpnd, ...)`.
Both paths retain `bounds_error=True`; interpolation order and precision are unchanged.

Six alternating pairs per case, using the same TG119 geometry and hardware as
the audit, produced these end-to-end median times. The legacy method is restored
in process by disabling prepared interpolators; all convolution and dose code is shared.

| Case | Backend | Legacy (s) | Reuse (s) | Speedup |
|---|---|---:|---:|---:|
| Unsampled influence | CuPy | 2.345 | 1.772 | **1.32×** |
| Sampled forward dose | CuPy | 2.563 | 2.001 | **1.28×** |
| Unsampled influence | NumPy | 1.279 | 1.234 | 1.04× |
| Sampled forward dose | NumPy | 1.495 | 1.469 | 1.02× |
| Single shaped-field forward dose | CuPy | 0.752 | 0.741 | 1.02× |
| Single shaped-field forward dose | NumPy | 1.056 | 1.055 | 1.00× |

All 36 comparisons had exactly identical dose outputs within their backend;
influence matrix indices and column pointers were also identical. The small CPU
and single-field differences should not be interpreted as robust speedups.
The main benefit is GPU evaluation across many beamlets.
See [paired records and summary](../../benchmark/pb_audit_results/interpolator_reuse/summary.json).
Here `baseline` means current production (reuse) and `variant` means legacy,
which is the reverse of the original prototype comparison.

Validation: six new regression cases passed across NumPy, CuPy, and strict Array
API, with float32/float64 inputs. They check construction counts, repeated queries,
different fluences, exact boundaries, out-of-bounds errors, and empty queries.
All six existing photon engine tests in `test_svdpb.py` and `test_svdpb_bio.py`
also passed, including the matRad dose fixture and shaped-field path. No public
data structure or calling API changed, and the GUI has no uses of these private
interpolators. The full repository suite was not run. Ruff passes for the new
tests and benchmark code; the engine retains the pre-existing `PLR0915` finding
in the unrelated `_init_beam` method.

Reproduce this follow-up:

```powershell
.venv/Scripts/python.exe -m pytest test/dose/test_svdpb_interpolators.py test/dose/test_svdpb.py test/dose/test_svdpb_bio.py -q
.venv/Scripts/python.exe benchmark/profile_photon_interpolator_reuse.py
```

The runner asserts exact parity and stores small JSON records. Large dose arrays
remain ignored. The original `prepared` audit variant now uses production code
unchanged; use `legacy_interpolators --paired` to compare against the old path.

## Reusable Array API interpolator

Backend dispatch now belongs to `core.xp_utils._interpolation` and the photon
engine constructs the public `ArrayBackendGridInterpolator`:

```python
from pyRadPlan.core.xp_utils import ArrayBackendGridInterpolator

interpolate = ArrayBackendGridInterpolator(
    (conv_grid, conv_grid), conv_mx, bounds_error=True, prefer_native=True
)
values = interpolate(query_points)
```

The object retains preparation for its lifetime; no global array cache is used.
Native SciPy/CuPy objects are constructed once. Generic Array API paths validate
and normalize grid ordering once. JAX uses a shared compiled evaluator with
grid/value arrays passed as arguments, avoiding compilation per object. Namespace
and device are resolved during construction; queries must stay on that namespace
and device. Inputs must remain unchanged while an object is used. A read-only
`implementation` property reports the selected implementation.

`prefer_native=False` forces the generic implementation (currently 2D/3D with at
least two points per axis). `bounds_error=False` still means clipping to the grid,
not native-library extrapolation or fill values. Native bounds checking is reused
when `bounds_error=True`, avoiding duplicate GPU synchronization. The historical
`compat.interpnd` import is preserved as a one-shot wrapper with the same options.
There are no interpolation-library imports or selection branches in the engine.

The initial comparison against the original implementation produced:

| Case | Backend | Original (s) | Backend object (s) | Speedup | Exact pairs |
|---|---|---:|---:|---:|---:|
| Unsampled influence | CuPy | 2.259 | 1.733 | **1.30×** | 6/6 |
| Sampled forward | CuPy | 2.524 | 1.933 | **1.31×** | **5/6** |
| Unsampled influence | NumPy | 1.243 | 1.212 | 1.03× | 6/6 |
| Sampled forward | NumPy | 1.439 | 1.380 | 1.04× | 6/6 |
| Single shaped-field forward | CuPy | 0.727 | 0.741 | 0.98× | 6/6 |
| Single shaped-field forward | NumPy | 1.019 | 1.019 | 1.00× | 6/6 |

These are six alternating pairs per configuration; all exact influence comparisons
also had identical sparse indices and column pointers. See the
[initial records](../../benchmark/pb_audit_results/array_backend_interpolator/summary.json).
Small CPU and single-field timing differences are not robust gains.

**Unresolved validation observation:** one sampled GPU forward pair differed by
0.00158686 in absolute dose, or 0.1316% of the reference peak. The other 35 pairs
were exact. This must not be described as universal bitwise parity. Follow-up
controls yielded 20/20 exact pairs using the original interpolation path on both
sides, and 6/6 using the new object on both sides. A subsequent 20-pair comparison
of the final object against the original path was entirely exact, with median
times 1.866 versus 2.426 s (**1.30×**). The outlier's cause remains unexplained;
these controls neither identify an existing engine bug nor prove the refactor
cannot contribute. No production tolerance was relaxed. The benchmark now retains
mismatching arrays locally and writes summary evidence before failing its exact
parity assertion. See [original-path controls](../../benchmark/pb_audit_results/array_backend_interpolator/legacy_repeatability/summary.json),
[new-path controls](../../benchmark/pb_audit_results/array_backend_interpolator/repeatability/summary.json),
and [20-pair recheck](../../benchmark/pb_audit_results/array_backend_interpolator/sampled_recheck/summary.json).

A final comparison with the previous **direct-native** engine integration produced
12/12 exact influence matrices. Median time differed by less than 0.8% in either
direction: the new object/reference ratios were 1.008 on NumPy and 0.993 on CuPy.
There was no material measured overhead from the final abstraction. Absolute
times were substantially higher for both implementations during this later run
(about 5.3 s NumPy and 9.1 s CuPy); its paired ratios must not be combined with
the earlier absolute timings. No lingering Python trace/profile hooks or coverage
environment variables were detected. See the
[direct-native control](../../benchmark/pb_audit_results/array_backend_interpolator/native_reference_final/summary.json).

The lifecycle benchmark now measures construction, repeated evaluation, and one-shot
calls separately, for one query and batches, native and generic, NumPy and CuPy.
`profile_interpnd_reference.py` freezes the original NumPy/CuPy interpolation path
and the direct-native integration from `9a045eeb`; changing production `interpnd`
therefore cannot silently change the reference. Raw timing samples stay local;
the [compact measurement record](../../benchmark/pb_audit_results/array_backend_interpolation_micro.json)
retains distribution statistics and environment metadata.
All 72 lifecycle cases passed their numerical checks; 24 unsupported/redundant
parameter combinations were skipped. In the final 2D single-query measurements,
native one-shot median overhead relative to the frozen reference was about +3.4%
on NumPy and -1.2% on CuPy. Retained-object evaluation took about 75% and 85% of
the respective one-shot reference times. Batch differences were small/noisy.
These microbenchmarks were run in the same slower runtime conditions as the final
direct-native comparison; they do not replace the paired full-dose measurements.

Validation covers native/fallback selection, construction reuse, fluence isolation,
descending/nonuniform axes, clipping and strict boundaries, empty queries, invalid
grids, mixed namespaces/devices, compatible query subclasses, and native 4D support.
The final core, compatibility, photon, and lazy-import test run passed **265 tests**
with 5 skips. The new module has **99% statement coverage** (the optional-library
ImportError handler was not executed). JAX float64 is explicitly enabled and restored
in the new tests. To avoid a local pydantic-settings startup error with pytest-cov,
settings were imported before starting pytest; the interpolation module was verified
not yet imported, so its coverage remained instrumented. The full repository suite
was not run. All **559 GUI tests** passed with the offscreen Qt platform; there are
no GUI uses of the changed private engine interpolators or `interpnd`. Existing
`interp1d` branch-count and photon `_init_beam` statement-count
lint findings remain unchanged.

Reproduce the backend measurements and dose checks:

```powershell
.venv/Scripts/python.exe benchmark/profile_grid_interpolation.py
.venv/Scripts/python.exe benchmark/profile_photon_interpolator_reuse.py --out benchmark/pb_audit_results/backend_recheck
.venv/Scripts/python.exe benchmark/profile_photon_interpolator_reuse.py --reference native --influence-only --out benchmark/pb_audit_results/native_recheck
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend cupy --mode photons --sampling --forward --variant legacy_interpolators --paired --pairs 20 --runs 1 --out benchmark/pb_audit_results/sampled_recheck
```

### Cached native versus cached generic evaluation

A focused follow-up compares **two prepared objects**, with `prefer_native=True`
and `False`. The caching-versus-reconstruction speedups above do not answer this
question. Historical `interpnd` already selected SciPy/CuPy directly; the separate,
more selective `interp1d` dispatch has not been changed.

`profile_interpnd_selection.py` captures the first photon convolution kernel and
its 121 actual beamlet query batches from the TG119 fixture. The grid is 782 by
782, axes and values are float64, and queries are float32. Smallest, median, and
largest batches contain 12,164, 13,063, and 13,133 points. Construction is excluded;
evaluation is warmed up, alternated for 15 pairs, and GPU-synchronized. These are
wall-clock interpolation timings, not full-dose speedups.

For the median-size batch:

| Backend | Bounds policy | Cached native (ms) | Cached generic (ms) | Native speedup |
|---|---|---:|---:|---:|
| NumPy | Raise (`True`, photon engine) | 0.444 | 1.854 | 4.18x |
| CuPy | Raise (`True`, photon engine) | 1.341 | 1.796 | 1.34x |
| NumPy | Clip (`False`) | 0.539 | 1.917 | 3.55x |
| CuPy | Clip (`False`) | 2.248 | 2.488 | 1.11x |

The smallest/largest batches give similar ratios: 4.03–4.24x on NumPy and
1.33–1.34x on CuPy with bounds checking. Every compared output passes
`rtol=atol=1e-12`; maximum absolute difference is 2.22e-16. The GPU here is an
RTX 4070 Laptop GPU, CuPy 13.3.0; NumPy is 2.3.5, Python 3.11.9. Native and generic
execute the same public bounds policy, including its per-call validation cost.

This supports keeping native preference for the current 2D photon workload. It
does **not** establish a universal native advantage: the earlier synthetic 3D
NumPy batch comparison was essentially tied. Backend selection should remain
inside the interpolation module, and any future automatic selection should be
justified by dimension/dtype/workload measurements. No production default or
numerical implementation was changed by this check.

See [NumPy results](../../benchmark/pb_audit_results/cached_native_generic/numpy.json)
and [CuPy results](../../benchmark/pb_audit_results/cached_native_generic/cupy.json).
The `capture/` records describe input generation only; their dose timings contain
instrumentation and must not be used to compare implementations.

```powershell
.venv/Scripts/python.exe benchmark/profile_interpnd_selection.py --backend numpy
.venv/Scripts/python.exe benchmark/profile_interpnd_selection.py --backend cupy
```

## Step 2 investigation: shared particle mask compaction

This follow-up isolates the indexing change in
`ParticlePencilBeamEngineAbstract._get_bixel_indices_on_ray`. The process-local
`compact_take` variant converts the mask once and gathers each field with:

```python
selected = xp.nonzero(mask)[0]
bixel["ix"] = xp.take(ray["ix"], selected, axis=0)
bixel["rad_depths"] = xp.take(ray["rad_depths"], selected, axis=0)
# Likewise for radial distances, optional lateral distances, and tissue fields.
```

The repeated kernel conversion in cutoff preparation is **unchanged**. So are
cutoffs, kernel interpolation, precision, dose accumulation, and sparse storage.
The earlier 1.18x combined-variant measurement must not be attributed to indexing
alone or subtracted from these timings to estimate the other change's contribution.

### Array API contract and dynamic shapes

The compatibility concern is real: older Array API versions did not require
integer-array bracket indexing. `take` has been standardized since 2022.12 and
explicitly supports selecting along one axis of a multidimensional array. Thus
`xp.take(..., axis=0)` also handles the `(N, 2)` lateral-distance field without
depending on NumPy's advanced-indexing semantics. See the
[2023.12 take specification](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.take.html).

`nonzero(mask)` still has a data-dependent output shape and can still synchronize
on a GPU. The benefit sought here is paying the compaction cost **once**, then
using its known-length index array for several gathers. It also introduces an
index-buffer allocation. This is not a static-shape/JIT solution; the
[nonzero specification](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.nonzero.html)
explicitly permits graph-based libraries to omit that operation.

Three-argument `where` can preserve fixed shapes, but a simple substitution would
leave rejected voxels in all downstream calculations and require changes to
mask propagation and sparse output. Fixed-capacity buffers with valid counts are
another architectural option. Neither is needed to evaluate this local change.

The full-dose experiment uses TG119, one proton beam, 72 rays and 908 beamlets,
with 8, 5 and 3 mm isotropic dose grids. Each NumPy/CuPy case runs in a fresh
process, with one warm-up and six alternating baseline/variant pairs. Only the
mask-application code is changed. Comparisons require identical dose values,
indices and column pointers. These are physical-dose influence measurements;
they do not establish biological-dose or other-modality performance.

### Full-dose results

Production baseline: `3ad8fae8`. Same Python 3.11.9 / NumPy 2.3.5 / CuPy 13.3.0
environment and RTX 4070 Laptop GPU as the preceding interpolation check.

| Backend | Grid (mm) | Selected voxels | Boolean median (s) | Shared take median (s) | Median paired speedup | Pair range |
|---|---:|---:|---:|---:|---:|---:|
| NumPy | 8 | 25,813 | 3.991 | 4.129 | 0.967x | 0.951–1.002x |
| NumPy | 5 | 108,871 | 4.522 | 4.463 | 0.999x | 0.966–1.048x |
| NumPy | 3 | 502,192 | 6.726 | 6.729 | 1.004x | 0.964–1.018x |
| CuPy | 8 | 25,813 | 11.719 | 10.667 | 1.102x | 1.080–1.128x |
| CuPy | 5 | 108,871 | 12.726 | 11.122 | 1.155x | 1.131–1.196x |
| CuPy | 3 | 502,192 | 13.794 | 12.373 | 1.123x | 1.101–1.147x |

Speedup is the median of the six within-pair baseline/variant ratios, not the
ratio of the two timing medians. The range is the observed minimum/maximum, not
a confidence interval. Absolute times were higher than the original audit;
use these paired comparisons rather than combining timings across sessions.

**All 36 pairs have bitwise-identical physical dose values and identical sparse
indices/column pointers.** The indexing-only change gives a consistent GPU
benefit across these voxel counts (roughly 9–13% less elapsed time), but no
convincing CPU benefit. The coarse CPU case is approximately 3% slower. These
measurements were captured before the production integration described below.

The [summary and individual pairs](../../benchmark/pb_audit_results/particle_indexing/summary.json)
retain configuration, timing samples, dimensions, and numerical comparisons.

### Isolated gathers and compatibility checks

The microbenchmark captures beamlets 0, 450 and 907 at 5 mm: 1,516–1,573 candidate
rows, 908–1,111 selected rows, and four fields (`ix`, `radial_dist_sq`, `rad_depths`,
and the two-column `lat_dists`). Each timed call includes `nonzero` for the
integer/take paths. There are 21 rounds with rotating execution order, warm-up,
exact output checks, and GPU synchronization. Input capture and host transfers
are outside the measured gather timings.

Representative beamlet 450:

| Backend | Repeated boolean (ms) | Shared integer brackets (ms) | Shared take (ms) |
|---|---:|---:|---:|
| NumPy | 0.0629 | 0.0552 | 0.0742 |
| CuPy | 2.8277 | 1.1551 | 1.0745 |

Across the three samples, GPU `take` is 2.4–2.6x faster than repeated boolean
indexing. Direct integer brackets have no consistent advantage over `take` on
CuPy. On NumPy, `take` is 10–19% slower than boolean indexing in these small
gathers; integer brackets are modestly faster, but this does not establish a
full-dose CPU speedup and would require handling older Array API versions and
multidimensional indexing carefully. See the
[NumPy gather record](../../benchmark/pb_audit_results/particle_indexing/micro/numpy.json)
and [CuPy gather record](../../benchmark/pb_audit_results/particle_indexing/micro/cupy.json).

`array_api_strict` verifies `take` against boolean selection for empty input,
empty selection, full selection and mixed selection, across API versions
2022.12, 2023.12, 2024.12 and 2025.12: **16 cases**, each covering integer/float
vectors and a two-column array. The `take` execution runs with boolean indexing
disabled and preserves shapes, values and dtypes. These are gather-level checks,
not a full strict-backend dose-engine test.

### Production integration

The shared particle engine now uses `nonzero` once and `take(..., axis=0)` for
every selected field on **all backends**. A single Array API implementation is
preferred over backend dispatch given the negligible full-dose CPU differences
at 3 and 5 mm and the small coarse-grid regression. The observed CPU tradeoff is
retained above; no CPU acceleration is claimed. Cutoff conversion, physics,
precision, and public calling interfaces are unchanged.

Regression tests exercise empty, rejected, full and mixed support; unsorted
global voxel IDs; constant and tabulated cutoffs; depth offsets; optional
two-column lateral distances and tissue fields; and preservation of shapes,
dtypes, devices and input arrays. NumPy, CuPy and strict Array API 2023.12/2025.12
are covered, with boolean indexing disabled on strict backends. Full-method
tests start at 2023.12 because the existing table interpolation uses `clip`;
the gather-only checks above continue to cover 2022.12.

The benchmark's `compact_take` variant now uses production unchanged.
`legacy_indexing` restores the old boolean selection for comparisons, while
`compact` retains the integer-bracket experiment. The runner keeps new records
in `particle_indexing/production/` and correctly labels boolean versus take
timings regardless of which is the current baseline.

Validation: **178 targeted tests pass**, including the new selection tests,
existing matRad particle dose fixtures (protons, helium, carbon and oxygen),
biological outputs, multi-Gaussian consistency and geometry. One additional
production-versus-boolean pair at 5 mm on each of NumPy/CuPy has exact dose and
sparse structure parity; these are integration checks, not new speedup estimates.
See the [production check](../../benchmark/pb_audit_results/particle_indexing/production/summary.json).
The full repository and GUI suites were not rerun: no public API or GUI binding
changed. The existing cutoff routine's Ruff statement-count finding is unchanged.

Reproduce sequentially, without overlapping benchmarks:

```powershell
.venv/Scripts/python.exe benchmark/profile_particle_indexing.py
.venv/Scripts/python.exe benchmark/profile_particle_indexing_micro.py --backend numpy
.venv/Scripts/python.exe benchmark/profile_particle_indexing_micro.py --backend cupy
```

## Cutoff preparation: conversion hoisting, batching and backend placement

These measurements describe the earlier CPU-default implementation. The current
[backend-native implementation](cutoff_backend_creation.md#production-integration)
uses the dose backend by default and offers `cutoff_force_numpy=True`.

Follow-up: [direct backend grid creation and GPU-resident kernel data](cutoff_backend_creation.md)
measures construction separately and tests its impact inside complete dose calculations.

Reference: `115e6ca8`, frozen in `benchmark/profile_cutoff_reference.py`, including
the original Hong physical-dose formulas. The new implementation keeps the
1,200 radial samples, 360 angles for `singleXY`, depth-selection procedure,
integration order, first-true cutoff rule, integral tolerance and compensation
convention. It does not substitute an analytic Gaussian cutoff or change physics.

### 1. Isolated conversion hoisting

Before refactoring, four alternating TG119 proton influence pairs per dose
backend compared only moving `base_kernel.to_namespace(np)` outside the depth
loop. The existing shared `take` indexing was present on both sides.

| Dose backend | Original median (s) | Hoisted median (s) | Median paired speedup | Pair range |
|---|---:|---:|---:|---:|
| NumPy | 0.913 | 0.903 | 1.024x | 0.945–1.062x |
| CuPy | 2.169 | 2.114 | 1.078x | 0.902–1.141x |

All eight pairs had identical dose values and sparse structure. These full-dose
differences are noisy; there is no robust standalone speedup claim. The isolated
preparation measurements below show where conversion cost is material. Original
records are under [hoist](../../benchmark/pb_audit_results/cutoff_preparation/hoist).

### 2. A common numerical evaluator

`CutoffQuadrature` retains the original radius-major sample order and prepares
the deterministic host geometry once per beam. It can transfer that geometry to
an Array API namespace once. `evaluate_cutoff` processes bounded depth batches,
integrates each profile, and selects radii/compensation using Array API operations.
Its callback leaves physical-dose evaluation with the engine.

Hong now shares `_lateral_profile` between normal dose evaluation and cutoff
preparation. The cutoff path interpolates IDD and lateral parameters once per
depth, broadcasts them over samples, and skips unused LET/biological interpolation
and evaluation. The energy kernel is converted once, outside the depth loop.
The base engine retains
a scalar-profile fallback; Hong subclasses overriding physical dose or kernel
interpolation use that fallback unless they explicitly provide a batched hook.
Future engines can supply `_cutoff_dose_profile` without duplicating quadrature
or backend-specific integration code.

The default batch limit is 1,000,000 depth-by-sample entries, with a minimum of
one complete depth. It is a per-array sizing rule, not a total-memory guarantee:
component count and live physics intermediates also consume memory. It fits all
35 isotropic depths in one batch and two focused depths per batch. A packed
`(n_depths, 3)` result permits one host transfer per energy for the existing
NumPy-backed machine model. Full cutoff (`1.0`) skips quadrature and conversion.

The historical `relative` method overwrites a scalar compensation factor at
each depth. Integration deliberately retains the final depth's value. Likewise,
an all-false threshold comparison retains the original argmax index-zero result.
Changing either convention would require a separate numerical-model change.

### 3. NumPy/CuPy measurements, including preparation and transfers

Each call constructs quadrature, prepares/uploads the kernel and depth arrays,
evaluates 35 depths, and downloads the small output. CUDA initialization, JIT and
allocation pools are warmed; prepared input arrays are **not** reused between
calls. Five rotating-order rounds were measured for integral cutoffs. Timings
are for one energy, not the complete dose engine. The full engine can reuse
quadrature across multiple energies, unlike this deliberately self-contained
single-energy comparison.

| Lateral model | Original (ms) | Hoisted only (ms) | Batched NumPy (ms) | Batched CuPy (ms) |
|---|---:|---:|---:|---:|
| Single Gaussian | 4.997 | 1.980 | **0.783** | 2.121 |
| Double Gaussian | 5.508 | 2.877 | **1.169** | 2.664 |
| Multi Gaussian, synthetic | 15.964 | 12.517 | **1.667** | 6.064 |
| `singleXY`, synthetic | 461.348 | 474.802 | 228.386 | **51.139** |

Single/double cases use an actual Generic proton kernel at 162.933 MeV. No
shipped Generic machine exposed a multi-Gaussian or focused kernel, so those
cases explicitly extend that kernel with three Gaussian components or unequal
x/y sigmas. They establish computational behavior, not validation of a clinical
focused machine. In `singleXY`, CuPy is about **4.47x faster than batched NumPy**
and **9.02x faster than the original**, with transfers included. Isotropic cases
favor batched NumPy by roughly 2.3–3.6x over CuPy.

NumPy outputs matched the frozen reference exactly for all measured models.
CuPy selected the same radius bins in both integral and relative measurements;
the largest difference in the relative compensation was 7.5e-14, and in the
reported integral-error percentage 8.5e-12. GPU cumulative reductions are not
bitwise identical to NumPy, so other inputs near a bin threshold can select a
neighboring radius. The benchmark requires exact radius selection for these
cases and `rtol=atol=1e-10` for compensation/error. Independent scalar-formula
tests use tighter tolerances. No production dose tolerance was relaxed.

Three additional rotating-order rounds for `relative` cutoffs show the same
placement trend. The records include separate preparation, evaluation and
download timings: [integral](../../benchmark/pb_audit_results/cutoff_preparation/preparation_integral.json),
[relative](../../benchmark/pb_audit_results/cutoff_preparation/preparation_relative.json).

### Full-dose impact and execution policy

Three alternating pairs per case compared the complete current engine against
the frozen original cutoff and physical-dose methods:

| Modality | Dose backend | Current median (s) | Original median (s) | Median paired speedup |
|---|---|---:|---:|---:|
| Protons | NumPy | 0.844 | 0.960 | 1.073x |
| Protons | CuPy | 1.820 | 1.865 | 1.032x |
| VHEE | NumPy | 0.969 | 0.972 | 0.996x |
| VHEE | CuPy | 1.029 | 1.048 | 1.032x |

VHEE changes are within observed timing noise; preparation gains do not imply
proportional engine gains. The single-energy synthetic focused speedup should
not be presented as a full VHEE speedup. Paired speedups are medians of the
within-pair ratios, not ratios of the displayed timing medians.

Six more pairs put the new cutoff evaluator on CuPy instead of NumPy while dose
remained on CuPy. They showed no convincing full-engine advantage for these
isotropic proton/VHEE fixtures. **All 18 full-dose pairs had bitwise-identical
physical dose values, indices and column pointers.** See the
[full-dose records](../../benchmark/pb_audit_results/cutoff_preparation/full_dose/summary.json).

NumPy remains the default preparation backend. The new plan/engine option
`cutoff_backend="dose"` chooses the dose namespace and device while using the
same numerical evaluator:

```python
pln.prop_dose_calc["cutoff_backend"] = "dose"  # e.g. CuPy when dose uses CuPy
```

There is no automatic threshold policy based on the synthetic focused case.
The option fits the existing generated GUI configuration form (Literal choices);
no widget-specific dispatch is introduced.

Validation: **262 targeted tests** and **559 GUI tests** pass. The new numerical
module has **100% statement coverage** from a clean 84-test numerical coverage
run. Tests cover all lateral models, both cutoff methods, multiple batch sizes,
empty/full cutoff, invalid physical dose, sample order, custom engine fallbacks,
placement configuration, and existing matRad/biological-dose regressions. An
initial broad instrumented run encountered sparse-matrix validation errors while
loading fixtures after pre-importing settings; the normal functional run passed,
and coverage was collected separately on the numerical tests. The full
repository suite was not run. All changed production modules pass Ruff.

Reproduce sequentially:

```powershell
.venv/Scripts/python.exe benchmark/profile_particle_cutoff.py
.venv/Scripts/python.exe benchmark/profile_particle_cutoff.py --method relative --rounds 3
.venv/Scripts/python.exe benchmark/profile_cutoff_suite.py
.venv/Scripts/python.exe -m pytest test/dose/test_particle_cutoff.py test/dose/test_particle_indexing.py test/dose/test_hongpb.py test/dose/test_hongpb_bio.py test/dose/test_base_pencilbeam.py test/dose/test_pb_geodist.py -q
```

The frozen `make_hoisted_cutoff()` remains available for standalone reproduction
of step 1; the audit's `hoist` variant now compares that frozen hoisted path
against current production. The earlier standalone numbers above were captured
before the batched implementation, so those references remain independent.

## Reproduce and inspect

Run from the repository root with the existing environment. All patches below
are installed only in the benchmark process and disappear on exit:

```powershell
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend numpy --mode protons --profile
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend cupy --mode protons --variant combined --paired
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend cupy --mode photons --variant prepared --paired
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend numpy --mode photons --sampling --variant scattermax --paired
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend cupy --mode protons --forward --variant device_forward --paired
.venv/Scripts/python.exe benchmark/benchmark_pb_architecture_audit.py
.venv/Scripts/python.exe benchmark/profile_pb_audit_summarize.py
```

[`profile_pb_audit.py`](../../benchmark/profile_pb_audit.py) contains the exact
prototype changes and timing/validation logic.
[`profile_pb_audit_suite.py`](../../benchmark/profile_pb_audit_suite.py) runs
cases sequentially; its default, `--followup`, and `--final` groups cover the
reported cases. Run groups sequentially, without overlapping CPU/GPU benchmarks.
[`benchmark_pb_architecture_audit.py`](../../benchmark/benchmark_pb_architecture_audit.py)
contains the synthetic schedule and primitive comparisons.

Raw configuration, repetitions, separate profile/stage results, paired comparisons,
and the aggregate [`summary.json`](../../benchmark/pb_audit_results/summary.json)
are under [`benchmark/pb_audit_results`](../../benchmark/pb_audit_results).
Generated matrices/images and binary profiles remain local and are ignored in
that folder. The original audit changed no production engine or public API. Its validation consists of
executed benchmark comparisons and lint checks, not a claim that a production
refactor has passed the complete test suite.

See [shared photon sampling and interpolation reductions](shared_photon_hotpaths.md)
for the production grouped maximum and multi-value interpolation follow-up.

See [compact sparse indices and device collection](sparse_output_performance.md)
for safe int32 storage, GPU construction measurements and the updated profile.
