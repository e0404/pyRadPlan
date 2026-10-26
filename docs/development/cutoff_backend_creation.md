# Cutoff grid creation and GPU-resident kernel data

This follow-up to the [pencil-beam performance audit](pencil_beam_performance_audit.md)
tests grid creation inside complete dose calculations and separates it from
kernel placement and cutoff evaluation. The experiment uses process-local
patches only; the original experiment did not change production code. The later
[production integration](#production-integration) is documented below.

## What is measured

`benchmark/profile_cutoff_creation.py` compares the current NumPy grid factory
plus transfer with a prototype that constructs the same sampling directly in
the selected Array API namespace. The prototype uses `linspace`, powers,
broadcasting and reshaping, retaining the historical radius-major sample order.
It also avoids some materialization through broadcasting, so this is a practical
alternative implementation, not a pure substitution of `np` with `cp`.

The isolated measurements use 35 depths and already-prepared kernel and depth
arrays. They time grid construction, numerical evaluation, and result download
separately, with CUDA synchronization at stage boundaries. Each timing includes
fresh grid construction; prepared quadrature is not cached across repetitions.
Single/double Gaussian cases use a real Generic proton kernel; the anisotropic
case is explicitly synthetic. Fifteen rotating-order rounds follow warm-up.
CUDA initialization and cold compilation are excluded.

Full-dose comparisons run the actual TG119 influence calculation on CuPy, with
5 mm dose voxels and 10 mm bixel spacing. The proton and VHEE cases use one and
three beams; carbon is also checked with one beam. Each policy gets one warm-up
and four measured runs, with rotating policy order. A separate instrumented run
records cutoff and conversion costs;
those extra synchronization points are absent from full-dose timing runs.
Quadrature is created once per beam, not once per energy or beamlet.

Seven policies are compared:

| Policy | Cutoff execution | Grid creation | Kernel preparation |
|---|---|---|---|
| `cpu` | NumPy | NumPy | Current lazy conversion |
| `gpu_host` | CuPy | NumPy then upload | Current lazy conversion |
| `gpu_native` | CuPy | Direct CuPy | Current lazy conversion |
| `resident_cpu_mirror` | NumPy | NumPy | CPU mirror for cutoffs; resident GPU arrays for dose |
| `resident_cpu_download` | NumPy | NumPy | Download resident GPU kernel arrays for cutoffs |
| `resident_gpu_host` | CuPy | NumPy then upload | Resident GPU arrays for cutoffs and dose |
| `resident_gpu_native` | CuPy | Direct CuPy | Resident GPU arrays for cutoffs and dose |

Resident variants pre-upload **every energy** in the machine database, then share
those immutable arrays across cutoff preparation and dose evaluation. Upload
cost is recorded separately and excluded from already-resident full-dose times.
For a first-use comparison, that setup cost must be added. This measures the
existing per-field upload interface, not a proposed packed database upload.
Beam-dependent cutoff tables are refreshed instead of being reused from the
database cache.

The existing CPU machine model remains available for validation, energy/depth
selection and serialization in every variant. `resident_cpu_download` forces
kernel-array downloads for profile evaluation to test that additional cost, but
does **not** represent a completely device-only machine model. A future engine
that also moves depth selection and cutoff consumption onto the GPU could avoid
additional transfers that this prototype retains. The prototype also retains the
current per-energy host-model deep copy; it only replaces kernel-array conversion
with cache reuse. Generic namespace/device checks on the cached arrays also
remain; a dedicated cache lookup could remove that small Python overhead.

## Isolated construction and evaluation

Median milliseconds, with kernels and query depths already resident:

| Model | NumPy creation | NumPy creation + GPU upload | Direct CuPy creation | NumPy evaluation | CuPy evaluation, native grid |
|---|---:|---:|---:|---:|---:|
| Single Gaussian | 0.109 | 0.346 | 0.288 | 0.484 | 0.893 |
| Double Gaussian | 0.105 | 0.341 | 0.285 | 0.735 | 1.152 |
| Anisotropic, synthetic | 11.489 | 12.105 | 0.965 | 212.977 | 36.759 |

Direct CuPy creation saves only about 0.06 ms per isotropic grid. For the much
larger anisotropic grid, it saves about 11 ms, roughly a 12.5-fold construction
speedup. Its complete single-energy cutoff preparation/evaluation/download falls
from 50.15 ms to 37.69 ms. That 1.33-fold improvement is specific to this
self-contained synthetic case: full engines reuse a grid across energies, so
the grid-creation saving is per beam, not per energy.

Even without kernel upload, NumPy evaluates the small isotropic grids faster.
Thus faster GPU grid creation alone does not establish that cutoff evaluation
should run on the GPU. Conversely, the large anisotropic workload benefits from
GPU evaluation and should not need to construct its expanded grid on the CPU.

Direct backend construction is not bitwise identical across libraries. The
benchmark checks all geometry arrays and cutoff results with tight tolerances;
the full-dose comparisons separately check sparse structure and dose values.

## Complete dose calculations

Median seconds; all dose calculations use CuPy. These columns vary cutoff
placement and grid construction while retaining lazy kernel conversion.

| Modality | Beams | NumPy cutoff | CuPy cutoff, host grid | CuPy cutoff, native grid |
|---|---:|---:|---:|---:|
| protons | 1 | 1.794 | 1.859 | 1.864 |
| protons | 3 | 4.376 | 4.490 | 4.492 |
| VHEE | 1 | 1.136 | 1.153 | 1.134 |
| VHEE | 3 | 2.250 | 2.292 | 2.299 |
| carbon | 1 | 2.197 | 2.172 | 2.188 |

For the same workloads with the entire kernel database already on the GPU:

| Modality | Beams | CPU, host mirror | CPU, kernel download | GPU, host grid | GPU, native grid |
|---|---:|---:|---:|---:|---:|
| protons | 1 | 1.836 | 1.833 | 1.850 | 1.822 |
| protons | 3 | 4.428 | 4.376 | 4.450 | 4.482 |
| VHEE | 1 | 1.138 | 1.121 | 1.148 | 1.180 |
| VHEE | 3 | 2.255 | 2.247 | 2.253 | 2.282 |
| carbon | 1 | 2.133 | 2.154 | 2.166 | 2.204 |

Differences from changing grid creation are small, inconsistent, and overlap
the observed run-to-run ranges. For example, the one-beam proton host-grid
GPU policy ranged from 1.830 to 1.923 s, while native creation ranged from
1.756 to 1.924 s. The three-beam medians differ by only 2 ms. These repetitions
do not establish a full-dose speedup from direct construction or full-database
residency for the tested isotropic workloads.

The separate synchronized stage measurements support distinguishing creation
from evaluation: one-beam proton cutoff preparation took about 17 ms on CPU
versus 36 ms on GPU, including depth selection, kernel preparation and output
handling. With resident kernels, the corresponding measurements were about
20 ms (CPU mirror) and 35 ms (GPU). For carbon they were about 21 ms with a
CPU mirror, 33 ms with forced kernel downloads, and 51 ms on GPU. These are
single diagnostic runs with extra synchronization, not standalone speedup
estimates. Grid creation itself accounts for a small fraction of these costs.

Whole-database upload, measured separately through the existing conversion API:

| Database | Energies | Kernel array storage (MB, decimal) | Median warmed upload (ms), one-beam setup |
|---|---:|---:|---:|
| protons | 114 | 1.502 | 35.526 |
| VHEE | 3 | 0.072 | 0.738 |
| carbon | 121 | 0.919 | 39.243 |

Storage counts flat numerical kernel arrays, excluding Python objects and
beam-dependent cutoff tables. Carbon has more energy entries here but a smaller
array payload than protons. Resident-cache setup is not free, and uploading
unused energies through many separate allocations/copies can outweigh savings
for a single calculation. Repeated calculations can amortize that setup.

All **140 measured full-dose runs** matched the reference dose values and sparse
indices/column pointers exactly. The largest direct-CuPy radius difference in
the isolated geometry comparison was approximately 1.14e-13 mm; the largest
cutoff-result difference was 8.46e-12 in the reported integral-error percentage.
There were no observed cutoff-bin changes. This is empirical validation of the
tested inputs, not a guarantee that roundoff cannot affect a threshold on other
inputs. Production tolerances and numerical code were not changed.

## Implications

- Make grid creation able to use the chosen execution namespace. Its cost does
  not justify forcing NumPy construction for small grids, and native creation
  helps the larger anisotropic case substantially.
- Treat a NumPy fallback for cutoff **evaluation** as a separate policy. Small
  isotropic workloads still favor CPU evaluation even with resident GPU kernels;
  grid creation is not the cause of that difference.
- Do not require a host round trip when a future implementation already has
  suitable device arrays. These experiments preserve the CPU machine model and
  depth-selection path, so they cannot settle placement for a fully resident
  architecture or for larger/finer sampling workloads.
- A private backend-aware factory with configurable sampling would fit these
  findings. There is no measured need for separate per-library quadrature
  implementations or a new general backend-selection interface.

Only the benchmark, its records, changelog and documentation were changed.
The full benchmark completed successfully and Ruff checks passed; no production
unit-test suite was rerun for these process-local experiments.

## Reproduction and records

Run sequentially to avoid competing GPU work:

```powershell
.venv/Scripts/python.exe benchmark/profile_cutoff_creation.py --rounds 4 --micro-rounds 15
.venv/Scripts/python.exe benchmark/profile_cutoff_creation.py --modes carbon --beams 1 --rounds 4 --micro-rounds 5 --out benchmark/pb_audit_results/cutoff_creation/carbon.json
```

The script records raw repetitions, timing medians, geometry/cutoff differences,
full-dose differences, database upload times, array storage, and instrumented
stages in [results.json](../../benchmark/pb_audit_results/cutoff_creation/results.json)
and [carbon.json](../../benchmark/pb_audit_results/cutoff_creation/carbon.json).
The second command repeats the same proton microbenchmarks as a consistency
check; its full-dose calculation uses the carbon machine. The isolated table
above uses the first command's 15-round measurements.

The measured GPU is an NVIDIA GeForce RTX 4070 Laptop GPU, with Python 3.11.9,
NumPy 2.3.5 and CuPy 13.3.0. Results describe these workloads and this hardware;
they do not establish a universal CPU/GPU dispatch threshold.

## Production integration

Cutoff sampling creation and batched evaluation now follow the dose namespace
and device by default. `cutoff_force_numpy=True` overrides both together. The
unreleased `cutoff_backend` selector has been replaced, not extended with more
backend choices. The GUI exposes the new boolean through its generated form.
Depth selection and serialized machine cutoff tables remain on the host.

`CutoffQuadrature` remains a frozen dataclass because it keeps the sampling
definition and the derived arrays together across depth batches. Its factory
accepts custom bin boundaries and angular resolution:

```python
from pyRadPlan.dose.engines._particle_cutoff import CutoffQuadrature

quadrature = CutoffQuadrature.create(
    "singleXY",
    xp=xp,
    device=device,
    radial_edges=[0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    n_angles=180,
)
```

The example illustrates the API, not a recommended dose-accuracy configuration.
The production engine retains the existing 1,200 radial bins and 360 angular
samples. Angular sampling is only used for `singleXY`; isotropic quadratures
record `n_angles=1` and integrate the angular factor analytically. The class
retains `radial_edges`, `n_angles`, and `lateral_model` alongside its integration
arrays. Array-valued custom edges can supply the namespace/device implicitly;
explicit `xp`/`device` remain available. Custom boundaries are copied and checked
for finite, strictly increasing values starting at zero. Change sampling by
creating a new quadrature; frozen fields do not make array contents immutable.

No host grid is constructed when a GPU namespace is requested. Default creation
uses Array API operations, including broadcast/reshape for radius-major angular
sampling. Custom host edges need an initial upload; device-resident edges need no host copy of the edge array, though validation
synchronizes scalar checks. The public engine configuration does not expose sampling
resolution yet; the configurable factory is an internal engine-building API.

The historical host factory is retained independently in the benchmark reference
module. `profile_particle_cutoff.py` now measures production-native creation;
`profile_cutoff_suite.py` compares the current native default against both the
frozen original and the NumPy override for protons, VHEE, and carbon. Historical
result files above have not been overwritten.

Production validation: the 304-test dose regression run passed, followed by a
129-test numerical coverage run including three additional cross-backend edge
cases. `_particle_cutoff.py` retains **100% statement coverage**. All **560 GUI
tests** passed, including the new boolean-editor round trip. Strict Array API
coverage includes a non-default logical device; the host-selected initial spot
variance is converted to a Python float before backend arithmetic. The full
repository test suite was not run.

All **27 full-dose pairs** matched dose values, sparse indices and column
pointers exactly: three pairs each for NumPy/CuPy versus the frozen original,
and three CuPy pairs versus the NumPy override, for all three modalities. See
[production full-dose records](../../benchmark/pb_audit_results/cutoff_native/full_dose/summary.json).
Timing noise remains material; the change makes placement follow the chosen
backend and does not claim GPU execution is faster for every machine.

Current integral preparation medians, including grid construction, kernel/depth
uploads and result download (five rounds, 35 depths per energy):

| Lateral model | Frozen original (ms) | Native NumPy (ms) | Native CuPy (ms) |
|---|---:|---:|---:|
| single | 5.024 | 0.851 | 2.436 |
| double | 6.243 | 1.155 | 2.665 |
| multi (synthetic) | 16.770 | 1.765 | 6.064 |
| singleXY (synthetic) | 508.779 | 235.148 | 38.663 |

The relative-cutoff benchmark also passes. Native CuPy radii differ from the
host reference by at most about 2.2e-14 mm in these preparation cases; no radius
bin changes were observed. Numerical checks allow only rounding-scale radius
differences, while production dose comparisons still require exact results.
The records are [integral](../../benchmark/pb_audit_results/cutoff_native/preparation_integral.json)
and [relative](../../benchmark/pb_audit_results/cutoff_native/preparation_relative.json).

```powershell
.venv/Scripts/python.exe benchmark/profile_particle_cutoff.py --out benchmark/pb_audit_results/cutoff_native
.venv/Scripts/python.exe benchmark/profile_particle_cutoff.py --method relative --rounds 3 --out benchmark/pb_audit_results/cutoff_native
.venv/Scripts/python.exe benchmark/profile_cutoff_suite.py --pairs 3
```
