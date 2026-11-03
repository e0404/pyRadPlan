# Bounded pencil-beam batching: first CPU experiments

2026-11-03, starting from `530209cc` on `refactor/dose-calc-performance`.
This follow-up investigates batching after prepared particle interpolation.
Production engine code and dispatch are unchanged. GPU testing and the full
baseline against `develop` remain deferred.

The first result is that batching **physics alone** gives modest PyTorch CPU
improvements, but no general NumPy benefit. The next useful production boundary
is an explicit bounded work-packet interface; larger GPU gains will likely need
support selection, interpolation and collection to consume those packets too.
That GPU expectation is an architectural inference, not a measured speedup.

## What is implemented in the experiment

[`benchmark_particle_batches.py`](../../benchmark/benchmark_particle_batches.py)
contains a working, CPU-only Hong-engine adapter with three strategies:

1. **Validation-only (`checks`)**: calculate physical dose and LET per beamlet,
   concatenate physical dose, then perform one host-visible validity decision
   for the packet. The NaN/negative-dose policy remains unchanged.
2. **Packed (`packed`)**: concatenate already selected voxel contributions and
   interpolated fields, evaluate the existing lateral model and LET calculation
   over that vector, check validity once, then split views for the collector.
3. **Padded (`padded`)**: retain a beamlet axis, pad to the largest selected voxel
   count, and broadcast scalar spot widths and compensation factors. Inactive
   outputs are masked before validation. Buffers are preallocated rather than
   repeatedly concatenating padding. Single-item batches use scalar physics.

All three retain the existing support selection and prepared interpolation **per
beamlet**. Biological evaluation also stays per beamlet. The experiment therefore
isolates the downstream physics/checking opportunity; it does not yet batch the
two most important preparation components or remove per-column output transfers.

Packets contain initialized bixels and their existing collector arguments. They
preserve beam, ray, beamlet, scenario and delivery-column identities. Work can
span adjacent rays and range/CT scenario slots within a beam/setup group without
reordering output writes. Packets are flushed before the next beam changes its
cutoff database, before a setup-group transition, and before finalization.
Contiguous incompatible field schemas (e.g. different Gaussian component counts)
split into separate evaluations.

The scheduler bounds both beamlet count and voxel-pair capacity. Packed capacity
is the sum of selected lengths; padded capacity is batch size times maximum
length. A single beamlet exceeding the budget uses the original scalar path,
without additional packed copies. The experiment reports accounted storage for
named batch arrays; this is **not** an allocator peak or the whole engine's memory
usage. It excludes original prepared bixels, interpolation arrays, small metadata
vectors and expression temporaries. Production should use a byte budget derived
from the engine's field schema and workspace requirements, rather than treating a pair cap as a universal
memory bound.

The adapter delays `_compute_bixel` physics until `_fill_dij` submits a bounded
packet. This is intentionally confined to the benchmark. It would be a poor
production contract: `_compute_bixel` should continue to mean a completed bixel
for existing engines and extensions.

## Architecture implications

### Common infrastructure

An explicit prepare/evaluate/collect interface is a better production boundary.
Keep the current single-bixel path as the default implementation; engines opt
into a batch evaluator when their preparation and physics support it.

```python
# Illustrative contract, not a new public API in this change.
for packet in bounded_work_packets(beam, scenarios, memory_budget):
    prepared = engine.prepare_batch(packet)
    quantities = engine.evaluate_batch(prepared)
    collector.add_batch(packet.destinations, prepared.indices, quantities)
```

The reusable scheduler and collector should own:

- bounded storage and flushing at machine/geometry-cache boundaries;
- explicit contribution identity: scenario slot, beam/ray, delivery column,
  optional parent column and contribution weight;
- output ordering, empty columns, quantity names, validity-check boundaries and
  the scalar fallback;
- collection into shared sparse structure or forward scoring, preserving the
  existing accumulation order where numerical reproducibility requires it.

The engine should own compatibility grouping, preparation, interpolation needs,
workspace estimates, and physical/biological evaluation. Backend dispatch stays
in numerical helpers, including the interpolation backend. This avoids requiring
photons, particles and future AAA implementations to share one physics formula
or one tensor layout.

**Do not make “one ray” the universal batch boundary.** The TG119 proton case has
908 beamlets on 72 rays, so energy batching within a ray is useful. The Generic
VHEE case has one beamlet per ray. Photon pencil beams commonly have the same
problem. Adjacent-ray packets are therefore necessary for shared infrastructure;
within-ray batching is an engine-specific optimization. Photon sampling must
also retain its existing random draw order; arbitrary regrouping is unsafe.

Future fine-sampled particle children should carry a parent delivery-column ID
and a reduction weight. They should not become independent deliverable columns.
The collector would need an explicit child-to-parent reduction, rather than the
current append-one-column operation. AAA can reuse scheduling and collection
while exposing its own transport components and workspace policy.

Reserve memory before preparing the next packet, including any look-ahead bixel
and temporary arrays. Add progress/cancellation checks at packet boundaries.
Keep scalar execution available for small CPU workloads and engines that have
not implemented the batch contract. Do not add engine-name checks to the shared
scheduler or silently bypass subclass overrides.

### Particle-specific next stages

The current prototypes still perform one cutoff interpolation, compaction and
depth interpolation for every beamlet. A more complete batch needs:

1. **Batch support selection on a shared ray.** Broadcast ray depth/radial
   coordinates against a bounded group of energies. Compact the combined mask
   once, yielding `(beamlet_owner, ray_voxel)` pairs. Preserve the exact existing
   cutoff predicates and grid-offset arithmetic.
2. **Grouped interpolation in `xp_utils`.** Evaluate several prepared 1D tables
   against owner-tagged queries without an engine-side interpolation fast path.
   Tables can have different depth grids and lengths. A portable segmented
   binary search is one candidate; native per-table NumPy interpolation remains
   an important CPU fallback. Compare complete preparation plus evaluation cost.
3. **Direct batch collection.** Preserve packed quantity arrays and segment
   offsets through collection. Splitting into bixel views and downloading each
   column individually, as this prototype does, leaves GPU transfer overhead.

Those stages can feed either packed or padded physics. Packed work avoids
inactive evaluations; padding can avoid repeating per-beamlet scalar metadata.
The best choice depends on occupancy, field count, device and batch size.
No universal layout or GPU speedup is established by the CPU experiment.

## CPU measurements

TG119, Generic machines, one beam, 5 mm grid, 10 mm bixel width, influence
calculation. Protons use single-Gaussian physics with LET; VHEE uses 200 MeV and
double-Gaussian physics without LET. Batch limit 16, pair capacity 65,536.
NumPy 2.3.5; PyTorch 2.13.0+cu126 on CPU with its default 24 threads. The CPU
reports Intel64 Family 6 Model 183 Stepping 1. Other workstation activity was not
controlled; the user's separate GPU task may also consume CPU resources.

Runtime changes below are ratios of the separate median full-dose times from
five alternating pairs (negative is faster). NumPy proton validation-only uses
a nine-pair repeat because its initial batch was noisy.

| Backend / workload | Validation-only | Packed physics | Padded physics |
|---|---:|---:|---:|
| NumPy / protons | -1.9% | +9.0% | +1.0% |
| PyTorch CPU / protons | -5.8% | -8.3% | -6.4% |
| NumPy / VHEE | -0.7% | +2.6% | +3.6% |
| PyTorch CPU / VHEE | -5.8% | -0.7% | -2.6% |

For example, PyTorch proton packed evaluation was 2.183 → 2.003 s; padded was
2.133 → 1.996 s. Both were faster in all five corresponding pairs. In contrast,
the proton validation-only pairs varied substantially: its PyTorch median
*paired* change was only -0.8%, although the ratio of separate medians was -5.8%.
Treat small changes as inconclusive and do not choose a universal CPU policy from
this table.

The initial NumPy proton validation-only run was 0.592 → 0.515 s (-13%); the
longer repeat was 0.483 → 0.474 s (-1.9%). Both raw runs are retained. A one-pair
control with every bixel forced back to scalar physics also showed a large timing
difference, reinforcing that one-off changes are not reliable speedups here.

Scheduling and accounted batch storage:

| Case | Beamlets → packets | Padded evaluated / selected pairs | Packed peak named arrays | Padded peak named arrays |
|---|---:|---:|---:|---:|
| Protons | 908 → 57 | 1.113× | 1,212,780 bytes | 1,008,528 bytes |
| Generic VHEE | 121 → 8 | 1.005× | 3,133,916 bytes | 2,453,520 bytes |

Padded arrays use less accounted memory in these cases because scalar spot widths
and compensation factors broadcast. Padding is not automatically the larger
representation. These are named physics arrays, not measured process/device
memory peaks. Restricting VHEE to one ray per packet produced 121 singleton
packets: no opportunity to batch its physics.

Physical-dose validity decisions drop from two per beamlet in the current valid
scalar path to one per packet: 1816 → 57 for this proton case, and 242 → 8 for
Generic VHEE. This count excludes support compaction, biological validation and
other host-visible operations. The validation prototype also combines the NaN
and negative predicates, so its timing is not an isolated measure of grouping
alone. GPU synchronization reductions have not been timed.

Raw timings and metadata are in
[`benchmark/pb_audit_results/batching_cpu/`](../../benchmark/pb_audit_results/batching_cpu/).
The `validation/`, `cap4096/` and `cap1/` runs are correctness checks with one
pair each, not performance evidence. All final full-dose comparisons had
**zero maximum absolute difference**, including LET, biological matrices and
weighted forward images; sparse structures and delivery identities also matched.

## Validation and reproducibility

The harness alternates scalar/batch order, warms both variants, and creates a
fresh engine for every full-dose measurement. Timings include support selection,
interpolation, packing, physics, checks, collection and setup. Small diagnostic
timers are included in the prototype timings. NumPy and PyTorch CPU use the same
NumPy-generated steering data. CUDA is hidden inside the harness as well as in
the invoking environment. Steering generation and Python engine construction are
outside the recorded dose interval.

`--self-check` passed 288 synthetic checks across all four lateral models on
NumPy, PyTorch CPU and strict Array API arrays, float32/float64, ragged/empty
batches, LET, and rejection of active NaN/negative dose. It also checks a
nontrivial Python-float compensation factor: materializing
this scalar with PyTorch's default float32 would silently lose precision, so the
batch uses the lateral result's dtype.

Additional completed full-dose checks cover two beams, setup/range scenarios,
Wedenberg biological influence matrices with CT-grid tracing, nonuniform weighted
forward dose with scenarios, and the Focused VHEE machine's anisotropic model.
The 4096-pair cap produced maxima of 4092 packed pairs and 4088 padded pairs;
a cap of one correctly sent all 908 proton bixels through the scalar fallback.
Pending work and metrics reset when a calculation starts, including engine reuse
after an interrupted run.

Full influence comparisons verify all quantities/scenario slots, sparse indices
and column pointers, and beam/ray/bixel numbering. Forward comparisons verify
result images and their metadata. Multiple CT phases, derived scenario aliases,
schema changes, custom engines, variable biological components and future
transport models still require dedicated tests before any production integration.

```powershell
.venv/Scripts/python.exe benchmark/benchmark_particle_batches.py --self-check
.venv/Scripts/python.exe benchmark/benchmark_particle_batches.py --backend numpy --sizes 16
.venv/Scripts/python.exe benchmark/benchmark_particle_batches.py --backend torch --sizes 16
.venv/Scripts/python.exe benchmark/benchmark_particle_batches.py --backend numpy --mode VHEE --sizes 16
.venv/Scripts/python.exe benchmark/benchmark_particle_batches.py --backend torch --mode VHEE --sizes 16
# Restrict scheduling to a ray to quantify lost batching opportunities:
.venv/Scripts/python.exe benchmark/benchmark_particle_batches.py --backend numpy --mode VHEE --scope ray --layouts packed --sizes 16
# Additional correctness checks:
.venv/Scripts/python.exe benchmark/benchmark_particle_batches.py --backend numpy --bio-model WED --scenarios --beams 2 --trace-on-ct-grid --sizes 16 --pairs 1 --out benchmark/pb_audit_results/batching_cpu/validation
.venv/Scripts/python.exe benchmark/benchmark_particle_batches.py --backend numpy --scenarios --beams 2 --forward --sizes 16 --pairs 1 --out benchmark/pb_audit_results/batching_cpu/validation
.venv/Scripts/python.exe benchmark/benchmark_particle_batches.py --backend numpy --max-pairs 4096 --sizes 16 --pairs 1 --out benchmark/pb_audit_results/batching_cpu/cap4096
```

The subsequent [CPU/GPU pipeline investigation](particle_pipeline_batching.md)
measures these ideas after the GPU became available, including support selection,
grouped interpolation and packet downloads. It also records the VHEE regression
from a shared candidate/physics packet limit and the separate-stage correction.
Production packet hooks still need a scalar default; avoid enabling copy-heavy
batching globally based on these CPU results.
