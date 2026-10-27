# Pencil-beam sparse indices and device collection

Follow-up to [shared photon hot paths](shared_photon_hotpaths.md), based on
`dd670e87`, on `refactor/dose-calc-performance`.

## Production change: safe compact indices

The shared pencil-beam collector now selects int32 for CSC row indices and
column pointers when the grid dimensions, column count and estimated stored-entry
count fit. It retains int64 for larger shapes/estimates. Actual accumulated entry
counts are checked again on every append: pointer arithmetic uses Python integers,
and both shared index arrays are promoted before storing an overflowing pointer.
This avoids an intermediate NumPy int32 addition wrapping before the check.

All quantities continue to share one index structure per scenario. Promotion
replaces both arrays in that shared structure, and final SciPy matrices retain
sharing. Dose values remain float32; computational backend voxel-selection arrays
are unchanged. Direct-dose accumulation is unchanged. This change applies to
photons, particle pencil-beam engines and VHEE through the common collector.

For the sampled five-beam TG119 photon workload, 64,572,864 CPU entries require:

- Before: 774,886,912 bytes (float32 dose, int64 rows/pointers).
- After: 516,589,184 bytes (float32 dose, int32 rows/pointers).
- Saving: 258,297,728 bytes, approximately 33.3% of the sparse buffers.

This is storage per physical-dose matrix, not total process memory. Multiple
quantities share indices, so their combined saving must count that shared buffer
only once. Sparse-buffer estimates do not include temporary computational arrays,
allocator caches or unused collector capacity.

SciPy 1.17.1 in the local environment supports int32 CSC/CSR matrices and arrays,
including format conversion and matrix-vector products. The helper checks dimensions
and stored-entry counts, not the potentially enormous product of matrix dimensions.

## Device collector experiment

The production collector allocates NumPy buffers and downloads each beamlet's
values and row indices before insertion. This was an implementation choice, not
a demonstrated speed advantage of host construction.

`benchmark/profile_device_collector.py` supplies process-local alternatives:

- `device_collect_full` retains beamlet columns on the backend, concatenates them,
  constructs actual CuPy CSC objects, then downloads completed row/value buffers.
- `device_collect_bounded` packs and downloads batches of approximately 64 MiB
  payload, bounding retained beamlet data independently of total matrix size.

Both return the existing SciPy output and use the existing final canonicalization
and validation. GPU sparse-object wrapping and bulk downloads are timed separately.
The full prototype temporarily holds both column fragments and concatenated arrays;
its peak payload can therefore substantially exceed final matrix storage. The batch
threshold is per scenario and can be exceeded by one column; concurrent scenarios,
concatenation temporaries and physics arrays also consume memory.

Sampled CuPy pool usage is recorded at collection/concatenation/wrapping boundaries.
It is a sampled peak of used pool bytes, not an exact device-wide high-water mark.
No production backend-placement switch or device-output contract is introduced by
these experiments. Returning a resident device matrix through all public Dij/GUI
paths would require a separate compatibility change; the benchmark compares device
collection while preserving today's SciPy result contract.

## Correctness and measurement method

`profile_sparse_output_reference.py` freezes the old allocation and insertion
methods. Paired runs alternate current production and the selected reference or
prototype. In JSON, `baseline` is the int32 production collector; `variant` is the
named experiment. Every pair compares full physical-dose values and sparse index
values/pointers. Index dtype changes are expected for the int64 reference.

Focused regression tests cover the real int32 pointer boundary without allocating
billions of entries, large grid row indices, growth from zero capacity, forced
promotion during collection, empty columns, inactive scenarios, multiple quantities,
shared final buffers and NumPy/CuPy/strict Array API inputs.

The selected dose suite passes: 109 tests (including the 16 new collector tests).
The initial combined dose/interface/GUI invocation reported 61 missing-fixture
errors before dose tests could execute. Collecting the dose tests from their
directory resolves fixture discovery without code changes. A separate Dij, namespace
conversion and GUI run passes 611 tests with four skips. Ruff checks also pass.

## Timing caveat

Initial default-thread runs had large runtime swings: the same NumPy collector
comparison ranged from 21 to 107 seconds and the GPU comparison from 9.6 to 58.7
seconds. These initial `numpy_indices/` and `device_full/` measurements remain in
the raw artifacts, but their medians must not be interpreted as speedups. Output
comparisons and buffer sizes remain valid. Repeat measurements are reported
separately; no cause for the timing variation has been established.

## Device assembly results

The stable five-field TG119 comparisons (two alternating pairs each) measured:

| Collector experiment | Host baseline | Prototype | Time reduction |
| --- | ---: | ---: | ---: |
| Full device collection, final bulk download | 9.967 s | 9.256 s | 7.1% |
| Approximately 64 MiB batches | 9.752 s | 9.647 s | 1.1% |

Every pair has identical physical-dose values, row indices and column pointers.
These are separate paired experiments; compare each prototype with its own
baseline. Two pairs support a preliminary workload-specific estimate, not a
universal speed claim, especially for the approximately 1% batching difference.

Full collection retains 516.6 MB of column payload and reaches about 1,094 MB
sampled live CuPy pool usage while packing. Bounded collection retains at most
67.4 MB of payload here and reaches about 225 MB sampled live pool usage. The
original collector has roughly two downloads per photon bixel (3,134 for this
plan); full collection uses two and batching uses sixteen (eight flushes).

The actual warm CuPy CSC constructors took 0.110 and 0.128 ms for the full matrix.
The two bulk downloads took 74 and 81 ms in total. Constructor work is therefore
not a reason to force host assembly. Nevertheless, returning SciPy still requires
one download, and most of the remaining calculation time is elsewhere.

These prototypes remain in `benchmark/`. The production change in this step is
safe index storage; it does not impose full-result GPU residency on large plans.
A future production collector should make the memory budget explicit internally,
preserve shared indices across quantities, support all scenarios, and separate
collection from the requested output representation. Device-resident output also
needs validation/GUI/export compatibility work, not just replacing a constructor.

The device prototypes cast row indices to the chosen output width before transfer;
the production int32 storage change alone still downloads the existing computation
indices (normally int64) and casts on insertion. Thus the prototypes combine fewer
transfers with reduced index-transfer volume. Their gains must not be attributed
solely to eliminating synchronization.

The stable NumPy int32/int64 repeat measured 20.853 vs 21.212 s (two alternating
pairs), a small 1.7% difference. The firm benefit is the 33.3% sparse-buffer saving;
this measurement does not establish a substantial dose-calculation speedup.

## Remaining work in the current profile

A separate GPU profile of the production collector used the same five-field
workload. Synchronized stage attribution measured approximately 5.53 s in photon
bixel computation, 2.02 s in ray geometry/selection, 0.57 s in output insertion,
and 0.62 s in Siddon tracing. Final sparse wrapping/validation was below 1 ms.
These instrumented stage timings explain costs; they are not paired speedup
measurements. The cProfile run additionally attributed about 2.61 s to shared
interpolation and 2.59 s to sampling, both inside bixel computation.

This supports keeping collection work proportionate: the full-device prototype
can save a fraction of a second here, while per-ray computation/dispatch remains
the larger target. The generic output collector should still be separated from
output representation if device-resident results are pursued for downstream reuse.

A supplementary proton workload (5 mm grid, 10 mm bixels, one beam, physical plus
LET dose) retained identical values and shared indices with the device prototype,
but measured 1.414 s for the host collector and 1.462 s for full device collection.
This single-pair check is not evidence of a universal GPU collection improvement.

The GPU int32/int64 repeat measured 10.226 vs 10.334 s (two alternating pairs).
An earlier single pair had the opposite ordering (9.969 vs 9.613 s), so there is
no convincing GPU runtime gain or regression from index width alone. The GPU
matrix's buffers shrink from 774,929,488 to 516,617,568 bytes. Its slightly different
entry count from NumPy predates this change; comparisons are within each backend.

The first full-device CuPy CSC construction in a fresh benchmark process took
about 62 ms, versus 0.11-0.13 ms after warm-up. Both construction and downloads
are included in the full-dose prototype timings. This distinguishes first-use
initialization from the cost of wrapping already prepared sparse buffers.

## Reproduce

```powershell
# Focused collector and existing photon/particle dose regression tests.
.venv/Scripts/python.exe -m pytest test/dose -q -k "sparse_output or svdpb or hongpb or dose_engine_base"

# Production int32 storage versus the frozen int64 collector; repeat with cupy.
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend numpy --mode photons --grid 3 --bixel 5 --beams 5 --sampling --runs 0 --variant legacy_sparse_indices --paired --pairs 2 --no-matrices --out benchmark/pb_audit_results/sparse_output/recheck

# Device assembly experiments; substitute device_collect_bounded for bounded batches.
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend cupy --mode photons --grid 3 --bixel 5 --beams 5 --sampling --runs 0 --variant device_collect_full --paired --pairs 2 --no-matrices --out benchmark/pb_audit_results/sparse_output/device_recheck

# Current GPU stage profile.
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend cupy --mode photons --grid 3 --bixel 5 --beams 5 --sampling --runs 0 --profile --no-matrices --out benchmark/pb_audit_results/sparse_output/profile_recheck
```

All 19 full sparse comparisons in the recorded experiments have zero dose
difference and identical index values/pointers. Proton checks also compare the
entire LET influence matrix and preserve physical/LET index sharing. The
supplementary proton/VHEE comparisons use one beam, a 5 mm dose grid and 10 mm
bixels; these verify modality coverage rather than establish small speed differences.

Raw timing, memory, comparison and profile results are under
`benchmark/pb_audit_results/sparse_output/`. The `*_repeat/` directories contain
the repeated measurements discussed above; initial timing swings are retained
separately. `--no-matrices` omits large local matrix files without skipping the
full output comparisons. Performance scripts and prototypes live in `benchmark/`
and are not imported by production code.
