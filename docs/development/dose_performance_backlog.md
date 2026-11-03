# Dose performance backlog

Updated 2026-11-03 during CPU/GPU particle pipeline batching experiments.
This is the durable record of the outstanding work discussed with the user.
The 2026-11-02 update merge itself contained no new performance optimization.

## Outstanding opportunities

1. **Particle/VHEE preparation and interpolation.** Prepare focus interpolators,
   energy lookups and scaled IDD tables with correct lifetimes. Investigate shared
   depth searches across IDD, widths, weights, LET and biological channels inside
   the Array API interpolation backend. Benchmark against native NumPy routines.
   **CPU implementation and measurements completed:** focus reuse, scaled fields
   and prepared shared searches are described in
   [particle interpolation preparation](particle_interpolation_preparation.md).
   GPU full-output comparisons now pass in the pipeline experiments below; an
   isolated GPU measurement of interpolation preparation versus its parent
   revision remains outstanding. Repeated beamlet energy lookup is now cached
   per calculation and source kernels are fetched only on converted-cache misses;
   see [metadata reuse](particle_metadata_followup.md). Its whole-dose CPU gain
   is too small to establish with the current measurements. PyTorch CPU
   CT-grid tracing exposed a tensor/scalar minimum incompatibility in image
   resampling. This is now fixed using Array API clipping, with PyTorch added to
   the image-resampling tests (220 checks passed across four backends).
2. **Shared conservative spatial candidates.** Avoid scanning every beam-valid
   voxel for every ray using BEV bins/depth slabs, then retain exact existing
   cutoff decisions. Cover divergence, anisotropy, range/setup changes and all CT
   phases. Measure the preprocessing crossover and memory overhead.
3. **Bounded execution batches.** Start with multiple particle energies on a ray,
   or photon rays sharing kernels. Amortize launches, allocations and dispatch.
   Keep engine-specific physics modular, with a single-bixel fallback. Later
   support voxel-owned forward scoring, fine-sample children reduced into parent
   delivery columns, and richer photon transport such as AAA.
   **Initial CPU exploration completed:** bounded validation/physics prototypes and the
   proposed shared prepare/evaluate/collect boundary are recorded in
   [particle batching exploration](particle_batching_exploration.md).
   **CPU/GPU pipeline investigation completed:** separate candidate/selected-work
   queues, grouped backend interpolation and packet downloads are measured in
   [particle pipeline batching](particle_pipeline_batching.md). The small CuPy
   proton case improves from 7.52 to 1.63 s; generic batching regresses NumPy.
   Automatic execution policies and adapters for other modalities remain outstanding.
   Optional shared hooks and a scalar default are now implemented; see
   [fusion and hook validation](cupy_fusion_and_batching_hooks.md).
   Preserve separate stage budgets: one candidate
   cap serialized VHEE physics and caused a large regression.
   **Reusable backend now available:** `ArrayBackend1DInterpolatorBank` implements
   portable table-bank execution and opt-in fused CuPy float64 kernels. See
   [interpolator bank follow-up](interpolator_bank_followup.md). Later GPU timings
   encountered unrelated device load and a changing scalar baseline. An idle-GPU
   alternating comparison now measures about 15% proton, 19% carbon and 8% VHEE
   full-pipeline time reductions from fusion. Automatic specialization remains
   unchanged; the particle adapter requests it explicitly. Existing dose engines
   still use their scalar defaults. The [overhead investigation](batching_hook_overhead.md)
   finds identical counted GPU operations in the proton comparison and about
   3 ms extra host scheduling work; a validation shortcut reduces that cost.
   The earlier 50 ms gap is not established as intrinsic hook overhead.
   **Opt-in production particle processor completed:** `batch_size` enables the
   [production adapter](production_particle_batching.md), with intact selected
   packet reuse and scalar defaults. Generic VHEE retains eight evaluation
   packets. Automatic backend/workload policy, photon integration and safe
   batching of biological evaluator contexts remain open.
4. **Device output collection.** Prototypes exist in benchmark/. Earlier photon
   measurements suggested about 7% for full-device collection and 1% for bounded
   collection, with markedly different memory costs; a small proton case did not
   improve. Keep scenario slots, shared quantity indices and output representation
   separate. Device-resident Dij needs GUI/serialization/downstream compatibility.
5. **Per-beamlet synchronization.** Investigate batch-level particle dose validity
   checks and prepared photon group-index guarantees. Preserve error detection,
   NaN behavior and diagnostics. Dynamic compaction and host-visible decisions
   remain even after the existing mask-compaction and cluster-count changes.

After CPU work on (1), the user requested starting (3). These experiments also
isolate the validity-check portion of (5). The GPU became available and was
measured; the full `develop` baseline is still deferred. The earlier
small proton profile attributed most GPU work to per-beamlet support selection
and physics/interpolation, not shared ray geometry or output collection. The
2026-10-28 recheck had unstable absolute times (NumPy 0.41-1.90 s; CuPy 7.80-8.52 s)
and does not establish a stable speed ratio or explain the change from earlier
runs. Establish comparable warmed baselines before claiming new speedups.

## Other retained items and constraints

- Beam preparation currently uploads ray coordinates, then the ray tracer can
  rebuild a validated host Beam. Reuse appropriate validated host geometry to
  avoid this round trip; preserve setup shifts, precision and engine modifications.
- Further SimpleITK geometry migration is deferred. Existing measurements do not
  justify replacing its fast CPU resampling indiscriminately with GPU execution.
- Keep interpolation specialization in xp_utils, not engine-specific fast paths.
- Preserve Array API-compatible operations where practical (including take).
- Preserve sampling policy and RNG draw order; the latest sampling cleanup has
  isolated benefits but no demonstrated full-dose speedup.
- Prepared data must reflect explicit ScenarioSet semantics: setup groups, CT
  phases, range values, computed/derived slots and voxel-parameter mappings.
- Cache lifetimes must cover device/dtype, machine/energy, geometry-dependent
  cutoffs, focus/range shifters and engine reuse as appropriate.
- Benchmark in benchmark/, compare complete dose/support outputs, and keep changes
  on refactor branches. Optimization and Monte Carlo acceleration are out of scope.

See [original audit](pencil_beam_performance_audit.md),
[sparse-output experiments](sparse_output_performance.md), and
[sampling results](photon_sampling_preparation.md).
