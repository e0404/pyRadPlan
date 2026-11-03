# Dose performance backlog

Updated 2026-11-03 during CPU particle batching experiments.
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
   GPU validation and energy-lookup preparation remain outstanding. PyTorch CPU
   CT-grid tracing exposed a pre-existing tensor/scalar minimum incompatibility
   in image resampling, which is recorded there for follow-up.
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
   Production batching and GPU measurements remain outstanding.
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
isolate the validity-check portion of (5). GPU work waits until the user's other
GPU task finishes; the full `develop` baseline is deferred. The latest
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
