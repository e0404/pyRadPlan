# TG119 five-field GUI photon performance recheck

2026-10-27. Follow-up to the reported NumPy change from roughly 16 to 35 seconds.
The earlier one-beam benchmarks did not represent this GUI workload, so their
results should not have been generalized to GUI-default five-field IMRT.

## Reproduction

The plan widget defaults to 3 mm isotropic dose voxels and 5 mm bixel spacing.
The five-field IMRT preset uses gantry angles 0, 72, 144, 216, 288 degrees and
zero couch angles. Photon SVD defaults enable dose sampling and trace on the
dose grid. The GUI compute callback generates steering and then computes the
influence matrix.

This recheck uses TG119, the Generic photon machine, those defaults, and NumPy
in the existing `.venv`. A fixed sampling seed (1234) permits exact comparison;
console progress is disabled. Headless calls exclude Qt signal/rendering costs.
Steering generation takes about 0.4 seconds and is reported separately from dose.
The workload has 1,567 rays/beamlets, 502,192 selected voxels, a 167 x 167 x 108
dose grid and 64,572,864 stored influence entries (~775 MB of sparse buffers).
No production code or installed dependencies were changed for this recheck.

Historical comparisons import complete `pyRadPlan` packages exported from Git,
in fresh processes with the same installed dependencies. The script verifies
the imported package path. Exports are cached in the ignored directory
`.benchmarks/gui_photon_history/`; no existing branch or worktree is modified.
NumPy is 2.3.5; the recorded runtime metadata includes BLAS/OpenMP thread pools.

## Historical timings

| Revision | Meaning | Cold dose | Warm dose |
|---|---|---:|---:|
| `dd06458d` | Immediately before array resampling/native dispatch | 41.16 s | 39.71 s |
| `4c5bef97` | Array resampling/native dispatch, before geometry refactor | 42.77 s | 39.93 s |
| `135da7a1` | Current compact ray geometry | 34.77 s | 35.52 s |

The older snapshots each have one cold and one warm run. Current production has
one cold and two warm runs (35.27 and 35.77 s); its warm entry is the median.
These measurements do not establish sub-percent differences between revisions.
The ~0.2 second difference across the resampling change is not evidence of a
regression; it is negligible relative to the reported 19-second difference.
The latest geometry refactor improves this sampled five-field workload by about
11% in elapsed time (1.12x speedup), consistent with its intended effect.

I reproduced the current ~35 seconds, but **did not reproduce the remembered
16-second NumPy baseline** under the same settings and environment. The available
comparison does not explain that earlier observation; it should not be assigned
to a particular configuration or runtime change without the earlier plan/log.

The earlier audit instead used a 5 mm dose grid, 10 mm bixels, one beam and
sampling disabled. It remains a valid comparison for that smaller workload,
but misses the expensive sampling path used here.

## Current profile

In a separate cProfile run (~36.2 seconds; timings below are inclusive):

| Work | Time |
|---|---:|
| Photon dose sampling | 13.7 s |
| Two stable sorts inside sampling | 10.8 s (part of sampling) |
| Three photon kernel interpolations per ray | 11.0 s |
| Ray geometry preparation and selection | 5.1 s |
| Kernel interpolator/FFT preparation, all five beams | 1.5 s |
| Ray tracing, all five beams | <0.9 s |
| Shared patient initialization | ~0.23 s |

The sort path performs 3,134 sorts to find maximum tail dose per radiological-
depth group. The interpolation path evaluates 4,701 interpolators, repeating
query preparation and index searches for the same coordinates and kernel grid.
These are larger opportunities than another broad geometry or SimpleITK port.

## Sampling prototype and correctness

A process-local NumPy prototype replaces the two sorts with a scatter maximum:

```python
cluster_dose = np.zeros(num_clusters, dtype=dose_tail.dtype)
np.maximum.at(cluster_dose, cluster, dose_tail)
threshold_dose = np.take(cluster_dose, cluster)
```

No sampling policy, random draw sequence, grouping or dose threshold is changed.
This is a benchmark prototype; native dispatch should live in the shared backend
utility before production integration.

Two alternating full-dose pairs measured:

| Implementation | Median dose time |
|---|---:|
| Current production | 35.35 s |
| Grouped-maximum prototype | 23.59 s |

This is **1.50x faster**, or **33.3% less elapsed time** in this workload.
The two prototype pairs had zero dose difference and identical sparse indices.
Complete-output SHA-256 comparisons also matched physical-dose data, row indices
and column pointers across **both historical snapshots, current production and
the prototype** (all 64,572,864 entries, not just a sample). Results are NumPy-only;
CuPy and strict Array API fallback validation remain part of the proposed next step.


## Suggested next steps

1. **Implement a shared backend grouped-maximum reduction and use it in photon
   sampling.** Put NumPy/CuPy native dispatch in `core.xp_utils`, with an Array
   API fallback, rather than introducing backend-specific code in the engine.
   Preserve empty groups, threshold behavior, random draws and selected row
   order. Validate NumPy/CuPy, strict Array API, boundary cases and full doses.
2. **Benchmark reuse of interpolation indices/weights across the three photon
   kernels inside the interpolation backend.** Prepared query state or suitable
   multi-value evaluation should avoid repeating grid searches. Do not assume
   that vector-valued SciPy interpolation or switching to the generic path is
   faster; compare with the current cached native implementation and check
   numerical agreement. This can become a shared facility for other engines.
3. **Keep the GUI-default workload in performance checks.** Record effective
   backend/device, grid, bixel width, beam/ray counts, sampling, JIT settings and
   cold/warm status with timings. Retain the smaller workloads for diagnosis,
   but include sampled five-field photons before making general speed claims.

## Reproduce and inspect artifacts

```powershell
.venv/Scripts/python.exe benchmark/profile_gui_photon_history.py --revision dd06458d --runs 1
.venv/Scripts/python.exe benchmark/profile_gui_photon_history.py --revision 4c5bef97 --runs 1
.venv/Scripts/python.exe benchmark/profile_gui_photon_history.py --revision 135da7a1 --runs 2
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend numpy --mode photons --grid 3 --bixel 5 --beams 5 --sampling --runs 2 --profile --out benchmark/pb_audit_results/gui_photon_recheck
.venv/Scripts/python.exe benchmark/profile_pb_audit.py --backend numpy --mode photons --grid 3 --bixel 5 --beams 5 --sampling --variant scattermax --runs 1 --paired --pairs 2 --out benchmark/pb_audit_results/gui_photon_recheck
.venv/Scripts/python.exe benchmark/profile_gui_photon_history.py --compare-only
```

JSON timings, profile summaries and complete-output hashes are in
`benchmark/pb_audit_results/gui_photon_recheck/`. Sparse matrices are saved
locally by the scripts and git-ignored. The first historical run's metadata
serialization failed after its matrix was saved; its JSON explicitly records
that its two timing rows were recovered verbatim from console output. The
harness now saves timings after every run and avoids serializing unrelated
plan model objects. In paired JSON, `baseline` is current production and
`variant` is the sampling prototype.
