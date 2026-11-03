# Particle metadata reuse

2026-11-03. Follow-up to the interpolation/batching investigation, using TG119,
one beam, 10 mm spot spacing, a 5 mm dose grid and NumPy. No dose formula,
sampling rule, public calling interface or GUI field changes.

## Integrated change

The particle engine caches exact requested energy values to their machine indices
for one calculation. The first lookup still delegates four-decimal matching and
ambiguity handling to the machine. Closely spaced requested energies remain
separate cache keys, even if the machine resolves them to the same index.

The cache resets before each calculation, including reuse with another machine.
It survives geometry/cutoff changes within that calculation because those do not
change the machine energy axis. Converted kernels and depth interpolation still
invalidate on their existing geometry boundaries. Source kernels are now fetched
only when their converted-cache entry is missing.

For the 908-beamlet proton case, 18 energy-cache entries remove 890 repeated
machine lookups. Full-calculation `get_energy_index` calls fall from 1,133 to 243;
the remaining calls are setup work. Source `get_kernel_by_index` calls fall from
1,151 before the change to 261 after moving the fetch into the cache-miss branch.
The focused profile shows energy-lookup cumulative time 20.85 ms uncached versus
3.79 ms cached, but **profiling overhead is included** in those numbers.

## Full-calculation timing: small and uncertain

The initial experiment memoized all machine energy lookups for a calculation.
The integrated version is deliberately narrower: it caches beamlet metadata in
the engine and leaves setup lookups alone. The production comparison disables
only this new cache in its reference run; both variants retain lazy source-kernel
fetching. It is not a frozen comparison against develop or the entire preceding
engine implementation.

| Experiment | Pairs | Reference median (s) | Cached median (s) | Median paired change | Approx. 95% interval |
|---|---:|---:|---:|---:|---:|
| Broad prototype, protons | 15 | 0.446 | 0.429 | -4.92% | [-8.51%, +0.87%] |
| Broad prototype, proton repeat | 31 | 0.469 | 0.447 | -4.03% | [-5.95%, -2.72%] |
| Integrated beamlet cache, protons | 31 | 0.461 | 0.461 | -2.81% | [-6.64%, +3.13%] |
| Integrated beamlet cache, VHEE | 15 | 0.482 | 0.480 | -0.61% | [-2.04%, +2.12%] |

Intervals bootstrap paired percentage changes (10,000 resamples, seed 41).
Ratios of separate medians and medians of paired ratios are different statistics.
Runs alternate reference/cached order and include fresh-engine setup and complete
influence calculation. The integrated change removes measured redundant work,
but these timings **do not establish a whole-calculation speedup** or rule out a
small regression. Do not attribute the broad prototype's 4% result to the shipped
cache. The `profiles/` one-pair run is diagnostic only.

All recorded paired dose and LET matrices match exactly, including sparse
structures and delivery identities. The dose/biology/scenario/cache suites passed
129 tests. Focused checks verify exact energy keys, lazy kernel fetching,
geometry invalidation, preservation of lookup errors, and engine reuse with both
CT-grid and dose-grid tracing; the final refined cache suite passed 12 tests.

The implementation is small and retains the existing engine architecture. More
elaborate setup-wide energy maps are lower priority than execution batching.
There is no GPU performance claim for this metadata change.

Raw data:
[`particle_metadata/`](../../benchmark/pb_audit_results/particle_metadata/).
The pipeline harness now records current source hashes instead of labeling all
future runs as the old `a780b35c` reference, so later changes remain distinguishable.

```powershell
.venv/Scripts/python.exe benchmark/benchmark_particle_metadata.py --production --pairs 31 --out benchmark/pb_audit_results/particle_metadata/recheck
.venv/Scripts/python.exe benchmark/benchmark_particle_metadata.py --production --mode VHEE --pairs 15 --out benchmark/pb_audit_results/particle_metadata/recheck
```
