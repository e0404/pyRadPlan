# Shared voxel-preparation performance

Measured 2026-10-26 on `refactor/dose-calc-performance`. The comparison restores
only voxel preparation from `4e747717`; interpolation, cutoff and all other dose
code are identical between each pair.

## Change

`DoseEngineBase._init_voxel_selection` now contains the shared selection,
resampling, coordinates and helper-mask setup. For a single CT:

- Union the existing SimpleITK VOI masks into the CT helper mask, then extract
  sorted indices once. This avoids materializing every VOI's integer indices,
  concatenating them and sorting overlapping selections.
- Reuse generated sorted indices. Canonicalize user-supplied indices internally
  without modifying the supplied array. Scenario membership is all true for
  the single CT, so no membership search is needed.
- Nearest-neighbor resampling of the existing float32 0/1 selection mask produces
  only 0/1 values. Extract its indices directly, omitting rounding, absolute
  value, conversion to bool and another sort/unique pass.

The essential union is:

```python
ct_mask = np.zeros(self._ct_grid.num_voxels, dtype=bool)
for voi in cst.vois:
    np.logical_or(ct_mask, sitk.GetArrayViewFromImage(voi.mask).ravel(), out=ct_mask)
self.voxel_sub_ix = np.flatnonzero(ct_mask)
self._vct_grid_mask = ct_mask
```

The bitmap is already needed downstream, so this does not add a persistent
full-grid allocation. Sparse custom selections bypass the VOI union. These are
host-side SimpleITK/NumPy preparation steps; subsequent numerical dose code
retains its existing Array API backend/device handling. No new public setting
or GUI API is introduced.

Multiple-CT cases retain the original union, resampling and membership path.
This change does not add or claim complete 4D dose support. Biological tissue
preparation, coordinate conventions, voxel ordering and dose physics are unchanged.

## Measurements

TG119, one beam, 10 mm bixels, Generic machine, unsampled influence calculation.
Photons were measured at 3, 5 and 10 mm; protons and VHEE at 5 mm. The CT has
3,597,681 voxels; the automatic union selects 601,736. Python 3.11.9, NumPy 2.3.5,
CuPy 13.3.0, CUDA runtime 12.6, RTX 4070 Laptop GPU; Windows x86-64 Intel family 6
model 183, 32 logical CPUs.

### Preparation stage

Five measured alternating rounds after warm-up, including mask union,
resampling, coordinate calculation and helper masks. Timing excludes dose
physics and biological tissue preparation. Speedup is the ratio of medians.

| Selection | Dose grid (mm) | Selected dose voxels | Before (ms) | After (ms) | Speedup | Traced peak before / after (MB) |
|---|---:|---:|---:|---:|---:|---:|
| automatic | 3 | 502,192 | 391.93 | 61.24 | 6.40x | 99.83 / 94.60 |
| custom | 3 | 56 | 31.44 | 28.54 | 1.10x | 50.54 / 33.06 |
| automatic | 5 | 108,871 | 305.66 | 40.39 | 7.57x | 77.27 / 75.18 |
| custom | 5 | 10 | 19.80 | 18.66 | 1.06x | 22.35 / 21.31 |
| automatic | 10 | 13,355 | 280.10 | 36.44 | 7.69x | 73.34 / 72.01 |
| custom | 10 | 2 | 16.51 | 16.19 | 1.02x | 18.43 / 18.43 |

The custom case supplies 66 unique CT indices in reverse order with duplicates.
Its smaller gains are expected: the old code already skips automatic VOI union.
Peaks were measured separately with `tracemalloc`; they include Python/NumPy
allocations, **not SimpleITK native allocations or process RSS**. MB is decimal.

### Complete influence calculation

Three alternating old/new pairs per case after warm-up, sequential CPU/GPU runs.
Times are medians. Speedups are medians of the three paired old/new ratios, so
may differ slightly from dividing the displayed median times. GPU execution is
synchronized before timing and before the dose call returns. Imports, phantom
loading and steering generation are outside these timings.

| Engine | Backend | Dose grid (mm) | Before (s) | After (s) | Paired speedup |
|---|---|---:|---:|---:|---:|
| VHEE | cupy | 5 | 1.083 | 0.832 | 1.33x |
| VHEE | numpy | 5 | 0.967 | 0.716 | 1.38x |
| photons | cupy | 3 | 1.867 | 1.528 | 1.23x |
| photons | numpy | 3 | 2.748 | 2.384 | 1.20x |
| photons | cupy | 5 | 1.651 | 1.344 | 1.23x |
| photons | numpy | 5 | 1.205 | 0.951 | 1.27x |
| photons | cupy | 10 | 1.509 | 1.301 | 1.16x |
| photons | numpy | 10 | 0.901 | 0.633 | 1.38x |
| protons | cupy | 5 | 1.861 | 1.601 | 1.16x |
| protons | numpy | 5 | 0.861 | 0.603 | 1.42x |

This removes roughly a quarter of a second of fixed setup work in many of these
cases. The percentage gain depends on how much time the remaining dose work
takes. These short laptop runs are noisy, particularly GPU runs; three pairs
are evidence of a useful local gain, not a precise general performance forecast.
The 6-8x preparation speedup must not be interpreted as a full-dose speedup.

## Validation and reproduction

- All 30 full-dose pairs had zero physical-dose difference and identical sparse
  indices/column pointers.
- Stage comparisons checked exact CT/dose indices, scenario membership, world
  coordinates and helper masks for automatic and custom selections at all three
  resolutions.
- Focused regression coverage includes overlapping and empty masks, nonbinary
  masks, unsorted/duplicate/empty custom selections, anisotropic resampling,
  oblique image geometry, invalid custom indices, and the retained multiple-CT
  membership fallback. The fallback test uses a controlled proxy, not a real
  clinical 4D case.
- Existing photon and ion fixture tests, including biological outputs, are run
  alongside the new tests (90 tests in total).

```powershell
.venv/Scripts/python.exe benchmark/profile_voxel_preparation.py
.venv/Scripts/python.exe -m pytest test/dose/test_voxel_preparation.py test/dose/test_dose_engine_base.py test/dose/test_svdpb.py test/dose/test_svdpb_bio.py test/dose/test_hongpb.py test/dose/test_hongpb_bio.py -q
```

The frozen reference is
[`profile_voxel_preparation_reference.py`](../../benchmark/profile_voxel_preparation_reference.py).
Raw timings, environment metadata and pairwise equality checks are in
[`voxel_preparation/`](../../benchmark/pb_audit_results/voxel_preparation/).
In the full-dose raw files, **`baseline` means current optimized code** and
**`variant` means the restored original voxel preparation**. The older audit's
`union` prototype is now a no-op; use `legacy_voxel_selection` for this comparison.
