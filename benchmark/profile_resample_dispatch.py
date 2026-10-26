"""Equivalent full-volume SimpleITK and Array API resampling, including transfers."""

import json
from pathlib import Path
import statistics
import time

import array_api_compat
import numpy as np
import SimpleITK as sitk

from pyRadPlan.core import Grid
from pyRadPlan.core.resample import ArrayBackendGridResampler, _resample_numpy_sitk
from pyRadPlan.core.xp_utils import to_numpy, to_namespace, synchronize


def main():
    """Measure complete one-shot and cached work for identical target voxels."""
    import cupy as cp  # noqa: PLC0415

    out = Path("benchmark/pb_audit_results/resample_dispatch")
    out.mkdir(exist_ok=True)
    rows = []
    for size in (48, 96, 192):
        source = Grid(
            resolution=dict(x=1.1, y=1.3, z=1.7),
            dimensions=(size, size, size),
            origin=np.array([3.4, -2.8, 7.1]),
        )
        target = source.resample(dict(x=1.5, y=1.9, z=2.1))
        values_np = np.random.default_rng(32).normal(size=(size, size, size)).astype(np.float32)
        for name, module in (("numpy", np), ("cupy", cp)):
            values = module.asarray(values_np)
            xp = array_api_compat.array_namespace(values)
            device = array_api_compat.device(values)
            for method in (sitk.sitkNearestNeighbor, sitk.sitkLinear):

                def native():
                    result = _resample_numpy_sitk(
                        to_numpy(values),
                        reference_grid=source,
                        target_grid=target,
                        interpolator=method,
                    )
                    return to_namespace(xp, result, device=device)

                def generic():
                    indices = xp.arange(target.num_voxels, dtype=xp.int64, device=device)
                    resampler = ArrayBackendGridResampler(
                        source, target, indices, interpolator=method
                    )
                    return xp.reshape(resampler(values), tuple(reversed(target.dimensions)))

                indices = xp.arange(target.num_voxels, dtype=xp.int64, device=device)
                cached = ArrayBackendGridResampler(source, target, indices, interpolator=method)
                methods = dict(
                    sitk=native,
                    array_api=generic,
                    cached_array_api=lambda: xp.reshape(
                        cached(values), tuple(reversed(target.dimensions))
                    ),
                )
                expected = to_numpy(native())
                samples = {key: [] for key in methods}
                for i in range(6):
                    for key in list(methods) if i % 2 else list(reversed(methods)):
                        synchronize(xp)
                        start = time.perf_counter()
                        actual = methods[key]()
                        synchronize(xp)
                        elapsed = time.perf_counter() - start
                        np.testing.assert_allclose(
                            to_numpy(actual), expected, rtol=2e-6, atol=2e-6
                        )
                        if i:
                            samples[key].append(elapsed)
                row = dict(
                    backend=name,
                    source_voxels=source.num_voxels,
                    target_voxels=target.num_voxels,
                    source_size=size,
                    interpolator="nearest" if method == sitk.sitkNearestNeighbor else "linear",
                    seconds=samples,
                    medians={key: statistics.median(ts) for key, ts in samples.items()},
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    report = dict(
        numpy=np.__version__, cupy=cp.__version__, sitk=sitk.Version_VersionString(), rows=rows
    )
    (out / "full_volume.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
