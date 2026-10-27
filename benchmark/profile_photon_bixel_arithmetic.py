"""Isolate photon bixel arithmetic with precomputed interpolation values.

This deliberately excludes interpolation and sampling. Full-dose paired timings
remain authoritative for end-to-end gains. All paths use identical backend inputs.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import array_api_compat
import numpy as np

from pyRadPlan.core.xp_utils import ArrayBackendGridInterpolator, to_numpy
from pyRadPlan.dose.engines._svdpb import PhotonPencilBeamSVDEngine
from profile_photon_bixel_reference import _compute_bixel as reference


def main():
    """Compare paired arithmetic calls, retaining numerical and timing results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "cupy":
        import cupy as module  # noqa: PLC0415
    else:
        module = np
    xp = array_api_compat.array_namespace(module.asarray([0.0]))
    engine = object.__new__(PhotonPencilBeamSVDEngine)
    engine.xp = xp
    engine.device = array_api_compat.device(module.asarray([0.0]))
    engine.enable_dij_sampling = False

    def sync():
        if args.backend == "cupy":
            module.cuda.get_current_stream().synchronize()

    results = []
    for dtype_name in ("float32", "float64"):
        dtype = getattr(xp, dtype_name)
        for n in (5000, 50000, 200000):
            rng = np.random.default_rng(43)
            queries = xp.asarray(rng.uniform(-1, 1, size=(n, 2)), dtype=dtype)
            grid = xp.asarray([-1, 0, 1], dtype=xp.float64)
            interpolator = ArrayBackendGridInterpolator(
                (grid, grid), xp.asarray(rng.normal(size=(3, 3, 3)), dtype=xp.float64)
            )
            # Preserve the real backend interpolator's output dtype and memory layout.
            fields = interpolator(queries)
            ray = dict(
                kernel=SimpleNamespace(m=0.0037, kernel_betas=np.array([0.0123, 0.0567, 0.0987])),
                kernel_depth_parameters={},
                rad_depths=xp.asarray(np.linspace(0, 200, n), dtype=dtype),
                iso_lat_dists=queries,
                kernel_interpolators=lambda queries: fields,
                geo_depths=xp.asarray(np.linspace(850, 1150, n), dtype=dtype),
                sad=1000.0,
                ix=xp.arange(n, dtype=xp.int64),
                beamlets=[dict(weight=1.0)],
            )
            functions = dict(
                current=lambda: engine._compute_bixel(ray, 0),
                previous=lambda: reference(engine, ray, 0),
            )
            outputs = {name: fn()["physical_dose"] for name, fn in functions.items()}
            np.testing.assert_array_equal(
                to_numpy(outputs["current"]), to_numpy(outputs["previous"])
            )
            for fn in functions.values():
                for _ in range(3):
                    fn()
            sync()
            record = dict(
                dtype=dtype_name, voxels=n, repeats=20, identical=True, current=[], previous=[]
            )
            for pair in range(5):
                order = ("current", "previous") if pair % 2 == 0 else ("previous", "current")
                for name in order:
                    sync()
                    start = time.perf_counter()
                    for _ in range(record["repeats"]):
                        functions[name]()
                    sync()
                    record[name].append((time.perf_counter() - start) / record["repeats"])
            results.append(record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(backend=args.backend, cases=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
