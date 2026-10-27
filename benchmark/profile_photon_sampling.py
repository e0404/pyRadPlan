"""Isolated sampler and final-selection timings on NumPy/CuPy."""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import array_api_compat
import numpy as np

from pyRadPlan.core.xp_utils import to_numpy
from pyRadPlan.dose.engines._svdpb import PhotonPencilBeamSVDEngine
from profile_photon_sampling_reference import _sample_dij as reference


def main():
    """Compare each sampling lever independently, with exact seeded output checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "cupy":
        import cupy as module  # noqa: PLC0415
    else:
        module = np
    xp = array_api_compat.array_namespace(module.asarray([0.0]))

    def sync():
        if args.backend == "cupy":
            module.cuda.get_current_stream().synchronize()

    def time_calls(functions, reset=lambda: None):
        timings = {name: [] for name in functions}
        for fn in functions.values():
            fn()
        sync()
        for pair in range(5):
            names = list(functions)
            if pair % 2:
                names.reverse()
            for name in names:
                reset()
                sync()
                start = time.perf_counter()
                for _ in range(10):
                    functions[name]()
                sync()
                timings[name].append((time.perf_counter() - start) / 10)
        return timings

    cases = []
    selections = []
    config = dict(type="radius", delta_rad_depth=5.0, lat_cut_off=10.0, rel_dose_threshold=0.5)
    for n in (5000, 50000, 200000):
        rng = np.random.default_rng(123)
        rows = xp.arange(n, dtype=xp.int64)
        dose = xp.asarray(rng.random(n), dtype=xp.float64)
        depths = xp.asarray(rng.uniform(-2, 200, n), dtype=xp.float64)
        radii = xp.asarray(rng.uniform(0, 900, n), dtype=xp.float64)
        engine = SimpleNamespace(dij_sampling=config, _rng=np.random.default_rng(77))
        prepared_count = 41
        sample = PhotonPencilBeamSVDEngine._sample_dij
        functions = dict(
            previous=lambda: reference(engine, rows, dose, depths, radii),
            count_only=lambda: reference(
                engine, rows, dose, depths, radii, num_clusters=prepared_count
            ),
            selection_only=lambda: sample(engine, rows, dose, depths, radii),
            both=lambda: sample(engine, rows, dose, depths, radii, num_clusters=prepared_count),
        )

        def reset():
            engine._rng = np.random.default_rng(77)

        expected = None
        for fn in functions.values():
            reset()
            output = tuple(to_numpy(x) for x in fn())
            if expected is None:
                expected = output
            else:
                for actual, target in zip(output, expected):
                    np.testing.assert_array_equal(actual, target)
        cases.append(dict(voxels=n, identical=True, seconds=time_calls(functions, reset)))
        for fraction in (0.1, 0.5, 0.9):
            keep = xp.asarray(rng.random(n) < fraction)

            def boolean():
                return rows[keep], dose[keep]

            def compact():
                indices = xp.nonzero(keep)[0]
                return xp.take(rows, indices, axis=0), xp.take(dose, indices, axis=0)

            for actual, target in zip(compact(), boolean()):
                np.testing.assert_array_equal(to_numpy(actual), to_numpy(target))
            selections.append(
                dict(
                    voxels=n,
                    fraction=fraction,
                    identical=True,
                    seconds=time_calls(dict(boolean=boolean, take=compact)),
                )
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(dict(backend=args.backend, sampler=cases, selection=selections), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
