"""Profile particle metadata lookup and measure a per-calculation cache experiment."""

import argparse
import cProfile
import hashlib
import json
import logging
from pathlib import Path
import pstats
import time
from unittest.mock import patch

import numpy as np

from pyRadPlan import IonPlan, generate_stf, load_tg119, settings
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine
from pyRadPlan.machines.particles import ParticleAccelerator


def compare_dose(a, b):
    """Compare complete influence outputs without depending on batching experiments."""
    assert a.quantities == b.quantities
    for key in ("beam_num", "ray_num", "bixel_num", "num_of_scenarios"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    errors = {}
    for key in a.quantities:
        aa, bb = getattr(a, key), getattr(b, key)
        assert aa.shape == bb.shape
        for i, (x, y) in enumerate(zip(aa.flat, bb.flat)):
            assert x.shape == y.shape
            np.testing.assert_array_equal(x.indptr, y.indptr)
            np.testing.assert_array_equal(x.indices, y.indices)
            np.testing.assert_allclose(x.data, y.data, rtol=2e-6, atol=1e-12)
            errors[f"{key}_{i}"] = float(np.max(np.abs(x.data - y.data), initial=0))
    return errors


class Uncached(dict):
    """Disable only the production lookup cache without changing dose arithmetic."""

    def get(self, key, default=None):
        """Require the engine to resolve the energy on every lookup."""
        return default

    def __setitem__(self, key, value):
        """Discard the resolved index for the reference run."""


def main():
    """Keep the experiment outside production; retain paired full-dose comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["protons", "VHEE"], default="protons")
    parser.add_argument("--pairs", type=int, default=15)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--out", default="benchmark/pb_audit_results/particle_metadata")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    settings.xp.prefer_gpu = False
    settings.xp.preferred_cpu_array_backend = "numpy"
    ct, cst = load_tg119()
    plan = IonPlan(radiation_mode=args.mode, machine="Generic")
    plan.prop_stf = dict(bixel_width=10, gantry_angles=[0], couch_angles=[0])
    if args.mode == "VHEE":
        plan.prop_stf["energy"] = 200
    plan.prop_dose_calc = dict(dose_grid={"resolution": [5] * 3}, calc_let=True)
    stf = generate_stf(ct, cst, plan)
    original = ParticleAccelerator.get_energy_index

    def run(cached=False):
        cache = {}

        def lookup(machine, energy, round_decimals=4):
            key = (id(machine), energy, round_decimals)
            if key not in cache:
                cache[key] = original(machine, energy, round_decimals)
            return cache[key]

        engine = ParticleHongPencilBeamEngine(plan.model_copy(deep=True))
        engine.console_progress = False
        if not args.production or not cached:
            engine._energy_index_cache = Uncached()
        with patch.object(
            ParticleAccelerator,
            "get_energy_index",
            lookup if cached and not args.production else original,
        ):
            start = time.perf_counter()
            result = engine.calc_dose_influence(ct, cst, stf.model_copy(deep=True))
            elapsed = time.perf_counter() - start
        return elapsed, result, len(engine._energy_index_cache) if args.production else len(cache)

    run()

    def profile_calls(cached):
        profile = cProfile.Profile()
        profile.runcall(run, cached)
        return {
            f"{Path(file).name}:{name}": dict(
                calls=value[1], self_s=value[2], cumulative_s=value[3]
            )
            for (file, line, name), value in pstats.Stats(profile).stats.items()
            if name in ("get_energy_index", "get_kernel_by_index", "_init_bixel")
        }

    selected = profile_calls(False)
    selected_cached = profile_calls(True)
    print("profile", selected, flush=True)
    warm_a, warm_b = run()[1], run(True)[1]
    compare_dose(warm_a, warm_b)
    samples = {"reference": [], "cached": []}
    for i in range(args.pairs):
        outputs = {}
        for name in ("reference", "cached")[:: 1 if i % 2 == 0 else -1]:
            elapsed, outputs[name], count = run(name == "cached")
            samples[name].append(elapsed)
            if name == "cached":
                cache_entries = count
        errors = compare_dose(outputs["reference"], outputs["cached"])
    print({k: float(np.median(v)) for k, v in samples.items()}, flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"numpy_{args.mode}.json").write_text(
        json.dumps(
            dict(
                config=vars(args),
                profile=selected,
                profile_cached=selected_cached,
                engine_source_sha256=hashlib.sha256(
                    (
                        Path(__file__).resolve().parents[1]
                        / "pyRadPlan/dose/engines/_base_pencilbeam_particle.py"
                    ).read_bytes()
                ).hexdigest(),
                samples_s=samples,
                cache_entries=cache_entries,
                max_abs_errors=errors,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
