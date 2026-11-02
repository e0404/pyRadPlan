"""CPU-only particle interpolation comparison against pre-change 6ea49b74.

The frozen reference methods below retain the original algorithm. GPU measurements
and comparisons to develop are deliberately outside this harness.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYRADPLAN_XP_PREFER_GPU"] = "false"

import argparse
import json
import logging
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
from scipy.interpolate import interp1d
from pyRadPlan.core.xp_utils.compat import interp1d as array_interp


def legacy_interpolate_kernels_in_depth(self, bixel):
    # xp = array_api_compat.array_namespace(bixel["rad_depths"])
    kernel = bixel["kernel"]
    depths = kernel["depths"]

    # xp = array_api_compat.array_namespace(bixel["rad_depths"])

    # Add potential offset
    # Ensure depths is on the correct device
    # depths = xp.asarray(depths, dtype=bixel["rad_depths"].dtype)
    depths = depths + kernel["offset"] - bixel["rad_depth_offset"]

    # Conversion factor from MeV cm^2/g per primary to Gy mm^2 per 1e6 primaries
    conversion_factor = 1.6021766208e-02

    # Find all values we need to interpolate
    used_kernels = {}
    used_kernels["idd"] = conversion_factor * kernel["idd"]

    # Lateral Kernel Model
    for name in self._lateral_kernel_names():
        used_kernels[name] = kernel[name]

    # LET (matrix output and/or input to the biological model)
    if self._use_let_kernel:
        used_kernels["let"] = kernel["let"]

    # Biological model kernels (1-D or (n_tissue_classes, n_depths) arrays); bixels
    # without tissue parameters (lateral cutoff calibration) only need physical dose
    if self._calc_bio_dose and "v_alpha_x" in bixel:
        bio_kernels = self._bio_evaluator.kernel_quantities(kernel)
        kernel_energy = float(kernel["energy"])
        if kernel_energy not in self._validated_bio_kernel_energies:
            self._bio_evaluator.validate_kernel_quantities(bio_kernels)
            self._validated_bio_kernel_energies.add(kernel_energy)
        used_kernels.update(bio_kernels)

    # Interpolate all fields in X
    return array_interp(bixel["rad_depths"], depths, used_kernels)


def legacy_calc_sigma_ini_focus_ix(
    self, energy: np.float64, focus_ix: int, ssd: np.floating
) -> np.float64:
    """
    Get the initial beam width for a specific focus.

    Parameters
    ----------
    energy : np.float64
        The energy of the beam.
    focus_ix : int
        The focus index.
    ssd : np.floating
        The source to surface distance.

    Returns
    -------
    np.float64
        The initial sigma for the given energy and focus index.
    """
    focus = self._machine.foci[energy][focus_ix]
    interpolator = interp1d(focus.dist, focus.sigma, fill_value="extrapolate")
    return interpolator(ssd)


def timed(fn, repeats=1):
    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = fn()
    return (time.perf_counter() - start) / repeats, result


def compare(a, b):
    if isinstance(a, dict):
        assert tuple(a) == tuple(b)
        return max(compare(a[k], b[k]) for k in a)
    from pyRadPlan.core.xp_utils import to_numpy

    a, b = to_numpy(a), to_numpy(b)
    assert a.dtype == b.dtype
    np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-12)
    return float(np.max(np.abs(a - b), initial=0))


def compare_dose(a, b):
    errors = {}
    for name in a.quantities:
        aa, bb = getattr(a, name), getattr(b, name)
        for i, (x, y) in enumerate(zip(aa.flat, bb.flat)):
            assert x.shape == y.shape
            diff = (x - y).tocoo()
            error = float(np.max(np.abs(diff.data), initial=0))
            scale = float(np.max(np.abs(x.data), initial=0))
            assert error <= 2e-6 * scale + 1e-12, (name, i, error, scale)
            errors[f"{name}_{i}"] = error
    return errors


def main():
    from copy import copy
    from pyRadPlan import IonPlan, generate_stf, load_tg119, settings
    from pyRadPlan.dose.engines._base_pencilbeam_particle import (
        ParticlePencilBeamEngineAbstract as Particle,
    )
    from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--mode", choices=["protons", "VHEE", "carbon"], default="protons")
    parser.add_argument("--grid", type=float, default=5)
    parser.add_argument("--trace-on-ct-grid", action="store_true")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--out", default="benchmark/pb_audit_results/interpolation_cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    settings.xp.prefer_gpu = False
    settings.xp.preferred_cpu_array_backend = "numpy"
    ct, cst = load_tg119()
    plan = IonPlan(radiation_mode=args.mode, machine="Generic", bio_model="none")
    plan.prop_stf = dict(bixel_width=10, gantry_angles=[0], couch_angles=[0])
    if args.mode == "VHEE":
        plan.prop_stf["energy"] = 200
    plan.prop_dose_calc = dict(
        dose_grid={"resolution": [args.grid] * 3},
        trace_on_dose_grid=not args.trace_on_ct_grid,
        calc_let=True,
    )
    stf = generate_stf(ct, cst, plan)
    settings.xp.preferred_cpu_array_backend = args.backend
    threads = None
    if args.backend == "torch":
        import torch

        if args.torch_threads is not None:
            torch.set_num_threads(args.torch_threads)
        threads = torch.get_num_threads()

    modern_depth = Particle._interpolate_kernels_in_depth
    modern_focus = Particle._calc_sigma_ini_focus_ix
    captured, effective = {}, {}

    def capture(engine, bixel):
        result = modern_depth(engine, bixel)
        n = bixel["rad_depths"].shape[0]
        # Retain just three representative interpolation workloads, including a
        # large bixel, without adding instrumentation to the timed full runs.
        if n and ("first" not in captured):
            captured["first"] = (copy(engine), dict(bixel))
        if n and ("largest" not in captured or n > captured["largest"][1]["rad_depths"].shape[0]):
            captured["largest"] = (copy(engine), dict(bixel))
        if n > 100 and "medium" not in captured:
            captured["medium"] = (copy(engine), dict(bixel))
        effective.update(lateral_model=engine.lateral_model)
        return result

    def run(legacy=False, capturing=False):
        depth = legacy_interpolate_kernels_in_depth if legacy else modern_depth
        focus = legacy_calc_sigma_ini_focus_ix if legacy else modern_focus
        if capturing:
            depth = capture
        with (
            patch.object(Particle, "_interpolate_kernels_in_depth", depth),
            patch.object(Particle, "_calc_sigma_ini_focus_ix", focus),
        ):
            engine = ParticleHongPencilBeamEngine(plan.model_copy(deep=True))
            engine.console_progress = False
            return engine.calc_dose_influence(ct, cst, stf.model_copy(deep=True))

    # Warm both paths; every measured calculation creates a fresh engine and
    # therefore includes all interpolator construction and field preparation.
    cold = {}
    cold["legacy"], old = timed(lambda: run(legacy=True))
    cold["prepared"], new = timed(lambda: run(capturing=True))
    errors = compare_dose(old, new)
    full = {"legacy": [], "prepared": []}
    for i in range(args.pairs):
        outputs = {}
        for name in ("legacy", "prepared") if i % 2 == 0 else ("prepared", "legacy"):
            elapsed, outputs[name] = timed(lambda: run(legacy=name == "legacy"))
            full[name].append(elapsed)
        errors = compare_dose(outputs["legacy"], outputs["prepared"])
        print("pair", i, {k: v[-1] for k, v in full.items()}, flush=True)

    micro = {}
    for name, (engine, bixel) in captured.items():
        # A copy of effective engine state keeps the per-calculation lateral
        # model, even though the public engine restores its settings afterwards.
        engine._depth_interpolator_cache = {}
        creation, obj = timed(lambda: engine._prepare_depth_interpolator(bixel))
        functions = {
            "legacy": lambda: legacy_interpolate_kernels_in_depth(engine, bixel),
            "prepared": lambda: modern_depth(engine, bixel),
        }
        error = compare(functions["legacy"](), functions["prepared"]())
        for fn in functions.values():
            for _ in range(10):
                fn()
        samples = {key: [] for key in functions}
        for i in range(9):
            order = tuple(functions) if i % 2 == 0 else tuple(reversed(functions))
            for key in order:
                elapsed, _ = timed(functions[key], repeats=100)
                samples[key].append(elapsed)
        micro[name] = dict(
            queries=bixel["rad_depths"].shape[0],
            depths=bixel["kernel"]["depths"].shape[0],
            preparation_s=creation,
            max_abs_error=error,
            samples_s=samples,
        )

    result = dict(
        reference="6ea49b74 frozen interpolation and focus methods",
        config=vars(args),
        effective=effective,
        cpu=platform.processor(),
        numpy=np.__version__,
        torch_threads=threads,
        torch=None if args.backend != "torch" else torch.__version__,
        beamlets=stf.total_number_of_bixels,
        first_runs_s=cold,
        full_s=full,
        dose_max_abs_errors=errors,
        micro=micro,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    filename = f"{args.backend}_{args.mode}_{args.grid:g}mm_{'ct' if args.trace_on_ct_grid else 'dose'}.json"
    (out / filename).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
