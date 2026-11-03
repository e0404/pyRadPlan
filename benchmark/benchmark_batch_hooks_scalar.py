"""Compare the unchanged scalar path with the pre-hook traversal from 51353010."""

import argparse
import ast
import json
import logging
from pathlib import Path
import subprocess
import textwrap
import time
from unittest.mock import patch

import numpy as np

from pyRadPlan import IonPlan, PhotonPlan, load_tg119, generate_stf, settings
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine, PhotonPencilBeamSVDEngine
from pyRadPlan.dose.engines import _base_pencilbeam as base
from benchmark_particle_batches import compare_dose


def main():
    """Alternate traversals on identical engines; all numerical code is shared."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["protons", "photons"], default="protons")
    parser.add_argument("--pairs", type=int, default=11)
    parser.add_argument("--out", default="benchmark/pb_audit_results/batching_hooks/scalar")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "-C",
            str(root),
            "show",
            "51353010:pyRadPlan/dose/engines/_base_pencilbeam.py",
        ],
        text=True,
    )
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "PencilBeamEngineAbstract"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_calc_dose")
    namespace = vars(base).copy()
    exec(textwrap.dedent(ast.get_source_segment(source, method)), namespace)
    reference = namespace["_calc_dose"]
    current = base.PencilBeamEngineAbstract._calc_dose
    logging.basicConfig(level=logging.ERROR)
    settings.xp.prefer_gpu = False
    settings.xp.preferred_cpu_array_backend = "numpy"
    ct, cst = load_tg119()
    photons = args.mode == "photons"
    plan = PhotonPlan() if photons else IonPlan(radiation_mode="protons")
    plan.prop_stf = dict(bixel_width=10, gantry_angles=[0], couch_angles=[0])
    plan.prop_dose_calc = dict(dose_grid={"resolution": [5] * 3})
    if photons:
        plan.prop_dose_calc["enable_dij_sampling"] = False
    stf = generate_stf(ct, cst, plan)
    engine_type = PhotonPencilBeamSVDEngine if photons else ParticleHongPencilBeamEngine

    def run(method):
        engine = engine_type(plan.model_copy(deep=True))
        engine.console_progress = False
        with patch.object(base.PencilBeamEngineAbstract, "_calc_dose", method):
            start = time.perf_counter()
            dose = engine.calc_dose_influence(ct, cst, stf.model_copy(deep=True))
            elapsed = time.perf_counter() - start
        return elapsed, dose

    compare_dose(run(reference)[1], run(current)[1])
    samples = {"before_hooks": [], "scalar_hooks": []}
    for i in range(args.pairs):
        outputs = {}
        for label in list(samples)[:: 1 if i % 2 == 0 else -1]:
            elapsed, outputs[label] = run(reference if label == "before_hooks" else current)
            samples[label].append(elapsed)
        errors = compare_dose(outputs["before_hooks"], outputs["scalar_hooks"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"numpy_{args.mode}.json").write_text(
        json.dumps(
            dict(
                config=vars(args),
                reference="51353010 _calc_dose only; all other code current",
                samples_s=samples,
                max_abs_errors=errors,
            ),
            indent=2,
        )
        + "\n"
    )
    print({k: float(np.median(v)) for k, v in samples.items()}, flush=True)


if __name__ == "__main__":
    main()
