"""Isolate depth sampling using radiological arrays captured from a real dose call."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

import profile_pb_audit as audit
from profile_patient_depth_reference import legacy_depth_selection
from pyRadPlan.core.xp_utils import to_numpy


def main():
    """Capture a proton beam, then compare identical depth-array consumers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    parser.add_argument("--trace-on-ct-grid", action="store_true")
    args = parser.parse_args()
    original = audit.PB._sample_radiological_depths
    captured = {}

    def capture(self, arrays):
        captured.update(engine=self, arrays=arrays)
        return original(self, arrays)

    audit.PB._sample_radiological_depths = capture
    out = Path("benchmark/pb_audit_results/patient_depth/depth_sampling")
    out.mkdir(exist_ok=True)
    sys.argv = [
        sys.argv[0],
        "--backend",
        args.backend,
        "--mode",
        "protons",
        "--runs",
        "1",
        "--out",
        str(out),
    ]
    if args.trace_on_ct_grid:
        sys.argv.append("--trace-on-ct-grid")
    audit.main()
    audit.PB._sample_radiological_depths = original
    engine, arrays = captured["engine"], captured["arrays"]
    methods = dict(current=original, legacy=legacy_depth_selection)
    seconds = {name: [] for name in methods}
    for i in range(6):
        outputs = {}
        for name in list(methods) if i % 2 else list(reversed(methods)):
            audit.sync(args.backend)
            start = time.perf_counter()
            outputs[name] = methods[name](engine, arrays)
            audit.sync(args.backend)
            if i:
                seconds[name].append(time.perf_counter() - start)
        for a, b in zip(outputs["current"], outputs["legacy"]):
            np.testing.assert_allclose(to_numpy(a), to_numpy(b), rtol=3e-6, atol=3e-5)
    engine._depth_resampler = None
    audit.sync(args.backend)
    start = time.perf_counter()
    original(engine, arrays)
    audit.sync(args.backend)
    with_creation = time.perf_counter() - start
    result = dict(
        backend=args.backend,
        trace_on_ct=args.trace_on_ct_grid,
        seconds=seconds,
        current_with_creation=with_creation,
    )
    label = args.backend + ("_ct" if args.trace_on_ct_grid else "_dose")
    (out / (label + "_sampling.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
