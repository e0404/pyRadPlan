"""Run audit cases sequentially so CPU and GPU timings do not contend."""

import subprocess
import sys
from pathlib import Path

out = Path("benchmark/pb_audit_results")
out.mkdir(exist_ok=True)
cases = []
for mode in ["photons", "carbon", "VHEE", "helium", "oxygen", "field"]:
    for backend in ["numpy", "cupy"]:
        cases.append(["--mode", mode, "--backend", backend, "--profile"])
for backend in ["numpy", "cupy"]:
    cases.append(["--mode", "VHEE", "--machine", "Focused", "--backend", backend, "--profile"])
for mode in ["protons", "photons", "VHEE"]:
    for backend in ["numpy", "cupy"]:
        cases.append(["--mode", mode, "--backend", backend, "--variant", "combined", "--profile"])
for backend in ["numpy", "cupy"]:
    cases.append(["--mode", "photons", "--backend", backend, "--sampling", "--profile"])
for mode in ["protons", "photons"]:
    for backend in ["numpy", "cupy"]:
        cases.append(["--mode", mode, "--backend", backend, "--forward"])

if "--followup" in sys.argv:
    cases = []
    for backend in ["numpy", "cupy"]:
        cases.append(["--mode", "protons", "--backend", backend, "--profile"])
        cases.append(
            [
                "--mode",
                "protons",
                "--backend",
                backend,
                "--variant",
                "combined",
                "--paired",
                "--profile",
            ]
        )
        cases.append(
            [
                "--mode",
                "photons",
                "--backend",
                backend,
                "--sampling",
                "--variant",
                "scattermax",
                "--paired",
            ]
        )
        for mode in ["protons", "photons", "VHEE"]:
            cases.append(["--mode", mode, "--backend", backend, "--forward"])
    cases.append(["--mode", "photons", "--backend", "numpy", "--variant", "union", "--paired"])
    cases.append(["--mode", "field", "--backend", "cupy", "--variant", "fft", "--paired"])
    cases.append(
        [
            "--mode",
            "VHEE",
            "--machine",
            "Focused",
            "--backend",
            "numpy",
            "--variant",
            "hoist",
            "--paired",
        ]
    )
    for backend in ["numpy", "cupy"]:
        cases.append(
            ["--mode", "protons", "--backend", backend, "--grid", "3", "--bixel", "5", "--profile"]
        )

if "--final" in sys.argv:
    cases = []
    for backend in ["numpy", "cupy"]:
        cases.append(
            [
                "--mode",
                "photons",
                "--backend",
                backend,
                "--variant",
                "prepared",
                "--paired",
                "--profile",
            ]
        )
    for mode in ["protons", "photons"]:
        cases.append(
            [
                "--mode",
                mode,
                "--backend",
                "cupy",
                "--forward",
                "--variant",
                "device_forward",
                "--paired",
            ]
        )
    for backend in ["numpy", "cupy"]:
        cases.append(["--mode", "protons", "--backend", backend, "--let", "--profile"])

for i, case in enumerate(cases):
    print(f"CASE {i + 1}/{len(cases)}: {case}", flush=True)
    prefix = (
        "final" if "--final" in sys.argv else ("followup" if "--followup" in sys.argv else "suite")
    )
    with (out / f"{prefix}_{i:02d}.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "benchmark/profile_pb_audit.py", "--runs", "3", *case],
            stdout=log,
            stderr=log,
            check=False,
        )
    print(f"EXIT {result.returncode}", flush=True)
