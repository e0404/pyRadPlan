"""Run interpolation lifecycle benchmarks and retain compact, reviewable results."""

import json
from pathlib import Path
import subprocess
import sys


def main():
    """Measure construction/evaluation/one-shot costs and omit raw timing samples."""
    root = Path("benchmark/pb_audit_results")
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "array_backend_interpolation_micro.raw.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "benchmark/benchmark_interpnd.py",
            "-k",
            "lifecycle",
            "--benchmark-min-rounds=5",
            "--benchmark-max-time=0.1",
            "--benchmark-warmup=on",
            "--benchmark-json=" + str(raw),
            "-q",
        ],
        check=True,
    )
    data = json.loads(raw.read_text())
    summary = {key: data[key] for key in ("machine_info", "commit_info", "datetime")}
    summary["benchmarks"] = [
        {
            "name": result["name"],
            "params": result["params"],
            "stats": {key: value for key, value in result["stats"].items() if key != "data"},
        }
        for result in data["benchmarks"]
    ]
    (root / "array_backend_interpolation_micro.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
