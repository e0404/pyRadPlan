"""Summarize saved audit measurements and compare CPU/GPU numerical output."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

root = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark/pb_audit_results")
records = []
for path in root.glob("*.json"):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "args" in data:
        records.append((path, data))

print(
    "BASELINES: mode machine backend grid bixel forward sampled LET rays bixels median [min,max]"
)
for path, data in records:
    a = data["args"]
    if a["variant"] != "baseline":
        continue
    print(
        a["mode"],
        a["machine"],
        a["backend"],
        a["grid"],
        a["bixel"],
        a["forward"],
        a["sampling"],
        a.get("with_let", False),
        data["rays"],
        data["beamlets"],
        round(data["median_seconds"], 4),
        [round(v, 4) for v in data["warm_seconds"]],
    )

paired_results = []
print("\nPAIRED: file baseline variant speedup range_of_pair_ratios worst_error")
for path in root.glob("*_paired.json"):
    data = json.loads(path.read_text())
    a, b = np.asarray(data["baseline"]), np.asarray(data["variant"])
    record = dict(
        file=path.name,
        baseline=float(np.median(a)),
        variant=float(np.median(b)),
        speedup=float(np.median(a) / np.median(b)),
        pair_ratio_range=[float(np.min(a / b)), float(np.max(a / b))],
        worst_peak_normalized_error=max(c["max_abs_over_peak"] for c in data["comparisons"]),
        identical_indices=(
            all(c["identical_indices"] for c in data["comparisons"])
            if all("identical_indices" in c for c in data["comparisons"])
            else None
        ),
    )
    paired_results.append(record)
    print(json.dumps(record))

comparisons = []
print("\nCPU/GPU COMPARISONS")
for path, data in records:
    a = data["args"]
    if a["backend"] != "numpy" or a["variant"] != "baseline":
        continue
    other = root / path.name.replace("_numpy_", "_cupy_")
    if not other.exists():
        continue
    details = {}
    if a["forward"]:
        x = np.load(path.with_suffix(".npy"))
        y = np.load(other.with_suffix(".npy"))
        error = x.astype(np.float64) - y
        max_abs = float(np.max(np.abs(error)))
        peak = float(np.max(np.abs(x)))
        l2 = float(np.linalg.norm(error.ravel()) / np.linalg.norm(x.ravel()))
        same_indices = None
        zeros = None
    else:
        x = sparse.load_npz(path.with_suffix(".npz"))
        y = sparse.load_npz(other.with_suffix(".npz"))
        error = x.astype(np.float64) - y
        max_abs = float(np.max(np.abs(error.data), initial=0))
        peak = float(np.max(np.abs(x.data)))
        l2 = float(np.linalg.norm(error.data) / np.linalg.norm(x.data))
        same_indices = bool(
            np.array_equal(x.indices, y.indices) and np.array_equal(x.indptr, y.indptr)
        )
        zeros = int(np.count_nonzero(x.data == 0))
        if not same_indices:
            support_delta = x.astype(bool).astype(np.int8) - y.astype(bool).astype(np.int8)
            details["support_mismatch_entries"] = int(support_delta.nnz)
            k = int(np.argmax(np.abs(error.data)))
            row = int(error.indices[k])
            col = int(np.searchsorted(error.indptr, k, side="right") - 1)
            details["maximum_error_entry"] = dict(
                row=row, column=col, numpy=float(x[row, col]), cupy=float(y[row, col])
            )
        if a.get("with_let"):
            let_path = path.with_name(path.stem + "_let_dose.npz")
            let_other = other.with_name(other.stem + "_let_dose.npz")
            if let_path.exists() and let_other.exists():
                lx, ly = sparse.load_npz(let_path), sparse.load_npz(let_other)
                ld = lx.astype(np.float64) - ly
                details["let_peak_normalized_error"] = float(
                    np.max(np.abs(ld.data), initial=0) / np.max(np.abs(lx.data))
                )
                details["let_relative_l2"] = float(
                    np.linalg.norm(ld.data) / np.linalg.norm(lx.data)
                )
    record = dict(
        file=path.name,
        peak_normalized_max_error=max_abs / peak,
        relative_l2=l2,
        same_indices=same_indices,
        stored_zeros_numpy=zeros,
        **details,
    )
    comparisons.append(record)
    print(json.dumps(record))

(root / "summary.json").write_text(
    json.dumps(dict(paired=paired_results, backend_comparisons=comparisons), indent=2)
)
