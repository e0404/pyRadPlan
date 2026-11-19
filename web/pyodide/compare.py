"""Compare two pipeline result files written by ``pipeline.py`` / ``run_pipeline.mjs``.

python web/pyodide/compare.py results/native_photons_fast.npz results/pyodide_photons_fast.npz
"""

import json
import sys

import numpy as np


def _rel(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    scale = float(np.abs(a).max()) or 1.0
    return {
        "max_abs_diff_rel_to_max": float(np.abs(a - b).max() / scale),
        "rel_l2": float(np.linalg.norm(a - b) / (np.linalg.norm(a) or 1.0)),
        "frac_voxels_gt_1pct_of_max": float(np.mean(np.abs(a - b) > 0.01 * scale)),
    }


def compare(path_a: str, path_b: str) -> dict:
    """Return relative differences of the arrays both files contain."""
    a, b = np.load(path_a), np.load(path_b)
    out = {"a": path_a, "b": path_b}
    for key in ("dose", "unit_dose", "weights", "bixel_sums", "ray_positions", "iso_centers"):
        if key in a and key in b:
            if a[key].shape != b[key].shape:
                out[key] = f"shape mismatch {a[key].shape} vs {b[key].shape}"
            else:
                out[key] = _rel(a[key], b[key])
    if "target" in a and "dose" in a and "dose" in b:
        t = a["target"]
        out["target_mean_a"] = float(a["dose"][t].mean())
        out["target_mean_b"] = float(b["dose"][t].mean())
    ia, ib = json.loads(str(a["info"])), json.loads(str(b["info"]))
    out["total_s"] = {"a": ia["total_s"], "b": ib["total_s"]}
    out["timings_s"] = {k: [ia["timings_s"][k], ib["timings_s"].get(k)] for k in ia["timings_s"]}
    return out


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=2))
