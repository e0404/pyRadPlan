"""Benchmark the outcome preprocessor's dose-grid gradient mapping: old vs new.

Two strategies map the CNN's model-grid gradient back to the optimization dose grid:

- **old** (forward remap): resample dose->model under ``no_grad`` with the model-grid
  dose as the autograd leaf, then map the model-grid gradient to the dose grid with a
  *second* forward ``grid_sample`` (model->dose). This is only an approximation of the
  transpose of the first resampling.
- **new** (exact adjoint): make the *dose-grid* dose the autograd leaf and backprop
  *through* the dose->model ``grid_sample``. ``dose.grad`` is then the exact adjoint —
  no second resample, and the model->dose coordinate tensor is not needed.

Both are driven by the *same* upstream model-grid gradient (a fixed random field standing
in for ``d loss / d dose_model``), so the comparison isolates the mapping itself.

Run directly for a table (time / memory / agreement)::

    python benchmark/benchmark_outcome_gradient.py

or via pytest-benchmark::

    pytest benchmark/benchmark_outcome_gradient.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None


# %% Grid geometry / coordinate helpers (mirrors OutcomeCnnPreprocessor)


def _grid(size, spacing, origin=(0.0, 0.0, 0.0)):
    return {
        "size": tuple(int(s) for s in size),
        "origin": tuple(float(o) for o in origin),
        "spacing": tuple(float(s) for s in spacing),
        "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }


def _compute_sample_coords(src_grid: dict, dst_grid: dict, device) -> "torch.Tensor":
    """Normalized [-1,1] coords mapping ``dst`` voxel centers into ``src`` (align_corners)."""
    dx, dy, dz = dst_grid["size"]

    ix = torch.arange(dx, dtype=torch.float32)
    iy = torch.arange(dy, dtype=torch.float32)
    iz = torch.arange(dz, dtype=torch.float32)
    gz, gy, gx = torch.meshgrid(iz, iy, ix, indexing="ij")
    indices = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)

    d_origin = torch.tensor(dst_grid["origin"], dtype=torch.float32)
    d_spacing = torch.tensor(dst_grid["spacing"], dtype=torch.float32)
    d_dir = torch.tensor(dst_grid["direction"], dtype=torch.float32).reshape(3, 3)
    phys = d_origin + (indices * d_spacing) @ d_dir.T

    s_origin = torch.tensor(src_grid["origin"], dtype=torch.float32)
    s_spacing = torch.tensor(src_grid["spacing"], dtype=torch.float32)
    s_dir = torch.tensor(src_grid["direction"], dtype=torch.float32).reshape(3, 3)
    s_dir_inv = torch.linalg.inv(s_dir)
    src_idx = ((phys - s_origin) @ s_dir_inv.T) / s_spacing

    s_size = torch.tensor(src_grid["size"], dtype=torch.float32)
    normalized = 2.0 * src_idx / (s_size - 1) - 1.0
    return normalized.reshape(1, int(dz), int(dy), int(dx), 3).to(device)


# %% The two gradient-mapping strategies (shared upstream gradient ``w``)


def grad_old(dose_flat, coords_d2m, coords_m2d, w, dose_shape):
    """Forward remap: model-grid leaf + a second grid_sample (model->dose)."""
    dose = dose_flat.reshape(dose_shape)
    with torch.no_grad():
        dose_model = F.grid_sample(
            dose[None, None], coords_d2m, mode="bilinear", padding_mode="zeros", align_corners=True
        )
    leaf = dose_model.detach().requires_grad_(True)
    (leaf * w).sum().backward()
    grad_dose = F.grid_sample(
        leaf.grad, coords_m2d, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return grad_dose.reshape(-1)


def grad_new(dose_flat, coords_d2m, w, dose_shape):
    """Exact adjoint: dose-grid leaf, backprop through the dose->model grid_sample."""
    dose = dose_flat.reshape(dose_shape).requires_grad_(True)
    dose_model = F.grid_sample(
        dose[None, None], coords_d2m, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    (dose_model * w).sum().backward()
    return dose.grad.reshape(-1)


# %% Case construction

_CASES = {
    # name: (model_size, model_spacing, dose_size, dose_spacing)  -- same physical extent
    "matched": ((100, 100, 40), (3.0, 3.0, 2.5), (100, 100, 40), (3.0, 3.0, 2.5)),
    "fine": ((100, 100, 40), (3.0, 3.0, 2.5), (200, 200, 80), (1.5, 1.5, 1.25)),
}


def _make_case(name: str, device):
    model_size, model_spacing, dose_size, dose_spacing = _CASES[name]
    model_grid = _grid(model_size, model_spacing)
    dose_grid = _grid(dose_size, dose_spacing)

    coords_d2m = _compute_sample_coords(dose_grid, model_grid, device)  # sample dose at model pts
    coords_m2d = _compute_sample_coords(model_grid, dose_grid, device)  # sample model at dose pts

    rng = np.random.default_rng(0)
    mz, my, mx = model_size[2], model_size[1], model_size[0]
    dz, dy, dx = dose_size[2], dose_size[1], dose_size[0]

    w = torch.tensor(rng.standard_normal((1, 1, mz, my, mx)), dtype=torch.float32, device=device)
    dose_flat = torch.tensor(
        rng.uniform(0.0, 60.0, size=dz * dy * dx), dtype=torch.float32, device=device
    )
    return dose_flat, coords_d2m, coords_m2d, w, (dz, dy, dx)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# %% pytest-benchmark tests

_BACKENDS = [("cpu", "cpu")]
if torch is not None and torch.cuda.is_available():
    _BACKENDS.append(("cuda", "cuda"))


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
@pytest.mark.parametrize("strategy", ["old", "new"])
@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("dev_name,dev", _BACKENDS, ids=[b[0] for b in _BACKENDS])
def test_grad_mapping(benchmark, strategy, case, dev_name, dev):
    device = torch.device(dev)
    dose_flat, c_d2m, c_m2d, w, dshape = _make_case(case, device)

    if strategy == "old":
        fn = lambda: grad_old(dose_flat, c_d2m, c_m2d, w, dshape)  # noqa: E731
    else:
        fn = lambda: grad_new(dose_flat, c_d2m, w, dshape)  # noqa: E731

    fn()  # warmup
    _sync(device)
    result = benchmark(lambda: (_sync(device), fn(), _sync(device))[1])
    assert torch.isfinite(result).all()


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_old_new_agreement():
    """Old and new should agree closely when the grids match; differ more when fine."""
    device = torch.device("cpu")
    for case, min_cos in (("matched", 0.98), ("fine", 0.5)):
        dose_flat, c_d2m, c_m2d, w, dshape = _make_case(case, device)
        g_old = grad_old(dose_flat, c_d2m, c_m2d, w, dshape)
        g_new = grad_new(dose_flat, c_d2m, w, dshape)
        assert torch.isfinite(g_old).all() and torch.isfinite(g_new).all()
        cos = torch.nn.functional.cosine_similarity(g_old, g_new, dim=0).item()
        assert cos >= min_cos, f"{case}: cosine {cos:.3f} < {min_cos}"


# %% Standalone table (time / memory / agreement)


def _agreement(g_old, g_new):
    diff = (g_old - g_new).abs()
    denom = g_new.norm().item() or 1.0
    cos = torch.nn.functional.cosine_similarity(g_old, g_new, dim=0).item()
    return diff.max().item(), (g_old - g_new).norm().item() / denom, cos


def _time(fn, device, reps=30):
    import time

    fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / reps * 1e3  # ms


def _peak_cuda(fn, device):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6  # MB


def _run_table():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        print(f"\n=== device: {device.type} ===")
        for case in _CASES:
            dose_flat, c_d2m, c_m2d, w, dshape = _make_case(case, device)
            f_old = lambda: grad_old(dose_flat, c_d2m, c_m2d, w, dshape)  # noqa: E731
            f_new = lambda: grad_new(dose_flat, c_d2m, w, dshape)  # noqa: E731

            t_old, t_new = _time(f_old, device), _time(f_new, device)
            g_old, g_new = f_old(), f_new()
            max_abs, rel_l2, cos = _agreement(g_old, g_new)

            coords_m2d_mb = c_m2d.numel() * 4 / 1e6  # persistent tensor old needs, new drops
            line = (
                f"[{case:>7}] old {t_old:7.2f} ms | new {t_new:7.2f} ms | "
                f"speedup {t_old / t_new:4.2f}x | drop coords_m2d {coords_m2d_mb:6.1f} MB | "
                f"agree: maxabs {max_abs:.2e} relL2 {rel_l2:.2e} cos {cos:.4f}"
            )
            if device.type == "cuda":
                p_old, p_new = _peak_cuda(f_old, device), _peak_cuda(f_new, device)
                line += f" | peak old {p_old:6.1f} new {p_new:6.1f} MB"
            print(line)


if __name__ == "__main__":
    if torch is None:
        raise SystemExit("PyTorch is required for this benchmark.")
    torch.manual_seed(0)
    _run_table()
