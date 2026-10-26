"""Controlled microbenchmarks; the forward model is a synthetic Gaussian, not a dose engine."""
# ruff: noqa: PLC0415

import json
from pathlib import Path
import time

import cupy as cp
import numpy as np


def host(a):
    """Return a host array, synchronizing device output when necessary."""
    return cp.asnumpy(a) if isinstance(a, cp.ndarray) else a


def measure(fn, gpu=False, repeats=7):
    """Measure warmed, synchronized wall times."""
    for _ in range(2):
        fn()
    samples = []
    for _ in range(repeats):
        if gpu:
            cp.cuda.get_current_stream().synchronize()
        t = time.perf_counter()
        value = fn()
        if gpu:
            cp.cuda.get_current_stream().synchronize()
        samples.append(time.perf_counter() - t)
    return {"median_s": float(np.median(samples)), "samples_s": samples}, value


def main():
    """Compare indexing, reductions, and synthetic forward-dose schedules."""
    rng = np.random.default_rng(42)
    results = {}
    for xp in (np, cp):
        backend = xp.__name__
        results[backend] = result = {}
        arrays = [xp.asarray(rng.normal(size=100_000)) for _ in range(4)]
        for density in (0.05, 0.5):
            mask = xp.asarray(rng.random(100_000) < density)

            def repeated():
                return [a[mask] for a in arrays]

            def compact():
                indices = xp.nonzero(mask)[0]
                return [a[indices] for a in arrays]

            a, av = measure(repeated, xp is cp, repeats=30)
            b, bv = measure(compact, xp is cp, repeats=30)
            assert all(np.array_equal(host(x), host(y)) for x, y in zip(av, bv))
            result[f"compact_{density}"] = {"repeated_mask": a, "one_compaction": b}

        cluster = xp.asarray(rng.integers(0, 300, 100_000))
        dose = xp.asarray(rng.random(100_000).astype(np.float32))

        def sorted_max():
            order = xp.argsort(dose, kind="stable")
            order = order[xp.argsort(cluster[order], kind="stable")]
            sc = cluster[order]
            end = xp.concatenate((sc[:-1] != sc[1:], xp.ones(1, dtype=bool)))
            out = xp.zeros(300, dtype=dose.dtype)
            out[sc[end]] = dose[order][end]
            return out

        def scatter_max():
            out = xp.zeros(300, dtype=dose.dtype)
            xp.maximum.at(out, cluster, dose)
            return out

        a, av = measure(sorted_max, xp is cp)
        b, bv = measure(scatter_max, xp is cp)
        assert np.array_equal(host(av), host(bv))
        result["cluster_max"] = {"two_sorts": a, "scatter_max": b}

    # Same synthetic inputs on CPU/GPU; compare execution schedules, not dose physics.
    nv, ns = 65536, 256
    x, z = rng.uniform(-100, 100, (2, nv)).astype(np.float32)
    depth = rng.uniform(0, 250, nv).astype(np.float32)
    sx, sz = rng.uniform(-40, 40, (2, ns)).astype(np.float32)
    sigma = rng.uniform(3, 10, ns).astype(np.float32)
    mu = rng.uniform(0.1, 1, ns).astype(np.float32)
    reference = None
    for xp in (np, cp):
        xx, zz, dd, bx, bz, ss, ww = [xp.asarray(v) for v in (x, z, depth, sx, sz, sigma, mu)]

        def beamlet_host_sink():
            out = np.zeros(nv, dtype=np.float32)
            for j in range(ns):
                r2 = (xx - bx[j]) ** 2 + (zz - bz[j]) ** 2
                s2 = ss[j] ** 2 + np.float32(0.02) * dd
                ix = xp.nonzero(r2 <= np.float32(10) * s2)[0]
                value = ww[j] * xp.exp(-r2[ix] / (2 * s2[ix])) / (2 * np.pi * s2[ix])
                out[host(ix)] += host(value)
            return out

        def tiled_device_sink():
            out = xp.zeros(nv, dtype=xp.float32)
            for start in range(0, ns, 16):
                sl = slice(start, start + 16)
                r2 = (xx[None, :] - bx[sl, None]) ** 2 + (zz[None, :] - bz[sl, None]) ** 2
                s2 = ss[sl, None] ** 2 + np.float32(0.02) * dd[None, :]
                values = ww[sl, None] * xp.exp(-r2 / (2 * s2)) / (2 * np.pi * s2)
                out += xp.sum(xp.where(r2 <= 10 * s2, values, 0), axis=0)
            return host(out)

        a, av = measure(beamlet_host_sink, xp is cp)
        b, bv = measure(tiled_device_sink, xp is cp)
        if reference is None:
            reference = av
        assert np.allclose(reference, bv, atol=2e-6 * reference.max(), rtol=2e-5)
        results[xp.__name__]["synthetic_forward"] = dict(
            voxels=nv,
            beamlets=ns,
            loop_host_sink=a,
            tile_16_device_sink=b,
            tile_peak_normalized_error=float(np.max(np.abs(reference - bv)) / reference.max()),
        )
        if xp is cp:
            kernel = cp.RawKernel(
                r"""
            extern "C" __global__ void voxel_forward(
                const float* x, const float* z, const float* depth,
                const float* bx, const float* bz, const float* sigma, const float* w,
                int nv, int ns, float* out) {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                if (i >= nv) return;
                float total = 0.0f;
                for (int j=0; j<ns; ++j) {
                    float dx=x[i]-bx[j], dz=z[i]-bz[j];
                    float r2=dx*dx+dz*dz;
                    float s2=sigma[j]*sigma[j]+0.02f*depth[i];
                    if(r2 <= 10.0f*s2)
                        total += w[j]*expf(-r2/(2.0f*s2))/(6.283185307179586f*s2);
                }
                out[i]=total;
            }""",
                "voxel_forward",
            )

            def voxel_gather():
                out = cp.empty(nv, dtype=cp.float32)
                kernel(
                    ((nv + 255) // 256,),
                    (256,),
                    (xx, zz, dd, bx, bz, ss, ww, np.int32(nv), np.int32(ns), out),
                )
                return cp.asnumpy(out)

            c, cv = measure(voxel_gather, True)
            assert np.allclose(reference, cv, atol=2e-6 * reference.max(), rtol=2e-5)
            results["cupy"]["synthetic_forward"].update(
                voxel_gather=c,
                voxel_peak_normalized_error=float(
                    np.max(np.abs(reference - cv)) / reference.max()
                ),
            )

    from pyRadPlan import load_tg119

    ct, cst = load_tg119()
    indices = [v.indices_numpy for v in cst.vois]

    def union_unique():
        return np.unique(np.concatenate(indices))

    def union_mask():
        mask = np.zeros(ct.grid.num_voxels, dtype=bool)
        for ix in indices:
            mask[ix] = True
        return np.flatnonzero(mask)

    a, av = measure(union_unique)
    b, bv = measure(union_mask)
    assert np.array_equal(av, bv)
    results["numpy"]["tg119_union"] = dict(
        unique=a, mask=b, voxels=ct.grid.num_voxels, input_indices=sum(v.size for v in indices)
    )
    output = Path("benchmark/pb_audit_results/microbenchmarks.json")
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
