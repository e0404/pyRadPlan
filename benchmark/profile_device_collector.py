"""Process-local device collection experiments; public dose output remains SciPy.

Full collection constructs a CuPy CSC object before downloading the completed
buffers. Bounded collection flushes roughly 64 MiB of payload at a time. Neither
prototype changes the production collector or the public Dij contract.
"""

import time

import numpy as np

from pyRadPlan.core.xp_utils import to_numpy
from pyRadPlan.dose.engines._sparse_output import ensure_index_capacity


def install(engine_class, *, batch_bytes=None):  # noqa: PLR0915
    """Install influence-only collection hooks for NumPy/CuPy audit processes."""
    original_allocate = engine_class._allocate_quantity_matrices
    original_fill = engine_class._fill_dij
    original_finalize = engine_class._finalize_dose

    def sample_pool(engine):
        if "cupy" in engine.xp.__name__:
            used = engine.xp.get_default_memory_pool().used_bytes()
            metrics = engine._collector_metrics
            metrics["sampled_peak_pool_bytes"] = max(metrics["sampled_peak_pool_bytes"], used)

    def allocate(self, dij, names):
        result = original_allocate(self, dij, names)
        if self._calc_dose_direct:
            return result
        if not any("pending_rows" in s for s in dij["shared_structure"].flat if s is not None):
            self._collector_metrics = dict(
                flushes=0,
                download_calls=0,
                peak_pending_bytes=0,
                sampled_peak_pool_bytes=0,
                device_wrap_s=0.0,
                bulk_download_s=0.0,
            )
        for i, shared in enumerate(dij["shared_structure"].flat):
            if shared is None:
                continue
            if "pending_rows" not in shared:
                shared.update(
                    pending_rows=[],
                    pending_values={},
                    host_rows=[],
                    host_values={},
                    pending_bytes=0,
                )
                shared["indices"] = np.empty(0, dtype=shared["indptr"].dtype)
            for name in names:
                shared["pending_values"][name] = []
                shared["host_values"][name] = []
                dij[name].flat[i]["data"] = np.empty(0, dtype=np.float32)
        return result

    def flush(self, dij, scenario):
        shared = dij["shared_structure"].flat[scenario]
        if not shared["pending_rows"]:
            return
        xp = self.xp
        metrics = self._collector_metrics
        rows = xp.concat(shared["pending_rows"])
        arrays = {q: xp.concat(shared["pending_values"][q]) for q in self._computed_quantities}
        sample_pool(self)
        if batch_bytes is None and "cupy" in xp.__name__:
            from cupyx.scipy.sparse import csc_matrix  # noqa: PLC0415

            ptr = xp.asarray(shared["indptr"])
            xp.cuda.get_current_stream().synchronize()
            start = time.perf_counter()
            matrices = [
                csc_matrix(
                    (arrays[q], rows, ptr),
                    shape=(self.dose_grid.num_voxels, self._num_of_columns_dij),
                )
                for q in self._computed_quantities
            ]
            xp.cuda.get_current_stream().synchronize()
            metrics["device_wrap_s"] += time.perf_counter() - start
            sample_pool(self)
            # Keep the native objects alive through the copy, exercising their
            # real constructor, not just timing an unused tuple of buffers.
            rows = matrices[0].indices
            arrays = {q: m.data for q, m in zip(self._computed_quantities, matrices)}
        if "cupy" in xp.__name__:
            xp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        shared["host_rows"].append(to_numpy(rows))
        for q in self._computed_quantities:
            shared["host_values"][q].append(to_numpy(arrays[q]))
            shared["pending_values"][q].clear()
        metrics["bulk_download_s"] += time.perf_counter() - start
        metrics["download_calls"] += len(arrays) + 1
        metrics["flushes"] += 1
        shared["pending_rows"].clear()
        shared["pending_bytes"] = 0

    def fill(self, bixel, dij, stf, scenario, beam, ray, bixel_index, column):
        if self._calc_dose_direct:
            return original_fill(self, bixel, dij, stf, scenario, beam, ray, bixel_index, column)
        xp = self.xp
        shared = dij["shared_structure"].flat[scenario]
        size = bixel["ix"].shape[0]
        end = int(shared["indptr"][column]) + size
        ensure_index_capacity(shared, end)
        shared["indptr"][column + 1] = end
        shared["nnz"] += size
        if size:
            dtype = xp.int32 if shared["indptr"].dtype == np.int32 else xp.int64
            rows = xp.astype(bixel["ix"], dtype, copy=False)
            shared["pending_rows"].append(rows)
            shared["pending_bytes"] += rows.nbytes
            for q in self._computed_quantities:
                values = xp.astype(bixel[q], xp.float32, copy=False)
                shared["pending_values"][q].append(values)
                shared["pending_bytes"] += values.nbytes
            pending = sum(
                s["pending_bytes"] for s in dij["shared_structure"].flat if s is not None
            )
            self._collector_metrics["peak_pending_bytes"] = max(
                self._collector_metrics["peak_pending_bytes"], pending
            )
            sample_pool(self)
            if batch_bytes is not None and shared["pending_bytes"] >= batch_bytes:
                flush(self, dij, scenario)
        dij["beam_num"][column] = beam
        dij["ray_num"][column] = ray
        dij["bixel_num"][column] = bixel_index

    def finalize(self, dij):
        if not self._calc_dose_direct:
            for i, shared in enumerate(dij["shared_structure"].flat):
                if shared is None:
                    continue
                flush(self, dij, i)
                if shared["host_rows"]:
                    chunks = shared["host_rows"]
                    shared["indices"] = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
                for q in self._computed_quantities:
                    if shared["host_values"][q]:
                        chunks = shared["host_values"][q]
                        dij[q].flat[i]["data"] = (
                            chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
                        )
                for key in ("pending_rows", "pending_values", "host_rows", "host_values"):
                    del shared[key]
        return original_finalize(self, dij)

    engine_class._allocate_quantity_matrices = allocate
    engine_class._fill_dij = fill
    engine_class._finalize_dose = finalize
