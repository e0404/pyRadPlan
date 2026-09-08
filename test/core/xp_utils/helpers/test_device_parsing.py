"""Tests for DLPack device parsing and detection helpers."""

import importlib.util

import array_api_compat
import array_api_strict as xps
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import cupy_available
from pyRadPlan.core.xp_utils.helpers import (
    DLPACK_CPU,
    _parse_device_to_dlpack,
    device_cache_key,
    dlpack_to_backend_device,
    get_device_info,
)

HAS_JAX = importlib.util.find_spec("jax") is not None


def test_parse_none_and_tuple():
    assert _parse_device_to_dlpack(None) is None
    assert _parse_device_to_dlpack((DLPACK_CPU, 0)) == (DLPACK_CPU, 0)


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("cpu", (1, 0)),
        ("gpu", (2, 0)),
        ("cuda", (2, 0)),
        ("cuda:1", (2, 1)),
        ("gpu:2", (2, 2)),
    ],
)
def test_parse_device_strings(spec, expected):
    assert _parse_device_to_dlpack(spec) == expected


def test_parse_array_api_strict_default_device():
    """The strict Device object must map to CPU (matched by type, not by repr)."""
    device = xps.asarray([1.0]).device
    assert _parse_device_to_dlpack(device) == (DLPACK_CPU, 0)


def test_parse_array_api_strict_non_default_device_rejected():
    """array-api-strict pseudo-devices have no DLPack equivalent and must not be silently accepted."""
    info = xps.__array_namespace_info__()
    others = [d for d in info.devices() if d != info.default_device()]
    if not others:
        pytest.skip("array-api-strict exposes no non-default devices")

    with pytest.raises(ValueError, match="no DLPack equivalent"):
        _parse_device_to_dlpack(others[0])


def test_device_repr_containing_cpu_is_not_classified_as_cpu():
    """A device whose repr merely mentions "cpu" must not be treated as a CPU device."""

    class SneakyDevice:
        def __repr__(self):
            return "cuda:1 (spilled from cpu)"

    with pytest.raises(ValueError, match="Invalid device specification"):
        _parse_device_to_dlpack(SneakyDevice())


def test_get_device_info_warns_for_unknown_object():
    """Assuming CPU for an unidentifiable array must not happen silently."""

    class NotAnArray:
        pass

    with pytest.warns(UserWarning, match="Cannot determine the device"):
        assert get_device_info(NotAnArray()) == (DLPACK_CPU, 0)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
def test_jax_device_round_trip():
    """A JAX device must survive parse -> backend-object conversion."""
    import jax
    import jax.numpy as jnp

    device = jax.devices()[0]
    parsed = _parse_device_to_dlpack(device)
    assert dlpack_to_backend_device(jnp, parsed) == device


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
def test_jax_device_index_out_of_range():
    import jax.numpy as jnp

    with pytest.raises(ValueError, match="out of range"):
        dlpack_to_backend_device(jnp, (DLPACK_CPU, 99))


def test_device_cache_key_prefers_the_backend_device_object():
    """Hashable device objects identify a device exactly; the DLPack tuple does not."""
    cpu = xps.asarray([1.0], device=xps.Device("CPU_DEVICE"))
    other = xps.asarray([1.0], device=xps.Device("device1"))

    # both report the same DLPack device, so the tuple alone cannot be the cache key
    assert get_device_info(cpu) == get_device_info(other) == (DLPACK_CPU, 0)
    assert device_cache_key(cpu) != device_cache_key(other)
    assert device_cache_key(cpu) == device_cache_key(xps.asarray([2.0, 3.0]))
    assert {device_cache_key(cpu), device_cache_key(other)}


def test_device_cache_key_is_hashable_for_numpy():
    assert hash(device_cache_key(np.asarray([1.0]))) is not None


@pytest.mark.skipif(not cupy_available(), reason="CuPy / CUDA is not available.")
def test_device_cache_key_falls_back_to_dlpack_for_unhashable_devices():
    """``cupy.cuda.Device`` is unhashable, so the DLPack tuple stands in for it."""
    import array_api_compat.cupy as cp  # noqa: PLC0415

    arr = cp.asarray([1.0])
    with pytest.raises(TypeError):
        hash(array_api_compat.device(arr))

    key = device_cache_key(arr)
    assert key == get_device_info(arr)
    assert hash(key) is not None
