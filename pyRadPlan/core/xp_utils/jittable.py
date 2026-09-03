"""Jit-compiled execution paths for performance-critical array kernels."""

import functools
import warnings
from typing import Callable, Iterable, Optional

import array_api_compat

from ..._settings import get_settings


def _enabled_jit_backends() -> set[str]:
    """Backends allowed to run jit-compiled kernels per ``settings.xp.jit_backends``."""
    value = get_settings().xp.jit_backends
    return {backend.strip().lower() for backend in value.split(",") if backend.strip()}


def _backend_name(arr) -> Optional[str]:
    if array_api_compat.is_numpy_array(arr):
        return "numpy"
    if array_api_compat.is_torch_array(arr):
        return "torch"
    if array_api_compat.is_cupy_array(arr):
        return "cupy"
    if array_api_compat.is_jax_array(arr):
        return "jax"
    return None


class JittableKernel:
    """
    Wrap a generic array-API kernel with jit-compiled execution paths per backend.

    Dispatch happens on the namespace of the first array argument, and only for
    backends enabled in ``settings.xp.jit_backends``:

    1. A jit-compiled implementation registered for that backend via
       :meth:`register` (e.g. a numba one for NumPy) is used when present. It may
       return ``NotImplemented`` to fall back (e.g. for an unsupported dtype).
    2. Otherwise, when the kernel declares the backend jit-capable
       (``backends=...``), the generic implementation is compiled with the
       backend's own jit (``jax.jit``, ``torch.compile``) and cached. This
       requires the kernel to be traceable with static output shapes; a failing
       compilation falls back to the generic code with a one-time warning.

    Disabled backends and every other case run the generic array-API
    implementation unchanged.
    """

    def __init__(self, fn: Callable, *, backends: Iterable[str] = ("jax",)):
        functools.update_wrapper(self, fn)
        self._generic = fn
        self._jit_backends = frozenset(backend.lower() for backend in backends)
        self._jitted: dict[str, Callable] = {}
        self._jit_validated: set[str] = set()
        self._jit_failed: set[str] = set()
        self._impls: dict[str, Callable] = {}

    @property
    def generic(self) -> Callable:
        """The generic array-API implementation (useful for differential testing)."""
        return self._generic

    def register(self, backend: str) -> Callable:
        """Register a compiled implementation for the given backend's arrays (decorator)."""

        def decorator(fn: Callable) -> Callable:
            self._impls[backend.lower()] = fn
            return fn

        return decorator

    def _compile(self, backend: str) -> Optional[Callable]:
        if backend == "jax":
            import jax  # noqa: PLC0415

            return jax.jit(self._generic)
        if backend == "torch":
            import torch  # noqa: PLC0415

            return torch.compile(self._generic)
        return None

    def _call_jitted(self, backend: str, args) -> object:
        """Run the jitted variant, falling back to NotImplemented on compile failure."""
        compiled = self._jitted.get(backend)
        if compiled is None:
            try:
                compiled = self._compile(backend)
            except Exception as exc:  # noqa: BLE001 - any compiler failure falls back
                self._jit_failed.add(backend)
                warnings.warn(
                    f"jit compilation of kernel '{self.__name__}' for backend "
                    f"'{backend}' failed ({exc!r}); using the generic implementation.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return NotImplemented
            if compiled is None:
                self._jit_failed.add(backend)
                return NotImplemented
            self._jitted[backend] = compiled

        if backend in self._jit_validated:
            return compiled(*args)

        # backends like torch.compile only compile on the first real call
        try:
            result = compiled(*args)
        except Exception as exc:  # noqa: BLE001 - any compiler failure falls back
            self._jit_failed.add(backend)
            warnings.warn(
                f"jit execution of kernel '{self.__name__}' for backend "
                f"'{backend}' failed ({exc!r}); using the generic implementation.",
                RuntimeWarning,
                stacklevel=3,
            )
            return NotImplemented
        self._jit_validated.add(backend)
        return result

    def __call__(self, *args):
        """Dispatch to the best implementation for the namespace of the array arguments."""
        first = next((a for a in args if array_api_compat.is_array_api_obj(a)), None)
        backend = _backend_name(first) if first is not None else None

        if backend is not None and backend in _enabled_jit_backends():
            impl = self._impls.get(backend)
            if impl is not None:
                result = impl(*args)
                if result is not NotImplemented:
                    return result

            if backend in self._jit_backends and backend not in self._jit_failed:
                result = self._call_jitted(backend, args)
                if result is not NotImplemented:
                    return result

        return self._generic(*args)


def jittable(fn: Optional[Callable] = None, *, backends: Iterable[str] = ("jax",)):
    """
    Turn a generic array-API function into a :class:`JittableKernel`.

    Use as ``@jittable`` or ``@jittable(backends=("jax", "torch"))``. ``backends``
    names the backends whose own jit may compile the generic code (which must be
    traceable with static output shapes there); implementations registered via
    :meth:`JittableKernel.register` (e.g. numba for NumPy) are independent of it.
    Which backends actually run jit-compiled paths is steered globally by
    ``settings.xp.jit_backends`` (``PYRADPLAN_XP_JIT_BACKENDS``).
    """
    if fn is None:
        return lambda f: JittableKernel(f, backends=backends)
    return JittableKernel(fn)
