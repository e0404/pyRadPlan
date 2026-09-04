"""Detection of clashing OpenMP runtimes in the current process.

Several scientific wheels vendor their own copy of the Intel/LLVM OpenMP runtime
(``libiomp5md.dll`` on Windows, ``libiomp5.so``/``libomp.so`` elsewhere).  When a
second copy is *initialized* in a process that already has one, that runtime
prints ``OMP: Error #15`` and calls ``abort()`` -- the interpreter dies on the
spot, with no Python exception to catch.

The abort happens at the first parallel region, not at load time, so an import
that appears to succeed can still take the process down much later.  The
functions here let pyRadPlan see the clash coming and refuse to use the offending
extension instead (see :mod:`pyRadPlan.optimization.solvers`, where ``ipyopt``
vendors a runtime that clashes with the one PyTorch ships).

Setting ``KMP_DUPLICATE_LIB_OK=TRUE`` tells the Intel runtime to continue anyway.
That is explicitly unsafe -- it may crash or silently produce wrong results -- so
pyRadPlan never *decides* to set it; it only honours the value the user has put in
the real environment or in pyRadPlan's ``.env`` file (see
:func:`duplicate_runtimes_allowed`).
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import sys
from functools import lru_cache
from importlib.util import find_spec
from typing import Optional

logger = logging.getLogger(__name__)

#: Runtime stems that abort the process when a second copy is initialized: the Intel
#: runtime and LLVM's, which share that check.  GNU's ``libgomp`` and MSVC's ``vcomp``
#: tolerate duplicates and are deliberately not listed.
_ABORTING_RUNTIMES = frozenset({"libiomp5md", "libiomp5", "libomp"})

#: Hash suffix that ``delvewheel``/``auditwheel`` append when vendoring a shared
#: library (e.g. ``libiomp5md-bbfd5d1c3843841454f68a54ee61f1f8.dll``).
_VENDOR_HASH = re.compile(r"-[0-9a-f]{8,32}$")

#: Directories a wheel may put its vendored shared libraries in, relative to the
#: package directory's parent: ``delvewheel`` uses ``<pkg>.libs``, ``delocate``
#: ``<pkg>.dylibs``; some wheels drop them into the package directory itself.
_VENDOR_DIR_SUFFIXES = (".libs", ".dylibs")


def _kmp_env_from_dotenv() -> Optional[str]:
    """Look up ``KMP_DUPLICATE_LIB_OK`` in pyRadPlan's ``.env`` file, if any.

    Only this one key is read -- never the rest of the file (which may hold API
    keys) -- and ``python-dotenv`` is an optional convenience here exactly as it
    is for :func:`pyRadPlan.ai.agents.load_ai_env`: silently unavailable if the
    package or the file is missing.
    """
    try:
        from dotenv import dotenv_values, find_dotenv  # noqa: PLC0415
    except ImportError:
        return None
    from pyRadPlan._settings import ENV_FILE  # noqa: PLC0415 - deferred to dodge import cycles

    path = find_dotenv(ENV_FILE, usecwd=True)
    if not path:
        return None
    return dotenv_values(path).get("KMP_DUPLICATE_LIB_OK")


def duplicate_runtimes_allowed() -> bool:
    """Return whether ``KMP_DUPLICATE_LIB_OK`` permits several OpenMP runtimes.

    Unlike the rest of pyRadPlan's configuration, this does not go through
    pydantic-settings: that layer only feeds a ``.env`` file into typed settings
    fields, it never touches ``os.environ`` -- and the Intel/LLVM OpenMP runtime
    reads this variable from the *real* process environment (C ``getenv()``), not
    from anything Python-side. So a value found only in ``.env`` is copied into
    ``os.environ`` here (an explicitly set, even empty, environment variable is
    never overridden by it), the same way :func:`pyRadPlan.ai.agents.load_ai_env`
    already does for provider API keys.

    Returns
    -------
    bool
        ``True`` if the user opted in to the (unsafe) duplicate-runtime workaround.
    """
    value = os.environ.get("KMP_DUPLICATE_LIB_OK")
    if value is None:
        value = _kmp_env_from_dotenv()
        if value is not None:
            os.environ["KMP_DUPLICATE_LIB_OK"] = value
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def runtime_key(path: str) -> Optional[str]:
    """Return the OpenMP runtime *path* provides, or ``None`` if it is not one.

    The file name is reduced to a comparable stem: the extension (including a
    ``.so.1`` style version) and any vendoring hash suffix are stripped, so
    ``libiomp5md-bbfd5d1c….dll`` and ``libiomp5md.dll`` compare equal.

    Parameters
    ----------
    path : str
        Path of a loaded or on-disk shared library.

    Returns
    -------
    str or None
        The runtime stem (e.g. ``"libiomp5md"``) for a runtime that aborts on
        duplicate initialization, otherwise ``None``.
    """
    name = re.split(r"[\\/]", path)[-1].lower()
    name = re.sub(r"\.(dll|dylib|pyd)$", "", name)
    name = re.sub(r"\.so(\.\d+)*$", "", name)
    name = _VENDOR_HASH.sub("", name)
    return name if name in _ABORTING_RUNTIMES else None


def loaded_runtimes() -> dict[str, set[str]]:
    """Return the OpenMP runtimes currently loaded into this process.

    Returns
    -------
    dict[str, set[str]]
        Runtime stem -> the (case-normalized) files it is loaded from.  Empty on
        platforms where the loaded modules cannot be enumerated.
    """
    runtimes: dict[str, set[str]] = {}
    for path in _loaded_modules():
        key = runtime_key(path)
        if key is not None:
            runtimes.setdefault(key, set()).add(os.path.normcase(path))
    return runtimes


def duplicate_loaded_runtimes() -> dict[str, set[str]]:
    """Return the OpenMP runtimes loaded from more than one file.

    A non-empty result means the process is already in the state that makes the
    runtime abort as soon as the second copy initializes.

    Returns
    -------
    dict[str, set[str]]
        Runtime stem -> its two or more distinct files.
    """
    return {key: paths for key, paths in loaded_runtimes().items() if len(paths) > 1}


@lru_cache(maxsize=None)
def runtimes_shipped_by(package: str) -> dict[str, frozenset[str]]:
    """Return the OpenMP runtimes *package* vendors, without importing it.

    Only the package directory and its wheel-vendored sibling directories are
    scanned, so a runtime linked statically into an extension module cannot be
    found this way.

    Parameters
    ----------
    package : str
        Top-level package name (e.g. ``"ipyopt"``).

    Returns
    -------
    dict[str, frozenset[str]]
        Runtime stem -> the (case-normalized) files the package ships it in.
        Empty if the package is not installed or ships no OpenMP runtime.
    """
    try:
        spec = find_spec(package)
    except (ImportError, ValueError):  # not installed, or a namespace/partial package
        return {}
    if spec is None or not spec.origin:
        return {}

    package_dir = os.path.dirname(spec.origin)
    parent = os.path.dirname(package_dir)
    name = os.path.basename(package_dir)
    directories = [package_dir]
    directories += [os.path.join(parent, name + suffix) for suffix in _VENDOR_DIR_SUFFIXES]

    runtimes: dict[str, set[str]] = {}
    for directory in directories:
        try:
            entries = os.scandir(directory)
        except OSError:  # a vendor directory only exists for the platform that uses it
            continue
        with entries:
            for entry in entries:
                key = runtime_key(entry.name) if entry.is_file() else None
                if key is not None:
                    runtimes.setdefault(key, set()).add(os.path.normcase(entry.path))
    return {key: frozenset(paths) for key, paths in runtimes.items()}


def blocked_by_openmp(package: str) -> Optional[str]:
    """Return why using *package* would abort this process, or ``None`` if it is safe.

    A clash is reported when *package* vendors an OpenMP runtime that is already
    loaded from a different file, or when such a runtime is already loaded twice
    (which is what the process looks like once *package* has been imported).
    Always ``None`` when :func:`duplicate_runtimes_allowed` is ``True``.

    Parameters
    ----------
    package : str
        Top-level package name (e.g. ``"ipyopt"``).

    Returns
    -------
    str or None
        A human-readable description of the clash, or ``None`` when there is none.
    """
    if duplicate_runtimes_allowed():
        return None

    loaded = loaded_runtimes()
    for key, shipped in sorted(runtimes_shipped_by(package).items()):
        others = loaded.get(key, set()) - shipped
        if others:
            return (
                f"{package} ships the OpenMP runtime '{key}' ({_shorten(shipped)}), "
                f"which is already loaded from {_shorten(others)}"
            )

    for key, paths in sorted(duplicate_loaded_runtimes().items()):
        return f"the OpenMP runtime '{key}' is loaded twice, from {_shorten(paths)}"

    return None


def _shorten(paths) -> str:
    """Render a set of module paths compactly for a log message."""
    shown = sorted(paths)
    text = ", ".join(
        os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p) for p in shown
    )
    return text


def _loaded_modules() -> list[str]:
    """Return the shared libraries loaded into this process (best effort)."""
    if sys.platform == "win32":
        return _loaded_modules_windows()
    if sys.platform.startswith("linux"):
        return _loaded_modules_proc_maps()
    return []  # macOS and friends: no dependency-free enumeration, assume no clash


def _loaded_modules_proc_maps() -> list[str]:
    """Return the mapped files from ``/proc/self/maps``."""
    paths = set()
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as maps:
            for line in maps:
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5].startswith("/"):
                    paths.add(fields[5].rstrip("\n"))
    except OSError:
        return []
    return sorted(paths)


def _loaded_modules_windows() -> list[str]:
    """Return the loaded modules via ``psapi.EnumProcessModules``."""
    from ctypes import wintypes  # noqa: PLC0415 - Windows-only import

    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except OSError:
        return []

    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.EnumProcessModules.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD

    process = kernel32.GetCurrentProcess()
    needed = wintypes.DWORD()
    capacity = 1024
    while True:
        modules = (wintypes.HMODULE * capacity)()
        if not psapi.EnumProcessModules(
            process, modules, ctypes.sizeof(modules), ctypes.byref(needed)
        ):
            logger.debug("EnumProcessModules failed (%d)", ctypes.get_last_error())
            return []
        if needed.value <= ctypes.sizeof(modules):
            break
        # The module list grew (or did not fit); retry with the size it asked for.
        capacity = needed.value // ctypes.sizeof(wintypes.HMODULE) + 64

    count = needed.value // ctypes.sizeof(wintypes.HMODULE)
    name = ctypes.create_unicode_buffer(32768)
    paths = []
    for index in range(count):
        if psapi.GetModuleFileNameExW(process, modules[index], name, len(name)):
            paths.append(name.value)
    return paths
