"""Record which SimpleITK functions and Image methods a native pipeline run touches.

Used to size the SimpleITK shim. Run with the native interpreter (real SimpleITK)::

    python web/pyodide/tools/trace_sitk.py --mode photons --preset fast
"""

import collections
import functools
import json
import sys
import types
from pathlib import Path

import SimpleITK as _sitk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

calls = collections.Counter()
_SKIP = {
    "__init_subclass__",
    "__subclasshook__",
    "__getattribute__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
    "__del__",
    "__new__",
    "__repr__",
    "__str__",
    "__dir__",
    "__class__",
    "__reduce__",
    "__reduce_ex__",
    "__sizeof__",
    "__format__",
    "__hash__",
    "__swig_destroy__",
    "__weakref__",
}


def _wrap(name, func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        calls[name] += 1
        return func(*args, **kwargs)

    return wrapper


for cls_name in dir(_sitk):
    cls = getattr(_sitk, cls_name)
    if not isinstance(cls, type) or cls.__module__ != "SimpleITK.SimpleITK":
        continue
    for attr, value in list(vars(cls).items()):
        if attr in _SKIP or not callable(value) or isinstance(value, (staticmethod, type)):
            continue
        try:
            setattr(cls, attr, _wrap(f"{cls_name}.{attr}", value))
        except (TypeError, AttributeError):
            pass


class _Proxy(types.ModuleType):
    def __getattr__(self, item):
        value = getattr(_sitk, item)
        if callable(value) and not isinstance(value, type):
            calls[item] += 1
        elif not isinstance(value, type):
            calls[f"const:{item}"] += 1
        return value


proxy = _Proxy("SimpleITK")
sys.modules["SimpleITK"] = proxy

import pipeline  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="photons")
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--max-iter", type=int, default=20)
    args = parser.parse_args()
    result = pipeline.run_pipeline(args.mode, args.preset, "scipy", args.max_iter)
    print(json.dumps(dict(sorted(calls.items())), indent=1))
