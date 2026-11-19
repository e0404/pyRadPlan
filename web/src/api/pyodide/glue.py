"""Bridge between the web worker and ``pyRadPlan.app.dispatch`` (runs inside Pyodide).

The worker calls :func:`web_dispatch` with the request as a JSON string plus the
request's binary buffers (JS ``Uint8Array``s), and an ``emit`` JS function that takes an
event as a JSON string. It returns a JS array ``[response_json, *buffers]`` where every
buffer is a fresh JS ``Uint8Array`` (one copy out of the wasm heap) that the worker then
transfers to the main thread without another copy.
"""

import inspect
import json
import math
import traceback

import numpy as np
from pyodide.ffi import to_js

from pyRadPlan import app
from pyRadPlan.core import ProgressReporter

# No console in a worker: tqdm bars would only flood stderr (and try to start a monitor
# thread); progress reaches the UI as protocol events instead.
ProgressReporter.console_progress = False


def _clean(obj):
    """Make *obj* strict-JSON serializable (numpy scalars, tuples, NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    item = getattr(obj, "item", None)
    if callable(item) and getattr(obj, "ndim", None) == 0:
        return _clean(item())
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return _clean(tolist())
    return str(obj)


def _dumps(obj):
    return json.dumps(_clean(obj), allow_nan=False, separators=(",", ":"))


def _buffer_to_js(buf):
    """Copy a bytes-like object (or numpy array) into a new JS Uint8Array."""
    if isinstance(buf, np.ndarray):
        buf = np.ascontiguousarray(buf)
        if buf.dtype.byteorder == ">":
            buf = buf.astype(buf.dtype.newbyteorder("<"))
    view = memoryview(buf)
    if view.ndim != 1 or view.format != "B":
        view = view.cast("B")
    return to_js(view)


def _error_response(request_id, exc):
    return {
        "id": request_id,
        "ok": False,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc) or type(exc).__name__,
            "traceback": "".join(traceback.format_exception(exc)),
        },
    }


async def web_dispatch(request_json, js_buffers, js_emit):
    """Dispatch one protocol request and return ``[response_json, *buffers]`` as a JS array.

    Parameters
    ----------
    request_json : str
        The request without its binary buffers, as JSON.
    js_buffers : list of JS Uint8Array
        The request's binary buffers, in the order the request references them.
    js_emit : JS function
        Called with each progress/status event of the request as a JSON string.
    """
    request = json.loads(request_json)
    request_id = request.get("id")
    request["buffers"] = [b.to_bytes() for b in js_buffers]

    def emit(event):
        js_emit(_dumps(event))

    try:
        response = app.dispatch(request, emit)
        if inspect.isawaitable(response):
            response = await response
    except BaseException as exc:  # noqa: BLE001 - KeyboardInterrupt from the interrupt buffer too
        response = _error_response(request_id, exc)

    buffers = list(response.pop("buffers", None) or [])
    response.setdefault("id", request_id)
    try:
        payload = _dumps(response)
        js_bufs = [_buffer_to_js(b) for b in buffers]
    except BaseException as exc:  # noqa: BLE001
        payload = _dumps(_error_response(request_id, exc))
        js_bufs = []
    return to_js([payload, *js_bufs])
