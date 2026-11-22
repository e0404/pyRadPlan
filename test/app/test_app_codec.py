import json
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from pyRadPlan.app import decode, encode


@pytest.mark.parametrize("dtype", ["uint8", "int16", "int32", "float32", "float64", "bool"])
def test_arrays_round_trip(dtype):
    array = (np.arange(24).reshape(2, 3, 4) % 5).astype(dtype)
    payload, buffers = encode({"volume": {"data": array}})
    assert payload["volume"]["data"] == {"$array": 0, "dtype": dtype, "shape": [2, 3, 4]}
    json.dumps(payload)
    restored = decode(payload, [bytes(b) for b in buffers])["volume"]["data"]
    assert restored.dtype == array.dtype
    np.testing.assert_array_equal(restored, array)


def test_buffers_are_c_contiguous_little_endian():
    array = np.arange(12, dtype=">f4").reshape(3, 4).T
    payload, buffers = encode([array])
    assert payload[0]["dtype"] == "float32" and payload[0]["shape"] == [4, 3]
    expected = np.ascontiguousarray(array).astype("<f4").tobytes()
    assert bytes(buffers[0]) == expected
    np.testing.assert_array_equal(decode(payload, buffers)[0], array)


def test_several_arrays_index_their_buffers():
    a, b = np.zeros(3, np.uint8), np.ones((2, 2), np.float32)
    payload, buffers = encode({"masks": [{"data": a}, {"data": b}]})
    assert [m["data"]["$array"] for m in payload["masks"]] == [0, 1]
    assert len(buffers) == 2
    restored = decode(payload, buffers)
    np.testing.assert_array_equal(restored["masks"][1]["data"], b)


def test_scalars_and_containers_become_json():
    class Mode(Enum):
        A = "a"

    payload, buffers = encode(
        {
            "i": np.int64(3),
            "f": np.float32(0.5),
            "b": np.bool_(True),
            "nan": float("nan"),
            "inf": np.float64("inf"),
            "t": (1, 2),
            "s": {3},
            "e": Mode.A,
            "p": Path("x"),
            "o": np.array(["a", "b"], dtype=object),
            "n": None,
        }
    )
    assert buffers == []
    assert payload == {
        "i": 3,
        "f": 0.5,
        "b": True,
        "nan": None,
        "inf": None,
        "t": [1, 2],
        "s": [3],
        "e": "a",
        "p": "x",
        "o": ["a", "b"],
        "n": None,
    }
    json.dumps(payload, allow_nan=False)


def test_unencodable_objects_raise():
    with pytest.raises(TypeError, match="object"):
        encode({"x": object()})
    with pytest.raises(TypeError, match="dtype"):
        encode(np.array([1 + 2j]))


def test_decode_rejects_missing_buffers():
    with pytest.raises(ValueError, match="missing buffer"):
        decode({"$array": 1, "dtype": "uint8", "shape": [1]}, [b"\x00"])


def test_decode_accepts_any_buffer_protocol_object():
    data = np.arange(4, dtype=np.uint8)
    for buffer in (data.tobytes(), bytearray(data.tobytes()), memoryview(data), data):
        restored = decode({"x": {"$array": 0, "dtype": "uint8", "shape": [4]}}, [buffer])
        np.testing.assert_array_equal(restored["x"], data)
