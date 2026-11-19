import { describe, expect, it } from "vitest";
import { CodecError, decode, encode, ndarray, viewBuffer } from "../src/api/codec";
import type { NDArray } from "../src/api/protocol";

describe("codec", () => {
  it("round-trips nested arrays through {$array} references", () => {
    const ct = ndarray(new Int16Array([-1000, 0, 40, 1200, 5, 6]), [1, 2, 3]);
    const dose = ndarray(new Float32Array([0, 1.5, 2.25, 60]), [4]);
    const value = { name: "x", grid: { shape: [1, 2, 3] }, data: ct, nested: [{ dose }], n: null };
    const { value: wire, buffers } = encode(value);
    expect(buffers).toHaveLength(2);
    expect(wire).toEqual({
      name: "x",
      grid: { shape: [1, 2, 3] },
      data: { $array: 0, dtype: "int16", shape: [1, 2, 3] },
      nested: [{ dose: { $array: 1, dtype: "float32", shape: [4] } }],
      n: null,
    });
    // JSON-safe wire part.
    const json = JSON.parse(JSON.stringify(wire));
    const back = decode<typeof value>(json, buffers);
    expect(back.data.dtype).toBe("int16");
    expect(Array.from(back.data.data)).toEqual([-1000, 0, 40, 1200, 5, 6]);
    expect(back.nested[0].dose.data).toBeInstanceOf(Float32Array);
    expect(Array.from(back.nested[0].dose.data)).toEqual([0, 1.5, 2.25, 60]);
  });

  it("is zero-copy for whole buffers and decodes as views", () => {
    const data = new Float32Array([1, 2, 3, 4]);
    const { buffers } = encode({ a: ndarray(data, [2, 2]) });
    expect(buffers[0]).toBe(data.buffer);
    const decoded = decode<{ a: NDArray }>({ a: { $array: 0, dtype: "float32", shape: [2, 2] } }, buffers);
    expect(decoded.a.data.buffer).toBe(data.buffer);
  });

  it("copies sub-views so a transferred buffer holds only the array", () => {
    const big = new Uint8Array(16).map((_, i) => i);
    const sub = big.subarray(4, 8);
    const { buffers } = encode({ data: sub });
    expect(buffers[0].byteLength).toBe(4);
    expect(Array.from(new Uint8Array(buffers[0]))).toEqual([4, 5, 6, 7]);
  });

  it("handles unaligned buffers by copying", () => {
    const bytes = new Uint8Array(9);
    new DataView(bytes.buffer).setFloat32(1, 3.5, true);
    const view = viewBuffer(new Uint8Array(bytes.buffer, 1, 4), "float32", [1]);
    expect(view.data[0]).toBe(3.5);
  });

  it("decodes bool as uint8 and rejects bad references", () => {
    const b = decode<NDArray>({ $array: 0, dtype: "bool", shape: [3] }, [new Uint8Array([1, 0, 1]).buffer]);
    expect(Array.from(b.data)).toEqual([1, 0, 1]);
    expect(() => decode({ $array: 2, dtype: "uint8", shape: [1] }, [])).toThrow(CodecError);
    expect(() => decode({ $array: 0, dtype: "float32", shape: [4] }, [new ArrayBuffer(8)])).toThrow(/too small/);
  });

  it("keeps a __proto__ key as data instead of replacing the prototype", () => {
    const decoded = decode(JSON.parse('{"__proto__": {"polluted": true}, "a": 1}')) as Record<string, unknown>;
    expect(Object.getPrototypeOf(decoded)).toBe(Object.prototype);
    expect(Object.keys(decoded)).toEqual(["__proto__", "a"]);
    expect((decoded as { polluted?: boolean }).polluted).toBeUndefined();
    expect(Object.getOwnPropertyDescriptor(decoded, "__proto__")?.value).toEqual({ polluted: true });
    expect(({} as { polluted?: boolean }).polluted).toBeUndefined();
  });

  it("checks shapes when building arrays", () => {
    expect(() => ndarray(new Float32Array(5), [2, 3])).toThrow(CodecError);
  });
});
