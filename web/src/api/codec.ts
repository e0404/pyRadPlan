import type { ArrayRef, DType, NDArray, TypedArray } from "./protocol";

type TypedArrayCtor = {
  new (buffer: ArrayBuffer, byteOffset?: number, length?: number): TypedArray;
  readonly BYTES_PER_ELEMENT: number;
};

const CTORS: Record<DType, TypedArrayCtor> = {
  float32: Float32Array,
  float64: Float64Array,
  int8: Int8Array,
  uint8: Uint8Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,
  bool: Uint8Array,
};

export class CodecError extends Error {}

export function isArrayRef(value: unknown): value is ArrayRef {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as ArrayRef).$array === "number" &&
    typeof (value as ArrayRef).dtype === "string" &&
    Array.isArray((value as ArrayRef).shape)
  );
}

export function isTypedArray(value: unknown): value is TypedArray {
  return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

export function isNDArray(value: unknown): value is NDArray {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as NDArray).dtype === "string" &&
    Array.isArray((value as NDArray).shape) &&
    isTypedArray((value as NDArray).data)
  );
}

export function dtypeOf(array: TypedArray): DType {
  if (array instanceof Float32Array) return "float32";
  if (array instanceof Float64Array) return "float64";
  if (array instanceof Int8Array) return "int8";
  if (array instanceof Uint8Array) return "uint8";
  if (array instanceof Int16Array) return "int16";
  if (array instanceof Uint16Array) return "uint16";
  if (array instanceof Int32Array) return "int32";
  if (array instanceof Uint32Array) return "uint32";
  throw new CodecError(`unsupported typed array ${Object.prototype.toString.call(array)}`);
}

export function numel(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export function ndarray<T extends TypedArray>(data: T, shape: readonly number[], dtype?: DType): NDArray<T> {
  if (numel(shape) !== data.length) {
    throw new CodecError(`shape [${shape.join(", ")}] does not match ${data.length} elements`);
  }
  return { dtype: dtype ?? dtypeOf(data), shape: [...shape], data };
}

/**
 * View a buffer as the typed array for `dtype`, zero-copy when the buffer is aligned
 * (a copy is made otherwise, since typed arrays require element alignment).
 */
export function viewBuffer(
  buffer: ArrayBuffer | ArrayBufferView,
  dtype: DType,
  shape: readonly number[],
): NDArray {
  const ctor = CTORS[dtype];
  if (!ctor) throw new CodecError(`unsupported dtype ${dtype}`);
  const count = numel(shape);
  const bytes = count * ctor.BYTES_PER_ELEMENT;
  let base: ArrayBuffer;
  let offset: number;
  let available: number;
  if (ArrayBuffer.isView(buffer)) {
    base = buffer.buffer as ArrayBuffer;
    offset = buffer.byteOffset;
    available = buffer.byteLength;
  } else {
    base = buffer;
    offset = 0;
    available = buffer.byteLength;
  }
  if (available < bytes) {
    throw new CodecError(`buffer of ${available} bytes too small for ${dtype}[${shape.join(", ")}]`);
  }
  if (offset % ctor.BYTES_PER_ELEMENT !== 0) {
    const copy = new Uint8Array(bytes);
    copy.set(new Uint8Array(base, offset, bytes));
    base = copy.buffer;
    offset = 0;
  }
  return { dtype, shape: [...shape], data: new ctor(base, offset, count) };
}

/** Replace every `{$array}` reference in `value` by the decoded array. */
export function decode<T = unknown>(value: unknown, buffers: readonly (ArrayBuffer | ArrayBufferView)[] = []): T {
  const walk = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(walk);
    if (v === null || typeof v !== "object") return v;
    if (isArrayRef(v)) {
      const buffer = buffers[v.$array];
      if (buffer === undefined) throw new CodecError(`missing buffer ${v.$array}`);
      return viewBuffer(buffer, v.dtype, v.shape);
    }
    if (isTypedArray(v)) return v;
    const out: Record<string, unknown> = {};
    for (const [k, item] of Object.entries(v)) {
      // A plain assignment of "__proto__" (an own key of parsed JSON) would replace the prototype.
      if (k === "__proto__") Object.defineProperty(out, k, { value: walk(item), enumerable: true, writable: true, configurable: true });
      else out[k] = walk(item);
    }
    return out;
  };
  return walk(value) as T;
}

/** Whole-buffer `ArrayBuffer` holding exactly the array's bytes (copies only views). */
function ownBuffer(array: TypedArray): ArrayBuffer {
  const buf = array.buffer;
  if (buf instanceof ArrayBuffer && array.byteOffset === 0 && array.byteLength === buf.byteLength) {
    return buf;
  }
  const copy = new Uint8Array(array.byteLength);
  copy.set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
  return copy.buffer;
}

/**
 * Replace every {@link NDArray} (or bare typed array) in `value` by an `{$array}`
 * reference and collect the buffers. Buffers are the arrays' own `ArrayBuffer`s when
 * they cover them exactly, so they can be transferred without a copy.
 */
export function encode(value: unknown): { value: unknown; buffers: ArrayBuffer[] } {
  const buffers: ArrayBuffer[] = [];
  const add = (array: TypedArray, dtype: DType, shape: readonly number[]): ArrayRef => {
    buffers.push(ownBuffer(array));
    return { $array: buffers.length - 1, dtype, shape: [...shape] };
  };
  const walk = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(walk);
    if (v === null || typeof v !== "object") return v;
    if (isNDArray(v)) return add(v.data, v.dtype, v.shape);
    if (isTypedArray(v)) return add(v, dtypeOf(v), [v.length]);
    const out: Record<string, unknown> = {};
    for (const [k, item] of Object.entries(v)) {
      if (item !== undefined) out[k] = walk(item);
    }
    return out;
  };
  return { value: walk(value), buffers };
}
