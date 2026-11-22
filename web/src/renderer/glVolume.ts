/**
 * A scalar volume as a WebGL2 3-D texture, sampled with `texelFetch` (nearest voxel, like
 * the Qt viewer), so integer formats work without conversion: int16 CT stays 2 bytes per
 * voxel. Volumes exceeding `MAX_3D_TEXTURE_SIZE` (or failing to allocate) fall back to
 * uploading only the displayed slice.
 */
import { extractSlice, texelMapping, type SliceLayout, type VolumeGeometry } from "./geometry";
import type { SamplerKind } from "./shaders";

export type VoxelArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array;

interface Format {
  internal: number;
  format: number;
  type: number;
  kind: SamplerKind;
  ctor: new (n: number) => VoxelArray;
}

export function textureFormat(gl: WebGL2RenderingContext, data: VoxelArray): Format {
  if (data instanceof Float32Array || data instanceof Float64Array) {
    return { internal: gl.R32F, format: gl.RED, type: gl.FLOAT, kind: "f", ctor: Float32Array };
  }
  if (data instanceof Int16Array) return { internal: gl.R16I, format: gl.RED_INTEGER, type: gl.SHORT, kind: "i", ctor: Int16Array };
  if (data instanceof Uint16Array) {
    return { internal: gl.R16UI, format: gl.RED_INTEGER, type: gl.UNSIGNED_SHORT, kind: "u", ctor: Uint16Array };
  }
  if (data instanceof Int8Array) return { internal: gl.R8I, format: gl.RED_INTEGER, type: gl.BYTE, kind: "i", ctor: Int8Array };
  if (data instanceof Uint8Array) {
    return { internal: gl.R8UI, format: gl.RED_INTEGER, type: gl.UNSIGNED_BYTE, kind: "u", ctor: Uint8Array };
  }
  if (data instanceof Int32Array) return { internal: gl.R32I, format: gl.RED_INTEGER, type: gl.INT, kind: "i", ctor: Int32Array };
  return { internal: gl.R32UI, format: gl.RED_INTEGER, type: gl.UNSIGNED_INT, kind: "u", ctor: Uint32Array };
}

export interface TexelMap {
  origin: Int32Array;
  dirH: Int32Array;
  dirV: Int32Array;
}

const SLICE_MAP: TexelMap = {
  origin: new Int32Array([0, 0, 0]),
  dirH: new Int32Array([1, 0, 0]),
  dirV: new Int32Array([0, 1, 0]),
};

export class GLVolume {
  readonly kind: SamplerKind;
  readonly sliceMode: boolean;
  readonly bytes: number;
  private readonly texture: WebGLTexture;
  private readonly format: Format;
  private uploadedSlice: string | null = null;

  constructor(
    private readonly gl: WebGL2RenderingContext,
    private readonly data: VoxelArray,
    private readonly geom: VolumeGeometry,
    forceSliceMode = false,
  ) {
    this.format = textureFormat(gl, data);
    this.kind = this.format.kind;
    const texture = gl.createTexture();
    if (!texture) throw new Error("createTexture failed");
    this.texture = texture;
    gl.bindTexture(gl.TEXTURE_3D, texture);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    const [nz, ny, nx] = geom.shape;
    const max3d = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE) as number;
    let sliceMode = forceSliceMode || nx > max3d || ny > max3d || nz > max3d;
    if (!sliceMode) {
      while (gl.getError() !== gl.NO_ERROR) {
        /* drain earlier errors */
      }
      gl.texImage3D(gl.TEXTURE_3D, 0, this.format.internal, nx, ny, nz, 0, this.format.format, this.format.type, this.upload(data));
      if (gl.getError() !== gl.NO_ERROR) sliceMode = true;
    }
    this.sliceMode = sliceMode;
    const elem = this.format.ctor === Float32Array ? 4 : (data.BYTES_PER_ELEMENT ?? 4);
    this.bytes = sliceMode ? 0 : nx * ny * nz * elem;
  }

  private upload(data: VoxelArray): ArrayBufferView {
    return data instanceof Float64Array ? new Float32Array(data) : data;
  }

  /** Bind to `unit` and return the texel mapping for the given slice. */
  bind(unit: number, layout: SliceLayout, index: number, plane: string): TexelMap {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_3D, this.texture);
    if (!this.sliceMode) return texelMapping(layout, index);
    const key = `${plane}:${index}`;
    if (this.uploadedSlice !== key) {
      const slice = extractSlice(this.data, this.geom, layout, index, this.format.ctor);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(
        gl.TEXTURE_3D,
        0,
        this.format.internal,
        slice.width,
        slice.height,
        1,
        0,
        this.format.format,
        this.format.type,
        this.upload(slice.values),
      );
      this.uploadedSlice = key;
    }
    return SLICE_MAP;
  }

  dispose(): void {
    this.gl.deleteTexture(this.texture);
  }
}

/** 1x1x1 placeholder bound when a volume is absent (keeps every sampler valid). */
export function dummyVolume(gl: WebGL2RenderingContext, kind: SamplerKind): WebGLTexture {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_3D, tex);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  if (kind === "f") gl.texImage3D(gl.TEXTURE_3D, 0, gl.R32F, 1, 1, 1, 0, gl.RED, gl.FLOAT, new Float32Array(1));
  else if (kind === "i") gl.texImage3D(gl.TEXTURE_3D, 0, gl.R16I, 1, 1, 1, 0, gl.RED_INTEGER, gl.SHORT, new Int16Array(1));
  else gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8UI, 1, 1, 1, 0, gl.RED_INTEGER, gl.UNSIGNED_BYTE, new Uint8Array(1));
  return tex;
}
