/** GLSL ES 3.00 sources of the slice program. */

export type SamplerKind = "f" | "i" | "u";

const SAMPLER: Record<SamplerKind, string> = {
  f: "sampler3D",
  i: "isampler3D",
  u: "usampler3D",
};

export const VERTEX_SHADER = `#version 300 es
// Full-screen triangle; the fragment shader works in canvas pixels.
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

export function fragmentShader(ct: SamplerKind, qty: SamplerKind): string {
  return `#version 300 es
precision highp float;
precision highp int;
precision highp sampler3D;
precision highp isampler3D;
precision highp usampler3D;

uniform ${SAMPLER[ct]} uCt;
uniform ${SAMPLER[qty]} uQty;
uniform sampler2D uCtLut;
uniform sampler2D uQtyLut;

uniform mat3 uFragToSlice;
uniform vec2 uSliceSize;
uniform ivec3 uCtOrigin;
uniform ivec3 uCtDirH;
uniform ivec3 uCtDirV;
uniform ivec3 uQtyOrigin;
uniform ivec3 uQtyDirH;
uniform ivec3 uQtyDirV;
uniform vec2 uCtWindow;
uniform vec2 uQtyWindow;
uniform float uOpacity;
uniform bool uShowCt;
uniform bool uShowQty;
uniform vec3 uBackground;

out vec4 outColor;

// Same as pyqtgraph's ImageItem: levels -> 256-entry LUT, values clamped to the ends.
int lutIndex(float v, vec2 win) {
  float w = win.y - win.x;
  float t = w > 0.0 ? (v - win.x) / w : (v > win.x ? 1.0 : 0.0);
  return int(clamp(floor(t * 256.0), 0.0, 255.0));
}

void main() {
  vec2 uv = (uFragToSlice * vec3(gl_FragCoord.xy, 1.0)).xy;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x >= uSliceSize.x || uv.y >= uSliceSize.y) {
    outColor = vec4(uBackground, 1.0);
    return;
  }
  ivec2 iuv = ivec2(floor(uv));
  vec3 color = vec3(0.0);
  if (uShowCt) {
    ivec3 t = uCtOrigin + iuv.x * uCtDirH + iuv.y * uCtDirV;
    float hu = float(texelFetch(uCt, t, 0).r);
    color = texelFetch(uCtLut, ivec2(lutIndex(hu, uCtWindow), 0), 0).rgb;
  }
  if (uShowQty) {
    ivec3 t = uQtyOrigin + iuv.x * uQtyDirH + iuv.y * uQtyDirV;
    float q = float(texelFetch(uQty, t, 0).r);
    // Zero is transparent, as the Qt viewer sets it to NaN before display.
    if (q != 0.0 && !isnan(q)) {
      vec3 c = texelFetch(uQtyLut, ivec2(lutIndex(q, uQtyWindow), 0), 0).rgb;
      color = mix(color, c, uOpacity);
    }
  }
  outColor = vec4(color, 1.0);
}
`;
}
