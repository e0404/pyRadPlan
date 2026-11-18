import { COLORMAP_DATA } from "./colormapData";

/** Colormaps offered by the Qt viewer options (same order), plus a few extras. */
export const COLORMAP_NAMES = [
  "jet",
  "viridis",
  "magma",
  "plasma",
  "inferno",
  "gray",
  "bone",
  "seismic",
  "coolwarm",
  "RdBu_r",
  "bwr",
  "hot",
  "turbo",
] as const;

export type ColormapName = (typeof COLORMAP_NAMES)[number];

export const LUT_SIZE = 256;

const cache = new Map<string, Uint8Array>();

function decodeBase64(text: string): Uint8Array {
  const bin = atob(text);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** 256 x RGB lookup table (uint8), as sampled by pyqtgraph from matplotlib. */
export function colormapRGB(name: string): Uint8Array {
  let lut = cache.get(name);
  if (!lut) {
    const data = COLORMAP_DATA[name] ?? COLORMAP_DATA.jet;
    lut = decodeBase64(data);
    if (lut.length !== LUT_SIZE * 3) throw new Error(`colormap ${name}: bad table size ${lut.length}`);
    cache.set(name, lut);
  }
  return lut;
}

/** 256 x RGBA lookup table for a texture upload. */
export function colormapRGBA(name: string): Uint8Array {
  const rgb = colormapRGB(name);
  const out = new Uint8Array(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i++) {
    out[4 * i] = rgb[3 * i];
    out[4 * i + 1] = rgb[3 * i + 1];
    out[4 * i + 2] = rgb[3 * i + 2];
    out[4 * i + 3] = 255;
  }
  return out;
}

/** LUT entry for normalized value `t` (clamped to [0, 1]), like `ImageItem` levels + LUT. */
export function lutIndex(t: number): number {
  if (!(t > 0)) return 0;
  if (t >= 1) return LUT_SIZE - 1;
  return Math.min(LUT_SIZE - 1, Math.floor(t * LUT_SIZE));
}

export function mapColor(name: string, t: number): [number, number, number] {
  const rgb = colormapRGB(name);
  const i = lutIndex(t);
  return [rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]];
}

/** Normalize `value` into [0, 1] for the window `[min, max]` (degenerate windows map to 0). */
export function normalize(value: number, min: number, max: number): number {
  const w = max - min;
  if (!(w > 0)) return value > min ? 1 : 0;
  return Math.min(1, Math.max(0, (value - min) / w));
}

export function cssColor(rgb: readonly number[], alpha = 1): string {
  return alpha >= 1 ? `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})` : `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}
