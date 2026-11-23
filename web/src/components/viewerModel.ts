import type { Volume } from "../api";
import { defaultIsolineLevels } from "../renderer/analysis";
import type { ViewerSettings } from "../store/appStore";

/** `[min, max]` of a `[center, width]` window, or the fallback range when unset. */
export function windowRange(window: [number, number] | null, fallback: [number, number]): [number, number] {
  if (!window) return fallback;
  const [c, w] = window;
  return [c - w / 2, c + w / 2];
}

export function centerWidth(range: [number, number]): [number, number] {
  return [(range[0] + range[1]) / 2, range[1] - range[0]];
}

/** A volume's `range`, with `null` entries (no finite values) replaced by a usable range. */
export function volumeDisplayRange(volume: Volume, fallback: [number, number]): [number, number] {
  const [lo, hi] = volume.range;
  if (lo === null || hi === null || !Number.isFinite(lo) || !Number.isFinite(hi)) return fallback;
  return [lo, hi];
}

export interface DisplayModel {
  quantity: Volume | null;
  ctRange: [number, number];
  quantityRange: [number, number];
  ctWindow: [number, number];
  quantityWindow: [number, number];
  isolineLevels: number[];
}

export function displayModel(ct: Volume | null, quantities: Record<string, Volume>, viewer: ViewerSettings): DisplayModel {
  const quantity = viewer.activeQuantity ? (quantities[viewer.activeQuantity] ?? null) : null;
  const ctRange = ct ? volumeDisplayRange(ct, [-1000, 1000]) : ([-1000, 1000] as [number, number]);
  const quantityRange = quantity ? volumeDisplayRange(quantity, [0, 1]) : ([0, 1] as [number, number]);
  return {
    quantity,
    ctRange,
    quantityRange,
    ctWindow: windowRange(viewer.ctWindow, ctRange),
    quantityWindow: windowRange(viewer.quantityWindow, quantityRange),
    isolineLevels: viewer.isolineLevels.length ? viewer.isolineLevels : defaultIsolineLevels(quantityRange[1]),
  };
}

/** About five "nice" tick values covering `[min, max]`. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!(max > min)) return [min];
  const span = max - min;
  const raw = span / Math.max(1, count - 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const first = Math.ceil(min / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let v = first; v <= max + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return ticks;
}
