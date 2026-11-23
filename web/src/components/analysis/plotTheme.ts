/**
 * Plot colours from the page's theme variables (Qt `widgets/_plot_theme.py`): the area around the
 * axes is `--window` (Qt `Window`), the data area `--base` (`Base`), text, ticks and axis lines
 * `--text` (`WindowText`), the grid a blend of foreground and data area.
 */

export type Rgb = readonly [number, number, number];

export interface PlotColors {
  figure: Rgb;
  axes: Rgb;
  foreground: Rgb;
  grid: Rgb;
}

/** Share of the foreground colour in the grid colour (the rest is the axes background). */
const GRID_WEIGHT = 0.3;

export const LIGHT_PLOT_COLORS: PlotColors = {
  figure: [240, 240, 240],
  axes: [255, 255, 255],
  foreground: [27, 27, 27],
  grid: blend([27, 27, 27], [255, 255, 255], GRID_WEIGHT),
};

function blend(fg: Rgb, bg: Rgb, weight: number): Rgb {
  return [0, 1, 2].map((i) => Math.round(weight * fg[i] + (1 - weight) * bg[i])) as unknown as Rgb;
}

/** `#rgb`, `#rrggbb` or `rgb(a)(r, g, b[, a])` as RGB; `null` for anything else. */
export function parseColor(text: string): Rgb | null {
  const s = text.trim();
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(s);
  if (hex) {
    const h = hex[1].length === 3 ? [...hex[1]].map((c) => c + c).join("") : hex[1];
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as unknown as Rgb;
  }
  const rgb = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i.exec(s);
  return rgb ? ([1, 2, 3].map((i) => Math.round(Number(rgb[i]))) as unknown as Rgb) : null;
}

/** Plot colours for the theme currently applied to `root` (default: the document element). */
export function plotColors(root: Element | null = globalThis.document?.documentElement ?? null): PlotColors {
  if (!root || typeof getComputedStyle !== "function") return LIGHT_PLOT_COLORS;
  const style = getComputedStyle(root);
  const read = (name: string, fallback: Rgb) => parseColor(style.getPropertyValue(name)) ?? fallback;
  const axes = read("--base", LIGHT_PLOT_COLORS.axes);
  const foreground = read("--text", LIGHT_PLOT_COLORS.foreground);
  return { figure: read("--window", LIGHT_PLOT_COLORS.figure), axes, foreground, grid: blend(foreground, axes, GRID_WEIGHT) };
}

export function rgba(color: Rgb | readonly number[], alpha = 1): string {
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
}
