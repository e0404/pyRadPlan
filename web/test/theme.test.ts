import { describe, expect, it } from "vitest";
import { LIGHT_PLOT_COLORS, parseColor, plotColors, rgba } from "../src/components/analysis/plotTheme";
import { busyStatusText } from "../src/components/WorkflowWidget";

describe("plot theme", () => {
  it("parses the CSS colour forms the theme variables use", () => {
    expect(parseColor("#2a2a2a")).toEqual([42, 42, 42]);
    expect(parseColor(" #fff ")).toEqual([255, 255, 255]);
    expect(parseColor("rgb(240, 240, 240)")).toEqual([240, 240, 240]);
    expect(parseColor("rgba(1 2 3 / 0.5)")).toEqual([1, 2, 3]);
    expect(parseColor("var(--x)")).toBeNull();
    expect(rgba([1, 2, 3], 0.5)).toBe("rgba(1, 2, 3, 0.5)");
  });

  it("derives figure, axes, foreground and a blended grid from the variables (Qt _plot_theme)", () => {
    const vars: Record<string, string> = { "--window": "#353535", "--base": "#2a2a2a", "--text": "#f0f0f0" };
    const root = {} as Element;
    const saved = globalThis.getComputedStyle;
    globalThis.getComputedStyle = (() => ({ getPropertyValue: (name: string) => vars[name] ?? "" })) as unknown as typeof getComputedStyle;
    try {
      const dark = plotColors(root);
      expect(dark.figure).toEqual([53, 53, 53]);
      expect(dark.axes).toEqual([42, 42, 42]);
      expect(dark.foreground).toEqual([240, 240, 240]);
      expect(dark.grid).toEqual([101, 101, 101]);
      vars["--base"] = "";
      expect(plotColors(root).axes).toEqual(LIGHT_PLOT_COLORS.axes);
    } finally {
      globalThis.getComputedStyle = saved;
    }
    expect(plotColors(null)).toEqual(LIGHT_PLOT_COLORS);
  });
});

describe("workflow status line", () => {
  it("shows the progress label, detail and fraction while busy (Qt shows the progress levels)", () => {
    expect(busyStatusText("Optimization", null)).toBe("Optimization…");
    expect(busyStatusText("Dose", { label: "Dose calculation", fraction: null })).toBe("Dose calculation…");
    expect(busyStatusText("Dose", { label: "Dose calculation", fraction: 0.42, detail: "beam 2/5" })).toBe("Dose calculation · beam 2/5 · 42 %");
  });
});
