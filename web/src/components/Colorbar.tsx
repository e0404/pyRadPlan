import { useEffect, useRef } from "react";
import { colormapRGB, LUT_SIZE } from "../renderer/colormaps";
import { niceTicks } from "./viewerModel";

/** Vertical colourbar with ticks and a rotated label (pyqtgraph `ColorBarItem` look). */
export function Colorbar(props: { colormap: string; range: [number, number]; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { colormap, range, label } = props;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      canvas.width = Math.max(1, Math.round(w * dpr));
      canvas.height = Math.max(1, Math.round(h * dpr));
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const top = 12;
      const bottom = h - 12;
      const barX = 6;
      const barW = 14;
      const lut = colormapRGB(colormap);
      const height = bottom - top;
      for (let i = 0; i < LUT_SIZE; i++) {
        const y0 = bottom - ((i + 1) / LUT_SIZE) * height;
        ctx.fillStyle = `rgb(${lut[3 * i]}, ${lut[3 * i + 1]}, ${lut[3 * i + 2]})`;
        ctx.fillRect(barX, y0, barW, height / LUT_SIZE + 0.6);
      }
      ctx.strokeStyle = "#888";
      ctx.strokeRect(barX + 0.5, top + 0.5, barW, height);
      const [min, max] = range;
      ctx.fillStyle = "#ddd";
      ctx.strokeStyle = "#ccc";
      ctx.font = "11px system-ui, sans-serif";
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      const span = max - min || 1;
      const ticks = niceTicks(min, max, Math.max(3, Math.floor(height / 45)));
      const decimals = Math.max(0, Math.min(3, -Math.floor(Math.log10(Math.abs(ticks[1] - ticks[0] || span))) ));
      for (const t of ticks) {
        const y = bottom - ((t - min) / span) * height;
        ctx.beginPath();
        ctx.moveTo(barX + barW, y + 0.5);
        ctx.lineTo(barX + barW + 4, y + 0.5);
        ctx.stroke();
        ctx.fillText(t.toFixed(decimals), barX + barW + 6, y);
      }
      ctx.save();
      ctx.translate(w - 8, top + height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText(label, 0, 0);
      ctx.restore();
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [colormap, range, label]);

  return (
    <div className="colorbar" data-testid="colorbar">
      <canvas ref={ref} />
    </div>
  );
}
