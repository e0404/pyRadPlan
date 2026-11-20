import { useMemo } from "react";
import { sliceRange } from "../renderer/analysis";
import { COLORMAP_NAMES } from "../renderer/colormaps";
import { DEFAULT_VIEW } from "../renderer/geometry";
import { DEFAULT_VIEWER, setViewer, useApp } from "../store/appStore";
import { NumberField } from "./common";
import { centerWidth, displayModel } from "./viewerModel";

const CT_PRESETS: Record<string, [number, number]> = {
  Abdomen: [40, 400],
  Lung: [-600, 1500],
  Bone: [400, 1800],
  Brain: [40, 80],
};

/** Mirror of `ViewerOptionsWidget`: mode toggle, colormap, CT window presets, center/width, range, opacity. */
export function ViewerOptionsWidget() {
  const viewer = useApp((s) => s.viewer);
  const ct = useApp((s) => s.ct);
  const quantities = useApp((s) => s.quantities);
  const model = useMemo(() => displayModel(ct, quantities, viewer), [ct, quantities, viewer]);
  const mode = viewer.mode;
  const isQ = mode === "quantity";

  const range = isQ ? model.quantityWindow : model.ctWindow;
  const [center, width] = centerWidth(range);
  const preset =
    !isQ && viewer.ctWindow
      ? (Object.entries(CT_PRESETS).find(([, v]) => v[0] === viewer.ctWindow![0] && v[1] === viewer.ctWindow![1])?.[0] ?? "Custom")
      : "Custom";

  // Slider limits as in the Qt widget: CT fixed, quantity padded around the data range.
  const dataMax = model.quantityRange[1];
  const dataMin = model.quantityRange[0];
  const qMin = dataMin < 0 ? Math.min(0, dataMin * 1.5) : 0;
  const qMax = Math.max(10, dataMax * 1.5);
  const limits = isQ ? { cMin: qMin, cMax: qMax, wMax: qMax - qMin, step: 0.1 } : { cMin: -2000, cMax: 4000, wMax: 4000, step: 1 };

  const setWindow = (c: number, w: number) => {
    const win: [number, number] = [c, Math.max(0, w)];
    setViewer(isQ ? { quantityWindow: win } : { ctWindow: win });
  };
  const decimals = isQ ? 2 : 0;

  return (
    <div className="opts" data-testid="viewer-options">
      <div className="row">
        <button
          className={`btn ${isQ ? "checked" : ""}`}
          title="Toggle between Quantity and CT mode"
          onClick={() => setViewer({ mode: isQ ? "ct" : "quantity" })}
          data-testid="mode-toggle"
        >
          {isQ ? "Quantity" : "CT"}
        </button>
        <select
          className="fill"
          value={isQ ? viewer.quantityColormap : viewer.ctColormap}
          onChange={(e) => setViewer(isQ ? { quantityColormap: e.target.value } : { ctColormap: e.target.value })}
          data-testid="colormap-select"
        >
          {COLORMAP_NAMES.map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
      </div>

      <label>CT window:</label>
      <select
        value={preset}
        disabled={isQ}
        onChange={(e) => {
          const p = CT_PRESETS[e.target.value];
          if (p) setViewer({ ctWindow: [...p] });
        }}
        data-testid="ct-preset"
      >
        {["Custom", ...Object.keys(CT_PRESETS)].map((n) => (
          <option key={n}>{n}</option>
        ))}
      </select>

      <div className="row">
        <label>Center:</label>
        <NumberField
          className="fill"
          value={center}
          decimals={decimals}
          step={limits.step}
          min={limits.cMin}
          max={limits.cMax}
          onChange={(c) => setWindow(c, width)}
          testId="wc-spin"
        />
      </div>
      <input
        type="range"
        min={limits.cMin}
        max={limits.cMax}
        step={limits.step}
        value={center}
        onChange={(e) => setWindow(Number(e.target.value), width)}
      />
      <div className="row">
        <label>Width:</label>
        <NumberField
          className="fill"
          value={width}
          decimals={decimals}
          step={limits.step}
          min={0}
          max={limits.wMax}
          onChange={(w) => setWindow(center, w)}
          testId="ww-spin"
        />
      </div>
      <input
        type="range"
        min={0}
        max={limits.wMax}
        step={limits.step}
        value={width}
        onChange={(e) => setWindow(center, Number(e.target.value))}
      />

      <div className="row">
        <label>Range:</label>
        <NumberField
          className="fill"
          value={range[0]}
          decimals={decimals}
          step={limits.step}
          onChange={(lo) => {
            const hi = Math.max(lo, range[1]);
            setWindow((lo + hi) / 2, hi - lo);
          }}
        />
        <span>-</span>
        <NumberField
          className="fill"
          value={range[1]}
          decimals={decimals}
          step={limits.step}
          onChange={(hi) => {
            const lo = Math.min(hi, range[0]);
            setWindow((lo + hi) / 2, hi - lo);
          }}
        />
      </div>
      <button
        className="btn"
        title="Set range to min/max of current slice"
        disabled={isQ ? !model.quantity : !ct}
        onClick={() => {
          const vol = isQ ? model.quantity : ct;
          if (!vol) return;
          const [lo, hi] = sliceRange(vol.data.data, vol.grid, viewer.plane, viewer.slices[viewer.plane], isQ, viewer.convention);
          setWindow((lo + hi) / 2, hi - lo);
        }}
      >
        Use local min/max
      </button>

      <label>Overlay opacity:</label>
      <input
        type="range"
        min={0}
        max={100}
        value={Math.round(viewer.opacity * 100)}
        onChange={(e) => setViewer({ opacity: Number(e.target.value) / 100 })}
        data-testid="opacity-slider"
      />
      <button
        className="btn"
        title="Reset all viewer options to defaults"
        onClick={() =>
          setViewer({
            mode: DEFAULT_VIEWER.mode,
            ctWindow: null,
            quantityWindow: null,
            ctColormap: DEFAULT_VIEWER.ctColormap,
            quantityColormap: DEFAULT_VIEWER.quantityColormap,
            opacity: DEFAULT_VIEWER.opacity,
            view: DEFAULT_VIEW,
          })
        }
      >
        Reset options
      </button>
    </div>
  );
}
