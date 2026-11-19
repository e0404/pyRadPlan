import { useEffect, useMemo, useRef, useState } from "react";
import { formatPick } from "../renderer/analysis";
import { PLANES, panBy, sliceLayout, zoomAt, DEFAULT_VIEW, type Plane } from "../renderer/geometry";
import { SliceRenderer, type RendererInfo } from "../renderer/sliceRenderer";
import { keyLabel, locateKey } from "../results/resultGroups";
import { actions, setSlice as storeSetSlice, setViewer, useApp } from "../store/appStore";
import { displayIsocenter, requestBeamAngle, resultGroups, selectQuantity, setDraftBeams, syncViews } from "../store/revisions";
import { Colorbar } from "./Colorbar";
import { displayModel } from "./viewerModel";

/** A slice picked by the user stops the slices from following the isocenter. */
function setSlice(plane: Plane, index: number): void {
  if (useApp.getState().followIsocenter) useApp.setState({ followIsocenter: false });
  storeSetSlice(plane, index);
}

export const PLANE_LABEL: Record<Plane, string> = { axial: "Axial", sagittal: "Sagittal", coronal: "Coronal" };

declare global {
  interface Window {
    /** Exposed for e2e checks and debugging. */
    __pyrpRenderer?: { info: RendererInfo; renderer: SliceRenderer };
    /** Store handles for e2e scripts (`e2e/viewer.mjs`). */
    __pyrpApp?: { useApp: typeof useApp; actions: typeof actions; setDraftBeams: typeof setDraftBeams; selectQuantity: typeof selectQuantity; syncViews: typeof syncViews };
  }
}

window.__pyrpApp = { useApp, actions, setDraftBeams, selectQuantity, syncViews };

/** React wrapper of {@link SliceRenderer}: Qt `QuantityWidget` (viewer + plane/slice row + colourbar). */
export function SliceViewer() {
  const glRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<SliceRenderer | null>(null);
  const [glError, setGlError] = useState<string | null>(null);
  const [readout, setReadout] = useState<string[] | null>(null);
  const [panning, setPanning] = useState(false);
  const [tooltip, setTooltip] = useState<string>("");
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);

  const ct = useApp((s) => s.ct);
  const quantities = useApp((s) => s.quantities);
  const maskSet = useApp((s) => s.maskSet);
  const viewer = useApp((s) => s.viewer);
  const state = useApp((s) => s.state);
  const beams = useApp((s) => s.beams);
  const showBeams = useApp((s) => s.showBeams);
  const model = useMemo(() => displayModel(ct, quantities, viewer), [ct, quantities, viewer]);
  const groups = useMemo(() => resultGroups(state), [state]);
  const isocenter = useMemo(() => displayIsocenter(), [state, beams, maskSet]);

  const layout = ct ? sliceLayout(ct.grid, viewer.plane, viewer.convention) : null;
  const slice = layout ? Math.min(layout.count - 1, Math.max(0, viewer.slices[viewer.plane])) : 0;

  useEffect(() => {
    if (!glRef.current || !overlayRef.current) return;
    try {
      const r = new SliceRenderer(glRef.current, overlayRef.current);
      if (new URLSearchParams(location.search).has("sliceMode")) r.forceSliceMode = true;
      rendererRef.current = r;
      window.__pyrpRenderer = { get info() { return r.info; }, renderer: r };
      const box = boxRef.current!;
      const ro = new ResizeObserver(() => r.resize(box.clientWidth, box.clientHeight));
      ro.observe(box);
      r.resize(box.clientWidth, box.clientHeight);
      return () => {
        ro.disconnect();
        r.dispose();
        rendererRef.current = null;
      };
    } catch (err) {
      setGlError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    const r = rendererRef.current;
    if (!r) return;
    r.setGeometry(ct ? ct.grid : null);
    r.setCT(ct ? ct.data.data : null);
  }, [ct]);

  useEffect(() => {
    rendererRef.current?.setQuantity(model.quantity ? model.quantity.data.data : null);
  }, [model.quantity, ct]);

  useEffect(() => {
    rendererRef.current?.setMasks(maskSet ? maskSet.masks : []);
  }, [maskSet, ct]);

  useEffect(() => {
    rendererRef.current?.update({
      plane: viewer.plane,
      slice,
      convention: viewer.convention,
      ctWindow: model.ctWindow,
      ctColormap: viewer.ctColormap,
      quantityWindow: model.quantityWindow,
      quantityColormap: viewer.quantityColormap,
      opacity: viewer.opacity,
      showCT: viewer.showCT,
      showQuantity: viewer.showQuantity,
      showContours: viewer.showCST,
      showIsolines: viewer.showIsolines,
      isolineLevels: model.isolineLevels,
      isolineColorMax: model.quantityRange[1],
      visibleStructures: new Set(Object.entries(viewer.visible).filter(([, v]) => v).map(([k]) => Number(k))),
      isocenter: viewer.showIsocenter ? isocenter : null,
      beams: beams?.beams ?? null,
      showBeams,
      showOrientation: viewer.showOrientation,
      view: viewer.view,
    });
  }, [viewer, model, slice, isocenter, beams, showBeams]);

  const localPos = (clientX: number, clientY: number): [number, number] => {
    const rect = boxRef.current!.getBoundingClientRect();
    return [clientX - rect.left, clientY - rect.top];
  };

  const hover = (clientX: number, clientY: number) => {
    const r = rendererRef.current;
    if (!r || !ct || !boxRef.current) return;
    const [x, y] = localPos(clientX, clientY);
    const beam = r.draggingBeam ? null : r.beamAt(x, y);
    setTooltip(beam ? beam.tooltip : "");
    boxRef.current.style.cursor = beam?.draggable ? "grab" : "";
    const pick = r.pick(x, y);
    if (!pick) {
      setReadout(null);
      r.update({ cursor: null });
      return;
    }
    r.update({ cursor: pick.uv });
    // One line per full quantity, named like the Qt readout (RTQuantity names).
    const qty = Object.entries(quantities)
      .filter(([key]) => {
        const found = locateKey(groups, key);
        return found ? found.component.kind === "full" : true;
      })
      .map(([key, v]) => {
        const li = (pick.index[2] * v.grid.shape[1] + pick.index[1]) * v.grid.shape[2] + pick.index[0];
        return { label: keyLabel(groups, key), unit: v.unit, value: v.data.data[li] ?? null };
      });
    setReadout(formatPick(pick.index, pick.position, pick.ct, qty));
  };

  const scroll = (delta: number) => {
    if (!layout) return;
    setSlice(viewer.plane, Math.min(layout.count - 1, Math.max(0, slice + delta)));
  };

  // Wheel needs a non-passive listener to prevent page scrolling.
  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = rendererRef.current;
      const { ct: vol, viewer: v } = useApp.getState();
      if (!r || !vol) return;
      const lay = sliceLayout(vol.grid, v.plane, v.convention);
      if (e.ctrlKey) {
        const rect = box.getBoundingClientRect();
        const [w, h] = r.size;
        setViewer({ view: zoomAt(lay, w, h, v.view, Math.exp(-e.deltaY * 0.0015), e.clientX - rect.left, e.clientY - rect.top) });
      } else {
        const cur = Math.min(lay.count - 1, Math.max(0, v.slices[v.plane]));
        const step = e.deltaY > 0 ? -1 : 1;
        setSlice(v.plane, Math.min(lay.count - 1, Math.max(0, cur + step)));
      }
    };
    box.addEventListener("wheel", onWheel, { passive: false });
    return () => box.removeEventListener("wheel", onWheel);
  }, []);

  // Escape aborts a beam drag (Qt shortcut while dragging).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && rendererRef.current?.draggingBeam) {
        rendererRef.current.endBeamDrag(true);
        setTooltip("");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    const r = rendererRef.current;
    if (!r || !layout) return;
    setMenu(null);
    const button = e.button;
    if (button !== 0 && button !== 1 && button !== 2) return;
    const target = e.currentTarget;
    target.setPointerCapture(e.pointerId);
    const [ax, ay] = localPos(e.clientX, e.clientY);

    // Left press on a draggable source marker rotates the beam on its gantry orbit.
    const beam = button === 0 ? r.beamAt(ax, ay) : null;
    if (beam && r.startBeamDrag(beam.beam.index)) {
      const moveBeam = (ev: PointerEvent) => {
        const [x, y] = localPos(ev.clientX, ev.clientY);
        r.dragBeamTo(x, y);
      };
      const upBeam = (ev: PointerEvent) => {
        target.releasePointerCapture(ev.pointerId);
        target.removeEventListener("pointermove", moveBeam as EventListener);
        target.removeEventListener("pointerup", upBeam as EventListener);
        const result = r.endBeamDrag();
        if (result) void requestBeamAngle(result.index, result.gantry);
      };
      target.addEventListener("pointermove", moveBeam as EventListener);
      target.addEventListener("pointerup", upBeam as EventListener);
      return;
    }

    let lastX = e.clientX;
    let lastY = e.clientY;
    const startView = useApp.getState().viewer.view;
    const startX = e.clientX;
    const startY = e.clientY;
    let moved = false;
    setPanning(button !== 2);
    const move = (ev: PointerEvent) => {
      const [w, h] = r.size;
      const v = useApp.getState().viewer;
      const lay = sliceLayout(useApp.getState().ct!.grid, v.plane, v.convention);
      if (Math.hypot(ev.clientX - startX, ev.clientY - startY) > 3) moved = true;
      if (button === 2) {
        // Right-drag zooms around the press position (pyqtgraph ViewBox behaviour).
        setViewer({ view: zoomAt(lay, w, h, startView, Math.exp((startY - ev.clientY) * 0.01), ax, ay) });
      } else {
        setViewer({ view: panBy(lay, w, h, v.view, ev.clientX - lastX, ev.clientY - lastY) });
      }
      lastX = ev.clientX;
      lastY = ev.clientY;
    };
    const up = (ev: PointerEvent) => {
      target.releasePointerCapture(ev.pointerId);
      target.removeEventListener("pointermove", move as EventListener);
      target.removeEventListener("pointerup", up as EventListener);
      setPanning(false);
      // A right click without drag opens the (reduced) pyqtgraph context menu.
      if (button === 2 && !moved) setMenu({ x: ax, y: ay });
    };
    target.addEventListener("pointermove", move as EventListener);
    target.addEventListener("pointerup", up as EventListener);
  };

  const group = model.quantity ? locateKey(groups, viewer.activeQuantity)?.group : undefined;
  const activeQuantityMode = viewer.mode === "quantity" && model.quantity !== null;
  const quantityLabel = group
    ? group.unit
      ? `${group.meta_label} [${group.unit}]`
      : group.meta_label
    : model.quantity
      ? `${model.quantity.name} [${model.quantity.unit}]`
      : "";
  const colorbar = activeQuantityMode
    ? { colormap: viewer.quantityColormap, range: model.quantityWindow, label: quantityLabel }
    : viewer.mode === "ct" && ct
      ? { colormap: viewer.ctColormap, range: model.ctWindow, label: "CT [HU]" }
      : null;

  return (
    <div className="viewer" data-testid="slice-viewer">
      <div className="viewer-main">
        <div
          ref={boxRef}
          className={`viewer-canvas ${panning ? "panning" : ""}`}
          tabIndex={0}
          title={tooltip}
          data-testid="viewer-canvas"
          onPointerDown={onPointerDown}
          onPointerMove={(e) => hover(e.clientX, e.clientY)}
          onPointerLeave={() => {
            setReadout(null);
            rendererRef.current?.update({ cursor: null });
          }}
          onDoubleClick={() => setViewer({ view: DEFAULT_VIEW })}
          onContextMenu={(e) => e.preventDefault()}
          onKeyDown={(e) => {
            if (e.key === "ArrowUp" || e.key === "PageUp") scroll(e.key === "PageUp" ? 10 : 1);
            else if (e.key === "ArrowDown" || e.key === "PageDown") scroll(e.key === "PageDown" ? -10 : -1);
            else return;
            e.preventDefault();
          }}
        >
          <canvas ref={glRef} />
          <canvas ref={overlayRef} />
          {readout && (
            <div className="viewer-info" data-testid="viewer-readout">
              {readout.join("\n")}
            </div>
          )}
          {menu && (
            <div className="menu-popup" style={{ left: menu.x, top: menu.y }} onPointerDown={(e) => e.stopPropagation()}>
              <button
                className="item"
                data-testid="viewer-view-all"
                onClick={() => {
                  setViewer({ view: DEFAULT_VIEW });
                  setMenu(null);
                }}
              >
                View All
              </button>
            </div>
          )}
          {(!ct || glError) && (
            <div className="viewer-empty">{glError ? `Viewer unavailable: ${glError}` : "No CT loaded — load a phantom or a file"}</div>
          )}
        </div>
        {colorbar && <Colorbar {...colorbar} />}
      </div>
      <div className="viewer-controls">
        <select value={viewer.plane} onChange={(e) => setViewer({ plane: e.target.value as Plane })} data-testid="plane-select">
          {PLANES.map((p) => (
            <option key={p} value={p}>
              {PLANE_LABEL[p]}
            </option>
          ))}
        </select>
        <input
          type="range"
          className="fill"
          min={0}
          max={Math.max(0, (layout?.count ?? 1) - 1)}
          value={slice}
          disabled={!layout}
          onChange={(e) => setSlice(viewer.plane, Number(e.target.value))}
          data-testid="slice-slider"
          style={{ flex: 1 }}
        />
        <input
          type="number"
          min={0}
          max={Math.max(0, (layout?.count ?? 1) - 1)}
          value={slice}
          disabled={!layout}
          data-testid="slice-spin"
          onChange={(e) => {
            const v = Number(e.target.value);
            if (Number.isFinite(v) && layout) setSlice(viewer.plane, Math.min(layout.count - 1, Math.max(0, Math.round(v))));
          }}
        />
        <span data-testid="slice-label" style={{ whiteSpace: "nowrap" }}>
          {layout ? `/ ${layout.count - 1} (${PLANE_LABEL[viewer.plane]})` : "Slice: -"}
        </span>
      </div>
    </div>
  );
}
