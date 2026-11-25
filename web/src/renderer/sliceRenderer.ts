/**
 * Framework-free slice viewer: WebGL2 for CT + quantity (window/level, LUTs and blending in
 * the fragment shader, volumes resident as 3-D textures), a Canvas2D overlay for structure
 * contours, isolines, the isocenter, orientation labels and the hover crosshair.
 */
import { colormapRGBA, cssColor, mapColor, LUT_SIZE } from "./colormaps";
import { ContourCache, IsolineCache, type MaskVolume } from "./contours";
import {
  DEFAULT_VIEW,
  canvasToSlice,
  fragToSliceMatrix,
  linearIndex,
  sliceLayout,
  sliceToCanvas,
  sliceVoxel,
  viewTransform,
  voxelToWorld,
  type Plane,
  type PlaneConvention,
  type SliceLayout,
  type ViewState,
  type ViewTransform,
  type VolumeGeometry,
} from "./geometry";
import { GLVolume, dummyVolume, type TexelMap, type VoxelArray } from "./glVolume";
import { VERTEX_SHADER, fragmentShader, type SamplerKind } from "./shaders";
import {
  BEAM_COLOR,
  BEAM_PERSPECTIVE,
  BEAM_WIDTH_PX,
  directionSymbol,
  indexToSliceMm,
  labelAnchor,
  orbitAngle,
  orbitPoint,
  pointToIndex,
  projectBeam,
  towardAt,
  type BeamInput,
  type ProjectedBeam,
} from "./beams";
import { ORIENTATION_MARGIN, drawMannequin, projectMannequin, screenAxesLps, type MarkerLayout } from "./orientation";

export interface RenderSettings {
  plane: Plane;
  slice: number;
  convention: PlaneConvention;
  ctWindow: [number, number];
  ctColormap: string;
  quantityWindow: [number, number];
  quantityColormap: string;
  /** 0..1 */
  opacity: number;
  showCT: boolean;
  showQuantity: boolean;
  showContours: boolean;
  showIsolines: boolean;
  /** Absolute isoline levels (quantity units). */
  isolineLevels: number[];
  /** Value mapped to the top of the colormap for isoline colours (volume max in Qt). */
  isolineColorMax: number;
  visibleStructures: ReadonlySet<number>;
  /** Isocenter in world mm, drawn as a cross on its slice. */
  isocenter: readonly [number, number, number] | null;
  /** Beams of `view.beams` (LPS mm), drawn when `showBeams`. */
  beams: BeamInput[] | null;
  showBeams: boolean;
  showOrientation: boolean;
  view: ViewState;
  /** Hover position in slice coordinates (crosshair), or null. */
  cursor: [number, number] | null;
}

export const DEFAULT_SETTINGS: RenderSettings = {
  plane: "axial",
  slice: 0,
  convention: "pyradplan",
  ctWindow: [-160, 240],
  ctColormap: "gray",
  quantityWindow: [0, 1],
  quantityColormap: "jet",
  opacity: 0.4,
  showCT: true,
  showQuantity: true,
  showContours: true,
  showIsolines: false,
  isolineLevels: [],
  isolineColorMax: 1,
  visibleStructures: new Set(),
  isocenter: null,
  beams: null,
  showBeams: false,
  showOrientation: true,
  view: DEFAULT_VIEW,
  cursor: null,
};

export interface PickResult {
  /** Voxel index `[i, j, k]` (x, y, z). */
  index: [number, number, number];
  /** World position (mm) of the voxel centre. */
  position: [number, number, number];
  ct: number | null;
  quantity: number | null;
  /** Continuous slice coordinates of the pick. */
  uv: [number, number];
}

export interface RendererInfo {
  renderer: string;
  max3DTextureSize: number;
  maxTextureSize: number;
  floatLinear: boolean;
  ctSliceMode: boolean;
  quantitySliceMode: boolean;
  textureBytes: number;
}

interface Program {
  program: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation | null>;
}

const UNIFORMS = [
  "uCt",
  "uQty",
  "uCtLut",
  "uQtyLut",
  "uFragToSlice",
  "uSliceSize",
  "uCtOrigin",
  "uCtDirH",
  "uCtDirV",
  "uQtyOrigin",
  "uQtyDirH",
  "uQtyDirV",
  "uCtWindow",
  "uQtyWindow",
  "uOpacity",
  "uShowCt",
  "uShowQty",
  "uBackground",
];

/** Slices on either side of the isocenter that show a half cross towards it. */
const ISO_HINT_SLICES = 5;
const ISO_CROSS: [number, number][] = [
  [-1, -1],
  [1, -1],
  [-1, 1],
  [1, 1],
];

export class SliceRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly overlay: CanvasRenderingContext2D;
  private readonly programs = new Map<string, Program>();
  private readonly dummies: Record<SamplerKind, WebGLTexture>;
  private readonly luts = new Map<string, WebGLTexture>();
  private readonly vao: WebGLVertexArrayObject;
  private geom: VolumeGeometry | null = null;
  private ct: { data: VoxelArray; tex: GLVolume } | null = null;
  private qty: { data: VoxelArray; tex: GLVolume } | null = null;
  private readonly contours = new ContourCache();
  private readonly isolines = new IsolineCache();
  private settings: RenderSettings = { ...DEFAULT_SETTINGS };
  private width = 1;
  private height = 1;
  private dpr = 1;
  private frame = 0;
  private disposed = false;
  private beamDrag: { index: number; gantry: number; start: number } | null = null;
  private marker: MarkerLayout | null = null;
  private markerKey = "";
  /** Force slice-mode textures (testing the fallback path). */
  forceSliceMode = false;

  constructor(
    private readonly glCanvas: HTMLCanvasElement,
    private readonly overlayCanvas: HTMLCanvasElement,
  ) {
    const gl = glCanvas.getContext("webgl2", {
      antialias: false,
      alpha: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;
    const ctx = overlayCanvas.getContext("2d");
    if (!ctx) throw new Error("Canvas2D is not available");
    this.overlay = ctx;
    this.vao = gl.createVertexArray()!;
    this.dummies = { f: dummyVolume(gl, "f"), i: dummyVolume(gl, "i"), u: dummyVolume(gl, "u") };
  }

  get info(): RendererInfo {
    const gl = this.gl;
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    return {
      renderer: String(gl.getParameter(dbg ? dbg.UNMASKED_RENDERER_WEBGL : gl.RENDERER)),
      max3DTextureSize: gl.getParameter(gl.MAX_3D_TEXTURE_SIZE) as number,
      maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
      floatLinear: gl.getExtension("OES_texture_float_linear") !== null,
      ctSliceMode: this.ct?.tex.sliceMode ?? false,
      quantitySliceMode: this.qty?.tex.sliceMode ?? false,
      textureBytes: (this.ct?.tex.bytes ?? 0) + (this.qty?.tex.bytes ?? 0),
    };
  }

  get geometry(): VolumeGeometry | null {
    return this.geom;
  }

  layout(plane: Plane = this.settings.plane): SliceLayout | null {
    return this.geom ? sliceLayout(this.geom, plane, this.settings.convention) : null;
  }

  transform(): ViewTransform | null {
    const layout = this.layout();
    return layout ? viewTransform(layout, this.width, this.height, this.settings.view) : null;
  }

  /** Geometry of all volumes; changing it drops the volumes. */
  setGeometry(geom: VolumeGeometry | null): void {
    const same =
      geom &&
      this.geom &&
      geom.shape.every((v, i) => v === this.geom!.shape[i]) &&
      geom.spacing.every((v, i) => v === this.geom!.spacing[i]) &&
      geom.origin.every((v, i) => v === this.geom!.origin[i]);
    if (same) return;
    this.geom = geom;
    this.setCT(null);
    this.setQuantity(null);
    this.setMasks([]);
  }

  setCT(data: VoxelArray | null): void {
    if (this.ct?.data === data) return;
    this.ct?.tex.dispose();
    this.ct = data && this.geom ? { data, tex: new GLVolume(this.gl, data, this.geom, this.forceSliceMode) } : null;
    this.requestRender();
  }

  setQuantity(data: VoxelArray | null): void {
    if (this.qty?.data === data) return;
    this.qty?.tex.dispose();
    this.qty = data && this.geom ? { data, tex: new GLVolume(this.gl, data, this.geom, this.forceSliceMode) } : null;
    this.isolines.setVolume(this.qty ? data : null, this.geom);
    this.requestRender();
  }

  setMasks(masks: MaskVolume[]): void {
    if (this.geom) this.contours.setMasks(masks, this.geom);
    this.requestRender();
  }

  update(partial: Partial<RenderSettings>): void {
    this.settings = { ...this.settings, ...partial };
    this.contours.convention = this.settings.convention;
    this.isolines.convention = this.settings.convention;
    this.requestRender();
  }

  get current(): Readonly<RenderSettings> {
    return this.settings;
  }

  /** Canvas size in CSS pixels. */
  resize(cssWidth: number, cssHeight: number, dpr = globalThis.devicePixelRatio || 1): void {
    this.dpr = dpr;
    this.width = Math.max(1, cssWidth);
    this.height = Math.max(1, cssHeight);
    for (const c of [this.glCanvas, this.overlayCanvas]) {
      c.width = Math.max(1, Math.round(cssWidth * dpr));
      c.height = Math.max(1, Math.round(cssHeight * dpr));
    }
    this.requestRender();
  }

  get size(): [number, number] {
    return [this.width, this.height];
  }

  pick(cssX: number, cssY: number): PickResult | null {
    const layout = this.layout();
    const geom = this.geom;
    if (!layout || !geom) return null;
    const t = viewTransform(layout, this.width, this.height, this.settings.view);
    const [u, v] = canvasToSlice(t, cssX, cssY);
    if (u < 0 || v < 0 || u >= layout.nh || v >= layout.nv) return null;
    const slice = Math.min(layout.count - 1, Math.max(0, this.settings.slice));
    const index = sliceVoxel(layout, Math.floor(u), Math.floor(v), slice);
    const li = linearIndex(geom, index);
    return {
      index,
      position: voxelToWorld(geom, index),
      ct: this.ct ? this.ct.data[li] : null,
      quantity: this.qty ? this.qty.data[li] : null,
      uv: [u, v],
    };
  }

  requestRender(): void {
    if (this.frame || this.disposed) return;
    const raf = globalThis.requestAnimationFrame;
    if (!raf) {
      this.render();
      return;
    }
    this.frame = raf(() => {
      this.frame = 0;
      this.render();
    });
  }

  /** Draw synchronously. */
  render(): void {
    if (this.disposed) return;
    this.renderGL();
    this.renderOverlay();
  }

  private program(ct: SamplerKind, qty: SamplerKind): Program {
    const key = ct + qty;
    let p = this.programs.get(key);
    if (p) return p;
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`shader: ${gl.getShaderInfoLog(s)}`);
      return s;
    };
    const program = gl.createProgram()!;
    gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentShader(ct, qty)));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(`link: ${gl.getProgramInfoLog(program)}`);
    const uniforms: Program["uniforms"] = {};
    for (const name of UNIFORMS) uniforms[name] = gl.getUniformLocation(program, name);
    p = { program, uniforms };
    this.programs.set(key, p);
    return p;
  }

  private lut(name: string): WebGLTexture {
    let tex = this.luts.get(name);
    if (tex) return tex;
    const gl = this.gl;
    tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, LUT_SIZE, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, colormapRGBA(name));
    this.luts.set(name, tex);
    return tex;
  }

  private renderGL(): void {
    const gl = this.gl;
    gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const layout = this.layout();
    if (!layout || !this.geom) return;
    const s = this.settings;
    const slice = Math.min(layout.count - 1, Math.max(0, s.slice));
    const showCt = s.showCT && this.ct !== null;
    const showQty = s.showQuantity && this.qty !== null;
    const ctKind = this.ct?.tex.kind ?? "f";
    const qtyKind = this.qty?.tex.kind ?? "f";
    const p = this.program(ctKind, qtyKind);
    const u = p.uniforms;
    gl.useProgram(p.program);

    // Transform in device pixels: scale the CSS-pixel transform by the DPR.
    const t = viewTransform(layout, this.width, this.height, s.view);
    const dt: ViewTransform = {
      scaleU: t.scaleU * this.dpr,
      scaleV: t.scaleV * this.dpr,
      offsetX: t.offsetX * this.dpr,
      offsetY: t.offsetY * this.dpr,
    };
    gl.uniformMatrix3fv(u.uFragToSlice, false, fragToSliceMatrix(dt, this.glCanvas.height));
    gl.uniform2f(u.uSliceSize, layout.nh, layout.nv);

    const bindVolume = (
      vol: { tex: GLVolume } | null,
      kind: SamplerKind,
      unit: number,
      sampler: string,
      prefix: "uCt" | "uQty",
    ) => {
      let map: TexelMap = { origin: new Int32Array(3), dirH: new Int32Array(3), dirV: new Int32Array(3) };
      if (vol) map = vol.tex.bind(unit, layout, slice, s.plane);
      else {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_3D, this.dummies[kind]);
      }
      gl.uniform1i(u[sampler], unit);
      gl.uniform3iv(u[`${prefix}Origin`], map.origin);
      gl.uniform3iv(u[`${prefix}DirH`], map.dirH);
      gl.uniform3iv(u[`${prefix}DirV`], map.dirV);
    };
    bindVolume(this.ct, ctKind, 0, "uCt", "uCt");
    bindVolume(this.qty, qtyKind, 1, "uQty", "uQty");

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.lut(s.ctColormap));
    gl.uniform1i(u.uCtLut, 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.lut(s.quantityColormap));
    gl.uniform1i(u.uQtyLut, 3);

    gl.uniform2f(u.uCtWindow, s.ctWindow[0], s.ctWindow[1]);
    gl.uniform2f(u.uQtyWindow, s.quantityWindow[0], s.quantityWindow[1]);
    gl.uniform1f(u.uOpacity, s.opacity);
    gl.uniform1i(u.uShowCt, showCt ? 1 : 0);
    gl.uniform1i(u.uShowQty, showQty ? 1 : 0);
    gl.uniform3f(u.uBackground, 0, 0, 0);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  private renderOverlay(): void {
    const ctx = this.overlay;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    const layout = this.layout();
    if (!layout || !this.geom) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const s = this.settings;
    const slice = Math.min(layout.count - 1, Math.max(0, s.slice));
    const t = viewTransform(layout, this.width, this.height, s.view);

    const strokeSegments = (segments: Float32Array, color: string, width: number) => {
      if (!segments.length) return;
      ctx.beginPath();
      for (let i = 0; i < segments.length; i += 4) {
        ctx.moveTo(t.offsetX + segments[i] * t.scaleU, t.offsetY + segments[i + 1] * t.scaleV);
        ctx.lineTo(t.offsetX + segments[i + 2] * t.scaleU, t.offsetY + segments[i + 3] * t.scaleV);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
    };
    ctx.lineCap = "round";

    if (s.showIsolines && s.showQuantity && this.qty) {
      const max = s.isolineColorMax > 0 ? s.isolineColorMax : 1;
      for (const level of s.isolineLevels) {
        const segs = this.isolines.isolines(s.plane, slice, level);
        strokeSegments(segs, cssColor(mapColor(s.quantityColormap, level / max)), 2);
      }
    }

    if (s.showContours) {
      for (const structure of this.contours.indices) {
        if (!s.visibleStructures.has(structure)) continue;
        const mask = this.contours.mask(structure)!;
        strokeSegments(this.contours.contour(structure, s.plane, slice), cssColor(mask.color), 1.5);
      }
    }

    this.drawBeams(ctx, layout, t);
    if (s.isocenter) this.drawIsocenter(ctx, layout, t, slice);
    if (s.showOrientation) this.drawOrientation(ctx, layout);

    if (s.cursor) {
      const [x, y] = sliceToCanvas(t, s.cursor[0], s.cursor[1]);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(this.width, y + 0.5);
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, this.height);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  private drawIsocenter(ctx: CanvasRenderingContext2D, layout: SliceLayout, t: ViewTransform, slice: number): void {
    const p = indexToSliceMm(layout, pointToIndex(this.geom!, this.settings.isocenter!));
    const offset = Math.round(p.n) - slice;
    if (Math.abs(offset) > ISO_HINT_SLICES) return;
    const [x, y] = this.mmToCanvas(t, layout, p.h, p.v);
    const r = 10;
    ctx.lineWidth = 5;
    ctx.lineCap = "square";
    ctx.beginPath();
    if (offset === 0) {
      ctx.strokeStyle = "rgb(255, 0, 0)";
      for (const [dx, dy] of ISO_CROSS) {
        ctx.moveTo(x, y);
        ctx.lineTo(x + dx * r, y + dy * r);
      }
    } else {
      // Half cross pointing towards the isocenter slice (the wheel scrolls up to higher slices).
      ctx.strokeStyle = "rgba(255, 0, 0, 0.35)";
      const dy = offset > 0 ? 1 : -1;
      ctx.moveTo(x, y);
      ctx.lineTo(x - r, y + dy * r);
      ctx.moveTo(x, y);
      ctx.lineTo(x + r, y + dy * r);
    }
    ctx.stroke();
  }

  private mmToCanvas(t: ViewTransform, layout: SliceLayout, h: number, v: number): [number, number] {
    return [t.offsetX + (h / layout.sh) * t.scaleU, t.offsetY + (v / layout.sv) * t.scaleV];
  }

  private canvasToMm(t: ViewTransform, layout: SliceLayout, x: number, y: number): { h: number; v: number } {
    return { h: ((x - t.offsetX) / t.scaleU) * layout.sh, v: ((y - t.offsetY) / t.scaleV) * layout.sv };
  }

  /** Beams projected onto the current plane. */
  projectedBeams(): ProjectedBeam[] {
    const layout = this.layout();
    const geom = this.geom;
    const beams = this.settings.beams;
    if (!layout || !geom || !beams || !this.settings.showBeams) return [];
    return beams.map((b) => projectBeam(geom, layout, b, this.settings.convention));
  }

  private drawBeams(ctx: CanvasRenderingContext2D, layout: SliceLayout, t: ViewTransform): void {
    const color = BEAM_COLOR.join(", ");
    ctx.font = "12px system-ui, sans-serif";
    for (const p of this.projectedBeams()) {
      const [ix, iy] = this.mmToCanvas(t, layout, p.iso.h, p.iso.v);
      if (p.perpendicular) {
        this.drawPerpendicularBeam(ctx, p, ix, iy, color);
        continue;
      }
      if (!p.source) continue;
      let source = p.source;
      let width = p.sourceWidth;
      let text = p.text;
      let unit = p.unit;
      const drag = this.beamDrag;
      if (drag && drag.index === p.beam.index) {
        const at = orbitPoint(p, drag.gantry);
        if (at) {
          source = at;
          const toward = towardAt(this.geom!, layout, p.beam, drag.gantry);
          width = Math.max(1, BEAM_WIDTH_PX * BEAM_PERSPECTIVE ** toward);
          text = `${p.beam.name}  ${+drag.gantry.toFixed(6)}°`;
          const n = Math.hypot(at.h - p.iso.h, at.v - p.iso.v) || 1;
          unit = { h: (at.h - p.iso.h) / n, v: (at.v - p.iso.v) / n };
          this.drawOrbit(ctx, layout, t, p);
        }
      }
      const [sx, sy] = this.mmToCanvas(t, layout, source.h, source.v);
      const dx = ix - sx;
      const dy = iy - sy;
      const len = Math.hypot(dx, dy);
      if (len > 1e-9) {
        // Wedge in screen pixels: wider source end for a source in front of the plane.
        const nx = -dy / len;
        const ny = dx / len;
        const ha = width / 2;
        const hb = BEAM_WIDTH_PX / 2;
        ctx.beginPath();
        ctx.moveTo(sx + nx * ha, sy + ny * ha);
        ctx.lineTo(ix + nx * hb, iy + ny * hb);
        ctx.lineTo(ix - nx * hb, iy - ny * hb);
        ctx.lineTo(sx - nx * ha, sy - ny * ha);
        ctx.closePath();
        ctx.fillStyle = `rgba(${color}, ${p.beam.pending ? 40 / 255 : 110 / 255})`;
        ctx.fill();
        ctx.strokeStyle = `rgb(${color})`;
        ctx.lineWidth = 1;
        ctx.setLineDash(p.beam.pending ? [5, 3] : []);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      const markerR = Math.max(7, width) / 2;
      ctx.fillStyle = `rgb(${color})`;
      ctx.beginPath();
      ctx.arc(sx, sy, markerR, 0, 2 * Math.PI);
      ctx.fill();
      const ax = labelAnchor(unit.h);
      const ay = labelAnchor(unit.v);
      ctx.textAlign = ax === 0 ? "left" : ax === 1 ? "right" : "center";
      ctx.textBaseline = ay === 0 ? "top" : ay === 1 ? "bottom" : "middle";
      ctx.fillText(text, sx + (ax === 0 ? markerR + 2 : ax === 1 ? -markerR - 2 : 0), sy + (ay === 0 ? markerR : ay === 1 ? -markerR : 0));
    }
  }

  private drawPerpendicularBeam(ctx: CanvasRenderingContext2D, p: ProjectedBeam, x: number, y: number, color: string): void {
    ctx.strokeStyle = `rgb(${color})`;
    ctx.fillStyle = `rgb(${color})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.beginPath();
    if (directionSymbol(p.toward) === "⊗") {
      ctx.moveTo(x - 3.5, y - 3.5);
      ctx.lineTo(x + 3.5, y + 3.5);
      ctx.moveTo(x + 3.5, y - 3.5);
      ctx.lineTo(x - 3.5, y + 3.5);
      ctx.stroke();
    } else {
      ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(p.beam.name, x + 8, y - 8);
  }

  private drawOrbit(ctx: CanvasRenderingContext2D, layout: SliceLayout, t: ViewTransform, p: ProjectedBeam): void {
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "rgba(200, 200, 200, 0.67)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let k = 0; k <= 240; k++) {
      const q = orbitPoint(p, (k / 240) * 360)!;
      const [qx, qy] = this.mmToCanvas(t, layout, q.h, q.v);
      if (k === 0) ctx.moveTo(qx, qy);
      else ctx.lineTo(qx, qy);
    }
    ctx.stroke();
    ctx.restore();
  }

  /** Beam whose source marker (or perpendicular marker) is under the CSS position. */
  beamAt(x: number, y: number): ProjectedBeam | null {
    const layout = this.layout();
    if (!layout) return null;
    const t = viewTransform(layout, this.width, this.height, this.settings.view);
    let best: ProjectedBeam | null = null;
    let bestD = Infinity;
    for (const p of this.projectedBeams()) {
      const at = p.source ?? (p.perpendicular ? p.iso : null);
      if (!at) continue;
      const [px, py] = this.mmToCanvas(t, layout, at.h, at.v);
      const d = Math.hypot(px - x, py - y);
      if (d <= Math.max(8, p.sourceWidth / 2 + 3) && d < bestD) {
        best = p;
        bestD = d;
      }
    }
    return best;
  }

  /** Start dragging beam `index` along its gantry orbit; false when it cannot be dragged. */
  startBeamDrag(index: number): boolean {
    const p = this.projectedBeams().find((q) => q.beam.index === index);
    if (!p || !p.draggable || !p.orbit) return false;
    this.beamDrag = { index, gantry: p.beam.gantry, start: p.beam.gantry };
    this.requestRender();
    return true;
  }

  /** Move the dragged beam towards the CSS position; returns the gantry angle shown. */
  dragBeamTo(x: number, y: number): number | null {
    const drag = this.beamDrag;
    const layout = this.layout();
    if (!drag || !layout) return null;
    const p = this.projectedBeams().find((q) => q.beam.index === drag.index);
    if (!p?.orbit) return null;
    const t = viewTransform(layout, this.width, this.height, this.settings.view);
    const at = this.canvasToMm(t, layout, x, y);
    const angle = orbitAngle(p.orbit, { h: at.h - p.iso.h, v: at.v - p.iso.v });
    if (angle !== null) drag.gantry = angle;
    this.requestRender();
    return drag.gantry;
  }

  /** Finish (with `cancel`: abort, Esc) the drag; returns the new angle when it changed. */
  endBeamDrag(cancel = false): { index: number; gantry: number } | null {
    const drag = this.beamDrag;
    this.beamDrag = null;
    this.requestRender();
    if (!drag || cancel) return null;
    const diff = ((((drag.gantry - drag.start + 180) % 360) + 360) % 360) - 180;
    return Math.abs(diff) > 1e-9 ? { index: drag.index, gantry: drag.gantry } : null;
  }

  get draggingBeam(): boolean {
    return this.beamDrag !== null;
  }

  private drawOrientation(ctx: CanvasRenderingContext2D, layout: SliceLayout): void {
    const { right, down } = screenAxesLps(this.geom!, layout);
    const key = `${right.join(",")}|${down.join(",")}`;
    if (this.markerKey !== key) {
      this.marker = projectMannequin(right, down);
      this.markerKey = key;
    }
    drawMannequin(ctx, this.marker!, ORIENTATION_MARGIN, ORIENTATION_MARGIN);
  }

  dispose(): void {
    this.disposed = true;
    if (this.frame) globalThis.cancelAnimationFrame?.(this.frame);
    this.ct?.tex.dispose();
    this.qty?.tex.dispose();
    const gl = this.gl;
    for (const p of this.programs.values()) gl.deleteProgram(p.program);
    for (const tex of this.luts.values()) gl.deleteTexture(tex);
    for (const tex of Object.values(this.dummies)) gl.deleteTexture(tex);
    gl.deleteVertexArray(this.vao);
  }
}
