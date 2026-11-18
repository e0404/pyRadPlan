/**
 * Mock handlers of package B (viewer, structures, analysis, snapshots): v1 `revisions` and
 * `results` in the SessionState, result components (per beam, RBE for ions) and snapshots,
 * `view.volume` encodings and component keys, compact mask encodings, `view.beams`,
 * `cst.get` / `cst.update_structure`, `analysis.dvh` v1, `analysis.qi`, `analysis.compare_units`
 * and `workflow.snapshot`.
 */
import { ndarray } from "../codec";
import { isoCenterPoint } from "../protocol";
import type { PlanSummary, SessionState, StructureDetail, TissueDraft, Vec3 } from "../protocol";
import type { BeamInfo, BeamsResult } from "../protocol/viewer";
import {
  QUANTITY_META,
  defaultQuantityKey,
  groupResultKeys,
  quantityName,
  RESERVED_TAGS,
  sanitizeTag,
  splitComponentSuffix,
  splitIndex,
} from "../../results/resultGroups";
import { MethodError, precondition, type MockContext, type MockHandlers, type MockSession } from "./mockBackend";
import { fakeDose, type Phantom } from "./phantom";

interface ViewerExt {
  seenData?: number;
  seenPlan?: number;
  seenStf?: number;
  doseRef: Float32Array | null;
  revMasks: number;
  revBeams: number;
  revCst: number;
  volumes: Record<string, number>;
  /** Result key -> lazy volume (snapshot keys share the factory of their source). */
  results: Map<string, () => Float32Array>;
  cache: Map<() => Float32Array, Float32Array>;
  tags: string[];
  overlap: Record<string, number>;
  tissue: Record<string, TissueDraft>;
}

let clock = 0;
const tick = () => ++clock;

function ext(s: MockSession): ViewerExt {
  let e = s.ext.viewer as ViewerExt | undefined;
  if (!e) {
    e = {
      doseRef: null,
      revMasks: tick(),
      revBeams: tick(),
      revCst: tick(),
      volumes: {},
      results: new Map(),
      cache: new Map(),
      tags: [],
      overlap: {},
      tissue: {},
    };
    s.ext.viewer = e;
  }
  return e;
}

function defaultOverlap(name: string, type: string): number {
  if (type === "TARGET") return 1;
  if (type === "EXTERNAL" || /body|external/i.test(name)) return 3;
  return 2;
}

function defaultTissue(): TissueDraft {
  return { model: "Generic", values: { alpha_x: 0.1, beta_x: 0.05 }, sources: { alpha_x: "default", beta_x: "default" } };
}

/** Bring the derived viewer state up to date with the mock's workspace (lazy, idempotent). */
export function syncViewer(s: MockSession): ViewerExt {
  const e = ext(s);
  if (s.computed.data !== e.seenData) {
    e.seenData = s.computed.data;
    e.results.clear();
    e.cache.clear();
    e.tags = [];
    e.doseRef = null;
    e.volumes = { ct: tick() };
    e.revMasks = tick();
    e.revBeams = tick();
    e.revCst = tick();
    e.overlap = {};
    e.tissue = {};
    for (const st of s.phantom?.structures ?? []) e.overlap[st.name] = defaultOverlap(st.name, st.type);
  }
  if (s.computed.plan !== e.seenPlan || s.computed.stf !== e.seenStf) {
    e.seenPlan = s.computed.plan;
    e.seenStf = s.computed.stf;
    e.revBeams = tick();
  }
  if (s.dose !== e.doseRef) {
    e.doseRef = s.dose;
    for (const key of [...e.results.keys()]) if (!isTagged(e, key)) e.results.delete(key);
    if (s.dose && s.phantom && s.plan) addCurrentResult(s, e, s.dose, s.phantom, s.plan);
  }
  return e;
}

function isTagged(e: ViewerExt, key: string): boolean {
  const stem = splitIndex(key)[0];
  return e.tags.some((t) => stem.endsWith("_" + t));
}

function addCurrentResult(s: MockSession, e: ViewerExt, dose: Float32Array, phantom: Phantom, plan: PlanSummary): void {
  const fresh = new Map<string, () => Float32Array>();
  fresh.set("physical_dose", () => dose);
  const n = plan.gantry_angles.length;
  const prescription = 2 * plan.num_of_fractions;
  const iso: Vec3 = isoCenterPoint(plan.iso_center) ?? [0, 0, 0];
  if (n > 1) {
    plan.gantry_angles.forEach((angle, i) => {
      fresh.set(`physical_dose_beam ${i}`, () => {
        const single = fakeDose(phantom, [angle], prescription, iso);
        for (let v = 0; v < single.length; v++) single[v] = Math.min(single[v] / n, dose[v]);
        return single;
      });
    });
  }
  if (plan.radiation_mode !== "photons") {
    fresh.set("rbe_x_dose", () => dose.map((v) => v * 1.1));
  }
  // The untagged result first (key order), then the snapshots kept so far.
  const merged = new Map([...fresh, ...[...e.results].filter(([k]) => isTagged(e, k))]);
  e.results = merged;
  for (const key of fresh.keys()) e.volumes[key] = tick();
  void s;
}

function volumeOf(e: ViewerExt, key: string): Float32Array {
  const factory = e.results.get(key);
  if (!factory) throw new MethodError("KeyError", `No volume quantity '${key}' in the result`);
  let data = e.cache.get(factory);
  if (!data) {
    data = factory();
    e.cache.set(factory, data);
  }
  return data;
}

function range(data: ArrayLike<number>): [number | null, number | null] {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return Number.isFinite(min) ? [min, max] : [null, null];
}

// ---------------------------------------------------------------------------
// State extension
// ---------------------------------------------------------------------------

export function viewerStateExtension(s: MockSession, state: SessionState): Partial<SessionState> {
  const e = syncViewer(s);
  const keys = [...e.results.keys()];
  const groups = groupResultKeys(keys);
  const fullKeys = keys.filter((k) => !splitIndex(k)[0].endsWith("_beam") && splitIndex(k)[1] === null);
  const workspace = {
    ct: s.computed.data ?? 0,
    cst: e.revCst,
    pln: s.computed.plan ?? 0,
    stf: s.computed.stf ?? 0,
    dij: s.computed.dij ?? 0,
    result: s.computed.optimization ?? 0,
  };
  const untagged = groups.filter((g) => !g.tag).flatMap((g) => g.components.map((c) => c.key));
  return {
    quantities: fullKeys,
    revisions: { workspace, volumes: { ...e.volumes }, masks: e.revMasks, beams: e.revBeams, base_data: 0 },
    results: {
      groups,
      scenario_labels: [],
      default: defaultQuantityKey(untagged.length ? untagged : keys),
      tags: [...e.tags],
      default_tag: s.plan ? `${s.plan.radiation_mode}_${s.plan.num_of_fractions}fx` : "result",
    },
    structures: state.structures.map((st) => ({
      ...st,
      overlap_priority: st.overlap_priority ?? e.overlap[st.name] ?? 0,
      num_voxels: st.num_voxels ?? countVoxels(s, st.name),
      num_objectives: st.num_objectives ?? 0,
    })),
    actions: { ...state.actions, snapshot: state.actions.snapshot ?? state.has.result },
  };
}

const voxelCounts = new WeakMap<Uint8Array, number>();
function countVoxels(s: MockSession, name: string): number {
  const mask = s.phantom?.structures.find((st) => st.name === name)?.mask;
  if (!mask) return 0;
  let n = voxelCounts.get(mask);
  if (n === undefined) {
    n = 0;
    for (let i = 0; i < mask.length; i++) n += mask[i];
    voxelCounts.set(mask, n);
  }
  return n;
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/** `lps.get_beam_rotation_matrix(gantry, couch) @ (0, -1, 0)`: isocenter -> source. */
export function sourceDirection(gantry: number, couch: number): Vec3 {
  const g = (gantry * Math.PI) / 180;
  const c = (couch * Math.PI) / 180;
  return [Math.sin(g) * Math.cos(c), -Math.cos(g), -Math.sin(g) * Math.sin(c)];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function centerOfMass(phantom: Phantom, mask: Uint8Array): Vec3 | null {
  const [nz, ny, nx] = phantom.grid.shape;
  let sx = 0;
  let sy = 0;
  let sz = 0;
  let n = 0;
  let i = 0;
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let c = 0; c < nx; c++, i++)
        if (mask[i]) {
          sx += c;
          sy += j;
          sz += k;
          n++;
        }
  if (!n) return null;
  const g = phantom.grid;
  return [g.origin[0] + (sx / n) * g.spacing[0], g.origin[1] + (sy / n) * g.spacing[1], g.origin[2] + (sz / n) * g.spacing[2]];
}

const comCache = new WeakMap<Uint8Array, Vec3 | null>();
function targetCenter(phantom: Phantom): Vec3 | null {
  const target = phantom.structures.find((st) => st.type === "TARGET");
  if (!target) return null;
  if (!comCache.has(target.mask)) comCache.set(target.mask, centerOfMass(phantom, target.mask));
  return comCache.get(target.mask) ?? null;
}

function sameAngles(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((v, i) => Math.abs((((v - b[i] + 180) % 360) + 360) % 360 - 180) < 1e-6);
}

export function beamsFor(
  plan: PlanSummary | null,
  iso: Vec3[],
  stfAngles: { gantry: number[]; couch: number[] } | null,
  draft?: { gantry_angles: number[]; couch_angles: number[] },
): BeamsResult {
  const gantry = draft?.gantry_angles ?? plan?.gantry_angles ?? [];
  const couchRaw = draft?.couch_angles ?? plan?.couch_angles ?? [];
  const couch = couchRaw.length === gantry.length ? couchRaw : gantry.map((_, i) => couchRaw[i] ?? couchRaw[0] ?? 0);
  if (!gantry.length || !iso.length) return { source: null, iso_centers: iso, beams: [] };
  const useStf = stfAngles !== null && sameAngles(stfAngles.gantry, gantry) && sameAngles(stfAngles.couch, couch);
  const applied = plan ? { g: plan.gantry_angles, c: plan.couch_angles } : null;
  const beams: BeamInfo[] = gantry.map((g, i) => {
    const c = couch[i] ?? 0;
    const u = sourceDirection(0, c);
    const w = sourceDirection(90, c);
    const pending = Boolean(
      draft && (!applied || applied.g.length !== gantry.length || Math.abs(applied.g[i] - g) > 1e-9 || Math.abs((applied.c[i] ?? 0) - c) > 1e-9),
    );
    return {
      index: i,
      name: `#${i}`,
      label: `Beam #${i}: gantry ${+g.toFixed(6)}°, couch ${+c.toFixed(6)}°`,
      gantry: g,
      couch: c,
      iso_center: iso[Math.min(i, iso.length - 1)],
      source_direction: sourceDirection(g, c),
      orbit_u: u,
      orbit_w: w,
      orbit_axis: cross(u, w),
      pending,
    };
  });
  return { source: useStf ? "stf" : "plan", iso_centers: iso, beams };
}

// ---------------------------------------------------------------------------
// Analysis helpers (DVH.compute / QICollection semantics)
// ---------------------------------------------------------------------------

/** `DVH.compute`: `numPoints` bins over `[0, max]`, left edges and cumulative volume in %. */
export function computeDvh(values: ArrayLike<number>, mask: Uint8Array, numPoints: number, maxValue?: number) {
  const voxels: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) voxels.push(values[i]);
  let max = maxValue ?? 0;
  if (maxValue === undefined) for (const v of voxels) if (v > max) max = v;
  if (maxValue === undefined) max *= 1.05;
  if (!(max > 0)) max = 1;
  const hist = new Float64Array(numPoints);
  for (const v of voxels) {
    const b = Math.floor((v / max) * numPoints);
    if (b >= 0 && b < numPoints) hist[b]++;
    else if (b === numPoints && v <= max) hist[numPoints - 1]++;
  }
  const dose = new Float32Array(numPoints);
  const volume = new Float32Array(numPoints);
  let above = voxels.length;
  for (let b = 0; b < numPoints; b++) {
    dose[b] = (b / numPoints) * max;
    volume[b] = voxels.length ? (100 * above) / voxels.length : 0;
    above -= hist[b];
  }
  let sum = 0;
  let mn = Infinity;
  let mx = -Infinity;
  for (const v of voxels) {
    sum += v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const stats = voxels.length ? { mean: sum / voxels.length, min: mn, max: mx } : { mean: null, min: null, max: null };
  return { dose, volume, stats, count: voxels.length };
}

/** numpy `percentile` (linear interpolation). */
export function percentile(sorted: Float64Array, q: number): number {
  if (!sorted.length) return NaN;
  const pos = (q / 100) * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function defaultRefDoses(values: ArrayLike<number>): number[] {
  let max = 0;
  for (let i = 0; i < values.length; i++) if (values[i] > max) max = values[i];
  if (max <= 0) return [];
  const out: number[] = [];
  for (let k = 1; k <= 5; k++) {
    const d = Math.round(((max / 5) * k) * 10) / 10;
    if (!out.includes(d)) out.push(d);
  }
  return out;
}

const fmt = (v: number) => String(+v.toFixed(6));

export function qiRow(values: ArrayLike<number>, mask: Uint8Array, refVols: number[], refDoses: number[]): Record<string, number | null> {
  const voxels: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) voxels.push(values[i]);
  const out: Record<string, number | null> = {};
  if (!voxels.length) {
    for (const k of ["mean", "std", "max", "min"]) out[k] = null;
    for (const v of refVols) out[`D${fmt(v)}`] = null;
    for (const d of refDoses) out[`V${fmt(d)}Gy`] = null;
    return out;
  }
  const sorted = Float64Array.from(voxels).sort();
  let sum = 0;
  for (const v of voxels) sum += v;
  const mean = sum / voxels.length;
  let sq = 0;
  for (const v of voxels) sq += (v - mean) ** 2;
  out.mean = mean;
  out.std = Math.sqrt(sq / voxels.length);
  out.max = sorted[sorted.length - 1];
  out.min = sorted[0];
  for (const v of refVols) out[`D${fmt(v)}`] = percentile(sorted, 100 - v);
  for (const d of refDoses) {
    let n = 0;
    for (const v of voxels) if (v >= d) n++;
    out[`V${fmt(d)}Gy`] = (100 * n) / voxels.length;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

function requirePhantom(s: MockSession): Phantom {
  if (!s.phantom) throw new MethodError("LookupError", "No CT / structures loaded");
  return s.phantom;
}

function unitOf(key: string): string {
  const base = splitComponentSuffix(splitIndex(key)[0])[0];
  for (const id of Object.keys(QUANTITY_META)) if (base === id || base.startsWith(id + "_")) return QUANTITY_META[id][1];
  return "";
}

function labelOf(key: string): string {
  const base = splitComponentSuffix(splitIndex(key)[0])[0];
  for (const id of Object.keys(QUANTITY_META)) if (base === id || base.startsWith(id + "_")) return QUANTITY_META[id][0];
  return quantityName(base);
}

function structureDetail(s: MockSession, e: ViewerExt, name: string): StructureDetail {
  const phantom = requirePhantom(s);
  const index = phantom.structures.findIndex((st) => st.name === name);
  if (index < 0) throw new MethodError("KeyError", `Unknown structure '${name}'`);
  const st = phantom.structures[index];
  const tissue = e.tissue[name] ?? defaultTissue();
  const ax = Number(tissue.values.alpha_x ?? 0.1);
  const bx = Number(tissue.values.beta_x ?? 0.05);
  return {
    name,
    index,
    type: st.type,
    color: [...st.color] as Vec3,
    visible: true,
    overlap_priority: e.overlap[name] ?? 0,
    num_voxels: countVoxels(s, name),
    num_scenarios: 1,
    center_of_mass: centerOfMass(phantom, st.mask),
    alpha_x: ax,
    beta_x: bx,
    tissue,
    tissue_summary: `Tissue model: ${tissue.model}; α_x = ${ax} Gy⁻¹, β_x = ${bx} Gy⁻²; α/β = ${bx ? +(ax / bx).toPrecision(4) : "∞"} Gy`,
    objectives: [],
  };
}

function bboxOf(mask: Uint8Array, shape: readonly number[]): [number, number, number, number, number, number] | null {
  const [nz, ny, nx] = shape;
  let z0 = nz, y0 = ny, x0 = nx, z1 = -1, y1 = -1, x1 = -1;
  let i = 0;
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let c = 0; c < nx; c++, i++)
        if (mask[i]) {
          if (k < z0) z0 = k;
          if (k > z1) z1 = k;
          if (j < y0) y0 = j;
          if (j > y1) y1 = j;
          if (c < x0) x0 = c;
          if (c > x1) x1 = c;
        }
  return z1 < 0 ? null : [z0, y0, x0, z1 + 1, y1 + 1, x1 + 1];
}

export const viewerHandlers: MockHandlers = {
  async "view.volume"(params, ctx: MockContext) {
    const s = ctx.session();
    const e = syncViewer(s);
    const name = String(params.name ?? "ct");
    const phantom = requirePhantom(s);
    const grid = phantom.grid;
    if (name === "ct") {
      // The app layer sends float32 CTs unless asked for "auto" (int16 when integral).
      const data = params.encoding === "auto" ? phantom.ct : Float32Array.from(phantom.ct);
      return { name: "ct", unit: "HU", data: ndarray(data, grid.shape), grid, range: range(phantom.ct) };
    }
    const data = volumeOf(e, name);
    return { name, unit: unitOf(name), data: ndarray(data, grid.shape), grid, range: range(data) };
  },

  async "view.structure_masks"(params, ctx) {
    const s = ctx.session();
    const phantom = requirePhantom(s);
    const names = params.names as string[] | undefined;
    const grid = phantom.grid;
    const list = phantom.structures.filter((st) => !names || names.includes(st.name));
    const encoding = (params.encoding as string | undefined) ?? "uint8";
    if (encoding === "uint8") return ctx.fallback();
    if (encoding === "crops") {
      return {
        grid,
        encoding: "crops",
        masks: list.map((st) => {
          const bbox = bboxOf(st.mask, grid.shape);
          if (!bbox) return { name: st.name, color: st.color, num_voxels: 0, bbox: null, data: null };
          const [z0, y0, x0, z1, y1, x1] = bbox;
          const [, ny, nx] = grid.shape;
          const crop = new Uint8Array((z1 - z0) * (y1 - y0) * (x1 - x0));
          let o = 0;
          for (let k = z0; k < z1; k++)
            for (let j = y0; j < y1; j++) {
              const row = (k * ny + j) * nx;
              for (let c = x0; c < x1; c++) crop[o++] = st.mask[row + c];
            }
          return { name: st.name, color: st.color, num_voxels: countVoxels(s, st.name), bbox, data: ndarray(crop, [z1 - z0, y1 - y0, x1 - x0]) };
        }),
      };
    }
    if (encoding === "bits") {
      const n = phantom.ct.length;
      const planes: Uint8Array[] = [];
      const masks = list.map((st, i) => {
        const plane = Math.floor(i / 8);
        const bit = i % 8;
        if (!planes[plane]) planes[plane] = new Uint8Array(n);
        const p = planes[plane];
        for (let v = 0; v < n; v++) if (st.mask[v]) p[v] |= 1 << bit;
        return { name: st.name, color: st.color, plane, bit, num_voxels: countVoxels(s, st.name), bbox: bboxOf(st.mask, grid.shape) };
      });
      return { grid, encoding: "bits", planes: planes.map((p) => ndarray(p, grid.shape)), masks };
    }
    throw new MethodError("ValueError", `Unknown mask encoding '${encoding}'`);
  },

  async "view.beams"(params, ctx) {
    const s = ctx.session();
    const phantom = requirePhantom(s);
    const draft = params.draft as { gantry_angles: number[]; couch_angles: number[]; iso_center?: Vec3 | Vec3[] | null } | undefined;
    const isoSource = draft?.iso_center !== undefined ? draft.iso_center : s.plan?.iso_center;
    const isoPoint = isoCenterPoint(isoSource ?? null) ?? targetCenter(phantom);
    const iso = isoPoint ? [isoPoint] : [];
    const stf = s.computed.stf !== undefined && s.plan ? { gantry: s.plan.gantry_angles, couch: s.plan.couch_angles } : null;
    return beamsFor(s.plan, iso, ctx.stage(s, "stf") === "done" ? stf : null, draft);
  },

  async "cst.get"(params, ctx) {
    const s = ctx.session();
    const e = syncViewer(s);
    const phantom = requirePhantom(s);
    const names = (params.names as string[] | undefined) ?? phantom.structures.map((st) => st.name);
    return { structures: names.map((n) => structureDetail(s, e, n)) };
  },

  async "cst.update_structure"(params, ctx) {
    const s = ctx.session();
    if (ctx.isBusy()) throw precondition("A computation is still running", "busy");
    const e = syncViewer(s);
    const phantom = requirePhantom(s);
    const name = String(params.name);
    const st = phantom.structures.find((x) => x.name === name);
    if (!st) throw new MethodError("KeyError", `Unknown structure '${name}'`);
    const changes = (params.changes ?? {}) as { type?: string; color?: Vec3; overlap_priority?: number; tissue?: TissueDraft };
    if (changes.type !== undefined) {
      if (!["TARGET", "OAR", "EXTERNAL", "HELPER"].includes(changes.type)) {
        throw new MethodError("ValidationError", `Unknown VOI type '${changes.type}'`, "invalid", [{ field: "type", message: "unknown type" }]);
      }
      st.type = changes.type as typeof st.type;
    }
    if (changes.color !== undefined) st.color = changes.color.map((c) => Math.max(0, Math.min(255, Math.round(c)))) as Vec3;
    if (changes.overlap_priority !== undefined) {
      const p = Number(changes.overlap_priority);
      if (!Number.isInteger(p) || p < 0 || p > 9999) {
        throw new MethodError("ValidationError", "Overlap priority must be 0..9999", "invalid", [{ field: "overlap_priority", message: "0..9999" }]);
      }
      e.overlap[name] = p;
    }
    if (changes.tissue !== undefined) {
      const ax = Number(changes.tissue.values.alpha_x);
      const bx = Number(changes.tissue.values.beta_x);
      if (!(ax > 0) || !(bx >= 0)) {
        if (!params.apply_anyway) {
          throw precondition("The tissue cannot be evaluated: α_x must be positive", "tissue_errors", [
            { structure: name, kind: "invalid_reference", severity: "error", message: "α_x must be positive" },
          ]);
        }
      }
      e.tissue[name] = changes.tissue;
    }
    e.revCst = tick();
    ctx.log(s, "info", `Structure ${name} updated`);
    return { state: ctx.publish(s), structure: structureDetail(s, e, name) };
  },

  async "workflow.snapshot"(params, ctx) {
    const s = ctx.session();
    if (ctx.isBusy()) throw precondition("A computation is still running", "busy");
    const e = syncViewer(s);
    if (!e.results.size) throw precondition("There is no result to snapshot", "not_available");
    const raw = params.tag === undefined || params.tag === null ? (s.plan ? `${s.plan.radiation_mode}_${s.plan.num_of_fractions}fx` : "result") : String(params.tag);
    const tag = sanitizeTag(raw);
    if (!tag) throw new MethodError("ValueError", "Empty snapshot name");
    if (RESERVED_TAGS.includes(tag)) {
      throw new MethodError("ValueError", `'${tag}' names per-${tag} results and cannot be used as a snapshot name.`);
    }
    for (const [key, factory] of [...e.results]) {
      if (isTagged(e, key)) continue;
      const [stem, index] = splitIndex(key);
      const tagged = index === null ? `${stem}_${tag}` : `${stem}_${tag} ${index}`;
      e.results.set(tagged, factory);
      // The same source object keeps its revision (v1 amendment 14).
      e.volumes[tagged] = e.volumes[key];
    }
    if (!e.tags.includes(tag)) e.tags.push(tag);
    ctx.log(s, "info", `Result snapshot '${tag}' saved`);
    return { state: ctx.publish(s), tag };
  },

  async "analysis.dvh"(params, ctx) {
    const keys = params.keys as string[] | undefined;
    const rangeMode = (params.range as string | undefined) ?? "shared";
    if (!keys && rangeMode === "shared" && params.num_points === undefined && !params.quantity) {
      const v0 = (await ctx.fallback()) as Record<string, unknown>;
      return { ...v0, label: "Dose", unit_info: { symbol: "Gy", dimensionless: false } };
    }
    const s = ctx.session();
    const e = syncViewer(s);
    const phantom = requirePhantom(s);
    const names = params.names as string[] | undefined;
    const numPoints = Number(params.num_points ?? 500);
    const setFor = (key: string) => {
      const data = volumeOf(e, key);
      const structures = phantom.structures.filter((st) => !names || names.includes(st.name));
      let shared: number | undefined;
      if (rangeMode !== "per_structure") {
        let mx = 0;
        for (const st of structures) for (let i = 0; i < data.length; i++) if (st.mask[i] && data[i] > mx) mx = data[i];
        shared = (mx || 1) * 1.05;
      }
      const curves = structures
        .map((st) => ({ st, dvh: computeDvh(data, st.mask, numPoints, shared) }))
        .filter(({ dvh }) => dvh.count > 0)
        .map(({ st, dvh }) => ({
          name: st.name,
          color: st.color,
          dose: ndarray(dvh.dose, [numPoints]),
          volume: ndarray(dvh.volume, [numPoints]),
          stats: dvh.stats,
        }));
      const unit = unitOf(key);
      return { key, label: labelOf(key), unit, unit_info: { symbol: unit, dimensionless: unit === "" }, curves };
    };
    if (keys) return { sets: keys.map(setFor) };
    const key = String(params.quantity ?? defaultQuantityKey([...e.results.keys()]) ?? "physical_dose");
    const set = setFor(key);
    return { quantity: key, unit: set.unit, curves: set.curves, label: set.label, unit_info: set.unit_info };
  },

  async "analysis.qi"(params, ctx) {
    const s = ctx.session();
    const e = syncViewer(s);
    const phantom = requirePhantom(s);
    const key = String(params.quantity);
    const data = volumeOf(e, key);
    const names = params.names as string[] | undefined;
    const validVols = (params.ref_vols as number[] | undefined)?.every((v) => Number.isFinite(v) && v >= 0 && v <= 100);
    const validDoses = (params.ref_doses as number[] | undefined)?.every((v) => Number.isFinite(v) && v >= 0);
    const refVols = validVols && (params.ref_vols as number[]).length ? (params.ref_vols as number[]) : [2, 5, 50, 95, 98];
    const refDoses = validDoses && (params.ref_doses as number[]).length ? (params.ref_doses as number[]) : defaultRefDoses(data);
    const unit = unitOf(key);
    const metrics = [
      ...(["mean", "std", "max", "min"] as const).map((id) => ({ id, label: unit ? `${id} [${unit}]` : id, kind: id, unit })),
      ...refVols.map((v) => ({ id: `D${fmt(v)}`, label: unit ? `D${fmt(v)} [${unit}]` : `D${fmt(v)}`, kind: "dx" as const, parameter: v, unit })),
      ...refDoses.map((d) => ({ id: `V${fmt(d)}Gy`, label: `V${fmt(d)}${unit} [%]`, kind: "vx" as const, parameter: d, unit: "%" })),
    ];
    const rows = phantom.structures
      .filter((st) => !names || names.includes(st.name))
      .map((st) => ({ name: st.name, color: st.color, values: qiRow(data, st.mask, refVols, refDoses) }));
    return { quantity: key, label: labelOf(key), unit, unit_symbol: unit, ref_vols: refVols, ref_doses: refDoses, metrics, rows };
  },

  async "analysis.compare_units"(params) {
    const a = unitOf(String(params.a));
    const b = unitOf(String(params.b));
    const base = (u: string) => u.replace(/\s*\(RBE\)/, "");
    return base(a) === base(b) ? { shareable: true, scale: 1 } : { shareable: false, scale: null };
  },
};
