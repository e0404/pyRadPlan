/**
 * Mock handlers of package B (viewer, structures, analysis, snapshots): v1 `revisions` and
 * `results` in the SessionState, the collection of named results (the active one with its
 * components per beam / scenario and RBE for ions, snapshots and imported doses as further results),
 * `view.volume` encodings and component keys, compact mask encodings, `view.beams`,
 * `cst.get` / `cst.update_structure`, `analysis.dvh` v1, `analysis.qi`, `analysis.compare_units`
 * and `workflow.snapshot`.
 */
import { ndarray } from "../codec";
import { isoCenterPoint } from "../protocol";
import type { PlanSummary, ResultGroup, SessionState, StructureDetail, TissueDraft, Vec3 } from "../protocol";
import type { BeamInfo, BeamsResult } from "../protocol/viewer";
import {
  SEQUENCED_SUFFIX,
  defaultQuantityKey,
  groupResultKeys,
  keyLabel,
  quantityMeta,
  quantityName,
  resultKey,
  sanitizeTag,
  splitComponentSuffix,
  splitIndex,
  splitResultKey,
  splitSequenced,
} from "../../results/resultGroups";
import { MethodError, precondition, resolveStructure, type MockContext, type MockHandlers, type MockSession } from "./mockBackend";
import { fakeDose, type Phantom } from "./phantom";
import { mockSequenceView, type MockSequence } from "./sequence";

interface ViewerExt {
  seenData?: number;
  seenPlan?: number;
  seenStf?: number;
  doseRef: Float32Array | null;
  revMasks: number;
  revBeams: number;
  revCst: number;
  revCt: number;
  /** Results by name, in workspace order; `active` names the one Optimize / Recalculate replace. */
  results: Map<string, MockResult>;
  active: string;
  cache: Map<() => Float32Array, Float32Array>;
  overlap: Record<number, number>;
  tissue: Record<number, TissueDraft>;
}

/** One result of the collection: its images by entry key (no result prefix) and its scenarios. */
interface MockResult {
  /** Entry key -> lazy volume (a snapshot shares the factories of the result it copies). */
  volumes: Map<string, () => Float32Array>;
  /** Revision per entry key (a snapshot keeps those of the images it shares, amendment 14). */
  revisions: Map<string, number>;
  /** Labels of the result's `_scenario` components, by flat scenario index. */
  scenarioLabels: string[];
  /** Delivery sequence of a sequenced result (`view.sequence`). */
  sequence?: MockSequence;
}

/** Name of the active result (`pyRadPlan.app.DEFAULT_RESULT_NAME`). */
export const ACTIVE_RESULT = "current";

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
      revCt: 0,
      results: new Map(),
      active: ACTIVE_RESULT,
      cache: new Map(),
      overlap: {},
      tissue: {},
    };
    s.ext.viewer = e;
  }
  return e;
}

/** `(tag, result)` of every result, the active one first with the tag `""` (`Workspace.result_entries`). */
function resultEntries(e: ViewerExt): [string, MockResult][] {
  const active = e.results.get(e.active);
  const others = [...e.results].filter(([name]) => name !== e.active);
  return active ? [["", active], ...others] : others;
}

/** Component keys of all results (prefixed with the result name except for the active one). */
function componentKeys(e: ViewerExt): string[] {
  return resultEntries(e).flatMap(([tag, r]) => [...r.volumes.keys()].map((k) => resultKey(tag, k)));
}

function catalogGroups(e: ViewerExt): ResultGroup[] {
  return groupResultKeys(componentKeys(e), Object.fromEntries(resultEntries(e).map(([tag, r]) => [tag, r.scenarioLabels])));
}

/** rbe_x_dose > physical_dose > first of the active result, else of the first other result. */
function defaultKey(e: ViewerExt): string | null {
  for (const [tag, r] of resultEntries(e)) {
    const key = defaultQuantityKey([...r.volumes.keys()]);
    if (key) return resultKey(tag, key);
  }
  return null;
}

/** `unique_result_name`: no whitespace and no `/`, made unique with `_2`, `_3`, ... */
function uniqueResultName(name: string, taken: Iterable<string>): string {
  const base = name.trim().replace(/[\s/]+/g, "_") || "result";
  const used = new Set(taken);
  if (!used.has(base)) return base;
  let i = 2;
  while (used.has(`${base}_${i}`)) i++;
  return `${base}_${i}`;
}

/** Component keys a `$result` reference may name (objective reference images). */
export function mockResultKeys(s: MockSession): string[] {
  return componentKeys(syncViewer(s));
}

/** Selector labels of the full quantities of all results (`objectives.catalog.reference_images`). */
export function mockReferenceImages(s: MockSession): { value: string; label: string; identifier: string }[] {
  return catalogGroups(syncViewer(s)).flatMap((g) =>
    g.components.filter((c) => c.kind === "full").map((c) => ({ value: c.key, label: g.label, identifier: c.key })),
  );
}

/**
 * Add an imported dose as a result of its own (quantity `physical_dose`), named after `name`
 * made unique. The mock reads no files: the dose is a single-field stand-in on the CT grid.
 */
export function addMockDoseResult(s: MockSession, name: string): string {
  const e = syncViewer(s);
  const phantom = requirePhantom(s);
  const unique = uniqueResultName(name, e.results.keys());
  e.results.set(unique, {
    volumes: new Map([["physical_dose", () => fakeDose(phantom, [90], 50, [0, 0, 0])]]),
    revisions: new Map([["physical_dose", tick()]]),
    scenarioLabels: [],
  });
  return unique;
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
    // A new patient replaces all results.
    e.results.clear();
    e.cache.clear();
    e.doseRef = null;
    e.revCt = tick();
    e.revMasks = tick();
    e.revBeams = tick();
    e.revCst = tick();
    e.overlap = {};
    e.tissue = {};
    (s.phantom?.structures ?? []).forEach((st, i) => (e.overlap[i] = defaultOverlap(st.name, st.type)));
  }
  if (s.computed.plan !== e.seenPlan || s.computed.stf !== e.seenStf) {
    e.seenPlan = s.computed.plan;
    e.seenStf = s.computed.stf;
    e.revBeams = tick();
  }
  if (s.dose !== e.doseRef) {
    e.doseRef = s.dose;
    // Optimize / Recalculate replace only the active result; the others are kept.
    if (s.dose && s.phantom && s.plan) e.results.set(e.active, activeResult(s, s.dose, s.phantom, s.plan));
    else e.results.delete(e.active);
  }
  return e;
}

/** Setup shift in mm of the mock's worst-case scenarios (the wcScen default `setup.sd`). */
const MOCK_SETUP_SHIFT = 2.25;

/**
 * Scenarios of a non-nominal scenario model, standing in for the dij's scenario set: the nominal
 * scenario and the six setup shifts (plus two range shifts for ions), labelled as
 * `Dij.scenario_labels()` does. Every model gets this set; the mock does not sample.
 */
export function mockScenarios(s: MockSession, plan: PlanSummary): { label: string; shift: Vec3; scale: number }[] {
  const model = (s.ext.plan as { scenario?: { model?: string } } | undefined)?.scenario?.model ?? "nomScen";
  if (model === "nomScen") return [];
  const d = MOCK_SETUP_SHIFT.toFixed(2);
  const out: { label: string; shift: Vec3; scale: number }[] = [{ label: "nominal", shift: [0, 0, 0], scale: 1 }];
  for (let axis = 0; axis < 3; axis++) {
    for (const sign of [-1, 1]) {
      const shift: Vec3 = [0, 0, 0];
      shift[axis] = sign * MOCK_SETUP_SHIFT;
      const parts = [0, 1, 2].map((i) => (i === axis ? `${sign < 0 ? "-" : "+"}${d}` : "0.0"));
      out.push({ label: `setup (${parts.join(", ")}) mm`, shift, scale: 1 });
    }
  }
  if (plan.radiation_mode !== "photons") {
    out.push({ label: "range -1.0 mm / -3.5 %", shift: [0, 0, 0], scale: 0.97 });
    out.push({ label: "range +1.0 mm / +3.5 %", shift: [0, 0, 0], scale: 1.03 });
  }
  return out;
}

function activeResult(s: MockSession, dose: Float32Array, phantom: Phantom, plan: PlanSummary): MockResult {
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
  const scenarios = mockScenarios(s, plan);
  scenarios.forEach((scenario, i) => {
    fresh.set(`physical_dose_scenario ${i}`, () => {
      if (i === 0) return dose;
      const shifted: Vec3 = [iso[0] - scenario.shift[0], iso[1] - scenario.shift[1], iso[2] - scenario.shift[2]];
      return fakeDose(phantom, plan.gantry_angles, prescription * scenario.scale, shifted);
    });
  });
  if (plan.radiation_mode !== "photons") {
    fresh.set("rbe_x_dose", () => dose.map((v) => v * 1.1));
  }
  return {
    volumes: fresh,
    revisions: new Map([...fresh.keys()].map((key) => [key, tick()])),
    scenarioLabels: scenarios.map((scenario) => scenario.label),
  };
}

/**
 * `workflow.sequence`: the active result is replaced by itself plus the `"<identifier>_sequenced"`
 * quantities (full and per beam) of its quantities, here the optimized ones times `scale`, and
 * carries `sequence`. Quantities of an earlier sequencing are replaced. Returns the keys added.
 */
export function addMockSequencedQuantities(s: MockSession, scale: number, sequence: MockSequence): string[] {
  const e = syncViewer(s);
  const active = e.results.get(e.active);
  if (!active) throw new MethodError("LookupError", "No result available to sequence.");
  const optimized = [...active.volumes].filter(([key]) => !splitSequenced(splitComponentSuffix(splitIndex(key)[0])[0])[1]);
  const volumes = new Map(optimized);
  const added: string[] = [];
  for (const [key, factory] of optimized) {
    const [stem, index] = splitIndex(key);
    const [base, kind] = splitComponentSuffix(stem);
    if (kind === "scenario" || (kind === "" && index !== null)) continue;
    const name = `${base}${SEQUENCED_SUFFIX}${kind ? `_${kind}` : ""}${index === null ? "" : ` ${index}`}`;
    volumes.set(name, () => factory().map((v) => v * scale));
    added.push(name);
  }
  e.results.set(e.active, {
    volumes,
    revisions: new Map([...volumes.keys()].map((key) => [key, tick()])),
    scenarioLabels: active.scenarioLabels,
    sequence,
  });
  return added;
}

function volumeOf(e: ViewerExt, key: string): Float32Array {
  const [tag, entry] = splitResultKey(key);
  const factory = e.results.get(tag || e.active)?.volumes.get(entry);
  if (!factory) throw new MethodError("KeyError", `No volume quantity '${key}' in the results`);
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
  const groups = catalogGroups(e);
  const fullKeys = groups.flatMap((g) => g.components.filter((c) => c.kind === "full").map((c) => c.key));
  const names = [...e.results.keys()];
  const volumes: Record<string, number> = { ct: e.revCt };
  for (const [tag, r] of resultEntries(e)) for (const [key, rev] of r.revisions) volumes[resultKey(tag, key)] = rev;
  const workspace = {
    ct: s.computed.data ?? 0,
    cst: e.revCst,
    pln: s.computed.plan ?? 0,
    stf: s.computed.stf ?? 0,
    dij: s.computed.dij ?? 0,
    result: s.computed.optimization ?? 0,
  };
  return {
    quantities: fullKeys,
    ...(state.sequencing ? { sequencing: { ...state.sequencing, sequenced: Boolean(e.results.get(e.active)?.sequence) } } : {}),
    revisions: { workspace, volumes, masks: e.revMasks, beams: e.revBeams, base_data: 0 },
    results: {
      groups,
      scenario_labels: [...(e.results.get(e.active)?.scenarioLabels ?? [])],
      default: defaultKey(e),
      names,
      active: e.active,
      tags: names.filter((name) => name !== e.active),
      default_tag: defaultTag(s),
    },
    structures: state.structures.map((st) => ({
      ...st,
      overlap_priority: st.overlap_priority ?? e.overlap[st.index] ?? 0,
      num_voxels: st.num_voxels ?? countVoxels(s, st.index),
      num_objectives: st.num_objectives ?? 0,
    })),
    // A snapshot copies the active result; saving works for any result.
    actions: { ...state.actions, snapshot: e.results.has(e.active), save_result: names.length > 0 },
  };
}

function defaultTag(s: MockSession): string {
  return s.plan ? `${s.plan.radiation_mode}_${s.plan.num_of_fractions}fx` : "result";
}

const voxelCounts = new WeakMap<Uint8Array, number>();
function countVoxels(s: MockSession, index: number): number {
  const mask = s.phantom?.structures[index]?.mask;
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

/** Quantity identifier (or key stem) of a component key of any result. */
function baseOf(key: string): string {
  return splitComponentSuffix(splitIndex(splitResultKey(key)[1])[0])[0];
}

function unitOf(key: string): string {
  return quantityMeta(baseOf(key))?.[1] ?? "";
}

function labelOf(key: string): string {
  const base = baseOf(key);
  return quantityMeta(base)?.[0] ?? quantityName(base);
}

function structureDetail(s: MockSession, e: ViewerExt, index: number): StructureDetail {
  const phantom = requirePhantom(s);
  const st = phantom.structures[index];
  const { name } = st;
  const tissue = e.tissue[index] ?? defaultTissue();
  const ax = Number(tissue.values.alpha_x ?? 0.1);
  const bx = Number(tissue.values.beta_x ?? 0.05);
  return {
    name,
    index,
    type: st.type,
    color: [...st.color] as Vec3,
    visible: true,
    overlap_priority: e.overlap[index] ?? 0,
    num_voxels: countVoxels(s, index),
    num_scenarios: 1,
    center_of_mass: centerOfMass(phantom, st.mask),
    alpha_x: ax,
    beta_x: bx,
    tissue,
    tissue_summary: `Tissue model: ${tissue.model}; α_x = ${ax} Gy⁻¹, β_x = ${bx} Gy⁻²; α/β = ${bx ? +(ax / bx).toPrecision(4) : "∞"} Gy`,
    objectives: [],
  };
}

/** Positions of the structures a request's `names` (positions or names) select; all without it. */
function selectedStructures(phantom: Phantom, names: unknown): Set<number> {
  if (!Array.isArray(names)) return new Set(phantom.structures.map((_, i) => i));
  return new Set(names.map((n) => resolveStructure(phantom.structures, n)));
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
    const wanted = selectedStructures(phantom, params.names);
    const grid = phantom.grid;
    const list = phantom.structures.map((st, index) => ({ st, index })).filter(({ index }) => wanted.has(index));
    const encoding = (params.encoding as string | undefined) ?? "uint8";
    if (encoding === "uint8") return ctx.fallback();
    if (encoding === "crops") {
      return {
        grid,
        encoding: "crops",
        masks: list.map(({ st, index }) => {
          const bbox = bboxOf(st.mask, grid.shape);
          if (!bbox) return { index, name: st.name, color: st.color, num_voxels: 0, bbox: null, data: null };
          const [z0, y0, x0, z1, y1, x1] = bbox;
          const [, ny, nx] = grid.shape;
          const crop = new Uint8Array((z1 - z0) * (y1 - y0) * (x1 - x0));
          let o = 0;
          for (let k = z0; k < z1; k++)
            for (let j = y0; j < y1; j++) {
              const row = (k * ny + j) * nx;
              for (let c = x0; c < x1; c++) crop[o++] = st.mask[row + c];
            }
          return { index, name: st.name, color: st.color, num_voxels: countVoxels(s, index), bbox, data: ndarray(crop, [z1 - z0, y1 - y0, x1 - x0]) };
        }),
      };
    }
    if (encoding === "bits") {
      const n = phantom.ct.length;
      const planes: Uint8Array[] = [];
      const masks = list.map(({ st, index }, i) => {
        const plane = Math.floor(i / 8);
        const bit = i % 8;
        if (!planes[plane]) planes[plane] = new Uint8Array(n);
        const p = planes[plane];
        for (let v = 0; v < n; v++) if (st.mask[v]) p[v] |= 1 << bit;
        return { index, name: st.name, color: st.color, plane, bit, num_voxels: countVoxels(s, index), bbox: bboxOf(st.mask, grid.shape) };
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

  async "view.sequence"(_params, ctx) {
    const s = ctx.session();
    const e = syncViewer(s);
    const sequence = e.results.get(e.active)?.sequence;
    if (!sequence || !s.plan) throw precondition("The active result has no delivery sequence.", "no_sequence");
    const phantom = requirePhantom(s);
    const iso = isoCenterPoint(s.plan.iso_center) ?? targetCenter(phantom) ?? [0, 0, 0];
    return mockSequenceView(phantom, s.plan, iso, sequence);
  },

  async "cst.get"(params, ctx) {
    const s = ctx.session();
    const e = syncViewer(s);
    const phantom = requirePhantom(s);
    const indices = Array.isArray(params.names) ? params.names.map((n) => resolveStructure(phantom.structures, n)) : phantom.structures.map((_, i) => i);
    return { structures: indices.map((i) => structureDetail(s, e, i)) };
  },

  async "cst.update_structure"(params, ctx) {
    const s = ctx.session();
    if (ctx.isBusy()) throw precondition("A computation is still running", "busy");
    const e = syncViewer(s);
    const phantom = requirePhantom(s);
    const index = resolveStructure(phantom.structures, params.name);
    const st = phantom.structures[index];
    const { name } = st;
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
      e.overlap[index] = p;
    }
    if (changes.tissue !== undefined) e.tissue[index] = changes.tissue;
    e.revCst = tick();
    ctx.log(s, "info", `Structure ${name} updated`);
    return { state: ctx.publish(s), structure: structureDetail(s, e, index) };
  },

  async "workflow.snapshot"(params, ctx) {
    const s = ctx.session();
    if (ctx.isBusy()) throw precondition("A computation is still running", "busy");
    const e = syncViewer(s);
    const active = e.results.get(e.active);
    if (!active) throw precondition("There is no active result to snapshot", "not_available");
    const raw = params.tag === undefined || params.tag === null ? defaultTag(s) : String(params.tag);
    const tag = sanitizeTag(raw);
    if (!tag) throw new MethodError("ValueError", "Empty snapshot name");
    if (tag === e.active) {
      throw new MethodError("ValueError", `'${tag}' is the name of the active result and cannot be used as a snapshot name.`);
    }
    // A result of its own sharing the images (and so their revisions, v1 amendment 14);
    // reusing a name overwrites that result.
    e.results.set(tag, {
      volumes: new Map(active.volumes),
      revisions: new Map(active.revisions),
      scenarioLabels: [...active.scenarioLabels],
    });
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
    const wanted = selectedStructures(phantom, params.names);
    const numPoints = Number(params.num_points ?? 500);
    const setFor = (key: string) => {
      const data = volumeOf(e, key);
      const structures = phantom.structures.map((st, index) => ({ ...st, index })).filter((st) => wanted.has(st.index));
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
          index: st.index,
          name: st.name,
          color: st.color,
          dose: ndarray(dvh.dose, [numPoints]),
          volume: ndarray(dvh.volume, [numPoints]),
          stats: dvh.stats,
        }));
      const unit = unitOf(key);
      return { key, label: labelOf(key), name: keyLabel(catalogGroups(e), key), unit, unit_info: { symbol: unit, dimensionless: unit === "" }, curves };
    };
    if (keys) return { sets: keys.map(setFor) };
    const key = String(params.quantity ?? defaultKey(e) ?? "physical_dose");
    const set = setFor(key);
    return { quantity: key, unit: set.unit, curves: set.curves, label: set.label, unit_info: set.unit_info };
  },

  async "analysis.qi"(params, ctx) {
    const s = ctx.session();
    const e = syncViewer(s);
    const phantom = requirePhantom(s);
    const key = String(params.quantity);
    const data = volumeOf(e, key);
    const wanted = selectedStructures(phantom, params.names);
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
      .map((st, index) => ({ st, index }))
      .filter(({ index }) => wanted.has(index))
      .map(({ st, index }) => ({ index, name: st.name, color: st.color, values: qiRow(data, st.mask, refVols, refDoses) }));
    return { quantity: key, label: labelOf(key), name: keyLabel(catalogGroups(e), key), unit, unit_symbol: unit, ref_vols: refVols, ref_doses: refDoses, metrics, rows };
  },

  async "analysis.compare_units"(params) {
    const a = unitOf(String(params.a));
    const b = unitOf(String(params.b));
    const base = (u: string) => u.replace(/\s*\(RBE\)/, "");
    return base(a) === base(b) ? { shareable: true, scale: 1 } : { shareable: false, scale: null };
  },
};
