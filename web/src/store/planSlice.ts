/**
 * State and actions of package A: plan detail / options / forms, objectives catalog and edits,
 * tissue methods and the optimization status window (Qt `OptimizationStatusWidget`).
 *
 * Kept in its own zustand store (like `uiMode.ts`), so `appStore.ts` needs no registration;
 * it reads the session, backend and state from `useApp` and writes `state` /
 * `structureDetails` after its write methods (the `state` events also trigger the
 * revision-driven refetching in `revisions.ts`).
 */
import { create } from "zustand";
import { BackendError } from "../api/backend";
import type {
  AppInfo,
  BackendEvent,
  FeatureId,
  FeatureStatus,
  FormSchema,
  Method,
  ParamsOf,
  ResultOf,
  SessionState,
  StructureDetail,
  ValidationDetail,
} from "../api/protocol";
import type {
  CurrentBio,
  FormKind,
  ObjectiveChanges,
  ObjectivesCatalog,
  PlanDetail,
  PlanOptions,
  PlanSetParams,
  StructureResult,
  TissueCheckResult,
} from "../api/protocol/plan";
import { actions as appActions, log, useApp } from "./appStore";

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export interface OptPoint {
  iteration: number;
  objective: number;
}

export interface OptStatus {
  open: boolean;
  running: boolean;
  label: string;
  points: OptPoint[];
  /** Last report's metrics (summary line). */
  last: Record<string, number> | null;
  paused: boolean;
  stopRequested: boolean;
  outcome: "done" | "stopped" | "failed" | null;
}

export interface PlanStore {
  detail: PlanDetail | null;
  /** `plan.get {detail}` failed or is not served: the form falls back to the v0 summary. */
  detailUnavailable: boolean;
  catalog: ObjectivesCatalog | null;
  objectivesStatus: string;
  status: OptStatus;
}

const EMPTY_STATUS: OptStatus = {
  open: false,
  running: false,
  label: "",
  points: [],
  last: null,
  paused: false,
  stopRequested: false,
  outcome: null,
};

export const usePlanStore = create<PlanStore>(() => ({
  detail: null,
  detailUnavailable: false,
  catalog: null,
  objectivesStatus: "",
  status: EMPTY_STATUS,
}));

const setPlan = usePlanStore.setState;
const getPlan = usePlanStore.getState;

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

export function hasMethod(method: string): boolean {
  const methods = useApp.getState().info?.methods;
  return !methods || methods.includes(method);
}

export async function request<M extends Method>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
  const { backend, session } = useApp.getState();
  if (!backend) throw new Error("Backend not connected");
  return backend.request(method, params, session ?? undefined);
}

/** Feature status from `app.info.capabilities`; the browser defaults for a backend without them. */
export function feature(info: AppInfo | null, id: FeatureId): FeatureStatus {
  const reported = info?.capabilities?.features[id];
  if (reported) return reported;
  const browser = info?.runtime !== "native" && useApp.getState().backend?.kind !== undefined;
  const fallbacks: Partial<Record<FeatureId, FeatureStatus>> = {
    ai: { available: false, reason: "AI features are not available in the browser (pydantic-ai is not installed in Pyodide)" },
    pause: {
      available: false,
      reason: "Pausing needs a second thread; in the browser pyRadPlan runs requests one at a time in a single worker",
    },
    ipopt: { available: false, reason: "IPOPT is not available in the browser runtime" },
    monte_carlo: { available: false, reason: "Monte Carlo engines cannot run in the browser runtime" },
  };
  return (browser && fallbacks[id]) || { available: true, reason: null };
}

/** Human text of a failed request like Qt's status lines: "<Type>: <message>". */
export function errorText(err: unknown): string {
  if (err instanceof BackendError) return `${err.type}: ${err.message}`;
  return err instanceof Error ? err.message : String(err);
}

export function validationDetails(err: unknown): ValidationDetail[] {
  if (!(err instanceof BackendError) || !Array.isArray(err.details)) return [];
  return (err.details as unknown[]).filter(
    (d): d is ValidationDetail => !!d && typeof d === "object" && typeof (d as ValidationDetail).field === "string",
  );
}

function applyState(state: SessionState | undefined): void {
  if (state && typeof state === "object" && "has" in state) useApp.setState({ state });
}

function mergeStructure(structure: StructureDetail | undefined): void {
  if (!structure) return;
  useApp.setState((s) => ({ structureDetails: { ...s.structureDetails, [structure.index]: structure } }));
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

let detailSeq = 0;

/** Fetch `plan.get {detail: true}` (null without a plan). */
export async function refreshPlanDetail(): Promise<PlanDetail | null> {
  const state = useApp.getState().state;
  const seq = ++detailSeq;
  if (!state?.has.pln) {
    setPlan({ detail: null });
    return null;
  }
  if (!hasMethod("plan.options")) {
    setPlan({ detail: null, detailUnavailable: true });
    return null;
  }
  try {
    const detail = (await request("plan.get", { detail: true })) as PlanDetail | null;
    if (seq === detailSeq) setPlan({ detail: detail && "dose_convention" in detail ? detail : null, detailUnavailable: !detail || !("dose_convention" in detail) });
    return detail && "dose_convention" in detail ? detail : null;
  } catch (err) {
    if (seq === detailSeq) setPlan({ detail: null, detailUnavailable: true });
    console.warn("plan.get {detail} failed", err);
    return null;
  }
}

const optionsCache = new Map<string, Promise<PlanOptions>>();

/** Options a v0 backend implies through `app.info` (no `plan.options`). */
export function fallbackPlanOptions(info: AppInfo | null, radiation_mode: string, bioModel: string | null): PlanOptions {
  const engines = info?.engines?.[radiation_mode];
  const engineNames = Array.isArray(engines) ? engines.map(String) : engines && typeof engines === "object" ? Object.keys(engines) : [];
  const model = bioModel ?? "none";
  return {
    radiation_modes: Object.keys(info?.machines ?? { photons: [] }).map((m) => ({ value: m, label: m })),
    machines: (info?.machines?.[radiation_mode] ?? ["Generic"]).map((m) => ({ value: m, label: m })),
    engines: engineNames.map((e) => ({ value: e, label: e, identifier: e })),
    bio_models: [{ value: model, label: model, identifier: model, model, parameters: {} }],
    default_bio_model: model,
    bio_selection: { option: model, model, replaced: false, unsupported: null, note: null },
    scenario_models: [{ value: "nomScen", label: "Nominal Scenario", identifier: "nomScen" }],
    problems: [{ value: "nonlin_fluence", label: "Fluence (nonlinear)", identifier: "nonlin_fluence" }],
    solvers: (info?.solvers ?? []).map((sv) => ({ value: sv, label: sv, identifier: sv, gpu_compatible: false })),
    default_solver: info?.solvers?.[0] ?? "scipy",
    beam_presets: [],
    fields: [],
  };
}

/**
 * `plan.options` for a draft context (cached per radiation mode / machine / engine / current
 * bio-model selection and plan revision). `currentBio` is the form's selection before the
 * change; without it `bio_selection` restores the plan's model.
 */
export function planOptions(radiation_mode: string, machine?: string, engine?: string | null, currentBio?: CurrentBio): Promise<PlanOptions> {
  if (!hasMethod("plan.options")) {
    return Promise.resolve(fallbackPlanOptions(useApp.getState().info, radiation_mode, useApp.getState().state?.plan?.bio_model ?? null));
  }
  const revisions = useApp.getState().state?.revisions;
  const key = JSON.stringify([radiation_mode, machine ?? "", engine ?? "", revisions?.base_data ?? 0, revisions?.workspace?.pln ?? 0, currentBio ?? null]);
  let p = optionsCache.get(key);
  if (!p) {
    const params: ParamsOf<"plan.options"> = { radiation_mode };
    if (machine) params.machine = machine;
    if (engine) params.engine = engine;
    if (currentBio) params.current_bio = currentBio;
    p = request("plan.options", params);
    p.catch(() => optionsCache.delete(key));
    optionsCache.set(key, p);
  }
  return p;
}

export function clearPlanOptionsCache(): void {
  optionsCache.clear();
}

export function planForm(kind: FormKind, name: string, radiation_mode: string, values?: Record<string, unknown>): Promise<FormSchema> {
  const params: ParamsOf<"plan.form"> = { kind, name, radiation_mode };
  if (values) params.values = values;
  return request("plan.form", params);
}

export async function tissueCheck(draft: ParamsOf<"plan.tissue_check">["draft"]): Promise<TissueCheckResult | null> {
  if (!hasMethod("plan.tissue_check")) return null;
  try {
    return await request("plan.tissue_check", draft ? { draft } : {});
  } catch (err) {
    console.warn("plan.tissue_check failed", err);
    return null;
  }
}

/** `plan.set` as the plan form's Apply (no busy bar: a write, not a long operation). */
export async function applyPlanChanges(params: PlanSetParams): Promise<SessionState> {
  const state = await request("plan.set", params);
  applyState(state);
  log("info", "Plan applied");
  await refreshPlanDetail();
  return state;
}

// ---------------------------------------------------------------------------
// Objectives
// ---------------------------------------------------------------------------

export async function refreshCatalog(): Promise<void> {
  if (!hasMethod("objectives.catalog") || !useApp.getState().state?.has.cst) return;
  try {
    setPlan({ catalog: await request("objectives.catalog", {}) });
  } catch (err) {
    console.warn("objectives.catalog failed", err);
  }
}

/** Refetch the structure details (`cst.get`) right away, e.g. while no revisions are sent. */
export async function refreshStructures(): Promise<void> {
  if (!hasMethod("cst.get") || !useApp.getState().state?.has.cst) return;
  try {
    const { structures } = await request("cst.get", {});
    useApp.setState({ structureDetails: Object.fromEntries(structures.map((d) => [d.index, d])) });
  } catch (err) {
    console.warn("cst.get failed", err);
  }
}

async function objectivesWrite<M extends Method>(method: M, params: ParamsOf<M>, what: string): Promise<boolean> {
  try {
    const result = (await request(method, params)) as unknown as Partial<StructureResult>;
    applyState(result.state);
    mergeStructure(result.structure);
    if (!result.structure) await refreshStructures();
    setPlan({ objectivesStatus: "" });
    return true;
  } catch (err) {
    const details = validationDetails(err);
    const text = details.length ? details.map((d) => `${d.field}: ${d.message}`).join("; ") : errorText(err);
    setPlan({ objectivesStatus: `Rejected ${what}: ${text}` });
    log("warning", `Objectives: rejected ${what}: ${text}`);
    return false;
  }
}

export const objectiveActions = {
  add: (structure: number) => objectivesWrite("objectives.add", { structure }, "new objective"),
  update: (structure: number, index: number, changes: ObjectiveChanges, what = "change") =>
    objectivesWrite("objectives.update", { structure, index, changes }, what),
  changeType: (structure: number, index: number, name: string) =>
    objectivesWrite("objectives.change_type", { structure, index, name }, `objective ${JSON.stringify(name)}`),
  remove: (structure: number, index: number) => objectivesWrite("objectives.remove", { structure, index }, "removal"),
  async setQuantityAll(quantity: string): Promise<boolean> {
    const ok = await objectivesWrite("objectives.set_quantity_all", { quantity }, `quantity ${JSON.stringify(quantity)}`);
    if (ok) await refreshStructures();
    return ok;
  },
};

// ---------------------------------------------------------------------------
// Optimization status window (fed from `progress` events)
// ---------------------------------------------------------------------------

const OPT_LABEL = /optimi[sz]/i;

function setStatus(patch: Partial<OptStatus> | ((s: OptStatus) => Partial<OptStatus>)): void {
  setPlan((p) => ({ status: { ...p.status, ...(typeof patch === "function" ? patch(p.status) : patch) } }));
}

/** Open (or raise) the window for a new optimization, clearing the previous curves. */
export function startOptimizationStatus(label = "Optimization"): void {
  setStatus({ ...EMPTY_STATUS, open: true, running: true, label });
}

function onProgress(event: Extract<BackendEvent, { event: "progress" }>): void {
  const data = event.data;
  const metric = data && (typeof data.objective === "number" || typeof data.iteration === "number");
  const status = getPlan().status;
  if (!status.running) {
    if (!metric && !OPT_LABEL.test(event.label)) return;
    startOptimizationStatus(event.label);
  }
  if (!data || !metric) return;
  setStatus((s) => {
    const points = typeof data.objective === "number" ? [...s.points, { iteration: data.iteration ?? s.points.length, objective: data.objective }] : s.points;
    return { points, last: { ...data } };
  });
}

function finishOptimizationStatus(): void {
  const { status } = getPlan();
  if (!status.running) return;
  const app = useApp.getState();
  const stopped = status.stopRequested || /stopped/i.test(app.statusMessage);
  const failed = !stopped && app.error !== null;
  setStatus({ running: false, paused: false, outcome: stopped ? "stopped" : failed ? "failed" : "done" });
}

export const statusActions = {
  close: () => setStatus({ open: false }),
  open: () => setStatus({ open: true }),
  async togglePause(): Promise<void> {
    const { status } = getPlan();
    const paused = !status.paused;
    try {
      await request(paused ? "session.pause" : "session.resume", {});
      setStatus({ paused });
    } catch (err) {
      log("warning", `${paused ? "Pause" : "Resume"}: ${errorText(err)}`);
    }
  },
  stop(): void {
    setStatus({ stopRequested: true });
    appActions.stop();
  },
};

/** Optimization history of the session after a reload (`session.history`). */
export async function restoreOptimizationHistory(): Promise<void> {
  if (!hasMethod("session.history")) return;
  try {
    const { optimization } = await request("session.history", { since: Number.MAX_SAFE_INTEGER });
    if (!optimization || !optimization.points.length) return;
    setStatus({
      open: !optimization.finished,
      running: !optimization.finished,
      label: optimization.label,
      points: optimization.points.map((p) => ({ iteration: p.iteration, objective: p.objective })),
      last: { ...optimization.points[optimization.points.length - 1] },
      outcome: optimization.outcome,
    });
  } catch (err) {
    console.warn("session.history failed", err);
  }
}

// ---------------------------------------------------------------------------
// Wiring: backend events and state transitions
// ---------------------------------------------------------------------------

let unsubscribeBackend: (() => void) | null = null;
let restoredFor: string | null = null;

function onBackendEvent(event: BackendEvent): void {
  const session = useApp.getState().session;
  if (event.session && session && event.session !== session) return;
  if (event.event === "progress") onProgress(event);
}

useApp.subscribe((s, prev) => {
  if (s.backend !== prev.backend) {
    unsubscribeBackend?.();
    unsubscribeBackend = s.backend ? s.backend.subscribe(onBackendEvent) : null;
    clearPlanOptionsCache();
  }
  if (s.busy !== prev.busy) {
    if (s.busy && OPT_LABEL.test(s.busy) && !getPlan().status.running) startOptimizationStatus(s.busy);
    if (!s.busy && prev.busy) queueMicrotask(finishOptimizationStatus);
  }
  if (s.session && s.state && s.session !== restoredFor) {
    restoredFor = s.session;
    void restoreOptimizationHistory();
  }
  const planChanged =
    s.state?.has.pln !== prev.state?.has.pln ||
    JSON.stringify(s.state?.plan) !== JSON.stringify(prev.state?.plan) ||
    s.state?.revisions?.workspace.pln !== prev.state?.revisions?.workspace.pln;
  if (planChanged) void refreshPlanDetail();
  const cstChanged =
    s.state?.has.cst !== prev.state?.has.cst ||
    s.state?.revisions?.workspace.ct !== prev.state?.revisions?.workspace.ct ||
    (s.state?.has.result ?? false) !== (prev.state?.has.result ?? false) ||
    s.state?.revisions?.workspace.result !== prev.state?.revisions?.workspace.result;
  if (cstChanged && s.state?.has.cst) void refreshCatalog();
  if (!s.state?.has.cst && getPlan().catalog) setPlan({ catalog: null });
});

if (useApp.getState().backend) unsubscribeBackend = useApp.getState().backend!.subscribe(onBackendEvent);
