/**
 * Application state: backend connection, session state from `pyRadPlan.app`, the volumes
 * shown by the viewer, log, progress, and viewer settings. Mirrors what the Qt GUI keeps in
 * its widgets and the `WorkspaceManager`.
 */
import { create } from "zustand";
import {
  BackendError,
  type AppInfo,
  type Backend,
  type BackendEvent,
  type BootStatus,
  type LogLevel,
  type Method,
  type ParamsOf,
  type PlanSummary,
  type SessionState,
  type StructureMasks,
  type Volume,
} from "../api";
import { ndarray } from "../api/codec";
import { viewerSliceInitial, type ViewerSlice } from "./viewerSlice";
import { scheduleViewSync, syncViews } from "./revisions";
import { DEFAULT_VIEW, type Plane, type PlaneConvention, type ViewState } from "../renderer/geometry";

export interface LogEntry {
  id: number;
  time: Date;
  level: LogLevel;
  message: string;
  /** Python logger of a backend record (Qt log format `HH:MM:SS [LEVEL] logger: message`). */
  logger?: string;
  /** Per-session sequence number of a backend record (`session.history` resumes after it). */
  seq?: number;
}

export interface ErrorInfo {
  title: string;
  message: string;
  details?: string;
}

export interface Progress {
  label: string;
  fraction: number | null;
  detail?: string | null;
}

export interface ConfirmInfo {
  title: string;
  message: string;
  confirmLabel: string;
  onConfirm: () => void;
}

export type ViewerMode = "quantity" | "ct";

export interface ViewerSettings {
  plane: Plane;
  slices: Record<Plane, number>;
  convention: PlaneConvention;
  /** Which volume the Viewer Options edit and the colourbar show. */
  mode: ViewerMode;
  /** `[center, width]`, or null for the volume's full range (Qt default). */
  ctWindow: [number, number] | null;
  quantityWindow: [number, number] | null;
  ctColormap: string;
  quantityColormap: string;
  opacity: number;
  showCT: boolean;
  showCST: boolean;
  showQuantity: boolean;
  showIsolines: boolean;
  showIsocenter: boolean;
  showOrientation: boolean;
  /** Custom isoline levels; empty for the default 50/80/95 % of the maximum. */
  isolineLevels: number[];
  activeQuantity: string | null;
  visible: Record<string, boolean>;
  view: ViewState;
}

export const DEFAULT_VIEWER: ViewerSettings = {
  plane: "axial",
  slices: { axial: 0, sagittal: 0, coronal: 0 },
  convention: "pyradplan",
  mode: "quantity",
  ctWindow: null,
  quantityWindow: null,
  ctColormap: "gray",
  quantityColormap: "jet",
  opacity: 0.4,
  showCT: true,
  showCST: true,
  showQuantity: true,
  showIsolines: false,
  showIsocenter: false,
  showOrientation: true,
  isolineLevels: [],
  activeQuantity: null,
  visible: {},
  view: DEFAULT_VIEW,
};

export interface AppStore extends ViewerSlice {
  backend: Backend | null;
  boot: BootStatus;
  info: AppInfo | null;
  session: string | null;
  state: SessionState | null;
  busy: string | null;
  progress: Progress | null;
  logs: LogEntry[];
  error: ErrorInfo | null;
  confirm: ConfirmInfo | null;
  statusMessage: string;
  ct: Volume | null;
  quantities: Record<string, Volume>;
  masks: StructureMasks | null;
  viewer: ViewerSettings;
  showLogPanel: boolean;
}

const MAX_LOGS = 2000;
let logId = 0;

export const useApp = create<AppStore>(() => ({
  backend: null,
  boot: { phase: "starting", message: "Starting…", fraction: null },
  info: null,
  session: null,
  state: null,
  busy: null,
  progress: null,
  logs: [],
  error: null,
  confirm: null,
  statusMessage: "",
  ct: null,
  quantities: {},
  masks: null,
  viewer: DEFAULT_VIEWER,
  showLogPanel: true,
  ...viewerSliceInitial,
}));

const get = useApp.getState;
const set = useApp.setState;

export function log(level: LogLevel, message: string, extra: { logger?: string; seq?: number; time?: Date } = {}): void {
  const entry: LogEntry = { id: ++logId, time: extra.time ?? new Date(), level, message, logger: extra.logger, seq: extra.seq };
  set((s) => ({ logs: [...s.logs, entry].slice(-MAX_LOGS) }));
}

export function setViewer(partial: Partial<ViewerSettings> | ((v: ViewerSettings) => Partial<ViewerSettings>)): void {
  set((s) => ({ viewer: { ...s.viewer, ...(typeof partial === "function" ? partial(s.viewer) : partial) } }));
}

export function setSlice(plane: Plane, index: number): void {
  setViewer((v) => ({ slices: { ...v.slices, [plane]: index } }));
}

const CANCELLED_TYPES = new Set(["KeyboardInterrupt", "ComputeCancelledError"]);

function reportError(title: string, err: unknown): void {
  if (err instanceof BackendError && CANCELLED_TYPES.has(err.type)) {
    log("warning", `${title} stopped${err.message ? `: ${err.message}` : ""}`);
    set({ statusMessage: `${title} stopped` });
    return;
  }
  const e =
    err instanceof BackendError
      ? { title, message: `${err.type}: ${err.message}`, details: err.traceback }
      : { title, message: err instanceof Error ? err.message : String(err), details: err instanceof Error ? err.stack : undefined };
  log("error", `${title}: ${e.message}`);
  set({ error: e, statusMessage: `${title} failed` });
}

export function dismissError(): void {
  set({ error: null });
}

export function dismissConfirm(): void {
  set({ confirm: null });
}

function onEvent(event: BackendEvent): void {
  const session = get().session;
  if (event.session && session && event.session !== session) return;
  switch (event.event) {
    case "state":
      set({ state: event.state });
      scheduleViewSync();
      break;
    case "progress":
      set({ progress: { label: event.label, fraction: event.fraction, detail: event.detail } });
      break;
    case "log":
      if (event.repeat) break;
      log(event.level, event.message, { logger: event.logger, seq: (event as { seq?: number }).seq });
      break;
  }
}

async function call<M extends Method>(method: M, params?: ParamsOf<M>) {
  const { backend, session } = get();
  if (!backend) throw new Error("Backend not connected");
  return backend.request(method, params, session ?? undefined);
}

export async function connect(backend: Backend): Promise<void> {
  set({ backend });
  backend.onBoot((boot) => set({ boot }));
  backend.subscribe(onEvent);
  try {
    await backend.ready();
    const info = await backend.request("app.info", {});
    const { session } = await backend.request("session.create", {});
    set({ info, session });
    const state = await backend.request("session.state", {}, session);
    set({ state, statusMessage: `Connected to ${info.backend} backend (pyRadPlan ${info.version})` });
    log("info", `Session ${session} on ${info.backend} backend · pyRadPlan ${info.version} · Python ${info.python}`);
  } catch (err) {
    reportError("Startup", err);
  }
}

type Reload = "data" | "result" | "none";

/** Methods accepting `physical_only` (PROTOCOL.md amendment 3). */
const PHYSICAL_ONLY_METHODS = new Set<Method>(["workflow.calc_dose_influence", "workflow.recalc", "workflow.run_all"]);

/** Run a workflow request with busy / progress handling and reload the affected views. */
async function run<M extends Method>(label: string, method: M, params: ParamsOf<M>, reload: Reload): Promise<boolean> {
  if (get().busy) return false;
  set({ busy: label, progress: { label, fraction: null }, statusMessage: `${label}…` });
  const started = performance.now();
  try {
    const result = await call(method, params);
    if (result && typeof result === "object" && "has" in result) set({ state: result as SessionState });
    const summary = (result as { optimization?: { solver?: string; num_iter?: number; objective?: number } } | null)?.optimization;
    if (summary) {
      log("info", `Optimization: ${summary.solver ?? "solver"}, ${summary.num_iter ?? "?"} iterations, objective ${summary.objective?.toPrecision(6) ?? "?"}`);
    }
    await syncViews(reload === "none" ? undefined : reload);
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    set({ statusMessage: `${label} finished (${seconds} s)` });
    return true;
  } catch (err) {
    if (err instanceof BackendError && err.code === "bio_model_unsupported" && PHYSICAL_ONLY_METHODS.has(method)) {
      // As the Qt dialog: offer to continue with physical dose only.
      log("warning", `${label}: ${err.message}`);
      set({
        statusMessage: `${label}: biological model not applicable`,
        confirm: {
          title: label,
          message: `${err.message}

Continue with physical dose only? The plan's biological model is set to "none".`,
          confirmLabel: "Physical dose only",
          onConfirm: () => {
            set({ confirm: null });
            void run(label, method, { ...(params as object), physical_only: true } as ParamsOf<M>, reload);
          },
        },
      });
      return false;
    }
    reportError(label, err);
    return false;
  } finally {
    set({ busy: null, progress: null });
  }
}

/** Refetch the viewer data after a data load (revision-driven, see `revisions.ts`). */
function reloadData(): Promise<void> {
  return syncViews("data");
}

/** Refetch the result quantities (revision-driven, see `revisions.ts`). */
function reloadQuantities(): Promise<void> {
  return syncViews("result");
}

/** Centre of mass (mm) of the first target structure, like the Qt "Auto" isocenter preview. */
export function targetCenter(masks: StructureMasks | null, state: SessionState | null): [number, number, number] | null {
  if (!masks || !state) return null;
  const target = state.structures.find((s) => s.type.toUpperCase() === "TARGET");
  const mask = masks.masks.find((m) => m.name === target?.name);
  if (!mask) return null;
  const [nz, ny, nx] = masks.grid.shape;
  let sx = 0;
  let sy = 0;
  let sz = 0;
  let n = 0;
  const d = mask.data.data;
  let i = 0;
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let c = 0; c < nx; c++, i++) {
        if (d[i]) {
          sx += c;
          sy += j;
          sz += k;
          n++;
        }
      }
    }
  }
  if (!n) return null;
  const g = masks.grid;
  return [g.origin[0] + (sx / n) * g.spacing[0], g.origin[1] + (sy / n) * g.spacing[1], g.origin[2] + (sz / n) * g.spacing[2]];
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export const actions = {
  loadPhantom: (name: string) => run(`Loading ${name}`, "data.load_phantom", { name }, "data"),

  async loadFile(file: File): Promise<boolean> {
    const bytes = new Uint8Array(await file.arrayBuffer());
    log("info", `Uploading ${file.name} (${(bytes.length / 1024).toFixed(0)} KiB)`);
    return run(`Loading ${file.name}`, "data.load_file", { filename: file.name, data: ndarray(bytes, [bytes.length]) }, "data");
  },

  applyPlan: (plan: Partial<PlanSummary>) => run("Applying plan", "plan.set", plan, "none"),
  generateStf: () => run("Generating STF", "workflow.generate_stf", {}, "none"),
  calcDoseInfluence: () => run("Dose influence calculation", "workflow.calc_dose_influence", {}, "none"),
  optimize: (params: ParamsOf<"workflow.optimize"> = {}) => run("Optimization", "workflow.optimize", params, "result"),
  recalc: () => run("Dose recalculation", "workflow.recalc", {}, "result"),
  runAll: () => run("Running all steps", "workflow.run_all", {}, "result"),

  async refresh(): Promise<void> {
    try {
      const state = await call("session.state", {});
      set({ state });
      await reloadData();
      await reloadQuantities();
    } catch (err) {
      reportError("Refresh", err);
    }
  },

  stop(): void {
    const { backend, busy } = get();
    if (!backend || !busy) return;
    if (backend.cancel()) {
      log("warning", `Stop requested for: ${busy}`);
      set({ statusMessage: "Stopping…" });
    } else {
      log("warning", "This backend cannot interrupt the running computation");
    }
  },
};

/** Internals for feature modules (`store/ioActions.ts`); not part of the component API. */
export const storeInternals = { call, run, reportError, reloadData, reloadQuantities };
