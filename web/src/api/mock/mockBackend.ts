/**
 * In-process mock of `pyRadPlan.app`: same methods, events and wire format (results are
 * encoded with `{$array}` references and decoded by {@link BaseBackend}), with synthetic
 * phantoms, simulated progress and a fake dose.
 */
import { BaseBackend } from "../backend";
import { encode, ndarray } from "../codec";
import { isoCenterPoint } from "../protocol";
import type {
  AppInfo,
  LogLevel,
  PlanSummary,
  SessionState,
  StageKey,
  Vec3,
  WireError,
  WireRequest,
  WireResponse,
} from "../protocol";
import { MOCK_PHANTOMS, cumulativeDvh, fakeDose, makePhantom, type Phantom } from "./phantom";
import { REGISTRY } from "./planData";
import { planHandlers, planStateExtension } from "./plan";
import { createIoHandlers, ioStateExtension, recordLogEvent } from "./io";
import { viewerHandlers, viewerStateExtension } from "./viewer";
import type { BackendEvent } from "../protocol";

/** API the feature handlers (`./<feature>.ts`) get for one request. */
export interface MockContext {
  readonly request: WireRequest;
  readonly params: Record<string, unknown>;
  readonly speed: number;
  /** The request's session (`KeyError` when unknown). */
  session(): MockSession;
  /** A session by id (e.g. the one `session.create` just created). */
  sessionById(id: string): MockSession | undefined;
  state(s: MockSession): SessionState;
  /** Emit and return the session state. */
  publish(s: MockSession): SessionState;
  /** Run a long operation (busy flag, `busy` refusal, cancellation). */
  run<T>(s: MockSession, label: string, fn: () => Promise<T>): Promise<T>;
  /** Simulated computation time; throws the cancellation error after Stop. */
  sleep(ms: number): Promise<void>;
  progress(s: MockSession, label: string, fraction: number | null, detail?: string, data?: Record<string, number>): void;
  log(s: MockSession, level: LogLevel, message: string): void;
  emit(event: BackendEvent): void;
  /** Listen to the events emitted while the request runs (returns the unsubscribe function). */
  subscribe(listener: (event: BackendEvent) => void): () => void;
  /** Mark a stage as (re)computed now. */
  touch(s: MockSession, key: StageKey): void;
  stage(s: MockSession, key: StageKey): "missing" | "done" | "stale";
  isBusy(): boolean;
  /** Result of the next handler for this method (finally the v0 implementation). */
  fallback(): Promise<unknown>;
}

export type MockHandler = (params: Record<string, unknown>, ctx: MockContext) => unknown;
export type MockHandlers = Record<string, MockHandler>;

/** Feature handler tables, consulted in order before the v0 methods; one line per package. */
const FEATURE_HANDLERS: MockHandlers[] = [
  createIoHandlers(),
  planHandlers,
  viewerHandlers,
];

/** Additions of the feature modules to every SessionState; one line per package. */
const STATE_EXTENSIONS: Array<(s: MockSession, state: SessionState) => Partial<SessionState>> = [
  planStateExtension,
  viewerStateExtension,
  ioStateExtension,
];

export interface MockOptions {
  /** Multiplier for the simulated computation times (0 = instant, for tests). */
  speed?: number;
  /** Simulated boot time in ms. */
  bootMs?: number;
}

class Cancelled extends Error {
  constructor() {
    super("Computation cancelled by the user");
    this.name = "KeyboardInterrupt";
  }
}

export class MethodError extends Error {
  constructor(
    readonly type: string,
    message: string,
    readonly code?: string,
    readonly details?: unknown,
  ) {
    super(message);
  }
}

export const precondition = (message: string, code: string, details?: unknown) =>
  new MethodError("PreconditionError", message, code, details);

/**
 * Position of the structure a request addresses: a number is its position, a string its name
 * (ambiguous when several structures share it, as in `pyRadPlan.app`).
 */
export function resolveStructure(structures: readonly { name: string }[], ref: unknown): number {
  if (typeof ref === "number") {
    if (!Number.isInteger(ref) || ref < 0 || ref >= structures.length) throw new MethodError("KeyError", `Unknown structure ${ref}`);
    return ref;
  }
  const hits = structures.flatMap((st, i) => (st.name === ref ? [i] : []));
  if (!hits.length) throw new MethodError("KeyError", `Unknown structure ${JSON.stringify(ref)}`);
  if (hits.length > 1) throw new MethodError("KeyError", `Structure name ${JSON.stringify(ref)} is ambiguous (positions ${hits.join(", ")}); address it by position`);
  return hits[0];
}

const METHODS = [
  "app.info",
  "session.create",
  "session.close",
  "session.state",
  "data.load_phantom",
  "data.load_file",
  "plan.get",
  "plan.set",
  "workflow.generate_stf",
  "workflow.calc_dose_influence",
  "workflow.optimize",
  "workflow.recalc",
  "workflow.run_all",
  "view.volume",
  "view.structure_masks",
  "analysis.dvh",
];

const UPSTREAM: Record<StageKey, StageKey[]> = {
  data: [],
  plan: ["data"],
  stf: ["data", "plan"],
  dij: ["data", "plan", "stf"],
  optimization: ["data", "plan", "stf", "dij"],
};

export interface MockSession {
  /** Per-feature state of the feature handlers, keyed by feature. */
  ext: Record<string, unknown>;
  id: string;
  phantom: Phantom | null;
  plan: PlanSummary | null;
  /** Logical time at which each stage was last computed (absent = missing). */
  computed: Partial<Record<StageKey, number>>;
  dose: Float32Array | null;
  doseMax: number;
  busy: string | null;
}

/** Default plan as applied by the app layer (PROTOCOL.md amendment 8). */
export function defaultPlan(): PlanSummary {
  return {
    radiation_mode: "photons",
    machine: "Generic",
    num_of_fractions: 30,
    gantry_angles: [0],
    couch_angles: [0],
    bixel_width: 5,
    iso_center: null,
    dose_grid_resolution: [3, 3, 3],
    engine: REGISTRY.engines.photons[0],
    bio_model: "none",
  };
}

export class MockBackend extends BaseBackend {
  readonly kind = "mock" as const;
  readonly canCancel = true;

  private readonly speed: number;
  private readonly booted: Promise<void>;
  private readonly sessions = new Map<string, MockSession>();
  private sessionCounter = 0;
  private clock = 0;
  private cancelRequested = false;
  private busy = false;

  constructor(options: MockOptions = {}) {
    super();
    this.speed = options.speed ?? 1;
    const bootMs = options.bootMs ?? 0;
    this.booted = (async () => {
      this.setBoot({ phase: "starting", message: "Starting mock backend…", fraction: 0 });
      if (bootMs > 0) await this.sleep(bootMs, false);
      this.setBoot({ phase: "ready", message: "Mock backend ready", fraction: 1 });
    })();
  }

  ready(): Promise<void> {
    return this.booted;
  }

  cancel(): boolean {
    if (!this.busy) return false;
    this.cancelRequested = true;
    return true;
  }

  protected emit(event: BackendEvent): void {
    const recorded = recordLogEvent(event);
    if (recorded) super.emit(recorded);
  }

  protected async transmit(request: WireRequest): Promise<WireResponse> {
    await this.booted;
    try {
      const result = await this.dispatch(request, 0);
      const { value, buffers } = encode(result);
      return { id: request.id, ok: true, result: value, buffers };
    } catch (err) {
      const error: WireError =
        err instanceof MethodError
          ? {
              type: err.type,
              message: err.message,
              ...(err.code ? { code: err.code } : {}),
              ...(err.details !== undefined ? { details: err.details } : {}),
            }
          : err instanceof Error
            ? { type: err.name || "Error", message: err.message, traceback: err.stack }
            : { type: "Error", message: String(err) };
      return { id: request.id, ok: false, error };
    }
  }

  // -------------------------------------------------------------------------

  private async dispatch(request: WireRequest, from: number): Promise<unknown> {
    for (let i = from; i < FEATURE_HANDLERS.length; i++) {
      const handler = FEATURE_HANDLERS[i][request.method];
      if (handler) return handler((request.params ?? {}) as Record<string, unknown>, this.context(request, i + 1));
    }
    return this.handle(request);
  }

  private context(request: WireRequest, next: number): MockContext {
    return {
      request,
      params: (request.params ?? {}) as Record<string, unknown>,
      speed: this.speed,
      session: () => this.session(request),
      sessionById: (id) => this.sessions.get(id),
      state: (s) => this.state(s),
      publish: (s) => this.publish(s),
      run: (s, label, fn) => this.run(s, label, fn),
      sleep: (ms) => this.sleep(ms),
      progress: (s, label, fraction, detail, data) => this.progress(s, label, fraction, detail, data),
      log: (s, level, message) => this.log(s, level, message),
      emit: (event) => this.emit(event),
      subscribe: (listener) => this.subscribe(listener),
      touch: (s, key) => this.touch(s, key),
      stage: (s, key) => this.stage(s, key),
      isBusy: () => this.busy,
      fallback: () => this.dispatch(request, next),
    };
  }

  private async sleep(ms: number, cancellable = true): Promise<void> {
    const scaled = ms * this.speed;
    if (scaled > 0) await new Promise((r) => setTimeout(r, scaled));
    else await Promise.resolve();
    if (cancellable && this.cancelRequested) {
      this.cancelRequested = false;
      throw new Cancelled();
    }
  }

  private log(s: MockSession, level: LogLevel, message: string, report = false): void {
    this.emit({ event: "log", session: s.id, level, message, ...(report ? { report: true } : {}) });
  }

  private progress(s: MockSession, label: string, fraction: number | null, detail?: string, data?: Record<string, number>): void {
    this.emit({ event: "progress", session: s.id, label, fraction, detail: detail ?? null, ...(data ? { data } : {}) });
  }

  private session(request: WireRequest): MockSession {
    const s = request.session ? this.sessions.get(request.session) : undefined;
    if (!s) throw new MethodError("KeyError", `Unknown session ${JSON.stringify(request.session)}`);
    return s;
  }

  private stage(s: MockSession, key: StageKey): "missing" | "done" | "stale" {
    const t = s.computed[key];
    if (t === undefined) return "missing";
    return UPSTREAM[key].some((u) => (s.computed[u] ?? -1) > t) ? "stale" : "done";
  }

  private touch(s: MockSession, key: StageKey): void {
    s.computed[key] = ++this.clock;
  }

  state(s: MockSession): SessionState {
    const hasData = s.phantom !== null;
    const has = {
      ct: hasData,
      cst: hasData,
      pln: s.plan !== null,
      stf: s.computed.stf !== undefined,
      dij: s.computed.dij !== undefined,
      result: s.dose !== null,
    };
    const ready = has.ct && has.cst && has.pln;
    let status = "No data loaded";
    if (has.ct && !has.pln) status = "CT and structures loaded – configure a plan to continue";
    else if (ready && !has.dij) status = "Ready for dose influence calculation";
    else if (has.dij && this.stage(s, "dij") === "stale") status = "Dose influence outdated — recalculate";
    else if (has.dij && !has.result) status = "Ready for optimization";
    else if (has.result && this.stage(s, "optimization") === "stale") status = "Result outdated — re-optimize";
    else if (has.result) status = "Optimization result available";
    const grid = s.phantom?.grid;
    const state: SessionState = {
      session: s.id,
      has,
      stages: {
        data: this.stage(s, "data"),
        plan: this.stage(s, "plan"),
        stf: this.stage(s, "stf"),
        dij: this.stage(s, "dij"),
        optimization: this.stage(s, "optimization"),
      },
      actions: {
        generate_stf: ready,
        calc_dose_influence: ready,
        optimize: ready && has.dij,
        recalc: ready && has.dij && has.result,
      },
      ct: grid
        ? { shape: grid.shape, spacing: grid.spacing, origin: grid.origin, num_scenarios: 1 }
        : null,
      plan: s.plan ? { ...s.plan } : null,
      quantities: has.result ? ["physical_dose"] : [],
      structures:
        s.phantom?.structures.map((st, index) => ({
          name: st.name,
          type: st.type,
          color: st.color,
          visible: true,
          index,
        })) ?? [],
      status: s.busy ?? status,
      busy: s.busy,
    };
    for (const extend of STATE_EXTENSIONS) Object.assign(state, extend(s, state));
    return state;
  }

  private publish(s: MockSession): SessionState {
    const state = this.state(s);
    this.emit({ event: "state", session: s.id, state });
    return state;
  }

  private async run<T>(s: MockSession, label: string, fn: () => Promise<T>): Promise<T> {
    if (this.busy) throw precondition(`${s.busy ?? "A computation"} is still running`, "busy");
    this.busy = true;
    s.busy = label;
    this.cancelRequested = false;
    let result: T;
    try {
      result = await fn();
    } finally {
      this.busy = false;
      s.busy = null;
      this.cancelRequested = false;
    }
    // A returned SessionState is taken after the operation, i.e. no longer busy.
    if (result && typeof result === "object" && "has" in result) return { ...result, ...this.publish(s) };
    return result;
  }

  private requireReady(s: MockSession): asserts s is MockSession & { phantom: Phantom; plan: PlanSummary } {
    if (!s.phantom) throw precondition("Load a CT and structure set first", "no_data");
    if (!s.plan) throw precondition("Configure a plan first", "no_data");
  }

  private async generateStf(s: MockSession): Promise<void> {
    this.requireReady(s);
    const n = s.plan.gantry_angles.length;
    this.log(s, "info", `Generating steering information for ${n} beam(s)…`);
    for (let b = 0; b < n; b++) {
      this.progress(s, "Generating STF", b / n, `beam ${b + 1}/${n}`);
      await this.sleep(80);
    }
    this.progress(s, "Generating STF", 1);
    this.touch(s, "stf");
    this.log(s, "info", `STF generated: ${n * 180} bixels`);
    this.publish(s);
  }

  private async calcDij(s: MockSession): Promise<void> {
    this.requireReady(s);
    // v1 amendment 20: the beams are always generated again (the automatic isocenter follows
    // the structures, whose edits do not outdate the stf).
    await this.generateStf(s);
    const n = s.plan.gantry_angles.length;
    this.log(s, "info", `Dose influence calculation (${s.plan.engine ?? "default engine"})…`);
    this.progress(s, "Dose calculation", null, "preparing");
    await this.sleep(250);
    const steps = 20;
    for (let k = 0; k < steps; k++) {
      const beam = Math.min(n, Math.floor((k / steps) * n) + 1);
      this.progress(s, "Dose calculation", k / steps, `beam ${beam}/${n}`);
      await this.sleep(60);
    }
    this.progress(s, "Dose calculation", 1, `beam ${n}/${n}`);
    this.touch(s, "dij");
    this.log(s, "info", "Dose influence matrix computed (mock: 900 bixels × 41k voxels)");
    this.publish(s);
  }

  private async optimize(
    s: MockSession,
    maxIter: number,
  ): Promise<{ solver: string; num_iter: number; objective: number; stopped?: boolean }> {
    this.requireReady(s);
    if (s.computed.dij === undefined) {
      throw precondition("Calculate the dose influence first", "not_available");
    }
    this.log(s, "info", `Fluence optimization (max ${maxIter} iterations)…`);
    let objective = 1520;
    let iterations = maxIter;
    for (let it = 1; it <= maxIter; it++) {
      objective = objective * 0.9 + 4;
      if (it % 5 === 0 || it === maxIter) {
        this.progress(s, "Optimization", it / maxIter, `iteration ${it}/${maxIter} · f = ${objective.toFixed(2)}`, { iteration: it, objective });
      }
      try {
        await this.sleep(12);
      } catch (err) {
        // Stop ends an optimization after this iteration with its last iterate (as the real backend).
        if (!(err instanceof Cancelled)) throw err;
        iterations = it;
        break;
      }
    }
    this.computeDose(s);
    this.touch(s, "optimization");
    this.log(s, "info", `Optimization finished: objective ${objective.toFixed(3)}, max dose ${s.doseMax.toFixed(2)} Gy`);
    this.progress(s, "Optimization", 1);
    this.publish(s);
    return { solver: "scipy", num_iter: iterations, objective, ...(iterations < maxIter ? { stopped: true } : {}) };
  }

  private computeDose(s: MockSession & { phantom: Phantom; plan: PlanSummary }): void {
    const prescription = 2 * s.plan.num_of_fractions;
    const iso: Vec3 = isoCenterPoint(s.plan.iso_center) ?? [0, 0, 0];
    s.dose = fakeDose(s.phantom, s.plan.gantry_angles, prescription, iso);
    let max = 0;
    for (const v of s.dose) if (v > max) max = v;
    s.doseMax = max;
  }

  private async handle(request: WireRequest): Promise<unknown> {
    const params = (request.params ?? {}) as Record<string, unknown>;
    switch (request.method) {
      case "app.info": {
        const info: AppInfo = {
          version: "0.0.0-mock",
          python: "n/a (mock)",
          platform: "browser",
          backend: "mock",
          phantoms: [...MOCK_PHANTOMS],
          machines: REGISTRY.machines,
          solvers: REGISTRY.solvers,
          engines: REGISTRY.engines,
          quantities: REGISTRY.quantities,
          methods: [...new Set([...METHODS, ...FEATURE_HANDLERS.flatMap((h) => Object.keys(h))])],
        };
        return info;
      }
      case "session.create": {
        const id = `s${++this.sessionCounter}`;
        const s: MockSession = { ext: {}, id, phantom: null, plan: null, computed: {}, dose: null, doseMax: 0, busy: null };
        this.sessions.set(id, s);
        this.publish(s);
        return { session: id };
      }
      case "session.close": {
        this.sessions.delete(this.session(request).id);
        return {};
      }
      case "session.state":
        return this.state(this.session(request));
      case "data.load_phantom": {
        const s = this.session(request);
        const name = String(params.name ?? "");
        return this.run(s, `Loading ${name}`, async () => {
          this.progress(s, `Loading ${name}`, null);
          await this.sleep(200);
          this.loadPhantom(s, makePhantom(name));
          this.log(s, "info", `Loaded phantom ${name}`);
          return this.publish(s);
        });
      }
      case "data.load_file": {
        const s = this.session(request);
        const filename = String(params.filename ?? "file");
        const data = params.data as { data?: Uint8Array } | undefined;
        const bytes = data?.data?.length ?? 0;
        return this.run(s, `Loading ${filename}`, async () => {
          this.progress(s, `Reading ${filename}`, null);
          await this.sleep(300);
          if (!/\.(mat|npz|pkl|pickle|nrrd|nii|nii\.gz|mha|dcm|zip)$/i.test(filename)) {
            throw new MethodError("ValueError", `Unsupported file type: ${filename}`);
          }
          // The mock cannot parse files; it stands in the TG119 phantom.
          this.loadPhantom(s, makePhantom("TG119"));
          this.log(s, "warning", `Mock backend: ${filename} (${bytes} bytes) replaced by the TG119 phantom`, true);
          return this.publish(s);
        });
      }
      case "plan.get":
        return this.session(request).plan;
      case "workflow.generate_stf": {
        const s = this.session(request);
        return this.run(s, "Generating STF", async () => {
          await this.generateStf(s);
          return this.state(s);
        });
      }
      case "workflow.calc_dose_influence": {
        const s = this.session(request);
        return this.run(s, "Dose influence calculation", async () => {
          await this.calcDij(s);
          return this.state(s);
        });
      }
      case "workflow.optimize": {
        const s = this.session(request);
        const maxIter = Number(params.max_iter ?? 60);
        return this.run(s, "Optimization", async () => {
          const optimization = await this.optimize(s, maxIter);
          return { ...this.state(s), optimization };
        });
      }
      case "workflow.recalc": {
        const s = this.session(request);
        return this.run(s, "Dose recalculation", async () => {
          this.requireReady(s);
          if (!s.dose) throw precondition("No optimized weights to recalculate", "no_weights");
          this.progress(s, "Dose recalculation", null);
          await this.sleep(300);
          this.computeDose(s);
          this.touch(s, "optimization");
          this.log(s, "info", `Dose recalculated, max ${s.doseMax.toFixed(2)} Gy`);
          return this.publish(s);
        });
      }
      case "workflow.run_all": {
        const s = this.session(request);
        const maxIter = Number(params.max_iter ?? 60);
        return this.run(s, "Running all steps", async () => {
          if (!s.phantom) throw precondition("Load a CT and structure set first", "no_data");
          if (!s.plan) {
            s.plan = defaultPlan();
            this.touch(s, "plan");
          }
          // Only the steps that are missing or stale.
          if (this.stage(s, "dij") !== "done") await this.calcDij(s);
          if (this.stage(s, "optimization") !== "done") await this.optimize(s, maxIter);
          return this.state(s);
        });
      }
      case "view.volume": {
        const s = this.session(request);
        const name = String(params.name ?? "ct");
        if (!s.phantom) throw new MethodError("ValueError", "No CT loaded");
        const grid = s.phantom.grid;
        if (name === "ct") {
          let min = Infinity;
          let max = -Infinity;
          for (const v of s.phantom.ct) {
            if (v < min) min = v;
            if (v > max) max = v;
          }
          return { name: "ct", unit: "HU", data: ndarray(s.phantom.ct, grid.shape), grid, range: [min, max] };
        }
        if (name === "physical_dose" && s.dose) {
          return { name, unit: "Gy", data: ndarray(s.dose, grid.shape), grid, range: [0, s.doseMax] };
        }
        throw new MethodError("KeyError", `Unknown quantity ${name}`);
      }
      case "view.structure_masks": {
        const s = this.session(request);
        if (!s.phantom) throw new MethodError("ValueError", "No structures loaded");
        const phantom = s.phantom;
        const wanted = Array.isArray(params.names) ? new Set(params.names.map((n) => resolveStructure(phantom.structures, n))) : null;
        const grid = phantom.grid;
        return {
          grid,
          masks: phantom.structures
            .map((st, index) => ({ st, index }))
            .filter(({ index }) => !wanted || wanted.has(index))
            .map(({ st, index }) => ({ index, name: st.name, color: st.color, data: ndarray(st.mask, grid.shape) })),
        };
      }
      case "analysis.dvh": {
        const s = this.session(request);
        if (!s.phantom || !s.dose) throw new MethodError("ValueError", "No result available");
        const dose = s.dose;
        const max = s.doseMax * 1.05;
        return {
          quantity: "physical_dose",
          unit: "Gy",
          curves: s.phantom.structures.map((st, index) => {
            const dvh = cumulativeDvh(dose, st.mask, max);
            return { index, name: st.name, color: st.color, dose: ndarray(dvh.dose, [dvh.dose.length]), volume: ndarray(dvh.volume, [dvh.volume.length]) };
          }),
        };
      }
      default:
        throw new MethodError("NotImplementedError", `Unknown method ${request.method}`);
    }
  }

  private loadPhantom(s: MockSession, phantom: Phantom): void {
    s.phantom = phantom;
    s.computed = {};
    s.dose = null;
    s.doseMax = 0;
    this.touch(s, "data");
    // As in the Qt GUI, a default plan is applied as soon as CT and structures are loaded.
    s.plan = defaultPlan();
    this.touch(s, "plan");
  }
}
