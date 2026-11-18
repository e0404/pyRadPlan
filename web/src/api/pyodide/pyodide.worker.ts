/**
 * Module web worker hosting Pyodide + pyRadPlan.
 *
 * Boot: load the Pyodide runtime (self-hosted from the npm package, see vite.config.ts),
 * install pyRadPlan with `installPyRadPlan` from `web/pyodide/bootstrap.js` using the wheels
 * listed in `<pyrp>/manifest.json`, then import `pyRadPlan.app` through `glue.py`.
 * Requests are forwarded one at a time to `pyRadPlan.app.dispatch(request, emit)`; events
 * stream back while a request runs.
 */
import type { PyodideAPI } from "pyodide";
import type { BackendEvent, WireRequest, WireResponse } from "../protocol";
import type { BootStatus } from "../backend";
import type { MainToWorker, WorkerConfig, WorkerToMain } from "./messages";
import glueSource from "./glue.py?raw";

interface WorkerScope {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((ev: MessageEvent) => void) | null;
}
const scope = self as unknown as WorkerScope;

/** The parts of `web/pyodide/bootstrap.js` used here. */
interface WheelSources {
  wheel: string;
  shimWheel: string;
  pureWheels: string[];
}

interface InstallProgress {
  stage: string;
  fraction: number;
  detail?: string;
}

interface BootstrapModule {
  installPyRadPlan: (
    pyodide: PyodideAPI,
    options: WheelSources & { plotting?: boolean; onProgress?: (event: InstallProgress) => void },
  ) => Promise<{ version: string; timings: Record<string, number> }>;
  wheelsFromManifest: (manifest: unknown, baseUrl: string) => WheelSources;
}

// Resolved at build time (bundled into the worker); empty when bootstrap.js is missing.
const bootstrapModules = import.meta.glob<BootstrapModule>("../../../pyodide/bootstrap.js");

const STAGE_LABEL: Record<string, string> = {
  "pyodide-packages": "Loading numpy, scipy and the other Pyodide packages…",
  "pure-wheels": "Installing pure-Python dependencies…",
  pyradplan: "Installing the pyRadPlan wheel…",
  import: "Importing pyRadPlan…",
  ready: "pyRadPlan installed",
};

let dispatchFn: ((req: string, buffers: Uint8Array[], emit: (json: string) => void) => Promise<unknown>) | null =
  null;
let interrupt: Int32Array | null = null;
/** The loaded runtime (wasm memory size requests). */
let runtime: PyodideAPI | null = null;
let booting: Promise<void> | null = null;
const queue: WireRequest[] = [];
let running = false;

function post(message: WorkerToMain, transfer: Transferable[] = []): void {
  scope.postMessage(message, transfer);
}

function boot(status: BootStatus): void {
  post({ type: "boot", status });
}

function progress(message: string, fraction: number | null): void {
  boot({ phase: "starting", message, fraction });
}

async function loadBootstrap(): Promise<BootstrapModule> {
  const loader = Object.values(bootstrapModules)[0];
  if (!loader) throw new Error("web/pyodide/bootstrap.js was not found at build time");
  return loader();
}

async function initialize(config: WorkerConfig): Promise<void> {
  progress("Loading Python runtime…", 0.02);
  const [bootstrap, manifest] = await Promise.all([
    loadBootstrap(),
    fetch(new URL("manifest.json", config.pyrpBaseUrl)).then((res) => {
      if (!res.ok) throw new Error(`pyRadPlan wheel manifest not found (${res.status} ${res.url})`);
      return res.json() as Promise<unknown>;
    }),
  ]);
  const mod = (await import(/* @vite-ignore */ new URL("pyodide.mjs", config.indexURL).href)) as {
    loadPyodide: (options: Record<string, unknown>) => Promise<PyodideAPI>;
  };
  const py = (runtime = await mod.loadPyodide({
    indexURL: config.indexURL,
    packageBaseUrl: config.packageBaseUrl,
    stdout: (text: string) => post({ type: "stdio", stream: "stdout", text }),
    stderr: (text: string) => post({ type: "stdio", stream: "stderr", text }),
  }));
  if (config.interruptBuffer) {
    interrupt = new Int32Array(config.interruptBuffer);
    py.setInterruptBuffer(interrupt);
  }
  progress(`Python ${py.runPython("import sys; sys.version.split()[0]")} ready`, 0.2);

  const wheels = bootstrap.wheelsFromManifest(manifest, config.pyrpBaseUrl);
  const { version, timings } = await bootstrap.installPyRadPlan(py, {
    ...wheels,
    onProgress: (event) => {
      const label = STAGE_LABEL[event.stage] ?? event.stage;
      progress(event.detail ? `${label} (${event.detail})` : label, 0.2 + 0.7 * event.fraction);
    },
  });
  post({
    type: "stdio",
    stream: "stdout",
    text: `pyRadPlan ${version} installed: ${Object.entries(timings)
      .map(([k, v]) => `${k} ${v.toFixed(1)} s`)
      .join(", ")}`,
  });

  progress("Importing pyRadPlan.app…", 0.93);
  const glue = py.toPy({});
  py.runPython(glueSource, { globals: glue });
  glue.get("_get_app")();
  dispatchFn = glue.get("web_dispatch");
  boot({ phase: "ready", message: `pyRadPlan ${version} ready`, fraction: 1 });
}

function errorResponse(id: number, err: unknown): WireResponse {
  const e = err instanceof Error ? err : new Error(String(err));
  return { id, ok: false, error: { type: e.name || "Error", message: e.message, traceback: e.stack } };
}

async function handle(request: WireRequest): Promise<void> {
  try {
    if (booting) await booting;
    if (!dispatchFn) throw new Error("pyRadPlan is not loaded");
    if (interrupt) Atomics.store(interrupt, 0, 0);
    const { buffers = [], ...rest } = request;
    const emit = (json: string) => post({ type: "event", event: JSON.parse(json) as BackendEvent });
    const proxy = (await dispatchFn(
      JSON.stringify(rest),
      buffers.map((b) => new Uint8Array(b)),
      emit,
    )) as unknown;
    // `to_js([json, *buffers])` yields a JS array (or a proxy of one in older Pyodide).
    const parts = (Array.isArray(proxy) ? proxy : Array.from(proxy as Iterable<unknown>)) as [string, ...Uint8Array[]];
    const [json, ...arrays] = parts;
    const response = JSON.parse(json) as WireResponse;
    const transfer: ArrayBuffer[] = arrays.map((a) =>
      a.byteOffset === 0 && a.byteLength === a.buffer.byteLength
        ? (a.buffer as ArrayBuffer)
        : (a.slice().buffer as ArrayBuffer),
    );
    if (response.ok) response.buffers = transfer;
    post({ type: "response", response }, transfer);
  } catch (err) {
    post({ type: "response", response: errorResponse(request.id, err) });
  }
}

async function pump(): Promise<void> {
  if (running) return;
  running = true;
  try {
    while (queue.length) await handle(queue.shift()!);
  } finally {
    running = false;
  }
}

scope.onmessage = (ev: MessageEvent<MainToWorker>) => {
  const msg = ev.data;
  if (msg.type === "init") {
    booting = initialize(msg.config).catch((err: unknown) => {
      const e = err instanceof Error ? err : new Error(String(err));
      boot({ phase: "failed", message: e.message, fraction: null });
      post({ type: "stdio", stream: "stderr", text: e.stack ?? e.message });
      throw e;
    });
    booting.catch(() => undefined);
  } else if (msg.type === "memory") {
    // Size of the wasm heap; it only grows, so this is the peak so far.
    const heap = (runtime as unknown as { _module?: { HEAPU8?: Uint8Array } } | null)?._module?.HEAPU8;
    post({ type: "memory", wasmBytes: heap ? heap.buffer.byteLength : null });
  } else if (msg.type === "request") {
    queue.push(msg.request);
    void pump();
  }
};
