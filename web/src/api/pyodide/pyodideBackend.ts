import { devDependencies } from "../../../package.json";
import { BaseBackend } from "../backend";
import type { WireRequest, WireResponse } from "../protocol";
import type { MainToWorker, WorkerConfig, WorkerToMain } from "./messages";

export interface PyodideBackendOptions {
  /** Pyodide runtime base URL; defaults to the self-hosted copy at `/pyodide-runtime/`. */
  indexURL?: string;
  /** Package wheel base URL; defaults to the jsDelivr CDN of the same Pyodide version. */
  packageBaseUrl?: string;
  /** pyRadPlan wheel + manifest directory; defaults to `/pyrp/`. */
  pyrpBaseUrl?: string;
}

/** Pyodide version of the npm package (`web/package.json`); the CDN path for package wheels depends on it. */
export const PYODIDE_VERSION: string = devDependencies.pyodide;
export const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

function absolute(url: string): string {
  return new URL(url, globalThis.location?.href ?? "http://localhost/").href;
}

/** Backend running `pyRadPlan.app` in Pyodide inside a module web worker. */
export class PyodideBackend extends BaseBackend {
  readonly kind = "pyodide" as const;

  private readonly worker: Worker;
  private readonly pending = new Map<number, (response: WireResponse) => void>();
  /** Why the worker is gone (it crashed or was disposed): every request fails with it. */
  private lost: string | null = null;
  private readonly memoryWaiters: ((bytes: number | null) => void)[] = [];
  private readonly interrupt: Int32Array | null;
  private readonly booted: Promise<void>;
  private inFlight = 0;

  constructor(options: PyodideBackendOptions = {}) {
    super();
    const base = import.meta.env.BASE_URL;
    const config: WorkerConfig = {
      indexURL: absolute(options.indexURL ?? `${base}pyodide-runtime/`),
      packageBaseUrl: absolute(options.packageBaseUrl ?? PYODIDE_CDN),
      pyrpBaseUrl: absolute(options.pyrpBaseUrl ?? `${base}pyrp/`),
    };
    if (globalThis.crossOriginIsolated && typeof SharedArrayBuffer !== "undefined") {
      config.interruptBuffer = new SharedArrayBuffer(4);
      this.interrupt = new Int32Array(config.interruptBuffer);
    } else {
      this.interrupt = null;
    }

    this.worker = new Worker(new URL("./pyodide.worker.ts", import.meta.url), {
      type: "module",
      name: "pyradplan-pyodide",
    });
    let resolveReady!: () => void;
    let rejectReady!: (err: Error) => void;
    this.booted = new Promise<void>((resolve, reject) => {
      resolveReady = resolve;
      rejectReady = reject;
    });
    this.booted.catch(() => undefined);

    this.worker.onmessage = (ev: MessageEvent<WorkerToMain>) => {
      const msg = ev.data;
      switch (msg.type) {
        case "boot":
          this.setBoot(msg.status);
          if (msg.status.phase === "ready") resolveReady();
          else if (msg.status.phase === "failed") rejectReady(new Error(msg.status.message));
          break;
        case "response": {
          const resolve = this.pending.get(msg.response.id);
          this.pending.delete(msg.response.id);
          this.inFlight = Math.max(0, this.inFlight - 1);
          resolve?.(msg.response);
          break;
        }
        case "event":
          this.emit(msg.event);
          break;
        case "memory":
          for (const resolve of this.memoryWaiters.splice(0)) resolve(msg.wasmBytes);
          break;
        case "stdio":
          this.emit({
            event: "log",
            session: "",
            level: msg.stream === "stderr" ? "warning" : "info",
            message: msg.text,
          });
          break;
      }
    };
    this.worker.onerror = (ev) => {
      const message = ev.message || "Worker failed to start";
      this.setBoot({ phase: "failed", message, fraction: null });
      rejectReady(new Error(message));
      this.failAll(`The Pyodide worker stopped: ${message}`);
    };
    this.worker.onmessageerror = () => this.failAll("The Pyodide worker sent a message that could not be read");
    this.post({ type: "init", config });
  }

  get canCancel(): boolean {
    return this.interrupt !== null;
  }

  ready(): Promise<void> {
    return this.booted;
  }

  cancel(): boolean {
    if (!this.interrupt || this.inFlight === 0) return false;
    // 2 = SIGINT: raises KeyboardInterrupt at the next bytecode check in Python.
    Atomics.store(this.interrupt, 0, 2);
    return true;
  }

  protected transmit(request: WireRequest, _transfer: ArrayBuffer[]): Promise<WireResponse> {
    return new Promise((resolve) => {
      if (this.lost !== null) {
        resolve(this.failure(request.id, this.lost));
        return;
      }
      this.pending.set(request.id, resolve);
      this.inFlight++;
      // Request buffers (uploads) are copied, not transferred: callers may send the same file
      // twice (e.g. base_data.add dry run, then the real add).
      this.post({ type: "request", request });
    });
  }

  /** Size of the worker's wasm memory in bytes (the peak so far; answered between requests). */
  wasmMemory(): Promise<number | null> {
    return new Promise((resolve) => {
      if (this.lost !== null) {
        resolve(null);
        return;
      }
      this.memoryWaiters.push(resolve);
      this.post({ type: "memory" });
    });
  }

  private post(message: MainToWorker, transfer: Transferable[] = []): void {
    this.worker.postMessage(message, transfer);
  }

  private failure(id: number, message: string): WireResponse {
    return { id, ok: false, error: { type: "RuntimeError", message } };
  }

  /** Fail every pending request and every later one: a dead worker never answers. */
  private failAll(message: string): void {
    this.lost ??= message;
    for (const [id, resolve] of this.pending) resolve(this.failure(id, message));
    this.pending.clear();
    this.inFlight = 0;
    for (const resolve of this.memoryWaiters.splice(0)) resolve(null);
  }

  dispose(): void {
    super.dispose();
    this.worker.terminate();
    this.failAll("Backend disposed");
  }
}
