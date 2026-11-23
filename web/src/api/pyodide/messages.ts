import type { BootStatus } from "../backend";
import type { BackendEvent, WireRequest, WireResponse } from "../protocol";

export interface WorkerConfig {
  /** Where `pyodide.mjs`, the wasm and the stdlib are served (trailing slash). */
  indexURL: string;
  /** Base URL of the Pyodide package wheels (numpy, scipy, ...). */
  packageBaseUrl: string;
  /** Directory with the pyRadPlan wheel and `manifest.json` (trailing slash). */
  pyrpBaseUrl: string;
  /** Shared Int32Array buffer for `pyodide.setInterruptBuffer` (cross-origin isolated only). */
  interruptBuffer?: SharedArrayBuffer;
}

export type MainToWorker =
  | { type: "init"; config: WorkerConfig }
  | { type: "request"; request: WireRequest }
  | { type: "memory" };

export type WorkerToMain =
  | { type: "boot"; status: BootStatus }
  | { type: "response"; response: WireResponse }
  | { type: "event"; event: BackendEvent }
  | { type: "stdio"; stream: "stdout" | "stderr"; text: string }
  | { type: "memory"; wasmBytes: number | null };
