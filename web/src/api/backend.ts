import { decode, encode } from "./codec";
import type {
  BackendEvent,
  Method,
  ParamsOf,
  ResultOf,
  WireError,
  WireRequest,
  WireResponse,
} from "./protocol";

/** Boot progress of a backend (Pyodide download, package install, import). */
export interface BootStatus {
  phase: "starting" | "ready" | "failed";
  message: string;
  fraction: number | null;
}

export type EventListener = (event: BackendEvent) => void;
export type BootListener = (status: BootStatus) => void;

export interface Backend {
  readonly kind: "mock" | "pyodide";
  /** Resolves when requests can be served; rejects when booting failed. */
  ready(): Promise<void>;
  request<M extends Method>(method: M, params?: ParamsOf<M>, session?: string): Promise<ResultOf<M>>;
  /** Subscribe to `state` / `progress` / `log` events; returns the unsubscribe function. */
  subscribe(listener: EventListener): () => void;
  onBoot(listener: BootListener): () => void;
  /** Ask the running computation to stop (best effort). */
  cancel(): boolean;
  readonly canCancel: boolean;
  dispose(): void;
}

/** Error raised for a `{ok: false}` response. */
export class BackendError extends Error {
  readonly type: string;
  readonly traceback?: string;
  readonly code?: string;
  /** v1 `error.details` (`ValidationError` fields, tissue issues). */
  readonly details?: unknown;
  readonly method: string;

  constructor(method: string, error: WireError) {
    super(error.message);
    this.name = "BackendError";
    this.type = error.type;
    this.traceback = error.traceback;
    this.code = error.code;
    this.details = error.details;
    this.method = method;
  }
}

/**
 * Shared client logic: request ids, the array codec, event fan-out and boot status.
 * Subclasses only move wire messages ({@link transmit}).
 */
export abstract class BaseBackend implements Backend {
  abstract readonly kind: "mock" | "pyodide";
  abstract readonly canCancel: boolean;

  private nextId = 1;
  private readonly listeners = new Set<EventListener>();
  private readonly bootListeners = new Set<BootListener>();
  protected bootStatus: BootStatus = { phase: "starting", message: "Starting…", fraction: null };

  protected abstract transmit(request: WireRequest, transfer: ArrayBuffer[]): Promise<WireResponse>;
  abstract ready(): Promise<void>;
  abstract cancel(): boolean;

  async request<M extends Method>(
    method: M,
    params?: ParamsOf<M>,
    session?: string,
  ): Promise<ResultOf<M>> {
    const { value, buffers } = encode(params ?? {});
    const request: WireRequest = { id: this.nextId++, method, params: value };
    if (session !== undefined) request.session = session;
    if (buffers.length) request.buffers = buffers;
    const response = await this.transmit(request, buffers);
    if (!response.ok) throw new BackendError(method, response.error);
    return decode<ResultOf<M>>(response.result, response.buffers ?? []);
  }

  subscribe(listener: EventListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  onBoot(listener: BootListener): () => void {
    this.bootListeners.add(listener);
    listener(this.bootStatus);
    return () => this.bootListeners.delete(listener);
  }

  protected emit(event: BackendEvent): void {
    for (const listener of this.listeners) listener(event);
  }

  protected setBoot(status: BootStatus): void {
    this.bootStatus = status;
    for (const listener of this.bootListeners) listener(status);
  }

  dispose(): void {
    this.listeners.clear();
    this.bootListeners.clear();
  }
}
