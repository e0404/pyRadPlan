import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BackendError } from "../src/api/backend";
import { PYODIDE_CDN, PYODIDE_VERSION, PyodideBackend } from "../src/api/pyodide/pyodideBackend";
import type { MainToWorker } from "../src/api/pyodide/messages";

class FakeWorker {
  static last: FakeWorker;
  posted: MainToWorker[] = [];
  terminated = false;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: { message: string }) => void) | null = null;
  onmessageerror: (() => void) | null = null;

  constructor() {
    FakeWorker.last = this;
  }

  postMessage(message: MainToWorker) {
    this.posted.push(message);
  }

  terminate() {
    this.terminated = true;
  }
}

function read(path: string): string {
  return readFileSync(new URL(path, import.meta.url), "utf8");
}

describe("Pyodide version", () => {
  it("is the same in the app, the harness and its bootstrap", () => {
    const app = JSON.parse(read("../package.json")) as { devDependencies: Record<string, string> };
    const harness = JSON.parse(read("../pyodide/package.json")) as { dependencies: Record<string, string> };
    const bootstrap = /export const PYODIDE_VERSION = "([^"]+)"/.exec(read("../pyodide/bootstrap.js"))?.[1];
    expect(PYODIDE_VERSION).toBe(app.devDependencies.pyodide);
    expect(harness.dependencies.pyodide).toBe(PYODIDE_VERSION);
    expect(bootstrap).toBe(PYODIDE_VERSION);
    expect(PYODIDE_CDN).toBe(`https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`);
  });
});

describe("PyodideBackend worker failures", () => {
  afterEach(() => vi.unstubAllGlobals());

  function started() {
    vi.stubGlobal("Worker", FakeWorker);
    const backend = new PyodideBackend();
    return { backend, worker: FakeWorker.last };
  }

  it("rejects the pending requests and every later one when the worker crashes", async () => {
    const { backend, worker } = started();
    const first = backend.request("session.create", {});
    const second = backend.request("app.info", {});
    worker.onerror?.({ message: "out of memory" });
    for (const request of [first, second]) {
      const error = await request.catch((e: unknown) => e);
      expect(error).toBeInstanceOf(BackendError);
      expect((error as BackendError).message).toMatch(/worker stopped: out of memory/);
    }
    await expect(backend.request("app.info", {})).rejects.toBeInstanceOf(BackendError);
    await expect(backend.ready()).rejects.toThrow("out of memory");
    expect(backend.cancel()).toBe(false);
    expect(await backend.wasmMemory()).toBeNull();
  });

  it("answers requests normally until it crashes, and fails them when disposed", async () => {
    const { backend, worker } = started();
    const request = backend.request("session.create", {});
    const sent = worker.posted.find((m) => m.type === "request");
    expect(sent).toBeDefined();
    const id = (sent as Extract<MainToWorker, { type: "request" }>).request.id;
    worker.onmessage?.({ data: { type: "response", response: { id, ok: true, result: { session: "s1" } } } } as MessageEvent);
    await expect(request).resolves.toEqual({ session: "s1" });
    const pending = backend.request("app.info", {});
    backend.dispose();
    await expect(pending).rejects.toThrow("Backend disposed");
    expect(worker.terminated).toBe(true);
  });
});
