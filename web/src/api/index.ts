import type { Backend } from "./backend";
import { MockBackend } from "./mock/mockBackend";

export type { Backend, BootStatus } from "./backend";
export { BackendError } from "./backend";
export * from "./protocol";

export type BackendKind = "mock" | "pyodide";

/**
 * Backend selected by the `backend` URL parameter (`?backend=mock|pyodide`).
 * Without the parameter the Pyodide backend is used; `VITE_BACKEND` overrides the default.
 */
export function backendKindFromUrl(search: string = globalThis.location?.search ?? ""): BackendKind {
  const param = new URLSearchParams(search).get("backend");
  const fallback = (import.meta.env.VITE_BACKEND as string | undefined) ?? "pyodide";
  const kind = (param ?? fallback).toLowerCase();
  return kind === "mock" ? "mock" : "pyodide";
}

export async function createBackend(kind: BackendKind, search: string = globalThis.location?.search ?? ""): Promise<Backend> {
  const params = new URLSearchParams(search);
  if (kind === "mock") {
    const speed = Number(params.get("speed") ?? "1");
    return new MockBackend({ speed: Number.isFinite(speed) ? speed : 1, bootMs: 400 });
  }
  // Loaded lazily so the mock build never touches the worker.
  const { PyodideBackend, PYODIDE_CDN } = await import("./pyodide/pyodideBackend");
  const source = params.get("pyodide");
  return new PyodideBackend({
    indexURL: source === "cdn" ? PYODIDE_CDN : (source ?? undefined),
    packageBaseUrl: params.get("packages") ?? undefined,
    pyrpBaseUrl: params.get("pyrp") ?? undefined,
  });
}
