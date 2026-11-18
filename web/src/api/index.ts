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

/** Where URL parameters may point the backend at code (see {@link trustedSource}). */
export interface SourceTrust {
  /** Development server: every value is honoured. */
  dev: boolean;
  /** Origin of the page; other origins are refused outside development. */
  origin: string;
  /** URL relative values resolve against. */
  base: string;
}

function pageTrust(): SourceTrust {
  const location = globalThis.location;
  return {
    dev: Boolean(import.meta.env.DEV),
    origin: location?.origin ?? "",
    base: location?.href ?? "http://localhost/",
  };
}

/**
 * The value of a URL parameter naming where the backend loads code from (`pyodide`,
 * `packages`, `pyrp`), or `undefined` for the default. Outside development only URLs on the
 * page's own origin are honoured: on a deployment, a crafted link must not make the app run
 * code from elsewhere.
 */
export function trustedSource(value: string | null, trust: SourceTrust = pageTrust()): string | undefined {
  if (value === null || value === "") return undefined;
  if (trust.dev) return value;
  try {
    if (new URL(value, trust.base).origin === trust.origin) return value;
  } catch {
    // Not a URL: refused like a foreign one.
  }
  console.warn(`Ignoring the URL parameter value "${value}": not on this site's origin`);
  return undefined;
}

/**
 * The Pyodide backend locations given by URL parameters, as far as they are trusted.
 * `pyodide=cdn` (the official Pyodide CDN, which also serves the default package wheels) is
 * always honoured; it names no location of its own.
 */
export function backendSources(search: string, trust: SourceTrust = pageTrust()) {
  const params = new URLSearchParams(search);
  const pyodide = params.get("pyodide");
  return {
    pyodide: pyodide === "cdn" ? "cdn" : trustedSource(pyodide, trust),
    packages: trustedSource(params.get("packages"), trust),
    pyrp: trustedSource(params.get("pyrp"), trust),
  };
}

export async function createBackend(kind: BackendKind, search: string = globalThis.location?.search ?? ""): Promise<Backend> {
  const params = new URLSearchParams(search);
  if (kind === "mock") {
    const speed = Number(params.get("speed") ?? "1");
    return new MockBackend({ speed: Number.isFinite(speed) ? speed : 1, bootMs: 400 });
  }
  // Loaded lazily so the mock build never touches the worker.
  const { PyodideBackend, PYODIDE_CDN } = await import("./pyodide/pyodideBackend");
  const sources = backendSources(search);
  return new PyodideBackend({
    indexURL: sources.pyodide === "cdn" ? PYODIDE_CDN : sources.pyodide,
    packageBaseUrl: sources.packages,
    pyrpBaseUrl: sources.pyrp,
  });
}
