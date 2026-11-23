import type { Backend } from "./backend";

export type { Backend, BootStatus } from "./backend";
export { BackendError } from "./backend";
export * from "./protocol";

export type BackendKind = "mock" | "pyodide";

/**
 * Whether this build contains the mock backend: the development server, or a build made with
 * `VITE_ENABLE_MOCK=1`. A deployment never offers it (and its bundle does not contain it), so a
 * crafted `?backend=mock` link cannot show invented plans and doses as results.
 */
export const MOCK_AVAILABLE: boolean = Boolean(import.meta.env.DEV) || import.meta.env.VITE_ENABLE_MOCK === "1";

/**
 * Backend selected by the `backend` URL parameter (`?backend=mock|pyodide`).
 * Without the parameter the Pyodide backend is used; `VITE_BACKEND` overrides the default.
 * `mock` only where the build has it ({@link MOCK_AVAILABLE}).
 */
export function backendKindFromUrl(search: string = globalThis.location?.search ?? "", mockAvailable: boolean = MOCK_AVAILABLE): BackendKind {
  const param = new URLSearchParams(search).get("backend");
  const fallback = (import.meta.env.VITE_BACKEND as string | undefined) ?? "pyodide";
  const kind = (param ?? fallback).toLowerCase();
  if (kind !== "mock") return "pyodide";
  if (mockAvailable) return "mock";
  console.warn("Ignoring backend=mock: this build has no mock backend");
  return "pyodide";
}

/** Where URL parameters may point the backend at code (see {@link trustedSource}). */
export interface SourceTrust {
  /** Development server: every value is honoured. */
  dev: boolean;
  /** Origin of the page; other origins are refused outside development. */
  origin: string;
  /** URL relative values resolve against. */
  base: string;
  /** Absolute URL of the app's own directory (`BASE_URL`); only URLs below it are honoured. */
  scope: string;
}

function pageTrust(): SourceTrust {
  const location = globalThis.location;
  const base = location?.href ?? "http://localhost/";
  return {
    dev: Boolean(import.meta.env.DEV),
    origin: location?.origin ?? "",
    base,
    scope: new URL(import.meta.env.BASE_URL, base).href,
  };
}

/**
 * The value of a URL parameter naming where the backend loads code from (`pyodide`,
 * `packages`, `pyrp`), or `undefined` for the default. Outside development only URLs below the
 * app's own directory are honoured: on a deployment, a crafted link must not make the app run
 * code from elsewhere, which includes other sites on a shared origin (GitHub Pages serves every
 * project of an organization from one) and the other channel of the same site.
 */
export function trustedSource(value: string | null, trust: SourceTrust = pageTrust()): string | undefined {
  if (value === null || value === "") return undefined;
  if (trust.dev) return value;
  try {
    const resolved = new URL(value, trust.base);
    if (resolved.origin === trust.origin && resolved.href.startsWith(trust.scope)) return value;
  } catch {
    // Not a URL: refused like a foreign one.
  }
  console.warn(`Ignoring the URL parameter value "${value}": not inside this app's directory`);
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
    if (import.meta.env.DEV || import.meta.env.VITE_ENABLE_MOCK === "1") {
      const { MockBackend } = await import("./mock/mockBackend");
      const speed = Number(params.get("speed") ?? "1");
      return new MockBackend({ speed: Number.isFinite(speed) ? speed : 1, bootMs: 400 });
    }
    throw new Error("This build has no mock backend");
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
