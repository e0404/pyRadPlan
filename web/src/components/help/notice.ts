/**
 * Research-use disclaimer and privacy notice of the web version, acknowledged once per browser.
 *
 * The disclaimer texts come from `pyRadPlan/app/about.json` (embedded by `vite.config.ts`, the
 * Qt About dialog reads the same file). The privacy paragraphs describe this frontend: what it
 * loads from where and what it stores. The acknowledgement lives in `localStorage` under a key
 * derived from all texts, so changing any of them asks again. The key is deliberately shared by
 * all deployments on one origin (`/latest/` and `/develop/` on GitHub Pages): the same text needs
 * one acknowledgement, a different text has a different key.
 */
import { create } from "zustand";

export const ABOUT = __PYRADPLAN_ABOUT__;

export const PRIVACY_TITLE = "Your data and privacy";

/** The privacy paragraphs; `{origin}` is replaced by the page's origin. */
export const PRIVACY = [
  "Everything runs in this browser tab: pyRadPlan computes in WebAssembly (Pyodide) on this computer, and the CT, structures, plans and results you load or create stay in the tab's memory. Nothing is uploaded; data leaves the browser only when you save a file.",
  "The app downloads only its own files from this site ({origin}) and the Python packages it needs (numpy, scipy and others) from the jsDelivr CDN (cdn.jsdelivr.net). Like any web server, both see your IP address and which files are requested.",
  "There is no telemetry, analytics, tracking or cookies. Besides the browser's normal cache, the app stores only your acknowledgement of this notice (local storage) and, where the Stop button needs a service worker (GitHub Pages), that worker and a per-tab reload flag.",
];

/** One-line version for the splash. */
export const SHORT_NOTICE = `Research and education only, ${ABOUT.disclaimer_title.toLowerCase()}. Your data stays in this browser.`;

export function privacyParagraphs(origin: string = globalThis.location?.origin ?? "this site"): string[] {
  return PRIVACY.map((p) => p.replace("{origin}", origin));
}

/** FNV-1a (32 bit) of the texts, as 8 hex digits. */
export function textHash(text: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

const NOTICE_TEXT = [ABOUT.disclaimer_title, ABOUT.disclaimer, ABOUT.license_note, PRIVACY_TITLE, ...PRIVACY].join("\n");

/** `localStorage` key of the acknowledgement of the current texts. */
export const NOTICE_KEY = `pyradplan.notice-acknowledged.${textHash(NOTICE_TEXT)}`;

function storage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

function readAcknowledged(): boolean {
  try {
    return Boolean(storage()?.getItem(NOTICE_KEY));
  } catch {
    return false;
  }
}

export const useNotice = create<{ acknowledged: boolean }>(() => ({ acknowledged: readAcknowledged() }));

/** Record the acknowledgement; without storage it holds for this page only. */
export function acknowledgeNotice(): void {
  try {
    storage()?.setItem(NOTICE_KEY, new Date().toISOString());
  } catch {
    // Storage blocked (private window, site data disabled): ask again on the next visit.
  }
  useNotice.setState({ acknowledged: true });
}

/** Resolves once the notice is acknowledged (immediately on later visits). */
export function noticeAcknowledged(): Promise<void> {
  if (useNotice.getState().acknowledged) return Promise.resolve();
  return new Promise((resolve) => {
    const unsubscribe = useNotice.subscribe((s) => {
      if (!s.acknowledged) return;
      unsubscribe();
      resolve();
    });
  });
}

/** Channel, version and commit the build was made from (`VITE_PYRADPLAN_*`), or `null`. */
export function buildLabel(env: Record<string, unknown> = import.meta.env): string | null {
  const text = (key: string) => (typeof env[key] === "string" && env[key] ? (env[key] as string) : "");
  const channel = text("VITE_PYRADPLAN_CHANNEL");
  const version = text("VITE_PYRADPLAN_VERSION");
  const commit = text("VITE_PYRADPLAN_COMMIT");
  const parts = [channel && `${channel} channel`, version, commit && `commit ${commit}`].filter(Boolean);
  return parts.length ? parts.join(" · ") : null;
}
