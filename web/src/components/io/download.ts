/** Browser download of an `export.save` result. */
import type { FileResult } from "../../api";

export interface DownloadEnv {
  createObjectURL(blob: Blob): string;
  revokeObjectURL(url: string): void;
  createAnchor(): { href: string; download: string; style: { display: string }; click(): void; remove(): void };
  append(node: unknown): void;
  defer(fn: () => void, ms: number): void;
}

function browserEnv(): DownloadEnv {
  return {
    createObjectURL: (blob) => URL.createObjectURL(blob),
    revokeObjectURL: (url) => URL.revokeObjectURL(url),
    createAnchor: () => document.createElement("a"),
    append: (node) => document.body.appendChild(node as Node),
    defer: (fn, ms) => void setTimeout(fn, ms),
  };
}

/** A file name safe to hand to the browser: no directories, no control characters. */
export function safeFilename(name: string): string {
  const base = name.split(/[\\/]/).pop() ?? "";
  // eslint-disable-next-line no-control-regex
  const cleaned = base.replace(/[\u0000-\u001f<>:"|?*]/g, "_").trim();
  return cleaned || "download";
}

export interface DownloadedFile {
  filename: string;
  mime: string;
  bytes: number;
}

/** Offer `result` as a download (anchor with the `download` attribute on an object URL). */
export function downloadFile(result: FileResult, env: DownloadEnv = browserEnv()): DownloadedFile {
  const filename = safeFilename(result.filename);
  const bytes = result.data.data;
  const blob = new Blob([bytes as BlobPart], { type: result.mime || "application/octet-stream" });
  const url = env.createObjectURL(blob);
  const anchor = env.createAnchor();
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  env.append(anchor);
  anchor.click();
  anchor.remove();
  env.defer(() => env.revokeObjectURL(url), 30_000);
  return { filename, mime: blob.type, bytes: bytes.length };
}

/** "plan.mat" or "result.zip (3 files)" for the confirmation after a save. */
export function describeDownload(result: FileResult, saved: DownloadedFile): string {
  const members = result.files?.length;
  return members && members > 0 ? `${saved.filename} (${members} file${members === 1 ? "" : "s"})` : saved.filename;
}
