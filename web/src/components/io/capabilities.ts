/** Helpers over `app.info.capabilities` (PROTOCOL.md "capabilities"). */
import type { AppInfo, ExportFormat, ExportObject, FeatureId, FeatureStatus, ImportFormat } from "../../api";

/** Feature status; a backend that reports no capabilities (protocol v0) is treated as capable. */
export function feature(info: AppInfo | null | undefined, id: FeatureId): FeatureStatus {
  return info?.capabilities?.features[id] ?? { available: true, reason: null };
}

export function exportFormats(info: AppInfo | null | undefined, object: ExportObject): ExportFormat[] {
  return (info?.capabilities?.formats.export ?? []).filter((f) => f.object === object);
}

/** Why none of an object's formats can be written, or `null` when at least one can. */
export function exportBlockedReason(info: AppInfo | null | undefined, object: ExportObject): string | null {
  if (!info) return null;
  if (!info.capabilities) return "This backend does not export files.";
  const formats = exportFormats(info, object);
  if (formats.some((f) => f.available)) return null;
  return formats[0]?.reason ?? "This object cannot be exported by this backend.";
}

/** Import formats of the backend; a v0 backend loads `.mat` files. */
export function importFormats(info: AppInfo | null | undefined): ImportFormat[] {
  const formats = info?.capabilities?.formats.import;
  if (formats) return formats;
  return [{ format: "mat", label: "matRad MATLAB", extensions: [".mat"], objects: [], available: true, reason: null }];
}

export const IMAGE_EXTENSIONS = [".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"] as const;

export function isImageFile(name: string): boolean {
  const lower = name.toLowerCase();
  return IMAGE_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

/** `accept` attribute of the Load File picker: every extension an available format reads. */
export function loadAccept(info: AppInfo | null | undefined): string {
  const extensions = importFormats(info)
    .filter((f) => f.available)
    .flatMap((f) => f.extensions);
  return [...new Set(extensions)].join(",");
}

/** The structure-set DICOM variant of an export entry (`structure_format` of `export.save`). */
export function structureFormatOf(format: ExportFormat): "rtstruct" | "seg" | undefined {
  const explicit = (format as ExportFormat & { structure_format?: string }).structure_format;
  if (explicit === "seg" || explicit === "rtstruct") return explicit;
  if (format.format !== "dcm" || format.object !== "cst") return undefined;
  return /\bseg\b/i.test(format.label) ? "seg" : "rtstruct";
}
