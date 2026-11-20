/**
 * v1 protocol types of package C: files, base data, settings, licenses, log level and
 * snapshots (PROTOCOL.md "v1", sections App, Session, Data and files, Base data, Workflow).
 */

import type {
  ExportObject,
  FileResult,
  FormSchema,
  LogHistoryEntry,
  LogLevel,
  NDArray,
  SessionState,
} from "../protocol";

// ---------------------------------------------------------------------------
// app.licenses / app.settings
// ---------------------------------------------------------------------------

export interface LicenseRow {
  name: string;
  version: string;
  license: string;
}

export interface LicenseDetail extends LicenseRow {
  /** `null`: no license text shipped with the distribution. */
  text: string | null;
}

export type SettingsSectionId = "general" | "xp" | "ai" | "gui";

export interface SettingsSection {
  id: SettingsSectionId;
  title: string;
  form: FormSchema;
}

export interface SettingsResult {
  sections: SettingsSection[];
}

// ---------------------------------------------------------------------------
// export.save / data.load_*
// ---------------------------------------------------------------------------

export interface ExportSaveParams {
  object: ExportObject;
  format: string;
  /** Component keys, `object: "result"` only; default all full quantities. */
  quantities?: string[];
  structure_format?: "rtstruct" | "seg";
  basename?: string;
}

export type LoadKind = "auto" | "ct_new" | "ct_replace" | "structures" | "dose";

export interface LoadFileParams {
  filename: string;
  data: NDArray<Uint8Array>;
  kind?: LoadKind;
  /** Result name of a dose import (`kind: "dose"`); default: the file stem. */
  name?: string;
}

// ---------------------------------------------------------------------------
// base_data.*
// ---------------------------------------------------------------------------

export type BaseDataKind = "machines" | "rbe_tables" | "stopping_power_tables" | "phantoms";
export type BaseDataSource = "bundled" | "user" | "shared";

export interface BaseDataFolder {
  source: BaseDataSource;
  label: string;
  path: string;
}

export interface BaseDataRow {
  name: string;
  file: string;
  source: BaseDataSource;
  removable: boolean;
  radiation_mode?: string;
  model?: string;
  tissue_classes?: string;
  reference?: string;
  /** Why the file is unreadable (the row is listed in orange). */
  problem?: string | null;
  /** Absolute path of the file (the row tooltip). */
  path?: string;
}

export interface BaseDataList {
  kind: BaseDataKind;
  folders: BaseDataFolder[];
  rows: BaseDataRow[];
}

export interface BaseDataAddParams {
  kind: BaseDataKind;
  filename: string;
  data: NDArray<Uint8Array>;
  name?: string;
  radiation_mode?: string;
  overwrite?: boolean;
  dry_run?: boolean;
}

export interface BaseDataAddResult {
  /** `null` for a `dry_run`. */
  row: BaseDataRow | null;
  suggested_name: string;
  radiation_mode: string;
  clash: { file: string; source: BaseDataSource } | null;
}

export interface IoMethods {
  "app.licenses": { params: { name?: string }; result: LicenseRow[] | LicenseDetail };
  "app.settings": { params: Record<string, never>; result: SettingsResult };
  "app.settings_set": {
    params: { section: SettingsSectionId; values: Record<string, unknown> };
    result: SettingsResult;
  };
  "data.load_file": { params: LoadFileParams; result: SessionState };
  "data.load_dij": { params: { filename: string; data: NDArray<Uint8Array> }; result: SessionState };
  "export.save": { params: ExportSaveParams; result: FileResult };
  "base_data.list": { params: { kind: BaseDataKind }; result: BaseDataList };
  "base_data.add": { params: BaseDataAddParams; result: BaseDataAddResult };
  "base_data.remove": { params: { kind: BaseDataKind; file: string }; result: Record<string, never> };
  "session.log_level": { params: { level: Exclude<LogLevel, "critical"> }; result: { level: string } };
  "workflow.snapshot": { params: { tag?: string }; result: { state: SessionState; tag: string } };
}

export type { LogHistoryEntry };
