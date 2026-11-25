/** State of package C that does not belong to the session: log level, phantom list, base data version. */
import { create } from "zustand";
import type { BaseDataRow, BaseDataFolder } from "../api/protocol/io";

export type LogThreshold = "debug" | "info" | "warning" | "error";

export const LOG_LEVELS: ReadonlyArray<{ id: LogThreshold; label: string }> = [
  { id: "debug", label: "Debug" },
  { id: "info", label: "Info" },
  { id: "warning", label: "Warning" },
  { id: "error", label: "Error" },
];

export const LEVEL_RANK: Record<string, number> = { debug: 10, info: 20, warning: 30, error: 40, critical: 50 };

export interface PhantomListing {
  rows: BaseDataRow[];
  folders: BaseDataFolder[];
}

export interface IoStore {
  /** Minimum level shown by the log console and set on the backend's `pyRadPlan` logger. */
  logLevel: LogThreshold;
  /** Incremented after base data was added or removed (plan widgets refetch `plan.options`). */
  baseDataVersion: number;
  /** `base_data.list {kind: "phantoms"}`; `null` until fetched (or not supported). */
  phantoms: PhantomListing | null;
}

export const useIo = create<IoStore>(() => ({
  logLevel: "info",
  baseDataVersion: 0,
  phantoms: null,
}));

export function bumpBaseData(): void {
  useIo.setState((s) => ({ baseDataVersion: s.baseDataVersion + 1 }));
}
