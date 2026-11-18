/** Structure edits of package B (`cst.update_structure`): colour pick, Edit VOI dialog. */
import { BackendError, type Issue } from "../api";
import type { StructureChanges } from "../api/protocol/viewer";
import { log, useApp } from "./appStore";
import { syncViews } from "./revisions";

export interface UpdateOutcome {
  ok: boolean;
  /** Tissue errors the user may override with "Apply anyway". */
  issues?: Issue[];
  error?: string;
}

/**
 * Write structure changes. Tissue errors (`PreconditionError` `tissue_errors`) are returned
 * so the caller can ask "Apply anyway / Keep editing" and retry with `applyAnyway`.
 */
export async function updateStructure(name: string, changes: StructureChanges, applyAnyway = false): Promise<UpdateOutcome> {
  const { backend, session } = useApp.getState();
  if (!backend || !session) return { ok: false, error: "Backend not connected" };
  try {
    const { state, structure } = await backend.request(
      "cst.update_structure",
      { name, changes, ...(applyAnyway ? { apply_anyway: true } : {}) },
      session,
    );
    useApp.setState((s) => ({ state, structureDetails: { ...s.structureDetails, [structure.name]: structure } }));
    await syncViews();
    return { ok: true };
  } catch (err) {
    if (err instanceof BackendError && err.code === "tissue_errors") {
      const details = err.details;
      return { ok: false, issues: Array.isArray(details) ? (details as Issue[]) : [], error: err.message };
    }
    const message = err instanceof Error ? err.message : String(err);
    log("error", `Structure ${name}: ${message}`);
    return { ok: false, error: message };
  }
}
