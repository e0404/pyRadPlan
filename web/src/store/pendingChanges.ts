/**
 * Unapplied editor changes offered before a step that uses the plan (Qt
 * `WorkflowWidget.add_pending_changes_source` / `_resolve_pending_changes`).
 *
 * An editor (the plan form) registers itself with `registerPendingSource`; the save and workflow
 * actions call `resolvePendingChanges(step)` first.
 */
import { messageBox } from "../components/io/messageBox";

export interface PendingSource {
  /** The editor as named to the user, e.g. "plan". */
  name: string;
  hasPending: () => boolean;
  /** Apply the changes; `false` when that failed (the editor reports why). */
  apply: () => boolean | Promise<boolean>;
}

const sources = new Set<PendingSource>();

/** Register an editor; returns the unregister function. */
export function registerPendingSource(source: PendingSource): () => void {
  sources.add(source);
  return () => {
    sources.delete(source);
  };
}

export function pendingSourceNames(): string[] {
  return [...sources].filter((s) => s.hasPending()).map((s) => s.name);
}

/**
 * Settle unapplied editor changes before `step`; `false` if the step must not run.
 * `outdatesInput` marks steps working from the dose influence: applying a plan change outdates it,
 * so such a step stops after applying and asks for a recalculation.
 */
export async function resolvePendingChanges(step: string, outdatesInput = false): Promise<boolean> {
  const pending = [...sources].filter((s) => s.hasPending());
  if (!pending.length) return true;
  const choice = await messageBox({
    title: step,
    icon: "question",
    text: `The ${pending.map((p) => p.name).join(" and ")} has changes that are not applied yet.`,
    informativeText: outdatesInput
      ? `Applying them outdates the dose influence, so it has to be recalculated before "${step}" can run.`
      : undefined,
    buttons: [
      { id: "apply", label: outdatesInput ? "Apply changes" : "Apply and continue", primary: true },
      { id: "continue", label: "Continue without applying" },
      { id: "cancel", label: "Cancel", reject: true },
    ],
    testId: "pending-changes-dialog",
  });
  if (choice === "continue") return true;
  if (choice !== "apply") return false;
  for (const source of pending) {
    if (!(await source.apply())) return false;
  }
  return !outdatesInput;
}
