/**
 * The plan form's "…" dialogs for the dose engine, the biological model, the scenario model
 * and the sequencer (Qt `ConfigFormDialog` opened by `_on_engine_config` / `_on_bio_config` /
 * `_on_scenario_config` / `_on_sequencer_config`), loaded with `plan.form`.
 */
import { useEffect, useState } from "react";
import type { FormSchema } from "../../api/protocol";
import type { FormKind } from "../../api/protocol/plan";
import { errorText, planForm } from "../../store/planSlice";
import { ConfigFormDialog } from "../forms/ConfigForm";

export interface PlanConfigDialogProps {
  name: string;
  radiationMode: string;
  /** The form's current entries for this algorithm (Qt keeps them per algorithm). */
  values?: Record<string, unknown>;
  educational?: boolean;
  onAccept: (values: Record<string, unknown>) => void;
  onCancel: () => void;
}

export function usePlanForm(kind: FormKind, name: string, radiationMode: string, values?: Record<string, unknown>) {
  const [schema, setSchema] = useState<FormSchema | null>(null);
  const [error, setError] = useState<string | null>(null);
  const key = JSON.stringify([kind, name, radiationMode, values ?? null]);
  useEffect(() => {
    let cancelled = false;
    setSchema(null);
    setError(null);
    planForm(kind, name, radiationMode, values)
      .then((s) => !cancelled && setSchema(s))
      .catch((err: unknown) => !cancelled && setError(errorText(err)));
    return () => {
      cancelled = true;
    };
  }, [key]);
  return { schema, error };
}

function PlanConfigDialog(props: PlanConfigDialogProps & { kind: FormKind; testId: string }) {
  const { schema, error } = usePlanForm(props.kind, props.name, props.radiationMode, props.values);
  return (
    <ConfigFormDialog
      schema={schema}
      title={schema?.title ?? `Configure ${props.name}`}
      loadError={error}
      educational={props.educational}
      onAccept={props.onAccept}
      onCancel={props.onCancel}
      testId={props.testId}
    />
  );
}

export function EngineDialog(props: PlanConfigDialogProps) {
  return <PlanConfigDialog {...props} kind="engine" testId="engine-dialog" />;
}

export function BioModelDialog(props: PlanConfigDialogProps) {
  return <PlanConfigDialog {...props} kind="bio_model" testId="bio-dialog" />;
}

export function ScenarioDialog(props: PlanConfigDialogProps) {
  return <PlanConfigDialog {...props} kind="scenario" testId="scenario-dialog" />;
}

export function SequencerDialog(props: PlanConfigDialogProps) {
  return <PlanConfigDialog {...props} kind="sequencer" testId="sequencer-dialog" />;
}
