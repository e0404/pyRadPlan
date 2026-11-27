/**
 * v1 protocol types of package A: plan detail and options, configuration forms, tissues,
 * objectives and the optimization status (PROTOCOL.md "v1", sections Plan / Structures, tissues
 * and objectives).
 */

import type {
  Choice,
  FieldDescriptor,
  FormSchema,
  Issue,
  OptimizationSummary,
  PlanSummary,
  SessionState,
  StructureDetail,
  TissueDraft,
  Vec3,
} from "../protocol";

export type DoseConvention = "per_fraction" | "total";

export interface NamedValues {
  name: string | null;
  values: Record<string, unknown>;
}

export interface PlanDetail extends PlanSummary {
  dose_convention: DoseConvention;
  /** Read-only (not shown in Qt, carried over). */
  prescribed_dose: number;
  /** "plan": the plan carries loaded base data ("(from plan)"). */
  machine_source: "name" | "plan";
  /** Target centre of mass when automatic. */
  resolved_iso_center: Vec3 | null;
  engine_config: NamedValues;
  bio: { model: string; option: string | null; values: Record<string, unknown> };
  scenario: { model: string; values: Record<string, unknown> };
  problem: { name: string; values: Record<string, unknown> };
  solver: NamedValues;
}

/** Value of the "(from plan)" machine entry (v1 amendment 11). */
export const PLAN_MACHINE = "__plan__";

/** Bio-model option ids are opaque strings (v1 amendment 11). */
export interface BioModelChoice extends Choice {
  model: string;
  parameters: Record<string, unknown>;
  /** Whether the model's "…" form has fields (v1 amendment 16); absent from older backends. */
  configurable?: boolean;
}

/** The unapplied plan form, as `plan.tissue_check` takes it (v1 amendment 16). */
export interface PlanDraftContext {
  radiation_mode?: string;
  machine?: string;
  engine?: string | null;
  bio_model?: string | { model: string; option: string | null; values?: Record<string, unknown> };
}

export interface SolverChoice extends Choice {
  gpu_compatible: boolean;
}

export interface BeamPreset {
  name: string;
  label: string;
  gantry_angles: number[];
  couch_angles: number[];
  description: string;
  typical: boolean;
}

/** The form's biological model selection before a re-listing (`plan.options {current_bio}`, v1 amendment 22). */
export interface CurrentBio {
  /** The radiation mode the selection was made in. */
  radiation_mode: string;
  model: string;
  /** Option id; null for the plan's model kept as the unsupported entry. */
  option: string | null;
  /** Model parameters besides the base-data selection. */
  values?: Record<string, unknown>;
  /** Label of the selected entry, for the replacement note. */
  label?: string;
}

/** The entry the plan form selects (v1 amendment 22); the rules are the backend's. */
export interface BioSelection {
  /** Option id; null while `unsupported` is selected (or without any entry). */
  option: string | null;
  model: string;
  /** The selection was replaced: take over the selected option's base-data parameters. */
  replaced: boolean;
  /** The plan's model the context cannot run, kept as a last, disabled entry. */
  unsupported: { model: string; label: string; description: string } | null;
  /** Status note (and its tooltip) when a selection the new context cannot run was replaced. */
  note: { text: string; reason: string } | null;
}

export interface PlanOptions {
  /** v1 amendment 11: the radiation mode the options describe. */
  radiation_mode?: string;
  radiation_modes: Choice[];
  machines: Choice[];
  engines: Choice[];
  bio_models: BioModelChoice[];
  default_bio_model: string;
  /** v1 amendment 22; absent from older backends. */
  bio_selection?: BioSelection;
  scenario_models: Choice[];
  problems: Choice[];
  solvers: SolverChoice[];
  default_solver: string;
  beam_presets: BeamPreset[];
  /** The fixed plan fields: bounds, defaults, advanced, editable. */
  fields: FieldDescriptor[];
}

export type FormKind = "engine" | "bio_model" | "scenario" | "problem" | "solver";

/**
 * Suggested values per field of an algorithm form (`plan.form_choices`, v1 amendment 28): what
 * the Qt dialogs list in their editable combo boxes. Free text stays allowed.
 */
export interface FormChoices {
  /** The form they belong to, as `FormSchema.id`. */
  id: string;
  choices: Record<string, Choice[]>;
}

export type BioModelSet = string | { model: string; option?: string | null; values?: Record<string, unknown> };

/** `plan.set` v1: partial v0 summary plus the v1 fields. */
export type PlanSetParams = Partial<Omit<PlanSummary, "bio_model">> & {
  dose_convention?: DoseConvention;
  bio_model?: BioModelSet | null;
  engine_config?: { values: Record<string, unknown> };
  scenarios?: { model: string; values: Record<string, unknown> };
  problem?: { name: string; values: Record<string, unknown> };
  solver?: string | { name: string; values: Record<string, unknown> };
};

export interface TissueCheckResult {
  /** The ⚠ / ℹ marker. */
  marker: "error" | "warning" | "";
  issues: Issue[];
}

export type TissueModelLabel = TissueDraft["model"];

export interface TissuePick {
  id: string;
  kind: "base_data" | "preset";
  label: string;
  group: string;
  exact: boolean;
  model: TissueModelLabel;
  values: Record<string, number | string | null>;
  /** Identity of the tissue applying the pick gives (amendment 25); null for a base data class without one. */
  cell_line?: string | null;
  cell_line_source?: string | null;
  description: string;
  /** v1 amendment 13: listed first (Qt: presets of the base data's provenance). */
  preferred?: boolean;
}

export interface TissueSchema {
  models: Array<{ label: TissueModelLabel; fields: FieldDescriptor[] }>;
  reference_radiations: Choice[];
  picks: TissuePick[];
  option_label: string;
}

export type TissueSource = "explicit" | "published" | "estimated" | "base_data" | "default" | "derived";

export interface TissueComparison {
  closest_class: string | null;
  matched: boolean;
  provenance_fix: boolean;
  parameters: Array<{ name: string; declared: unknown; value: unknown; matches: boolean }>;
}

export interface TissueEvalRow {
  structure: string;
  tissue: TissueDraft;
  issues: Issue[];
  comparison: TissueComparison | null;
  /** Id of the `tissue.schema` pick the tissue corresponds to (the class it matches exactly or the preset it was applied from), null if none. */
  pick?: string | null;
  /** Not sent since v1 amendment 13 (the sources are in `tissue.sources`). */
  sources?: Record<string, TissueSource>;
}

export type TissueOp = "none" | "derive" | "match" | "relabel" | "apply_pick";

export interface TissueEvaluateParams {
  rows: Array<{ structure: string; tissue: TissueDraft }>;
  op?: TissueOp;
  pick?: string;
  targets?: string[];
  edited?: { structure: string; field: string };
  /** Evaluate against the unapplied plan form (v1 amendment 16). */
  draft?: PlanDraftContext;
}

export interface TissueEvaluateResult {
  rows: TissueEvalRow[];
  notes: string[];
  can_match: boolean;
  can_relabel: string[];
  can_derive: boolean;
}

export interface ObjectiveType {
  name: string;
  scenario_aware: boolean;
  parameters: FieldDescriptor[];
  robustness: Choice[];
}

export interface ObjectivesCatalog {
  objectives: ObjectiveType[];
  quantities: Choice[];
  robustness: Choice[];
  /** `{TARGET: "Squared Deviation", "*": "Squared Overdosing"}`. */
  default_by_type: Record<string, string>;
  reference_images: Choice[];
}

export interface ObjectiveChanges {
  priority?: number;
  quantity?: string;
  robustness?: string;
  parameters?: Record<string, unknown>;
}

export interface StructureResult {
  state: SessionState;
  structure: StructureDetail;
}

/** Optimizer metrics in `progress.data` during `workflow.optimize`. */
export interface OptimizerMetrics {
  iteration?: number;
  objective?: number;
  constraint_violation?: number;
  dual_inf?: number;
  step?: number;
  kkt_error?: number;
  mu?: number;
}

export interface PlanMethods {
  "plan.get": { params: { detail?: boolean }; result: PlanSummary | PlanDetail | null };
  "plan.options": {
    params: { radiation_mode?: string; machine?: string; engine?: string; current_bio?: CurrentBio };
    result: PlanOptions;
  };
  "plan.form": {
    params: { kind: FormKind; name: string; radiation_mode?: string; values?: Record<string, unknown> };
    result: FormSchema;
  };
  "plan.form_choices": {
    params: { kind: FormKind; name: string; radiation_mode?: string };
    result: FormChoices;
  };
  "plan.set": { params: PlanSetParams; result: SessionState };
  "plan.tissue_check": {
    params: { draft?: { radiation_mode: string; machine: string; engine: string | null; bio_model: BioModelSet | null } };
    result: TissueCheckResult;
  };
  "tissue.schema": {
    params: { bio_model?: string; machine?: string; engine?: string; draft?: PlanDraftContext };
    result: TissueSchema;
  };
  "tissue.evaluate": { params: TissueEvaluateParams; result: TissueEvaluateResult };
  "tissue.apply": {
    params: { rows: Array<{ structure: number; tissue: TissueDraft }>; apply_anyway?: boolean };
    result: { state: SessionState };
  };
  "objectives.catalog": { params: Record<string, never>; result: ObjectivesCatalog };
  "objectives.add": { params: { structure: number }; result: StructureResult };
  "objectives.update": {
    params: { structure: number; index: number; changes: ObjectiveChanges };
    result: StructureResult;
  };
  "objectives.change_type": {
    params: { structure: number; index: number; name: string };
    result: StructureResult;
  };
  "objectives.remove": { params: { structure: number; index: number }; result: StructureResult };
  "objectives.set_quantity_all": { params: { quantity: string }; result: { state: SessionState } };
  "workflow.optimize": {
    params: { solver?: string; max_iter?: number };
    result: SessionState & { optimization?: OptimizationSummary };
  };
  "session.pause": { params: Record<string, never>; result: Record<string, never> };
  "session.resume": { params: Record<string, never>; result: Record<string, never> };
}
