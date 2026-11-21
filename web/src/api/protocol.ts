/**
 * Types of the pyRadPlan application protocol (see web/PROTOCOL.md).
 *
 * "Wire" types describe what crosses the transport (JSON + a list of binary buffers);
 * the decoded types replace every `{$array}` reference with an {@link NDArray}.
 */

import type { PlanMethods } from "./protocol/plan";
import type { IoMethods } from "./protocol/io";
import type { ViewerMethods } from "./protocol/viewer";

export type DType =
  | "float32"
  | "float64"
  | "int8"
  | "uint8"
  | "int16"
  | "uint16"
  | "int32"
  | "uint32"
  | "bool";

export type TypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array;

/** Array reference inside a wire message; `$array` indexes the message's `buffers`. */
export interface ArrayRef {
  $array: number;
  dtype: DType;
  shape: number[];
}

/** A decoded n-d array (C order, axis order as sent, i.e. `(z, y, x)` for volumes). */
export interface NDArray<T extends TypedArray = TypedArray> {
  readonly dtype: DType;
  readonly shape: readonly number[];
  readonly data: T;
}

export type Vec3 = [number, number, number];

export type StageStatus = "missing" | "done" | "stale";

export interface StructureInfo {
  name: string;
  type: string;
  color: Vec3;
  visible: boolean;
  index: number;
  /** v1 additions. */
  overlap_priority?: number;
  num_objectives?: number;
  num_voxels?: number;
}

export interface PlanSummary {
  radiation_mode: string;
  machine: string;
  num_of_fractions: number;
  gantry_angles: number[];
  couch_angles: number[];
  bixel_width: number;
  /** `null` = automatic (target centre of mass), one point, or one point per beam. */
  iso_center: Vec3 | Vec3[] | null;
  dose_grid_resolution: Vec3 | null;
  engine: string | null;
  bio_model?: string | null;
}

/** The single isocenter of a plan (the first one for per-beam isocenters), or null for "auto". */
export function isoCenterPoint(iso: PlanSummary["iso_center"] | undefined): Vec3 | null {
  if (!iso || !iso.length) return null;
  return Array.isArray(iso[0]) ? (iso[0] as Vec3) : (iso as Vec3);
}

export interface CtSummary {
  shape: Vec3;
  spacing: Vec3;
  origin: Vec3;
  direction?: number[];
  num_scenarios: number;
}

export type PipelineKey = "ct" | "cst" | "pln" | "stf" | "dij" | "result";
export type StageKey = "data" | "plan" | "stf" | "dij" | "optimization";
export type ActionKey = "generate_stf" | "calc_dose_influence" | "optimize" | "recalc";

export interface SessionState {
  session: string;
  has: Record<PipelineKey, boolean>;
  stages: Record<StageKey, StageStatus>;
  actions: Record<ActionKey, boolean> & Record<string, boolean>;
  ct: CtSummary | null;
  plan: PlanSummary | null;
  quantities: string[];
  structures: StructureInfo[];
  status: string;
  /** Label of the operation running in the session, if any. */
  busy?: string | null;
  /** v1 (PROTOCOL.md "SessionState v1 additions"); absent from a v0 backend. */
  revisions?: Revisions;
  results?: ResultsInfo;
  options?: SessionOptions;
  /** Whether `workflow.sequence` can run on the active result, else why not (amendment 27). */
  sequencing?: { available: boolean; reason: string | null };
}

export interface AppInfo {
  version: string;
  python: string;
  backend: string;
  phantoms: string[];
  machines: Record<string, string[]>;
  solvers: string[];
  engines: Record<string, unknown>;
  platform?: string;
  /** Registered protocol methods. */
  methods?: string[];
  /** v1 extensions of `app.info`. */
  runtime?: "pyodide" | "native";
  capabilities?: Capabilities;
  quantities?: QuantityInfo[];
  default_profile?: Profile;
}

export interface Grid {
  /** `(nz, ny, nx)`. */
  shape: Vec3;
  /** `(sx, sy, sz)` in mm. */
  spacing: Vec3;
  /** `(x, y, z)` of voxel `[0, 0, 0]` in mm, LPS. */
  origin: Vec3;
  /** 3x3 direction cosines, row-major. */
  direction?: number[];
}

export interface Volume {
  name: string;
  unit: string;
  data: NDArray;
  grid: Grid;
  /** Entries are `null` when the volume has no finite values. */
  range: [number | null, number | null];
}

export interface StructureMask {
  index: number;
  name: string;
  color: Vec3;
  data: NDArray<Uint8Array>;
}

export interface StructureMasks {
  grid: Grid;
  masks: StructureMask[];
}

export interface DvhCurve {
  index: number;
  name: string;
  color: Vec3;
  dose: NDArray;
  volume: NDArray;
  stats?: { mean: number | null; min: number | null; max: number | null };
}

export interface DvhResult {
  quantity: string;
  unit: string;
  curves: DvhCurve[];
}

export interface OptimizationSummary {
  solver?: string;
  num_iter?: number;
  objective?: number;
}

/** v0 request params and decoded results per method. */
export interface V0Methods {
  "app.info": { params: Record<string, never>; result: AppInfo };
  "session.create": { params: Record<string, never>; result: { session: string } };
  "session.close": { params: Record<string, never>; result: Record<string, never> };
  "session.state": { params: Record<string, never>; result: SessionState };
  "data.load_phantom": { params: { name: string }; result: SessionState };
  "data.load_file": { params: { filename: string; data: NDArray<Uint8Array> }; result: SessionState };
  "plan.get": { params: Record<string, never>; result: PlanSummary | null };
  "plan.set": { params: Partial<PlanSummary>; result: SessionState };
  "workflow.generate_stf": { params: Record<string, never>; result: SessionState };
  "workflow.calc_dose_influence": { params: { physical_only?: boolean }; result: SessionState };
  "workflow.optimize": {
    params: { solver?: string; max_iter?: number };
    result: SessionState & { optimization?: OptimizationSummary };
  };
  "workflow.recalc": { params: { physical_only?: boolean }; result: SessionState };
  "workflow.run_all": {
    params: { solver?: string; max_iter?: number; physical_only?: boolean };
    result: SessionState;
  };
  "view.volume": { params: { name: string; phase?: number }; result: Volume };
  "view.structure_masks": { params: { names?: number[] }; result: StructureMasks };
  "analysis.dvh": { params: { quantity?: string; names?: number[]; num_points?: number }; result: DvhResult };
}

// ---------------------------------------------------------------------------
// Shared v1 types (PROTOCOL.md "v1"), used by every frontend package.
// Feature-specific v1 types live in `./protocol/<feature>.ts`.
// ---------------------------------------------------------------------------

export type Profile = "native" | "web";

export interface Choice {
  /** What is sent back. */
  value: string | number | boolean | null;
  /** Qt display text. */
  label: string;
  /** Python identifier (popup hint). */
  identifier?: string;
  /** Default true. */
  enabled?: boolean;
  /** Why disabled. */
  reason?: string;
  /** Section header in the list. */
  group?: string;
  /** Tooltip. */
  description?: string;
}

export type FieldType =
  | "bool"
  | "int"
  | "float"
  | "str"
  | "choice"
  | "number_list"
  | "json"
  | "group"
  | "color"
  | "image_ref";

export interface FieldDescriptor {
  /** Key in `values`; dot path inside groups ("setup.sd"). */
  name: string;
  label: string;
  type: FieldType;
  /** JSON value of the model default. */
  default?: unknown;
  /** Optional[...]: null allowed ("None" / indeterminate checkbox). */
  nullable?: boolean;
  /** Default true. */
  editable?: boolean;
  /** Why not editable (locked, runtime, Qt shows it read-only). */
  reason?: string;
  /** Locked in the educational UI mode. */
  advanced?: boolean;
  /** Pydantic Field description (tooltip). */
  description?: string;
  /** Display unit, e.g. "mm", "Gy", "%". */
  unit?: string;
  /** Unit is that of the objective's quantity (objective parameters). */
  unit_from?: "quantity";
  min?: number;
  max?: number;
  exclusive_min?: number;
  exclusive_max?: number;
  step?: number;
  decimals?: number;
  /** Type "choice" and "image_ref". */
  choices?: Choice[];
  /** Choice that also accepts free text (Union[Literal, str]). */
  editable_choice?: boolean;
  /** Type "number_list" (`length` for tuples). */
  item_type?: "int" | "float";
  length?: number;
  /** Type "group": nested model/dataclass, shown collapsed. */
  fields?: FieldDescriptor[];
}

export interface FormSchema {
  /** E.g. "engine:SVDPB", "solver:scipy", "scenario:wcScen". */
  id: string;
  /** Qt dialog title, e.g. "Configure SciPy minimize". */
  title: string;
  fields: FieldDescriptor[];
  /** Current values; groups as nested override dicts. */
  values: Record<string, unknown>;
  /** Names whose values are set (reported back even if unchanged). */
  explicit: string[];
  /** E.g. "Solver 'ipopt' is unavailable: ...". */
  notes?: string[];
}

/** Every save returns one file; folders and several files are zipped. */
export interface FileResult {
  /** Suggested download name. */
  filename: string;
  mime: string;
  data: NDArray<Uint8Array>;
  /** Archive members when mime is application/zip. */
  files?: string[];
}

/** `pyRadPlan.biology.models.TissueIssue`. */
export interface Issue {
  structure: string | null;
  kind: string;
  severity: "error" | "warning";
  message: string;
  closest_class?: string | null;
}

/** `error.details` entry of a `ValidationError`. */
export interface ValidationDetail {
  field: string;
  message: string;
}

export interface Revisions {
  /** +1 per write of the key. */
  workspace: Record<PipelineKey, number>;
  /** Per volume key ("ct" and every component key): +1 when its source object changes. */
  volumes: Record<string, number>;
  /** +1 when a mask, the structure name list or the CT changes. */
  masks: number;
  /** +1 when pln, stf or cst (automatic isocenter) changes. */
  beams: number;
  /** Process-wide, +1 per base_data.add / remove. */
  base_data: number;
}

export type ComponentKind = "full" | "beam" | "scenario" | "other";

export interface ResultComponent {
  key: string;
  label: string;
  kind: ComponentKind;
}

export interface ResultGroup {
  base: string;
  /** Name of the result the group belongs to; "" = the active result. */
  tag: string;
  /** Selector label: RTQuantity.name (+ " (<result name>)"). */
  label: string;
  /** Colorbar / save-dialog label and unit (QUANTITY_META). */
  meta_label: string;
  unit: string;
  components: ResultComponent[];
  /** Scenario labels of the group's own result, by flat scenario index. */
  scenario_labels: string[];
}

export interface ResultsInfo {
  groups: ResultGroup[];
  /** Scenario labels of the active result (compatibility; use the group's own list). */
  scenario_labels: string[];
  /** rbe_x_dose > physical_dose > first (Qt default_quantity_key). */
  default: string | null;
  /** Names of all results, in workspace order. */
  names: string[];
  /** Name of the active result (default "current"); it need not exist yet. */
  active: string;
  /** Names of the results other than the active one (snapshots, imported doses). */
  tags: string[];
  /** Suggested snapshot name, "<radiation_mode>_<n>fx". */
  default_tag: string;
}

export interface SessionOptions {
  profile: Profile;
  default_plan: Partial<PlanSummary>;
  locked_fields: string[];
}

/** v1 `actions` entries in addition to the v0 ones. */
export type ActionKeyV1 =
  | "snapshot"
  | "load_dij"
  | "save_pln"
  | "save_dij"
  | "save_cst"
  | "save_result"
  | "save_workspace";

export type FeatureId =
  | "image_file_io"
  | "dicom_import"
  | "dicom_export"
  | "pickle_load"
  | "ai"
  | "ipopt"
  | "gpu"
  | "numba"
  | "monte_carlo"
  | "pause"
  | "open_folder"
  | "host_paths"
  | "base_data_persistence";

export interface FeatureStatus {
  available: boolean;
  reason: string | null;
}

export interface ImportFormat {
  format: string;
  label: string;
  extensions: string[];
  objects: string[];
  available: boolean;
  reason: string | null;
  note?: string;
}

export type ExportObject = "pln" | "dij" | "cst" | "result" | "workspace";

export interface ExportFormat {
  object: ExportObject;
  format: string;
  label: string;
  extension: string;
  mime: string;
  available: boolean;
  reason: string | null;
  keeps_objectives?: boolean;
  archive: boolean;
  note?: string;
}

export interface Capabilities {
  features: Partial<Record<FeatureId, FeatureStatus>> & Record<string, FeatureStatus>;
  formats: { import: ImportFormat[]; export: ExportFormat[] };
}

export interface QuantityInfo {
  identifier: string;
  name: string;
  meta_label: string;
  unit: string;
}

/** Feature status from `app.info.capabilities` (available when the backend does not report it). */
export function featureStatus(info: AppInfo | null | undefined, id: FeatureId): FeatureStatus {
  return info?.capabilities?.features[id] ?? { available: true, reason: null };
}

/** `pyRadPlan.biology` tissue parameters of one structure. */
export interface TissueDraft {
  model: "Generic" | "LEM" | "MKM" | "mMKM";
  /** NaN-default parameters as null. */
  values: Record<string, number | string | null>;
  /** reference_sources. */
  sources?: Record<string, string>;
  /** Cell line the parameters describe; the backend sends it (null if none), a draft without it keeps the structure's. */
  cell_line?: string | null;
  /** Where the parameters of the cell line come from (a preset's citation). */
  cell_line_source?: string | null;
}

export interface ObjectiveData {
  index: number;
  name: string;
  priority: number;
  quantity: string;
  robustness: string;
  /** Image references: `{"$result": "<component key>"}` or `{"$custom": "<summary>"}`. */
  parameters: Record<string, unknown>;
}

export interface StructureDetail {
  name: string;
  index: number;
  type: string;
  color: Vec3;
  visible: boolean;
  overlap_priority: number;
  num_voxels: number;
  num_scenarios: number;
  center_of_mass: Vec3 | null;
  alpha_x: number;
  beta_x: number;
  tissue: TissueDraft;
  /** Qt VOI row tooltip text. */
  tissue_summary: string;
  objectives: ObjectiveData[];
}

export interface LogHistoryEntry extends Omit<LogEvent, "event" | "session"> {
  event?: "log";
  session?: string;
  seq?: number;
}

export interface OptimizationHistory {
  label: string;
  points: Array<{ iteration: number; objective: number } & Record<string, number>>;
  finished: boolean;
  outcome: "done" | "stopped" | "failed" | null;
}

/** v1 methods used by more than one package. */
export interface SharedV1Methods {
  "app.info": { params: Record<string, never>; result: AppInfo };
  "session.create": {
    params: {
      profile?: Profile;
      options?: {
        default_plan?: Partial<PlanSummary>;
        locked_fields?: string[];
        auto_plan?: boolean;
        solver_preference?: string[];
        max_iter?: number;
        log_level?: LogLevel;
      };
    };
    result: { session: string; options?: SessionOptions };
  };
  "session.history": {
    params: { since?: number };
    result: { log: LogHistoryEntry[]; optimization: OptimizationHistory | null };
  };
  "cst.get": { params: { names?: number[] }; result: { structures: StructureDetail[] } };
}

/**
 * v1 method tables of the feature modules (`./protocol/<feature>.ts`); each package adds one
 * `& <Feature>Methods` line. Entries override the v0 entry of the same name.
 */
type V1Methods = SharedV1Methods
  & PlanMethods
  & IoMethods
  & ViewerMethods;

/** Request params and decoded results per method. */
export interface Methods extends Omit<V0Methods, keyof V1Methods>, V1Methods {}

export type Method = keyof Methods;
export type ParamsOf<M extends Method> = Methods[M]["params"];
export type ResultOf<M extends Method> = Methods[M]["result"];

/** Methods that run without a session. */
export const SESSIONLESS_METHODS: ReadonlySet<string> = new Set(["app.info", "session.create"]);

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

export interface WireRequest {
  id: number;
  session?: string;
  method: string;
  params: unknown;
  /** Binary payloads referenced by `{$array}` entries in `params`. */
  buffers?: ArrayBuffer[];
}

export interface WireError {
  type: string;
  message: string;
  traceback?: string;
  /** Machine-readable reason of a `PreconditionError` (e.g. `bio_model_unsupported`). */
  code?: string;
  /** v1: `[{field, message}]` of a `ValidationError`, the issues of `tissue_errors`. */
  details?: unknown;
}

export type WireResponse =
  | { id: number; ok: true; result: unknown; buffers?: ArrayBuffer[] }
  | { id: number; ok: false; error: WireError };

export interface StateEvent {
  event: "state";
  session: string;
  state: SessionState;
}

export interface ProgressEvent {
  event: "progress";
  session: string;
  label: string;
  fraction: number | null;
  detail?: string | null;
  data?: Record<string, number>;
}

export type LogLevel = "debug" | "info" | "warning" | "error" | "critical";

export interface LogEvent {
  event: "log";
  session: string;
  level: LogLevel;
  message: string;
  logger?: string;
  /** An algorithm `WarningReport` (Qt shows these once after the run; amendment 17). */
  report?: boolean;
  /** A report whose message was already logged as a record (the log shows it once). */
  repeat?: boolean;
}

export type BackendEvent = StateEvent | ProgressEvent | LogEvent;
