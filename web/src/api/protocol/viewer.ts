/**
 * Protocol v1 types of the viewer / structures / analysis / results package (package B),
 * see web/PROTOCOL.md "v1". Registered in `protocol.ts` via `ViewerMethods`; the shared v1
 * types come from the shared block in `protocol.ts`.
 */
import type {
  ArrayRef,
  ComponentKind,
  DvhResult,
  Grid,
  NDArray,
  ResultComponent,
  ResultGroup,
  ResultsInfo,
  Revisions,
  SessionState,
  StructureDetail,
  StructureMasks,
  TissueDraft,
  Vec3,
  Volume,
} from "../protocol";

export type { Revisions, ResultsInfo, ResultGroup, ResultComponent, ComponentKind, StructureDetail, TissueDraft };

/** Changes accepted by `cst.update_structure`. */
export interface StructureChanges {
  type?: string;
  color?: Vec3;
  visible?: boolean;
  overlap_priority?: number;
  tissue?: TissueDraft;
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

export type VolumeEncoding = "float32" | "auto";
export type MaskEncoding = "uint8" | "bits" | "crops";

export interface BitsMaskEntry {
  index: number;
  name: string;
  color: Vec3;
  plane: number;
  bit: number;
  num_voxels: number;
  bbox: [number, number, number, number, number, number] | null;
}

export interface BitsMasks {
  grid: Grid;
  encoding: "bits";
  planes: NDArray[];
  masks: BitsMaskEntry[];
}

export interface CropMaskEntry {
  index: number;
  name: string;
  color: Vec3;
  num_voxels: number;
  /** `[z0, y0, x0, z1, y1, x1]`, ends exclusive. */
  bbox: [number, number, number, number, number, number] | null;
  data: NDArray<Uint8Array> | null;
}

export interface CropMasks {
  grid: Grid;
  encoding: "crops";
  masks: CropMaskEntry[];
}

export interface BeamDraft {
  gantry_angles: number[];
  couch_angles: number[];
  iso_center?: Vec3 | Vec3[] | null;
}

export interface BeamInfo {
  index: number;
  name: string;
  label: string;
  gantry: number;
  couch: number;
  iso_center: Vec3;
  /** Unit vector isocenter -> source, LPS. */
  source_direction: Vec3;
  orbit_u?: Vec3;
  orbit_w?: Vec3;
  orbit_axis?: Vec3;
  pending: boolean;
}

export interface BeamsResult {
  source: "stf" | "plan" | null;
  iso_centers: Vec3[];
  beams: BeamInfo[];
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

export interface UnitInfo {
  symbol: string;
  dimensionless: boolean;
}

export interface DvhCurveV1 {
  index: number;
  name: string;
  color: Vec3;
  dose: NDArray;
  volume: NDArray;
  stats?: { mean: number | null; min: number | null; max: number | null };
}

export interface DvhSet {
  key: string;
  /** Axis label (`QUANTITY_META`). */
  label: string;
  /** Selector label ("dose, beam 0"), v1 amendment 5. */
  name?: string;
  unit: string;
  unit_info: UnitInfo;
  curves: DvhCurveV1[];
}

export interface DvhSetsResult {
  sets: DvhSet[];
}

export type MetricKind = "mean" | "std" | "max" | "min" | "dx" | "vx";

export interface QiMetric {
  id: string;
  label: string;
  kind: MetricKind;
  parameter?: number;
  unit: string;
}

export interface QiResult {
  quantity: string;
  label: string;
  /** Selector label ("dose, beam 0"), v1 amendment 5. */
  name?: string;
  unit: string;
  unit_symbol: string;
  ref_vols: number[];
  ref_doses: number[];
  metrics: QiMetric[];
  rows: { index: number; name: string; color: Vec3; values: Record<string, number | null> }[];
}

/** One MLC shape of a beam's delivery sequence; positions in mm in the beam's eye view. */
export interface SequenceShape {
  leaf_width: number;
  /** Lower edge of every leaf row, ascending. */
  boundaries: number[];
  /** Left and right leaf of every row. */
  leaf_positions: [number, number][];
}

/** `[travel_min, travel_max, stack_min, stack_max]` in mm. */
export type BevWindow = [number, number, number, number];

export interface SequenceFluence {
  window: BevWindow;
  bixel_width: number | null;
  /** `(stack, travel)`, NaN where the beam has no ray. */
  optimized: NDArray<Float32Array> | null;
  delivered: NDArray<Float32Array> | null;
}

export interface SequenceTarget {
  /** `(travel, stack)` of the centre of the first bin. */
  origin: [number, number];
  spacing: [number, number];
  /** `(stack, travel)`, 1 where the projected target lies. */
  mask: NDArray<Uint8Array>;
}

export interface SequenceBeam {
  index: number;
  label: string;
  orientation: "X" | "Y" | null;
  window: BevWindow | null;
  weights: number[];
  shapes: SequenceShape[];
  fluence: SequenceFluence | null;
  target: SequenceTarget | null;
}

/** `view.sequence` (amendment 30): the delivery sequence of the active result. */
export interface SequenceView {
  sequencer: string | null;
  num_of_levels: number | null;
  beams: SequenceBeam[];
}

export interface ViewerMethods {
  "view.sequence": { params: Record<string, never>; result: SequenceView };
  "view.volume": { params: { name: string; phase?: number; encoding?: VolumeEncoding }; result: Volume };
  "view.structure_masks": {
    params: { names?: number[]; encoding?: MaskEncoding };
    result: StructureMasks | BitsMasks | CropMasks;
  };
  "view.beams": { params: { draft?: BeamDraft }; result: BeamsResult };
  "cst.update_structure": {
    params: { name: number; changes: StructureChanges; apply_anyway?: boolean };
    result: { state: SessionState; structure: StructureDetail };
  };
  "analysis.dvh": {
    params: {
      quantity?: string;
      names?: number[];
      num_points?: number;
      keys?: string[];
      range?: "shared" | "per_structure";
    };
    result: DvhResult & Partial<DvhSetsResult> & { label?: string; unit_info?: UnitInfo };
  };
  "analysis.qi": {
    params: { quantity: string; names?: number[]; ref_vols?: number[]; ref_doses?: number[] };
    result: QiResult;
  };
  "analysis.compare_units": { params: { a: string; b: string }; result: { shareable: boolean; scale: number | null } };
  "workflow.snapshot": { params: { tag?: string }; result: { state: SessionState; tag: string } };
}

/** Reference to an array in a wire message (re-exported for mock handlers). */
export type { ArrayRef };
