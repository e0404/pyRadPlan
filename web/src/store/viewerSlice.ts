/**
 * Store slice of package B (viewer data beyond CT / quantities, structures, beams, analysis
 * window). Registered in `appStore.ts` with one spread line; data is filled by `revisions.ts`.
 */
import type { Grid, StructureDetail, StructureInfo } from "../api/protocol";
import type { BeamDraft, BeamsResult } from "../api/protocol/viewer";
import type { MaskVolume } from "../renderer/contours";

/** Revisions of the data currently held (compared with `state.revisions`). */
export interface AppliedRevisions {
  ct: number | null;
  masks: number | null;
  beams: number | null;
  cst: number | null;
  result: number | null;
  volumes: Record<string, number>;
  /** Structure list (positions and names) the visibility defaults were derived from. */
  structureNames: string;
}

export interface MaskSet {
  grid: Grid;
  masks: MaskVolume[];
}

export interface ViewerSlice {
  maskSet: MaskSet | null;
  structureDetails: Record<number, StructureDetail>;
  beams: BeamsResult | null;
  /** Beam angles edited in the plan form but not applied yet (drawn dashed with "*"). */
  draftBeams: BeamDraft | null;
  showBeams: boolean;
  appliedRevisions: AppliedRevisions;
  analysisOpen: boolean;
  editVoi: number | null;
  /** The slices follow the isocenter until the user picks a slice (Qt `_follow_isocenter`). */
  followIsocenter: boolean;
}

export const EMPTY_REVISIONS: AppliedRevisions = {
  ct: null,
  masks: null,
  beams: null,
  cst: null,
  result: null,
  volumes: {},
  structureNames: "",
};

export const viewerSliceInitial: ViewerSlice = {
  maskSet: null,
  structureDetails: {},
  beams: null,
  draftBeams: null,
  showBeams: false,
  appliedRevisions: EMPTY_REVISIONS,
  analysisOpen: false,
  editVoi: null,
  followIsocenter: true,
};

/**
 * Qt default checked set (`VOIsWidget._determine_checked_vois`): structures with objectives,
 * else the first EXTERNAL / TARGET / OAR by type, filled by name ("body", "target", "oar");
 * empty structures are never checked. Without v1 fields the backend's `visible` flags are used.
 */
export function defaultCheckedVois(structures: readonly StructureInfo[]): Set<number> {
  const hasV1 = structures.some((s) => s.num_objectives !== undefined || s.num_voxels !== undefined);
  if (!hasV1) return new Set(structures.filter((s) => s.visible).map((s) => s.index));
  const sorted = [...structures].sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
  const empty = new Set(sorted.filter((s) => s.num_voxels === 0).map((s) => s.index));
  const withObjectives = sorted.filter((s) => (s.num_objectives ?? 0) > 0).map((s) => s.index);
  let picked: (number | undefined)[];
  if (withObjectives.length) picked = withObjectives;
  else {
    const byType = (t: string) => sorted.find((s) => s.type.toUpperCase() === t)?.index;
    const byName = (n: string) => sorted.find((s) => s.name.toLowerCase().includes(n))?.index;
    picked = [byType("EXTERNAL") ?? byName("body"), byType("TARGET") ?? byName("target"), byType("OAR") ?? byName("oar")];
  }
  return new Set(picked.filter((i): i is number => i !== undefined && !empty.has(i)));
}

// ---------------------------------------------------------------------------
// Beam editor (Qt `ViewingWidget.set_beam_editor`): the plan form registers how a dragged
// gantry angle enters its draft, and reports its edited angles with `setDraftBeams`.
// ---------------------------------------------------------------------------

type BeamEditor = (index: number, gantry: number) => void;
let beamEditor: BeamEditor | null = null;

/** Register the plan form's handler for dragged beams; returns the unregister function. */
export function registerBeamEditor(editor: BeamEditor): () => void {
  beamEditor = editor;
  return () => {
    if (beamEditor === editor) beamEditor = null;
  };
}

export function currentBeamEditor(): BeamEditor | null {
  return beamEditor;
}
