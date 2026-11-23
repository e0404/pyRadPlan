/**
 * Beam overlay geometry, ported from `quantity_widget.py` (`_update_beams`, `_toward_viewer`,
 * `_beam_radius`, `_start_beam_drag`, ...). Works in "slice millimetres": `H = u * sh`,
 * `V = v * sv` with the slice coordinates `(u, v)` of `geometry.ts` (voxel centre at
 * `i + 0.5`), so screen distances are isotropic. Inputs are LPS millimetres (`view.beams`);
 * the CT direction matrix maps them to voxel axes (Qt ignores it, ambiguity 8).
 */
import { PLANES, sliceLayout, type PlaneConvention, type SliceLayout, type VolumeGeometry } from "./geometry";

export type V3 = readonly [number, number, number];

export const BEAM_COLOR: readonly [number, number, number] = [255, 210, 0];
export const BEAM_WIDTH_PX = 3;
export const BEAM_PERSPECTIVE = 6;
export const BEAM_MIN_IN_PLANE = 0.035;
export const BEAM_MIN_OUT_OF_PLANE = 0.087;
export const BEAM_MIN_ORBIT_TILT = 0.3;

export interface BeamInput {
  index: number;
  name: string;
  label: string;
  gantry: number;
  couch: number;
  iso_center: V3;
  source_direction: V3;
  orbit_u?: V3;
  orbit_w?: V3;
  orbit_axis?: V3;
  pending: boolean;
}

function direction(geom: VolumeGeometry): number[] {
  return geom.direction ? [...geom.direction] : [1, 0, 0, 0, 1, 0, 0, 0, 1];
}

/** LPS vector -> components along the voxel axes (`D^T v`). */
export function toAxes(geom: VolumeGeometry, v: V3): [number, number, number] {
  const d = direction(geom);
  return [d[0] * v[0] + d[3] * v[1] + d[6] * v[2], d[1] * v[0] + d[4] * v[1] + d[7] * v[2], d[2] * v[0] + d[5] * v[1] + d[8] * v[2]];
}

/** LPS point (mm) -> continuous voxel index (voxel centres at integers). */
export function pointToIndex(geom: VolumeGeometry, p: V3): [number, number, number] {
  const rel = toAxes(geom, [p[0] - geom.origin[0], p[1] - geom.origin[1], p[2] - geom.origin[2]]);
  return [rel[0] / geom.spacing[0], rel[1] / geom.spacing[1], rel[2] / geom.spacing[2]];
}

/** Continuous voxel index -> slice mm `(H, V)` and the slice index along the normal. */
export function indexToSliceMm(layout: SliceLayout, idx: readonly number[]): { h: number; v: number; n: number } {
  const { axes, nh, nv, sh, sv } = layout;
  const u = (axes.flipH ? nh - 1 - idx[axes.h] : idx[axes.h]) + 0.5;
  const v = (axes.flipV ? nv - 1 - idx[axes.v] : idx[axes.v]) + 0.5;
  return { h: u * sh, v: v * sv, n: idx[axes.slice] };
}

/** LPS direction -> slice mm components `(dH, dV)` and the signed sine towards the viewer. */
export function directionInSlice(geom: VolumeGeometry, layout: SliceLayout, dir: V3): { dh: number; dv: number; toward: number } {
  const m = toAxes(geom, dir);
  const norm = Math.hypot(m[0], m[1], m[2]) || 1;
  const { axes } = layout;
  const right: [number, number, number] = [0, 0, 0];
  const down: [number, number, number] = [0, 0, 0];
  right[axes.h] = axes.flipH ? -1 : 1;
  down[axes.v] = axes.flipV ? -1 : 1;
  const into = [right[1] * down[2] - right[2] * down[1], right[2] * down[0] - right[0] * down[2], right[0] * down[1] - right[1] * down[0]];
  const toward = -(m[0] * into[0] + m[1] * into[1] + m[2] * into[2]) / norm;
  return { dh: (right[axes.h] * m[axes.h]) / norm, dv: (down[axes.v] * m[axes.v]) / norm, toward };
}

/** Largest radius (mm) of a circle around `(h, v)` within the slice's voxel centres. */
export function beamRadius(layout: SliceLayout, h: number, v: number): number | null {
  const uh = h / layout.sh;
  const uv = v / layout.sv;
  const r = Math.min(Math.min(uh - 0.5, layout.nh - 0.5 - uh) * layout.sh, Math.min(uv - 0.5, layout.nv - 0.5 - uv) * layout.sv);
  return r > 0 ? r : null;
}

export interface ProjectedBeam {
  beam: BeamInput;
  /** Isocenter in slice mm. */
  iso: { h: number; v: number };
  toward: number;
  perpendicular: boolean;
  /** Source position in slice mm (absent for perpendicular beams or outside the slice). */
  source?: { h: number; v: number };
  radius?: number;
  sourceWidth: number;
  draggable: boolean;
  /** Unit in-plane direction iso -> source (screen axes). */
  unit: { h: number; v: number };
  /** `[[u_h, w_h], [u_v, w_v]]` of the gantry orbit (slice mm per unit radius). */
  orbit?: [[number, number], [number, number]];
  text: string;
  tooltip: string;
}

export function orbitTilt(geom: VolumeGeometry, layout: SliceLayout, beam: BeamInput): number {
  if (!beam.orbit_axis) return 0;
  return Math.abs(directionInSlice(geom, layout, beam.orbit_axis).toward);
}

export function outOfPlaneNote(toward: number): string {
  const angle = (Math.asin(Math.min(Math.abs(toward), 1)) * 180) / Math.PI;
  if (angle < 1) return "in the displayed plane";
  return `${angle.toFixed(0)}° out of plane, source ${toward > 0 ? "towards" : "away from"} the viewer`;
}

/** ⊗ (tail) for a beam going into the screen (source in front), ⊙ (tip) coming out. */
export function directionSymbol(toward: number): string {
  return toward > 0 ? "⊗" : "⊙";
}

/** Plane in which the beam's gantry orbit is seen most face-on (where it can be dragged). */
export function bestDragPlane(geom: VolumeGeometry, beam: BeamInput, convention?: PlaneConvention): string {
  let best = PLANES[0];
  let tilt = -1;
  for (const plane of PLANES) {
    const t = orbitTilt(geom, sliceLayout(geom, plane, convention), beam);
    if (t > tilt) {
      tilt = t;
      best = plane;
    }
  }
  return best[0].toUpperCase() + best.slice(1);
}

export function projectBeam(geom: VolumeGeometry, layout: SliceLayout, beam: BeamInput, convention?: PlaneConvention): ProjectedBeam {
  const iso = indexToSliceMm(layout, pointToIndex(geom, beam.iso_center));
  const { dh, dv, toward } = directionInSlice(geom, layout, beam.source_direction);
  const inPlane = Math.hypot(dh, dv);
  const perpendicular = inPlane < 1e-6 || Math.sqrt(Math.max(0, 1 - toward * toward)) < BEAM_MIN_IN_PLANE;
  const sourceWidth = Math.max(1, BEAM_WIDTH_PX * BEAM_PERSPECTIVE ** toward);
  const tilt = orbitTilt(geom, layout, beam);
  const draggable = !perpendicular && tilt >= BEAM_MIN_ORBIT_TILT;
  const lines = [beam.label, outOfPlaneNote(toward)];
  if (beam.pending) lines.push("Not applied yet: apply or discard it in the plan panel");
  if (beam.orbit_axis) {
    if (tilt >= BEAM_MIN_ORBIT_TILT) lines.push("Drag the source marker to change the gantry angle");
    else lines.push(`Drag it in the ${bestDragPlane(geom, beam, convention)} view to change the gantry angle`);
  }
  let text = beam.name + (beam.pending ? "*" : "");
  if (!perpendicular && Math.abs(toward) >= BEAM_MIN_OUT_OF_PLANE) text = `${text} ${directionSymbol(toward)}`;
  const projected: ProjectedBeam = {
    beam,
    iso,
    toward,
    perpendicular,
    sourceWidth,
    draggable,
    unit: inPlane > 0 ? { h: dh / inPlane, v: dv / inPlane } : { h: 0, v: 0 },
    text,
    tooltip: lines.filter(Boolean).join("\n"),
  };
  if (perpendicular) return projected;
  const radius = beamRadius(layout, iso.h, iso.v);
  if (radius === null) return projected;
  projected.radius = radius;
  projected.source = { h: iso.h + dh * radius, v: iso.v + dv * radius };
  if (beam.orbit_u && beam.orbit_w) {
    const u = directionInSlice(geom, layout, beam.orbit_u);
    const w = directionInSlice(geom, layout, beam.orbit_w);
    projected.orbit = [
      [u.dh, w.dh],
      [u.dv, w.dv],
    ];
  }
  return projected;
}

/** Gantry angle (degrees, rounded, 0..360) whose orbit point lies towards `offset` (slice mm). */
export function orbitAngle(orbit: [[number, number], [number, number]], offset: { h: number; v: number }): number | null {
  const [[a, b], [c, d]] = orbit;
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-12 || (offset.h === 0 && offset.v === 0)) return null;
  const x = (d * offset.h - b * offset.v) / det;
  const y = (-c * offset.h + a * offset.v) / det;
  const deg = Math.round((Math.atan2(y, x) * 180) / Math.PI);
  return ((deg % 360) + 360) % 360;
}

/** Orbit point (slice mm) at `gantry` degrees. */
export function orbitPoint(p: ProjectedBeam, gantry: number): { h: number; v: number } | null {
  if (!p.orbit || p.radius === undefined) return null;
  const t = (gantry * Math.PI) / 180;
  const [[a, b], [c, d]] = p.orbit;
  return { h: p.iso.h + p.radius * (a * Math.cos(t) + b * Math.sin(t)), v: p.iso.v + p.radius * (c * Math.cos(t) + d * Math.sin(t)) };
}

/** Text anchor for a tip label pointing along `component` (0 = start at point, 1 = end, 0.5 centre). */
export function labelAnchor(component: number): number {
  if (component > 0.3) return 0;
  if (component < -0.3) return 1;
  return 0.5;
}

/** Sign of the source direction out of the plane for a dragged beam at `gantry`. */
export function towardAt(geom: VolumeGeometry, layout: SliceLayout, beam: BeamInput, gantry: number): number {
  if (!beam.orbit_u || !beam.orbit_w) return 0;
  const t = (gantry * Math.PI) / 180;
  const dir: V3 = [
    Math.cos(t) * beam.orbit_u[0] + Math.sin(t) * beam.orbit_w[0],
    Math.cos(t) * beam.orbit_u[1] + Math.sin(t) * beam.orbit_w[1],
    Math.cos(t) * beam.orbit_u[2] + Math.sin(t) * beam.orbit_w[2],
  ];
  return directionInSlice(geom, layout, dir).toward;
}
