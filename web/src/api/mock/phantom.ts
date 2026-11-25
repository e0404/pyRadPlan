/**
 * Synthetic phantoms for the mock backend: a TG119-like water phantom (C-shaped target
 * around a core) and a box phantom, plus a fake dose distribution shaped by the plan's
 * gantry angles. Pure functions, no DOM.
 */
import type { Grid, Vec3 } from "../protocol";

export interface PhantomStructure {
  name: string;
  type: "TARGET" | "OAR" | "EXTERNAL" | "HELPER";
  color: Vec3;
  mask: Uint8Array;
}

export interface Phantom {
  name: string;
  grid: Grid;
  ct: Int16Array;
  structures: PhantomStructure[];
  /** Signed distance (mm, negative inside) to the target, used by the fake dose. */
  targetDistance: Float32Array;
  coreDistance: Float32Array | null;
}

export const MOCK_PHANTOMS = ["TG119", "BOXPHANTOM"] as const;

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** Deterministic pseudo-random noise in [-1, 1] (no Math.random, so tests are stable). */
function hashNoise(i: number): number {
  let h = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return ((h >>> 0) / 0xffffffff) * 2 - 1;
}

function makeGrid(shape: Vec3, spacing: Vec3, origin: Vec3): Grid {
  return { shape, spacing, origin, direction: [...IDENTITY] };
}

/** Iterate voxels with their world position in mm. */
function forEachVoxel(grid: Grid, fn: (i: number, x: number, y: number, z: number) => void): void {
  const [nz, ny, nx] = grid.shape;
  const [sx, sy, sz] = grid.spacing;
  const [ox, oy, oz] = grid.origin;
  let i = 0;
  for (let k = 0; k < nz; k++) {
    const z = oz + k * sz;
    for (let j = 0; j < ny; j++) {
      const y = oy + j * sy;
      for (let c = 0; c < nx; c++, i++) fn(i, ox + c * sx, y, z);
    }
  }
}

export function makeTG119(): Phantom {
  const grid = makeGrid([129, 167, 167], [3, 3, 2.5], [-250, -250, -160]);
  const n = grid.shape[0] * grid.shape[1] * grid.shape[2];
  const ct = new Int16Array(n);
  const body = new Uint8Array(n);
  const target = new Uint8Array(n);
  const core = new Uint8Array(n);
  const targetDistance = new Float32Array(n);
  const coreDistance = new Float32Array(n);

  forEachVoxel(grid, (i, x, y, z) => {
    // Elliptic water cylinder with rounded ends, a couch below it.
    const e = Math.hypot(x / 150, y / 110);
    const inBody = e <= 1 && Math.abs(z) <= 130;
    let hu = -1000;
    if (inBody) {
      body[i] = 1;
      hu = 0;
      // A few inserts to make the window/level visible: two "bone" rods, one "lung" rod.
      if (Math.hypot(x - 95, y - 20) < 14) hu = 650;
      if (Math.hypot(x + 95, y - 20) < 14) hu = 650;
      if (Math.hypot(x, y - 75) < 18) hu = -720;
      if (e > 0.96) hu = 60;
      hu += 12 * hashNoise(i);
    } else if (y > 118 && y < 130 && Math.abs(x) < 210) {
      hu = 200 + 20 * hashNoise(i);
    }
    ct[i] = Math.round(hu);

    // C-shaped target: annulus 15..37 mm, opening anteriorly (-y) around the core.
    const r = Math.hypot(x, y);
    const dRadial = Math.max(15 - r, r - 37);
    const dGap = Math.min(12 - Math.abs(x), -y);
    const dZ = Math.abs(z) - 40;
    const dTarget = Math.max(dRadial, dGap, dZ);
    targetDistance[i] = dTarget;
    if (dTarget <= 0 && inBody) target[i] = 1;

    const dCore = Math.max(r - 10, Math.abs(z) - 50);
    coreDistance[i] = dCore;
    if (dCore <= 0) core[i] = 1;
  });

  return {
    name: "TG119",
    grid,
    ct,
    structures: [
      { name: "Core", type: "OAR", color: [85, 85, 0], mask: core },
      { name: "OuterTarget", type: "TARGET", color: [255, 255, 85], mask: target },
      { name: "BODY", type: "OAR", color: [170, 0, 255], mask: body },
    ],
    targetDistance,
    coreDistance,
  };
}

export function makeBoxPhantom(): Phantom {
  const grid = makeGrid([80, 100, 100], [3, 3, 3], [-150, -150, -120]);
  const n = grid.shape[0] * grid.shape[1] * grid.shape[2];
  const ct = new Int16Array(n);
  const body = new Uint8Array(n);
  const target = new Uint8Array(n);
  const oar = new Uint8Array(n);
  const targetDistance = new Float32Array(n);
  const coreDistance = new Float32Array(n);
  const box = (x: number, y: number, z: number, c: Vec3, h: Vec3) =>
    Math.max(Math.abs(x - c[0]) - h[0], Math.abs(y - c[1]) - h[1], Math.abs(z - c[2]) - h[2]);
  forEachVoxel(grid, (i, x, y, z) => {
    const dBody = box(x, y, z, [0, 0, 0], [120, 120, 100]);
    if (dBody <= 0) {
      body[i] = 1;
      ct[i] = Math.round(10 * hashNoise(i));
    } else {
      ct[i] = -1000;
    }
    const dT = box(x, y, z, [0, 0, 0], [24, 24, 24]);
    targetDistance[i] = dT;
    if (dT <= 0) target[i] = 1;
    const dO = box(x, y, z, [0, 48, 0], [30, 15, 30]);
    coreDistance[i] = dO;
    if (dO <= 0) oar[i] = 1;
  });
  return {
    name: "BOXPHANTOM",
    grid,
    ct,
    structures: [
      { name: "Target", type: "TARGET", color: [255, 0, 0], mask: target },
      { name: "OAR", type: "OAR", color: [0, 160, 255], mask: oar },
      { name: "BODY", type: "EXTERNAL", color: [0, 200, 0], mask: body },
    ],
    targetDistance,
    coreDistance,
  };
}

export function makePhantom(name: string): Phantom {
  switch (name.toUpperCase()) {
    case "TG119":
      return makeTG119();
    case "BOXPHANTOM":
      return makeBoxPhantom();
    default:
      throw new Error(`Unknown phantom "${name}" (available: ${MOCK_PHANTOMS.join(", ")})`);
  }
}

const sigmoid = (t: number) => 1 / (1 + Math.exp(-t));

/**
 * Fake dose (Gy): conformal high dose to the target, partial core sparing and a low-dose
 * bath along each beam (beam axes in the axial plane at the given gantry angles, IEC-like:
 * gantry 0 enters from anterior, i.e. -y).
 */
export function fakeDose(
  phantom: Phantom,
  gantryAngles: readonly number[],
  prescription: number,
  isoCenter: Vec3 = [0, 0, 0],
): Float32Array {
  const { grid, targetDistance, coreDistance } = phantom;
  const body = phantom.structures.find((s) => s.name === "BODY")?.mask;
  const n = targetDistance.length;
  const dose = new Float32Array(n);
  const beams = gantryAngles.map((a) => {
    const t = (a * Math.PI) / 180;
    return { dx: Math.sin(t), dy: -Math.cos(t) };
  });
  const bathPerBeam = beams.length ? 0.55 / beams.length : 0;
  forEachVoxel(grid, (i, x, y, z) => {
    if (body && !body[i]) return;
    const px = x - isoCenter[0];
    const py = y - isoCenter[1];
    const pz = z - isoCenter[2];
    const conformal = sigmoid(-targetDistance[i] / 3.5);
    let bath = 0;
    for (const b of beams) {
      // Distance from the beam axis (in-plane) and depth from the source side.
      const lateral = Math.abs(px * b.dy - py * b.dx);
      const depth = -(px * b.dx + py * b.dy);
      const field = sigmoid((45 - lateral) / 4) * sigmoid((50 - Math.abs(pz)) / 4);
      bath += bathPerBeam * field * Math.exp(-0.004 * (depth + 150));
    }
    let d = Math.max(conformal, bath * 1.4);
    if (coreDistance) d *= 1 - 0.5 * sigmoid(-(coreDistance[i] + 2) / 2.5);
    d = d * (1 + 0.015 * hashNoise(i * 7 + 3));
    dose[i] = d > 0.025 ? prescription * d : 0;
  });
  return dose;
}

/** Cumulative DVH (dose bins in Gy, volume in %) of `dose` inside `mask`. */
export function cumulativeDvh(
  dose: Float32Array,
  mask: Uint8Array,
  maxDose: number,
  bins = 200,
): { dose: Float32Array; volume: Float32Array } {
  const hist = new Float64Array(bins + 1);
  let count = 0;
  const scale = maxDose > 0 ? bins / maxDose : 0;
  for (let i = 0; i < dose.length; i++) {
    if (!mask[i]) continue;
    count++;
    hist[Math.min(bins, Math.floor(dose[i] * scale))]++;
  }
  const doseAxis = new Float32Array(bins + 1);
  const volume = new Float32Array(bins + 1);
  let above = count;
  for (let b = 0; b <= bins; b++) {
    doseAxis[b] = (b / bins) * maxDose;
    volume[b] = count ? (100 * above) / count : 0;
    above -= hist[b];
  }
  return { dose: doseAxis, volume };
}
