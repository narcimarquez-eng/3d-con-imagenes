import * as THREE from 'three';
import { computeHumanRig, computeSkinWeightMaps, sampleSkinWeights, type HumanRig } from './humanRig';

// ============================================================================
// HUMAN BODY TEMPLATE
// Builds a closed 3D body that follows the drawing's silhouette exactly and
// gets its depth from human proportions instead of from image brightness:
//   - head: oval, a bit deeper than wide
//   - neck: round
//   - torso: wider than deep (chest ≈ 0.6, waist ≈ 0.52, hips ≈ 0.58)
//   - legs: two round cylinders, even when cloth hangs between them
//   - arms, hands and anything else: round tubes sized by their own width
// Front and back surfaces meet directly on the outline, so there is no
// separate "joint" strip with its own colours.
// ============================================================================

export interface HumanBodyInputs {
  /** Silhouette mask, 1 = body, size w*w. */
  mask: Uint8Array;
  /** Tensor size (square). */
  w: number;
  /** Normalised photo relief of the front / back image, size w*w. */
  frontRelief: Float32Array;
  backRelief: Float32Array;
  /** Marks which vertices are in use (the viewport reads it). */
  vertexUsedMask: Uint8Array;
  maxVertices: number;
  atlas?: AtlasRepairInput;
}

export interface AtlasRepairInput {
  /** RGBA8 atlas (front quadrant at x 0..w-1, back quadrant at x w..2w-1, rows 0..w-1). */
  albedo: Uint8Array;
  /** RGBA8 roughness (G) / metalness (B) atlas with the same layout. */
  roughMetal?: Uint8Array;
  atlasWidth: number;
  /** Background colour of the front / back photos (0..1). */
  background: [number, number, number];
  backBackground?: [number, number, number];
}

export interface HumanBodyOptions {
  /** Overall body depth multiplier (1 = natural human proportions). */
  depthScale: number;
  /** How much of the photo relief is added on top (0..1). */
  reliefAmount: number;
  /** Mesh rows over the figure height. */
  rows: number;
  /** Also build a skeleton and skin weights (skinIndex / skinWeight attributes). */
  rig?: boolean;
  /** Add shoes (flat sole, toe forward). Default true. */
  shoes?: boolean;
}

export interface HumanBodyLandmarks {
  top: number;
  bottom: number;
  axisX: number;
  neckY: number;
  shoulderY: number;
  armpitY: number;
  crotchY: number;
}

export interface HumanBodyStats {
  vertexCount: number;
  triangleCount: number;
  landmarks: HumanBodyLandmarks;
  rig?: HumanRig;
}

// World units: the existing pipeline maps the full tensor width to 2.85 units.
const WORLD_SPAN = 2.85;

let _size = 0;
let _field = new Float32Array(0);
let _tmp = new Float32Array(0);
let _dist = new Float32Array(0);
let _radius = new Float32Array(0);
let _depth = new Float32Array(0);
let _frontD = new Float32Array(0);
let _backD = new Float32Array(0);
let _coreMask = new Uint8Array(0);
let _limbMask = new Uint8Array(0);
let _limbDist = new Float32Array(0);
let _limbRadius = new Float32Array(0);
let _rowCoreDepth = new Float32Array(0);

function ensureBuffers(w: number) {
  if (_size === w) return;
  _size = w;
  const n = w * w;
  _field = new Float32Array(n);
  _tmp = new Float32Array(n);
  _dist = new Float32Array(n);
  _radius = new Float32Array(n);
  _depth = new Float32Array(n);
  _frontD = new Float32Array(n);
  _backD = new Float32Array(n);
  _coreMask = new Uint8Array(n);
  _limbMask = new Uint8Array(n);
  _limbDist = new Float32Array(n);
  _limbRadius = new Float32Array(n);
  _rowCoreDepth = new Float32Array(w);
}

const K5 = [0.0625, 0.25, 0.375, 0.25, 0.0625];

/** Separable 5-tap blur; when `mask` is given, pixels outside it are forced to 0. */
function blur5(src: Float32Array, w: number, passes: number, mask?: Uint8Array) {
  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let o = -2; o <= 2; o++) {
          const sx = x + o < 0 ? 0 : x + o >= w ? w - 1 : x + o;
          s += src[y * w + sx] * K5[o + 2];
        }
        _tmp[y * w + x] = s;
      }
    }
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (mask && mask[i] !== 1) {
          src[i] = 0;
          continue;
        }
        let s = 0;
        for (let o = -2; o <= 2; o++) {
          const sy = y + o < 0 ? 0 : y + o >= w ? w - 1 : y + o;
          s += _tmp[sy * w + x] * K5[o + 2];
        }
        src[i] = s;
      }
    }
  }
}

function sample(buf: Float32Array, w: number, x: number, y: number): number {
  const cx = Math.max(0, Math.min(w - 1.001, x));
  const cy = Math.max(0, Math.min(w - 1.001, y));
  const x0 = cx | 0;
  const y0 = cy | 0;
  const fx = cx - x0;
  const fy = cy - y0;
  const i = y0 * w + x0;
  const a = buf[i] * (1 - fx) + buf[i + 1] * fx;
  const b = buf[i + w] * (1 - fx) + buf[i + w + 1] * fx;
  return a * (1 - fy) + b * fy;
}

/** Chamfer distance to the outside plus the radius of the largest inscribed disk covering each pixel. */
function distanceAndRadius(mask: Uint8Array, w: number, D: Float32Array = _dist, R: Float32Array = _radius) {
  const INF = 1e6;
  for (let i = 0; i < w * w; i++) D[i] = mask[i] === 1 ? INF : 0;
  for (let y = 0; y < w; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (D[i] === 0) continue;
      let d = D[i];
      if (x === 0 || y === 0) d = Math.min(d, 1);
      if (x > 0) d = Math.min(d, D[i - 1] + 1);
      if (y > 0) {
        d = Math.min(d, D[i - w] + 1);
        if (x > 0) d = Math.min(d, D[i - w - 1] + 1.4142);
        if (x < w - 1) d = Math.min(d, D[i - w + 1] + 1.4142);
      }
      D[i] = d;
    }
  }
  for (let y = w - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      if (D[i] === 0) continue;
      let d = D[i];
      if (x === w - 1 || y === w - 1) d = Math.min(d, 1);
      if (x < w - 1) d = Math.min(d, D[i + 1] + 1);
      if (y < w - 1) {
        d = Math.min(d, D[i + w] + 1);
        if (x < w - 1) d = Math.min(d, D[i + w + 1] + 1.4142);
        if (x > 0) d = Math.min(d, D[i + w - 1] + 1.4142);
      }
      D[i] = d;
    }
  }
  for (let i = 0; i < w * w; i++) if (D[i] > 0) D[i] -= 0.5;

  R.fill(0);
  for (let y = 0; y < w; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const r = D[i];
      if (r < 1) continue;
      if (
        (x > 0 && D[i - 1] > r + 0.5) ||
        (x < w - 1 && D[i + 1] > r + 0.5) ||
        (y > 0 && D[i - w] > r + 0.5) ||
        (y < w - 1 && D[i + w] > r + 0.5)
      ) {
        continue;
      }
      const ri = Math.ceil(r);
      const r2 = r * r;
      for (let dy = -ri; dy <= ri; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= w) continue;
        for (let dx = -ri; dx <= ri; dx++) {
          if (dx * dx + dy * dy > r2) continue;
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const j = yy * w + xx;
          if (R[j] < r) R[j] = r;
        }
      }
    }
  }
}

type Run = [number, number];

function rowRuns(mask: Uint8Array, w: number, y: number): Run[] {
  const runs: Run[] = [];
  let start = -1;
  for (let x = 0; x < w; x++) {
    const on = mask[y * w + x] === 1;
    if (on && start < 0) start = x;
    if (!on && start >= 0) {
      runs.push([start, x - 1]);
      start = -1;
    }
  }
  if (start >= 0) runs.push([start, w - 1]);
  return runs;
}

function runNearest(runs: Run[], x: number): number {
  let best = -1;
  let bestD = Infinity;
  for (let k = 0; k < runs.length; k++) {
    const [a, b] = runs[k];
    const d = x < a ? a - x : x > b ? x - b : 0;
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  }
  return best;
}

function smooth1D(arr: Float32Array, from: number, to: number, radius: number) {
  const out = new Float32Array(arr.length);
  for (let y = from; y <= to; y++) {
    let s = 0;
    let ws = 0;
    for (let o = -radius; o <= radius; o++) {
      const yy = y + o;
      if (yy < from || yy > to) continue;
      const wt = Math.exp(-(o * o) / (2 * (radius * 0.5) ** 2));
      s += arr[yy] * wt;
      ws += wt;
    }
    out[y] = ws > 0 ? s / ws : arr[y];
  }
  for (let y = from; y <= to; y++) arr[y] = out[y];
}

/** Superellipse cross-section height: depth at centre, 0 at ±halfWidth. */
function crossSection(dx: number, halfWidth: number, depth: number, p: number): number {
  if (halfWidth <= 0) return 0;
  const t = Math.abs(dx) / halfWidth;
  if (t >= 1) return 0;
  return depth * Math.pow(1 - Math.pow(t, p), 1 / p);
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

/** Torso depth/width ratio along the torso (t = 0 at neck, 1 at crotch). */
function torsoDepthRatio(t: number): number {
  if (t < 0.25) return lerp(0.8, 0.74, t / 0.25); // upper chest / shoulders
  if (t < 0.6) return lerp(0.74, 0.64, (t - 0.25) / 0.35); // chest -> waist
  return lerp(0.64, 0.7, Math.min(1, (t - 0.6) / 0.4)); // waist -> hips
}

/**
 * Builds the half-thickness map (in texels) of a human body that fills the silhouette.
 */
function buildHumanDepthMap(mask: Uint8Array, w: number): HumanBodyLandmarks {
  distanceAndRadius(mask, w);

  // --- Landmarks -----------------------------------------------------------
  let top = -1;
  let bottom = -1;
  let massX = 0;
  let mass = 0;
  for (let y = 0; y < w; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x] !== 1) continue;
      if (top < 0) top = y;
      bottom = y;
      massX += x;
      mass++;
    }
  }
  _depth.fill(0);
  if (top < 0 || mass < 50) {
    return { top: 0, bottom: 0, axisX: w / 2, neckY: 0, shoulderY: 0, armpitY: 0, crotchY: 0 };
  }
  const H = bottom - top + 1;
  const axisX = massX / mass;
  const ax = Math.round(axisX);

  const runsByRow: Run[][] = [];
  for (let y = 0; y < w; y++) runsByRow.push(y >= top && y <= bottom ? rowRuns(mask, w, y) : []);
  const axisRun = (y: number) => {
    const runs = runsByRow[y];
    const k = runNearest(runs, axisX);
    return k >= 0 ? runs[k] : null;
  };

  const axisWidth = (y: number) => {
    const r = axisRun(y);
    return r ? r[1] - r[0] + 1 : 0;
  };

  // Head width: median axis width in the upper part of the head.
  const headSamples: number[] = [];
  for (let y = top + Math.round(H * 0.025); y <= top + Math.round(H * 0.07); y++) headSamples.push(axisWidth(y));
  headSamples.sort((p, q) => p - q);
  const headW = Math.max(3, headSamples[headSamples.length >> 1] || H * 0.08);

  // Shoulders start where the body run becomes much wider than the head.
  let shoulderY = top + Math.round(H * 0.2);
  for (let y = top + Math.round(H * 0.05); y <= top + Math.round(H * 0.3); y++) {
    if (axisWidth(y) > headW * 1.8) {
      shoulderY = y;
      break;
    }
  }

  // Neck: narrowest axis run between the head and the shoulders.
  let neckY = Math.max(top + 1, shoulderY - 1);
  let neckW = Infinity;
  for (let y = top + Math.round(H * 0.06); y < shoulderY; y++) {
    const wd = axisWidth(y);
    if (wd > 0 && wd <= neckW) {
      neckW = wd;
      neckY = y;
    }
  }

  // Crotch: top of the gap between the legs (on the body axis), but never lower
  // than an anatomical limit, so cloth hanging between the legs is not torso.
  let crotchY = top + Math.round(H * 0.53);
  for (let y = bottom - Math.round(H * 0.03); y > top + Math.round(H * 0.35); y--) {
    const gap = mask[y * w + ax] !== 1 && mask[y * w + Math.min(w - 1, ax + 1)] !== 1;
    if (!gap) {
      if (y < bottom - Math.round(H * 0.06)) {
        crotchY = Math.min(y + 1, top + Math.round(H * 0.56));
        break;
      }
    }
  }
  crotchY = Math.max(crotchY, neckY + Math.round(H * 0.25));

  // Armpit: first row below the shoulders where real arms hang free on both
  // sides and the body run is clearly narrower than across the shoulders.
  // (Small spikes of shoulder pads next to the head do not count.)
  let maxShoulderW = 0;
  let armpitY = shoulderY + Math.round(H * 0.1);
  for (let y = shoulderY; y < crotchY; y++) {
    const runs = runsByRow[y];
    const k = runNearest(runs, axisX);
    if (k < 0) continue;
    const wd = runs[k][1] - runs[k][0] + 1;
    maxShoulderW = Math.max(maxShoulderW, wd);
    const leftArm = k > 0 && runs[k - 1][1] - runs[k - 1][0] + 1 >= 3;
    const rightArm = k < runs.length - 1 && runs[k + 1][1] - runs[k + 1][0] + 1 >= 3;
    if (y > shoulderY + H * 0.03 && leftArm && rightArm && wd < maxShoulderW * 0.8) {
      armpitY = y;
      break;
    }
  }

  // --- Torso core width per row -------------------------------------------
  const coreHalf = new Float32Array(w);
  const coreCenter = new Float32Array(w);
  let chestHalf = 0;
  let chestCount = 0;
  for (let y = armpitY; y <= Math.min(crotchY, armpitY + Math.round(H * 0.06)); y++) {
    const r = axisRun(y);
    if (!r) continue;
    chestHalf += (r[1] - r[0] + 1) / 2;
    chestCount++;
  }
  chestHalf = chestCount > 0 ? chestHalf / chestCount : H * 0.12;
  const neckHalf = Math.max(2, (Number.isFinite(neckW) ? neckW + 1 : H * 0.06) / 2);

  for (let y = neckY; y <= crotchY; y++) {
    const r = axisRun(y);
    if (!r) continue;
    const runHalf = (r[1] - r[0] + 1) / 2;
    const runCenter = (r[0] + r[1]) / 2;
    if (y >= armpitY) {
      coreHalf[y] = runHalf;
      coreCenter[y] = runCenter;
    } else {
      // Shoulders: grow from neck width to chest width over the upper part.
      const t = Math.min(1, (y - neckY) / Math.max(1, (armpitY - neckY) * 0.45));
      const s = t * t * (3 - 2 * t);
      coreHalf[y] = Math.min(runHalf, lerp(neckHalf, chestHalf, s));
      coreCenter[y] = lerp(axisX, runCenter, 0.3);
    }
  }
  smooth1D(coreHalf, neckY, crotchY, 4);
  smooth1D(coreCenter, neckY, crotchY, 4);

  // --- Depth map, pass 1: head, torso and legs ("core") ----------------------
  const D = _depth;
  _coreMask.fill(0);
  _rowCoreDepth.fill(0);
  for (let y = top; y <= bottom; y++) {
    const runs = runsByRow[y];
    const kAxis = runNearest(runs, axisX);
    for (let k = 0; k < runs.length; k++) {
      const [a, b] = runs[k];
      const runHalf = (b - a + 1) / 2;
      const runCenter = (a + b) / 2;
      for (let x = a; x <= b; x++) {
        const i = y * w + x;
        let d = 0;
        if (y < neckY && k === kAxis) {
          // Head: oval, slightly deeper than wide.
          d = crossSection(x - runCenter, runHalf, runHalf * 1.08, 2.0);
        } else if (y >= neckY && y <= crotchY) {
          const t = (y - neckY) / Math.max(1, crotchY - neckY);
          const half = coreHalf[y];
          // Flaring clothes (skirts, tabards) widen the outline but not the body depth.
          const depth = Math.min(half * torsoDepthRatio(t), chestHalf * 0.85);
          _rowCoreDepth[y] = depth;
          d = crossSection(x - coreCenter[y], half, depth, 2.3);
        } else if (y > crotchY && k === kAxis && a < ax && b > ax) {
          // Legs joined by cloth: two leg cylinders split at the axis.
          const left = x <= ax;
          const la = left ? a : ax;
          const lb = left ? ax : b;
          const lh = (lb - la + 1) / 2;
          d = crossSection(x - (la + lb) / 2, lh, lh * 0.95, 2.0);
        } else if (y > crotchY) {
          // Separate legs / feet / hands: round section from their own width.
          d = crossSection(x - runCenter, runHalf, runHalf * 0.92, 2.0);
        }
        D[i] = d;
        if (d > 0) _coreMask[i] = 1;
      }
    }
  }

  // --- Pass 2: arms, shoulder pads and everything else ------------------------
  // Their thickness comes from their own shape (the silhouette minus the core),
  // so a shoulder pad attached to the torso does not inherit the torso's depth.
  for (let i = 0; i < w * w; i++) _limbMask[i] = mask[i] === 1 && _coreMask[i] === 0 ? 1 : 0;
  distanceAndRadius(_limbMask, w, _limbDist, _limbRadius);
  for (let y = top; y <= bottom; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (mask[i] !== 1) continue;
      const outline = _dist[i];
      if (_limbMask[i] === 1) {
        const r = Math.max(_limbRadius[i], _limbDist[i], 1);
        const d = Math.min(outline, r);
        let limb = Math.sqrt(Math.max(0, 2 * r * d - d * d)) * 0.95;
        if (_rowCoreDepth[y] > 0) limb = Math.min(limb, _rowCoreDepth[y]);
        // Smooth union so there is no crease where a shoulder pad meets the torso.
        const k = 4;
        const a = D[i];
        const hmix = Math.max(0, Math.min(1, 0.5 + (0.5 * (limb - a)) / k));
        D[i] = Math.max(a, limb, a * (1 - hmix) + limb * hmix + k * hmix * (1 - hmix));
      }
      // Never thinner than a rounded rim near the outline.
      const e = Math.min(outline, 3);
      D[i] = Math.max(D[i], Math.sqrt(Math.max(0, 6 * e - e * e)) * 0.8);
    }
  }

  // Blend the parts together (head/neck, torso/legs, torso/arms) without seams.
  blur5(D, w, 3, mask);

  return { top, bottom, axisX, neckY, shoulderY, armpitY, crotchY };
}

/** Blurred mask whose 0.5 level is the smooth outline. */
function buildOutlineField(mask: Uint8Array, w: number) {
  for (let i = 0; i < w * w; i++) _field[i] = mask[i] === 1 ? 1 : 0;
  blur5(_field, w, 2);
}

/** High-pass of the photo relief (local detail only), faded near the outline. */
function addPhotoDetail(dst: Float32Array, base: Float32Array, relief: Float32Array, mask: Uint8Array, w: number, amount: number) {
  // Local mean of the relief.
  for (let i = 0; i < w * w; i++) _radius[i] = relief[i];
  blur5(_radius, w, 4, mask);
  for (let i = 0; i < w * w; i++) {
    if (mask[i] !== 1) {
      dst[i] = 0;
      continue;
    }
    const hp = relief[i] - _radius[i];
    const fade = Math.min(1, Math.max(0, (_dist[i] - 1) / 5));
    dst[i] = Math.max(0, base[i] + hp * amount * fade);
  }
}

/**
 * Cleans the rim of the front and back textures: pixels near the outline that
 * still show the photo background (e.g. the white margin of a back view that
 * does not line up exactly) are replaced by the nearest body colour. Pixels
 * just outside the outline are filled too, so texture filtering never pulls
 * background in. This is what removes the grey/white "joint" on the rim.
 */
export function repairAtlasRim(input: AtlasRepairInput, mask: Uint8Array, w: number) {
  const { albedo, atlasWidth } = input;
  const band = 6; // texels inside the outline that may hold background
  const grow = 4; // texels outside the outline to fill
  const valid = new Uint8Array(w * w);
  const region = new Uint8Array(w * w);
  const nextR = new Float32Array(w * w);
  const nextG = new Float32Array(w * w);
  const nextB = new Float32Array(w * w);

  // Region = mask dilated by `grow`.
  for (let i = 0; i < w * w; i++) region[i] = mask[i];
  for (let g = 0; g < grow; g++) {
    const prev = region.slice();
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (prev[i]) continue;
        if ((x > 0 && prev[i - 1]) || (x < w - 1 && prev[i + 1]) || (y > 0 && prev[i - w]) || (y < w - 1 && prev[i + w])) {
          region[i] = 1;
        }
      }
    }
  }

  for (const qx of [0, w]) {
    const isBack = qx > 0;
    const [br, bg, bb] = isBack && input.backBackground ? input.backBackground : input.background;
    const px = (x: number, y: number) => (y * atlasWidth + qx + x) * 4;
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (mask[i] !== 1) {
          valid[i] = 0;
          continue;
        }
        if (_dist[i] > band) {
          valid[i] = 1;
          continue;
        }
        const o = px(x, y);
        const dr = albedo[o] / 255 - br;
        const dg = albedo[o + 1] / 255 - bg;
        const db = albedo[o + 2] / 255 - bb;
        const r = albedo[o] / 255;
        const g = albedo[o + 1] / 255;
        const b = albedo[o + 2] / 255;
        const mx = Math.max(r, g, b);
        const mn = Math.min(r, g, b);
        // Background, or the pale anti-aliased blend of background and outline.
        // The back view is re-aligned to the front outline, so its rim often
        // holds smeared background/outline greys: be stricter there.
        const nearBg = dr * dr + dg * dg + db * db < (isBack ? 0.3 : 0.12);
        const paleGrey = isBack ? mx > 0.4 && mx - mn < 0.1 : mx > 0.72 && mx - mn < 0.12;
        valid[i] = nearBg || paleGrey ? 0 : 1;
      }
    }
    // Grow valid colours into the invalid part of the region.
    for (let pass = 0; pass < band + grow + 4; pass++) {
      let changed = 0;
      for (let y = 0; y < w; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          nextR[i] = -1;
          if (valid[i] || !region[i]) continue;
          let r = 0, g = 0, b = 0, n = 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const xx = x + dx;
              const yy = y + dy;
              if ((dx === 0 && dy === 0) || xx < 0 || yy < 0 || xx >= w || yy >= w) continue;
              const j = yy * w + xx;
              if (!valid[j]) continue;
              const o = px(xx, yy);
              r += albedo[o];
              g += albedo[o + 1];
              b += albedo[o + 2];
              n++;
            }
          }
          if (n > 0) {
            nextR[i] = r / n;
            nextG[i] = g / n;
            nextB[i] = b / n;
          }
        }
      }
      for (let i = 0; i < w * w; i++) {
        if (nextR[i] < 0) continue;
        const x = i % w;
        const y = (i / w) | 0;
        const o = px(x, y);
        albedo[o] = Math.round(nextR[i]);
        albedo[o + 1] = Math.round(nextG[i]);
        albedo[o + 2] = Math.round(nextB[i]);
        valid[i] = 1;
        changed++;
      }
      if (!changed) break;
    }
  }
}

/**
 * The roughness/metalness atlas is classified per pixel and comes out noisy,
 * which shows as sparkling dots. Blur it (inside the silhouette) per quadrant.
 */
function smoothRoughMetal(rm: Uint8Array, atlasWidth: number, mask: Uint8Array, w: number) {
  const chan = new Float32Array(w * w);
  const tmp = new Float32Array(w * w);
  for (const qx of [0, w]) {
    for (const c of [1, 2]) {
      for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) chan[y * w + x] = rm[(y * atlasWidth + qx + x) * 4 + c];
      for (let p = 0; p < 3; p++) {
        for (let y = 0; y < w; y++) {
          for (let x = 0; x < w; x++) {
            let sum = 0;
            for (let o = -2; o <= 2; o++) {
              const sx = x + o < 0 ? 0 : x + o >= w ? w - 1 : x + o;
              sum += chan[y * w + sx] * K5[o + 2];
            }
            tmp[y * w + x] = sum;
          }
        }
        for (let y = 0; y < w; y++) {
          for (let x = 0; x < w; x++) {
            let sum = 0;
            for (let o = -2; o <= 2; o++) {
              const sy = y + o < 0 ? 0 : y + o >= w ? w - 1 : y + o;
              sum += tmp[sy * w + x] * K5[o + 2];
            }
            chan[y * w + x] = sum;
          }
        }
      }
      for (let y = 0; y < w; y++) {
        for (let x = 0; x < w; x++) {
          if (mask[y * w + x] !== 1) continue;
          rm[(y * atlasWidth + qx + x) * 4 + c] = Math.round(chan[y * w + x]);
        }
      }
    }
  }
}

/**
 * Replaces `geometry` with a closed human-body mesh fitted to the silhouette.
 * UVs use the same atlas layout as the relief pipeline: front image in the
 * top-left quadrant, back image (already mirrored/aligned) in the top-right.
 */
export function buildHumanBodyTemplateGeometry(
  geometry: THREE.BufferGeometry,
  inputs: HumanBodyInputs,
  options: HumanBodyOptions
): HumanBodyStats {
  const { mask, w } = inputs;
  ensureBuffers(w);

  // The depth map is computed before the outline field because both use _tmp.
  const landmarks = buildHumanDepthMap(mask, w);
  if (inputs.atlas) {
    repairAtlasRim(inputs.atlas, mask, w);
    if (inputs.atlas.roughMetal) smoothRoughMetal(inputs.atlas.roughMetal, inputs.atlas.atlasWidth, mask, w);
  }
  // Note: buildHumanDepthMap also fills _dist, which addPhotoDetail uses for fading.
  const reliefTexels = 9 * Math.max(0, Math.min(1, options.reliefAmount));
  addPhotoDetail(_frontD, _depth, inputs.frontRelief, mask, w, reliefTexels);
  addPhotoDetail(_backD, _depth, inputs.backRelief, mask, w, reliefTexels);
  buildOutlineField(mask, w);

  const { top, bottom } = landmarks;
  const rows = Math.max(40, Math.min(320, Math.round(options.rows)));
  const H = Math.max(1, bottom - top + 1);
  const cs = (H + 4) / rows; // texels per cell

  // Horizontal extent of the figure.
  let minX = w;
  let maxX = -1;
  for (let y = top; y <= bottom; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x] === 1) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
      }
    }
  }
  if (maxX < 0) {
    geometry.setIndex([]);
    return { vertexCount: 0, triangleCount: 0, landmarks };
  }

  const gx0 = minX - 2;
  const gy0 = top - 2;
  const nCols = Math.ceil((maxX - minX + 5) / cs) + 1;
  const nRows = rows + 1;
  const nNodes = nCols * nRows;

  const nodeX = new Float32Array(nNodes);
  const nodeY = new Float32Array(nNodes);
  const inside = new Uint8Array(nNodes);
  for (let j = 0; j < nRows; j++) {
    for (let i = 0; i < nCols; i++) {
      const n = j * nCols + i;
      nodeX[n] = gx0 + i * cs;
      nodeY[n] = gy0 + j * cs;
      inside[n] = sample(_field, w, nodeX[n], nodeY[n]) >= 0.5 ? 1 : 0;
    }
  }

  const qCols = nCols - 1;
  const qRows = nRows - 1;
  const quadOn = new Uint8Array(qCols * qRows);
  for (let j = 0; j < qRows; j++) {
    for (let i = 0; i < qCols; i++) {
      const a = j * nCols + i;
      quadOn[j * qCols + i] =
        inside[a] && inside[a + 1] && inside[a + nCols] && inside[a + nCols + 1] ? 1 : 0;
    }
  }

  // Boundary edges -> neighbour links.
  const nbA = new Int32Array(nNodes).fill(-1);
  const nbB = new Int32Array(nNodes).fill(-1);
  const nbCount = new Uint8Array(nNodes);
  const used = new Uint8Array(nNodes);
  const link = (p: number, q: number) => {
    for (const [s, t] of [
      [p, q],
      [q, p],
    ]) {
      const c = nbCount[s];
      if (c === 0) nbA[s] = t;
      else if (c === 1) nbB[s] = t;
      nbCount[s] = Math.min(255, c + 1);
    }
  };
  const qOn = (i: number, j: number) => i >= 0 && j >= 0 && i < qCols && j < qRows && quadOn[j * qCols + i] === 1;
  for (let j = 0; j < qRows; j++) {
    for (let i = 0; i < qCols; i++) {
      if (!quadOn[j * qCols + i]) continue;
      const a = j * nCols + i;
      const b = a + 1;
      const c = a + nCols;
      const d = c + 1;
      used[a] = used[b] = used[c] = used[d] = 1;
      if (!qOn(i, j - 1)) link(a, b);
      if (!qOn(i, j + 1)) link(c, d);
      if (!qOn(i - 1, j)) link(a, c);
      if (!qOn(i + 1, j)) link(b, d);
    }
  }
  const isBoundary = (n: number) => nbCount[n] > 0;

  // 1) Snap boundary nodes onto the smooth outline.
  for (let n = 0; n < nNodes; n++) {
    if (!used[n] || !isBoundary(n)) continue;
    const x0 = nodeX[n];
    const y0 = nodeY[n];
    let x = x0;
    let y = y0;
    for (let it = 0; it < 6; it++) {
      const f = sample(_field, w, x, y) - 0.5;
      const gx = 0.5 * (sample(_field, w, x + 1, y) - sample(_field, w, x - 1, y));
      const gy = 0.5 * (sample(_field, w, x, y + 1) - sample(_field, w, x, y - 1));
      const g2 = gx * gx + gy * gy;
      if (g2 < 1e-6) break;
      x -= (f * gx) / g2;
      y -= (f * gy) / g2;
      if (Math.abs(f) < 0.01) break;
    }
    let dx = x - x0;
    let dy = y - y0;
    const len = Math.hypot(dx, dy);
    const lim = 1.05 * cs;
    if (len > lim) {
      dx *= lim / len;
      dy *= lim / len;
    }
    nodeX[n] = x0 + dx;
    nodeY[n] = y0 + dy;
  }

  // 2) Smooth the outline as a curve (Taubin, no shrinking).
  const tmpX = new Float32Array(nNodes);
  const tmpY = new Float32Array(nNodes);
  for (let it = 0; it < 12; it++) {
    const k = it % 2 === 0 ? 0.5 : -0.53;
    for (let n = 0; n < nNodes; n++) {
      if (nbCount[n] !== 2) continue;
      const a = nbA[n];
      const b = nbB[n];
      tmpX[n] = nodeX[n] + k * (0.5 * (nodeX[a] + nodeX[b]) - nodeX[n]);
      tmpY[n] = nodeY[n] + k * (0.5 * (nodeY[a] + nodeY[b]) - nodeY[n]);
    }
    for (let n = 0; n < nNodes; n++) {
      if (nbCount[n] !== 2) continue;
      nodeX[n] = tmpX[n];
      nodeY[n] = tmpY[n];
    }
  }

  // 3) Even out the first rows inside the outline.
  for (let it = 0; it < 4; it++) {
    for (let n = 0; n < nNodes; n++) {
      tmpX[n] = NaN;
      if (!used[n] || isBoundary(n)) continue;
      const dist = sample(_dist, w, nodeX[n], nodeY[n]);
      if (dist > cs * 2.6) continue;
      let sx = 0;
      let sy = 0;
      let c = 0;
      for (const m of [n - 1, n + 1, n - nCols, n + nCols]) {
        if (m < 0 || m >= nNodes || !used[m]) continue;
        sx += nodeX[m];
        sy += nodeY[m];
        c++;
      }
      if (c < 3) continue;
      const k = 0.5 * (1 - dist / (cs * 3.1));
      tmpX[n] = nodeX[n] + k * (sx / c - nodeX[n]);
      tmpY[n] = nodeY[n] + k * (sy / c - nodeY[n]);
    }
    for (let n = 0; n < nNodes; n++) {
      if (Number.isNaN(tmpX[n])) continue;
      nodeX[n] = tmpX[n];
      nodeY[n] = tmpY[n];
    }
  }

  // 4) Vertices: one front + one back vertex per used node (boundary nodes share position).
  const frontIdx = new Int32Array(nNodes).fill(-1);
  let nUsed = 0;
  for (let n = 0; n < nNodes; n++) if (used[n]) frontIdx[n] = nUsed++;
  const vCount = Math.min(nUsed * 2, inputs.maxVertices);
  if (nUsed * 2 > inputs.maxVertices) {
    geometry.setIndex([]);
    return { vertexCount: 0, triangleCount: 0, landmarks };
  }

  const pos = new Float32Array(vCount * 3);
  const nor = new Float32Array(vCount * 3);
  const uv = new Float32Array(vCount * 2);
  const col = new Float32Array(vCount * 3).fill(1);
  const texelWorld = WORLD_SPAN / (w - 1);
  const zScale = texelWorld * Math.max(0, options.depthScale);

  for (let n = 0; n < nNodes; n++) {
    const f = frontIdx[n];
    if (f < 0) continue;
    const b = f + nUsed;
    const tx = nodeX[n];
    const ty = nodeY[n];
    const wx = (tx / (w - 1) - 0.5) * WORLD_SPAN;
    const wy = (0.5 - ty / (w - 1)) * WORLD_SPAN;
    const onEdge = isBoundary(n);
    const zf = onEdge ? 0 : sample(_frontD, w, tx, ty) * zScale;
    const zb = onEdge ? 0 : sample(_backD, w, tx, ty) * zScale;
    pos[f * 3] = wx;
    pos[f * 3 + 1] = wy;
    pos[f * 3 + 2] = zf;
    pos[b * 3] = wx;
    pos[b * 3 + 1] = wy;
    pos[b * 3 + 2] = -zb;

    // Texture lookup moves slightly inside near the outline, so the drawing's
    // dark outline / background never paints the rim.
    let sx = tx;
    let sy = ty;
    const dist = sample(_dist, w, tx, ty);
    const inset = Math.max(0, 1.6 - dist);
    if (inset > 0) {
      const gx = 0.5 * (sample(_field, w, tx + 1, ty) - sample(_field, w, tx - 1, ty));
      const gy = 0.5 * (sample(_field, w, tx, ty + 1) - sample(_field, w, tx, ty - 1));
      const gl = Math.hypot(gx, gy);
      if (gl > 1e-5) {
        sx += (gx / gl) * inset;
        sy += (gy / gl) * inset;
      }
    }
    const u = Math.max(0, Math.min(1, sx / (w - 1)));
    const v = Math.max(0, Math.min(1, sy / (w - 1)));
    uv[f * 2] = 0.002 + u * 0.495;
    uv[f * 2 + 1] = 0.002 + v * 0.495;
    uv[b * 2] = 0.502 + u * 0.495;
    uv[b * 2 + 1] = 0.002 + v * 0.495;
  }

  // 5) Triangles. Texel y grows downwards, world y upwards.
  let activeQuads = 0;
  for (let q = 0; q < quadOn.length; q++) activeQuads += quadOn[q];
  const index = new Uint32Array(activeQuads * 12);
  let ip = 0;
  for (let j = 0; j < qRows; j++) {
    for (let i = 0; i < qCols; i++) {
      if (!quadOn[j * qCols + i]) continue;
      const a = frontIdx[j * nCols + i];
      const b = frontIdx[j * nCols + i + 1];
      const c = frontIdx[(j + 1) * nCols + i];
      const d = frontIdx[(j + 1) * nCols + i + 1];
      // Front (+Z)
      index[ip++] = a; index[ip++] = c; index[ip++] = b;
      index[ip++] = b; index[ip++] = c; index[ip++] = d;
      // Back (-Z), reversed winding
      index[ip++] = a + nUsed; index[ip++] = b + nUsed; index[ip++] = c + nUsed;
      index[ip++] = b + nUsed; index[ip++] = d + nUsed; index[ip++] = c + nUsed;
    }
  }

  // 6) Smooth normals; front and back copies of an outline node share one normal
  //    so the rim is shaded as one continuous surface.
  for (let t = 0; t < ip; t += 3) {
    const i0 = index[t] * 3;
    const i1 = index[t + 1] * 3;
    const i2 = index[t + 2] * 3;
    const e1x = pos[i1] - pos[i0];
    const e1y = pos[i1 + 1] - pos[i0 + 1];
    const e1z = pos[i1 + 2] - pos[i0 + 2];
    const e2x = pos[i2] - pos[i0];
    const e2y = pos[i2 + 1] - pos[i0 + 1];
    const e2z = pos[i2 + 2] - pos[i0 + 2];
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    for (const vi of [i0, i1, i2]) {
      nor[vi] += nx;
      nor[vi + 1] += ny;
      nor[vi + 2] += nz;
    }
  }
  for (let n = 0; n < nNodes; n++) {
    const f = frontIdx[n];
    if (f < 0 || !isBoundary(n)) continue;
    const b = f + nUsed;
    for (let c = 0; c < 3; c++) {
      const s = nor[f * 3 + c] + nor[b * 3 + c];
      nor[f * 3 + c] = s;
      nor[b * 3 + c] = s;
    }
  }
  for (let v = 0; v < vCount; v++) {
    const l = Math.hypot(nor[v * 3], nor[v * 3 + 1], nor[v * 3 + 2]) || 1;
    nor[v * 3] /= l;
    nor[v * 3 + 1] /= l;
    nor[v * 3 + 2] /= l;
  }

  // 7) Shoes: one per leg, with a flat sole, the toe forward and the heel back.
  const shoes = options.shoes === false ? [] : buildShoes(mask, w, landmarks, texelWorld);
  let shoeVerts = 0;
  let shoeIdx = 0;
  for (const sh of shoes) {
    shoeVerts += sh.pos.length / 3;
    shoeIdx += sh.index.length;
  }
  const total = vCount + shoeVerts;
  if (total > inputs.maxVertices) shoes.length = 0;
  const allCount = shoes.length ? total : vCount;
  const P = new Float32Array(allCount * 3);
  const N = new Float32Array(allCount * 3);
  const U = new Float32Array(allCount * 2);
  const C = new Float32Array(allCount * 3).fill(1);
  const I = new Uint32Array(ip + (shoes.length ? shoeIdx : 0));
  P.set(pos);
  N.set(nor);
  U.set(uv);
  I.set(index.subarray(0, ip));
  let vb = vCount;
  let ib = ip;
  const shoeRanges: Array<{ start: number; count: number; side: -1 | 1; soleY: number; height: number }> = [];
  for (const sh of shoes) {
    const n = sh.pos.length / 3;
    P.set(sh.pos, vb * 3);
    N.set(sh.nor, vb * 3);
    U.set(sh.uv, vb * 2);
    for (let k = 0; k < sh.index.length; k++) I[ib + k] = sh.index[k] + vb;
    shoeRanges.push({ start: vb, count: n, side: sh.side, soleY: sh.soleY, height: sh.height });
    vb += n;
    ib += sh.index.length;
  }

  // Skeleton + skin weights (optional).
  let rig: HumanRig | undefined;
  if (options.rig) {
    rig = computeHumanRig(mask, w, landmarks, WORLD_SPAN);
    const maps = computeSkinWeightMaps(rig, mask, w, WORLD_SPAN);
    const skinIndex = new Uint16Array(allCount * 4);
    const skinWeight = new Float32Array(allCount * 4);
    for (let n = 0; n < nNodes; n++) {
      const f = frontIdx[n];
      if (f < 0) continue;
      sampleSkinWeights(maps, mask, w, nodeX[n], nodeY[n], skinIndex, skinWeight, f * 4);
      const b = f + nUsed;
      for (let k = 0; k < 4; k++) {
        skinIndex[b * 4 + k] = skinIndex[f * 4 + k];
        skinWeight[b * 4 + k] = skinWeight[f * 4 + k];
      }
    }
    // Shoes follow the foot, blending into the shin at the top.
    const boneIndex = (name: string) => Math.max(0, rig!.bones.findIndex((b) => b.name === name));
    for (const r of shoeRanges) {
      const sideName = r.side < 0 ? 'Right' : 'Left';
      const foot = boneIndex(`${sideName}Foot`);
      const leg = boneIndex(`${sideName}Leg`);
      for (let v = r.start; v < r.start + r.count; v++) {
        const t = (P[v * 3 + 1] - r.soleY) / Math.max(1e-6, r.height);
        const wLeg = Math.max(0, Math.min(0.5, (t - 0.55) * 1.2));
        skinIndex[v * 4] = foot;
        skinWeight[v * 4] = 1 - wLeg;
        skinIndex[v * 4 + 1] = leg;
        skinWeight[v * 4 + 1] = wLeg;
      }
    }
    geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(skinIndex, 4));
    geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(skinWeight, 4));
  } else {
    if (geometry.getAttribute('skinIndex')) geometry.deleteAttribute('skinIndex');
    if (geometry.getAttribute('skinWeight')) geometry.deleteAttribute('skinWeight');
  }

  inputs.vertexUsedMask.fill(0);
  inputs.vertexUsedMask.fill(1, 0, allCount);

  geometry.setAttribute('position', new THREE.BufferAttribute(P, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(N, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(U, 2));
  geometry.setAttribute('color', new THREE.BufferAttribute(C, 3));
  geometry.setIndex(new THREE.BufferAttribute(I, 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  return { vertexCount: allCount, triangleCount: I.length / 3, landmarks, rig };
}

interface ShoePart {
  side: -1 | 1;
  pos: Float32Array;
  nor: Float32Array;
  uv: Float32Array;
  index: Uint32Array;
  soleY: number;
  height: number;
}

/**
 * Builds a shoe for each leg: an ellipsoid stretched along Z (toe forward,
 * heel back), lower towards the toe and cut flat at the sole. It is textured
 * with the front image of the boot and covers the end of the leg.
 */
function buildShoes(mask: Uint8Array, w: number, lm: HumanBodyLandmarks, texelWorld: number): ShoePart[] {
  const H = lm.bottom - lm.top + 1;
  if (H < 20) return [];
  const ax = lm.axisX;
  const ankleY = Math.round(lm.bottom - 0.075 * H);
  const parts: ShoePart[] = [];
  const toWX = (tx: number) => (tx / (w - 1) - 0.5) * WORLD_SPAN;
  const toWY = (ty: number) => (0.5 - ty / (w - 1)) * WORLD_SPAN;

  for (const side of [-1, 1] as const) {
    // Extent of this foot in the bottom rows.
    let minX = Infinity;
    let maxX = -Infinity;
    for (let y = ankleY; y <= lm.bottom; y++) {
      const runs = rowRuns(mask, w, y);
      for (const [a, b] of runs) {
        // Keep the part of the run on this side of the axis.
        const lo = side < 0 ? a : Math.max(a, Math.ceil(ax));
        const hi = side < 0 ? Math.min(b, Math.floor(ax)) : b;
        if (hi < lo) continue;
        // Ignore runs far from the legs (e.g. a weapon on the floor).
        if (Math.abs((lo + hi) / 2 - ax) > 0.3 * H) continue;
        minX = Math.min(minX, lo);
        maxX = Math.max(maxX, hi);
      }
    }
    if (!Number.isFinite(minX) || maxX - minX < 2) continue;

    const halfW = ((maxX - minX + 1) / 2) * texelWorld;
    const cx = toWX((minX + maxX) / 2);
    const soleY = toWY(lm.bottom + 0.5);
    const height0 = 0.105 * H * texelWorld; // heel / ankle height
    const length = 0.17 * H * texelWorld;
    const heelZ = -halfW * 1.0;

    // Boot shape from a sphere: boxier section, flat sole, high heel side, low toe.
    const box = (v: number, e: number) => Math.sign(v) * Math.pow(Math.abs(v), e);
    const sphere = new THREE.SphereGeometry(1, 32, 20);
    const sp = sphere.getAttribute('position') as THREE.BufferAttribute;
    const n = sp.count;
    const pos = new Float32Array(n * 3);
    const uv = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      const ux = box(sp.getX(i), 0.55);
      const uy = box(sp.getY(i), 0.5);
      const uz = box(sp.getZ(i), 0.7);
      const t = (uz + 1) / 2; // 0 = heel, 1 = toe
      const toeT = Math.max(0, Math.min(1, (t - 0.35) / 0.65));
      const hp = 1 - 0.58 * toeT * toeT * (3 - 2 * toeT);
      const wp = 0.9 + 0.16 * Math.sin(Math.PI * Math.min(1, t * 1.1)) - 0.12 * toeT * toeT;
      const X = cx + ux * halfW * 0.98 * wp;
      const Y = soleY + ((uy + 1) / 2) * height0 * hp;
      const Z = heelZ + t * length;
      pos[i * 3] = X;
      pos[i * 3 + 1] = Y;
      pos[i * 3 + 2] = Z;
      // Front projection of the boot, kept inside the foot's own pixels.
      const tx = Math.max(minX + 1, Math.min(maxX - 1, (X / WORLD_SPAN + 0.5) * (w - 1)));
      const ty = Math.max(ankleY - 0.06 * H, Math.min(lm.bottom - 1.5, (0.5 - Y / WORLD_SPAN) * (w - 1)));
      uv[i * 2] = 0.002 + (tx / (w - 1)) * 0.495;
      uv[i * 2 + 1] = 0.002 + (ty / (w - 1)) * 0.495;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(sphere.getIndex());
    g.computeVertexNormals();
    const idx = sphere.getIndex()!;
    const index = new Uint32Array(idx.count);
    for (let k = 0; k < idx.count; k++) index[k] = idx.getX(k);
    parts.push({
      side,
      pos,
      nor: (g.getAttribute('normal') as THREE.BufferAttribute).array as Float32Array,
      uv,
      index,
      soleY,
      height: height0,
    });
    sphere.dispose();
    g.dispose();
  }
  return parts;
}

/** Debug helper: depth map (texels), core mask and landmarks for a silhouette. */
export function debugHumanDepthMap(mask: Uint8Array, w: number) {
  ensureBuffers(w);
  const landmarks = buildHumanDepthMap(mask, w);
  return { landmarks, depth: _depth.slice(), core: _coreMask.slice() };
}
