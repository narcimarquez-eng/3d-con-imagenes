import * as THREE from 'three';

// ============================================================================
// AUTOMATIC HUMAN RIG
// Places a standard humanoid skeleton inside the drawing's silhouette and
// computes skin weights, then provides a few procedural animation clips.
// Bone names follow the common humanoid convention (Hips, Spine, LeftArm…)
// so the exported .glb also works in Blender, Unity, Godot or three.js.
// "Left"/"Right" are the character's own sides: the front image shows the
// character facing the viewer, so the character's right arm is on the
// image's left (world -X).
// ============================================================================

export interface RigLandmarks {
  top: number;
  bottom: number;
  axisX: number;
  neckY: number;
  shoulderY: number;
  armpitY: number;
  crotchY: number;
}

export interface RigBone {
  name: string;
  parent: string | null;
  /** Joint position in world units (bind pose). */
  head: [number, number, number];
  /** End of the bone (child joint or tip), world units. Used for skinning. */
  tail: [number, number, number];
}

export interface HumanRig {
  bones: RigBone[];
}

export interface SkinWeightMaps {
  /** 4 bone indices per texel. */
  index: Uint16Array;
  /** 4 weights per texel (sum 1 inside the silhouette). */
  weight: Float32Array;
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

function nearestRun(runs: Run[], x: number): number {
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

/** Least-squares line x = a + b·y through (y, x) samples. */
function fitLine(samples: Array<[number, number]>): [number, number] {
  if (samples.length === 0) return [0, 0];
  if (samples.length === 1) return [samples[0][1], 0];
  let sy = 0, sx = 0, syy = 0, sxy = 0;
  for (const [y, x] of samples) {
    sy += y;
    sx += x;
    syy += y * y;
    sxy += x * y;
  }
  const n = samples.length;
  const den = n * syy - sy * sy;
  const b = Math.abs(den) < 1e-6 ? 0 : (n * sxy - sx * sy) / den;
  const a = (sx - b * sy) / n;
  return [a, b];
}

/**
 * Works out joint positions from the silhouette.
 * Coordinates are computed in texels and converted to world units
 * (same mapping as the mesh: full tensor width = `worldSpan`).
 */
export function computeHumanRig(
  mask: Uint8Array,
  w: number,
  lm: RigLandmarks,
  worldSpan: number
): HumanRig {
  const H = Math.max(1, lm.bottom - lm.top + 1);
  const ax = lm.axisX;
  const toWorld = (tx: number, ty: number, z = 0): [number, number, number] => [
    (tx / (w - 1) - 0.5) * worldSpan,
    (0.5 - ty / (w - 1)) * worldSpan,
    z,
  ];
  const texelWorld = worldSpan / (w - 1);

  const runsAt = (y: number) => rowRuns(mask, w, Math.max(0, Math.min(w - 1, Math.round(y))));

  // --- Spine ---------------------------------------------------------------
  const hipsY = lm.crotchY - 0.05 * H;
  const neckBaseY = lm.neckY + 0.5 * Math.max(2, lm.shoulderY - lm.neckY) + 0.01 * H;
  const spineY = hipsY - (hipsY - neckBaseY) * 0.33;
  const chestY = hipsY - (hipsY - neckBaseY) * 0.66;
  const headY = lm.neckY;

  // --- Arms: centre line of the arm runs hanging beside the torso ----------
  const armChain = (side: -1 | 1) => {
    const samples: Array<[number, number]> = [];
    let lastY = -1;
    let misses = 0;
    const limitY = Math.min(lm.bottom, lm.crotchY + 0.3 * H);
    for (let y = lm.armpitY; y <= limitY; y++) {
      const runs = runsAt(y);
      const k = nearestRun(runs, ax);
      const j = k + side;
      if (k >= 0 && j >= 0 && j < runs.length) {
        const [a, b] = runs[j];
        samples.push([y, (a + b) / 2]);
        lastY = y;
        misses = 0;
      } else if (samples.length > 0 && ++misses > 3) {
        break;
      }
    }
    // Fallback: arms touching the body — put them along the torso sides.
    if (samples.length < 6) {
      const r = runsAt(lm.armpitY + 0.05 * H);
      const k = nearestRun(r, ax);
      const edge = k >= 0 ? (side < 0 ? r[k][0] : r[k][1]) : ax + side * 0.15 * H;
      const x0 = edge - side * 0.03 * H;
      samples.length = 0;
      for (let y = lm.armpitY; y <= lm.armpitY + 0.32 * H; y++) samples.push([y, x0 + side * (y - lm.armpitY) * 0.15]);
      lastY = lm.armpitY + 0.32 * H;
    }
    const upper = samples.slice(0, Math.max(3, Math.round(samples.length * 0.35)));
    const [ua, ub] = fitLine(upper);
    const xAtCenter = (y: number) => {
      let best = samples[0];
      for (const s of samples) if (Math.abs(s[0] - y) < Math.abs(best[0] - y)) best = s;
      return best[1];
    };

    const shoulderJY = lm.shoulderY + 0.45 * (lm.armpitY - lm.shoulderY);
    const shoulderJX = ua + ub * shoulderJY;
    const handEndY = lastY;
    const wristY = handEndY - 0.075 * H;
    const elbowY = (shoulderJY + wristY) / 2;
    const elbowX = elbowY < lm.armpitY ? ua + ub * elbowY : xAtCenter(elbowY);
    return {
      clavicle: toWorld(ax + side * 0.02 * H, neckBaseY + 0.02 * H),
      shoulder: toWorld(shoulderJX, shoulderJY),
      elbow: toWorld(elbowX, elbowY, -0.01 * H * texelWorld),
      wrist: toWorld(xAtCenter(wristY), wristY),
      handEnd: toWorld(xAtCenter(handEndY), handEndY),
    };
  };

  // --- Legs: centre of each leg below the crotch ---------------------------
  const legX = (side: -1 | 1, y: number) => {
    const runs = runsAt(y);
    const k = nearestRun(runs, ax);
    if (k < 0) return ax + side * 0.06 * H;
    const [a, b] = runs[k];
    if (a < ax && b > ax) {
      // One run spanning both legs: take this side's half.
      return side < 0 ? (a + ax) / 2 : (ax + b) / 2;
    }
    // Separate runs: the one on this side closest to the axis.
    let best = -1;
    let bestD = Infinity;
    for (let j = 0; j < runs.length; j++) {
      const c = (runs[j][0] + runs[j][1]) / 2;
      if ((c - ax) * side <= 0) continue;
      const d = Math.abs(c - ax);
      if (d < bestD) {
        bestD = d;
        best = j;
      }
    }
    return best >= 0 ? (runs[best][0] + runs[best][1]) / 2 : ax + side * 0.06 * H;
  };
  const legChain = (side: -1 | 1) => {
    const hipY = lm.crotchY - 0.03 * H;
    const ankleY = lm.bottom - 0.06 * H;
    const kneeY = (hipY + ankleY) / 2;
    // Toe of a normal foot (15.5% of the height) with the ankle a quarter from the heel.
    const footLen = 0.155 * 0.75 * H * texelWorld;
    return {
      hip: toWorld(legX(side, lm.crotchY + 0.03 * H), hipY),
      knee: toWorld(legX(side, kneeY), kneeY, 0.01 * H * texelWorld),
      ankle: toWorld(legX(side, ankleY), ankleY),
      toe: toWorld(legX(side, lm.bottom), lm.bottom, footLen),
    };
  };

  const R = armChain(-1);
  const L = armChain(1);
  const RL = legChain(-1);
  const LL = legChain(1);

  const hips = toWorld(ax, hipsY);
  const spine = toWorld(ax, spineY);
  const chest = toWorld(ax, chestY);
  const neck = toWorld(ax, neckBaseY);
  const head = toWorld(ax, headY);
  const headTop = toWorld(ax, lm.top);

  const bones: RigBone[] = [
    { name: 'Hips', parent: null, head: hips, tail: spine },
    { name: 'Spine', parent: 'Hips', head: spine, tail: chest },
    { name: 'Chest', parent: 'Spine', head: chest, tail: neck },
    { name: 'Neck', parent: 'Chest', head: neck, tail: head },
    { name: 'Head', parent: 'Neck', head: head, tail: headTop },
    { name: 'RightShoulder', parent: 'Chest', head: R.clavicle, tail: R.shoulder },
    { name: 'RightArm', parent: 'RightShoulder', head: R.shoulder, tail: R.elbow },
    { name: 'RightForeArm', parent: 'RightArm', head: R.elbow, tail: R.wrist },
    { name: 'RightHand', parent: 'RightForeArm', head: R.wrist, tail: R.handEnd },
    { name: 'LeftShoulder', parent: 'Chest', head: L.clavicle, tail: L.shoulder },
    { name: 'LeftArm', parent: 'LeftShoulder', head: L.shoulder, tail: L.elbow },
    { name: 'LeftForeArm', parent: 'LeftArm', head: L.elbow, tail: L.wrist },
    { name: 'LeftHand', parent: 'LeftForeArm', head: L.wrist, tail: L.handEnd },
    { name: 'RightUpLeg', parent: 'Hips', head: RL.hip, tail: RL.knee },
    { name: 'RightLeg', parent: 'RightUpLeg', head: RL.knee, tail: RL.ankle },
    { name: 'RightFoot', parent: 'RightLeg', head: RL.ankle, tail: RL.toe },
    { name: 'LeftUpLeg', parent: 'Hips', head: LL.hip, tail: LL.knee },
    { name: 'LeftLeg', parent: 'LeftUpLeg', head: LL.knee, tail: LL.ankle },
    { name: 'LeftFoot', parent: 'LeftLeg', head: LL.ankle, tail: LL.toe },
  ];
  return { bones };
}

const K5 = [0.0625, 0.25, 0.375, 0.25, 0.0625];

/**
 * Skin weights per texel.
 * 1) Every silhouette pixel is bound to the bone it reaches first when walking
 *    *inside* the silhouette from all bones at once, so the side of the torso
 *    goes to the spine and not to the arm hanging next to it.
 * 2) Each bone's region is blurred so joints bend smoothly.
 * 3) A pixel only keeps weights of its own bone, its parent and its children.
 */
export function computeSkinWeightMaps(
  rig: HumanRig,
  mask: Uint8Array,
  w: number,
  worldSpan: number
): SkinWeightMaps {
  const n = w * w;
  const nb = rig.bones.length;
  const toTex = (p: [number, number, number]): [number, number] => [
    (p[0] / worldSpan + 0.5) * (w - 1),
    (0.5 - p[1] / worldSpan) * (w - 1),
  ];

  const label = new Int16Array(n).fill(-1);
  const queue = new Int32Array(n);
  let qh = 0;
  let qt = 0;

  // Seed: rasterise every bone segment (inside the silhouette).
  for (let b = 0; b < nb; b++) {
    const [x0, y0] = toTex(rig.bones[b].head);
    const [x1, y1] = toTex(rig.bones[b].tail);
    const steps = Math.max(2, Math.ceil(Math.hypot(x1 - x0, y1 - y0) * 2));
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      // Leave the very ends to neighbouring bones so joints sit between regions.
      if (t < 0.08 && rig.bones[b].parent !== null) continue;
      const x = Math.round(x0 + (x1 - x0) * t);
      const y = Math.round(y0 + (y1 - y0) * t);
      if (x < 0 || y < 0 || x >= w || y >= w) continue;
      const i = y * w + x;
      if (mask[i] !== 1 || label[i] >= 0) continue;
      label[i] = b;
      queue[qt++] = i;
    }
  }
  // Geodesic flood inside the silhouette.
  while (qh < qt) {
    const i = queue[qh++];
    const x = i % w;
    const y = (i / w) | 0;
    const l = label[i];
    if (x > 0 && mask[i - 1] === 1 && label[i - 1] < 0) { label[i - 1] = l; queue[qt++] = i - 1; }
    if (x < w - 1 && mask[i + 1] === 1 && label[i + 1] < 0) { label[i + 1] = l; queue[qt++] = i + 1; }
    if (y > 0 && mask[i - w] === 1 && label[i - w] < 0) { label[i - w] = l; queue[qt++] = i - w; }
    if (y < w - 1 && mask[i + w] === 1 && label[i + w] < 0) { label[i + w] = l; queue[qt++] = i + w; }
  }
  // Pixels not reached (islands) go to the Hips.
  for (let i = 0; i < n; i++) if (mask[i] === 1 && label[i] < 0) label[i] = 0;

  // Parent / children lookup.
  const parentIdx = rig.bones.map((b) => rig.bones.findIndex((p) => p.name === b.parent));
  const related: number[][] = rig.bones.map((_, b) => {
    const set = [b];
    if (parentIdx[b] >= 0) set.push(parentIdx[b]);
    for (let c = 0; c < nb; c++) if (parentIdx[c] === b) set.push(c);
    return set;
  });

  // Blur each bone's region.
  const maps: Float32Array[] = [];
  const tmp = new Float32Array(n);
  const passes = 5;
  for (let b = 0; b < nb; b++) {
    const m = new Float32Array(n);
    for (let i = 0; i < n; i++) m[i] = label[i] === b ? 1 : 0;
    for (let p = 0; p < passes; p++) {
      for (let y = 0; y < w; y++) {
        for (let x = 0; x < w; x++) {
          let s = 0;
          for (let o = -2; o <= 2; o++) {
            const sx = x + o < 0 ? 0 : x + o >= w ? w - 1 : x + o;
            s += m[y * w + sx] * K5[o + 2];
          }
          tmp[y * w + x] = s;
        }
      }
      for (let y = 0; y < w; y++) {
        for (let x = 0; x < w; x++) {
          let s = 0;
          for (let o = -2; o <= 2; o++) {
            const sy = y + o < 0 ? 0 : y + o >= w ? w - 1 : y + o;
            s += tmp[sy * w + x] * K5[o + 2];
          }
          m[y * w + x] = s;
        }
      }
    }
    maps.push(m);
  }

  const index = new Uint16Array(n * 4);
  const weight = new Float32Array(n * 4);
  const cand: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    const l = label[i];
    if (l < 0) continue;
    cand.length = 0;
    for (const b of related[l]) cand.push([b, maps[b][i]]);
    cand.sort((p, q) => q[1] - p[1]);
    let sum = 0;
    for (let k = 0; k < 4 && k < cand.length; k++) sum += cand[k][1];
    if (sum <= 1e-6) {
      index[i * 4] = l;
      weight[i * 4] = 1;
      continue;
    }
    for (let k = 0; k < 4; k++) {
      if (k < cand.length) {
        index[i * 4 + k] = cand[k][0];
        weight[i * 4 + k] = cand[k][1] / sum;
      }
    }
  }
  return { index, weight };
}

/** Looks up skin weights for a texel position, searching nearby if it lies just outside the silhouette. */
export function sampleSkinWeights(
  maps: SkinWeightMaps,
  mask: Uint8Array,
  w: number,
  tx: number,
  ty: number,
  outIndex: Uint16Array | number[],
  outWeight: Float32Array | number[],
  outOffset: number
) {
  const cx = Math.round(tx);
  const cy = Math.round(ty);
  for (let r = 0; r <= 4; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        const x = cx + dx;
        const y = cy + dy;
        if (x < 0 || y < 0 || x >= w || y >= w) continue;
        const i = y * w + x;
        if (mask[i] !== 1) continue;
        for (let k = 0; k < 4; k++) {
          outIndex[outOffset + k] = maps.index[i * 4 + k];
          outWeight[outOffset + k] = maps.weight[i * 4 + k];
        }
        return;
      }
    }
  }
  outIndex[outOffset] = 0;
  outWeight[outOffset] = 1;
}

// ============================================================================
// SKELETON + ANIMATIONS (three.js objects)
// ============================================================================

export const RIG_ANIMATIONS = [
  { id: 'idle', label: 'Reposo' },
  { id: 'walk', label: 'Caminar' },
  { id: 'run', label: 'Correr' },
  { id: 'wave', label: 'Saludar' },
] as const;

export type RigAnimationId = (typeof RIG_ANIMATIONS)[number]['id'];

/** Creates THREE.Bones (bind pose = identity rotations, joint offsets as positions). */
export function createSkeletonBones(rig: HumanRig): { root: THREE.Bone; bones: THREE.Bone[] } {
  const byName = new Map<string, THREE.Bone>();
  const bones: THREE.Bone[] = [];
  for (const b of rig.bones) {
    const bone = new THREE.Bone();
    bone.name = b.name;
    byName.set(b.name, bone);
    bones.push(bone);
  }
  let root = bones[0];
  rig.bones.forEach((b, i) => {
    const bone = bones[i];
    if (b.parent) {
      const parent = byName.get(b.parent)!;
      const ph = rig.bones.find((p) => p.name === b.parent)!.head;
      bone.position.set(b.head[0] - ph[0], b.head[1] - ph[1], b.head[2] - ph[2]);
      parent.add(bone);
    } else {
      bone.position.set(b.head[0], b.head[1], b.head[2]);
      root = bone;
    }
    // Remembered so exports can put the skeleton back in its rest pose.
    bone.userData.bindPosition = bone.position.toArray();
  });
  return { root, bones };
}

const DEG = Math.PI / 180;

/**
 * Procedural animation clips for the rig: idle, walk, run and wave.
 * Bones start with identity rotation, so each track is just the motion.
 */
export function createHumanAnimationClips(rig: HumanRig): THREE.AnimationClip[] {
  const boneHead = new Map(rig.bones.map((b) => [b.name, b]));
  const hipsPos = boneHead.get('Hips')!.head;
  const height = Math.max(0.5, boneHead.get('Head')!.tail[1] - Math.min(rig.bones.find((b) => b.name === 'RightFoot')!.tail[1], 0));

  // Angle (around Z) that turns an arm from its rest direction to "raised up and out".
  const raiseAngle = (armName: string, side: -1 | 1) => {
    const b = boneHead.get(armName)!;
    const dx = b.tail[0] - b.head[0];
    const dy = b.tail[1] - b.head[1];
    const rest = Math.atan2(dy, dx);
    const target = Math.atan2(0.92, side * 0.4);
    let a = target - rest;
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a < -Math.PI) a += 2 * Math.PI;
    return a;
  };

  type Pose = Record<string, [number, number, number]>; // Euler XYZ in radians
  const build = (
    name: string,
    duration: number,
    samples: number,
    pose: (p: number) => { rot: Pose; hipsOffset?: [number, number, number] }
  ) => {
    const times: number[] = [];
    const rotValues = new Map<string, number[]>();
    const posValues: number[] = [];
    const q = new THREE.Quaternion();
    const e = new THREE.Euler();
    for (let s = 0; s <= samples; s++) {
      const p = s / samples;
      times.push(p * duration);
      const { rot, hipsOffset } = pose(p % 1);
      for (const b of rig.bones) {
        const r = rot[b.name] || [0, 0, 0];
        e.set(r[0], r[1], r[2], 'XYZ');
        q.setFromEuler(e);
        if (!rotValues.has(b.name)) rotValues.set(b.name, []);
        rotValues.get(b.name)!.push(q.x, q.y, q.z, q.w);
      }
      const o = hipsOffset || [0, 0, 0];
      posValues.push(hipsPos[0] + o[0], hipsPos[1] + o[1], hipsPos[2] + o[2]);
    }
    const tracks: THREE.KeyframeTrack[] = [];
    for (const [bone, values] of rotValues) {
      // Skip bones that never move.
      let moves = false;
      for (let i = 0; i < values.length; i += 4) if (Math.abs(values[i + 3] - 1) > 1e-5) moves = true;
      if (!moves) continue;
      tracks.push(new THREE.QuaternionKeyframeTrack(`${bone}.quaternion`, times, values));
    }
    tracks.push(new THREE.VectorKeyframeTrack('Hips.position', times, posValues));
    return new THREE.AnimationClip(name, duration, tracks);
  };

  const TWO_PI = Math.PI * 2;

  const idle = build('Reposo', 4, 40, (p) => {
    const b = Math.sin(TWO_PI * p);
    return {
      rot: {
        Chest: [2 * DEG * b, 0, 0],
        Spine: [0, 0, 1.2 * DEG * Math.sin(TWO_PI * p + 1)],
        Head: [-1.5 * DEG * b, 4 * DEG * Math.sin(TWO_PI * p * 0.5 * 2 + 0.5), 0],
        RightArm: [0, 0, -2 * DEG * b],
        LeftArm: [0, 0, 2 * DEG * b],
        RightForeArm: [-6 * DEG, 0, 0],
        LeftForeArm: [-6 * DEG, 0, 0],
      },
      hipsOffset: [0, 0.004 * height * b, 0],
    };
  });

  const gait = (name: string, duration: number, amp: number, knee: number, elbow: number, lean: number, bob: number) =>
    build(name, duration, 32, (p) => {
      const s = Math.sin(TWO_PI * p);
      const c = Math.cos(TWO_PI * p);
      // Knee bends while the leg swings forward.
      const kneeR = knee * Math.max(0, Math.sin(TWO_PI * p + 0.9)) + 5 * DEG;
      const kneeL = knee * Math.max(0, Math.sin(TWO_PI * p + 0.9 + Math.PI)) + 5 * DEG;
      return {
        rot: {
          Hips: [0, 5 * DEG * s * amp, 0],
          Spine: [lean * 0.5, -3 * DEG * s * amp, 0],
          Chest: [lean * 0.5, -4 * DEG * s * amp, 0],
          Head: [-lean * 0.6, 3 * DEG * s * amp, 0],
          RightUpLeg: [-26 * DEG * amp * s, 0, 0],
          LeftUpLeg: [26 * DEG * amp * s, 0, 0],
          RightLeg: [kneeR, 0, 0],
          LeftLeg: [kneeL, 0, 0],
          RightFoot: [-8 * DEG * amp * s, 0, 0],
          LeftFoot: [8 * DEG * amp * s, 0, 0],
          RightArm: [22 * DEG * amp * s, 0, 0],
          LeftArm: [-22 * DEG * amp * s, 0, 0],
          RightForeArm: [-elbow - 10 * DEG * amp * Math.max(0, -s), 0, 0],
          LeftForeArm: [-elbow - 10 * DEG * amp * Math.max(0, s), 0, 0],
        },
        hipsOffset: [0, bob * height * Math.abs(c) - bob * height * 0.5, 0],
      };
    });

  const walk = gait('Caminar', 1.1, 1, 38 * DEG, 12 * DEG, 3 * DEG, 0.012);
  const run = gait('Correr', 0.68, 1.7, 75 * DEG, 70 * DEG, 12 * DEG, 0.03);

  const raiseR = raiseAngle('RightArm', -1);
  const wave = build('Saludar', 2, 48, (p) => {
    // Raise in the first 20%, wave, lower in the last 15%.
    const up = p < 0.2 ? Math.sin((p / 0.2) * Math.PI * 0.5) : p > 0.85 ? Math.cos(((p - 0.85) / 0.15) * Math.PI * 0.5) : 1;
    const waveT = Math.sin(TWO_PI * p * 4);
    const b = Math.sin(TWO_PI * p);
    return {
      rot: {
        Chest: [1.5 * DEG * b, 0, -2 * DEG * up],
        Head: [0, -6 * DEG * up, 3 * DEG * up],
        RightShoulder: [0, 0, -8 * DEG * up],
        RightArm: [0, 0, raiseR * up],
        RightForeArm: [0, 0, (-20 * DEG + 22 * DEG * waveT) * up],
        RightHand: [0, 0, 10 * DEG * waveT * up],
        LeftArm: [0, 0, 2 * DEG * b],
        LeftForeArm: [-6 * DEG, 0, 0],
      },
    };
  });

  return [idle, walk, run, wave];
}

export function clipNameForAnimation(id: string): string | null {
  const found = RIG_ANIMATIONS.find((a) => a.id === id);
  return found ? found.label : null;
}
