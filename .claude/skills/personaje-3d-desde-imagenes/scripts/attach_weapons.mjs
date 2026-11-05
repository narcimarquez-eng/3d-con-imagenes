#!/usr/bin/env node
// Attaches weapon pieces to the hands of a rigged body GLB, and writes a normal
// and a reduced GLB. Weapon files are static glTF files with one node per piece.
//
// Usage (needs @gltf-transform/core and the gltf-transform CLI, see SKILL):
//   node attach_weapons.mjs --body cuerpo.glb --weapon arma.glb --out salida
//     --right 1-16 [--right-grip x,y,z] [--right-rot 180]
//     --left 17-34 [--left-center x,y,z] [--left-rot 90]
//     [--gltf-transform ./node_modules/.bin/gltf-transform]
//
// Piece numbers are the node order in the weapon file (the first node is the
// root, so pieces start at 1). A piece range is "lo-hi" (inclusive).
// The anchor (grip for the right hand, centre for the left) goes to the hand
// bone; the piece is rotated around X by --right-rot / --left-rot degrees first.
// Output: <salida>_normal.glb and <salida>_reducido.glb (25 % of the triangles,
// 256 px textures, no meshopt, for games).
import { NodeIO } from '@gltf-transform/core';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const need = (name) => {
  const v = opt(name);
  if (v === undefined) throw new Error(`falta --${name}`);
  return v;
};
const parseRange = (s) => {
  const [lo, hi] = s.split('-').map(Number);
  return [lo, hi];
};
const parseVec = (s) => s.split(',').map(Number);
const degToQuat = (deg) => {
  const h = (deg * Math.PI) / 360;
  return [Math.sin(h), 0, 0, Math.cos(h)]; // rotation around X
};
const rotX = (p, deg) => {
  const r = (deg * Math.PI) / 180;
  const c = Math.cos(r);
  const s = Math.sin(r);
  return [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c];
};

const bodyPath = need('body');
const weaponPath = need('weapon');
const outBase = need('out');
const gt = opt('gltf-transform', './node_modules/.bin/gltf-transform');

const io = new NodeIO();
const doc = await io.read(bodyPath);
const weapon = await io.read(weaponPath);
const texCache = new Map();
const matCache = new Map();

const copyTex = (t) => {
  if (!t) return null;
  if (texCache.has(t)) return texCache.get(t);
  const n = doc.createTexture(t.getName() || 'tex').setImage(t.getImage().slice()).setMimeType(t.getMimeType());
  texCache.set(t, n);
  return n;
};
const copyMat = (m) => {
  if (!m) return null;
  if (matCache.has(m)) return matCache.get(m);
  const n = doc.createMaterial(m.getName() || 'mat')
    .setBaseColorFactor(m.getBaseColorFactor())
    .setMetallicFactor(m.getMetallicFactor())
    .setRoughnessFactor(m.getRoughnessFactor())
    .setEmissiveFactor(m.getEmissiveFactor())
    .setAlphaMode(m.getAlphaMode())
    .setDoubleSided(m.getDoubleSided());
  if (m.getBaseColorTexture()) n.setBaseColorTexture(copyTex(m.getBaseColorTexture()));
  if (m.getMetallicRoughnessTexture()) n.setMetallicRoughnessTexture(copyTex(m.getMetallicRoughnessTexture()));
  if (m.getNormalTexture()) n.setNormalTexture(copyTex(m.getNormalTexture()));
  if (m.getEmissiveTexture()) n.setEmissiveTexture(copyTex(m.getEmissiveTexture()));
  if (m.getOcclusionTexture()) n.setOcclusionTexture(copyTex(m.getOcclusionTexture()));
  matCache.set(m, n);
  return n;
};
const copyMesh = (mesh, name) => {
  const out = doc.createMesh(name);
  for (const p of mesh.listPrimitives()) {
    const np = doc.createPrimitive().setMode(p.getMode()).setMaterial(copyMat(p.getMaterial()));
    for (const sem of p.listSemantics()) {
      const src = p.getAttribute(sem);
      np.setAttribute(sem, doc.createAccessor(sem).setType(src.getType()).setArray(src.getArray().slice()).setNormalized(src.getNormalized()));
    }
    if (p.getIndices()) np.setIndices(doc.createAccessor('idx').setType('SCALAR').setArray(p.getIndices().getArray().slice()));
    out.addPrimitive(np);
  }
  return out;
};

// Pieces of the weapon, numbered by node order (root = 0).
const wNodes = weapon.getRoot().listNodes();
const piecesIn = (range) => {
  const [lo, hi] = parseRange(range);
  return wNodes.map((n, i) => [n, i]).filter(([n, i]) => n.getMesh() && i >= lo && i <= hi).map(([n, i]) => [n.getMesh(), i]);
};
// Centre of the pieces' bounding box, in weapon coordinates.
const boxCenter = (parts) => {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const [mesh] of parts) for (const p of mesh.listPrimitives()) {
    const pos = p.getAttribute('POSITION');
    const mn = pos.getMin([]);
    const mx = pos.getMax([]);
    for (let i = 0; i < 3; i++) { min[i] = Math.min(min[i], mn[i]); max[i] = Math.max(max[i], mx[i]); }
  }
  return min.map((v, i) => (v + max[i]) / 2);
};

const bodyNodes = doc.getRoot().listNodes();
const findNode = (name) => bodyNodes.find((n) => n.getName() === name);

const attach = (handName, label, parts, anchor, deg) => {
  const hand = findNode(handName);
  if (!hand) throw new Error(`no existe el hueso ${handName}`);
  const R = rotX(anchor, deg);
  const t = [-R[0], -R[1], -R[2]];
  const group = doc.createNode(label).setTranslation(t).setRotation(degToQuat(deg)).setScale([1, 1, 1]);
  for (const [mesh, i] of parts) group.addChild(doc.createNode(`${label}_${i}`).setMesh(copyMesh(mesh, `${label}_${i}`)));
  hand.addChild(group);
};

const rightRange = opt('right');
const leftRange = opt('left');
if (rightRange) {
  const parts = piecesIn(rightRange);
  const anchor = opt('right-grip') ? parseVec(opt('right-grip')) : boxCenter(parts);
  attach('RightHand', 'Arma_derecha', parts, anchor, Number(opt('right-rot', '180')));
  console.log(`derecha: piezas ${rightRange} (${parts.length} mallas) ancla ${anchor.map((v) => v.toFixed(3))}`);
}
if (leftRange) {
  const parts = piecesIn(leftRange);
  const anchor = opt('left-center') ? parseVec(opt('left-center')) : boxCenter(parts);
  attach('LeftHand', 'Arma_izquierda', parts, anchor, Number(opt('left-rot', '90')));
  console.log(`izquierda: piezas ${leftRange} (${parts.length} mallas) ancla ${anchor.map((v) => v.toFixed(3))}`);
}

const normal = `${outBase}_normal.glb`;
const reduced = `${outBase}_reducido.glb`;
await io.write(normal, doc);
console.log('escrito', normal);

// Reduced: fewer triangles, smaller textures, no meshopt (standard glTF for games).
if (!existsSync(gt)) {
  console.log(`no encuentro el CLI de gltf-transform en ${gt}; no genero el reducido`);
} else {
  const steps = [
    ['simplify', normal, `${outBase}_s1.glb`, '--ratio', '0.25', '--error', '0.01'],
    ['resize', `${outBase}_s1.glb`, `${outBase}_s2.glb`, '--width', '256', '--height', '256'],
    ['prune', `${outBase}_s2.glb`, `${outBase}_s3.glb`],
    ['dedup', `${outBase}_s3.glb`, reduced],
  ];
  for (const [cmd, input, output, ...extra] of steps) execFileSync(gt, [cmd, input, output, ...extra], { stdio: 'pipe' });
  console.log('escrito', reduced);
}
