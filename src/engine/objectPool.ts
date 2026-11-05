import * as THREE from 'three';

/**
 * Pre-allocated module-scope math registers.
 * Eliminates `new THREE.Vector3()`, `new THREE.Matrix4()`, and `new THREE.Color()`
 * inside the main animation loop and spatial brush raycasting routines.
 */
export const _vec3A = new THREE.Vector3();
export const _vec3B = new THREE.Vector3();
export const _vec3C = new THREE.Vector3();
export const _normalA = new THREE.Vector3();
export const _mat4A = new THREE.Matrix4();
export const _quatA = new THREE.Quaternion();
export const _scaleVec = new THREE.Vector3();
export const _colorA = new THREE.Color();
export const _colorB = new THREE.Color();
export const _eulerA = new THREE.Euler();

/**
 * Pooled Sculpt / Deform operation node to prevent heap allocations during
 * continuous pointermove 3D sculpting strokes.
 */
export interface PooledVertexDelta {
  vertexIndex: number;
  dx: number;
  dy: number;
  dz: number;
  weight: number;
}

export class VertexDeltaPool {
  private readonly pool: PooledVertexDelta[];
  private activeCount = 0;
  public totalReused = 0;

  constructor(capacity: number = 16384) {
    this.pool = new Array(capacity);
    for (let i = 0; i < capacity; i++) {
      this.pool[i] = { vertexIndex: 0, dx: 0, dy: 0, dz: 0, weight: 0 };
    }
  }

  public acquire(
    vertexIndex: number,
    dx: number,
    dy: number,
    dz: number,
    weight: number
  ): PooledVertexDelta | null {
    if (this.activeCount >= this.pool.length) {
      return null;
    }
    const item = this.pool[this.activeCount++];
    item.vertexIndex = vertexIndex;
    item.dx = dx;
    item.dy = dy;
    item.dz = dz;
    item.weight = weight;
    this.totalReused++;
    return item;
  }

  public getActiveBuffer(): readonly PooledVertexDelta[] {
    return this.pool;
  }

  public getActiveLength(): number {
    return this.activeCount;
  }

  public releaseAll(): void {
    this.activeCount = 0;
  }
}

/**
 * Spatial math utilities using squared Euclidean distance (distSq)
 * to avoid expensive `Math.sqrt()` calls in inner vertex loops, collision checks, and LOD evaluation.
 */
export function distSq3D(
  x1: number,
  y1: number,
  z1: number,
  x2: number,
  y2: number,
  z2: number
): number {
  const dx = x1 - x2;
  const dy = y1 - y2;
  const dz = z1 - z2;
  return dx * dx + dy * dy + dz * dz;
}

/**
 * Smooth polynomial C2 falloff kernel (Wyvill kernel) evaluated strictly from squared distance:
 * W(d^2) = (1 - d^2 / R^2)^3 for d^2 < R^2, else 0.
 * Zero Math.sqrt calls required.
 */
export function wyvillFalloffFromDistSq(distSq: number, radiusSq: number): number {
  if (distSq >= radiusSq || radiusSq <= 1e-8) return 0;
  const t = 1.0 - distSq / radiusSq;
  return t * t * t;
}

/**
 * Reference code block showcasing the refactored Zero-GC main loop & InstancedMesh spatial batching.
 */
export const OPTIMIZED_LOOP_REFERENCE_CODE = `// ============================================================================
// MOTOR DE OPTIMIZACIÓN ZERO-GC (THREE.JS / JAVASCRIPT)
// Eliminado del bucle: new THREE.Vector3(), new THREE.Matrix4(), new THREE.Color(),
// Math.sqrt() en colisiones/LOD, y draw calls individuales reemplazados por InstancedMesh.
// ============================================================================

// 1. PRE-ASIGNACIÓN FUERA DEL BUCLE (Registros estáticos reutilizables en módulo)
const _pos = new THREE.Vector3();
const _camPos = new THREE.Vector3();
const _normal = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _mat4 = new THREE.Matrix4();
const _color = new THREE.Color();

// Umbrales espaciales pre-calculados al cuadrado (evita Math.sqrt por entidad)
const LOD_NEAR_DIST_SQ = 12.0 * 12.0; // 144.0
const LOD_FAR_DIST_SQ  = 32.0 * 32.0; // 1024.0
const COLLISION_RAD_SQ = 0.85 * 0.85; // 0.7225

// 2. OBJECT POOLING EN MEMORIA CONTIGUA (TypedArrays sin Garbage Collector)
class EntityPool {
  constructor(maxCount) {
    this.maxCount = maxCount;
    this.activeCount = 0;
    this.positions = new Float32Array(maxCount * 3);
    this.velocities = new Float32Array(maxCount * 3);
    this.normals = new Float32Array(maxCount * 3);
    this.scales = new Float32Array(maxCount);
  }
  spawn(x, y, z, nx, ny, nz, scale) {
    if (this.activeCount >= this.maxCount) return -1;
    const i = this.activeCount++;
    const i3 = i * 3;
    this.positions[i3] = x; this.positions[i3 + 1] = y; this.positions[i3 + 2] = z;
    this.normals[i3] = nx;  this.normals[i3 + 1] = ny;  this.normals[i3 + 2] = nz;
    this.scales[i] = scale;
    return i;
  }
  release(index) {
    // Swap-and-pop O(1) sin splice() ni reasignación de arrays
    const last = --this.activeCount;
    if (index !== last) {
      const i3 = index * 3, l3 = last * 3;
      this.positions[i3] = this.positions[l3];
      this.positions[i3 + 1] = this.positions[l3 + 1];
      this.positions[i3 + 2] = this.positions[l3 + 2];
      this.scales[index] = this.scales[last];
    }
  }
}

// 3. BATCHING CON INSTANCEDMESH (1 único Draw Call para N entidades)
function updateMainLoopZeroGC(dt, pool, instancedMesh, camera, brushHitPoint) {
  _camPos.copy(camera.position);
  const count = pool.activeCount;
  const posArr = pool.positions;
  const normArr = pool.normals;
  const scaleArr = pool.scales;

  for (let i = 0; i < count; i++) {
    const i3 = i * 3;
    let px = posArr[i3], py = posArr[i3 + 1], pz = posArr[i3 + 2];

    // 4. CÁLCULO ESPACIAL CON DISTANCIA AL CUADRADO (distSq) — Cero Math.sqrt()
    const dxCam = px - _camPos.x;
    const dyCam = py - _camPos.y;
    const dzCam = pz - _camPos.z;
    const distCamSq = dxCam * dxCam + dyCam * dyCam + dzCam * dzCam;

    // LOD dinámico por umbrales cuadráticos
    let lodFactor = 1.0;
    if (distCamSq > LOD_FAR_DIST_SQ) {
      lodFactor = 0.0; // Culling suave de instancias lejanas
    } else if (distCamSq > LOD_NEAR_DIST_SQ) {
      lodFactor = 1.0 - (distCamSq - LOD_NEAR_DIST_SQ) / (LOD_FAR_DIST_SQ - LOD_NEAR_DIST_SQ);
    }

    // Colisión espacial con el pincel 3D usando kernel polinomial C2 sobre distSq
    if (brushHitPoint) {
      const bx = px - brushHitPoint.x;
      const by = py - brushHitPoint.y;
      const bz = pz - brushHitPoint.z;
      const brushDistSq = bx * bx + by * by + bz * bz;
      if (brushDistSq < COLLISION_RAD_SQ) {
        const falloff = 1.0 - brushDistSq / COLLISION_RAD_SQ;
        const impulse = falloff * falloff * dt * 0.5;
        px += normArr[i3] * impulse;
        py += normArr[i3 + 1] * impulse;
        pz += normArr[i3 + 2] * impulse;
        posArr[i3] = px; posArr[i3 + 1] = py; posArr[i3 + 2] = pz;
      }
    }

    // Composición directa sobre _mat4 preasignada (sin crear objetos temporales)
    _pos.set(px, py, pz);
    const s = scaleArr[i] * lodFactor;
    _scale.set(s, s, s);
    _mat4.compose(_pos, _quat, _scale);
    instancedMesh.setMatrixAt(i, _mat4);
  }

  instancedMesh.count = count;
  instancedMesh.instanceMatrix.needsUpdate = true;
}`;
