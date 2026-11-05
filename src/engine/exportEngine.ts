import * as THREE from 'three';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';
import {
  ATLAS_TEX_SIZE,
  CNN_TENSOR_SIZE,
  CNNReconstructionBuffers,
  DEFAULT_ZONES,
  PRESET_REFERENCE_IMAGES,
  buildReconstructedMeshGeometry,
  renderPresetImageToCanvas,
  runCNNReconstructionPipeline,
} from './cnnReconstruction';
import {
  NormalReconstructionMode,
  ReconstructionParams,
  ZoneConfig,
} from '../types/studio';

/**
 * Triggers a browser download for a Blob or ArrayBuffer.
 */
export function uint8ToBlob(bytes: Uint8Array, mimeType: string): Blob {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Blob([copy.buffer], { type: mimeType });
}

export function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.style.display = 'none';
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(url), 3000);
}

/**
 * Converts an HTMLCanvasElement into a PNG Blob.
 */
export function canvasToPNGBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('No se pudo codificar la imagen PNG.'));
    }, 'image/png');
  });
}

/**
 * Converts an HTMLCanvasElement into a PNG Uint8Array.
 */
export async function canvasToPNGBytes(
  canvas: HTMLCanvasElement
): Promise<Uint8Array> {
  const blob = await canvasToPNGBlob(canvas);
  const buf = await blob.arrayBuffer();
  return new Uint8Array(buf);
}

/**
 * Renders the RGB Input Tensor (256x256) into a dedicated Canvas for export.
 */
export function createRGBCanvasFromBuffers(
  buffers: CNNReconstructionBuffers
): HTMLCanvasElement {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;

  const imgData = ctx.createImageData(w, h);
  const data = imgData.data;
  const len = w * h;

  for (let i = 0; i < len; i++) {
    const i3 = i * 3;
    const i4 = i * 4;
    data[i4] = Math.min(255, Math.max(0, Math.round(buffers.rgbInput[i3] * 255)));
    data[i4 + 1] = Math.min(
      255,
      Math.max(0, Math.round(buffers.rgbInput[i3 + 1] * 255))
    );
    data[i4 + 2] = Math.min(
      255,
      Math.max(0, Math.round(buffers.rgbInput[i3 + 2] * 255))
    );
    data[i4 + 3] = buffers.texelCutoutMask[i] ? 255 : 0;
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

/**
 * Renders the Tangent-Space Normal Tensor (Nx, Ny, Nz) into a 256x256 PNG Canvas.
 */
export function createNormalMapCanvasFromBuffers(
  buffers: CNNReconstructionBuffers
): HTMLCanvasElement {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;

  const imgData = ctx.createImageData(w, h);
  const data = imgData.data;
  const len = w * h;

  for (let i = 0; i < len; i++) {
    const i3 = i * 3;
    const i4 = i * 4;
    data[i4] = Math.min(
      255,
      Math.max(0, Math.round((buffers.normalTensor[i3] * 0.5 + 0.5) * 255))
    );
    data[i4 + 1] = Math.min(
      255,
      Math.max(0, Math.round((buffers.normalTensor[i3 + 1] * 0.5 + 0.5) * 255))
    );
    data[i4 + 2] = Math.min(
      255,
      Math.max(0, Math.round((buffers.normalTensor[i3 + 2] * 0.5 + 0.5) * 255))
    );
    data[i4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

/**
 * Renders the Poisson-Integrated Depth / Heightmap Tensor into a 256x256 Grayscale/Height PNG Canvas.
 */
export function createDepthMapCanvasFromBuffers(
  buffers: CNNReconstructionBuffers
): HTMLCanvasElement {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;

  const imgData = ctx.createImageData(w, h);
  const data = imgData.data;
  const len = w * h;

  for (let i = 0; i < len; i++) {
    const i4 = i * 4;
    const v = Math.min(255, Math.max(0, Math.round(buffers.depthTensor[i] * 255)));
    data[i4] = v;
    data[i4 + 1] = v;
    data[i4 + 2] = v;
    data[i4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

/**
 * Converts a raw RGBA8 DataTexture into a canvas-backed texture.
 * GLTFExporter re-draws some maps (e.g. normal maps that need a channel flip)
 * with drawImage(), which only accepts real images/canvases, not DataTextures.
 */
function dataTextureToCanvasTexture(tex: THREE.Texture): THREE.Texture {
  const img = tex.image as { data?: ArrayLike<number>; width: number; height: number } | undefined;
  if (!(tex as THREE.DataTexture).isDataTexture || !img || !img.data) return tex;

  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return tex;

  // Rows are copied as-is and flipY is preserved, so GLTFExporter keeps the
  // same UV orientation it would use for the original texture.
  const imageData = ctx.createImageData(img.width, img.height);
  imageData.data.set(img.data as ArrayLike<number>);
  ctx.putImageData(imageData, 0, 0);

  const out = new THREE.CanvasTexture(canvas);
  out.colorSpace = tex.colorSpace;
  out.wrapS = tex.wrapS;
  out.wrapT = tex.wrapT;
  out.minFilter = tex.minFilter;
  out.magFilter = tex.magFilter;
  out.channel = tex.channel;
  out.flipY = tex.flipY;
  return out;
}

const EXPORTABLE_MAP_KEYS = [
  'map',
  'normalMap',
  'roughnessMap',
  'metalnessMap',
  'aoMap',
  'emissiveMap',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
] as const;

/**
 * Returns an export-safe copy of the object: materials are cloned and every
 * DataTexture is converted to a canvas texture. The live scene is untouched.
 */
function prepareObjectForGLTFExport(root: THREE.Object3D): THREE.Object3D {
  // SkeletonUtils.clone keeps skinned meshes bound to their (cloned) bones.
  const copy = SkeletonUtils.clone(root);
  // Bones go back to the rest pose: the clip, not the frame being shown, holds the motion.
  copy.traverse((o) => {
    const bone = o as THREE.Bone;
    if (bone.isBone && Array.isArray(bone.userData.bindPosition)) {
      bone.position.fromArray(bone.userData.bindPosition as number[]);
      bone.quaternion.identity();
      bone.scale.set(1, 1, 1);
    }
  });
  // The viewport sways/turns the model for display; export it facing +Z, upright.
  copy.position.set(0, 0, 0);
  copy.rotation.set(0, 0, 0);
  copy.scale.set(1, 1, 1);
  copy.updateMatrix();
  const converted = new Map<THREE.Texture, THREE.Texture>();
  const convert = (t: THREE.Texture) => {
    let c = converted.get(t);
    if (!c) {
      c = dataTextureToCanvasTexture(t);
      converted.set(t, c);
    }
    return c;
  };

  copy.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh) return;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    const cloned = mats.map((m) => {
      const mc = m.clone() as THREE.Material & Record<string, unknown>;
      for (const key of EXPORTABLE_MAP_KEYS) {
        const t = mc[key];
        if (t && (t as THREE.Texture).isTexture) mc[key] = convert(t as THREE.Texture);
      }
      return mc;
    });
    mesh.material = Array.isArray(mesh.material) ? cloned : cloned[0];
  });
  return copy;
}

/**
 * Exports a THREE.Mesh to binary .GLB (ArrayBuffer).
 */
export function exportMeshToGLBBytes(
  mesh: THREE.Object3D,
  animations: THREE.AnimationClip[] = []
): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const exporter = new GLTFExporter();
    exporter.parse(
      prepareObjectForGLTFExport(mesh),
      (result) => {
        if (result instanceof ArrayBuffer) {
          resolve(new Uint8Array(result));
        } else {
          reject(new Error('Se esperaba un ArrayBuffer binario desde GLTFExporter.'));
        }
      },
      (err) => reject(err),
      { binary: true, onlyVisible: false, animations }
    );
  });
}

/**
 * Serializes a THREE.BufferGeometry into Wavefront .OBJ ASCII format (vertices, UVs, normals, faces).
 */
export function exportGeometryToWavefrontOBJ(
  geometry: THREE.BufferGeometry,
  objectName = 'vortex3d_mesh'
): Uint8Array {
  const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
  const normAttr = geometry.getAttribute('normal') as THREE.BufferAttribute;
  const uvAttr = geometry.getAttribute('uv') as THREE.BufferAttribute;
  const indexAttr = geometry.getIndex();

  const lines: string[] = [
    `# Vortex3D — Estudio de Reconstruccion 3D Neuronal y Poisson`,
    `o ${objectName}`,
  ];

  if (posAttr) {
    const arr = posAttr.array as Float32Array;
    for (let i = 0; i < posAttr.count; i++) {
      const i3 = i * 3;
      lines.push(
        `v ${arr[i3].toFixed(5)} ${arr[i3 + 1].toFixed(5)} ${arr[i3 + 2].toFixed(5)}`
      );
    }
  }

  if (uvAttr) {
    const arr = uvAttr.array as Float32Array;
    for (let i = 0; i < uvAttr.count; i++) {
      const i2 = i * 2;
      lines.push(`vt ${arr[i2].toFixed(5)} ${arr[i2 + 1].toFixed(5)}`);
    }
  }

  if (normAttr) {
    const arr = normAttr.array as Float32Array;
    for (let i = 0; i < normAttr.count; i++) {
      const i3 = i * 3;
      lines.push(
        `vn ${arr[i3].toFixed(5)} ${arr[i3 + 1].toFixed(5)} ${arr[i3 + 2].toFixed(5)}`
      );
    }
  }

  if (indexAttr) {
    const indices = indexAttr.array;
    for (let i = 0; i < indexAttr.count; i += 3) {
      const a = indices[i] + 1;
      const b = indices[i + 1] + 1;
      const c = indices[i + 2] + 1;
      lines.push(`f ${a}/${a}/${a} ${b}/${b}/${b} ${c}/${c}/${c}`);
    }
  }

  return new TextEncoder().encode(lines.join('\n'));
}

// ============================================================================
// ZERO-DEPENDENCY STANDARD PKZIP ARCHIVE BUILDER (CRC32 + Store Method)
// Packages multiple .glb, .obj, and .png files into a single clean .zip file
// ============================================================================

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }
  return table;
})();

function computeCRC32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc = CRC32_TABLE[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export interface ZipFileEntry {
  filename: string;
  data: Uint8Array;
}

export function createZipArchiveBlob(entries: ZipFileEntry[]): Blob {
  const encoder = new TextEncoder();
  const encodedNames = entries.map((e) => encoder.encode(e.filename));
  const crcs = entries.map((e) => computeCRC32(e.data));

  let localTotalSize = 0;
  let centralTotalSize = 0;

  for (let i = 0; i < entries.length; i++) {
    localTotalSize += 30 + encodedNames[i].length + entries[i].data.length;
    centralTotalSize += 46 + encodedNames[i].length;
  }

  const totalSize = localTotalSize + centralTotalSize + 22;
  const out = new Uint8Array(totalSize);
  const view = new DataView(out.buffer);

  const localOffsets: number[] = [];
  let offset = 0;

  // 1. Write Local File Headers + File Payloads
  for (let i = 0; i < entries.length; i++) {
    localOffsets.push(offset);
    const nameBytes = encodedNames[i];
    const payload = entries[i].data;
    const crc = crcs[i];

    view.setUint32(offset, 0x04034b50, true); // Local file header signature
    view.setUint16(offset + 4, 20, true); // Version needed to extract (2.0)
    view.setUint16(offset + 6, 0x0800, true); // UTF-8 filename flag
    view.setUint16(offset + 8, 0, true); // Compression method: 0 (Store)
    view.setUint16(offset + 10, 0, true); // Last mod time
    view.setUint16(offset + 12, 0, true); // Last mod date
    view.setUint32(offset + 14, crc, true); // CRC-32
    view.setUint32(offset + 18, payload.length, true); // Compressed size
    view.setUint32(offset + 22, payload.length, true); // Uncompressed size
    view.setUint16(offset + 26, nameBytes.length, true); // Filename length
    view.setUint16(offset + 28, 0, true); // Extra field length
    offset += 30;

    out.set(nameBytes, offset);
    offset += nameBytes.length;

    out.set(payload, offset);
    offset += payload.length;
  }

  const centralDirStart = offset;

  // 2. Write Central Directory Headers
  for (let i = 0; i < entries.length; i++) {
    const nameBytes = encodedNames[i];
    const payload = entries[i].data;
    const crc = crcs[i];

    view.setUint32(offset, 0x02014b50, true); // Central directory signature
    view.setUint16(offset + 4, 20, true); // Version made by
    view.setUint16(offset + 6, 20, true); // Version needed
    view.setUint16(offset + 8, 0x0800, true); // UTF-8 flag
    view.setUint16(offset + 10, 0, true); // Compression method: 0 (Store)
    view.setUint16(offset + 12, 0, true); // Mod time
    view.setUint16(offset + 14, 0, true); // Mod date
    view.setUint32(offset + 16, crc, true); // CRC-32
    view.setUint32(offset + 20, payload.length, true); // Compressed size
    view.setUint32(offset + 24, payload.length, true); // Uncompressed size
    view.setUint16(offset + 28, nameBytes.length, true); // Filename length
    view.setUint16(offset + 30, 0, true); // Extra field length
    view.setUint16(offset + 32, 0, true); // File comment length
    view.setUint16(offset + 34, 0, true); // Disk number start
    view.setUint16(offset + 36, 0, true); // Internal file attributes
    view.setUint32(offset + 38, 0, true); // External file attributes
    view.setUint32(offset + 42, localOffsets[i], true); // Relative offset of local header
    offset += 46;

    out.set(nameBytes, offset);
    offset += nameBytes.length;
  }

  // 3. Write End of Central Directory Record (EOCD)
  view.setUint32(offset, 0x06054b50, true);
  view.setUint16(offset + 4, 0, true);
  view.setUint16(offset + 6, 0, true);
  view.setUint16(offset + 8, entries.length, true);
  view.setUint16(offset + 10, entries.length, true);
  view.setUint32(offset + 12, centralTotalSize, true);
  view.setUint32(offset + 16, centralDirStart, true);
  view.setUint16(offset + 20, 0, true);

  return new Blob([out], { type: 'application/zip' });
}

/**
 * Generates and downloads a single preset model's asset (RGB, Normal Map, or 3D .GLB)
 * or builds the complete ZIP package containing ALL 6 application models (3D .GLB + .OBJ + RGB + Normales + Profundidad).
 */
export async function buildSinglePresetModelAssets(
  presetId: string,
  baseParams: ReconstructionParams,
  zones: ZoneConfig[] = DEFAULT_ZONES
): Promise<{
  rgbBytes: Uint8Array;
  normalBytes: Uint8Array;
  depthBytes: Uint8Array;
  glbBytes: Uint8Array;
  objBytes: Uint8Array;
}> {
  const preset =
    PRESET_REFERENCE_IMAGES.find((p) => p.id === presetId) ||
    PRESET_REFERENCE_IMAGES[0];

  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = CNN_TENSOR_SIZE;
  srcCanvas.height = CNN_TENSOR_SIZE;
  renderPresetImageToCanvas(preset.id, srcCanvas);

  const tempBuffers = new CNNReconstructionBuffers();
  runCNNReconstructionPipeline(srcCanvas, tempBuffers);

  const modelParams: ReconstructionParams = {
    ...baseParams,
    topology: preset.recommendedTopology,
    normalReconMode:
      preset.recommendedNormalMode ||
      NormalReconstructionMode.HybridScreenedPoisson,
    renderNormalShading: false,
  };

  const albedoTex = new THREE.DataTexture(
    new Uint8Array(ATLAS_TEX_SIZE * ATLAS_TEX_SIZE * 4),
    ATLAS_TEX_SIZE,
    ATLAS_TEX_SIZE,
    THREE.RGBAFormat
  );
  albedoTex.colorSpace = THREE.SRGBColorSpace;
  albedoTex.needsUpdate = true;

  const roughMetalTex = new THREE.DataTexture(
    new Uint8Array(ATLAS_TEX_SIZE * ATLAS_TEX_SIZE * 4),
    ATLAS_TEX_SIZE,
    ATLAS_TEX_SIZE,
    THREE.RGBAFormat
  );
  const normalTex = new THREE.DataTexture(
    new Uint8Array(ATLAS_TEX_SIZE * ATLAS_TEX_SIZE * 4),
    ATLAS_TEX_SIZE,
    ATLAS_TEX_SIZE,
    THREE.RGBAFormat
  );

  const geo = new THREE.BufferGeometry();
  buildReconstructedMeshGeometry(
    geo,
    tempBuffers,
    modelParams,
    zones,
    albedoTex,
    roughMetalTex,
    normalTex
  );

  const mat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: albedoTex,
    normalMap: normalTex,
    roughnessMap: roughMetalTex,
    metalnessMap: roughMetalTex,
    roughness: 0.35,
    metalness: 0.65,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = preset.id;

  const [rgbBytes, normalBytes, depthBytes, glbBytes] = await Promise.all([
    canvasToPNGBytes(createRGBCanvasFromBuffers(tempBuffers)),
    canvasToPNGBytes(createNormalMapCanvasFromBuffers(tempBuffers)),
    canvasToPNGBytes(createDepthMapCanvasFromBuffers(tempBuffers)),
    exportMeshToGLBBytes(mesh),
  ]);
  const objBytes = exportGeometryToWavefrontOBJ(geo, preset.id);

  geo.dispose();
  mat.dispose();
  albedoTex.dispose();
  roughMetalTex.dispose();
  normalTex.dispose();

  return { rgbBytes, normalBytes, depthBytes, glbBytes, objBytes };
}

/**
 * Generates a complete ZIP archive containing ALL 6 application models:
 * For each model:
 *   - `<id>/<id>_modelo_3d.glb`
 *   - `<id>/<id>_malla_3d.obj`
 *   - `<id>/<id>_textura_rgb.png`
 *   - `<id>/<id>_mapa_normales.png`
 *   - `<id>/<id>_profundidad_poisson.png`
 */
export async function exportAllApplicationModelsToZip(
  baseParams: ReconstructionParams,
  onProgress?: (current: number, total: number, title: string) => void
): Promise<Blob> {
  const entries: ZipFileEntry[] = [];
  const total = PRESET_REFERENCE_IMAGES.length;

  for (let i = 0; i < total; i++) {
    const preset = PRESET_REFERENCE_IMAGES[i];
    onProgress?.(i + 1, total, preset.title);
    // Yield briefly to UI thread so progress updates render smoothly
    await new Promise((r) => setTimeout(r, 15));

    const assets = await buildSinglePresetModelAssets(preset.id, baseParams);
    const folder = `${String(i + 1).padStart(2, '0')}_${preset.id}`;

    entries.push(
      { filename: `${folder}/${preset.id}_modelo_3d.glb`, data: assets.glbBytes },
      { filename: `${folder}/${preset.id}_malla_3d.obj`, data: assets.objBytes },
      { filename: `${folder}/${preset.id}_textura_rgb.png`, data: assets.rgbBytes },
      { filename: `${folder}/${preset.id}_mapa_normales.png`, data: assets.normalBytes },
      { filename: `${folder}/${preset.id}_profundidad_poisson.png`, data: assets.depthBytes }
    );
  }

  return createZipArchiveBlob(entries);
}
