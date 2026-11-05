import * as THREE from 'three';
import { buildHumanBodyTemplateGeometry } from './humanBodyTemplate';
import type { HumanRig } from './humanRig';
import {
  BackImageSourceMode,
  BackgroundCutoutMode,
  CNNFeatureView,
  LateralJointProfile,
  LimbUnionMode,
  MathTopology,
  NormalReconstructionMode,
  PresetBaseShape,
  PresetReferenceImage,
  ReconstructionParams,
  ShapeSilhouetteMode,
  SideImageSourceMode,
  SmoothingAlgorithm,
  TexturePattern,
  ZoneConfig,
} from '../types/studio';

export const CNN_TENSOR_SIZE = 256;
export const ATLAS_TEX_SIZE = 512;
const TENSOR_LEN = CNN_TENSOR_SIZE * CNN_TENSOR_SIZE;

export const PRESET_REFERENCE_IMAGES: PresetReferenceImage[] = [
  {
    id: 'royal_medallion',
    title: 'Emblema Solar Dorado',
    subtitle: 'Escudo en relieve de oro bruñido, esmalte azul cobalto y rubí',
    recommendedTopology: MathTopology.MonolithRelief,
    recommendedNormalMode: NormalReconstructionMode.HybridScreenedPoisson,
  },
  {
    id: 'tangent_normal_shield',
    title: 'Mapa de Normales RGB (Búho Real)',
    subtitle: 'Mapa de normales en espacio tangente (R=Nx, G=Ny, B=Nz) para integración pura de Poisson',
    recommendedTopology: MathTopology.MonolithRelief,
    recommendedNormalMode: NormalReconstructionMode.DirectNormalMapRGB,
  },
  {
    id: 'cyber_samurai',
    title: 'Máscara Mecánica Oni',
    subtitle: 'Armadura facial en titanio rojo carmesí con visores ópticos cian',
    recommendedTopology: MathTopology.SphericalHarmonics,
    recommendedNormalMode: NormalReconstructionMode.HybridScreenedPoisson,
  },
  {
    id: 'ammonite_fossil',
    title: 'Amonites Espiral Áureo',
    subtitle: 'Concha logarítmica mineralizada en pirita sobre matriz de pizarra',
    recommendedTopology: MathTopology.MonolithRelief,
    recommendedNormalMode: NormalReconstructionMode.PurePoissonIntegration,
  },
  {
    id: 'classical_amphora',
    title: 'Ánfora Real Policromada',
    subtitle: 'Vasija de revolución en terracota, pan de oro y grecas clásicas',
    recommendedTopology: MathTopology.AxialRevolution,
    recommendedNormalMode: NormalReconstructionMode.HybridScreenedPoisson,
  },
  {
    id: 'emerald_geode',
    title: 'Núcleo Cristalino Esmeralda',
    subtitle: 'Estructura biomimética facetada en verde esmeralda y vetas áureas',
    recommendedTopology: MathTopology.ImplicitGyroid,
    recommendedNormalMode: NormalReconstructionMode.HybridScreenedPoisson,
  },
];

export const DEFAULT_ZONES: ZoneConfig[] = [
  {
    id: 0,
    name: 'Zona 1 · Relieve Principal',
    description: 'Zonas altas y primer plano reconstruidas por integración de normales',
    extrusion: 0.22,
    microFrequency: 6,
    microAmplitude: 0.0,
    radialScale: 1.04,
    colorHex: '#f59e0b',
    zoneTintMix: 0.0,
    pattern: TexturePattern.OriginalPhoto,
    roughness: 0.24,
    metalness: 0.78,
  },
  {
    id: 1,
    name: 'Zona 2 · Contorno Medio',
    description: 'Pendientes estructurales donde el gradiente de normales (p, q) es moderado',
    extrusion: 0.08,
    microFrequency: 8,
    microAmplitude: 0.0,
    radialScale: 1.0,
    colorHex: '#38bdf8',
    zoneTintMix: 0.0,
    pattern: TexturePattern.OriginalPhoto,
    roughness: 0.36,
    metalness: 0.45,
  },
  {
    id: 2,
    name: 'Zona 3 · Fondo y Base',
    description: 'Regiones planas donde la normal apunta hacia +Z (Nz ≈ 1)',
    extrusion: -0.1,
    microFrequency: 4,
    microAmplitude: 0.0,
    radialScale: 0.96,
    colorHex: '#1e293b',
    zoneTintMix: 0.0,
    pattern: TexturePattern.OriginalPhoto,
    roughness: 0.72,
    metalness: 0.2,
  },
  {
    id: 3,
    name: 'Zona 4 · Crestas de Alta Curvatura',
    description: 'Aristas donde la divergencia del campo de normales (∇·n) es máxima',
    extrusion: 0.3,
    microFrequency: 12,
    microAmplitude: 0.0,
    radialScale: 1.06,
    colorHex: '#fde047',
    zoneTintMix: 0.0,
    pattern: TexturePattern.OriginalPhoto,
    roughness: 0.16,
    metalness: 0.92,
  },
];

/**
 * Pre-allocated CNN, Poisson Normal Integration, and Spatial Hash Welding Buffers (Zero-GC)
 */
export class CNNReconstructionBuffers {
  public readonly rgbInput = new Float32Array(TENSOR_LEN * 3);
  public readonly alphaInput = new Float32Array(TENSOR_LEN);
  public readonly luminance = new Float32Array(TENSOR_LEN);
  public readonly convGradX = new Float32Array(TENSOR_LEN);
  public readonly convGradY = new Float32Array(TENSOR_LEN);
  public readonly convLaplacian = new Float32Array(TENSOR_LEN);
  public readonly blurTemp = new Float32Array(TENSOR_LEN);
  public readonly lowFreqVolume = new Float32Array(TENSOR_LEN);
  public readonly rawCnnDepth = new Float32Array(TENSOR_LEN);

  // Normal-Based Reconstruction (Poisson Solver) Pre-allocated Tensors
  public readonly slopeP = new Float32Array(TENSOR_LEN);
  public readonly slopeQ = new Float32Array(TENSOR_LEN);
  public readonly divergenceField = new Float32Array(TENSOR_LEN);
  public readonly poissonSurface = new Float32Array(TENSOR_LEN);
  public readonly customNormalTilt = new Float32Array(TENSOR_LEN * 2);

  public readonly depthTensor = new Float32Array(TENSOR_LEN);
  public readonly normalTensor = new Float32Array(TENSOR_LEN * 3);
  public readonly zoneMap = new Uint8Array(TENSOR_LEN);

  // Outer Frame & Color/Tone Cutout Pre-allocated Masks (1 = active foreground, 0 = removed background)
  public readonly texelCutoutMask = new Uint8Array(TENSOR_LEN);
  public readonly drawingSilhouetteMask = new Uint8Array(TENSOR_LEN);
  public readonly rowSilhouetteMinU = new Float32Array(CNN_TENSOR_SIZE);
  public readonly rowSilhouetteMaxU = new Float32Array(CNN_TENSOR_SIZE);
  public readonly rowSilhouetteTempMinU = new Float32Array(CNN_TENSOR_SIZE);
  public readonly rowSilhouetteTempMaxU = new Float32Array(CNN_TENSOR_SIZE);
  public readonly candidateBgMask = new Uint8Array(TENSOR_LEN);
  public readonly exteriorBgMask = new Uint8Array(TENSOR_LEN);
  public readonly floodVisited = new Uint8Array(TENSOR_LEN);
  public readonly floodQueue = new Int32Array(TENSOR_LEN);
  public readonly morphTempMask = new Uint8Array(TENSOR_LEN);
  public readonly borderDistField = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);
  public autoDetectedBgR = 0.047;
  public autoDetectedBgG = 0.071;
  public autoDetectedBgB = 0.125;
  public autoDetectedBgHex = '#0c1220';
  // Auto-Detected Drawing Silhouette Bounding Box & Proportions (Zero-GC)
  public autoSilhouetteMinU = 0.08;
  public autoSilhouetteMaxU = 0.92;
  public autoSilhouetteMinV = 0.06;
  public autoSilhouetteMaxV = 0.94;
  public autoSilhouetteWidth = 0.84;
  public autoSilhouetteHeight = 0.88;
  public autoSilhouetteAspect = 0.95;
  public autoSilhouetteCenterX = 0.0;
  public autoSilhouetteCenterY = 0.0;
  public autoDetectedArmLengthScale = 1.0;
  public autoDetectedArmThicknessScale = 1.0;
  public autoDetectedLegLengthScale = 1.0;
  public autoDetectedLegThicknessScale = 1.0;
  public autoDetectedPoseArmAngleDeg = 35;

  // Straight-Line Structure Tensor & Anti-Mound Planar Facet Pre-allocated Tensors (Zero-GC)
  public readonly lineCoherenceTensor = new Float32Array(TENSOR_LEN);
  public readonly lineTangentX = new Float32Array(TENSOR_LEN);
  public readonly lineTangentY = new Float32Array(TENSOR_LEN);

  // Multi-View Solid Reconstruction (Back Image + Symmetric Rules with Back Depths + Lateral Seam Band)
  public readonly customBackRgb = new Float32Array(TENSOR_LEN * 3);
  public readonly customBackAlpha = new Float32Array(TENSOR_LEN);
  public readonly customBackDepth = new Float32Array(TENSOR_LEN);
  public readonly customBackFgMask = new Uint8Array(TENSOR_LEN);
  public readonly backRowSilhouetteMinU = new Float32Array(CNN_TENSOR_SIZE);
  public readonly backRowSilhouetteMaxU = new Float32Array(CNN_TENSOR_SIZE);
  public readonly backRowTempMinU = new Float32Array(CNN_TENSOR_SIZE);
  public readonly backRowTempMaxU = new Float32Array(CNN_TENSOR_SIZE);
  public readonly backRawCnnDepth = new Float32Array(TENSOR_LEN);
  public readonly backLowFreqVolume = new Float32Array(TENSOR_LEN);
  public readonly backConvGradX = new Float32Array(TENSOR_LEN);
  public readonly backConvGradY = new Float32Array(TENSOR_LEN);
  public readonly backConvLaplacian = new Float32Array(TENSOR_LEN);
  public readonly backDivergenceField = new Float32Array(TENSOR_LEN);
  public readonly backPoissonSurface = new Float32Array(TENSOR_LEN);
  public readonly backLineCoherence = new Float32Array(TENSOR_LEN);
  public readonly backLineTangentX = new Float32Array(TENSOR_LEN);
  public readonly backLineTangentY = new Float32Array(TENSOR_LEN);
  public readonly backZoneMap = new Uint8Array(TENSOR_LEN);
  public readonly backRgbActive = new Float32Array(TENSOR_LEN * 3);
  public readonly backDepthTensor = new Float32Array(TENSOR_LEN);
  public readonly backNormalTensor = new Float32Array(TENSOR_LEN * 3);
  public hasCustomBackImage = false;
  public backAutoBgR = 0.047;
  public backAutoBgG = 0.071;
  public backAutoBgB = 0.125;
  public backAutoSilhouetteMinU = 0.08;
  public backAutoSilhouetteMaxU = 0.92;
  public backAutoSilhouetteMinV = 0.06;
  public backAutoSilhouetteMaxV = 0.94;
  public backAutoSilhouetteWidth = 0.84;
  public backAutoSilhouetteHeight = 0.88;

  public readonly customSideRgb = new Float32Array(TENSOR_LEN * 3);
  public readonly customSideDepth = new Float32Array(TENSOR_LEN);
  public readonly sideRgbActive = new Float32Array(TENSOR_LEN * 3);
  public readonly sideDepthTensor = new Float32Array(TENSOR_LEN);
  public readonly sideNormalTensor = new Float32Array(TENSOR_LEN * 3);
  public hasCustomSideImage = false;

  // Zero-GC Voronoï Multi-Source Foreground Image Color Inpainting Buffers (ensures the joint/seam always takes the clean image color)
  public readonly frontInpaintedRgb = new Float32Array(TENSOR_LEN * 3);
  public readonly backInpaintedRgb = new Float32Array(TENSOR_LEN * 3);
  public readonly nearestFgTexelIdx = new Int32Array(TENSOR_LEN);

  // Sculpt, Smoothing, Multi-View Solid & Close-Point Vertex Welding Pre-allocated Buffers
  public static readonly MAX_VERTICES = 196608;
  public static readonly HASH_TABLE_SIZE = 262144; // Power of 2 for bitwise mask
  public static readonly MAX_SEAM_EDGES = 16384;
  public readonly sculptOffsets = new Float32Array(CNNReconstructionBuffers.MAX_VERTICES * 3);
  public readonly customZoneOverrides = new Int8Array(CNNReconstructionBuffers.MAX_VERTICES);
  public readonly laplacianTemp = new Float32Array(CNNReconstructionBuffers.MAX_VERTICES * 3);
  public readonly analyticalSideNormals = new Float32Array(CNNReconstructionBuffers.MAX_VERTICES * 3);
  public readonly vertexCutoutMask = new Uint8Array(CNNReconstructionBuffers.MAX_VERTICES);
  public readonly vertexLineCoherence = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);
  public readonly vertexLineTanX = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);
  public readonly vertexLineTanY = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);
  public readonly vertexOutNx = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);
  public readonly vertexOutNy = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);
  public readonly seamEdgeVStart = new Int32Array(CNNReconstructionBuffers.MAX_SEAM_EDGES);
  public readonly seamEdgeVEnd = new Int32Array(CNNReconstructionBuffers.MAX_SEAM_EDGES);
  public readonly seamEdgeBaseV = new Int32Array(CNNReconstructionBuffers.MAX_SEAM_EDGES);
  public readonly quadActiveMask = new Uint8Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);
  public readonly activeIndices = new Uint32Array(CNNReconstructionBuffers.MAX_VERTICES * 6);
  public readonly weldHead = new Int32Array(CNNReconstructionBuffers.HASH_TABLE_SIZE);
  public readonly weldNext = new Int32Array(CNNReconstructionBuffers.MAX_VERTICES);
  public readonly weldClusterRep = new Int32Array(CNNReconstructionBuffers.MAX_VERTICES);

  constructor() {
    this.texelCutoutMask.fill(1);
    this.vertexCutoutMask.fill(1);
    this.resetSculptAndZoneOverrides();
  }

  public resetSculptAndZoneOverrides(): void {
    this.sculptOffsets.fill(0);
    this.customZoneOverrides.fill(-1);
    this.customNormalTilt.fill(0);
  }
}

function sampleOwlCrestHeight(nx: number, ny: number): number {
  const r = Math.hypot(nx, ny);
  if (r > 0.92) return 0;

  const rim = Math.exp(-Math.pow((r - 0.82) / 0.06, 2)) * 0.45;
  const shieldDome = Math.sqrt(Math.max(0, 0.85 - r * r)) * 0.55;

  const eyeL = Math.hypot(nx + 0.26, ny + 0.18);
  const eyeR = Math.hypot(nx - 0.26, ny + 0.18);
  const eyeRingL = Math.exp(-Math.pow((eyeL - 0.16) / 0.045, 2)) * 0.42;
  const eyeRingR = Math.exp(-Math.pow((eyeR - 0.16) / 0.045, 2)) * 0.42;
  const pupilL = Math.max(0, 0.09 - eyeL) * 3.2;
  const pupilR = Math.max(0, 0.09 - eyeR) * 3.2;

  const beak =
    Math.abs(nx) < 0.12 && ny > -0.18 && ny < 0.16
      ? (0.12 - Math.abs(nx)) * (1.0 - Math.abs(ny - 0.02) * 4.0) * 3.2
      : 0;
  const brow =
    ny < -0.1 && ny > -0.45
      ? Math.exp(-Math.pow((ny + 0.18 + Math.abs(nx) * 0.45) / 0.05, 2)) * 0.38
      : 0;

  const inChest = Math.abs(nx) < 0.42 && ny > 0.12 && ny < 0.68;
  const feathers = inChest
    ? Math.abs(Math.sin(nx * 28) * Math.cos(ny * 24)) * 0.24
    : 0;

  return (
    shieldDome +
    rim +
    eyeRingL +
    eyeRingR +
    pupilL +
    pupilR +
    Math.max(0, beak) +
    brow +
    feathers
  );
}

export function renderPresetImageToCanvas(
  presetId: string,
  canvas: HTMLCanvasElement
): void {
  canvas.width = CNN_TENSOR_SIZE;
  canvas.height = CNN_TENSOR_SIZE;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const cx = w * 0.5;
  const cy = h * 0.5;

  ctx.fillStyle = '#0c1220';
  ctx.fillRect(0, 0, w, h);

  if (presetId === 'tangent_normal_shield') {
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;
    const eps = 2.0 / w;

    for (let y = 0; y < h; y++) {
      const ny = (y - cy) / (h * 0.5);
      for (let x = 0; x < w; x++) {
        const nx = (x - cx) / (w * 0.5);
        const hR = sampleOwlCrestHeight(nx + eps, ny);
        const hL = sampleOwlCrestHeight(nx - eps, ny);
        const hU = sampleOwlCrestHeight(nx, ny - eps);
        const hD = sampleOwlCrestHeight(nx, ny + eps);

        const dzdx = (hR - hL) / (2 * eps);
        const dzdy = (hU - hD) / (2 * eps);

        const sx = -dzdx * 1.35;
        const sy = -dzdy * 1.35;
        const sz = 1.0;
        const invLen = 1.0 / Math.sqrt(sx * sx + sy * sy + sz * sz);

        const idx = (y * w + x) * 4;
        data[idx] = Math.min(
          255,
          Math.max(0, Math.round((sx * invLen * 0.5 + 0.5) * 255))
        );
        data[idx + 1] = Math.min(
          255,
          Math.max(0, Math.round((sy * invLen * 0.5 + 0.5) * 255))
        );
        data[idx + 2] = Math.min(
          255,
          Math.max(0, Math.round((sz * invLen * 0.5 + 0.5) * 255))
        );
        data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
    return;
  }

  if (presetId === 'royal_medallion') {
    const bgGrad = ctx.createRadialGradient(cx, cy, 10, cx, cy, w * 0.46);
    bgGrad.addColorStop(0, '#1e40af');
    bgGrad.addColorStop(0.55, '#1d4ed8');
    bgGrad.addColorStop(0.75, '#991b1b');
    bgGrad.addColorStop(0.92, '#f59e0b');
    bgGrad.addColorStop(1, '#0c1220');
    ctx.fillStyle = bgGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, w * 0.45, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 10;
    ctx.beginPath();
    ctx.arc(cx, cy, w * 0.41, 0, Math.PI * 2);
    ctx.stroke();

    for (let i = 0; i < 16; i++) {
      const angle = (i * Math.PI * 2) / 16;
      const isMajor = i % 2 === 0;
      const rOuter = isMajor ? w * 0.37 : w * 0.29;
      ctx.fillStyle = isMajor ? '#fde047' : '#f59e0b';
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(
        cx + Math.cos(angle - 0.11) * (w * 0.16),
        cy + Math.sin(angle - 0.11) * (w * 0.16)
      );
      ctx.lineTo(cx + Math.cos(angle) * rOuter, cy + Math.sin(angle) * rOuter);
      ctx.lineTo(
        cx + Math.cos(angle + 0.11) * (w * 0.16),
        cy + Math.sin(angle + 0.11) * (w * 0.16)
      );
      ctx.closePath();
      ctx.fill();
    }

    for (let i = 0; i < 12; i++) {
      const a = (i * Math.PI * 2) / 12;
      const rx = cx + Math.cos(a) * (w * 0.41);
      const ry = cy + Math.sin(a) * (w * 0.41);
      ctx.fillStyle = '#fef08a';
      ctx.beginPath();
      ctx.arc(rx, ry, 5.5, 0, Math.PI * 2);
      ctx.fill();
    }

    const bossGrad = ctx.createRadialGradient(
      cx - 8,
      cy - 8,
      2,
      cx,
      cy,
      w * 0.17
    );
    bossGrad.addColorStop(0, '#fef9c3');
    bossGrad.addColorStop(0.5, '#f59e0b');
    bossGrad.addColorStop(0.85, '#b45309');
    bossGrad.addColorStop(1, '#78350f');
    ctx.fillStyle = bossGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, w * 0.17, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(cx, cy, w * 0.07, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fecaca';
    ctx.beginPath();
    ctx.arc(cx - 4, cy - 4, w * 0.025, 0, Math.PI * 2);
    ctx.fill();
  } else if (presetId === 'cyber_samurai') {
    const maskGrad = ctx.createRadialGradient(cx, cy, 10, cx, cy, w * 0.44);
    maskGrad.addColorStop(0, '#ef4444');
    maskGrad.addColorStop(0.55, '#b91c1c');
    maskGrad.addColorStop(0.85, '#1e293b');
    maskGrad.addColorStop(1, '#090d16');
    ctx.fillStyle = maskGrad;
    ctx.beginPath();
    ctx.ellipse(cx, cy, w * 0.36, h * 0.43, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#e2e8f0';
    ctx.beginPath();
    ctx.moveTo(cx, cy - 35);
    ctx.lineTo(cx - 68, cy - 62);
    ctx.lineTo(cx - 52, cy - 82);
    ctx.lineTo(cx, cy - 58);
    ctx.lineTo(cx + 52, cy - 82);
    ctx.lineTo(cx + 68, cy - 62);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = '#38bdf8';
    ctx.shadowColor = '#38bdf8';
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.ellipse(cx - 38, cy - 22, 22, 9, 0.22, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(cx + 38, cy - 22, 22, 9, -0.22, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.ellipse(cx - 38, cy - 22, 11, 4, 0.22, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(cx + 38, cy - 22, 11, 4, -0.22, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#f8fafc';
    ctx.beginPath();
    ctx.moveTo(cx, cy - 26);
    ctx.lineTo(cx - 16, cy + 18);
    ctx.lineTo(cx + 16, cy + 18);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = '#fbbf24';
    for (let i = -3; i <= 3; i++) {
      const vx = cx + i * 16;
      const vh = 28 - Math.abs(i) * 4;
      ctx.fillRect(vx - 5, cy + 32, 10, vh);
    }
  } else if (presetId === 'ammonite_fossil') {
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const dx = (x - cx) / (w * 0.44);
        const dy = (y - cy) / (h * 0.44);
        const r = Math.hypot(dx, dy);
        const theta = Math.atan2(dy, dx) + Math.PI;
        const idx = (y * w + x) * 4;

        if (r > 0.96) {
          data[idx] = 14;
          data[idx + 1] = 19;
          data[idx + 2] = 30;
          data[idx + 3] = 255;
          continue;
        }

        const spiralPhase =
          Math.log(Math.max(0.04, r)) * 3.8 - theta / (Math.PI * 2);
        const whorl = spiralPhase - Math.floor(spiralPhase);
        const tubeProfile = Math.sin(whorl * Math.PI);
        const ribs = 0.5 + 0.5 * Math.cos(theta * 18 + r * 14);
        const heightVal =
          Math.max(0, 1 - r * 0.45) * tubeProfile * (0.68 + 0.32 * ribs);

        data[idx] = Math.min(255, Math.floor(45 + heightVal * 210));
        data[idx + 1] = Math.min(255, Math.floor(35 + heightVal * 165));
        data[idx + 2] = Math.min(255, Math.floor(25 + heightVal * 95));
        data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  } else if (presetId === 'classical_amphora') {
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;
    for (let y = 0; y < h; y++) {
      const ny = y / h;
      const bandPattern = Math.sin(ny * Math.PI * 16);
      const isGoldBand =
        Math.abs(ny - 0.25) < 0.045 || Math.abs(ny - 0.65) < 0.045;
      const isTurquoiseBand = Math.abs(ny - 0.45) < 0.06;

      for (let x = 0; x < w; x++) {
        const nx = (x - cx) / (w * 0.46);
        const flute = 0.5 + 0.5 * Math.cos(nx * Math.PI * 12);
        const cylinderShade = Math.sqrt(Math.max(0.08, 1 - nx * nx * 0.85));
        const relief = cylinderShade * (0.76 + 0.24 * flute);

        let rC = 215;
        let gC = 105;
        let bC = 55;

        if (isGoldBand) {
          rC = 250;
          gC = 204;
          bC = 21;
        } else if (isTurquoiseBand) {
          rC = 56;
          gC = 189;
          bC = 248;
        } else if (bandPattern > 0.75) {
          rC = 245;
          gC = 185;
          bC = 110;
        }

        const idx = (y * w + x) * 4;
        data[idx] = Math.min(255, Math.floor(rC * relief));
        data[idx + 1] = Math.min(255, Math.floor(gC * relief));
        data[idx + 2] = Math.min(255, Math.floor(bC * relief));
        data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  } else {
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const u = ((x - cx) / w) * Math.PI * 6.0;
        const v = ((y - cy) / h) * Math.PI * 6.0;
        const dist = Math.hypot(x - cx, y - cy) / (w * 0.47);
        const mask = Math.max(0, 1 - dist * dist * 0.85);
        const g =
          (Math.sin(u) * Math.cos(v) +
            Math.sin(v) * Math.cos(u * 0.8 - v * 0.6) +
            Math.cos(u * 1.3 + v * 0.9)) /
          3;
        const val = Math.max(0, Math.min(1, (g * 0.5 + 0.5) * mask));
        const isGoldVein = Math.abs(g) < 0.06 && dist < 0.85;
        const idx = (y * w + x) * 4;

        if (isGoldVein) {
          data[idx] = 250;
          data[idx + 1] = 204;
          data[idx + 2] = 45;
        } else {
          data[idx] = Math.floor(val * 45 + Math.pow(val, 3) * 90);
          data[idx + 1] = Math.floor(val * 225 + 20);
          data[idx + 2] = Math.floor(val * 165 + 25);
        }
        data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }
}

export function runCNNReconstructionPipeline(
  sourceCanvas: HTMLCanvasElement,
  buffers: CNNReconstructionBuffers
): number {
  const t0 = performance.now();
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;

  const ctx = sourceCanvas.getContext('2d');
  if (!ctx) return 0;

  const imgData = ctx.getImageData(0, 0, w, h).data;
  const {
    rgbInput,
    alphaInput,
    luminance,
    convGradX,
    convGradY,
    convLaplacian,
    blurTemp,
    lowFreqVolume,
    rawCnnDepth,
  } = buffers;

  for (let i = 0; i < TENSOR_LEN; i++) {
    const i4 = i * 4;
    const i3 = i * 3;
    const r = imgData[i4] / 255;
    const g = imgData[i4 + 1] / 255;
    const b = imgData[i4 + 2] / 255;
    const a = imgData[i4 + 3] / 255;

    rgbInput[i3] = r;
    rgbInput[i3 + 1] = g;
    rgbInput[i3 + 2] = b;
    alphaInput[i] = a;
  }

  // Automatically sample the 4 outer corners (8x8 patches) to detect background frame color
  let bgSumR = 0;
  let bgSumG = 0;
  let bgSumB = 0;
  let bgCount = 0;
  const cornerCoords = [
    [2, 2],
    [w - 10, 2],
    [2, h - 10],
    [w - 10, h - 10],
  ];
  for (let c = 0; c < 4; c++) {
    const startX = cornerCoords[c][0];
    const startY = cornerCoords[c][1];
    for (let dy = 0; dy < 8; dy++) {
      for (let dx = 0; dx < 8; dx++) {
        const idx = (startY + dy) * w + (startX + dx);
        if (alphaInput[idx] < 0.2) continue;
        const idx3 = idx * 3;
        bgSumR += rgbInput[idx3];
        bgSumG += rgbInput[idx3 + 1];
        bgSumB += rgbInput[idx3 + 2];
        bgCount++;
      }
    }
  }
  if (bgCount > 0) {
    buffers.autoDetectedBgR = bgSumR / bgCount;
    buffers.autoDetectedBgG = bgSumG / bgCount;
    buffers.autoDetectedBgB = bgSumB / bgCount;
    const cr = Math.min(255, Math.max(0, Math.round(buffers.autoDetectedBgR * 255)));
    const cg = Math.min(255, Math.max(0, Math.round(buffers.autoDetectedBgG * 255)));
    const cb = Math.min(255, Math.max(0, Math.round(buffers.autoDetectedBgB * 255)));
    buffers.autoDetectedBgHex =
      '#' +
      cr.toString(16).padStart(2, '0') +
      cg.toString(16).padStart(2, '0') +
      cb.toString(16).padStart(2, '0');
  }

  const bgR = buffers.autoDetectedBgR;
  const bgG = buffers.autoDetectedBgG;
  const bgB = buffers.autoDetectedBgB;
  const bgLum = 0.299 * bgR + 0.587 * bgG + 0.114 * bgB;
  const isLightBackground = bgLum > 0.46;

  // Flood-fill exterior background from the 4 outer edges using color similarity to corner background (Zero-GC)
  const { exteriorBgMask, floodQueue } = buffers;
  exteriorBgMask.fill(0);
  let qHead = 0;
  let qTail = 0;
  const extColorTolSq = isLightBackground ? 0.085 : 0.045;

  const trySeedExterior = (idx: number) => {
    if (exteriorBgMask[idx] === 1) return;
    if (alphaInput[idx] < 0.15) {
      exteriorBgMask[idx] = 1;
      floodQueue[qTail++] = idx;
      return;
    }
    const i3 = idx * 3;
    const dr = rgbInput[i3] - bgR;
    const dg = rgbInput[i3 + 1] - bgG;
    const db = rgbInput[i3 + 2] - bgB;
    if (dr * dr + dg * dg + db * db <= extColorTolSq) {
      exteriorBgMask[idx] = 1;
      floodQueue[qTail++] = idx;
    }
  };

  for (let x = 0; x < w; x++) {
    trySeedExterior(x);
    trySeedExterior((h - 1) * w + x);
  }
  for (let y = 1; y < h - 1; y++) {
    trySeedExterior(y * w);
    trySeedExterior(y * w + (w - 1));
  }

  while (qHead < qTail) {
    const curr = floodQueue[qHead++];
    const cx = curr % w;
    const cy = (curr / w) | 0;

    if (cx > 0) trySeedExterior(curr - 1);
    if (cx < w - 1) trySeedExterior(curr + 1);
    if (cy > 0) trySeedExterior(curr - w);
    if (cy < h - 1) trySeedExterior(curr + w);
  }

  // Compute foreground-aware luminance/height signal so light/white backgrounds drop to 0.0
  // while dark armor/outlines inside the foreground subject retain solid positive depth!
  for (let i = 0; i < TENSOR_LEN; i++) {
    const a = alphaInput[i];
    if (a < 0.12 || exteriorBgMask[i] === 1) {
      luminance[i] = 0.0;
      continue;
    }
    const i3 = i * 3;
    const r = rgbInput[i3];
    const g = rgbInput[i3 + 1];
    const b = rgbInput[i3 + 2];
    const rawLum = (0.299 * r + 0.587 * g + 0.114 * b) * a;

    if (isLightBackground) {
      // On a light/white background, foreground subject has solid base volume (0.34) + internal shading relief
      const dr = r - bgR;
      const dg = g - bgG;
      const db = b - bgB;
      const bgDist = Math.min(1.0, Math.sqrt((dr * dr + dg * dg + db * db) * 1.8));
      const internalShading = 0.55 * rawLum + 0.45 * (1.0 - rawLum * 0.5);
      luminance[i] = (0.34 + 0.66 * internalShading * (0.4 + 0.6 * bgDist)) * a;
    } else {
      // On a dark background, ensure interior foreground pixels keep a minimum base volume so internal dark details never vanish
      luminance[i] = Math.max(0.24 * a, rawLum);
    }
  }

  const kernel5 = [0.0625, 0.25, 0.375, 0.25, 0.0625];
  for (let pass = 0; pass < 2; pass++) {
    const src = pass === 0 ? luminance : lowFreqVolume;
    for (let y = 0; y < h; y++) {
      const rowOffset = y * w;
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let k = -2; k <= 2; k++) {
          const sx = Math.min(w - 1, Math.max(0, x + k));
          sum += src[rowOffset + sx] * kernel5[k + 2];
        }
        blurTemp[rowOffset + x] = sum;
      }
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let k = -2; k <= 2; k++) {
          const sy = Math.min(h - 1, Math.max(0, y + k));
          sum += blurTemp[sy * w + x] * kernel5[k + 2];
        }
        lowFreqVolume[y * w + x] = sum;
      }
    }
  }

  let maxDepth = 1e-5;
  for (let y = 0; y < h; y++) {
    const ym1 = Math.max(0, y - 1) * w;
    const y0 = y * w;
    const yp1 = Math.min(h - 1, y + 1) * w;

    for (let x = 0; x < w; x++) {
      const xm1 = Math.max(0, x - 1);
      const xp1 = Math.min(w - 1, x + 1);

      const tl = luminance[ym1 + xm1];
      const tc = luminance[ym1 + x];
      const tr = luminance[ym1 + xp1];
      const ml = luminance[y0 + xm1];
      const mc = luminance[y0 + x];
      const mr = luminance[y0 + xp1];
      const bl = luminance[yp1 + xm1];
      const bc = luminance[yp1 + x];
      const br = luminance[yp1 + xp1];

      const gx = (3 * tr + 10 * mr + 3 * br - (3 * tl + 10 * ml + 3 * bl)) / 16;
      const gy = (3 * bl + 10 * bc + 3 * br - (3 * tl + 10 * tc + 3 * tr)) / 16;
      const lap =
        mc * 4.0 - (tc + bc + ml + mr) * 0.7 - (tl + tr + bl + br) * 0.3;

      const idx = y0 + x;
      convGradX[idx] = gx;
      convGradY[idx] = gy;
      const activeLap = lap > 0 ? lap : lap * 0.15;
      convLaplacian[idx] = activeLap;

      if (exteriorBgMask[idx] === 1) {
        rawCnnDepth[idx] = 0.0;
      } else {
        const fused =
          0.52 * lowFreqVolume[idx] +
          0.36 * mc +
          0.12 * Math.min(1.0, Math.abs(activeLap) * 2.0);
        rawCnnDepth[idx] = fused;
        if (fused > maxDepth) maxDepth = fused;
      }
    }
  }

  const invMax = 1.0 / maxDepth;
  for (let i = 0; i < TENSOR_LEN; i++) {
    rawCnnDepth[i] =
      exteriorBgMask[i] === 1
        ? 0.0
        : Math.max(0.18, Math.min(1, rawCnnDepth[i] * invMax));
  }

  return performance.now() - t0;
}

const _reliefTemp = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);

/**
 * Joint bilateral filter on the depth map, guided by the colour image.
 * Pixels only average with nearby pixels of similar colour, so a flat plate
 * becomes smooth but the step to the next plate (different colour) survives.
 */
function smoothReliefEdgeAware(
  depth: Float32Array,
  rgb: Float32Array,
  bgMask: Uint8Array,
  strength: number
): void {
  if (strength <= 0.001) return;
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const radius = 3;
  const sigmaS = 2.0;
  const sigmaR = 0.06 + 0.1 * strength; // colour distance tolerance
  const invS = -1 / (2 * sigmaS * sigmaS);
  const invR = -1 / (2 * sigmaR * sigmaR);
  const passes = strength > 0.5 ? 2 : 1;

  for (let pass = 0; pass < passes; pass++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (bgMask[i] === 1) {
          _reliefTemp[i] = depth[i];
          continue;
        }
        const r0 = rgb[i * 3];
        const g0 = rgb[i * 3 + 1];
        const b0 = rgb[i * 3 + 2];
        let sum = 0;
        let wsum = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= h) continue;
          for (let dx = -radius; dx <= radius; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= w) continue;
            const j = yy * w + xx;
            if (bgMask[j] === 1) continue;
            const dr = rgb[j * 3] - r0;
            const dg = rgb[j * 3 + 1] - g0;
            const db = rgb[j * 3 + 2] - b0;
            const wt = Math.exp((dx * dx + dy * dy) * invS + (dr * dr + dg * dg + db * db) * invR);
            sum += depth[j] * wt;
            wsum += wt;
          }
        }
        const filtered = wsum > 0 ? sum / wsum : depth[i];
        _reliefTemp[i] = depth[i] + (filtered - depth[i]) * strength;
      }
    }
    depth.set(_reliefTemp);
  }
}

export function solvePoissonSurfaceFromNormals(
  buffers: CNNReconstructionBuffers,
  params: ReconstructionParams
): number {
  const t0 = performance.now();
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;

  const {
    rgbInput,
    convGradX,
    convGradY,
    convLaplacian,
    rawCnnDepth,
    blurTemp,
    slopeP,
    slopeQ,
    divergenceField,
    poissonSurface,
    customNormalTilt,
    depthTensor,
    normalTensor,
    zoneMap,
    lineCoherenceTensor,
    lineTangentX,
    lineTangentY,
  } = buffers;

  const isDirectRGBNormalMap =
    params.normalReconMode === NormalReconstructionMode.DirectNormalMapRGB;
  const ySign = params.invertNormalGreenY ? -1.0 : 1.0;
  const slopeGain = params.gradientSlopeGain;

  const detectLines = params.detectStraightLines !== false;
  const lineSens = Math.max(0.0, Math.min(1.0, params.straightLineSensitivity ?? 0.78));
  const facetPlanarity = Math.max(0.0, Math.min(1.0, params.flatFacetPlanarity ?? 0.68));

  // Evaluate 2D Structure Tensor Anisotropy & Collinear Straight-Line Coherence (Zero-GC)
  computeStraightLineStructureTensorZeroGC(
    convGradX,
    convGradY,
    buffers.exteriorBgMask,
    lineCoherenceTensor,
    lineTangentX,
    lineTangentY,
    lineSens
  );

  for (let i = 0; i < TENSOR_LEN; i++) {
    const i3 = i * 3;
    const i2 = i * 2;
    let nx = 0;
    let ny = 0;
    let nz = 1.0;

    if (isDirectRGBNormalMap) {
      nx = (rgbInput[i3] * 2.0 - 1.0) * slopeGain;
      ny = (rgbInput[i3 + 1] * 2.0 - 1.0) * ySign * slopeGain;
      nz = Math.max(0.18, rgbInput[i3 + 2] * 2.0 - 1.0);
    } else {
      nx = -convGradX[i] * 3.4 * slopeGain;
      ny = convGradY[i] * 3.4 * ySign * slopeGain;
      nz = 1.0;
    }

    // When Straight-Line Detection is active, project gradients onto the orthogonal normal of the detected straight line
    // so straight edges stay crisp and do not wobble or curve
    if (detectLines) {
      const coh = lineCoherenceTensor[i];
      if (coh > 0.15) {
        const tx = lineTangentX[i];
        const ty = lineTangentY[i];
        // Component along tangent is dampened so slope along a straight line is flat (0 curvature along line!)
        const dotTan = nx * tx + ny * ty;
        const damp = coh * lineSens * 0.85;
        nx -= dotTan * tx * damp;
        ny -= dotTan * ty * damp;
      }
    }

    nx += customNormalTilt[i2];
    ny += customNormalTilt[i2 + 1];

    const invLen = 1.0 / Math.sqrt(nx * nx + ny * ny + nz * nz);
    nx *= invLen;
    ny *= invLen;
    nz *= invLen;

    normalTensor[i3] = nx;
    normalTensor[i3 + 1] = ny;
    normalTensor[i3 + 2] = nz;

    const safeNz = Math.max(0.22, nz);
    slopeP[i] = -nx / safeNz;
    slopeQ[i] = ny / safeNz;
  }

  for (let y = 0; y < h; y++) {
    const ym1 = Math.max(0, y - 1) * w;
    const y0 = y * w;
    const yp1 = Math.min(h - 1, y + 1) * w;

    for (let x = 0; x < w; x++) {
      const xm1 = Math.max(0, x - 1);
      const xp1 = Math.min(w - 1, x + 1);
      const idx = y0 + x;

      const dpdx = 0.5 * (slopeP[y0 + xp1] - slopeP[y0 + xm1]);
      const dqdy = 0.5 * (slopeQ[yp1 + x] - slopeQ[ym1 + x]);
      let div = dpdx + dqdy;

      // Anti-Mound ("Evitar Montículos"):
      // In Poisson integration, negative divergence inside flat areas inflates surfaces into rounded parabolic mounds (Z ~ 1 - r^2).
      // When detectLines is enabled, suppress interior mound divergence in low-gradient flat zones while keeping sharp step edges!
      if (detectLines && facetPlanarity > 0.01) {
        const gMagSq = convGradX[idx] * convGradX[idx] + convGradY[idx] * convGradY[idx];
        const coh = lineCoherenceTensor[idx];
        const flatRegionMask = Math.max(0.0, 1.0 - Math.min(1.0, gMagSq * 18.0 + coh * 0.9));
        // Dampen divergence inside flat facets so Poisson solves Laplace's equation (∇²Z = 0 -> perfectly planar facet, zero mounds!)
        div *= 1.0 - flatRegionMask * facetPlanarity * 0.88;
      }

      divergenceField[idx] = div;
      poissonSurface[idx] = isDirectRGBNormalMap
        ? 0.0
        : rawCnnDepth[idx] * 0.5;
    }
  }

  const iterations = Math.max(8, Math.min(120, params.poissonIterations));
  const baseLambda =
    params.normalReconMode === NormalReconstructionMode.PurePoissonIntegration ||
    isDirectRGBNormalMap
      ? 0.015
      : params.screenedLambda;
  // Boost screened anchoring when anti-mound planarity is active so flat plateaus stay level instead of ballooning
  const lambda = detectLines
    ? baseLambda + facetPlanarity * 0.22
    : baseLambda;
  const invDiag = 1.0 / (4.0 + lambda);
  const omega = 1.68;

  const { exteriorBgMask } = buffers;

  for (let iter = 0; iter < iterations; iter++) {
    for (let y = 1; y < h - 1; y++) {
      const y0 = y * w;
      const ym1 = y0 - w;
      const yp1 = y0 + w;

      for (let x = 1; x < w - 1; x++) {
        const idx = y0 + x;
        if (!isDirectRGBNormalMap && exteriorBgMask[idx] === 1) {
          poissonSurface[idx] = 0.0;
          continue;
        }
        const neighborSum =
          poissonSurface[idx - 1] +
          poissonSurface[idx + 1] +
          poissonSurface[ym1 + x] +
          poissonSurface[yp1 + x];

        const prior = isDirectRGBNormalMap ? 0.0 : rawCnnDepth[idx];
        const targetZ =
          (neighborSum - divergenceField[idx] * 0.65 + lambda * prior) *
          invDiag;

        poissonSurface[idx] += omega * (targetZ - poissonSurface[idx]);
      }
    }
  }

  let minZ = Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < TENSOR_LEN; i++) {
    if (!isDirectRGBNormalMap && exteriorBgMask[i] === 1) continue;
    const z = poissonSurface[i];
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minZ) || !Number.isFinite(maxZ)) {
    minZ = 0.0;
    maxZ = 1.0;
  }

  const rangeZ = Math.max(1e-5, maxZ - minZ);
  const invRangeZ = 1.0 / rangeZ;
  const normWeight =
    params.normalReconMode === NormalReconstructionMode.PurePoissonIntegration ||
    isDirectRGBNormalMap
      ? 1.0
      : params.normalIntegrationWeight;

  for (let i = 0; i < TENSOR_LEN; i++) {
    if (!isDirectRGBNormalMap && exteriorBgMask[i] === 1) {
      poissonSurface[i] = 0.0;
      depthTensor[i] = 0.0;
      zoneMap[i] = 2;
      continue;
    }
    const normZ = Math.max(0.0, Math.min(1.0, (poissonSurface[i] - minZ) * invRangeZ));
    poissonSurface[i] = normZ;

    let finalD = normZ * normWeight + rawCnnDepth[i] * (1.0 - normWeight);

    // Anti-Mound Planar Facet Stabilization ("Evitar Montículos"):
    // 1. Replace puffy lowFreqVolume dome component with crisp terraced planar facets in non-curved regions
    // 2. Keep straight-line architectural creases sharp without rounded pillow domes
    if (detectLines && facetPlanarity > 0.01) {
      const coh = lineCoherenceTensor[i];
      const rawNoMound = Math.max(0.0, rawCnnDepth[i] - buffers.lowFreqVolume[i] * 0.26 * facetPlanarity);
      // Soft terrace quantization (10 planar levels) to eliminate micro-mounds on flat armor/geometric plates
      const levels = 10.0;
      const terraced = Math.round(rawNoMound * levels) / levels;
      const planarTarget = rawNoMound * (1.0 - 0.38 * facetPlanarity) + terraced * (0.38 * facetPlanarity);
      const blendPlanar = facetPlanarity * (0.55 + 0.35 * coh);
      finalD = finalD * (1.0 - blendPlanar) + planarTarget * blendPlanar;
    }

    depthTensor[i] = Math.max(0.0, Math.min(1.0, finalD));

    const divMag = Math.abs(divergenceField[i]);
    const lapMag = Math.abs(convLaplacian[i]);

    if (divMag > 0.24 || (lapMag > 0.18 && finalD > 0.22) || (detectLines && lineCoherenceTensor[i] > 0.58)) {
      zoneMap[i] = 3;
    } else if (finalD >= 0.58) {
      zoneMap[i] = 0;
    } else if (finalD >= 0.26) {
      zoneMap[i] = 1;
    } else {
      zoneMap[i] = 2;
    }
  }

  // Optional Multi-Pass Gaussian Tensor Pre-Smoothing (anisotropic when detectStraightLines is active so straight lines do not blur)
  const tensorBlurPasses = Math.max(0, Math.min(8, params.tensorBlurPasses || 0));
  const kernel5 = [0.0625, 0.25, 0.375, 0.25, 0.0625];
  for (let pass = 0; pass < tensorBlurPasses; pass++) {
    for (let y = 0; y < h; y++) {
      const rowOffset = y * w;
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let k = -2; k <= 2; k++) {
          const sx = Math.min(w - 1, Math.max(0, x + k));
          sum += depthTensor[rowOffset + sx] * kernel5[k + 2];
        }
        blurTemp[rowOffset + x] = sum;
      }
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let k = -2; k <= 2; k++) {
          const sy = Math.min(h - 1, Math.max(0, y + k));
          sum += blurTemp[sy * w + x] * kernel5[k + 2];
        }
        depthTensor[y * w + x] = sum;
      }
    }
  }

  // Edge-aware relief smoothing: removes lumps/spikes caused by shading and
  // specular highlights while keeping the borders between armour pieces.
  smoothReliefEdgeAware(
    depthTensor,
    rgbInput,
    buffers.exteriorBgMask,
    Math.max(0.0, Math.min(1.0, params.reliefEdgeSmoothing ?? 0.7))
  );

  // Evaluate Outer Frame & Color / Hue Tone Cutout Mask (Zero-GC)
  evaluateBackgroundCutoutMask(buffers, params);

  // Synthesize or process Back Image & Side Seam Image Tensors (Zero-GC)
  evaluateMultiViewBackAndSideTensors(buffers, params);

  return performance.now() - t0;
}

/**
 * Computes the 2D Structure Tensor J = [Jxx, Jxy; Jxy, Jyy] and Collinear Straight-Line Coherence (Zero-GC).
 * Detects straight geometric lines so they do not curve or inflate into rounded mounds ("montículos").
 */
function computeStraightLineStructureTensorZeroGC(
  gradX: Float32Array,
  gradY: Float32Array,
  exteriorMask: Uint8Array,
  outCoherence: Float32Array,
  outTanX: Float32Array,
  outTanY: Float32Array,
  sensitivity: number
): void {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const sensScale = 0.6 + sensitivity * 1.15;

  for (let y = 0; y < h; y++) {
    const rowOff = y * w;
    for (let x = 0; x < w; x++) {
      const idx = rowOff + x;
      if (exteriorMask[idx] === 1 || x < 2 || x >= w - 2 || y < 2 || y >= h - 2) {
        outCoherence[idx] = 0.0;
        outTanX[idx] = 1.0;
        outTanY[idx] = 0.0;
        continue;
      }

      // Accumulate 3x3 Structure Tensor J = sum(w * [gx^2, gx*gy; gx*gy, gy^2])
      let jxx = 0.0;
      let jxy = 0.0;
      let jyy = 0.0;

      for (let dy = -1; dy <= 1; dy++) {
        const nr = (y + dy) * w;
        const wy = dy === 0 ? 0.5 : 0.25;
        for (let dx = -1; dx <= 1; dx++) {
          const wx = dx === 0 ? 0.5 : 0.25;
          const weight = wx * wy;
          const nIdx = nr + (x + dx);
          const gx = gradX[nIdx];
          const gy = gradY[nIdx];
          jxx += weight * gx * gx;
          jxy += weight * gx * gy;
          jyy += weight * gy * gy;
        }
      }

      const trace = jxx + jyy;
      if (trace < 1e-5) {
        outCoherence[idx] = 0.0;
        outTanX[idx] = 1.0;
        outTanY[idx] = 0.0;
        continue;
      }

      const diff = jxx - jyy;
      const anisotropy = Math.sqrt(diff * diff + 4.0 * jxy * jxy) / (trace + 1e-5);

      // Principal gradient orientation angle: 2*theta = atan2(2*Jxy, Jxx - Jyy)
      const gradAngle = 0.5 * Math.atan2(2.0 * jxy, diff);
      // Straight line tangent is orthogonal to principal gradient: (-sin(gradAngle), cos(gradAngle))
      const tx = -Math.sin(gradAngle);
      const ty = Math.cos(gradAngle);
      outTanX[idx] = tx;
      outTanY[idx] = ty;

      // Verify collinear straightness by sampling gradients at ±2 pixels along the tangent direction (tx, ty)
      const stepX = Math.round(tx * 2.0);
      const stepY = Math.round(ty * 2.0);
      const fwdIdx = (y + stepY) * w + (x + stepX);
      const bwdIdx = (y - stepY) * w + (x - stepX);

      const gx0 = gradX[idx];
      const gy0 = gradY[idx];
      const gLen0Sq = gx0 * gx0 + gy0 * gy0;

      let collinearBoost = 0.65;
      if (gLen0Sq > 1e-5) {
        const gxF = gradX[fwdIdx];
        const gyF = gradY[fwdIdx];
        const gxB = gradX[bwdIdx];
        const gyB = gradY[bwdIdx];
        const lenFSq = gxF * gxF + gyF * gyF;
        const lenBSq = gxB * gxB + gyB * gyB;
        if (lenFSq > 1e-5 && lenBSq > 1e-5) {
          const dotF = (gx0 * gxF + gy0 * gyF) / Math.sqrt(gLen0Sq * lenFSq);
          const dotB = (gx0 * gxB + gy0 * gyB) / Math.sqrt(gLen0Sq * lenBSq);
          // High dotF & dotB near +1 means the edge continues in a straight line without curving!
          if (dotF > 0.7 && dotB > 0.7) {
            collinearBoost = 0.5 * (dotF + dotB) * 1.25;
          } else {
            collinearBoost = 0.35;
          }
        }
      }

      const energyGate = Math.min(1.0, trace * 14.0);
      const rawCoherence = anisotropy * collinearBoost * energyGate * sensScale;
      outCoherence[idx] = Math.max(0.0, Math.min(1.0, rawCoherence));
    }
  }
}

/**
 * Processes an uploaded Back Image ("Imagen de la Parte de Atrás") into customBackRgb, customBackDepth,
 * customBackFgMask, and per-row silhouette bounds (backRowSilhouetteMinU/MaxU) using Zero-GC border flood-fill
 * so the Back Image automatically conforms to the 3D Mold and shared silhouette.
 */
export function processUploadedBackImage(
  sourceCanvas: HTMLCanvasElement,
  buffers: CNNReconstructionBuffers
): void {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const ctx = sourceCanvas.getContext('2d');
  if (!ctx) return;

  const imgData = ctx.getImageData(0, 0, w, h).data;
  const {
    customBackRgb,
    customBackAlpha,
    customBackDepth,
    customBackFgMask,
    backRowSilhouetteMinU,
    backRowSilhouetteMaxU,
    backRowTempMinU,
    backRowTempMaxU,
    floodVisited,
    floodQueue,
    morphTempMask,
  } = buffers;

  // Sample border perimeter of the back image to auto-detect its background color & luminance
  let bgR = 0;
  let bgG = 0;
  let bgB = 0;
  let borderLumSum = 0;
  let borderAlphaSum = 0;
  let borderSamples = 0;
  for (let x = 0; x < w; x += 2) {
    const top4 = x * 4;
    const bot4 = ((h - 1) * w + x) * 4;
    const tr = imgData[top4] / 255;
    const tg = imgData[top4 + 1] / 255;
    const tb = imgData[top4 + 2] / 255;
    const br = imgData[bot4] / 255;
    const bg = imgData[bot4 + 1] / 255;
    const bb = imgData[bot4 + 2] / 255;
    bgR += tr + br;
    bgG += tg + bg;
    bgB += tb + bb;
    borderLumSum += 0.299 * (tr + br) + 0.587 * (tg + bg) + 0.114 * (tb + bb);
    borderAlphaSum += (imgData[top4 + 3] + imgData[bot4 + 3]) / 255;
    borderSamples += 2;
  }
  const invSamples = 1.0 / Math.max(1, borderSamples);
  bgR *= invSamples;
  bgG *= invSamples;
  bgB *= invSamples;
  buffers.backAutoBgR = bgR;
  buffers.backAutoBgG = bgG;
  buffers.backAutoBgB = bgB;

  const avgBorderLum = borderLumSum * invSamples;
  const avgBorderAlpha = borderAlphaSum * invSamples;
  const isLightBackBg = avgBorderAlpha > 0.5 && avgBorderLum > 0.58;
  const bgTolSq = 0.048; // Squared RGB tolerance for background candidate classification

  let maxD = 1e-5;
  for (let i = 0; i < TENSOR_LEN; i++) {
    const i4 = i * 4;
    const i3 = i * 3;
    const r = imgData[i4] / 255;
    const g = imgData[i4 + 1] / 255;
    const b = imgData[i4 + 2] / 255;
    const a = imgData[i4 + 3] / 255;

    customBackRgb[i3] = r;
    customBackRgb[i3 + 1] = g;
    customBackRgb[i3 + 2] = b;
    customBackAlpha[i] = a;

    const rawLum = 0.299 * r + 0.587 * g + 0.114 * b;
    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    const sat = maxC > 0.02 ? (maxC - minC) / maxC : 0;

    const dr = r - bgR;
    const dg = g - bgG;
    const db = b - bgB;
    const distBgSq = dr * dr + dg * dg + db * db;

    // Candidate background if transparent or close to perimeter background color
    const isBgCandidate = a < 0.12 || distBgSq <= bgTolSq;
    morphTempMask[i] = isBgCandidate ? 1 : 0;

    let dVal = (rawLum * 0.72 + sat * 0.28) * a;
    if (isLightBackBg) {
      const distFromBg = Math.max(0, avgBorderLum - rawLum);
      dVal =
        distBgSq <= bgTolSq * 0.65 && sat < 0.12
          ? 0.0
          : Math.min(
              1.0,
              (distFromBg * 0.75 + sat * 0.35 + (1.0 - rawLum * 0.35) * 0.3) * a
            );
    } else {
      const distFromDark = Math.sqrt(distBgSq);
      dVal = Math.min(1.0, (rawLum * 0.62 + distFromDark * 0.42 + sat * 0.22) * a);
    }

    customBackDepth[i] = dVal;
    if (dVal > maxD) maxD = dVal;
  }

  // Zero-GC Flood-Fill from outer border of Back Image through morphTempMask === 1
  // so internal dark/light details inside the back character remain 100% solid foreground
  floodVisited.fill(0);
  let qHead = 0;
  let qTail = 0;
  for (let i = 0; i < TENSOR_LEN; i++) {
    const tx = i % w;
    const ty = (i / w) | 0;
    if (
      (tx === 0 || tx === w - 1 || ty === 0 || ty === h - 1) &&
      morphTempMask[i] === 1
    ) {
      floodVisited[i] = 1;
      floodQueue[qTail++] = i;
    }
  }
  while (qHead < qTail) {
    const curr = floodQueue[qHead++];
    const cx = curr % w;
    const cy = (curr / w) | 0;
    if (cx > 0) {
      const n = curr - 1;
      if (floodVisited[n] === 0 && morphTempMask[n] === 1) {
        floodVisited[n] = 1;
        floodQueue[qTail++] = n;
      }
    }
    if (cx < w - 1) {
      const n = curr + 1;
      if (floodVisited[n] === 0 && morphTempMask[n] === 1) {
        floodVisited[n] = 1;
        floodQueue[qTail++] = n;
      }
    }
    if (cy > 0) {
      const n = curr - w;
      if (floodVisited[n] === 0 && morphTempMask[n] === 1) {
        floodVisited[n] = 1;
        floodQueue[qTail++] = n;
      }
    }
    if (cy < h - 1) {
      const n = curr + w;
      if (floodVisited[n] === 0 && morphTempMask[n] === 1) {
        floodVisited[n] = 1;
        floodQueue[qTail++] = n;
      }
    }
  }

  const invMax = 1.0 / maxD;
  let fgCount = 0;
  let minX = w - 1;
  let maxX = 0;
  let minY = h - 1;
  let maxY = 0;

  for (let ty = 0; ty < h; ty++) {
    const rOff = ty * w;
    let rMin = w;
    let rMax = -1;
    for (let tx = 0; tx < w; tx++) {
      const idx = rOff + tx;
      const isFg = floodVisited[idx] === 1 ? 0 : 1;
      customBackFgMask[idx] = isFg;
      customBackDepth[idx] =
        isFg === 0 ? 0.0 : Math.max(0.14, Math.min(1.0, customBackDepth[idx] * invMax));

      if (isFg === 1) {
        fgCount++;
        if (tx < minX) minX = tx;
        if (tx > maxX) maxX = tx;
        if (ty < minY) minY = ty;
        if (ty > maxY) maxY = ty;
        if (tx < rMin) rMin = tx;
        if (tx > rMax) rMax = tx;
      }
    }
    if (rMax >= rMin) {
      backRowTempMinU[ty] = rMin / (w - 1);
      backRowTempMaxU[ty] = rMax / (w - 1);
    } else {
      backRowTempMinU[ty] = -1.0;
      backRowTempMaxU[ty] = -1.0;
    }
  }

  if (fgCount > 28 && fgCount < TENSOR_LEN * 0.985 && maxX > minX && maxY > minY) {
    buffers.backAutoSilhouetteMinU = minX / (w - 1);
    buffers.backAutoSilhouetteMaxU = maxX / (w - 1);
    buffers.backAutoSilhouetteMinV = 1.0 - maxY / (h - 1);
    buffers.backAutoSilhouetteMaxV = 1.0 - minY / (h - 1);
  } else {
    buffers.backAutoSilhouetteMinU = 0.08;
    buffers.backAutoSilhouetteMaxU = 0.92;
    buffers.backAutoSilhouetteMinV = 0.06;
    buffers.backAutoSilhouetteMaxV = 0.94;
  }

  buffers.backAutoSilhouetteWidth = Math.max(
    0.16,
    buffers.backAutoSilhouetteMaxU - buffers.backAutoSilhouetteMinU
  );
  buffers.backAutoSilhouetteHeight = Math.max(
    0.16,
    buffers.backAutoSilhouetteMaxV - buffers.backAutoSilhouetteMinV
  );

  for (let ty = 0; ty < h; ty++) {
    if (backRowTempMaxU[ty] < 0.0) {
      backRowTempMinU[ty] = buffers.backAutoSilhouetteMinU;
      backRowTempMaxU[ty] = buffers.backAutoSilhouetteMaxU;
    }
  }
  for (let ty = 0; ty < h; ty++) {
    let sMin = 0.0;
    let sMax = 0.0;
    let cnt = 0;
    for (let dy = -2; dy <= 2; dy++) {
      const sy = Math.min(h - 1, Math.max(0, ty + dy));
      sMin += backRowTempMinU[sy];
      sMax += backRowTempMaxU[sy];
      cnt++;
    }
    backRowSilhouetteMinU[ty] = sMin / cnt;
    backRowSilhouetteMaxU[ty] = sMax / cnt;
  }

  buffers.hasCustomBackImage = true;
}

/**
 * Processes an uploaded Lateral/Side Image ("Imagen del Lateral") into customSideRgb and customSideDepth.
 */
export function processUploadedSideImage(
  sourceCanvas: HTMLCanvasElement,
  buffers: CNNReconstructionBuffers
): void {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const ctx = sourceCanvas.getContext('2d');
  if (!ctx) return;

  const imgData = ctx.getImageData(0, 0, w, h).data;
  const { customSideRgb, customSideDepth } = buffers;

  let maxD = 1e-5;
  for (let i = 0; i < TENSOR_LEN; i++) {
    const i4 = i * 4;
    const i3 = i * 3;
    const r = imgData[i4] / 255;
    const g = imgData[i4 + 1] / 255;
    const b = imgData[i4 + 2] / 255;
    const a = imgData[i4 + 3] / 255;

    customSideRgb[i3] = r;
    customSideRgb[i3 + 1] = g;
    customSideRgb[i3 + 2] = b;

    const lum = (0.299 * r + 0.587 * g + 0.114 * b) * a;
    customSideDepth[i] = lum;
    if (lum > maxD) maxD = lum;
  }

  const invMax = 1.0 / maxD;
  for (let i = 0; i < TENSOR_LEN; i++) {
    customSideDepth[i] = Math.max(0, Math.min(1, customSideDepth[i] * invMax));
  }
  buffers.hasCustomSideImage = true;
}

/**
 * Renders the active Back Image ("Parte de Atrás") into a 2D preview canvas with cutout checkerboard
 * and subtle 3D mold / dorsal normal shading so the user sees the molded Back Image directly.
 */
export function renderBackPreviewToCanvas(
  buffers: CNNReconstructionBuffers,
  targetCanvas: HTMLCanvasElement
): void {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  targetCanvas.width = w;
  targetCanvas.height = h;
  const ctx = targetCanvas.getContext('2d');
  if (!ctx) return;

  const imgData = ctx.createImageData(w, h);
  const data = imgData.data;

  for (let i = 0; i < TENSOR_LEN; i++) {
    const i4 = i * 4;
    const i3 = i * 3;
    if (buffers.texelCutoutMask[i] === 0) {
      const tx = i % w;
      const ty = (i / w) | 0;
      const checker = ((tx >> 3) ^ (ty >> 3)) & 1;
      const cVal = checker ? 18 : 10;
      data[i4] = cVal;
      data[i4 + 1] = cVal + 3;
      data[i4 + 2] = cVal + 8;
      data[i4 + 3] = 255;
      continue;
    }
    const bDepth = buffers.backDepthTensor[i];
    const bnz = buffers.backNormalTensor[i3 + 2];
    const moldShade = 0.86 + 0.14 * bDepth + 0.06 * bnz;
    data[i4] = Math.min(
      255,
      Math.max(0, Math.round(buffers.backRgbActive[i3] * moldShade * 255))
    );
    data[i4 + 1] = Math.min(
      255,
      Math.max(0, Math.round(buffers.backRgbActive[i3 + 1] * moldShade * 255))
    );
    data[i4 + 2] = Math.min(
      255,
      Math.max(0, Math.round(buffers.backRgbActive[i3 + 2] * moldShade * 255))
    );
    data[i4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

/**
 * Renders the active Side / Lateral Band Image ("El Lateral") into a 2D preview canvas.
 */
export function renderSidePreviewToCanvas(
  buffers: CNNReconstructionBuffers,
  targetCanvas: HTMLCanvasElement
): void {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  targetCanvas.width = w;
  targetCanvas.height = h;
  const ctx = targetCanvas.getContext('2d');
  if (!ctx) return;

  const imgData = ctx.createImageData(w, h);
  const data = imgData.data;

  for (let i = 0; i < TENSOR_LEN; i++) {
    const i4 = i * 4;
    const i3 = i * 3;
    data[i4] = Math.min(255, Math.max(0, Math.round(buffers.sideRgbActive[i3] * 255)));
    data[i4 + 1] = Math.min(255, Math.max(0, Math.round(buffers.sideRgbActive[i3 + 1] * 255)));
    data[i4 + 2] = Math.min(255, Math.max(0, Math.round(buffers.sideRgbActive[i3 + 2] * 255)));
    data[i4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

/**
 * Evaluates and updates the active Back Image and Side Lateral Seam Band tensors (Zero-GC).
 * Automatically molds and aligns the Back Image to the 3D Preset Base Shape and shared silhouette,
 * and runs Screened Poisson integration + Low-Frequency Pillow extraction on the Back Image.
 */
function evaluateMultiViewBackAndSideTensors(
  buffers: CNNReconstructionBuffers,
  params: ReconstructionParams
): void {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const {
    rgbInput,
    depthTensor,
    texelCutoutMask,
    customBackRgb,
    customBackDepth,
    backRowSilhouetteMinU,
    backRowSilhouetteMaxU,
    rowSilhouetteMinU,
    rowSilhouetteMaxU,
    backRgbActive,
    backDepthTensor,
    backNormalTensor,
    backLowFreqVolume,
    backDivergenceField,
    backPoissonSurface,
    blurTemp,
    customSideRgb,
    customSideDepth,
    sideRgbActive,
    sideDepthTensor,
    sideNormalTensor,
  } = buffers;

  const {
    exteriorBgMask,
    candidateBgMask,
    floodQueue,
    frontInpaintedRgb,
    backInpaintedRgb,
    nearestFgTexelIdx,
  } = buffers;

  // ============================================================================
  // STEP 0: ZERO-GC VORONOÏ FOREGROUND COLOR & DEPTH PROPAGATION FOR SEAMS/JOINTS
  // Strips the 1-2px background anti-aliasing halo from the silhouette edge and
  // propagates clean interior image colors outward into all joints, bridged zones,
  // and silhouette borders so the joint ALWAYS takes the color of the image!
  // ============================================================================
  nearestFgTexelIdx.fill(-1);
  let qHead = 0;
  let qTail = 0;
  const bgR = buffers.autoDetectedBgR;
  const bgG = buffers.autoDetectedBgG;
  const bgB = buffers.autoDetectedBgB;

  for (let ty = 1; ty < h - 1; ty++) {
    const rOff = ty * w;
    for (let tx = 1; tx < w - 1; tx++) {
      const idx = rOff + tx;
      if (
        texelCutoutMask[idx] === 1 &&
        exteriorBgMask[idx] === 0 &&
        candidateBgMask[idx] === 0 &&
        exteriorBgMask[idx - 1] === 0 &&
        exteriorBgMask[idx + 1] === 0 &&
        exteriorBgMask[idx - w] === 0 &&
        exteriorBgMask[idx + w] === 0 &&
        candidateBgMask[idx - 1] === 0 &&
        candidateBgMask[idx + 1] === 0 &&
        candidateBgMask[idx - w] === 0 &&
        candidateBgMask[idx + w] === 0
      ) {
        const i3 = idx * 3;
        const dr = rgbInput[i3] - bgR;
        const dg = rgbInput[i3 + 1] - bgG;
        const db = rgbInput[i3 + 2] - bgB;
        if (dr * dr + dg * dg + db * db > 0.022) {
          nearestFgTexelIdx[idx] = idx;
          floodQueue[qTail++] = idx;
        }
      }
    }
  }

  // Fallback for ultra-thin line art or full-frame textures
  if (qTail < 32) {
    qTail = 0;
    for (let i = 0; i < TENSOR_LEN; i++) {
      if (texelCutoutMask[i] === 1 && exteriorBgMask[i] === 0) {
        nearestFgTexelIdx[i] = i;
        floodQueue[qTail++] = i;
      }
    }
  }
  if (qTail === 0) {
    for (let i = 0; i < TENSOR_LEN; i++) {
      nearestFgTexelIdx[i] = i;
    }
  } else {
    while (qHead < qTail) {
      const curr = floodQueue[qHead++];
      const seedIdx = nearestFgTexelIdx[curr];
      const cx = curr % w;
      const cy = (curr / w) | 0;

      if (cx > 0) {
        const n = curr - 1;
        if (nearestFgTexelIdx[n] === -1) {
          nearestFgTexelIdx[n] = seedIdx;
          floodQueue[qTail++] = n;
        }
      }
      if (cx < w - 1) {
        const n = curr + 1;
        if (nearestFgTexelIdx[n] === -1) {
          nearestFgTexelIdx[n] = seedIdx;
          floodQueue[qTail++] = n;
        }
      }
      if (cy > 0) {
        const n = curr - w;
        if (nearestFgTexelIdx[n] === -1) {
          nearestFgTexelIdx[n] = seedIdx;
          floodQueue[qTail++] = n;
        }
      }
      if (cy < h - 1) {
        const n = curr + w;
        if (nearestFgTexelIdx[n] === -1) {
          nearestFgTexelIdx[n] = seedIdx;
          floodQueue[qTail++] = n;
        }
      }
    }
  }

  // Populate frontInpaintedRgb and smoothly bridge depthTensor across united gaps/joints
  for (let i = 0; i < TENSOR_LEN; i++) {
    const seedIdx = nearestFgTexelIdx[i] >= 0 ? nearestFgTexelIdx[i] : i;
    const i3 = i * 3;
    const s3 = seedIdx * 3;
    frontInpaintedRgb[i3] = rgbInput[s3];
    frontInpaintedRgb[i3 + 1] = rgbInput[s3 + 1];
    frontInpaintedRgb[i3 + 2] = rgbInput[s3 + 2];

    // If this pixel is part of an active bridged union zone (where raw image had background),
    // inherit smooth volume from the nearest foreground seed so the joint is covered without trenches!
    if (
      texelCutoutMask[i] === 1 &&
      (exteriorBgMask[i] === 1 || candidateBgMask[i] === 1) &&
      seedIdx !== i
    ) {
      const seedDepth = depthTensor[seedIdx];
      if (depthTensor[i] < seedDepth * 0.85) {
        depthTensor[i] = seedDepth * 0.85;
      }
    }
  }

  // Smooth the inpainted border/joint pixels (where nearestFgTexelIdx[i] !== i) using a 3x3 neighborhood average
  // so adjacent image regions blend their colors organically across any joint!
  for (let ty = 1; ty < h - 1; ty++) {
    const rOff = ty * w;
    for (let tx = 1; tx < w - 1; tx++) {
      const idx = rOff + tx;
      if (nearestFgTexelIdx[idx] === idx) continue;
      let sr = 0.0;
      let sg = 0.0;
      let sb = 0.0;
      for (let dy = -1; dy <= 1; dy++) {
        const nr = (ty + dy) * w;
        for (let dx = -1; dx <= 1; dx++) {
          const n3 = (nr + tx + dx) * 3;
          sr += frontInpaintedRgb[n3];
          sg += frontInpaintedRgb[n3 + 1];
          sb += frontInpaintedRgb[n3 + 2];
        }
      }
      const i3 = idx * 3;
      frontInpaintedRgb[i3] = sr / 9.0;
      frontInpaintedRgb[i3 + 1] = sg / 9.0;
      frontInpaintedRgb[i3 + 2] = sb / 9.0;
    }
  }

  // Compute average foreground RGB to tint procedural backplate & lateral edge bands realistically
  let avgR = 0;
  let avgG = 0;
  let avgB = 0;
  let fgCount = 0;
  for (let i = 0; i < TENSOR_LEN; i += 4) {
    if (texelCutoutMask[i] === 1) {
      const i3 = i * 3;
      avgR += frontInpaintedRgb[i3];
      avgG += frontInpaintedRgb[i3 + 1];
      avgB += frontInpaintedRgb[i3 + 2];
      fgCount++;
    }
  }
  if (fgCount > 0) {
    avgR /= fgCount;
    avgG /= fgCount;
    avgB /= fgCount;
  } else {
    avgR = 0.72;
    avgG = 0.58;
    avgB = 0.28;
  }

  const backMode = params.backSourceMode || BackImageSourceMode.SymmetricRelief;
  const mirrorX = params.mirrorBackImageX !== false;
  const useSymmetricRules = params.symmetricBackRules !== false;
  const detectLines = params.detectStraightLines !== false;
  const lineSens = Math.max(0.0, Math.min(1.0, params.straightLineSensitivity ?? 0.78));
  const facetPlanarity = Math.max(0.0, Math.min(1.0, params.flatFacetPlanarity ?? 0.68));

  // 3D Preset Base Shape & Silhouette Conformity parameters for the Back Image
  const presetShape = params.presetBaseShape || PresetBaseShape.None;
  const hasPresetShape = presetShape !== PresetBaseShape.None;
  const autoFitMold = params.autoFitMoldToSilhouette !== false;
  const silConformity = Math.max(0.0, Math.min(1.0, params.silhouetteConformity ?? 0.94));
  const moldStrength = Math.max(0.0, Math.min(1.6, params.shapeMoldStrength ?? 0.75));
  const wrapUVWeight = Math.max(0.0, Math.min(1.0, params.shapeWrapUV ?? 0.45));
  const userScaleX = Math.max(0.4, Math.min(1.8, params.shapeScaleX ?? 1.0));
  const userScaleY = Math.max(0.4, Math.min(1.8, params.shapeScaleY ?? 1.0));
  const userAspect = Math.max(0.5, Math.min(1.6, params.shapeAspectRatio ?? 1.0));
  const shapeOffX = Math.max(-0.35, Math.min(0.35, params.shapeOffsetX ?? 0.0));
  const shapeOffY = Math.max(-0.35, Math.min(0.35, params.shapeOffsetY ?? 0.0));
  const poseArmAngleDeg = Math.max(15.0, Math.min(90.0, params.poseArmAngleDeg ?? 35.0));
  const limbUnionMode = params.limbUnionMode || LimbUnionMode.TubularCylindersFull;
  const limbUnionCurvature = Math.max(0.05, Math.min(1.0, params.limbUnionCurvature ?? 0.68));
  const autoLimb = params.autoFitLimbSizes !== false;
  const effArmLen = Math.max(
    0.4,
    Math.min(
      1.85,
      (params.armLengthScale ?? 1.0) * (autoLimb ? buffers.autoDetectedArmLengthScale : 1.0)
    )
  );
  const effArmThick = Math.max(
    0.4,
    Math.min(
      1.85,
      (params.armThicknessScale ?? 1.0) * (autoLimb ? buffers.autoDetectedArmThicknessScale : 1.0)
    )
  );
  const effLegLen = Math.max(
    0.4,
    Math.min(
      1.85,
      (params.legLengthScale ?? 1.0) * (autoLimb ? buffers.autoDetectedLegLengthScale : 1.0)
    )
  );
  const effLegThick = Math.max(
    0.4,
    Math.min(
      1.85,
      (params.legThicknessScale ?? 1.0) * (autoLimb ? buffers.autoDetectedLegThicknessScale : 1.0)
    )
  );

  const autoScaleX = autoFitMold ? buffers.autoSilhouetteWidth / 0.78 : 1.0;
  const autoScaleY = autoFitMold ? buffers.autoSilhouetteHeight / 0.88 : 1.0;
  const shapeScaleX = Math.max(0.25, autoScaleX * userScaleX * userAspect);
  const shapeScaleY = Math.max(0.25, autoScaleY * userScaleY);
  const effMoldCenterX = (autoFitMold ? buffers.autoSilhouetteCenterX : 0.0) + shapeOffX;
  const effMoldCenterY = (autoFitMold ? buffers.autoSilhouetteCenterY : 0.0) + shapeOffY;

  const frontMinV = buffers.autoSilhouetteMinV;
  const frontSpanV = Math.max(0.12, buffers.autoSilhouetteMaxV - frontMinV);
  const backMinV = buffers.backAutoSilhouetteMinV;
  const backSpanV = Math.max(0.12, buffers.backAutoSilhouetteMaxV - backMinV);

  const {
    backRawCnnDepth,
    backConvGradX,
    backConvGradY,
    backConvLaplacian,
    backLineCoherence,
    backLineTangentX,
    backLineTangentY,
    backZoneMap,
  } = buffers;

  // 1. Populate Back Image RGB and Initial Depth Tensors (with automatic Silhouette & 3D Mold Conformity!)
  for (let ty = 0; ty < h; ty++) {
    const v = 1.0 - ty / (h - 1);
    const ny = (ty / (h - 1)) * 2.0 - 1.0;
    const rowOffset = ty * w;

    const fRowMinU = rowSilhouetteMinU[ty];
    const fRowMaxU = rowSilhouetteMaxU[ty];
    const fRowSpanU = Math.max(0.06, fRowMaxU - fRowMinU);
    const fRowCenterU = 0.5 * (fRowMinU + fRowMaxU) - 0.5;
    const rowAlignCenterX =
      effMoldCenterX * (1.0 - silConformity * 0.45) +
      (fRowCenterU + shapeOffX) * (silConformity * 0.45);

    // Conform vertical coordinate v from target silhouette span to Back Image's own silhouette span
    const relV = Math.max(0.0, Math.min(1.0, (v - frontMinV) / frontSpanV));
    const vBackAligned = backMinV + relV * backSpanV;
    const vBack = v * (1.0 - silConformity * 0.88) + vBackAligned * (silConformity * 0.88);
    const srcTyBase = Math.min(h - 1, Math.max(0, Math.round((1.0 - vBack) * (h - 1))));

    const bRowMinRaw = backRowSilhouetteMinU[srcTyBase];
    const bRowMaxRaw = backRowSilhouetteMaxU[srcTyBase];
    // Account for horizontal mirroring when aligning the Back Image's row span to the Front silhouette
    const bRowMinU = mirrorX ? 1.0 - bRowMaxRaw : bRowMinRaw;
    const bRowMaxU = mirrorX ? 1.0 - bRowMinRaw : bRowMaxRaw;
    const bRowSpanU = Math.max(0.06, bRowMaxU - bRowMinU);

    for (let tx = 0; tx < w; tx++) {
      const u = tx / (w - 1);
      const nx = u * 2.0 - 1.0;
      const dstIdx = rowOffset + tx;
      const dst3 = dstIdx * 3;

      // Evaluate 3D Preset Base Shape on the Back so the 3D Mold shapes the Back Image as well
      let moldBackZ = 0.0;
      let uSample = u;
      let vSample = v;
      if (hasPresetShape) {
        const uMold = 0.5 + (u - 0.5 - rowAlignCenterX);
        const vMold = 0.5 + (v - 0.5 - effMoldCenterY);
        evaluatePresetBaseShapeZeroGC(
          presetShape,
          uMold,
          vMold,
          shapeScaleX,
          shapeScaleY,
          poseArmAngleDeg,
          limbUnionMode,
          limbUnionCurvature,
          effArmLen,
          effArmThick,
          effLegLen,
          effLegThick
        );
        if (_shapeInside === 1) {
          moldBackZ = _shapeBackZ;
          if (wrapUVWeight > 0.01) {
            const warpLocalU = _shapeWarpU + rowAlignCenterX;
            const warpLocalV = _shapeWarpV + effMoldCenterY;
            uSample = Math.max(
              0.0,
              Math.min(1.0, u * (1.0 - wrapUVWeight * 0.65) + warpLocalU * (wrapUVWeight * 0.65))
            );
            vSample = Math.max(
              0.0,
              Math.min(1.0, v * (1.0 - wrapUVWeight * 0.65) + warpLocalV * (wrapUVWeight * 0.65))
            );
          }
        }
      }

      if (backMode === BackImageSourceMode.CustomBackUpload && buffers.hasCustomBackImage) {
        // Conform horizontal coordinate uSample from target row silhouette span [fRowMinU..fRowMaxU]
        // into the Back Image's own row silhouette span [bRowMinU..bRowMaxU] so the Back Image fits the silhouette & mold 100%!
        const relU = Math.max(0.0, Math.min(1.0, (uSample - fRowMinU) / fRowSpanU));
        const uBackAligned = bRowMinU + relU * bRowSpanU;
        const uBackFinal =
          uSample * (1.0 - silConformity * 0.92) + uBackAligned * (silConformity * 0.92);

        const relVSample = Math.max(0.0, Math.min(1.0, (vSample - frontMinV) / frontSpanV));
        const vBackSampleAligned = backMinV + relVSample * backSpanV;
        const vBackFinal =
          vSample * (1.0 - silConformity * 0.88) + vBackSampleAligned * (silConformity * 0.88);

        const rawBackTx = Math.min(w - 1, Math.max(0, Math.round(uBackFinal * (w - 1))));
        const srcTx = mirrorX ? w - 1 - rawBackTx : rawBackTx;
        const srcTy = Math.min(h - 1, Math.max(0, Math.round((1.0 - vBackFinal) * (h - 1))));
        const srcIdx = srcTy * w + srcTx;
        const src3 = srcIdx * 3;

        backRgbActive[dst3] = customBackRgb[src3];
        backRgbActive[dst3 + 1] = customBackRgb[src3 + 1];
        backRgbActive[dst3 + 2] = customBackRgb[src3 + 2];

        // Combine the Back Image's own depth with the 3D Mold's dorsal carrier so the mold shapes the back image
        const imgBackD = customBackDepth[srcIdx];
        backRawCnnDepth[dstIdx] = hasPresetShape
          ? Math.min(1.0, imgBackD * 0.78 + moldBackZ * 0.42 * moldStrength)
          : imgBackD;
      } else if (backMode === BackImageSourceMode.EngravedBackplate) {
        const r = Math.hypot(nx, ny);
        const angle = Math.atan2(ny, nx);
        const ringWave = 0.5 + 0.5 * Math.cos(r * Math.PI * 18.0);
        const radialRays = 0.5 + 0.5 * Math.cos(angle * 36.0 + r * 6.0);
        const outerRim = Math.exp(-Math.pow((r - 0.78) / 0.055, 2)) * 0.65;
        const innerSeal = Math.exp(-Math.pow((r - 0.42) / 0.035, 2)) * 0.48;
        const frontImprint = depthTensor[dstIdx] * 0.38;
        const moldImprint = hasPresetShape ? moldBackZ * 0.45 * moldStrength : 0.0;

        const plateRelief = Math.max(
          0,
          Math.min(
            1,
            0.22 +
              outerRim +
              innerSeal +
              ringWave * 0.1 * (r < 0.76 ? 1 : 0) +
              radialRays * 0.06 * (r > 0.44 && r < 0.76 ? 1 : 0) +
              frontImprint +
              moldImprint
          )
        );
        const metalShade = 0.52 + 0.48 * plateRelief;
        backRgbActive[dst3] = Math.min(1, (avgR * 0.68 + 0.22) * metalShade);
        backRgbActive[dst3 + 1] = Math.min(1, (avgG * 0.68 + 0.2) * metalShade);
        backRgbActive[dst3 + 2] = Math.min(1, (avgB * 0.68 + 0.16) * metalShade);
        backRawCnnDepth[dstIdx] = plateRelief;
      } else if (backMode === BackImageSourceMode.SmoothCurvedBack) {
        const rSq = nx * nx + ny * ny;
        const baseDome = detectLines
          ? Math.max(0, Math.min(0.42, (1.0 - Math.max(Math.abs(nx), Math.abs(ny))) * 0.65))
          : Math.max(0, 1.0 - rSq * 0.82) * 0.46;
        const dome = hasPresetShape
          ? Math.min(1.0, baseDome * 0.45 + moldBackZ * 0.85 * moldStrength)
          : baseDome;
        const brushed = 0.94 + 0.06 * Math.sin(ty * 0.85 + Math.cos(tx * 0.25));
        const shade = (0.68 + 0.32 * dome) * brushed;
        backRgbActive[dst3] = Math.min(1, (avgR * 0.75 + 0.15) * shade);
        backRgbActive[dst3 + 1] = Math.min(1, (avgG * 0.75 + 0.15) * shade);
        backRgbActive[dst3 + 2] = Math.min(1, (avgB * 0.75 + 0.15) * shade);
        backRawCnnDepth[dstIdx] = dome;
      } else {
        // SymmetricRelief: Aligned with the Front Shell silhouette, clean inpainted RGB, depth tensor & 3D Mold dorsal anatomy
        backRgbActive[dst3] = frontInpaintedRgb[dst3];
        backRgbActive[dst3 + 1] = frontInpaintedRgb[dst3 + 1];
        backRgbActive[dst3 + 2] = frontInpaintedRgb[dst3 + 2];
        backRawCnnDepth[dstIdx] = hasPresetShape
          ? Math.min(1.0, depthTensor[dstIdx] * 0.82 + moldBackZ * 0.38 * moldStrength)
          : depthTensor[dstIdx];
      }
    }
  }

  // Zero-GC Voronoï Inpainting on backRgbActive -> backInpaintedRgb so the Back Image joint also has zero background bleed
  for (let i = 0; i < TENSOR_LEN; i++) {
    const seedIdx = nearestFgTexelIdx[i] >= 0 ? nearestFgTexelIdx[i] : i;
    const i3 = i * 3;
    const s3 = seedIdx * 3;
    if (backMode === BackImageSourceMode.CustomBackUpload && buffers.hasCustomBackImage) {
      const br = backRgbActive[i3];
      const bg = backRgbActive[i3 + 1];
      const bb = backRgbActive[i3 + 2];
      const dbr = br - buffers.backAutoBgR;
      const dbg = bg - buffers.backAutoBgG;
      const dbb = bb - buffers.backAutoBgB;
      const isBackBg = dbr * dbr + dbg * dbg + dbb * dbb <= 0.032;
      if (isBackBg && seedIdx !== i) {
        backInpaintedRgb[i3] = backRgbActive[s3];
        backInpaintedRgb[i3 + 1] = backRgbActive[s3 + 1];
        backInpaintedRgb[i3 + 2] = backRgbActive[s3 + 2];
        backRgbActive[i3] = backInpaintedRgb[i3];
        backRgbActive[i3 + 1] = backInpaintedRgb[i3 + 1];
        backRgbActive[i3 + 2] = backInpaintedRgb[i3 + 2];
      } else {
        backInpaintedRgb[i3] = br;
        backInpaintedRgb[i3 + 1] = bg;
        backInpaintedRgb[i3 + 2] = bb;
      }
    } else {
      backInpaintedRgb[i3] = backRgbActive[s3];
      backInpaintedRgb[i3 + 1] = backRgbActive[s3 + 1];
      backInpaintedRgb[i3 + 2] = backRgbActive[s3 + 2];
    }
  }

  // 1A. Compute Multi-Scale Low-Frequency Pillow Volume (backLowFreqVolume) on the Back Image (Zero-GC)
  // so Anti-Inflation Relief Extraction works identically on the Back Shell without bloating!
  backLowFreqVolume.set(backRawCnnDepth);
  const radius = 5;
  const invSpan = 1.0 / (radius * 2 + 1);
  for (let pass = 0; pass < 3; pass++) {
    for (let y = 0; y < h; y++) {
      const rOff = y * w;
      for (let x = 0; x < w; x++) {
        let sum = 0.0;
        for (let k = -radius; k <= radius; k++) {
          const sx = x + k < 0 ? 0 : x + k >= w ? w - 1 : x + k;
          sum += backLowFreqVolume[rOff + sx];
        }
        blurTemp[rOff + x] = sum * invSpan;
      }
    }
    for (let y = 0; y < h; y++) {
      const rOff = y * w;
      for (let x = 0; x < w; x++) {
        let sum = 0.0;
        for (let k = -radius; k <= radius; k++) {
          const sy = y + k < 0 ? 0 : y + k >= h ? h - 1 : y + k;
          sum += blurTemp[sy * w + x];
        }
        backLowFreqVolume[rOff + x] = sum * invSpan;
      }
    }
  }

  // 1B. Apply the Exact Same CNN Sobel/Laplacian + Screened Poisson + Straight-Line Anti-Mound + 4-Zone Rules to the Back Shell
  for (let y = 0; y < h; y++) {
    const ym1 = Math.max(0, y - 1) * w;
    const y0 = y * w;
    const yp1 = Math.min(h - 1, y + 1) * w;

    for (let x = 0; x < w; x++) {
      const xm1 = Math.max(0, x - 1);
      const xp1 = Math.min(w - 1, x + 1);
      const idx = y0 + x;

      if (texelCutoutMask[idx] === 0) {
        backConvGradX[idx] = 0.0;
        backConvGradY[idx] = 0.0;
        backConvLaplacian[idx] = 0.0;
        continue;
      }

      const tl = backRawCnnDepth[ym1 + xm1];
      const tc = backRawCnnDepth[ym1 + x];
      const tr = backRawCnnDepth[ym1 + xp1];
      const ml = backRawCnnDepth[y0 + xm1];
      const mc = backRawCnnDepth[idx];
      const mr = backRawCnnDepth[y0 + xp1];
      const bl = backRawCnnDepth[yp1 + xm1];
      const bc = backRawCnnDepth[yp1 + x];
      const br = backRawCnnDepth[yp1 + xp1];

      const gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
      const gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
      const lap = tc + ml + mr + bc - 4.0 * mc;

      backConvGradX[idx] = gx * 0.25;
      backConvGradY[idx] = gy * 0.25;
      backConvLaplacian[idx] = lap;
    }
  }

  // Compute Straight-Line Structure Tensor on the Back Shell's own gradients
  computeStraightLineStructureTensorZeroGC(
    backConvGradX,
    backConvGradY,
    buffers.exteriorBgMask,
    backLineCoherence,
    backLineTangentX,
    backLineTangentY,
    lineSens
  );

  // 1C. Fast Screened Poisson Surface Integration on the Back Image (backPoissonSurface)
  const slopeGain = params.gradientSlopeGain;
  for (let y = 0; y < h; y++) {
    const ym1 = Math.max(0, y - 1) * w;
    const y0 = y * w;
    const yp1 = Math.min(h - 1, y + 1) * w;
    for (let x = 0; x < w; x++) {
      const xm1 = Math.max(0, x - 1);
      const xp1 = Math.min(w - 1, x + 1);
      const idx = y0 + x;
      if (texelCutoutMask[idx] === 0) {
        backDivergenceField[idx] = 0.0;
        backPoissonSurface[idx] = 0.0;
        continue;
      }
      const dpdx = 0.5 * (backConvGradX[y0 + xp1] - backConvGradX[y0 + xm1]) * 2.8 * slopeGain;
      const dqdy = 0.5 * (backConvGradY[yp1 + x] - backConvGradY[ym1 + x]) * 2.8 * slopeGain;
      let div = dpdx + dqdy;
      if (detectLines && facetPlanarity > 0.01) {
        const gMagSq = backConvGradX[idx] * backConvGradX[idx] + backConvGradY[idx] * backConvGradY[idx];
        const coh = backLineCoherence[idx];
        const flatMask = Math.max(0.0, 1.0 - Math.min(1.0, gMagSq * 18.0 + coh * 0.9));
        div *= 1.0 - flatMask * facetPlanarity * 0.88;
      }
      backDivergenceField[idx] = div;
      backPoissonSurface[idx] = backRawCnnDepth[idx];
    }
  }

  const backPoissonIters = Math.max(8, Math.min(28, Math.round(params.poissonIterations * 0.55)));
  const backLambda = (params.screenedLambda || 0.12) + (detectLines ? facetPlanarity * 0.22 : 0.0);
  const backInvDiag = 1.0 / (4.0 + backLambda);
  const backOmega = 1.62;

  for (let iter = 0; iter < backPoissonIters; iter++) {
    for (let y = 1; y < h - 1; y++) {
      const y0 = y * w;
      const ym1 = y0 - w;
      const yp1 = y0 + w;
      for (let x = 1; x < w - 1; x++) {
        const idx = y0 + x;
        if (texelCutoutMask[idx] === 0) {
          backPoissonSurface[idx] = 0.0;
          continue;
        }
        const nSum =
          backPoissonSurface[idx - 1] +
          backPoissonSurface[idx + 1] +
          backPoissonSurface[ym1 + x] +
          backPoissonSurface[yp1 + x];
        const targetZ =
          (nSum - backDivergenceField[idx] * 0.62 + backLambda * backRawCnnDepth[idx]) *
          backInvDiag;
        backPoissonSurface[idx] += backOmega * (targetZ - backPoissonSurface[idx]);
      }
    }
  }

  let bMinZ = Infinity;
  let bMaxZ = -Infinity;
  for (let i = 0; i < TENSOR_LEN; i++) {
    if (texelCutoutMask[i] === 0) continue;
    const z = backPoissonSurface[i];
    if (z < bMinZ) bMinZ = z;
    if (z > bMaxZ) bMaxZ = z;
  }
  if (!Number.isFinite(bMinZ) || !Number.isFinite(bMaxZ)) {
    bMinZ = 0.0;
    bMaxZ = 1.0;
  }
  const bInvRange = 1.0 / Math.max(1e-5, bMaxZ - bMinZ);
  const normWeight = params.normalIntegrationWeight ?? 0.78;

  // Refine backDepthTensor and backZoneMap using the same rules as the front (symmetric + depth-aware)
  for (let i = 0; i < TENSOR_LEN; i++) {
    if (texelCutoutMask[i] === 0) {
      backPoissonSurface[i] = 0.0;
      backDepthTensor[i] = 0.0;
      backZoneMap[i] = 2;
      continue;
    }

    const normBackZ = Math.max(0.0, Math.min(1.0, (backPoissonSurface[i] - bMinZ) * bInvRange));
    backPoissonSurface[i] = normBackZ;

    let bDepth = normBackZ * normWeight + backRawCnnDepth[i] * (1.0 - normWeight);
    const coh = backLineCoherence[i];
    const lapMag = Math.abs(backConvLaplacian[i]);

    if (useSymmetricRules && detectLines && facetPlanarity > 0.01) {
      // Apply the exact same anti-mound planar facet stabilization to the Back Shell's depth
      const rawNoMound = Math.max(
        0.0,
        backRawCnnDepth[i] - backLowFreqVolume[i] * 0.26 * facetPlanarity
      );
      const levels = 10.0;
      const terraced = Math.round(rawNoMound * levels) / levels;
      const planarTarget = rawNoMound * (1.0 - 0.38 * facetPlanarity) + terraced * (0.38 * facetPlanarity);
      const blendPlanar = facetPlanarity * (0.52 + 0.35 * coh);
      bDepth = bDepth * (1.0 - blendPlanar) + planarTarget * blendPlanar;
    }

    backDepthTensor[i] = Math.max(0.0, Math.min(1.0, bDepth));

    // Classify Back Shell 4-Zone Map from its own depths and ridges (or mirror Front zoneMap when SymmetricRelief)
    if (backMode === BackImageSourceMode.SymmetricRelief && useSymmetricRules) {
      backZoneMap[i] = buffers.zoneMap[i];
    } else if (lapMag > 0.16 || (detectLines && coh > 0.58)) {
      backZoneMap[i] = 3;
    } else if (bDepth >= 0.58) {
      backZoneMap[i] = 0;
    } else if (bDepth >= 0.26) {
      backZoneMap[i] = 1;
    } else {
      backZoneMap[i] = 2;
    }
  }

  // Compute Back Normals from backDepthTensor + 3D Preset Base Shape dorsal normals
  for (let ty = 0; ty < h; ty++) {
    const v = 1.0 - ty / (h - 1);
    const ym1 = Math.max(0, ty - 1) * w;
    const y0 = ty * w;
    const yp1 = Math.min(h - 1, ty + 1) * w;
    const fRowCenterU = 0.5 * (rowSilhouetteMinU[ty] + rowSilhouetteMaxU[ty]) - 0.5;
    const rowAlignCenterX =
      effMoldCenterX * (1.0 - silConformity * 0.45) +
      (fRowCenterU + shapeOffX) * (silConformity * 0.45);

    for (let tx = 0; tx < w; tx++) {
      const u = tx / (w - 1);
      const xm1 = Math.max(0, tx - 1);
      const xp1 = Math.min(w - 1, tx + 1);
      const idx = y0 + tx;
      const dzdx = (backDepthTensor[y0 + xp1] - backDepthTensor[y0 + xm1]) * 2.4 * slopeGain;
      const dzdy = (backDepthTensor[yp1 + tx] - backDepthTensor[ym1 + tx]) * 2.4 * slopeGain;
      let nx = -dzdx;
      let ny = dzdy;
      let nz = 1.0;

      if (hasPresetShape) {
        const uMold = 0.5 + (u - 0.5 - rowAlignCenterX);
        const vMold = 0.5 + (v - 0.5 - effMoldCenterY);
        evaluatePresetBaseShapeZeroGC(
          presetShape,
          uMold,
          vMold,
          shapeScaleX,
          shapeScaleY,
          poseArmAngleDeg,
          limbUnionMode,
          limbUnionCurvature,
          effArmLen,
          effArmThick,
          effLegLen,
          effLegThick
        );
        if (_shapeInside === 1) {
          nx += _shapeNx * 0.55 * moldStrength;
          ny += _shapeNy * 0.55 * moldStrength;
        }
      }

      if (detectLines) {
        const coh = backLineCoherence[idx];
        if (coh > 0.15) {
          const ltx = backLineTangentX[idx];
          const lty = backLineTangentY[idx];
          const dotTan = nx * ltx + ny * lty;
          const damp = coh * lineSens * 0.85;
          nx -= dotTan * ltx * damp;
          ny -= dotTan * lty * damp;
        }
      }

      const invL = 1.0 / Math.sqrt(nx * nx + ny * ny + nz * nz);
      const i3 = idx * 3;
      backNormalTensor[i3] = nx * invL;
      backNormalTensor[i3 + 1] = ny * invL;
      backNormalTensor[i3 + 2] = nz * invL;
    }
  }

  // 2. Populate Lateral Seam Band ("El Lateral") RGB, Depth & Normals
  const sideMode = params.sideSourceMode || SideImageSourceMode.AutoEdgeExtension;

  for (let ty = 0; ty < h; ty++) {
    // vAcross in [0, 1] goes from Front Joint (0) -> Center of Lateral Band (0.5) -> Back Joint (1)
    const vAcross = ty / (h - 1);
    const distFromCenter = Math.abs(vAcross - 0.5) * 2.0; // 0 at center seam, 1 at front/back borders
    const rowOffset = ty * w;

    for (let tx = 0; tx < w; tx++) {
      // uPerim in [0, 1] wraps around the 360° perimeter of the model
      const uPerim = tx / (w - 1);
      const angle = uPerim * Math.PI * 2.0;
      const idx = rowOffset + tx;
      const i3 = idx * 3;

      // Ray-march inward from the outer perimeter along angle to find the true foreground contour pixel for this perimeter angle
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      let foundIdx = Math.round(h * 0.5) * w + Math.round(w * 0.5);
      for (let rStep = 0.47; rStep >= 0.08; rStep -= 0.025) {
        const rx = Math.min(w - 1, Math.max(0, Math.round((0.5 + cosA * rStep) * (w - 1))));
        const ry = Math.min(h - 1, Math.max(0, Math.round((0.5 + sinA * rStep) * (h - 1))));
        const testIdx = ry * w + rx;
        if (texelCutoutMask[testIdx] === 1) {
          const innerR = Math.max(0.05, rStep - 0.03);
          const ix = Math.min(w - 1, Math.max(0, Math.round((0.5 + cosA * innerR) * (w - 1))));
          const iy = Math.min(h - 1, Math.max(0, Math.round((0.5 + sinA * innerR) * (h - 1))));
          foundIdx = iy * w + ix;
          break;
        }
      }
      const sIdx = nearestFgTexelIdx[foundIdx] >= 0 ? nearestFgTexelIdx[foundIdx] : foundIdx;
      const s3 = sIdx * 3;
      const frontR = frontInpaintedRgb[s3];
      const frontG = frontInpaintedRgb[s3 + 1];
      const frontB = frontInpaintedRgb[s3 + 2];
      const backR = backInpaintedRgb[s3];
      const backG = backInpaintedRgb[s3 + 1];
      const backB = backInpaintedRgb[s3 + 2];
      const localImgR = frontR * (1.0 - vAcross) + backR * vAcross;
      const localImgG = frontG * (1.0 - vAcross) + backG * vAcross;
      const localImgB = frontB * (1.0 - vAcross) + backB * vAcross;

      if (sideMode === SideImageSourceMode.CustomSideUpload && buffers.hasCustomSideImage) {
        sideRgbActive[i3] = customSideRgb[i3];
        sideRgbActive[i3 + 1] = customSideRgb[i3 + 1];
        sideRgbActive[i3 + 2] = customSideRgb[i3 + 2];
        sideDepthTensor[idx] = customSideDepth[idx];
      } else if (sideMode === SideImageSourceMode.MilledCoinEdge) {
        // Reeded Coin Edge tinted directly with the local Image Color at that contour position
        const reeds = 0.5 + 0.5 * Math.cos(angle * 112.0);
        const borderBand = distFromCenter > 0.78 ? 0.92 : reeds;
        const heightVal = borderBand;
        const shade = 0.78 + 0.22 * heightVal;

        sideRgbActive[i3] = Math.min(1, localImgR * shade);
        sideRgbActive[i3 + 1] = Math.min(1, localImgG * shade);
        sideRgbActive[i3 + 2] = Math.min(1, localImgB * shade);
        sideDepthTensor[idx] = heightVal;
      } else if (sideMode === SideImageSourceMode.FiligreeSeamBand) {
        // Engraved Filigree Seam Band tinted directly with the local Image Color
        const waveA = Math.sin(angle * 24.0 + vAcross * Math.PI * 3.0);
        const waveB = Math.cos(angle * 48.0 - vAcross * Math.PI * 2.0);
        const centerCord = Math.exp(-Math.pow((vAcross - 0.5) / 0.11, 2)) * (0.65 + 0.35 * Math.cos(angle * 64.0));
        const rails = Math.exp(-Math.pow((distFromCenter - 0.72) / 0.09, 2)) * 0.75;
        const filigree = Math.max(0, Math.min(1, 0.32 + 0.22 * waveA * waveB + centerCord + rails));
        const shade = 0.76 + 0.24 * filigree;

        sideRgbActive[i3] = Math.min(1, localImgR * shade);
        sideRgbActive[i3 + 1] = Math.min(1, localImgG * shade);
        sideRgbActive[i3 + 2] = Math.min(1, localImgB * shade);
        sideDepthTensor[idx] = filigree;
      } else {
        // AutoEdgeExtension: Pure seamless local Image Color from Front to Back (zero dark groove!)
        sideRgbActive[i3] = localImgR;
        sideRgbActive[i3 + 1] = localImgG;
        sideRgbActive[i3 + 2] = localImgB;

        const frontD = depthTensor[sIdx];
        const backD = backDepthTensor[sIdx];
        const crownProfile = Math.sin(vAcross * Math.PI);
        sideDepthTensor[idx] = Math.max(
          0,
          Math.min(1, (0.35 + 0.45 * (frontD * (1.0 - vAcross) + backD * vAcross)) * (0.68 + 0.32 * crownProfile))
        );
      }
    }
  }

  for (let ty = 0; ty < h; ty++) {
    const ym1 = Math.max(0, ty - 1) * w;
    const y0 = ty * w;
    const yp1 = Math.min(h - 1, ty + 1) * w;
    for (let tx = 0; tx < w; tx++) {
      const xm1 = (tx - 1 + w) % w;
      const xp1 = (tx + 1) % w;
      const dzdx = (sideDepthTensor[y0 + xp1] - sideDepthTensor[y0 + xm1]) * 2.6;
      const dzdy = (sideDepthTensor[yp1 + tx] - sideDepthTensor[ym1 + tx]) * 2.6;
      const nx = -dzdx;
      const ny = dzdy;
      const nz = 1.0;
      const invL = 1.0 / Math.sqrt(nx * nx + ny * ny + nz * nz);
      const i3 = (y0 + tx) * 3;
      sideNormalTensor[i3] = nx * invL;
      sideNormalTensor[i3 + 1] = ny * invL;
      sideNormalTensor[i3 + 2] = nz * invL;
    }
  }
}

// Module-scoped HSL registers for Zero-GC color/hue tone matching
let _hslH = 0;
let _hslS = 0;
let _hslL = 0;

function computeHSLInPlace(r: number, g: number, b: number): void {
  const max = r > g ? (r > b ? r : b) : g > b ? g : b;
  const min = r < g ? (r < b ? r : b) : g < b ? g : b;
  const l = (max + min) * 0.5;
  const d = max - min;

  if (d < 1e-5) {
    _hslH = 0;
    _hslS = 0;
    _hslL = l;
    return;
  }

  const s = l > 0.5 ? d / (2.0 - max - min) : d / (max + min);
  let h = 0;
  if (max === r) {
    h = (g - b) / d + (g < b ? 6.0 : 0.0);
  } else if (max === g) {
    h = (b - r) / d + 2.0;
  } else {
    h = (r - g) / d + 4.0;
  }
  _hslH = h / 6.0;
  _hslS = s;
  _hslL = l;
}

function parseHexToRGBNormalized(hex: string): [number, number, number] {
  const clean = hex.replace('#', '');
  if (clean.length !== 6) return [0.05, 0.07, 0.12];
  const num = parseInt(clean, 16);
  if (Number.isNaN(num)) return [0.05, 0.07, 0.12];
  return [
    ((num >> 16) & 255) / 255,
    ((num >> 8) & 255) / 255,
    (num & 255) / 255,
  ];
}

function evaluateBackgroundCutoutMask(
  buffers: CNNReconstructionBuffers,
  params: ReconstructionParams
): void {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const {
    rgbInput,
    alphaInput,
    depthTensor,
    slopeP,
    slopeQ,
    exteriorBgMask,
    candidateBgMask,
    floodVisited,
    floodQueue,
    morphTempMask,
    texelCutoutMask,
  } = buffers;

  const mode = params.cutoutMode || BackgroundCutoutMode.None;
  const borderMargin = Math.max(0, Math.min(0.48, params.cutoutBorderMargin || 0));
  const flatThreshold = Math.max(0, Math.min(0.65, params.cutoutFlatThreshold || 0));
  const colorTol = Math.max(0.01, Math.min(0.85, params.cutoutColorTolerance || 0.22));
  const colorTolSq = colorTol * colorTol * 3.0; // Normalized across 3 RGB channels
  const hueTol = Math.max(0.01, Math.min(0.5, params.cutoutHueTolerance || 0.14));

  let targetR = buffers.autoDetectedBgR;
  let targetG = buffers.autoDetectedBgG;
  let targetB = buffers.autoDetectedBgB;

  if (mode === BackgroundCutoutMode.CustomColorAndHue) {
    const parsed = parseHexToRGBNormalized(params.cutoutTargetColorHex || '#0c1220');
    targetR = parsed[0];
    targetG = parsed[1];
    targetB = parsed[2];
  }

  computeHSLInPlace(targetR, targetG, targetB);
  const targetH = _hslH;
  const targetS = _hslS;
  const targetL = _hslL;

  const maxRadiusNorm = 1.0 - borderMargin;
  const maxRadiusSq = maxRadiusNorm * maxRadiusNorm;

  // Step 1: Classify candidate foreground (1) vs candidate background (0) in candidateBgMask (1 = background candidate)
  for (let ty = 0; ty < h; ty++) {
    const ny = (ty / (h - 1)) * 2.0 - 1.0; // -1 to +1
    const rowOffset = ty * w;

    for (let tx = 0; tx < w; tx++) {
      const nx = (tx / (w - 1)) * 2.0 - 1.0; // -1 to +1
      const i = rowOffset + tx;

      // 1. Always discard transparent alpha pixels from uploaded PNGs
      if (alphaInput[i] < 0.12) {
        candidateBgMask[i] = 1;
        texelCutoutMask[i] = 0;
        continue;
      }

      // 2. Outer Frame / Radial Margin Cutout (removes outer flat frame corners/border)
      const rSq = nx * nx + ny * ny;
      if (borderMargin > 0.002 && rSq > maxRadiusSq) {
        candidateBgMask[i] = 1;
        texelCutoutMask[i] = 0;
        continue;
      }

      if (mode === BackgroundCutoutMode.None) {
        candidateBgMask[i] = 0;
        texelCutoutMask[i] = 1;
        continue;
      }

      const d = depthTensor[i];
      const p = slopeP[i];
      const q = slopeQ[i];
      const slopeMagSq = p * p + q * q;

      const i3 = i * 3;
      const pr = rgbInput[i3];
      const pg = rgbInput[i3 + 1];
      const pb = rgbInput[i3 + 2];

      if (mode === BackgroundCutoutMode.FlatReliefAndBorder) {
        // Check both flat relief depth AND similarity to auto-detected corner background color (handles white, light, or dark backgrounds!)
        const drBg = pr - buffers.autoDetectedBgR;
        const dgBg = pg - buffers.autoDetectedBgG;
        const dbBg = pb - buffers.autoDetectedBgB;
        const distSqAutoBg = drBg * drBg + dgBg * dgBg + dbBg * dbBg;
        const autoBgTolSq = Math.max(0.025, flatThreshold * 1.45) * 0.45;

        const isOuterFlatZone =
          exteriorBgMask[i] === 1 ||
          (distSqAutoBg <= autoBgTolSq && slopeMagSq < 0.24) ||
          (d <= flatThreshold && (slopeMagSq < 0.14 || rSq > 0.68));
        const isOutsideDefaultMedallion =
          rSq > 0.92 && (d <= flatThreshold + 0.08 || distSqAutoBg <= autoBgTolSq * 1.5);

        let keep = !(isOuterFlatZone || isOutsideDefaultMedallion);
        if (params.invertCutout) keep = !keep;
        candidateBgMask[i] = keep ? 0 : 1;
        texelCutoutMask[i] = keep ? 1 : 0;
        continue;
      }

      // Color / Hue Tone Matching (for AutoCornerColor and CustomColorAndHue)
      let matchesColorOrTone = false;

      if (params.matchByHueTone) {
        computeHSLInPlace(pr, pg, pb);
        const pixH = _hslH;
        const pixS = _hslS;
        const pixL = _hslL;

        if (targetS < 0.14) {
          // Target is dark/gray/white achromatic tone: match low saturation & similar luminance
          matchesColorOrTone =
            pixS < 0.24 + hueTol * 0.55 &&
            Math.abs(pixL - targetL) <= colorTol * 0.9;
        } else {
          // Chromatic tone: match angular Hue distance + reasonable saturation
          const rawDiff = Math.abs(pixH - targetH);
          const angularHueDiff = rawDiff > 0.5 ? 1.0 - rawDiff : rawDiff;
          matchesColorOrTone =
            angularHueDiff <= hueTol &&
            pixS >= Math.max(0.05, targetS - colorTol * 0.75) &&
            Math.abs(pixL - targetL) <= Math.max(0.28, colorTol);
        }
      } else {
        // Euclidean squared RGB distance (Zero Math.sqrt)
        const dr = pr - targetR;
        const dg = pg - targetG;
        const db = pb - targetB;
        const distSq = dr * dr + dg * dg + db * db;
        matchesColorOrTone = distSq <= colorTolSq;
      }

      const isFlatBg = flatThreshold > 0.01 && d <= flatThreshold && slopeMagSq < 0.1;

      let keep = !(matchesColorOrTone || isFlatBg);
      if (params.invertCutout) {
        keep = matchesColorOrTone && !isFlatBg;
      }
      candidateBgMask[i] = keep ? 0 : 1;
      texelCutoutMask[i] = keep ? 1 : 0;
    }
  }

  // Step 2: Close Internal Holes & Bridge Separated Zones (Zero-GC)
  const closeInternal = params.closeInternalHoles !== false;
  const fillRadius = Math.max(0, Math.min(12, Math.round(params.holeFillRadius ?? 4)));

  if (mode !== BackgroundCutoutMode.None && closeInternal) {
    // 2A. Flood-fill background ONLY from the outer border through candidateBgMask === 1
    // This prevents dark shadows or armor lines inside the character from ever turning into gaping holes!
    floodVisited.fill(0);
    let qHead = 0;
    let qTail = 0;

    for (let i = 0; i < TENSOR_LEN; i++) {
      const tx = i % w;
      const ty = (i / w) | 0;
      const nx = (tx / (w - 1)) * 2.0 - 1.0;
      const ny = (ty / (h - 1)) * 2.0 - 1.0;
      const onOuterBorder =
        tx === 0 ||
        tx === w - 1 ||
        ty === 0 ||
        ty === h - 1 ||
        (borderMargin > 0.002 && nx * nx + ny * ny >= maxRadiusSq * 0.96);

      if (onOuterBorder && candidateBgMask[i] === 1) {
        floodVisited[i] = 1;
        floodQueue[qTail++] = i;
      }
    }

    while (qHead < qTail) {
      const curr = floodQueue[qHead++];
      const cx = curr % w;
      const cy = (curr / w) | 0;

      if (cx > 0) {
        const n = curr - 1;
        if (floodVisited[n] === 0 && candidateBgMask[n] === 1) {
          floodVisited[n] = 1;
          floodQueue[qTail++] = n;
        }
      }
      if (cx < w - 1) {
        const n = curr + 1;
        if (floodVisited[n] === 0 && candidateBgMask[n] === 1) {
          floodVisited[n] = 1;
          floodQueue[qTail++] = n;
        }
      }
      if (cy > 0) {
        const n = curr - w;
        if (floodVisited[n] === 0 && candidateBgMask[n] === 1) {
          floodVisited[n] = 1;
          floodQueue[qTail++] = n;
        }
      }
      if (cy < h - 1) {
        const n = curr + w;
        if (floodVisited[n] === 0 && candidateBgMask[n] === 1) {
          floodVisited[n] = 1;
          floodQueue[qTail++] = n;
        }
      }
    }

    for (let i = 0; i < TENSOR_LEN; i++) {
      texelCutoutMask[i] = floodVisited[i] === 1 ? 0 : 1;
    }
  }

  // 2B. Morphological Closing (Dilation + Interior Fill + Erosion) to fuse separated zones & eliminate gaps
  if (mode !== BackgroundCutoutMode.None && fillRadius > 0) {
    // Horizontal Dilation (texelCutoutMask -> morphTempMask)
    for (let ty = 0; ty < h; ty++) {
      const rowOff = ty * w;
      for (let tx = 0; tx < w; tx++) {
        const x0 = Math.max(0, tx - fillRadius);
        const x1 = Math.min(w - 1, tx + fillRadius);
        let anyFg = 0;
        for (let sx = x0; sx <= x1; sx++) {
          if (texelCutoutMask[rowOff + sx] === 1) {
            anyFg = 1;
            break;
          }
        }
        morphTempMask[rowOff + tx] = anyFg;
      }
    }
    // Vertical Dilation (morphTempMask -> texelCutoutMask)
    for (let ty = 0; ty < h; ty++) {
      const y0 = Math.max(0, ty - fillRadius);
      const y1 = Math.min(h - 1, ty + fillRadius);
      const rowOff = ty * w;
      for (let tx = 0; tx < w; tx++) {
        let anyFg = 0;
        for (let sy = y0; sy <= y1; sy++) {
          if (morphTempMask[sy * w + tx] === 1) {
            anyFg = 1;
            break;
          }
        }
        texelCutoutMask[rowOff + tx] = anyFg;
      }
    }

    // Horizontal Erosion (texelCutoutMask -> morphTempMask)
    for (let ty = 0; ty < h; ty++) {
      const rowOff = ty * w;
      for (let tx = 0; tx < w; tx++) {
        if (tx === 0 || tx === w - 1 || ty === 0 || ty === h - 1) {
          morphTempMask[rowOff + tx] = texelCutoutMask[rowOff + tx];
          continue;
        }
        const x0 = Math.max(0, tx - fillRadius);
        const x1 = Math.min(w - 1, tx + fillRadius);
        let allFg = 1;
        for (let sx = x0; sx <= x1; sx++) {
          if (texelCutoutMask[rowOff + sx] === 0) {
            allFg = 0;
            break;
          }
        }
        morphTempMask[rowOff + tx] = allFg;
      }
    }
    // Vertical Erosion (morphTempMask -> texelCutoutMask)
    for (let ty = 0; ty < h; ty++) {
      const y0 = Math.max(0, ty - fillRadius);
      const y1 = Math.min(h - 1, ty + fillRadius);
      const rowOff = ty * w;
      for (let tx = 0; tx < w; tx++) {
        const nx = (tx / (w - 1)) * 2.0 - 1.0;
        const ny = (ty / (h - 1)) * 2.0 - 1.0;
        if (borderMargin > 0.002 && nx * nx + ny * ny > maxRadiusSq) {
          texelCutoutMask[rowOff + tx] = 0;
          continue;
        }
        let allFg = 1;
        for (let sy = y0; sy <= y1; sy++) {
          if (morphTempMask[sy * w + tx] === 0) {
            allFg = 0;
            break;
          }
        }
        texelCutoutMask[rowOff + tx] = allFg;
      }
    }

    // Keep only the main connected foreground subject (removes stray disconnected background scraps in corners)
    if (closeInternal) {
      floodVisited.fill(0);
      let qHead = 0;
      let qTail = 0;
      const cMinX = Math.floor(w * 0.25);
      const cMaxX = Math.floor(w * 0.75);
      const cMinY = Math.floor(h * 0.22);
      const cMaxY = Math.floor(h * 0.78);

      for (let ty = cMinY; ty <= cMaxY; ty++) {
        const rOff = ty * w;
        for (let tx = cMinX; tx <= cMaxX; tx++) {
          const idx = rOff + tx;
          if (texelCutoutMask[idx] === 1) {
            floodVisited[idx] = 1;
            floodQueue[qTail++] = idx;
          }
        }
      }

      if (qTail > 0) {
        while (qHead < qTail) {
          const curr = floodQueue[qHead++];
          const cx = curr % w;
          const cy = (curr / w) | 0;

          if (cx > 0) {
            const n = curr - 1;
            if (floodVisited[n] === 0 && texelCutoutMask[n] === 1) {
              floodVisited[n] = 1;
              floodQueue[qTail++] = n;
            }
          }
          if (cx < w - 1) {
            const n = curr + 1;
            if (floodVisited[n] === 0 && texelCutoutMask[n] === 1) {
              floodVisited[n] = 1;
              floodQueue[qTail++] = n;
            }
          }
          if (cy > 0) {
            const n = curr - w;
            if (floodVisited[n] === 0 && texelCutoutMask[n] === 1) {
              floodVisited[n] = 1;
              floodQueue[qTail++] = n;
            }
          }
          if (cy < h - 1) {
            const n = curr + w;
            if (floodVisited[n] === 0 && texelCutoutMask[n] === 1) {
              floodVisited[n] = 1;
              floodQueue[qTail++] = n;
            }
          }
        }

        for (let i = 0; i < TENSOR_LEN; i++) {
          texelCutoutMask[i] = floodVisited[i];
        }
      }
    }
  }

  // Step 2.5: Capture Drawing's True Silhouette Bounding Box, Width, Height, Proportions & Row Spans (Zero-GC)
  const {
    drawingSilhouetteMask,
    rowSilhouetteMinU,
    rowSilhouetteMaxU,
    rowSilhouetteTempMinU,
    rowSilhouetteTempMaxU,
  } = buffers;

  let fgCount = 0;
  let minX = w - 1;
  let maxX = 0;
  let minY = h - 1;
  let maxY = 0;

  for (let ty = 0; ty < h; ty++) {
    const rOff = ty * w;
    let rMin = w;
    let rMax = -1;
    for (let tx = 0; tx < w; tx++) {
      const idx = rOff + tx;
      // When mode !== None, use texelCutoutMask; when mode === None, use exteriorBgMask === 0 so auto-fit still knows the drawing bounds!
      const isFg =
        mode !== BackgroundCutoutMode.None
          ? texelCutoutMask[idx]
          : exteriorBgMask[idx] === 0
          ? 1
          : 0;
      drawingSilhouetteMask[idx] = isFg;
      if (isFg === 1) {
        fgCount++;
        if (tx < minX) minX = tx;
        if (tx > maxX) maxX = tx;
        if (ty < minY) minY = ty;
        if (ty > maxY) maxY = ty;
        if (tx < rMin) rMin = tx;
        if (tx > rMax) rMax = tx;
      }
    }
    if (rMax >= rMin) {
      rowSilhouetteTempMinU[ty] = rMin / (w - 1);
      rowSilhouetteTempMaxU[ty] = rMax / (w - 1);
    } else {
      rowSilhouetteTempMinU[ty] = -1.0;
      rowSilhouetteTempMaxU[ty] = -1.0;
    }
  }

  if (fgCount > 28 && fgCount < TENSOR_LEN * 0.985 && maxX > minX && maxY > minY) {
    buffers.autoSilhouetteMinU = minX / (w - 1);
    buffers.autoSilhouetteMaxU = maxX / (w - 1);
    buffers.autoSilhouetteMinV = 1.0 - maxY / (h - 1);
    buffers.autoSilhouetteMaxV = 1.0 - minY / (h - 1);
  } else {
    buffers.autoSilhouetteMinU = 0.08;
    buffers.autoSilhouetteMaxU = 0.92;
    buffers.autoSilhouetteMinV = 0.06;
    buffers.autoSilhouetteMaxV = 0.94;
  }

  buffers.autoSilhouetteWidth = Math.max(
    0.16,
    buffers.autoSilhouetteMaxU - buffers.autoSilhouetteMinU
  );
  buffers.autoSilhouetteHeight = Math.max(
    0.16,
    buffers.autoSilhouetteMaxV - buffers.autoSilhouetteMinV
  );
  buffers.autoSilhouetteAspect =
    buffers.autoSilhouetteWidth / Math.max(0.16, buffers.autoSilhouetteHeight);
  buffers.autoSilhouetteCenterX =
    (buffers.autoSilhouetteMinU + buffers.autoSilhouetteMaxU) * 0.5 - 0.5;
  buffers.autoSilhouetteCenterY =
    (buffers.autoSilhouetteMinV + buffers.autoSilhouetteMaxV) * 0.5 - 0.5;

  // Fill empty rows with global silhouette span and apply a 5-tap vertical box filter for smooth per-row conformity
  for (let ty = 0; ty < h; ty++) {
    if (rowSilhouetteTempMaxU[ty] < 0.0) {
      rowSilhouetteTempMinU[ty] = buffers.autoSilhouetteMinU;
      rowSilhouetteTempMaxU[ty] = buffers.autoSilhouetteMaxU;
    }
  }
  for (let ty = 0; ty < h; ty++) {
    let sumMin = 0.0;
    let sumMax = 0.0;
    let cnt = 0;
    for (let dy = -2; dy <= 2; dy++) {
      const sy = Math.min(h - 1, Math.max(0, ty + dy));
      sumMin += rowSilhouetteTempMinU[sy];
      sumMax += rowSilhouetteTempMaxU[sy];
      cnt++;
    }
    rowSilhouetteMinU[ty] = sumMin / cnt;
    rowSilhouetteMaxU[ty] = sumMax / cnt;
  }

  // Step 2.6: Zero-GC Automatic Arm & Leg Proportion Detection from Silhouette (Longitud y Grosor de Brazos y Piernas)
  if (fgCount > 28 && maxX > minX && maxY > minY) {
    const silSpanX = Math.max(16, maxX - minX);
    const silSpanY = Math.max(16, maxY - minY);
    const midX = (minX + maxX) >> 1;

    // 1. Measure core torso width around the waist/abdomen band [0.38..0.52 of height]
    const waistY0 = Math.min(h - 1, Math.max(0, Math.round(minY + silSpanY * 0.38)));
    const waistY1 = Math.min(h - 1, Math.max(waistY0, Math.round(minY + silSpanY * 0.52)));
    let torsoCoreSumPx = 0;
    let torsoRowsCnt = 0;
    for (let ty = waistY0; ty <= waistY1; ty++) {
      const rOff = ty * w;
      // Measure contiguous central torso chord around midX so separated arms don't inflate torso width
      let leftTorso = midX;
      while (leftTorso > minX && drawingSilhouetteMask[rOff + leftTorso] === 1) {
        leftTorso--;
      }
      let rightTorso = midX;
      while (rightTorso < maxX && drawingSilhouetteMask[rOff + rightTorso] === 1) {
        rightTorso++;
      }
      const chord = rightTorso - leftTorso;
      if (chord > 4) {
        torsoCoreSumPx += chord;
        torsoRowsCnt++;
      } else {
        const spanU = rowSilhouetteMaxU[ty] - rowSilhouetteMinU[ty];
        torsoCoreSumPx += spanU * (w - 1) * 0.52;
        torsoRowsCnt++;
      }
    }
    const avgTorsoCorePx =
      torsoRowsCnt > 0
        ? Math.max(12, Math.min(silSpanX * 0.64, torsoCoreSumPx / torsoRowsCnt))
        : silSpanX * 0.42;
    const torsoHalfPx = Math.max(6, Math.round(avgTorsoCorePx * 0.5));

    // 2. Measure Upper-Body Arm Reach, Arm Thickness & Pose Angle in [0.18..0.58 of height]
    const armY0 = Math.min(h - 1, Math.max(0, Math.round(minY + silSpanY * 0.18)));
    const armY1 = Math.min(h - 1, Math.max(armY0, Math.round(minY + silSpanY * 0.58)));
    let maxArmSpanPx = avgTorsoCorePx;
    let armPeakTy = Math.round(minY + silSpanY * 0.36);
    let lateralArmPixels = 0;
    let activeArmCols = 0;

    for (let tx = minX; tx <= maxX; tx++) {
      if (Math.abs(tx - midX) <= torsoHalfPx) continue;
      let colFg = 0;
      for (let ty = armY0; ty <= armY1; ty++) {
        if (drawingSilhouetteMask[ty * w + tx] === 1) {
          colFg++;
        }
      }
      if (colFg > 0) {
        lateralArmPixels += colFg;
        activeArmCols++;
      }
    }

    for (let ty = armY0; ty <= armY1; ty++) {
      const rowSpanPx = (rowSilhouetteMaxU[ty] - rowSilhouetteMinU[ty]) * (w - 1);
      if (rowSpanPx > maxArmSpanPx) {
        maxArmSpanPx = rowSpanPx;
        armPeakTy = ty;
      }
    }

    const singleArmReachPx = Math.max(6, (maxArmSpanPx - avgTorsoCorePx * 0.72) * 0.5);
    const canonicalArmReachPx = Math.max(8, silSpanY * 0.31);
    const rawArmLenRatio = singleArmReachPx / canonicalArmReachPx;
    buffers.autoDetectedArmLengthScale = Math.max(
      0.68,
      Math.min(1.42, 0.35 + 0.65 * rawArmLenRatio)
    );

    const avgArmChordPx =
      activeArmCols > 4 ? lateralArmPixels / activeArmCols : silSpanY * 0.115;
    const canonicalArmThickPx = Math.max(5, silSpanY * 0.118);
    const rawArmThickRatio = avgArmChordPx / canonicalArmThickPx;
    buffers.autoDetectedArmThicknessScale = Math.max(
      0.68,
      Math.min(1.42, 0.38 + 0.62 * rawArmThickRatio)
    );

    const shoulderTy = minY + silSpanY * 0.21;
    const dropY = Math.max(2, armPeakTy - shoulderTy);
    const estAngle = (Math.atan2(singleArmReachPx, dropY) * 180.0) / Math.PI;
    buffers.autoDetectedPoseArmAngleDeg = Math.round(
      Math.max(15, Math.min(90, estAngle))
    );

    // 3. Measure Lower-Body Leg Length & Leg Thickness in [0.46..1.00 of height]
    const crouchSearchY0 = Math.min(h - 1, Math.max(0, Math.round(minY + silSpanY * 0.45)));
    const crouchSearchY1 = Math.min(h - 1, Math.max(crouchSearchY0, Math.round(minY + silSpanY * 0.68)));
    let detectedCrotchTy = Math.round(minY + silSpanY * 0.52);
    for (let ty = crouchSearchY0; ty <= crouchSearchY1; ty++) {
      const rOff = ty * w;
      // Check if a central gap opens between left and right leg columns
      const centerGap =
        drawingSilhouetteMask[rOff + midX] === 0 ||
        drawingSilhouetteMask[rOff + Math.max(0, midX - 2)] === 0 ||
        drawingSilhouetteMask[rOff + Math.min(w - 1, midX + 2)] === 0;
      if (centerGap) {
        detectedCrotchTy = ty;
        break;
      }
    }

    const legSpanPx = Math.max(10, maxY - detectedCrotchTy);
    const canonicalLegSpanPx = Math.max(10, silSpanY * 0.48);
    const rawLegLenRatio = legSpanPx / canonicalLegSpanPx;
    buffers.autoDetectedLegLengthScale = Math.max(
      0.68,
      Math.min(1.42, 0.35 + 0.65 * rawLegLenRatio)
    );

    const legSampleY0 = Math.min(h - 1, Math.max(0, Math.round(minY + silSpanY * 0.58)));
    const legSampleY1 = Math.min(h - 1, Math.max(legSampleY0, Math.round(minY + silSpanY * 0.86)));
    let legFgWidthSum = 0;
    let legRowCnt = 0;
    for (let ty = legSampleY0; ty <= legSampleY1; ty++) {
      const rOff = ty * w;
      let leftLegFg = 0;
      let rightLegFg = 0;
      for (let tx = minX; tx < midX; tx++) {
        if (drawingSilhouetteMask[rOff + tx] === 1) leftLegFg++;
      }
      for (let tx = midX; tx <= maxX; tx++) {
        if (drawingSilhouetteMask[rOff + tx] === 1) rightLegFg++;
      }
      if (leftLegFg + rightLegFg > 4) {
        legFgWidthSum += (leftLegFg + rightLegFg) * 0.5;
        legRowCnt++;
      }
    }
    const avgSingleLegPx =
      legRowCnt > 0 ? legFgWidthSum / legRowCnt : silSpanX * 0.145;
    const canonicalLegThickPx = Math.max(5, silSpanX * 0.148);
    const rawLegThickRatio = avgSingleLegPx / canonicalLegThickPx;
    buffers.autoDetectedLegThicknessScale = Math.max(
      0.68,
      Math.min(1.42, 0.38 + 0.62 * rawLegThickRatio)
    );
  } else {
    buffers.autoDetectedArmLengthScale = 1.0;
    buffers.autoDetectedArmThicknessScale = 1.0;
    buffers.autoDetectedLegLengthScale = 1.0;
    buffers.autoDetectedLegThicknessScale = 1.0;
    buffers.autoDetectedPoseArmAngleDeg = 35;
  }

  // Step 3: Apply Preset 3D Base Shape Silhouette Conformity ("Amoldar Dibujo a la Silueta y Forma 3D")
  const presetShape = params.presetBaseShape || PresetBaseShape.None;
  if (presetShape !== PresetBaseShape.None) {
    const silMode =
      params.shapeSilhouetteMode || ShapeSilhouetteMode.AdaptiveImageToShape;
    const autoFit = params.autoFitMoldToSilhouette !== false;
    const silConformity = Math.max(
      0.0,
      Math.min(1.0, params.silhouetteConformity ?? 0.94)
    );
    const userScaleX = Math.max(0.4, Math.min(1.8, params.shapeScaleX ?? 1.0));
    const userScaleY = Math.max(0.4, Math.min(1.8, params.shapeScaleY ?? 1.0));
    const userAspect = Math.max(
      0.5,
      Math.min(1.6, params.shapeAspectRatio ?? 1.0)
    );
    const offX = Math.max(-0.35, Math.min(0.35, params.shapeOffsetX ?? 0.0));
    const offY = Math.max(-0.35, Math.min(0.35, params.shapeOffsetY ?? 0.0));
    const poseArmAngleDeg = Math.max(
      15.0,
      Math.min(90.0, params.poseArmAngleDeg ?? 35.0)
    );
    const limbUnionMode =
      params.limbUnionMode || LimbUnionMode.TubularCylindersFull;
    const limbUnionCurvature = Math.max(
      0.05,
      Math.min(1.0, params.limbUnionCurvature ?? 0.68)
    );
    const autoLimb = params.autoFitLimbSizes !== false;
    const effArmLen = Math.max(
      0.4,
      Math.min(
        1.85,
        (params.armLengthScale ?? 1.0) *
          (autoLimb ? buffers.autoDetectedArmLengthScale : 1.0)
      )
    );
    const effArmThick = Math.max(
      0.4,
      Math.min(
        1.85,
        (params.armThicknessScale ?? 1.0) *
          (autoLimb ? buffers.autoDetectedArmThicknessScale : 1.0)
      )
    );
    const effLegLen = Math.max(
      0.4,
      Math.min(
        1.85,
        (params.legLengthScale ?? 1.0) *
          (autoLimb ? buffers.autoDetectedLegLengthScale : 1.0)
      )
    );
    const effLegThick = Math.max(
      0.4,
      Math.min(
        1.85,
        (params.legThicknessScale ?? 1.0) *
          (autoLimb ? buffers.autoDetectedLegThicknessScale : 1.0)
      )
    );

    // Canonical preset mold occupies ~0.78 width and ~0.88 height of UV space
    const autoScaleX = autoFit ? buffers.autoSilhouetteWidth / 0.78 : 1.0;
    const autoScaleY = autoFit ? buffers.autoSilhouetteHeight / 0.88 : 1.0;
    const effScaleX = Math.max(0.25, autoScaleX * userScaleX * userAspect);
    const effScaleY = Math.max(0.25, autoScaleY * userScaleY);
    const effCenterX = (autoFit ? buffers.autoSilhouetteCenterX : 0.0) + offX;
    const effCenterY = (autoFit ? buffers.autoSilhouetteCenterY : 0.0) + offY;

    const hasDistinctDrawingSilhouette =
      mode !== BackgroundCutoutMode.None ||
      (fgCount > 36 && fgCount < TENSOR_LEN * 0.93);

    for (let ty = 0; ty < h; ty++) {
      // v in [0, 1] bottom-to-top
      const v = 1.0 - ty / (h - 1);
      const rowOff = ty * w;
      const rCenterU = 0.5 * (rowSilhouetteMinU[ty] + rowSilhouetteMaxU[ty]) - 0.5;
      const blendCenterX =
        effCenterX * (1.0 - silConformity * 0.45) +
        (rCenterU + offX) * (silConformity * 0.45);

      for (let tx = 0; tx < w; tx++) {
        const u = tx / (w - 1);
        const idx = rowOff + tx;
        const uAligned = 0.5 + (u - 0.5 - blendCenterX);
        const vAligned = 0.5 + (v - 0.5 - effCenterY);

        evaluatePresetBaseShapeZeroGC(
          presetShape,
          uAligned,
          vAligned,
          effScaleX,
          effScaleY,
          poseArmAngleDeg,
          limbUnionMode,
          limbUnionCurvature,
          effArmLen,
          effArmThick,
          effLegLen,
          effLegThick
        );
        const inShape = _shapeInside;

        if (silMode === ShapeSilhouetteMode.StrictShapeSilhouette) {
          // 100% Preset 3D Shape Silhouette: molds any photo/texture directly onto the Humanoid A-Pose, Sphere, Quadruped, etc.
          texelCutoutMask[idx] = inShape;
        } else if (silMode === ShapeSilhouetteMode.UnionImageAndShape) {
          // Union: keeps both the 3D preset mannequin and any extending parts of the image
          texelCutoutMask[idx] =
            texelCutoutMask[idx] === 1 || inShape === 1 ? 1 : 0;
        } else {
          // AdaptiveImageToShape ("Amoldar Dibujo a la Silueta - Igual en la Unión"):
          // When the image has a foreground silhouette, keep the EXACT silhouette of the drawing at the union zone!
          // The 3D mold volume conforms internally to this exact silhouette rather than cutting off the drawing.
          if (hasDistinctDrawingSilhouette) {
            texelCutoutMask[idx] =
              mode !== BackgroundCutoutMode.None
                ? texelCutoutMask[idx]
                : drawingSilhouetteMask[idx];
          } else {
            texelCutoutMask[idx] = inShape;
          }
        }
      }
    }
  }
}

export interface AutoSilhouetteFitMetrics {
  scaleX: number;
  scaleY: number;
  aspectRatio: number;
  offsetX: number;
  offsetY: number;
  widthPct: number;
  heightPct: number;
  armLengthScale: number;
  armThicknessScale: number;
  legLengthScale: number;
  legThicknessScale: number;
  armLengthPct: number;
  armThicknessPct: number;
  legLengthPct: number;
  legThicknessPct: number;
  poseArmAngleDeg: number;
}

const _autoMetricsResult: AutoSilhouetteFitMetrics = {
  scaleX: 1.0,
  scaleY: 1.0,
  aspectRatio: 1.0,
  offsetX: 0.0,
  offsetY: 0.0,
  widthPct: 84,
  heightPct: 88,
  armLengthScale: 1.0,
  armThicknessScale: 1.0,
  legLengthScale: 1.0,
  legThicknessScale: 1.0,
  armLengthPct: 100,
  armThicknessPct: 100,
  legLengthPct: 100,
  legThicknessPct: 100,
  poseArmAngleDeg: 35,
};

export function getDetectedSilhouetteFitMetrics(
  buffers: CNNReconstructionBuffers
): AutoSilhouetteFitMetrics {
  _autoMetricsResult.scaleX = Number(
    Math.max(0.45, Math.min(1.65, buffers.autoSilhouetteWidth / 0.78)).toFixed(2)
  );
  _autoMetricsResult.scaleY = Number(
    Math.max(0.45, Math.min(1.65, buffers.autoSilhouetteHeight / 0.88)).toFixed(2)
  );
  _autoMetricsResult.aspectRatio = Number(
    Math.max(0.55, Math.min(1.55, buffers.autoSilhouetteAspect)).toFixed(2)
  );
  _autoMetricsResult.offsetX = Number(
    Math.max(-0.3, Math.min(0.3, buffers.autoSilhouetteCenterX)).toFixed(2)
  );
  _autoMetricsResult.offsetY = Number(
    Math.max(-0.3, Math.min(0.3, buffers.autoSilhouetteCenterY)).toFixed(2)
  );
  _autoMetricsResult.widthPct = Math.round(buffers.autoSilhouetteWidth * 100);
  _autoMetricsResult.heightPct = Math.round(buffers.autoSilhouetteHeight * 100);
  _autoMetricsResult.armLengthScale = Number(
    buffers.autoDetectedArmLengthScale.toFixed(2)
  );
  _autoMetricsResult.armThicknessScale = Number(
    buffers.autoDetectedArmThicknessScale.toFixed(2)
  );
  _autoMetricsResult.legLengthScale = Number(
    buffers.autoDetectedLegLengthScale.toFixed(2)
  );
  _autoMetricsResult.legThicknessScale = Number(
    buffers.autoDetectedLegThicknessScale.toFixed(2)
  );
  _autoMetricsResult.armLengthPct = Math.round(
    buffers.autoDetectedArmLengthScale * 100
  );
  _autoMetricsResult.armThicknessPct = Math.round(
    buffers.autoDetectedArmThicknessScale * 100
  );
  _autoMetricsResult.legLengthPct = Math.round(
    buffers.autoDetectedLegLengthScale * 100
  );
  _autoMetricsResult.legThicknessPct = Math.round(
    buffers.autoDetectedLegThicknessScale * 100
  );
  _autoMetricsResult.poseArmAngleDeg = buffers.autoDetectedPoseArmAngleDeg;
  return _autoMetricsResult;
}

export function renderCNNFeaturePreview(
  view: CNNFeatureView,
  buffers: CNNReconstructionBuffers,
  zones: ZoneConfig[],
  targetCanvas: HTMLCanvasElement
): void {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  targetCanvas.width = w;
  targetCanvas.height = h;
  const ctx = targetCanvas.getContext('2d');
  if (!ctx) return;

  const imgData = ctx.createImageData(w, h);
  const data = imgData.data;

  const zoneRGB = zones.map((z) => {
    const c = new THREE.Color(z.colorHex);
    return [
      Math.floor(c.r * 255),
      Math.floor(c.g * 255),
      Math.floor(c.b * 255),
    ];
  });

  for (let i = 0; i < TENSOR_LEN; i++) {
    const i4 = i * 4;
    const i3 = i * 3;

    // Show dark transparency checkerboard on cut-out background pixels in the 2D preview
    if (buffers.texelCutoutMask[i] === 0) {
      const tx = i % w;
      const ty = (i / w) | 0;
      const checker = ((tx >> 3) ^ (ty >> 3)) & 1;
      const cVal = checker ? 18 : 10;
      data[i4] = cVal;
      data[i4 + 1] = cVal + 3;
      data[i4 + 2] = cVal + 8;
      data[i4 + 3] = 255;
      continue;
    }

    if (view === CNNFeatureView.RGBInput) {
      data[i4] = Math.floor(buffers.rgbInput[i3] * 255);
      data[i4 + 1] = Math.floor(buffers.rgbInput[i3 + 1] * 255);
      data[i4 + 2] = Math.floor(buffers.rgbInput[i3 + 2] * 255);
    } else if (view === CNNFeatureView.NormalTensor) {
      data[i4] = Math.floor((buffers.normalTensor[i3] * 0.5 + 0.5) * 255);
      data[i4 + 1] = Math.floor(
        (buffers.normalTensor[i3 + 1] * 0.5 + 0.5) * 255
      );
      data[i4 + 2] = Math.floor(
        (buffers.normalTensor[i3 + 2] * 0.5 + 0.5) * 255
      );
    } else if (view === CNNFeatureView.PoissonSurface) {
      const d = buffers.depthTensor[i];
      data[i4] = Math.floor(Math.min(255, Math.pow(d, 1.4) * 250 + 15));
      data[i4 + 1] = Math.floor(Math.min(255, d * 215 + 20));
      data[i4 + 2] = Math.floor(
        Math.min(255, (0.3 + 0.7 * Math.sin(d * Math.PI * 0.75)) * 245)
      );
    } else if (view === CNNFeatureView.DivergenceCurvature) {
      const div = buffers.divergenceField[i];
      const pos = Math.min(1, Math.max(0, -div * 1.6));
      const neg = Math.min(1, Math.max(0, div * 1.6));
      data[i4] = Math.floor(24 + pos * 230);
      data[i4 + 1] = Math.floor(32 + pos * 165 + neg * 175);
      data[i4 + 2] = Math.floor(48 + neg * 205);
    } else if (view === CNNFeatureView.StraightLines) {
      // Visualize Structure Tensor Straight-Line Coherence (Cyan/Gold Straight Edges over Planar Facets)
      const coh = buffers.lineCoherenceTensor[i];
      const d = buffers.depthTensor[i];
      const tx = Math.abs(buffers.lineTangentX[i]);
      const ty = Math.abs(buffers.lineTangentY[i]);
      data[i4] = Math.min(255, Math.floor(18 + d * 45 + coh * (140 + tx * 110)));
      data[i4 + 1] = Math.min(255, Math.floor(24 + d * 55 + coh * (195 + ty * 55)));
      data[i4 + 2] = Math.min(255, Math.floor(38 + d * 70 + coh * 245));
    } else {
      const zId = buffers.zoneMap[i] & 3;
      const rgb = zoneRGB[zId] || [200, 200, 200];
      const shade = 0.55 + 0.45 * buffers.depthTensor[i];
      data[i4] = Math.floor(rgb[0] * shade);
      data[i4 + 1] = Math.floor(rgb[1] * shade);
      data[i4 + 2] = Math.floor(rgb[2] * shade);
    }
    data[i4 + 3] = 255;
  }

  ctx.putImageData(imgData, 0, 0);
}

function evaluateSphericalHarmonic(
  theta: number,
  phi: number,
  l: number,
  m: number
): number {
  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);
  const mAngle = m * phi;
  const angular = m === 0 ? 1.0 : Math.cos(mAngle) + 0.5 * Math.sin(mAngle);

  if (l === 1) return cosTheta * angular;
  if (l === 2) return 0.5 * (3 * cosTheta * cosTheta - 1) * sinTheta * angular;
  if (l === 3)
    return 0.5 * cosTheta * (5 * cosTheta * cosTheta - 3) * angular;
  if (l === 4)
    return (
      0.125 *
      (35 * Math.pow(cosTheta, 4) - 30 * cosTheta * cosTheta + 3) *
      angular
    );
  return Math.sin(l * theta) * Math.cos(m * phi);
}

function evaluateZonePattern(
  pattern: TexturePattern,
  u: number,
  v: number
): number {
  if (
    pattern === TexturePattern.OriginalPhoto ||
    pattern === TexturePattern.Solid
  ) {
    return 1.0;
  }

  const su = u * 24.0;
  const sv = v * 24.0;

  switch (pattern) {
    case TexturePattern.CarbonFiber: {
      const cx = Math.floor(su * 2) & 1;
      const cy = Math.floor(sv * 2) & 1;
      const diag = Math.sin((su + sv) * Math.PI * 2) * 0.5 + 0.5;
      return (cx ^ cy ? 0.76 : 1.04) * (0.85 + 0.15 * diag);
    }
    case TexturePattern.VeinedMarble: {
      const turb =
        Math.sin(su * 1.5 + Math.sin(sv * 2.2) * 1.8) +
        0.5 * Math.cos(sv * 3.1 + Math.sin(su * 2.7));
      const vein = Math.pow(Math.abs(Math.sin(su * 1.2 + turb * 1.6)), 0.35);
      return 0.68 + 0.36 * vein;
    }
    case TexturePattern.BrushedMetal: {
      const streak =
        0.5 +
        0.25 * Math.sin(sv * 42.0 + Math.sin(su * 4.0)) +
        0.25 * Math.cos(sv * 74.0);
      return 0.85 + 0.2 * streak;
    }
    case TexturePattern.HexScales: {
      const hx = su * 1.8;
      const hy = sv * 2.08;
      const cell =
        Math.cos(hx * Math.PI) * Math.cos(hy * Math.PI) +
        Math.cos((hx * 0.5 + hy * 0.866) * Math.PI);
      return cell > 0.2 ? 1.08 : 0.74;
    }
    case TexturePattern.NeuralCircuit: {
      const gx = Math.abs(((su * 1.5) % 1) - 0.5);
      const gy = Math.abs(((sv * 1.5) % 1) - 0.5);
      return Math.min(gx, gy) < 0.08 ? 1.28 : 0.92;
    }
    default:
      return 1.0;
  }
}

// ============================================================================
// ZERO-GC PRESET 3D BASE SHAPE EVALUATOR
// ("Cuerpo Humano en Pose A", "Esfera 3D", "Animal Cuadrúpedo", "Busto", "Coraza", "Cilindro")
// Uses pre-allocated module-level scalars and squared distances (distSq) to avoid GC pauses.
// ============================================================================
let _shapeInside = 0;
let _shapeFrontZ = 0.0;
let _shapeBackZ = 0.0;
let _shapeNx = 0.0;
let _shapeNy = 0.0;
let _shapeNz = 1.0;
let _shapeWarpU = 0.5;
let _shapeWarpV = 0.5;

let _capNx = 0.0;
let _capNy = 0.0;
let _tubularWallFloor = 0.28;

/**
 * Evaluates a 3D tapered capsule/cylinder segment from (ax, ay) to (bx, by) with radii rA -> rB.
 * Uses squared Euclidean distance (distSq) for rejection before computing the 3D cross-section height.
 * When _tubularWallFloor > 0, the cylinder maintains a round tubular wall height right up to the lateral union!
 */
function evalCapsule3D(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
  rA: number,
  rB: number,
  zScale: number
): number {
  const abx = bx - ax;
  const aby = by - ay;
  const lenSq = abx * abx + aby * aby;
  let t = 0.0;
  if (lenSq > 1e-6) {
    t = ((px - ax) * abx + (py - ay) * aby) / lenSq;
    if (t < 0.0) t = 0.0;
    else if (t > 1.0) t = 1.0;
  }
  const cx = ax + t * abx;
  const cy = ay + t * aby;
  const r = rA + t * (rB - rA);
  const rSq = r * r;
  const dx = px - cx;
  const dy = py - cy;
  const distSq = dx * dx + dy * dy;
  if (distSq >= rSq) {
    return 0.0;
  }
  const qRaw = 1.0 - distSq / rSq;
  // Blend qRaw with _tubularWallFloor so tubular limbs keep their cylindrical wall thickness at the union seam
  const q = _tubularWallFloor + (1.0 - _tubularWallFloor) * qRaw;
  const invR = 1.0 / Math.max(1e-5, r);
  _capNx = dx * invR;
  _capNy = dy * invR;
  return r * zScale * Math.sqrt(q);
}

/**
 * Polynomial smooth-maximum (smax) to blend intersecting 3D anatomical tubes/cylinders with a curved fillet.
 */
function smax3D(a: number, b: number, k: number): number {
  if (a <= 0.0) return b;
  if (b <= 0.0) return a;
  const kSafe = Math.max(0.008, k);
  const h = Math.max(kSafe - Math.abs(a - b), 0.0) / kSafe;
  return Math.max(a, b) + h * h * kSafe * 0.25;
}

/**
 * Evaluates the 3D Preset Base Shape at normalized UV coordinates (u, v) in [0, 1]^2.
 * Populates _shapeInside, _shapeFrontZ, _shapeBackZ, _shapeNx, _shapeNy, _shapeNz, _shapeWarpU, _shapeWarpV.
 */
export function evaluatePresetBaseShapeZeroGC(
  shape: PresetBaseShape,
  u: number,
  v: number,
  scaleX = 1.0,
  scaleY = 1.0,
  poseArmAngleDeg = 35.0,
  limbUnionMode: LimbUnionMode = LimbUnionMode.TubularCylindersFull,
  limbUnionCurvature = 0.68,
  armLengthScale = 1.0,
  armThicknessScale = 1.0,
  legLengthScale = 1.0,
  legThicknessScale = 1.0
): void {
  _shapeInside = 1;
  _shapeFrontZ = 0.0;
  _shapeBackZ = 0.0;
  _shapeNx = 0.0;
  _shapeNy = 0.0;
  _shapeNz = 1.0;
  _shapeWarpU = u;
  _shapeWarpV = v;

  if (!shape || shape === PresetBaseShape.None) {
    return;
  }

  const armLen = Math.max(0.4, Math.min(1.85, armLengthScale));
  const armThick = Math.max(0.4, Math.min(1.85, armThicknessScale));
  const legLen = Math.max(0.4, Math.min(1.85, legLengthScale));
  const legThick = Math.max(0.4, Math.min(1.85, legThicknessScale));

  // Configure tubular cylinder wall floor & inter-limb union curvature based on user-selected LimbUnionMode
  const curveVal = Math.max(0.05, Math.min(1.0, limbUnionCurvature));
  let kUnionScale = 0.3 + curveVal * 1.45;
  let volumeProfileMult = 0.82; // Calibrated sleek, non-bloated anatomical proportion

  if (limbUnionMode === LimbUnionMode.TubularCylindersFull) {
    // True cylindrical/tubular cross-sections of the full thickness of each limb/torso segment
    _tubularWallFloor = 0.34;
    kUnionScale = 0.35 + curveVal * 1.55;
    volumeProfileMult = 0.84;
  } else if (limbUnionMode === LimbUnionMode.SmoothCurvedUnions) {
    // Extra-smooth organic curved transitions between parts
    _tubularWallFloor = 0.18;
    kUnionScale = 0.55 + curveVal * 1.85;
    volumeProfileMult = 0.78;
  } else if (limbUnionMode === LimbUnionMode.AnatomicalArticulated) {
    // Defined muscular/articulated creases at joints
    _tubularWallFloor = 0.22;
    kUnionScale = 0.15 + curveVal * 0.65;
    volumeProfileMult = 0.80;
  } else {
    // SlimBasRelief: Low initial volume so high-detail photo reliefs sit on a sleek base without any bloating
    _tubularWallFloor = 0.12;
    kUnionScale = 0.35 + curveVal * 1.1;
    volumeProfileMult = 0.45;
  }

  // Normalized coordinates x, y in [-1, +1] scaled by user proportion sliders
  const x = ((u - 0.5) * 2.0) / Math.max(0.4, scaleX);
  const y = ((v - 0.5) * 2.0) / Math.max(0.4, scaleY);
  const sx = Math.abs(x);
  const signX = x >= 0.0 ? 1.0 : -1.0;

  // Clamp pose angle between 15° (low A-pose) and 90° (full horizontal T-pose)
  const clampedArmDeg = Math.max(15.0, Math.min(90.0, poseArmAngleDeg));
  const armAngleRad = (clampedArmDeg * Math.PI) / 180.0;
  const sinArm = Math.sin(armAngleRad);
  const cosArm = Math.cos(armAngleRad);

  if (shape === PresetBaseShape.Sphere3D) {
    // Full 3D Volumetric Sphere (R = 0.88)
    const R = 0.88;
    const rSq = x * x + y * y;
    const RSq = R * R;
    if (rSq > RSq) {
      _shapeInside = 0;
      return;
    }
    const qRaw = Math.max(0.0, 1.0 - rSq / RSq);
    const zUnit = Math.sqrt(_tubularWallFloor * 0.6 + (1.0 - _tubularWallFloor * 0.6) * qRaw);
    _shapeFrontZ = zUnit * 0.62 * volumeProfileMult;
    _shapeBackZ = zUnit * 0.62 * volumeProfileMult;
    _shapeNx = (x / R) * 0.85;
    _shapeNy = (y / R) * 0.85;
    _shapeNz = Math.max(0.25, zUnit);

    // Spherical conformal UV projection so flat images wrap around the sphere curvature
    const lon = Math.asin(Math.max(-0.99, Math.min(0.99, x / R))) / (Math.PI * 0.5);
    const lat = Math.asin(Math.max(-0.99, Math.min(0.99, y / R))) / (Math.PI * 0.5);
    _shapeWarpU = 0.5 + lon * 0.46;
    _shapeWarpV = 0.5 + lat * 0.46;
    return;
  }

  if (shape === PresetBaseShape.HumanoidAPose) {
    // ========================================================================
    // CUERPO HUMANO EN POSE A ↔ POSE T (Graduable de 15° hasta 90°)
    // Calibrated sleek cylindrical/tubular anatomy without bloating + curved unions
    // Supports adjustable & automatic Arm/Leg Length & Thickness scaling
    // ========================================================================
    let hFront = 0.0;
    let accNx = 0.0;
    let accNy = 0.0;

    // 1. Head & Cranium (Cabeza y Rostro — proporción esbelta)
    const hHead = evalCapsule3D(sx, y, 0.0, 0.62, 0.0, 0.78, 0.122, 0.138, 1.05);
    if (hHead > 0.0) {
      hFront = hHead;
      accNx = _capNx * signX;
      accNy = _capNy;
    }

    // 2. Neck Cylinder (Tubo de Cuello con unión curva hacia cabeza y tórax)
    const hNeck = evalCapsule3D(sx, y, 0.0, 0.44, 0.0, 0.62, 0.096, 0.080, 1.0);
    if (hNeck > 0.0) {
      if (hNeck > hFront) {
        accNx = _capNx * signX;
        accNy = _capNy;
      }
      hFront = smax3D(hFront, hNeck, 0.042 * kUnionScale);
    }

    // 3. Upper Chest & Deltoid Shoulder Bridge (Tórax Superior y Puente de Hombros)
    const deltoidR = 0.135 * (0.4 + 0.6 * armThick);
    const hChest = evalCapsule3D(sx, y, 0.0, 0.36, 0.23, 0.38, 0.185, deltoidR, 0.92);
    if (hChest > 0.0) {
      if (hChest > hFront) {
        accNx = _capNx * signX;
        accNy = _capNy;
      }
      hFront = smax3D(hFront, hChest, 0.055 * kUnionScale);
    }

    // 4. Ribcage & Abdomen (Tronco y Abdomen — elíptico esbelto sin abultamiento)
    const hTorso = evalCapsule3D(sx, y, 0.0, 0.36, 0.0, -0.06, 0.215, 0.162, 0.86);
    if (hTorso > 0.0) {
      if (hTorso > hFront) {
        accNx = _capNx * signX * 0.85;
        accNy = _capNy;
      }
      hFront = smax3D(hFront, hTorso, 0.062 * kUnionScale);
    }

    // 5. Pelvis & Hips (Pelvis y Cadera con unión curva hacia abdomen y muslos)
    const hipR = 0.148 * (0.45 + 0.55 * legThick);
    const hPelvis = evalCapsule3D(sx, y, 0.0, -0.08, 0.14, -0.11, 0.168, hipR, 0.88);
    if (hPelvis > 0.0) {
      if (hPelvis > hFront) {
        accNx = _capNx * signX;
        accNy = _capNy;
      }
      hFront = smax3D(hFront, hPelvis, 0.055 * kUnionScale);
    }

    // 6. Articulated Cylindrical Arms from Shoulder Pivot (0.23, 0.38) graduated 15°..90° & scaled by armLen / armThick
    const shoulderX = 0.23;
    const shoulderY = 0.38;
    const upperArmLen = 0.29 * armLen;
    const forearmLen = 0.26 * armLen;
    const handLen = 0.11 * armLen;

    const elbowX = shoulderX + upperArmLen * sinArm;
    const elbowY = shoulderY - upperArmLen * cosArm;

    const wristX = elbowX + forearmLen * sinArm;
    const wristY = elbowY - forearmLen * cosArm;

    const handX = wristX + handLen * sinArm;
    const handY = wristY - handLen * cosArm;

    const hUpperArm = evalCapsule3D(
      sx,
      y,
      shoulderX,
      shoulderY,
      elbowX,
      elbowY,
      0.104 * armThick,
      0.086 * armThick,
      1.02
    );
    if (hUpperArm > 0.0) {
      if (hUpperArm > hFront) {
        accNx = _capNx * signX;
        accNy = _capNy;
      }
      hFront = smax3D(hFront, hUpperArm, 0.048 * kUnionScale);
    }

    const hForearm = evalCapsule3D(
      sx,
      y,
      elbowX,
      elbowY,
      wristX,
      wristY,
      0.084 * armThick,
      0.066 * armThick,
      1.02
    );
    if (hForearm > 0.0) {
      if (hForearm > hFront) {
        accNx = _capNx * signX;
        accNy = _capNy;
      }
      hFront = smax3D(hFront, hForearm, 0.042 * kUnionScale);
    }

    const hHand = evalCapsule3D(
      sx,
      y,
      wristX,
      wristY,
      handX,
      handY,
      0.068 * armThick,
      0.054 * armThick,
      0.92
    );
    if (hHand > 0.0) {
      if (hHand > hFront) {
        accNx = _capNx * signX;
        accNy = _capNy;
      }
      hFront = smax3D(hFront, hHand, 0.035 * kUnionScale);
    }

    // 7. Cylindrical Legs in A-Stance scaled by legLen / legThick
    const hipX = 0.14;
    const hipY = -0.11;
    const kneeX = hipX + 0.06 * legLen;
    const kneeY = hipY - 0.39 * legLen;
    const ankleX = kneeX + 0.04 * legLen;
    const ankleY = kneeY - 0.36 * legLen;
    const footX = ankleX + 0.03 * legLen;
    const footY = ankleY - 0.07 * legLen;

    const hThigh = evalCapsule3D(
      sx,
      y,
      hipX,
      hipY,
      kneeX,
      kneeY,
      0.138 * legThick,
      0.102 * legThick,
      1.02
    );
    if (hThigh > 0.0) {
      if (hThigh > hFront) {
        accNx = _capNx * signX;
        accNy = _capNy;
      }
      hFront = smax3D(hFront, hThigh, 0.052 * kUnionScale);
    }

    const hCalf = evalCapsule3D(
      sx,
      y,
      kneeX,
      kneeY,
      ankleX,
      ankleY,
      0.098 * legThick,
      0.070 * legThick,
      1.02
    );
    if (hCalf > 0.0) {
      if (hCalf > hFront) {
        accNx = _capNx * signX;
        accNy = _capNy;
      }
      hFront = smax3D(hFront, hCalf, 0.042 * kUnionScale);
    }

    const hFoot = evalCapsule3D(
      sx,
      y,
      ankleX,
      ankleY,
      footX,
      footY,
      0.080 * legThick,
      0.066 * legThick,
      1.08
    );
    if (hFoot > 0.0) {
      hFront = smax3D(hFront, hFoot, 0.038 * kUnionScale);
    }

    if (hFront <= 0.001) {
      _shapeInside = 0;
      return;
    }

    // Subtle anatomical contours (scaled by volumeProfileMult so they never bloat the chest or back)
    const pecBulge =
      Math.exp(-((sx - 0.10) * (sx - 0.10)) / 0.009 - ((y - 0.34) * (y - 0.34)) / 0.008) * 0.026;
    const sternumGroove =
      Math.exp(-(sx * sx) / 0.0012 - ((y - 0.22) * (y - 0.22)) / 0.045) * 0.016;
    // Dorsal anatomy (Espalda: omóplatos/escápulas, dorsal ancho, surco espinal, glúteos y gemelos)
    const occipitalDome =
      Math.exp(-(sx * sx) / 0.008 - ((y - 0.72) * (y - 0.72)) / 0.009) * 0.024;
    const scapulaBulge =
      Math.exp(-((sx - 0.11) * (sx - 0.11)) / 0.0085 - ((y - 0.34) * (y - 0.34)) / 0.010) * 0.030;
    const latDorsi =
      Math.exp(-((sx - 0.13) * (sx - 0.13)) / 0.011 - ((y - 0.18) * (y - 0.18)) / 0.016) * 0.022;
    const spineGroove =
      Math.exp(-(sx * sx) / 0.0016 - ((y - 0.20) * (y - 0.20)) / 0.068) * 0.026;
    const lumbarHollow =
      Math.exp(-(sx * sx) / 0.009 - ((y - 0.02) * (y - 0.02)) / 0.009) * 0.018;
    const gluteBulge =
      Math.exp(-((sx - 0.095) * (sx - 0.095)) / 0.0085 - ((y + 0.12) * (y + 0.12)) / 0.0095) * 0.038;
    const calfMidX = 0.5 * (kneeX + ankleX);
    const calfMidY = 0.5 * (kneeY + ankleY) + 0.05;
    const calfBulge =
      Math.exp(
        -((sx - calfMidX) * (sx - calfMidX)) / (0.0045 * legThick * legThick) -
          ((y - calfMidY) * (y - calfMidY)) / (0.014 * legLen * legLen)
      ) *
      0.022 *
      legThick;

    _shapeFrontZ = Math.max(0.0, (hFront + pecBulge - sternumGroove) * volumeProfileMult);
    _shapeBackZ = Math.max(
      0.0,
      (hFront + occipitalDome + scapulaBulge + latDorsi + gluteBulge + calfBulge - spineGroove - lumbarHollow) *
        volumeProfileMult
    );

    const nLen = Math.sqrt(accNx * accNx + accNy * accNy + 1.0);
    _shapeNx = (accNx / nLen) * 0.65;
    _shapeNy = (accNy / nLen) * 0.65;
    _shapeNz = 1.0 / nLen;

    // Slight cylindrical UV wrap across limbs/torso
    _shapeWarpU = Math.max(0.0, Math.min(1.0, u + accNx * 0.03));
    _shapeWarpV = Math.max(0.0, Math.min(1.0, v + accNy * 0.02));
    return;
  }

  if (shape === PresetBaseShape.QuadrupedAnimal) {
    // ========================================================================
    // ANIMAL CUADRÚPEDO 3D (Tubular Limbs & Curved Unions without Bloating)
    // ========================================================================
    let hQuad = 0.0;
    let accNx = 0.0;
    let accNy = 0.0;

    // 1. Snout & Head (Hocico y Cráneo)
    const hSnout = evalCapsule3D(x, y, -0.80, 0.34, -0.55, 0.46, 0.105, 0.165, 1.02);
    if (hSnout > 0.0) {
      hQuad = hSnout;
      accNx = _capNx;
      accNy = _capNy;
    }

    // 2. Ears (Orejas erguidas)
    const hEars = evalCapsule3D(x, y, -0.54, 0.48, -0.48, 0.70, 0.082, 0.038, 0.88);
    if (hEars > 0.0) {
      hQuad = smax3D(hQuad, hEars, 0.038 * kUnionScale);
    }

    // 3. Muscular Neck (Cuello con unión curva)
    const hNeck = evalCapsule3D(x, y, -0.54, 0.42, -0.28, 0.15, 0.155, 0.205, 1.02);
    if (hNeck > 0.0) {
      if (hNeck > hQuad) {
        accNx = _capNx;
        accNy = _capNy;
      }
      hQuad = smax3D(hQuad, hNeck, 0.062 * kUnionScale);
    }

    // 4. Chest & Ribcage Barrel (Pecho y Tronco Cilíndrico)
    const hBarrel = evalCapsule3D(x, y, -0.28, 0.12, 0.34, 0.13, 0.242, 0.202, 1.05);
    if (hBarrel > 0.0) {
      if (hBarrel > hQuad) {
        accNx = _capNx;
        accNy = _capNy;
      }
      hQuad = smax3D(hQuad, hBarrel, 0.068 * kUnionScale);
    }

    // 5. Haunches / Pelvis (Grupa y Cuartos Traseros)
    const hHaunch = evalCapsule3D(x, y, 0.30, 0.13, 0.46, 0.08, 0.212, 0.192, 1.02);
    if (hHaunch > 0.0) {
      if (hHaunch > hQuad) {
        accNx = _capNx;
        accNy = _capNy;
      }
      hQuad = smax3D(hQuad, hHaunch, 0.062 * kUnionScale);
    }

    // 6. Tail (Cola Tubular)
    const hTail1 = evalCapsule3D(x, y, 0.46, 0.12, 0.68, 0.02, 0.082, 0.062, 0.96);
    const hTail2 = evalCapsule3D(x, y, 0.68, 0.02, 0.85, -0.20, 0.062, 0.045, 0.95);
    if (hTail1 > 0.0) hQuad = smax3D(hQuad, hTail1, 0.04 * kUnionScale);
    if (hTail2 > 0.0) hQuad = smax3D(hQuad, hTail2, 0.035 * kUnionScale);

    // 7. Front Pair of Legs (2 Patas Delanteras Tubulares — escaladas con armLen / armThick)
    const flKneeY = 0.02 - 0.44 * armLen;
    const flFootY = flKneeY - 0.42 * armLen;
    const hFL1 = evalCapsule3D(x, y, -0.33, 0.02, -0.42, flKneeY, 0.115 * armThick, 0.082 * armThick, 1.02);
    const hFL2 = evalCapsule3D(x, y, -0.42, flKneeY, -0.45, flFootY, 0.080 * armThick, 0.064 * armThick, 1.02);
    if (hFL1 > 0.0) hQuad = smax3D(hQuad, hFL1, 0.048 * kUnionScale);
    if (hFL2 > 0.0) hQuad = smax3D(hQuad, hFL2, 0.040 * kUnionScale);

    const frKneeY = 0.00 - 0.44 * armLen;
    const frFootY = frKneeY - 0.41 * armLen;
    const hFR1 = evalCapsule3D(x, y, -0.16, 0.00, -0.14, frKneeY, 0.110 * armThick, 0.080 * armThick, 0.98);
    const hFR2 = evalCapsule3D(x, y, -0.14, frKneeY, -0.15, frFootY, 0.078 * armThick, 0.062 * armThick, 0.98);
    if (hFR1 > 0.0) hQuad = smax3D(hQuad, hFR1, 0.048 * kUnionScale);
    if (hFR2 > 0.0) hQuad = smax3D(hQuad, hFR2, 0.040 * kUnionScale);

    // 8. Hind Pair of Legs (2 Patas Traseras Tubulares — escaladas con legLen / legThick)
    const hlKneeY = 0.02 - 0.44 * legLen;
    const hlFootY = hlKneeY - 0.43 * legLen;
    const hHL1 = evalCapsule3D(x, y, 0.28, 0.02, 0.23, hlKneeY, 0.132 * legThick, 0.086 * legThick, 1.02);
    const hHL2 = evalCapsule3D(x, y, 0.23, hlKneeY, 0.19, hlFootY, 0.084 * legThick, 0.065 * legThick, 1.02);
    if (hHL1 > 0.0) hQuad = smax3D(hQuad, hHL1, 0.052 * kUnionScale);
    if (hHL2 > 0.0) hQuad = smax3D(hQuad, hHL2, 0.040 * kUnionScale);

    const hrKneeY = 0.02 - 0.42 * legLen;
    const hrFootY = hrKneeY - 0.43 * legLen;
    const hHR1 = evalCapsule3D(x, y, 0.44, 0.02, 0.54, hrKneeY, 0.132 * legThick, 0.086 * legThick, 1.04);
    const hHR2 = evalCapsule3D(x, y, 0.54, hrKneeY, 0.51, hrFootY, 0.084 * legThick, 0.065 * legThick, 1.04);
    if (hHR1 > 0.0) hQuad = smax3D(hQuad, hHR1, 0.052 * kUnionScale);
    if (hHR2 > 0.0) hQuad = smax3D(hQuad, hHR2, 0.040 * kUnionScale);

    if (hQuad <= 0.001) {
      _shapeInside = 0;
      return;
    }

    _shapeFrontZ = hQuad * volumeProfileMult;
    _shapeBackZ = hQuad * volumeProfileMult;
    const nLen = Math.sqrt(accNx * accNx + accNy * accNy + 1.0);
    _shapeNx = (accNx / nLen) * 0.65;
    _shapeNy = (accNy / nLen) * 0.65;
    _shapeNz = 1.0 / nLen;
    _shapeWarpU = Math.max(0.0, Math.min(1.0, u + accNx * 0.03));
    _shapeWarpV = Math.max(0.0, Math.min(1.0, v + accNy * 0.03));
    return;
  }

  if (shape === PresetBaseShape.AnatomicalBust) {
    // ========================================================================
    // BUSTO ANATÓMICO 3D (Cabeza, Rostro, Cuello y Hombros Escultóricos)
    // ========================================================================
    let hBust = 0.0;
    let accNx = 0.0;
    let accNy = 0.0;

    // Cranium & Face
    const hHead = evalCapsule3D(sx, y, 0.0, 0.14, 0.0, 0.54, 0.25, 0.30, 0.95);
    if (hHead > 0.0) {
      hBust = hHead;
      accNx = _capNx * signX;
      accNy = _capNy;
    }
    // Ears
    const hEar = evalCapsule3D(sx, y, 0.28, 0.24, 0.31, 0.38, 0.065, 0.072, 0.82);
    if (hEar > 0.0) hBust = smax3D(hBust, hEar, 0.038 * kUnionScale);

    // Neck
    const hNeck = evalCapsule3D(sx, y, 0.0, -0.26, 0.0, 0.10, 0.185, 0.165, 0.92);
    if (hNeck > 0.0) {
      if (hNeck > hBust) {
        accNx = _capNx * signX;
        accNy = _capNy;
      }
      hBust = smax3D(hBust, hNeck, 0.062 * kUnionScale);
    }

    // Broad Sculptural Shoulders & Upper Chest Pedestal
    const hShoulders = evalCapsule3D(
      sx,
      y,
      0.0,
      -0.44,
      0.58 * (0.4 + 0.6 * armLen),
      -0.52,
      0.36,
      0.24 * armThick,
      0.86
    );
    if (hShoulders > 0.0) {
      if (hShoulders > hBust) {
        accNx = _capNx * signX;
        accNy = _capNy;
      }
      hBust = smax3D(hBust, hShoulders, 0.068 * kUnionScale);
    }

    if (hBust <= 0.001) {
      _shapeInside = 0;
      return;
    }

    // Nose & Brow prominence on Front; Occipital & Trapezius sculpt on Back
    const noseRidge =
      Math.exp(-(sx * sx) / 0.0028 - ((y - 0.30) * (y - 0.30)) / 0.015) * 0.055;
    const occipitalBack =
      Math.exp(-(sx * sx) / 0.022 - ((y - 0.36) * (y - 0.36)) / 0.025) * 0.038;
    const cervicalGroove =
      Math.exp(-(sx * sx) / 0.0025 - ((y + 0.22) * (y + 0.22)) / 0.035) * 0.024;
    _shapeFrontZ = (hBust + noseRidge) * volumeProfileMult;
    _shapeBackZ = Math.max(0.0, (hBust + occipitalBack - cervicalGroove) * volumeProfileMult);

    const nLen = Math.sqrt(accNx * accNx + accNy * accNy + 1.0);
    _shapeNx = (accNx / nLen) * 0.65;
    _shapeNy = (accNy / nLen) * 0.65;
    _shapeNz = 1.0 / nLen;
    _shapeWarpU = Math.max(0.0, Math.min(1.0, u + accNx * 0.035));
    _shapeWarpV = Math.max(0.0, Math.min(1.0, v + accNy * 0.025));
    return;
  }

  if (shape === PresetBaseShape.HeroicCuirass) {
    // ========================================================================
    // CORAZA / ARMADURA TORSO (Peto Anatómico con Hombreras y Quilla Central)
    // ========================================================================
    let hArmor = 0.0;
    let accNx = 0.0;
    let accNy = 0.0;

    // Main Torso Armor Core
    const hCore = evalCapsule3D(sx, y, 0.0, 0.44, 0.0, -0.52, 0.52, 0.38, 0.78);
    if (hCore > 0.0) {
      hArmor = hCore;
      accNx = _capNx * signX;
      accNy = _capNy;
    }
    // Pauldrons (Hombreras) graduated with poseArmAngleDeg (15° to 90°) & armLen/armThick
    const pauldronEndX = 0.38 + 0.34 * armLen * sinArm;
    const pauldronEndY = 0.42 - 0.24 * armLen * cosArm;
    const hPauldron = evalCapsule3D(
      sx,
      y,
      0.38,
      0.42,
      pauldronEndX,
      pauldronEndY,
      0.24 * armThick,
      0.18 * armThick,
      0.92
    );
    if (hPauldron > 0.0) {
      if (hPauldron > hArmor) {
        accNx = _capNx * signX;
        accNy = _capNy;
      }
      hArmor = smax3D(hArmor, hPauldron, 0.06 * kUnionScale);
    }

    if (hArmor <= 0.001) {
      _shapeInside = 0;
      return;
    }

    // Central sternum keel ridge on Front; Scapular plates & spinal channel on Back
    const keel = Math.max(0.0, 1.0 - sx * 3.2) * 0.045;
    const backPlates =
      Math.exp(-((sx - 0.18) * (sx - 0.18)) / 0.018 - ((y - 0.18) * (y - 0.18)) / 0.035) * 0.038;
    const backSpineChannel =
      Math.exp(-(sx * sx) / 0.0022 - (y * y) / 0.12) * 0.025;
    _shapeFrontZ = (hArmor + keel) * volumeProfileMult;
    _shapeBackZ = Math.max(0.0, (hArmor + backPlates - backSpineChannel) * volumeProfileMult);

    const nLen = Math.sqrt(accNx * accNx + accNy * accNy + 1.0);
    _shapeNx = (accNx / nLen) * 0.65;
    _shapeNy = (accNy / nLen) * 0.65;
    _shapeNz = 1.0 / nLen;
    _shapeWarpU = Math.max(0.0, Math.min(1.0, u + accNx * 0.035));
    _shapeWarpV = v;
    return;
  }

  if (shape === PresetBaseShape.CylindricalTotem) {
    // ========================================================================
    // CILINDRO / TÓTEM VERTICAL 3D
    // ========================================================================
    const R = 0.74;
    if (sx > R || Math.abs(y) > 0.90) {
      _shapeInside = 0;
      return;
    }
    const zCyl = Math.sqrt(Math.max(0.0, 1.0 - (x * x) / (R * R)));
    const capTaper = Math.min(1.0, (0.90 - Math.abs(y)) * 12.0);
    _shapeFrontZ = zCyl * 0.64 * capTaper;
    _shapeBackZ = zCyl * 0.64 * capTaper;
    _shapeNx = (x / R) * 0.85;
    _shapeNy = 0.0;
    _shapeNz = Math.max(0.2, zCyl);

    const lon = Math.asin(Math.max(-0.99, Math.min(0.99, x / R))) / (Math.PI * 0.5);
    _shapeWarpU = 0.5 + lon * 0.47;
    _shapeWarpV = v;
  }
}

let _limbSlabX = 0.0;
let _limbSlabY = 0.0;
let _limbZMult = 1.0;

/**
 * Zero-GC Spatial Anatomical Limb Scaling for the 3D Mesh.
 * Smoothly expands or contracts the 3D coordinates (slabX, slabY) and Z volume multiplier
 * in the arm and leg anatomical zones when the user adjusts Arm/Leg Length or Thickness.
 */
function evaluateAnatomicalLimbSpatialWarpZeroGC(
  slabX: number,
  slabY: number,
  uMold: number,
  vMold: number,
  scaleX: number,
  scaleY: number,
  poseArmAngleDeg: number,
  userArmLen: number,
  userArmThick: number,
  userLegLen: number,
  userLegThick: number
): void {
  _limbSlabX = slabX;
  _limbSlabY = slabY;
  _limbZMult = 1.0;

  if (
    Math.abs(userArmLen - 1.0) < 0.005 &&
    Math.abs(userArmThick - 1.0) < 0.005 &&
    Math.abs(userLegLen - 1.0) < 0.005 &&
    Math.abs(userLegThick - 1.0) < 0.005
  ) {
    return;
  }

  const x = ((uMold - 0.5) * 2.0) / Math.max(0.4, scaleX);
  const y = ((vMold - 0.5) * 2.0) / Math.max(0.4, scaleY);
  const sx = Math.abs(x);
  const signX = slabX >= 0.0 ? 1.0 : -1.0;

  // 1. Upper-Body Arm Region Deformation (outside torso core sx > 0.16, y in [-0.24..0.64])
  if (
    sx > 0.16 &&
    y > -0.24 &&
    y < 0.64 &&
    (Math.abs(userArmLen - 1.0) >= 0.005 || Math.abs(userArmThick - 1.0) >= 0.005)
  ) {
    const tx = Math.max(0.0, Math.min(1.0, (sx - 0.16) / 0.14));
    const wx = tx * tx * (3.0 - 2.0 * tx);

    const tyLow = Math.max(0.0, Math.min(1.0, (y + 0.24) / 0.18));
    const wyLow = tyLow * tyLow * (3.0 - 2.0 * tyLow);

    const tyHigh = Math.max(0.0, Math.min(1.0, (0.64 - y) / 0.16));
    const wyHigh = tyHigh * tyHigh * (3.0 - 2.0 * tyHigh);

    const wArm = wx * wyLow * wyHigh;
    if (wArm > 0.001) {
      const clampedArmDeg = Math.max(15.0, Math.min(90.0, poseArmAngleDeg));
      const armRad = (clampedArmDeg * Math.PI) / 180.0;
      const dirX = signX * Math.sin(armRad);
      const dirY = -Math.cos(armRad);
      const perpX = signX * Math.cos(armRad);
      const perpY = Math.sin(armRad);

      const pivotX = signX * 0.21 * 1.425 * scaleX;
      const pivotY = 0.36 * 1.425 * scaleY;
      const dx = _limbSlabX - pivotX;
      const dy = _limbSlabY - pivotY;

      const dPar = dx * dirX + dy * dirY;
      const dPerp = dx * perpX + dy * perpY;

      const effPar = dPar * (1.0 + wArm * (userArmLen - 1.0));
      const effPerp = dPerp * (1.0 + wArm * (userArmThick - 1.0));

      _limbSlabX = pivotX + effPar * dirX + effPerp * perpX;
      _limbSlabY = pivotY + effPar * dirY + effPerp * perpY;
      _limbZMult *= Math.max(0.35, 1.0 + wArm * (userArmThick - 1.0) * 0.72);
    }
  }

  // 2. Lower-Body Leg Region Deformation (below hip line y < -0.06)
  if (
    y < -0.06 &&
    (Math.abs(userLegLen - 1.0) >= 0.005 || Math.abs(userLegThick - 1.0) >= 0.005)
  ) {
    const tyLeg = Math.max(0.0, Math.min(1.0, (-0.06 - y) / 0.18));
    const wLeg = tyLeg * tyLeg * (3.0 - 2.0 * tyLeg);
    if (wLeg > 0.001) {
      const hipPivotY = -0.08 * 1.425 * scaleY;
      const dyLeg = _limbSlabY - hipPivotY;
      _limbSlabY = hipPivotY + dyLeg * (1.0 + wLeg * (userLegLen - 1.0));

      // Scale horizontal leg thickness around each leg's anatomical center axis
      const legProgress = Math.max(0.0, Math.min(1.0, (-0.08 - y) / 0.82));
      const legAxisX = signX * (0.14 + 0.10 * legProgress) * 1.425 * scaleX;
      const dxFromLegAxis = _limbSlabX - legAxisX;
      _limbSlabX =
        legAxisX * (1.0 + wLeg * (userLegThick - 1.0) * 0.22) +
        dxFromLegAxis * (1.0 + wLeg * (userLegThick - 1.0));
      _limbZMult *= Math.max(0.35, 1.0 + wLeg * (userLegThick - 1.0) * 0.72);
    }
  }
}

/**
 * Performs 1 pass of 8-Neighbor Isotropic, Bilateral, or Anisotropic Straight-Line-Preserving Smoothing over posArr.
 * Supports `vertOffset` so both the Front Shell (vertOffset = 0) and Back Shell (vertOffset = gridVerts)
 * can be smoothed with identical rules!
 */
function applyMeshSmoothingPass(
  posArr: Float32Array,
  tempArr: Float32Array,
  segs: number,
  cols: number,
  weightLambda: number,
  isBilateral: boolean,
  wrapSeamX: boolean,
  wrapSeamY: boolean,
  vertexMask?: Uint8Array,
  vertOffset = 0,
  lineCoherence?: Float32Array,
  lineTanX?: Float32Array,
  lineTanY?: Float32Array,
  lineLockStrength = 0.0
): void {
  const vertexCount = cols * cols;
  const floatOffset = vertOffset * 3;
  tempArr.set(posArr.subarray(floatOffset, floatOffset + vertexCount * 3));

  const yStart = wrapSeamY ? 0 : 1;
  const yEnd = wrapSeamY ? segs - 1 : segs - 1;
  const xStart = wrapSeamX ? 0 : 1;
  const xEnd = wrapSeamX ? segs - 1 : segs - 1;

  // Inverse bilateral range variance (squared distance threshold)
  const invSigmaSq = 1.0 / (0.045 * 0.045);

  // 8-neighbor unit direction vectors in grid space (dx, dy)
  const dirX = [-1, 1, 0, 0, -0.7071, 0.7071, -0.7071, 0.7071];
  const dirY = [0, 0, -1, 1, -0.7071, -0.7071, 0.7071, 0.7071];

  for (let iy = yStart; iy <= yEnd; iy++) {
    const iyUp = iy === 0 ? (wrapSeamY ? segs - 1 : 0) : iy - 1;
    const iyDn = iy === segs ? (wrapSeamY ? 1 : segs) : iy + 1;

    const r0 = iy * cols;
    const rUp = iyUp * cols;
    const rDn = iyDn * cols;

    for (let ix = xStart; ix <= xEnd; ix++) {
      const centerIdx = r0 + ix;
      if (vertexMask && vertexMask[vertOffset + centerIdx] === 0) continue;

      const ixL = ix === 0 ? (wrapSeamX ? segs - 1 : 0) : ix - 1;
      const ixR = ix === segs ? (wrapSeamX ? 1 : segs) : ix + 1;

      const c3 = centerIdx * 3;
      const cx = tempArr[c3];
      const cy = tempArr[c3 + 1];
      const cz = tempArr[c3 + 2];

      const coh =
        lineCoherence && lineLockStrength > 0.01
          ? lineCoherence[centerIdx] * lineLockStrength
          : 0.0;
      const ltx = lineTanX ? lineTanX[centerIdx] : 1.0;
      const lty = lineTanY ? lineTanY[centerIdx] : 0.0;

      const idxNeighbors = [
        r0 + ixL,
        r0 + ixR,
        rUp + ix,
        rDn + ix,
        rUp + ixL,
        rUp + ixR,
        rDn + ixL,
        rDn + ixR,
      ];
      const baseW = [0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1];

      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      let sumW = 0;

      for (let k = 0; k < 8; k++) {
        const nVertIdx = idxNeighbors[k];
        if (vertexMask && vertexMask[vertOffset + nVertIdx] === 0) continue;

        const nk = nVertIdx * 3;
        let w = baseW[k];

        // Anisotropic Straight-Line Preservation:
        // On a detected straight line (coh > 0.1), weight neighbors collinear with the line tangent (ltx, lty)
        // and reject cross-line neighbors so straight lines stay razor-straight and never curve or round into mounds!
        if (coh > 0.1) {
          const align = Math.abs(dirX[k] * ltx + dirY[k] * lty); // 1.0 along straight line, 0.0 across line
          const anisoFactor = (1.0 - coh) + coh * (align * align * align);
          w *= anisoFactor;
        }

        if (isBilateral || coh > 0.15) {
          const dx = tempArr[nk] - cx;
          const dy = tempArr[nk + 1] - cy;
          const dz = tempArr[nk + 2] - cz;
          const dSq = dx * dx + dy * dy + dz * dz;
          w *= 1.0 / (1.0 + dSq * invSigmaSq * (1.0 + coh * 3.0));
        }

        sumX += tempArr[nk] * w;
        sumY += tempArr[nk + 1] * w;
        sumZ += tempArr[nk + 2] * w;
        sumW += w;
      }

      if (sumW > 1e-5) {
        const invW = 1.0 / sumW;
        // Lock X/Y lateral drift on straight lines so straight contours never bend
        const xyLambda = weightLambda * (1.0 - coh * 0.75);
        const zLambda = weightLambda * (1.0 - coh * 0.45);
        const out3 = floatOffset + c3;
        posArr[out3] = cx + xyLambda * (sumX * invW - cx);
        posArr[out3 + 1] = cy + xyLambda * (sumY * invW - cy);
        posArr[out3 + 2] = cz + zLambda * (sumZ * invW - cz);
      }
    }
  }

  // Synchronize closed seam column ix = segs with ix = 0
  if (wrapSeamX) {
    for (let iy = 0; iy <= segs; iy++) {
      const i0 = floatOffset + (iy * cols + 0) * 3;
      const iS = floatOffset + (iy * cols + segs) * 3;
      posArr[iS] = posArr[i0];
      posArr[iS + 1] = posArr[i0 + 1];
      posArr[iS + 2] = posArr[i0 + 2];
    }
  }
  if (wrapSeamY) {
    for (let ix = 0; ix <= segs; ix++) {
      const i0 = floatOffset + (0 * cols + ix) * 3;
      const iS = floatOffset + (segs * cols + ix) * 3;
      posArr[iS] = posArr[i0];
      posArr[iS + 1] = posArr[i0 + 1];
      posArr[iS + 2] = posArr[i0 + 2];
    }
  }
}

/**
 * Spatial Hash Close-Point Vertex Welding ("Unión de Punto Cercano" — Zero-GC, Zero Math.sqrt).
 * Finds vertices within `weldTolerance` using a 3D spatial hash table and squared distance `distSq <= tolSq`,
 * snaps them to their shared cluster position, and unifies their vertex normals so micro-gaps and seams fuse seamlessly.
 */
function weldCloseVerticesZeroGC(
  posArr: Float32Array,
  vertexCount: number,
  weldTolerance: number,
  buffers: CNNReconstructionBuffers
): number {
  if (weldTolerance <= 1e-5) {
    for (let i = 0; i < vertexCount; i++) {
      buffers.weldClusterRep[i] = i;
    }
    return 0;
  }

  const tolSq = weldTolerance * weldTolerance;
  const invCellSize = 1.0 / weldTolerance;
  const hashMask = CNNReconstructionBuffers.HASH_TABLE_SIZE - 1;
  const { weldHead, weldNext, weldClusterRep, vertexCutoutMask } = buffers;

  weldHead.fill(-1);
  let weldedCount = 0;

  for (let i = 0; i < vertexCount; i++) {
    if (vertexCutoutMask[i] === 0) {
      weldClusterRep[i] = i;
      continue;
    }

    const i3 = i * 3;
    const px = posArr[i3];
    const py = posArr[i3 + 1];
    const pz = posArr[i3 + 2];

    const cx = Math.floor(px * invCellSize);
    const cy = Math.floor(py * invCellSize);
    const cz = Math.floor(pz * invCellSize);

    let foundRep = -1;

    // Search 3x3x3 neighboring spatial hash cells for an existing representative within tolSq
    outerLoop: for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const h =
            (((cx + dx) * 73856093) ^
              ((cy + dy) * 19349663) ^
              ((cz + dz) * 83492791)) &
            hashMask;

          let candidate = weldHead[h];
          while (candidate !== -1) {
            const c3 = candidate * 3;
            const ddx = posArr[c3] - px;
            const ddy = posArr[c3 + 1] - py;
            const ddz = posArr[c3 + 2] - pz;
            const distSq = ddx * ddx + ddy * ddy + ddz * ddz;

            if (distSq <= tolSq) {
              foundRep = candidate;
              break outerLoop;
            }
            candidate = weldNext[candidate];
          }
        }
      }
    }

    if (foundRep !== -1) {
      weldClusterRep[i] = foundRep;
      const r3 = foundRep * 3;
      const mx = 0.5 * (posArr[r3] + px);
      const my = 0.5 * (posArr[r3 + 1] + py);
      const mz = 0.5 * (posArr[r3 + 2] + pz);
      posArr[r3] = mx;
      posArr[r3 + 1] = my;
      posArr[r3 + 2] = mz;
      posArr[i3] = mx;
      posArr[i3 + 1] = my;
      posArr[i3 + 2] = mz;
      weldedCount++;
    } else {
      weldClusterRep[i] = i;
      const h =
        ((cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791)) & hashMask;
      weldNext[i] = weldHead[h];
      weldHead[h] = i;
    }
  }

  if (weldedCount > 0) {
    for (let i = 0; i < vertexCount; i++) {
      const rep = weldClusterRep[i];
      if (rep !== i) {
        const i3 = i * 3;
        const r3 = rep * 3;
        posArr[i3] = posArr[r3];
        posArr[i3 + 1] = posArr[r3 + 1];
        posArr[i3 + 2] = posArr[r3 + 2];
      }
    }
  }

  return weldedCount;
}

// ============================================================================
// SILHOUETTE INFLATION (rounded volume from the drawing's own outline)
// Every part of the silhouette gets a circular cross-section whose radius is
// the local half-width of that part: arms/legs become tubes, the torso gets
// more depth, and the border meets the back shell with a vertical tangent
// instead of a thin flat flange.
// ============================================================================
const _inflDist = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);
const _inflRadius = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);
const _inflHeight = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);
const _inflTemp = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);

function buildSilhouetteInflation(mask: Uint8Array): void {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const INF = 1e6;
  const D = _inflDist;

  // 1) Distance to the outside (3-4 chamfer, scaled to texels).
  for (let i = 0; i < w * h; i++) D[i] = mask[i] === 1 ? INF : 0;
  for (let y = 0; y < h; y++) {
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
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      if (D[i] === 0) continue;
      let d = D[i];
      if (x === w - 1 || y === h - 1) d = Math.min(d, 1);
      if (x < w - 1) d = Math.min(d, D[i + 1] + 1);
      if (y < h - 1) {
        d = Math.min(d, D[i + w] + 1);
        if (x < w - 1) d = Math.min(d, D[i + w + 1] + 1.4142);
        if (x > 0) d = Math.min(d, D[i + w - 1] + 1.4142);
      }
      D[i] = d;
    }
  }
  // Border pixels sit half a texel inside the real outline.
  for (let i = 0; i < w * h; i++) if (D[i] > 0) D[i] -= 0.5;

  // 2) Local radius: every inside pixel stamps its inscribed disk with its
  //    own distance; each pixel keeps the largest disk that covers it.
  const R = _inflRadius;
  R.fill(0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const r = D[y * w + x];
      if (r < 1) continue;
      // Only near-medial pixels matter; skip pixels dominated by a neighbour.
      const i = y * w + x;
      if (
        (x > 0 && D[i - 1] > r + 0.5) ||
        (x < w - 1 && D[i + 1] > r + 0.5) ||
        (y > 0 && D[i - w] > r + 0.5) ||
        (y < h - 1 && D[i + w] > r + 0.5)
      ) {
        continue;
      }
      const ri = Math.ceil(r);
      const r2 = r * r;
      for (let dy = -ri; dy <= ri; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
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

  // 3) Circular cross-section: height = sqrt(2·R·d − d²).
  const H = _inflHeight;
  for (let i = 0; i < w * h; i++) {
    const d = D[i];
    if (d <= 0) {
      H[i] = 0;
      continue;
    }
    const r = Math.max(R[i], d);
    H[i] = Math.sqrt(Math.max(0, 2 * r * d - d * d));
  }

  // 4) Blur (masked, so the border stays at 0) to hide chamfer/disk quantisation.
  const k5 = [0.0625, 0.25, 0.375, 0.25, 0.0625];
  for (let pass = 0; pass < 3; pass++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let o = -2; o <= 2; o++) {
          const sx = x + o < 0 ? 0 : x + o >= w ? w - 1 : x + o;
          sum += H[y * w + sx] * k5[o + 2];
        }
        _inflTemp[y * w + x] = sum;
      }
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (mask[i] !== 1) {
          H[i] = 0;
          continue;
        }
        let sum = 0;
        for (let o = -2; o <= 2; o++) {
          const sy = y + o < 0 ? 0 : y + o >= h ? h - 1 : y + o;
          sum += _inflTemp[sy * w + x] * k5[o + 2];
        }
        H[i] = sum;
      }
    }
  }
}

/** Inflation height (in texels) at grid coordinates u, v in [0, 1]. */
function sampleInflationHeight(u: number, v: number): number {
  const w = CNN_TENSOR_SIZE;
  const x = Math.max(0, Math.min(w - 1.001, u * (w - 1)));
  const y = Math.max(0, Math.min(w - 1.001, (1 - v) * (w - 1)));
  const x0 = x | 0;
  const y0 = y | 0;
  const fx = x - x0;
  const fy = y - y0;
  const i = y0 * w + x0;
  const a = _inflHeight[i] * (1 - fx) + _inflHeight[i + 1] * fx;
  const b = _inflHeight[i + w] * (1 - fx) + _inflHeight[i + w + 1] * fx;
  return a * (1 - fy) + b * fy;
}

// ============================================================================
// SUB-PIXEL CONTOUR SNAPPING (removes the "saw-tooth" silhouette)
// The cutout keeps whole grid quads, so the raw border follows grid steps.
// Here every border vertex is slid onto the real 0.5 iso-line of a softly
// blurred cutout mask, which follows the drawing's outline smoothly.
// ============================================================================
const _contourField = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);
const _contourTemp = new Float32Array(CNN_TENSOR_SIZE * CNN_TENSOR_SIZE);
const _snapDu = new Float32Array(CNNReconstructionBuffers.MAX_VERTICES);
const _snapDv = new Float32Array(CNNReconstructionBuffers.MAX_VERTICES);
const _snapDx = new Float32Array(CNNReconstructionBuffers.MAX_VERTICES);
const _snapDy = new Float32Array(CNNReconstructionBuffers.MAX_VERTICES);
const _bNbrA = new Int32Array(CNNReconstructionBuffers.MAX_VERTICES);
const _bNbrB = new Int32Array(CNNReconstructionBuffers.MAX_VERTICES);
const _bNbrCount = new Uint8Array(CNNReconstructionBuffers.MAX_VERTICES);
const _bTmp = new Float32Array(CNNReconstructionBuffers.MAX_VERTICES * 4);
const _uvBefore = new Float32Array(CNNReconstructionBuffers.MAX_VERTICES * 2);

function buildSmoothContourField(mask: Uint8Array): void {
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const k = [0.0625, 0.25, 0.375, 0.25, 0.0625];
  for (let i = 0; i < w * h; i++) _contourField[i] = mask[i] === 1 ? 1.0 : 0.0;
  // Two separable 5-tap passes ≈ Gaussian with sigma ~1.4 texels: enough to
  // turn pixel stairs into a smooth ramp whose 0.5 level is the outline.
  for (let pass = 0; pass < 2; pass++) {
    for (let y = 0; y < h; y++) {
      const r = y * w;
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let o = -2; o <= 2; o++) {
          const sx = x + o < 0 ? 0 : x + o >= w ? w - 1 : x + o;
          sum += _contourField[r + sx] * k[o + 2];
        }
        _contourTemp[r + x] = sum;
      }
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let o = -2; o <= 2; o++) {
          const sy = y + o < 0 ? 0 : y + o >= h ? h - 1 : y + o;
          sum += _contourTemp[sy * w + x] * k[o + 2];
        }
        _contourField[y * w + x] = sum;
      }
    }
  }
}

function sampleContourField(tx: number, ty: number): number {
  const w = CNN_TENSOR_SIZE;
  const x = Math.max(0, Math.min(w - 1.001, tx));
  const y = Math.max(0, Math.min(w - 1.001, ty));
  const x0 = x | 0;
  const y0 = y | 0;
  const fx = x - x0;
  const fy = y - y0;
  const i = y0 * w + x0;
  const a = _contourField[i] * (1 - fx) + _contourField[i + 1] * fx;
  const b = _contourField[i + w] * (1 - fx) + _contourField[i + w + 1] * fx;
  return a * (1 - fy) + b * fy;
}

/**
 * Moves the front-shell border vertices onto the smooth outline.
 * Positions (x, y) are shifted in place; the matching UV shift is stored in
 * _snapDu/_snapDv so the back shell (built later) can apply it too.
 */
function snapFrontBorderToContour(
  posArr: Float32Array,
  uvArr: Float32Array,
  vertexCutoutMask: Uint8Array,
  borderDistField: Float32Array,
  texelCutoutMask: Uint8Array,
  quadActiveMask: Uint8Array,
  segs: number,
  strength: number
): void {
  const cols = segs + 1;
  const gridVerts = cols * cols;
  for (let i = 0; i < gridVerts * 2; i++) _uvBefore[i] = uvArr[i];
  _snapDu.fill(0, 0, gridVerts);
  _snapDv.fill(0, 0, gridVerts);
  _snapDx.fill(0, 0, gridVerts);
  _snapDy.fill(0, 0, gridVerts);
  if (strength <= 0.001) return;

  buildSmoothContourField(texelCutoutMask);

  const w = CNN_TENSOR_SIZE;
  const cellTex = (w - 1) / segs; // texels per grid cell
  const maxOut = 1.05 * cellTex; // never move further than ~one cell
  const maxIn = 0.45 * cellTex; // moving inwards is riskier (triangle flips)

  for (let iy = 1; iy < segs; iy++) {
    for (let ix = 1; ix < segs; ix++) {
      const vIdx = iy * cols + ix;
      if (vertexCutoutMask[vIdx] !== 1 || borderDistField[vIdx] > 0.0) continue;

      const tx0 = (ix / segs) * (w - 1);
      const ty0 = (1 - iy / segs) * (w - 1);
      let tx = tx0;
      let ty = ty0;

      // A few Newton steps along the field gradient towards the 0.5 level.
      for (let it = 0; it < 5; it++) {
        const f = sampleContourField(tx, ty) - 0.5;
        const gx = 0.5 * (sampleContourField(tx + 1, ty) - sampleContourField(tx - 1, ty));
        const gy = 0.5 * (sampleContourField(tx, ty + 1) - sampleContourField(tx, ty - 1));
        const g2 = gx * gx + gy * gy;
        if (g2 < 1e-6) break;
        tx -= (f * gx) / g2;
        ty -= (f * gy) / g2;
        if (Math.abs(f) < 0.01) break;
      }

      let dtx = tx - tx0;
      let dty = ty - ty0;
      const len = Math.hypot(dtx, dty);
      if (len < 1e-4) continue;
      // Outward = towards lower field values.
      const outward =
        sampleContourField(tx0 + dtx / len, ty0 + dty / len) < sampleContourField(tx0, ty0);
      const lim = outward ? maxOut : maxIn;
      if (len > lim) {
        dtx *= lim / len;
        dty *= lim / len;
      }
      dtx *= strength;
      dty *= strength;

      // Texel offset -> grid-cell offset (grid v grows upwards, texel y downwards).
      const di = dtx / cellTex;
      const dj = -dty / cellTex;

      // Local world-space axes of the grid at this vertex (handles mold warps).
      const pR = (vIdx + 1) * 3;
      const pL = (vIdx - 1) * 3;
      const pU = (vIdx + cols) * 3;
      const pD = (vIdx - cols) * 3;
      const axX = 0.5 * (posArr[pR] - posArr[pL]);
      const axY = 0.5 * (posArr[pR + 1] - posArr[pL + 1]);
      const ayX = 0.5 * (posArr[pU] - posArr[pD]);
      const ayY = 0.5 * (posArr[pU + 1] - posArr[pD + 1]);

      _snapDx[vIdx] = di * axX + dj * ayX;
      _snapDy[vIdx] = di * axY + dj * ayY;
      _snapDu[vIdx] = (di / segs) * 0.495;
      _snapDv[vIdx] = -(dj / segs) * 0.495;
    }
  }

  // Apply after all offsets are known so neighbours read unmodified positions.
  for (let vIdx = 0; vIdx < gridVerts; vIdx++) {
    if (_snapDx[vIdx] === 0 && _snapDy[vIdx] === 0) continue;
    posArr[vIdx * 3] += _snapDx[vIdx];
    posArr[vIdx * 3 + 1] += _snapDy[vIdx];
    uvArr[vIdx * 2] += _snapDu[vIdx];
    uvArr[vIdx * 2 + 1] += _snapDv[vIdx];
  }

  smoothBorderChain(posArr, uvArr, quadActiveMask, segs, strength);
  relaxNearBorderRing(posArr, uvArr, vertexCutoutMask, borderDistField, segs, strength);

  // Record the total UV shift (snap + chain smoothing) for the back shell.
  for (let vIdx = 0; vIdx < gridVerts; vIdx++) {
    _snapDu[vIdx] = uvArr[vIdx * 2] - _uvBefore[vIdx * 2];
    _snapDv[vIdx] = uvArr[vIdx * 2 + 1] - _uvBefore[vIdx * 2 + 1];
  }
}

/**
 * After the outline moved, the first rows inside it are unevenly spaced
 * (one row hugs the border, the next is a full cell away), which shows up as
 * horizontal ripples on the sides. A few Laplacian steps on those rows only
 * (the outline itself stays pinned) even the spacing out.
 */
function relaxNearBorderRing(
  posArr: Float32Array,
  uvArr: Float32Array,
  vertexCutoutMask: Uint8Array,
  borderDistField: Float32Array,
  segs: number,
  strength: number
): void {
  const cols = segs + 1;
  const band = 2.6;
  const iterations = Math.round(4 * strength);
  for (let it = 0; it < iterations; it++) {
    for (let iy = 1; iy < segs; iy++) {
      for (let ix = 1; ix < segs; ix++) {
        const v = iy * cols + ix;
        const d = borderDistField[v];
        if (vertexCutoutMask[v] !== 1 || d <= 0.0 || d > band) continue;
        const o = v * 4;
        let sx = 0, sy = 0, su = 0, sv = 0, n = 0;
        for (const nb of [v - 1, v + 1, v - cols, v + cols]) {
          if (vertexCutoutMask[nb] !== 1) continue;
          sx += posArr[nb * 3];
          sy += posArr[nb * 3 + 1];
          su += uvArr[nb * 2];
          sv += uvArr[nb * 2 + 1];
          n++;
        }
        if (n < 3) {
          _bTmp[o] = NaN;
          continue;
        }
        // Weaker pull deeper inside so the change fades out smoothly.
        const k = 0.5 * (1 - d / (band + 0.5));
        _bTmp[o] = posArr[v * 3] + k * (sx / n - posArr[v * 3]);
        _bTmp[o + 1] = posArr[v * 3 + 1] + k * (sy / n - posArr[v * 3 + 1]);
        _bTmp[o + 2] = uvArr[v * 2] + k * (su / n - uvArr[v * 2]);
        _bTmp[o + 3] = uvArr[v * 2 + 1] + k * (sv / n - uvArr[v * 2 + 1]);
      }
    }
    for (let iy = 1; iy < segs; iy++) {
      for (let ix = 1; ix < segs; ix++) {
        const v = iy * cols + ix;
        const d = borderDistField[v];
        if (vertexCutoutMask[v] !== 1 || d <= 0.0 || d > band) continue;
        const o = v * 4;
        if (Number.isNaN(_bTmp[o])) continue;
        posArr[v * 3] = _bTmp[o];
        posArr[v * 3 + 1] = _bTmp[o + 1];
        uvArr[v * 2] = _bTmp[o + 2];
        uvArr[v * 2 + 1] = _bTmp[o + 3];
      }
    }
  }
}

/**
 * Smooths the outline as a curve: each border vertex with exactly two border
 * neighbours is pulled towards their midpoint (Taubin λ/μ so the silhouette
 * does not shrink). Removes the remaining notches and single-vertex spikes.
 * UVs are smoothed with the same weights so the texture stays glued on.
 */
function smoothBorderChain(
  posArr: Float32Array,
  uvArr: Float32Array,
  quadActiveMask: Uint8Array,
  segs: number,
  strength: number
): void {
  const cols = segs + 1;
  const gridVerts = cols * cols;
  _bNbrCount.fill(0, 0, gridVerts);

  const link = (a: number, b: number) => {
    const ca = _bNbrCount[a];
    if (ca === 0) _bNbrA[a] = b;
    else if (ca === 1) _bNbrB[a] = b;
    _bNbrCount[a] = Math.min(255, ca + 1);
    const cb = _bNbrCount[b];
    if (cb === 0) _bNbrA[b] = a;
    else if (cb === 1) _bNbrB[b] = a;
    _bNbrCount[b] = Math.min(255, cb + 1);
  };

  // Same boundary-edge rule the lateral seam uses.
  for (let iy = 0; iy < segs; iy++) {
    for (let ix = 0; ix < segs; ix++) {
      if (quadActiveMask[iy * segs + ix] === 0) continue;
      const a = iy * cols + ix;
      const b = a + 1;
      const c = a + cols;
      const d = c + 1;
      if (iy === 0 || quadActiveMask[(iy - 1) * segs + ix] === 0) link(a, b);
      if (iy === segs - 1 || quadActiveMask[(iy + 1) * segs + ix] === 0) link(c, d);
      if (ix === 0 || quadActiveMask[iy * segs + ix - 1] === 0) link(a, c);
      if (ix === segs - 1 || quadActiveMask[iy * segs + ix + 1] === 0) link(b, d);
    }
  }

  const iterations = Math.round(6 * strength);
  const lambda = 0.5;
  const mu = -0.53;
  for (let it = 0; it < iterations * 2; it++) {
    const k = it % 2 === 0 ? lambda : mu;
    for (let v = 0; v < gridVerts; v++) {
      if (_bNbrCount[v] !== 2) continue;
      const a = _bNbrA[v];
      const b = _bNbrB[v];
      const o = v * 4;
      _bTmp[o] = posArr[v * 3] + k * (0.5 * (posArr[a * 3] + posArr[b * 3]) - posArr[v * 3]);
      _bTmp[o + 1] =
        posArr[v * 3 + 1] + k * (0.5 * (posArr[a * 3 + 1] + posArr[b * 3 + 1]) - posArr[v * 3 + 1]);
      _bTmp[o + 2] = uvArr[v * 2] + k * (0.5 * (uvArr[a * 2] + uvArr[b * 2]) - uvArr[v * 2]);
      _bTmp[o + 3] =
        uvArr[v * 2 + 1] + k * (0.5 * (uvArr[a * 2 + 1] + uvArr[b * 2 + 1]) - uvArr[v * 2 + 1]);
    }
    for (let v = 0; v < gridVerts; v++) {
      if (_bNbrCount[v] !== 2) continue;
      const o = v * 4;
      posArr[v * 3] = _bTmp[o];
      posArr[v * 3 + 1] = _bTmp[o + 1];
      uvArr[v * 2] = _bTmp[o + 2];
      uvArr[v * 2 + 1] = _bTmp[o + 3];
    }
  }
}

export function buildReconstructedMeshGeometry(
  geometry: THREE.BufferGeometry,
  buffers: CNNReconstructionBuffers,
  params: ReconstructionParams,
  zones: ZoneConfig[],
  albedoDataTex: THREE.DataTexture,
  roughMetalDataTex: THREE.DataTexture,
  normalDataTex: THREE.DataTexture
): {
  elapsedMs: number;
  poissonSolveMs: number;
  vertexCount: number;
  weldedVerticesCount: number;
  cutoutTrianglesRemoved: number;
  stitchedSideFacesCount: number;
  triangleCount: number;
  rig?: HumanRig;
} {
  const poissonSolveMs = solvePoissonSurfaceFromNormals(buffers, params);
  const t0 = performance.now();

  // Rounded volume derived from the silhouette (world units per texel ≈ slab width / texels).
  const inflationAmount = Math.max(0.0, Math.min(1.5, params.silhouetteInflation ?? 0.75));
  if (inflationAmount > 0.001) buildSilhouetteInflation(buffers.texelCutoutMask);
  const inflationWorldScale = (2.85 / (CNN_TENSOR_SIZE - 1)) * inflationAmount;
  // Inflation for a grid vertex; exactly 0 on the border so the outline stays thin
  // and the lateral seam closes it smoothly.
  // Photo relief fades out over the last cells before the outline, so the outline
  // keeps the clean rounded profile and front/back meet without a ridge.
  const reliefFadeCells =
    inflationAmount > 0.001 ? Math.max(2.5, Math.max(32, Math.min(144, params.gridResolution)) * 0.035) : 0.0;
  const reliefBorderFade = (gridIdx: number) => {
    if (reliefFadeCells <= 0.0) return 1.0;
    const t = Math.min(1.0, Math.max(0.0, buffers.borderDistField[gridIdx] / reliefFadeCells));
    return t * t * (3.0 - 2.0 * t);
  };
  const inflationAt = (gridIdx: number, gu: number, gv: number) =>
    buffers.borderDistField[gridIdx] <= 0.0 ? 0.0 : sampleInflationHeight(gu, gv);

  const segs = Math.max(32, Math.min(144, params.gridResolution));
  const cols = segs + 1;
  const rows = segs + 1;
  const gridVerts = cols * rows;
  const quadCount = segs * segs;

  // Always keep MonolithRelief 100% closed with Back Shell + 5-Ring Lateral Joint so the joint is ALWAYS covered ("siempre esté tapada")
  const isDualSolid = params.topology === MathTopology.MonolithRelief;

  // Convert user's lateral union width in centimeters (0.0 cm .. 6.0 cm) into 3D world units
  const sideCm = Math.max(0.0, Math.min(8.0, params.sideThicknessCm ?? 0.45));
  const lateralThicknessWorld = sideCm * 0.12; // 0.0 cm = 0.0 units (flush union with zero gap!)
  const halfGap = isDualSolid ? lateralThicknessWorld * 0.5 : 0.0;

  const zoneColors = zones.map((z) => new THREE.Color(z.colorHex));
  const w = CNN_TENSOR_SIZE;
  const h = CNN_TENSOR_SIZE;
  const alpha = params.cnnDepthWeight;
  const beta = params.mathModulationWeight;
  const degL = params.harmonicDegreeL;
  const ordM = params.harmonicOrderM;

  // Always stitch seams on 360° topologies as well so there is never an open seam
  const isClosedSeamX = params.topology !== MathTopology.MonolithRelief;
  const isClosedSeamY = params.topology === MathTopology.RiemannianTorus;
  const hasSphericalPoles =
    params.topology === MathTopology.SphericalHarmonics ||
    params.topology === MathTopology.ImplicitGyroid;

  const {
    vertexCutoutMask,
    texelCutoutMask,
    quadActiveMask,
    borderDistField,
    activeIndices,
  } = buffers;

  // Determine how many boundary edges exist when isDualSolid is active so we allocate exact BufferAttribute size
  for (let iy = 0; iy <= segs; iy++) {
    const v = iy / segs;
    const ty = Math.min(h - 1, Math.floor((1.0 - v) * (h - 1)));
    for (let ix = 0; ix <= segs; ix++) {
      const uSample = isClosedSeamX && ix === segs ? 0.0 : ix / segs;
      const tx = Math.min(w - 1, Math.floor(uSample * (w - 1)));
      const texIdx = ty * w + tx;
      vertexCutoutMask[iy * cols + ix] = texelCutoutMask[texIdx];
    }
  }

  let boundaryEdgeCount = 0;
  let activeFrontQuads = 0;

  for (let iy = 0; iy < segs; iy++) {
    const r0 = iy * segs;
    for (let ix = 0; ix < segs; ix++) {
      const a = iy * cols + ix;
      const b = a + 1;
      const c = a + cols;
      const d = c + 1;
      const isQuadActive =
        vertexCutoutMask[a] === 1 &&
        vertexCutoutMask[b] === 1 &&
        vertexCutoutMask[c] === 1 &&
        vertexCutoutMask[d] === 1;
      quadActiveMask[r0 + ix] = isQuadActive ? 1 : 0;
      if (isQuadActive) activeFrontQuads++;
    }
  }

  // Compute 2-Pass Chamfer Distance Transform (borderDistField) on vertex grid (0 .. gridVerts - 1)
  // Every vertex that touches an inactive quad or outer grid edge has borderDistField = 0.0
  for (let iy = 0; iy <= segs; iy++) {
    const r0 = iy * cols;
    for (let ix = 0; ix <= segs; ix++) {
      const vIdx = r0 + ix;
      if (
        ix === 0 ||
        ix === segs ||
        iy === 0 ||
        iy === segs ||
        vertexCutoutMask[vIdx] === 0
      ) {
        borderDistField[vIdx] = 0.0;
        continue;
      }
      // Check the 4 quads sharing vertex (ix, iy)
      const qTL = quadActiveMask[(iy - 1) * segs + (ix - 1)];
      const qTR = quadActiveMask[(iy - 1) * segs + ix];
      const qBL = quadActiveMask[iy * segs + (ix - 1)];
      const qBR = quadActiveMask[iy * segs + ix];
      if (qTL === 0 || qTR === 0 || qBL === 0 || qBR === 0) {
        borderDistField[vIdx] = 0.0;
      } else {
        borderDistField[vIdx] = 1000.0;
      }
    }
  }

  // Forward Chamfer Pass
  for (let iy = 1; iy < segs; iy++) {
    const r0 = iy * cols;
    const rUp = (iy - 1) * cols;
    for (let ix = 1; ix < segs; ix++) {
      const vIdx = r0 + ix;
      let d = borderDistField[vIdx];
      if (d > 0.0) {
        const dL = borderDistField[vIdx - 1] + 1.0;
        const dU = borderDistField[rUp + ix] + 1.0;
        const dUL = borderDistField[rUp + ix - 1] + 1.4142;
        const dUR = borderDistField[rUp + ix + 1] + 1.4142;
        if (dL < d) d = dL;
        if (dU < d) d = dU;
        if (dUL < d) d = dUL;
        if (dUR < d) d = dUR;
        borderDistField[vIdx] = d;
      }
    }
  }
  // Backward Chamfer Pass
  for (let iy = segs - 1; iy >= 1; iy--) {
    const r0 = iy * cols;
    const rDn = (iy + 1) * cols;
    for (let ix = segs - 1; ix >= 1; ix--) {
      const vIdx = r0 + ix;
      let d = borderDistField[vIdx];
      if (d > 0.0) {
        const dR = borderDistField[vIdx + 1] + 1.0;
        const dD = borderDistField[rDn + ix] + 1.0;
        const dDL = borderDistField[rDn + ix - 1] + 1.4142;
        const dDR = borderDistField[rDn + ix + 1] + 1.4142;
        if (dR < d) d = dR;
        if (dD < d) d = dD;
        if (dDL < d) d = dDL;
        if (dDR < d) d = dDR;
        borderDistField[vIdx] = d;
      }
    }
  }

  // Each boundary edge uses 5 cross-sectional rings (10 vertices, 8 triangles) to represent Flat, 90° Curve, or 90° Right-Angle Steps
  const maxAllowedBoundaryEdges = Math.min(
    CNNReconstructionBuffers.MAX_SEAM_EDGES,
    Math.floor((CNNReconstructionBuffers.MAX_VERTICES - gridVerts * 2) / 10)
  );

  if (isDualSolid) {
    for (let iy = 0; iy < segs; iy++) {
      for (let ix = 0; ix < segs; ix++) {
        if (quadActiveMask[iy * segs + ix] === 0) continue;
        // Top edge (iy == 0 or quad above inactive)
        if (iy === 0 || quadActiveMask[(iy - 1) * segs + ix] === 0) {
          if (boundaryEdgeCount < maxAllowedBoundaryEdges) boundaryEdgeCount++;
        }
        // Bottom edge (iy == segs - 1 or quad below inactive)
        if (iy === segs - 1 || quadActiveMask[(iy + 1) * segs + ix] === 0) {
          if (boundaryEdgeCount < maxAllowedBoundaryEdges) boundaryEdgeCount++;
        }
        // Left edge (ix == 0 or quad left inactive)
        if (ix === 0 || quadActiveMask[iy * segs + (ix - 1)] === 0) {
          if (boundaryEdgeCount < maxAllowedBoundaryEdges) boundaryEdgeCount++;
        }
        // Right edge (ix == 0 or quad right inactive)
        if (ix === segs - 1 || quadActiveMask[iy * segs + (ix + 1)] === 0) {
          if (boundaryEdgeCount < maxAllowedBoundaryEdges) boundaryEdgeCount++;
        }
      }
    }
  }

  const totalVertices = isDualSolid
    ? gridVerts * 2 + boundaryEdgeCount * 10
    : gridVerts;

  let posAttr = geometry.getAttribute('position') as
    | THREE.BufferAttribute
    | undefined;
  let normAttr = geometry.getAttribute('normal') as
    | THREE.BufferAttribute
    | undefined;
  let uvAttr = geometry.getAttribute('uv') as THREE.BufferAttribute | undefined;
  let colorAttr = geometry.getAttribute('color') as
    | THREE.BufferAttribute
    | undefined;

  if (!posAttr || posAttr.count !== totalVertices) {
    posAttr = new THREE.BufferAttribute(new Float32Array(totalVertices * 3), 3);
    normAttr = new THREE.BufferAttribute(new Float32Array(totalVertices * 3), 3);
    uvAttr = new THREE.BufferAttribute(new Float32Array(totalVertices * 2), 2);
    colorAttr = new THREE.BufferAttribute(new Float32Array(totalVertices * 3), 3);

    geometry.setAttribute('position', posAttr);
    geometry.setAttribute('normal', normAttr);
    geometry.setAttribute('uv', uvAttr);
    geometry.setAttribute('color', colorAttr);
  }

  const posArr = posAttr.array as Float32Array;
  const normArr = normAttr!.array as Float32Array;
  const uvArr = uvAttr!.array as Float32Array;
  const colArr = colorAttr!.array as Float32Array;

  const detectLines = params.detectStraightLines !== false;
  const lineSens = Math.max(0.0, Math.min(1.0, params.straightLineSensitivity ?? 0.78));
  const facetPlanarity = Math.max(0.0, Math.min(1.0, params.flatFacetPlanarity ?? 0.68));
  const useSymmetricBackRules = params.symmetricBackRules !== false;

  const presetShape = params.presetBaseShape || PresetBaseShape.None;
  const hasPresetShape = presetShape !== PresetBaseShape.None;
  const moldStrength = Math.max(0.0, Math.min(1.6, params.shapeMoldStrength ?? 0.75));
  const wrapUVWeight = Math.max(0.0, Math.min(1.0, params.shapeWrapUV ?? 0.45));
  const autoFitMold = params.autoFitMoldToSilhouette !== false;
  const silConformity = Math.max(0.0, Math.min(1.0, params.silhouetteConformity ?? 0.94));
  const userScaleX = Math.max(0.4, Math.min(1.8, params.shapeScaleX ?? 1.0));
  const userScaleY = Math.max(0.4, Math.min(1.8, params.shapeScaleY ?? 1.0));
  const userAspect = Math.max(0.5, Math.min(1.6, params.shapeAspectRatio ?? 1.0));
  const shapeOffX = Math.max(-0.35, Math.min(0.35, params.shapeOffsetX ?? 0.0));
  const shapeOffY = Math.max(-0.35, Math.min(0.35, params.shapeOffsetY ?? 0.0));

  // Effective mold dimensions combining Auto-Silhouette detection with explicit Width, Height & Proportion controls
  const autoScaleX = autoFitMold ? buffers.autoSilhouetteWidth / 0.78 : 1.0;
  const autoScaleY = autoFitMold ? buffers.autoSilhouetteHeight / 0.88 : 1.0;
  const shapeScaleX = Math.max(0.25, autoScaleX * userScaleX * userAspect);
  const shapeScaleY = Math.max(0.25, autoScaleY * userScaleY);
  const effMoldCenterX = (autoFitMold ? buffers.autoSilhouetteCenterX : 0.0) + shapeOffX;
  const effMoldCenterY = (autoFitMold ? buffers.autoSilhouetteCenterY : 0.0) + shapeOffY;

  const poseArmAngleDeg = Math.max(15.0, Math.min(90.0, params.poseArmAngleDeg ?? 35.0));
  const limbUnionMode = params.limbUnionMode || LimbUnionMode.TubularCylindersFull;
  const limbUnionCurvature = Math.max(0.05, Math.min(1.0, params.limbUnionCurvature ?? 0.68));
  const autoLimb = params.autoFitLimbSizes !== false;
  const userArmLen = Math.max(0.45, Math.min(1.75, params.armLengthScale ?? 1.0));
  const userArmThick = Math.max(0.45, Math.min(1.75, params.armThicknessScale ?? 1.0));
  const userLegLen = Math.max(0.45, Math.min(1.75, params.legLengthScale ?? 1.0));
  const userLegThick = Math.max(0.45, Math.min(1.75, params.legThicknessScale ?? 1.0));
  const effArmLen = Math.max(
    0.4,
    Math.min(1.85, userArmLen * (autoLimb ? buffers.autoDetectedArmLengthScale : 1.0))
  );
  const effArmThick = Math.max(
    0.4,
    Math.min(1.85, userArmThick * (autoLimb ? buffers.autoDetectedArmThicknessScale : 1.0))
  );
  const effLegLen = Math.max(
    0.4,
    Math.min(1.85, userLegLen * (autoLimb ? buffers.autoDetectedLegLengthScale : 1.0))
  );
  const effLegThick = Math.max(
    0.4,
    Math.min(1.85, userLegThick * (autoLimb ? buffers.autoDetectedLegThicknessScale : 1.0))
  );
  const reliefInflationSupp = hasPresetShape
    ? Math.max(0.0, Math.min(1.0, params.reliefInflationSuppression ?? 0.85))
    : Math.max(0.0, Math.min(1.0, (params.reliefInflationSuppression ?? 0.0) * 0.45));

  // Radius in grid cells over which the silhouette dome arch rises from 0 at the silhouette border to full 3D volume
  const silArchRadiusCells = Math.max(
    3.5,
    Math.min(22.0, segs * 0.15 * buffers.autoSilhouetteWidth)
  );

  const {
    vertexLineCoherence,
    vertexLineTanX,
    vertexLineTanY,
    lineCoherenceTensor,
    lineTangentX,
    lineTangentY,
    backLineCoherence,
    backLineTangentX,
    backLineTangentY,
    rowSilhouetteMinU,
    rowSilhouetteMaxU,
  } = buffers;

  // ============================================================================
  // STEP 1A: FRONT SHELL RECONSTRUCTION (Vertices 0 .. gridVerts - 1)
  // ============================================================================
  for (let iy = 0; iy <= segs; iy++) {
    const v = iy / segs;
    const tyRow = Math.min(h - 1, Math.max(0, Math.floor((1.0 - v) * (h - 1))));
    const rowCenterU = 0.5 * (rowSilhouetteMinU[tyRow] + rowSilhouetteMaxU[tyRow]) - 0.5;
    const rowAlignCenterX =
      effMoldCenterX * (1.0 - silConformity * 0.45) +
      (rowCenterU + shapeOffX) * (silConformity * 0.45);

    for (let ix = 0; ix <= segs; ix++) {
      const uSample = isClosedSeamX && ix === segs ? 0.0 : ix / segs;
      const u = ix / segs;
      const vIdx = iy * cols + ix;

      // Evaluate 3D Preset Base Shape aligned and scaled to the drawing's silhouette
      let sampleU = uSample;
      let sampleV = v;
      const uMold = 0.5 + (u - 0.5 - rowAlignCenterX);
      const vMold = 0.5 + (v - 0.5 - effMoldCenterY);
      if (hasPresetShape) {
        evaluatePresetBaseShapeZeroGC(
          presetShape,
          uMold,
          vMold,
          shapeScaleX,
          shapeScaleY,
          poseArmAngleDeg,
          limbUnionMode,
          limbUnionCurvature,
          effArmLen,
          effArmThick,
          effLegLen,
          effLegThick
        );
        if (_shapeInside === 1 && wrapUVWeight > 0.01) {
          const warpLocalU = _shapeWarpU + rowAlignCenterX;
          const warpLocalV = _shapeWarpV + effMoldCenterY;
          sampleU = Math.max(
            0.0,
            Math.min(1.0, uSample * (1.0 - wrapUVWeight) + warpLocalU * wrapUVWeight)
          );
          sampleV = Math.max(
            0.0,
            Math.min(1.0, v * (1.0 - wrapUVWeight) + warpLocalV * wrapUVWeight)
          );
        }
      }

      let ty = Math.min(h - 1, Math.max(0, Math.floor((1.0 - sampleV) * (h - 1))));
      let tx = Math.min(w - 1, Math.max(0, Math.floor(sampleU * (w - 1))));
      let texIdx = ty * w + tx;

      // Only pull UVs inward if StrictShapeSilhouette or UnionImageAndShape extends outside the drawing
      if (
        hasPresetShape &&
        _shapeInside === 1 &&
        buffers.exteriorBgMask[texIdx] === 1 &&
        params.shapeSilhouetteMode !== ShapeSilhouetteMode.AdaptiveImageToShape
      ) {
        sampleU = 0.5 + (sampleU - 0.5) * 0.54;
        sampleV = 0.5 + (sampleV - 0.5) * 0.54;
        ty = Math.min(h - 1, Math.max(0, Math.floor((1.0 - sampleV) * (h - 1))));
        tx = Math.min(w - 1, Math.max(0, Math.floor(sampleU * (w - 1))));
        texIdx = ty * w + tx;
      }

      const v3 = vIdx * 3;
      const v2 = vIdx * 2;

      const vCoh = detectLines ? lineCoherenceTensor[texIdx] : 0.0;
      vertexLineCoherence[vIdx] = vCoh;
      vertexLineTanX[vIdx] = lineTangentX[texIdx];
      // Invert Y tangent from image space (top-down) to 3D mesh UV space (bottom-up)
      vertexLineTanY[vIdx] = -lineTangentY[texIdx];

      // Map Front Shell UVs to Top-Left Quadrant [0.002..0.498] x [0.002..0.498] of 512x512 Multi-View Atlas
      uvArr[v2] = 0.002 + sampleU * 0.495;
      uvArr[v2 + 1] = 0.002 + (1.0 - sampleV) * 0.495;

      const overrideZone = buffers.customZoneOverrides[vIdx];
      const zoneId =
        overrideZone >= 0 && overrideZone < 4
          ? overrideZone
          : buffers.zoneMap[texIdx] & 3;
      const zone = zones[zoneId] || zones[0];

      const rawIntegratedDepth = buffers.depthTensor[texIdx];
      // Anti-Inflation Surface Relief Extraction:
      // Subtract the broad low-frequency pillow volume (lowFreqVolume) so photo relief carves/embosses crisp surface details
      // without bloating or inflating the human/animal body!
      const lowFreqPillow = buffers.lowFreqVolume[texIdx] * 0.78;
      const integratedDepth = Math.max(
        0.0,
        rawIntegratedDepth - lowFreqPillow * reliefInflationSupp
      );

      // Suppress sinusoidal microWave on straight lines & planar facets so they don't form ripples/mounds
      const microSuppression = detectLines
        ? Math.max(0.0, 1.0 - (vCoh * 0.85 + facetPlanarity * 0.65))
        : 1.0;
      const microWave =
        zone.microAmplitude *
        microSuppression *
        Math.sin(sampleU * Math.PI * 2 * zone.microFrequency) *
        Math.cos(sampleV * Math.PI * 2 * zone.microFrequency);

      const zoneExtrusionEffective = hasPresetShape
        ? zone.extrusion * (1.0 - reliefInflationSupp * 0.65)
        : zone.extrusion;
      const totalDisplacement =
        alpha * integratedDepth + zoneExtrusionEffective + microWave;

      let px = 0;
      let py = 0;
      let pz = 0;

      if (params.topology === MathTopology.MonolithRelief) {
        const rawSlabX = (u - 0.5) * 2.85 * zone.radialScale;
        const rawSlabY = (v - 0.5) * 2.85 * zone.radialScale;
        evaluateAnatomicalLimbSpatialWarpZeroGC(
          rawSlabX,
          rawSlabY,
          uMold,
          vMold,
          shapeScaleX,
          shapeScaleY,
          poseArmAngleDeg,
          userArmLen,
          userArmThick,
          userLegLen,
          userLegThick
        );
        const slabX = _limbSlabX;
        const slabY = _limbSlabY;

        const distFromCenter = Math.hypot(u - 0.5, v - 0.5) * 2.0;
        const rimBevel = detectLines
          ? Math.max(0.25, 1.0 - Math.max(0, distFromCenter - 0.88) * 3.5 * (1.0 - facetPlanarity * 0.65))
          : Math.max(
              0.08,
              1.0 - Math.pow(Math.max(0, distFromCenter - 0.82) * 4.5, 2)
            );

        // Suppress sinusoidal harmonicWave mounding when straight-line & planar facet detection is active!
        const harmonicSuppression = detectLines
          ? Math.max(0.0, 1.0 - facetPlanarity * 0.95 - vCoh * 0.9)
          : 1.0;
        const harmonicWave =
          beta *
          0.16 *
          harmonicSuppression *
          Math.cos((u - 0.5) * Math.PI * degL) *
          Math.cos((v - 0.5) * Math.PI * Math.max(1, ordM));

        const rawPedestal = detectLines ? 0.08 * (1.0 - facetPlanarity * 0.5) : 0.12;
        const basePedestal = rawPedestal * (1.0 - reliefInflationSupp * 0.88);
        const imageRelief =
          Math.max(0, (totalDisplacement * 0.85 + harmonicWave + basePedestal) * rimBevel) *
          reliefBorderFade(vIdx);

        // Silhouette Distance Dome: 0.0 right at the drawing's silhouette border (union zone), rising smoothly inside
        const dSil = Math.min(1.0, borderDistField[vIdx] / silArchRadiusCells);
        const silDomeArch = Math.sqrt(Math.sin(dSil * Math.PI * 0.5));

        if (hasPresetShape) {
          // Combine the auto-fitted 3D preset shape with the drawing's exact silhouette dome so:
          // 1) Any drawing detail inside the silhouette (arms, armor, cape, weapons) gets natural 3D volume.
          // 2) Right at the union zone (borderDistField === 0), the volume conforms 100% to the drawing's silhouette!
          const rawShapeZ = _shapeInside === 1 ? _shapeFrontZ : 0.0;
          const silFloorZ = silDomeArch * 0.28 * silConformity * _limbZMult;
          const blendedCarrierZ = smax3D(rawShapeZ * _limbZMult, silFloorZ, 0.065);
          const silEdgeEnvelope =
            (1.0 - silConformity * 0.86) + (silConformity * 0.86) * silDomeArch;
          const moldVolumeZ = blendedCarrierZ * silEdgeEnvelope * moldStrength;
          const shapeVolumeZ =
            inflationAmount > 0.001
              ? smax3D(moldVolumeZ, inflationAt(vIdx, u, v) * inflationWorldScale * _limbZMult, 0.05)
              : moldVolumeZ;

          const surfaceDetailScale = 1.0 - reliefInflationSupp * 0.42;
          // Lock XY displacement to 0 at the silhouette border so the union is 100% equal to the silhouette
          const xyDisplaceScale =
            0.12 * surfaceDetailScale * (1.0 - silConformity * 0.85) * silDomeArch;
          px = slabX + _shapeNx * imageRelief * xyDisplaceScale;
          py = slabY + _shapeNy * imageRelief * xyDisplaceScale;
          pz =
            halfGap +
            shapeVolumeZ +
            imageRelief * surfaceDetailScale * (0.45 + 0.55 * _shapeNz) * _limbZMult;
        } else {
          px = slabX;
          py = slabY;
          const inflZ =
            inflationAmount > 0.001 ? inflationAt(vIdx, u, v) * inflationWorldScale : 0.0;
          pz = halfGap + (inflZ + imageRelief) * _limbZMult;
        }
      } else if (params.topology === MathTopology.SphericalHarmonics) {
        const theta = (1.0 - v) * Math.PI;
        const phi = (uSample - 0.5) * Math.PI * 2.0;
        const ylm = evaluateSphericalHarmonic(theta, phi, degL, ordM);
        const radius =
          (1.18 + 0.52 * totalDisplacement + beta * 0.28 * ylm) *
          zone.radialScale;

        const sinT = Math.sin(theta);
        px = radius * sinT * Math.sin(phi);
        py = radius * Math.cos(theta);
        pz = radius * sinT * Math.cos(phi);
      } else if (params.topology === MathTopology.AxialRevolution) {
        const phi = (uSample - 0.5) * Math.PI * 2.0;
        const heightY = (v - 0.5) * 2.75;
        const baseProfile =
          0.82 +
          0.24 * Math.sin(v * Math.PI * degL) +
          beta * 0.16 * Math.cos(phi * ordM);
        const rho =
          Math.max(0.15, baseProfile + 0.48 * totalDisplacement) *
          zone.radialScale;

        px = rho * Math.sin(phi);
        py = heightY;
        pz = rho * Math.cos(phi);
      } else if (params.topology === MathTopology.ImplicitGyroid) {
        const theta = (1.0 - v) * Math.PI;
        const phi = (uSample - 0.5) * Math.PI * 2.0;
        const sinT = Math.sin(theta);
        const bx = sinT * Math.sin(phi);
        const by = Math.cos(theta);
        const bz = sinT * Math.cos(phi);

        const freq = (degL + 1.5) * 2.1;
        const gyroidField =
          Math.sin(bx * freq) * Math.cos(by * freq) +
          Math.sin(by * freq) * Math.cos(bz * freq) +
          Math.sin(bz * freq) * Math.cos(bx * freq);

        const radius =
          (1.2 +
            0.48 * totalDisplacement +
            (0.18 + beta * 0.26) * gyroidField) *
          zone.radialScale;

        px = bx * radius;
        py = by * radius;
        pz = bz * radius;
      } else {
        const uAngle = (uSample - 0.25) * Math.PI * 2.0;
        const vScroll = isClosedSeamY && iy === segs ? 0.0 : v;
        const vAngle = (vScroll - 0.5) * Math.PI * 2.0;
        const majorR = 1.22 * zone.radialScale;
        const torsion = beta * 0.22 * Math.sin(degL * uAngle + ordM * vAngle);
        const minorR = 0.48 + 0.36 * totalDisplacement + torsion;

        const ringR = majorR + minorR * Math.cos(vAngle);
        px = ringR * Math.cos(uAngle);
        py = minorR * Math.sin(vAngle);
        pz = ringR * Math.sin(uAngle);
      }

      px += buffers.sculptOffsets[v3];
      py += buffers.sculptOffsets[v3 + 1];
      pz += buffers.sculptOffsets[v3 + 2];

      posArr[v3] = px;
      posArr[v3 + 1] = py;
      posArr[v3 + 2] = pz;

      const zCol = zoneColors[zoneId] || zoneColors[0];
      const patMod = evaluateZonePattern(zone.pattern, u, v);
      // Sample frontInpaintedRgb so silhouette borders, joints, and bridged areas always take the clean image color!
      const imgR = buffers.frontInpaintedRgb[texIdx * 3];
      const imgG = buffers.frontInpaintedRgb[texIdx * 3 + 1];
      const imgB = buffers.frontInpaintedRgb[texIdx * 3 + 2];

      const mix = zone.zoneTintMix;
      colArr[v3] = Math.min(1, (imgR * (1.0 - mix) + zCol.r * mix) * patMod);
      colArr[v3 + 1] = Math.min(
        1,
        (imgG * (1.0 - mix) + zCol.g * mix) * patMod
      );
      colArr[v3 + 2] = Math.min(
        1,
        (imgB * (1.0 - mix) + zCol.b * mix) * patMod
      );
    }
  }

  const jointProfile =
    params.sideJointProfile || LateralJointProfile.SeamlessSmoothUnion;
  const seamlessSmooth = Math.max(
    0.0,
    Math.min(
      1.0,
      jointProfile === LateralJointProfile.SeamlessSmoothUnion
        ? Math.max(0.88, params.seamlessBorderSmoothing ?? 0.92)
        : params.seamlessBorderSmoothing ?? 0.85
    )
  );
  const seamlessBorderWidth = Math.max(
    2.0,
    Math.min(20.0, params.seamlessBorderWidth ?? 8.5)
  );
  const cornerSharpness = Math.max(
    0,
    Math.min(
      1,
      (params.sideCornerSharpness ??
        (jointProfile === LateralJointProfile.RightAngle90 ||
        jointProfile === LateralJointProfile.ChamferBevel ||
        jointProfile === LateralJointProfile.FlatStraight
          ? 0.92
          : 0.08)) *
        (1.0 - seamlessSmooth * 0.92)
    )
  );
  const seamClosure = Math.max(0, Math.min(1, params.seamEdgeClosure ?? 0.96));

  // Step 2: Apply Advanced Multi-Algorithm Mesh Smoothing to Front Shell (0 .. gridVerts - 1)
  // When detectLines is active, smoothing is anisotropic along straight lines so straight edges do not curve!
  const smoothSteps = Math.max(0, Math.min(30, params.laplacianSmoothingSteps));
  const lambda = Math.max(0.1, Math.min(0.92, params.smoothingLambda || 0.55));
  const algo =
    params.smoothingAlgorithm || SmoothingAlgorithm.TaubinVolumePreserving;
  const tempArr = buffers.laplacianTemp;
  const lineLock = detectLines ? lineSens : 0.0;

  for (let step = 0; step < smoothSteps; step++) {
    if (algo === SmoothingAlgorithm.TaubinVolumePreserving) {
      applyMeshSmoothingPass(
        posArr,
        tempArr,
        segs,
        cols,
        lambda,
        false,
        isClosedSeamX,
        isClosedSeamY,
        vertexCutoutMask,
        0,
        vertexLineCoherence,
        vertexLineTanX,
        vertexLineTanY,
        lineLock
      );
      // Reduce negative Taubin inflation (−μ) on flat facets when detectLines is enabled so flat surfaces never puff into mounds
      const inflationDamp = detectLines ? 1.0 - facetPlanarity * 0.55 : 1.0;
      const mu = (-lambda / (1.0 - 0.05 * lambda)) * inflationDamp;
      applyMeshSmoothingPass(
        posArr,
        tempArr,
        segs,
        cols,
        mu,
        false,
        isClosedSeamX,
        isClosedSeamY,
        vertexCutoutMask,
        0,
        vertexLineCoherence,
        vertexLineTanX,
        vertexLineTanY,
        lineLock
      );
    } else if (algo === SmoothingAlgorithm.BilateralEdgePreserving) {
      applyMeshSmoothingPass(
        posArr,
        tempArr,
        segs,
        cols,
        lambda,
        true,
        isClosedSeamX,
        isClosedSeamY,
        vertexCutoutMask,
        0,
        vertexLineCoherence,
        vertexLineTanX,
        vertexLineTanY,
        lineLock
      );
    } else {
      applyMeshSmoothingPass(
        posArr,
        tempArr,
        segs,
        cols,
        lambda,
        false,
        isClosedSeamX,
        isClosedSeamY,
        vertexCutoutMask,
        0,
        vertexLineCoherence,
        vertexLineTanX,
        vertexLineTanY,
        lineLock
      );
    }
  }

  // Check if full-thickness Tubular / Cylindrical Union mode is active
  const isTubularUnion =
    jointProfile === LateralJointProfile.TubularCylinderFull ||
    (hasPresetShape && limbUnionMode === LimbUnionMode.TubularCylindersFull);

  // Effective border smoothing band width in grid cells
  const baseTaperCells = isTubularUnion
    ? 6.0
    : detectLines
    ? Math.max(2.5, 6.5 * (1.0 - facetPlanarity * 0.55))
    : 6.5;
  const taperBandCells = Math.max(
    baseTaperCells,
    seamlessBorderWidth * (0.35 + 0.65 * seamlessSmooth)
  );

  if (
    params.topology === MathTopology.MonolithRelief &&
    (seamClosure > 0.001 || seamlessSmooth > 0.001)
  ) {
    for (let vIdx = 0; vIdx < gridVerts; vIdx++) {
      const dist = borderDistField[vIdx];
      if (dist < taperBandCells) {
        const t = Math.min(1.0, Math.max(0.0, dist / taperBandCells));
        const smoothDome = t * t * (3.0 - 2.0 * t);
        // Rounded elliptical arch that reaches 0.0 smoothly at dist = 0 so Front and Back meet without a step!
        const seamlessDome =
          Math.pow(Math.sin(t * Math.PI * 0.5), 0.68) *
          (0.35 + 0.65 * smoothDome);

        let stdRetention = 1.0;
        if (isTubularUnion) {
          const cylArc = Math.sqrt(Math.max(0.18, Math.sin(t * Math.PI * 0.5)));
          stdRetention = (1.0 - seamClosure * 0.72) + seamClosure * 0.72 * cylArc;
        } else {
          const rampProfile = detectLines
            ? smoothDome * (1.0 - facetPlanarity * 0.7) + t * (facetPlanarity * 0.7)
            : smoothDome;
          stdRetention = (1.0 - seamClosure) + seamClosure * rampProfile;
        }

        const edgeRetention =
          stdRetention * (1.0 - seamlessSmooth) + seamlessDome * seamlessSmooth;
        // Taper halfGap to 0 at the silhouette border when seamlessSmooth is high so both shells meet flush on the silhouette!
        const localHalfGap =
          halfGap * ((1.0 - seamlessSmooth * 0.96) + seamlessSmooth * 0.96 * smoothDome);

        const z3 = vIdx * 3 + 2;
        const relZ = Math.max(0.0, posArr[z3] - halfGap);
        // Keep the rounded silhouette volume: the taper may only remove what lies above it.
        const inflFloor =
          inflationAmount > 0.001
            ? Math.min(relZ, inflationAt(vIdx, (vIdx % cols) / segs, ((vIdx / cols) | 0) / segs) * inflationWorldScale)
            : 0.0;
        posArr[z3] = localHalfGap + Math.max(relZ * edgeRetention, inflFloor);
      }
    }
  }

  // Step 2A: slide border vertices onto the real outline (removes grid stair-steps)
  if (params.topology === MathTopology.MonolithRelief) {
    snapFrontBorderToContour(
      posArr,
      uvArr,
      vertexCutoutMask,
      borderDistField,
      texelCutoutMask,
      quadActiveMask,
      segs,
      Math.max(0.0, Math.min(1.0, params.contourSnapStrength ?? 1.0))
    );
  } else {
    _snapDu.fill(0, 0, gridVerts);
    _snapDv.fill(0, 0, gridVerts);
  }

  // ============================================================================
  // STEP 2B: BACK SHELL ("PARTE DE ATRÁS") & 5-RING MULTI-PROFILE LATERAL JOINT
  // Uses the EXACT SAME rules as the Front Shell (4-Zone extrusion, microWave,
  // pattern, Straight-Line & Anti-Mound detection, multi-pass smoothing, and
  // contour closure) symmetrically along -Z and taking into account its own depths!
  // ============================================================================
  let activeIndexPtr = 0;
  let stitchedSideFacesCount = 0;

  // Emit Front Shell active triangles
  for (let iy = 0; iy < segs; iy++) {
    const r0 = iy * segs;
    for (let ix = 0; ix < segs; ix++) {
      if (quadActiveMask[r0 + ix] === 0) continue;
      const a = iy * cols + ix;
      const b = a + 1;
      const c = a + cols;
      const d = c + 1;

      activeIndices[activeIndexPtr++] = a;
      activeIndices[activeIndexPtr++] = c;
      activeIndices[activeIndexPtr++] = b;

      activeIndices[activeIndexPtr++] = b;
      activeIndices[activeIndexPtr++] = c;
      activeIndices[activeIndexPtr++] = d;
    }
  }

  const { analyticalSideNormals } = buffers;

  if (isDualSolid) {
    const backWeight = Math.max(0, Math.min(1.8, params.backDepthWeight ?? 0.85));
    const effectiveBackAlpha = useSymmetricBackRules ? alpha * backWeight : backWeight;

    // Populate Back Shell vertices (gridVerts .. 2 * gridVerts - 1) using symmetric Front rules + Back Depths + 3D Preset Shape Back Volume
    for (let iy = 0; iy <= segs; iy++) {
      const v = iy / segs;
      const r0 = iy * cols;
      const tyRow = Math.min(h - 1, Math.max(0, Math.floor((1.0 - v) * (h - 1))));
      const rowCenterU = 0.5 * (rowSilhouetteMinU[tyRow] + rowSilhouetteMaxU[tyRow]) - 0.5;
      const rowAlignCenterX =
        effMoldCenterX * (1.0 - silConformity * 0.45) +
        (rowCenterU + shapeOffX) * (silConformity * 0.45);

      for (let ix = 0; ix <= segs; ix++) {
        const u = ix / segs;
        const frontVIdx = r0 + ix;

        let sampleU = u;
        let sampleV = v;
        let backShapeZ = 0.0;
        let backShapeNz = 1.0;
        const uMold = 0.5 + (u - 0.5 - rowAlignCenterX);
        const vMold = 0.5 + (v - 0.5 - effMoldCenterY);
        evaluateAnatomicalLimbSpatialWarpZeroGC(
          (u - 0.5) * 2.85,
          (v - 0.5) * 2.85,
          uMold,
          vMold,
          shapeScaleX,
          shapeScaleY,
          poseArmAngleDeg,
          userArmLen,
          userArmThick,
          userLegLen,
          userLegThick
        );
        if (hasPresetShape) {
          evaluatePresetBaseShapeZeroGC(
            presetShape,
            uMold,
            vMold,
            shapeScaleX,
            shapeScaleY,
            poseArmAngleDeg,
            limbUnionMode,
            limbUnionCurvature,
            effArmLen,
            effArmThick,
            effLegLen,
            effLegThick
          );
          const dSil = Math.min(1.0, borderDistField[frontVIdx] / silArchRadiusCells);
          const silDomeArch = Math.sqrt(Math.sin(dSil * Math.PI * 0.5));
          const rawBackShapeZ = _shapeInside === 1 ? _shapeBackZ : 0.0;
          const silFloorZ = silDomeArch * 0.28 * silConformity * _limbZMult;
          const blendedBackCarrierZ = smax3D(rawBackShapeZ * _limbZMult, silFloorZ, 0.065);
          const silEdgeEnvelope =
            (1.0 - silConformity * 0.86) + (silConformity * 0.86) * silDomeArch;
          backShapeZ = blendedBackCarrierZ * silEdgeEnvelope * moldStrength;
          if (inflationAmount > 0.001) {
            backShapeZ = smax3D(
              backShapeZ,
              inflationAt(frontVIdx, u, v) * inflationWorldScale * _limbZMult,
              0.05
            );
          }
        } else if (inflationAmount > 0.001) {
          backShapeZ = inflationAt(frontVIdx, u, v) * inflationWorldScale * _limbZMult;

          if (_shapeInside === 1) {
            backShapeNz = _shapeNz;
            if (wrapUVWeight > 0.01) {
              const warpLocalU = _shapeWarpU + rowAlignCenterX;
              const warpLocalV = _shapeWarpV + effMoldCenterY;
              sampleU = Math.max(
                0.0,
                Math.min(1.0, u * (1.0 - wrapUVWeight) + warpLocalU * wrapUVWeight)
              );
              sampleV = Math.max(
                0.0,
                Math.min(1.0, v * (1.0 - wrapUVWeight) + warpLocalV * wrapUVWeight)
              );
            }
          }
        }

        let ty = Math.min(h - 1, Math.max(0, Math.floor((1.0 - sampleV) * (h - 1))));
        let tx = Math.min(w - 1, Math.max(0, Math.floor(sampleU * (w - 1))));
        let texIdx = ty * w + tx;

        if (
          hasPresetShape &&
          _shapeInside === 1 &&
          buffers.exteriorBgMask[texIdx] === 1 &&
          params.shapeSilhouetteMode !== ShapeSilhouetteMode.AdaptiveImageToShape
        ) {
          sampleU = 0.5 + (sampleU - 0.5) * 0.54;
          sampleV = 0.5 + (sampleV - 0.5) * 0.54;
          ty = Math.min(h - 1, Math.max(0, Math.floor((1.0 - sampleV) * (h - 1))));
          tx = Math.min(w - 1, Math.max(0, Math.floor(sampleU * (w - 1))));
          texIdx = ty * w + tx;
        }

        const backVIdx = gridVerts + frontVIdx;
        const f3 = frontVIdx * 3;
        const b3 = backVIdx * 3;
        const b2 = backVIdx * 2;

        vertexCutoutMask[backVIdx] = vertexCutoutMask[frontVIdx];

        // Store Back Shell straight-line coherence & tangents for Back Shell smoothing
        const bCoh = detectLines ? backLineCoherence[texIdx] : 0.0;
        vertexLineCoherence[frontVIdx] = bCoh;
        vertexLineTanX[frontVIdx] = backLineTangentX[texIdx];
        vertexLineTanY[frontVIdx] = -backLineTangentY[texIdx];

        // Map Back Shell UVs to Top-Right Quadrant [0.502..0.998] x [0.002..0.498] of 512x512 Atlas
        uvArr[b2] = 0.502 + sampleU * 0.495;
        uvArr[b2 + 1] = 0.002 + (1.0 - sampleV) * 0.495;

        const overrideZone = buffers.customZoneOverrides[frontVIdx];
        const backZoneId =
          overrideZone >= 0 && overrideZone < 4
            ? overrideZone
            : buffers.backZoneMap[texIdx] & 3;
        const backZone = zones[backZoneId] || zones[0];

        const rawBackDepth = buffers.backDepthTensor[texIdx];
        const backLowFreqPillow = buffers.backLowFreqVolume[texIdx] * 0.78;
        const backDepth = Math.max(
          0.0,
          rawBackDepth - backLowFreqPillow * reliefInflationSupp
        );

        const distFromCenter = Math.hypot(u - 0.5, v - 0.5) * 2.0;
        const rimBevel = detectLines
          ? Math.max(0.25, 1.0 - Math.max(0, distFromCenter - 0.88) * 3.5 * (1.0 - facetPlanarity * 0.65))
          : Math.max(
              0.08,
              1.0 - Math.pow(Math.max(0, distFromCenter - 0.82) * 4.5, 2)
            );

        let localBackRelief = 0;
        if (useSymmetricBackRules) {
          // Exact same formula as the Front Shell, but using backDepth, backZone, and backLineCoherence
          const microSuppression = detectLines
            ? Math.max(0.0, 1.0 - (bCoh * 0.85 + facetPlanarity * 0.65))
            : 1.0;
          const backMicroWave =
            backZone.microAmplitude *
            microSuppression *
            Math.sin(sampleU * Math.PI * 2 * backZone.microFrequency) *
            Math.cos(sampleV * Math.PI * 2 * backZone.microFrequency);

          const backZoneExtrusion = hasPresetShape
            ? backZone.extrusion * (1.0 - reliefInflationSupp * 0.65)
            : backZone.extrusion;
          const backTotalDisplacement =
            effectiveBackAlpha * backDepth + backZoneExtrusion * backWeight + backMicroWave;

          const harmonicSuppression = detectLines
            ? Math.max(0.0, 1.0 - facetPlanarity * 0.95 - bCoh * 0.9)
            : 1.0;
          const backHarmonicWave =
            beta *
            0.16 *
            harmonicSuppression *
            Math.cos((u - 0.5) * Math.PI * degL) *
            Math.cos((v - 0.5) * Math.PI * Math.max(1, ordM));

          const rawPedestal = detectLines ? 0.08 * (1.0 - facetPlanarity * 0.5) : 0.12;
          const basePedestal = rawPedestal * (1.0 - reliefInflationSupp * 0.88);
          localBackRelief = Math.max(
            0,
            (backTotalDisplacement * 0.85 + backHarmonicWave + basePedestal * backWeight) * rimBevel
          );
        } else {
          localBackRelief = Math.max(
            0,
            (backDepth * backWeight * 0.85 + 0.06 * (1.0 - reliefInflationSupp * 0.85)) * rimBevel
          );
        }

        localBackRelief *= reliefBorderFade(frontVIdx);

        const surfaceDetailScale = hasPresetShape ? 1.0 - reliefInflationSupp * 0.42 : 1.0;
        // Maintain 100% of the 3D Mold dorsal volume on the Back Shell (never squashed when backWeight < 1),
        // while scaling the Back Image's surface relief by backWeight!
        const moldCarrierBackScale = Math.max(1.0, backWeight);
        const totalBackZ =
          backShapeZ * moldCarrierBackScale +
          localBackRelief * surfaceDetailScale * (0.45 + 0.55 * backShapeNz) * _limbZMult;

        // Apply symmetric sculpt offsets: X/Y match Front silhouette, while Z sculpt is mirrored symmetrically (-Z)
        posArr[b3] = posArr[f3];
        posArr[b3 + 1] = posArr[f3 + 1];
        posArr[b3 + 2] = -halfGap - totalBackZ - buffers.sculptOffsets[f3 + 2] * backWeight;

        const zCol = zoneColors[backZoneId] || zoneColors[0];
        const patMod = useSymmetricBackRules ? evaluateZonePattern(backZone.pattern, sampleU, sampleV) : 1.0;
        const bRgb3 = texIdx * 3;
        const bImgR = buffers.backInpaintedRgb[bRgb3];
        const bImgG = buffers.backInpaintedRgb[bRgb3 + 1];
        const bImgB = buffers.backInpaintedRgb[bRgb3 + 2];
        const mix = useSymmetricBackRules ? backZone.zoneTintMix : 0.0;

        colArr[b3] = Math.min(1, (bImgR * (1.0 - mix) + zCol.r * mix) * patMod);
        colArr[b3 + 1] = Math.min(1, (bImgG * (1.0 - mix) + zCol.g * mix) * patMod);
        colArr[b3 + 2] = Math.min(1, (bImgB * (1.0 - mix) + zCol.b * mix) * patMod);
      }
    }

    // Apply the Exact Same Multi-Algorithm Mesh Smoothing to the Back Shell (vertOffset = gridVerts)
    if (smoothSteps > 0) {
      for (let step = 0; step < smoothSteps; step++) {
        if (algo === SmoothingAlgorithm.TaubinVolumePreserving) {
          applyMeshSmoothingPass(
            posArr,
            tempArr,
            segs,
            cols,
            lambda,
            false,
            isClosedSeamX,
            isClosedSeamY,
            vertexCutoutMask,
            gridVerts,
            vertexLineCoherence,
            vertexLineTanX,
            vertexLineTanY,
            lineLock
          );
          const inflationDamp = detectLines ? 1.0 - facetPlanarity * 0.55 : 1.0;
          const mu = (-lambda / (1.0 - 0.05 * lambda)) * inflationDamp;
          applyMeshSmoothingPass(
            posArr,
            tempArr,
            segs,
            cols,
            mu,
            false,
            isClosedSeamX,
            isClosedSeamY,
            vertexCutoutMask,
            gridVerts,
            vertexLineCoherence,
            vertexLineTanX,
            vertexLineTanY,
            lineLock
          );
        } else if (algo === SmoothingAlgorithm.BilateralEdgePreserving) {
          applyMeshSmoothingPass(
            posArr,
            tempArr,
            segs,
            cols,
            lambda,
            true,
            isClosedSeamX,
            isClosedSeamY,
            vertexCutoutMask,
            gridVerts,
            vertexLineCoherence,
            vertexLineTanX,
            vertexLineTanY,
            lineLock
          );
        } else {
          applyMeshSmoothingPass(
            posArr,
            tempArr,
            segs,
            cols,
            lambda,
            false,
            isClosedSeamX,
            isClosedSeamY,
            vertexCutoutMask,
            gridVerts,
            vertexLineCoherence,
            vertexLineTanX,
            vertexLineTanY,
            lineLock
          );
        }
      }
    }

    // Lock Back Shell X/Y to Front Shell X/Y and apply Cross-Part Seamless Border Smoothing & Contour Closure
    for (let frontVIdx = 0; frontVIdx < gridVerts; frontVIdx++) {
      const backVIdx = gridVerts + frontVIdx;
      const f3 = frontVIdx * 3;
      const b3 = backVIdx * 3;
      posArr[b3] = posArr[f3];
      posArr[b3 + 1] = posArr[f3 + 1];

      if (seamClosure > 0.001 || seamlessSmooth > 0.001) {
        const dist = borderDistField[frontVIdx];
        if (dist < taperBandCells) {
          const t = Math.min(1.0, Math.max(0.0, dist / taperBandCells));
          const smoothDome = t * t * (3.0 - 2.0 * t);
          const seamlessDome =
            Math.pow(Math.sin(t * Math.PI * 0.5), 0.68) *
            (0.35 + 0.65 * smoothDome);

          let stdRetention = 1.0;
          if (isTubularUnion) {
            const cylArc = Math.sqrt(Math.max(0.18, Math.sin(t * Math.PI * 0.5)));
            stdRetention = (1.0 - seamClosure * 0.72) + seamClosure * 0.72 * cylArc;
          } else {
            const rampProfile = detectLines
              ? smoothDome * (1.0 - facetPlanarity * 0.7) + t * (facetPlanarity * 0.7)
              : smoothDome;
            stdRetention = (1.0 - seamClosure) + seamClosure * rampProfile;
          }

          const edgeRetention =
            stdRetention * (1.0 - seamlessSmooth) + seamlessDome * seamlessSmooth;
          const localHalfGap =
            halfGap * ((1.0 - seamlessSmooth * 0.96) + seamlessSmooth * 0.96 * smoothDome);

          const relBackZ = Math.max(0.0, -halfGap - posArr[b3 + 2]);
          const backInflFloor =
            inflationAmount > 0.001
              ? Math.min(
                  relBackZ,
                  inflationAt(frontVIdx, (frontVIdx % cols) / segs, ((frontVIdx / cols) | 0) / segs) *
                    inflationWorldScale
                )
              : 0.0;
          let newBackZ = localHalfGap + Math.max(relBackZ * edgeRetention, backInflFloor);
          let newFrontZ = Math.max(0.0, posArr[f3 + 2]);

          // Harmonize Front & Back border curvature and color near the union line so both parts merge seamlessly!
          if (seamlessSmooth > 0.02 && t < 0.75) {
            const symBlend =
              seamlessSmooth * Math.pow(1.0 - t / 0.75, 1.5) * 0.48;
            const avgSpan = 0.5 * (newFrontZ + newBackZ);
            newFrontZ = newFrontZ * (1.0 - symBlend) + avgSpan * symBlend;
            newBackZ = newBackZ * (1.0 - symBlend) + avgSpan * symBlend;
            posArr[f3 + 2] = newFrontZ;

            const colBlend =
              seamlessSmooth * Math.pow(1.0 - t / 0.75, 1.8) * 0.45;
            const mR = 0.5 * (colArr[f3] + colArr[b3]);
            const mG = 0.5 * (colArr[f3 + 1] + colArr[b3 + 1]);
            const mB = 0.5 * (colArr[f3 + 2] + colArr[b3 + 2]);
            colArr[f3] = colArr[f3] * (1.0 - colBlend) + mR * colBlend;
            colArr[f3 + 1] = colArr[f3 + 1] * (1.0 - colBlend) + mG * colBlend;
            colArr[f3 + 2] = colArr[f3 + 2] * (1.0 - colBlend) + mB * colBlend;
            colArr[b3] = colArr[b3] * (1.0 - colBlend) + mR * colBlend;
            colArr[b3 + 1] = colArr[b3 + 1] * (1.0 - colBlend) + mG * colBlend;
            colArr[b3 + 2] = colArr[b3 + 2] * (1.0 - colBlend) + mB * colBlend;
          }

          posArr[b3 + 2] = -newBackZ;
        }
      }
    }

    // Emit Back Shell active triangles (reversed winding so normals point toward -Z)
    for (let iy = 0; iy < segs; iy++) {
      const r0 = iy * segs;
      for (let ix = 0; ix < segs; ix++) {
        if (quadActiveMask[r0 + ix] === 0) continue;
        const a = gridVerts + iy * cols + ix;
        const b = a + 1;
        const c = a + cols;
        const d = c + 1;

        activeIndices[activeIndexPtr++] = a;
        activeIndices[activeIndexPtr++] = b;
        activeIndices[activeIndexPtr++] = c;

        activeIndices[activeIndexPtr++] = b;
        activeIndices[activeIndexPtr++] = d;
        activeIndices[activeIndexPtr++] = c;
      }
    }

    // Precompute the 5-Ring Cross-Sectional Profile (r = 0, 1, 2, 3, 4) based on LateralJointProfile and sideJointAngleDeg (0°..90°)
    const jointAngleDeg = Math.max(
      0,
      Math.min(90, params.sideJointAngleDeg ?? 90)
    );
    const angleFactor = jointAngleDeg / 90.0; // 0.0 = 0° (Plana), 1.0 = 90° (Curva 90° o Ángulo Recto 90°)
    const bevelRaw = Math.max(0, Math.min(1.2, params.sideSeamBevel ?? 0.55));
    // Suppress outward XY protrusion at the union zone when seamlessSmooth or silConformity is active so the union equals the silhouette!
    const silhouetteLockDamp =
      (1.0 - seamlessSmooth * 0.96) * (1.0 - silConformity * 0.88);
    const bevelScale =
      bevelRaw *
      Math.max(0.02, Math.min(0.18, lateralThicknessWorld * 0.45 + 0.025)) *
      silhouetteLockDamp;
    const maxProfileExt = bevelScale * angleFactor;
    const sideReliefAmp =
      Math.max(0, Math.min(0.6, params.sideReliefAmplitude ?? 0.12)) *
      0.08 *
      silhouetteLockDamp;
    const isAutoEdge =
      (params.sideSourceMode || SideImageSourceMode.AutoEdgeExtension) ===
      SideImageSourceMode.AutoEdgeExtension;

    // Cross-sectional arrays for the 5 rings:
    const ringZFrac = [0.0, 0.25, 0.5, 0.75, 1.0];
    const ringOutBase = [0.0, 0.0, 0.0, 0.0, 0.0];
    const ringTubeSin = [0.0, 0.7071, 1.0, 0.7071, 0.0];
    const ringRelMask = [0.0, 0.75, 1.0, 0.75, 0.0];
    const ringNormOut = [0.0, 1.0, 1.0, 1.0, 0.0];
    const ringNormZ = [1.0, 0.0, 0.0, 0.0, -1.0];

    if (
      jointProfile === LateralJointProfile.SeamlessSmoothUnion ||
      isTubularUnion
    ) {
      // Smooth rounded 180° transition between Front and Back locked to the silhouette contour:
      ringZFrac[0] = 0.0;
      ringZFrac[1] = 0.1464;
      ringZFrac[2] = 0.5;
      ringZFrac[3] = 0.8536;
      ringZFrac[4] = 1.0;

      ringNormOut[0] = 0.7071;
      ringNormZ[0] = 0.7071;
      ringNormOut[1] = 0.9239;
      ringNormZ[1] = 0.3827;
      ringNormOut[2] = 1.0;
      ringNormZ[2] = 0.0;
      ringNormOut[3] = 0.9239;
      ringNormZ[3] = -0.3827;
      ringNormOut[4] = 0.7071;
      ringNormZ[4] = -0.7071;
    } else if (jointProfile === LateralJointProfile.FlatStraight) {
      const flatExt = maxProfileExt * 0.2;
      ringZFrac[0] = 0.0;
      ringZFrac[1] = 0.25;
      ringZFrac[2] = 0.5;
      ringZFrac[3] = 0.75;
      ringZFrac[4] = 1.0;

      ringOutBase[0] = 0.0;
      ringOutBase[1] = flatExt * 0.5;
      ringOutBase[2] = flatExt;
      ringOutBase[3] = flatExt * 0.5;
      ringOutBase[4] = 0.0;

      for (let r = 0; r < 5; r++) {
        ringNormOut[r] = 1.0;
        ringNormZ[r] = 0.0;
      }
    } else if (jointProfile === LateralJointProfile.RightAngle90) {
      const stepOut = Math.max(0.0, maxProfileExt);
      const orthoSlope = 0.18 * (1.0 - angleFactor);
      ringZFrac[0] = 0.0;
      ringZFrac[1] = orthoSlope;
      ringZFrac[2] = 0.5;
      ringZFrac[3] = 1.0 - orthoSlope;
      ringZFrac[4] = 1.0;

      ringOutBase[0] = 0.0;
      ringOutBase[1] = stepOut;
      ringOutBase[2] = stepOut;
      ringOutBase[3] = stepOut;
      ringOutBase[4] = 0.0;

      ringRelMask[0] = 0.0;
      ringRelMask[1] = 1.0;
      ringRelMask[2] = 1.0;
      ringRelMask[3] = 1.0;
      ringRelMask[4] = 0.0;

      ringNormOut[0] = stepOut > 0.005 ? 0.35 : 1.0;
      ringNormZ[0] = stepOut > 0.005 ? 0.93 : 0.0;
      ringNormOut[1] = 1.0;
      ringNormZ[1] = 0.0;
      ringNormOut[2] = 1.0;
      ringNormZ[2] = 0.0;
      ringNormOut[3] = 1.0;
      ringNormZ[3] = 0.0;
      ringNormOut[4] = stepOut > 0.005 ? 0.35 : 1.0;
      ringNormZ[4] = stepOut > 0.005 ? -0.93 : 0.0;
    } else if (jointProfile === LateralJointProfile.ChamferBevel) {
      const chamferZ = Math.max(0.08, 0.34 * angleFactor);
      const stepOut = Math.max(0.005, maxProfileExt);
      ringZFrac[0] = 0.0;
      ringZFrac[1] = chamferZ;
      ringZFrac[2] = 0.5;
      ringZFrac[3] = 1.0 - chamferZ;
      ringZFrac[4] = 1.0;

      ringOutBase[0] = 0.0;
      ringOutBase[1] = stepOut;
      ringOutBase[2] = stepOut;
      ringOutBase[3] = stepOut;
      ringOutBase[4] = 0.0;

      ringNormOut[0] = 0.707;
      ringNormZ[0] = 0.707;
      ringNormOut[1] = 0.92;
      ringNormZ[1] = 0.38;
      ringNormOut[2] = 1.0;
      ringNormZ[2] = 0.0;
      ringNormOut[3] = 0.92;
      ringNormZ[3] = -0.38;
      ringNormOut[4] = 0.707;
      ringNormZ[4] = -0.707;
    } else if (jointProfile === LateralJointProfile.ConcaveCove90) {
      const coveDepth = Math.max(0.01, maxProfileExt * 0.95);
      const thetaMid = Math.PI * 0.25 * angleFactor;
      const thetaMax = Math.PI * 0.5 * angleFactor;
      ringZFrac[0] = 0.0;
      ringZFrac[1] = 0.22;
      ringZFrac[2] = 0.5;
      ringZFrac[3] = 0.78;
      ringZFrac[4] = 1.0;

      ringOutBase[0] = 0.0;
      ringOutBase[1] = -coveDepth * Math.sin(thetaMid);
      ringOutBase[2] = -coveDepth * Math.sin(thetaMax);
      ringOutBase[3] = -coveDepth * Math.sin(thetaMid);
      ringOutBase[4] = 0.0;

      ringNormOut[0] = 0.707;
      ringNormZ[0] = -0.707;
      ringNormOut[1] = 0.86;
      ringNormZ[1] = -0.5;
      ringNormOut[2] = 1.0;
      ringNormZ[2] = 0.0;
      ringNormOut[3] = 0.86;
      ringNormZ[3] = 0.5;
      ringNormOut[4] = 0.707;
      ringNormZ[4] = 0.707;
    } else {
      const theta1 = Math.PI * 0.25 * angleFactor;
      const theta2 = Math.PI * 0.5 * angleFactor;
      ringZFrac[0] = 0.0;
      ringZFrac[1] = 0.22;
      ringZFrac[2] = 0.5;
      ringZFrac[3] = 0.78;
      ringZFrac[4] = 1.0;

      ringOutBase[0] = 0.0;
      ringOutBase[1] = maxProfileExt * Math.sin(theta1);
      ringOutBase[2] = maxProfileExt * Math.sin(theta2);
      ringOutBase[3] = maxProfileExt * Math.sin(theta1);
      ringOutBase[4] = 0.0;

      ringNormOut[0] = Math.cos(theta2 * 0.85);
      ringNormZ[0] = Math.sin(theta2 * 0.85);
      ringNormOut[1] = Math.cos(theta1);
      ringNormZ[1] = Math.sin(theta1);
      ringNormOut[2] = 1.0;
      ringNormZ[2] = 0.0;
      ringNormOut[3] = Math.cos(theta1);
      ringNormZ[3] = -Math.sin(theta1);
      ringNormOut[4] = Math.cos(theta2 * 0.85);
      ringNormZ[4] = -Math.sin(theta2 * 0.85);
    }

    // Back shell shares the front outline, so give its UVs the same contour shift.
    for (let vIdx = 0; vIdx < gridVerts; vIdx++) {
      const du = _snapDu[vIdx];
      const dv = _snapDv[vIdx];
      if (du === 0 && dv === 0) continue;
      const b2 = (gridVerts + vIdx) * 2;
      uvArr[b2] += du;
      uvArr[b2 + 1] += dv;
    }

    const {
      vertexOutNx,
      vertexOutNy,
      seamEdgeVStart,
      seamEdgeVEnd,
      seamEdgeBaseV,
    } = buffers;

    // Precompute shared per-vertex 2D outward contour normals (vertexOutNx, vertexOutNy)
    // so adjacent boundary edges meeting at 90° grid corners share the EXACT same outward displacement vector
    // and never split open at corners!
    vertexOutNx.fill(0);
    vertexOutNy.fill(0);

    const accumulateBoundaryNormal = (vStart: number, vEnd: number) => {
      const f0 = vStart * 3;
      const f1 = vEnd * 3;
      const edx = posArr[f1] - posArr[f0];
      const edy = posArr[f1 + 1] - posArr[f0 + 1];
      const eLen = Math.max(1e-6, Math.hypot(edx, edy));
      const enx = edy / eLen;
      const eny = -edx / eLen;
      vertexOutNx[vStart] += enx;
      vertexOutNy[vStart] += eny;
      vertexOutNx[vEnd] += enx;
      vertexOutNy[vEnd] += eny;
    };

    for (let iy = 0; iy < segs; iy++) {
      for (let ix = 0; ix < segs; ix++) {
        if (quadActiveMask[iy * segs + ix] === 0) continue;
        const a = iy * cols + ix;
        const b = a + 1;
        const c = a + cols;
        const d = c + 1;
        if (iy === 0 || quadActiveMask[(iy - 1) * segs + ix] === 0) {
          accumulateBoundaryNormal(b, a);
        }
        if (iy === segs - 1 || quadActiveMask[(iy + 1) * segs + ix] === 0) {
          accumulateBoundaryNormal(c, d);
        }
        if (ix === 0 || quadActiveMask[iy * segs + (ix - 1)] === 0) {
          accumulateBoundaryNormal(a, c);
        }
        if (ix === segs - 1 || quadActiveMask[iy * segs + (ix + 1)] === 0) {
          accumulateBoundaryNormal(d, b);
        }
      }
    }

    for (let v = 0; v < gridVerts; v++) {
      const nx = vertexOutNx[v];
      const ny = vertexOutNy[v];
      const lenSq = nx * nx + ny * ny;
      if (lenSq > 1e-8) {
        const invL = 1.0 / Math.sqrt(lenSq);
        vertexOutNx[v] = nx * invL;
        vertexOutNy[v] = ny * invL;
      }
    }

    let sideVertCursor = gridVerts * 2;
    let builtEdges = 0;

    const appendLateralSeamEdge = (vStart: number, vEnd: number) => {
      if (builtEdges >= boundaryEdgeCount) return;
      const edgeIdx = builtEdges;
      builtEdges++;

      const f0 = vStart * 3;
      const f1 = vEnd * 3;
      const b0 = (gridVerts + vStart) * 3;
      const b1 = (gridVerts + vEnd) * 3;

      const x0 = posArr[f0];
      const y0 = posArr[f0 + 1];
      const zf0 = posArr[f0 + 2];
      const zb0 = posArr[b0 + 2];

      const x1 = posArr[f1];
      const y1 = posArr[f1 + 1];
      const zf1 = posArr[f1 + 2];
      const zb1 = posArr[b1 + 2];

      // Local total thickness of the object/human at boundary vertices vStart and vEnd,
      // damped by silhouetteLockDamp so the union never protrudes outside the drawing's silhouette!
      const localSpanZ0 = Math.max(0.005, zf0 - zb0);
      const localSpanZ1 = Math.max(0.005, zf1 - zb1);
      const tubeRadius0 =
        localSpanZ0 * (0.28 + bevelRaw * 0.34) * angleFactor * silhouetteLockDamp;
      const tubeRadius1 =
        localSpanZ1 * (0.28 + bevelRaw * 0.34) * angleFactor * silhouetteLockDamp;

      // Shared per-vertex outward 2D normals so adjacent boundary edges meet at identical 3D points at every corner!
      let outNx0 = vertexOutNx[vStart];
      let outNy0 = vertexOutNy[vStart];
      let outNx1 = vertexOutNx[vEnd];
      let outNy1 = vertexOutNy[vEnd];
      if (outNx0 === 0 && outNy0 === 0) {
        const edx = x1 - x0;
        const edy = y1 - y0;
        const eLen = Math.max(1e-6, Math.hypot(edx, edy));
        outNx0 = edy / eLen;
        outNy0 = -edx / eLen;
      }
      if (outNx1 === 0 && outNy1 === 0) {
        outNx1 = outNx0;
        outNy1 = outNy0;
      }

      // Perimeter UV coordinate u0, u1 in [0, 1] from polar angle around model center (used only for custom uploaded side textures)
      const ang0 = (Math.atan2(y0, x0) + Math.PI) / (Math.PI * 2.0);
      let ang1 = (Math.atan2(y1, x1) + Math.PI) / (Math.PI * 2.0);
      if (Math.abs(ang1 - ang0) > 0.5) {
        ang1 = ang1 < ang0 ? ang1 + 1.0 : ang1 - 1.0;
      }
      const u0 = Math.max(0, Math.min(1, ang0));
      const u1 = Math.max(0, Math.min(1, ang1));

      const stx0 = Math.min(w - 1, Math.floor(u0 * (w - 1)));
      const stx1 = Math.min(w - 1, Math.floor(u1 * (w - 1)));

      const baseV = sideVertCursor;
      sideVertCursor += 10;

      seamEdgeVStart[edgeIdx] = vStart;
      seamEdgeVEnd[edgeIdx] = vEnd;
      seamEdgeBaseV[edgeIdx] = baseV;

      // Dedicated 4x2 texel cell in the Bottom Half of the 512x512 Atlas (y: 256..511) for this boundary edge!
      // Eliminates any UV jump across the atlas so the joint ALWAYS takes the exact local color of the image!
      const cellX = (edgeIdx & 127) << 2;
      const cellY = 256 + ((edgeIdx >> 7) << 1);
      const patchU0 = (cellX + 1.0) / ATLAS_TEX_SIZE;
      const patchU1 = (cellX + 3.0) / ATLAS_TEX_SIZE;

      const uAtlas0 = 0.002 + u0 * 0.996;
      const uAtlas1 = 0.002 + u1 * 0.996;
      const useCustomSideAtlas =
        params.sideSourceMode === SideImageSourceMode.CustomSideUpload &&
        buffers.hasCustomSideImage;

      // Emit 5 rings (10 vertices: baseV + r*2 and baseV + r*2 + 1)
      for (let r = 0; r < 5; r++) {
        const v0Idx = baseV + r * 2;
        const v1Idx = v0Idx + 1;
        vertexCutoutMask[v0Idx] = 1;
        vertexCutoutMask[v1Idx] = 1;

        const zFrac = ringZFrac[r];
        const sty = Math.min(h - 1, Math.floor((r / 4.0) * (h - 1)));
        const rowOff = sty * w;

        const sRel0 =
          r === 0 || r === 4 || isAutoEdge
            ? 0.0
            : (buffers.sideDepthTensor[rowOff + stx0] - 0.35) *
              sideReliefAmp *
              ringRelMask[r];
        const sRel1 =
          r === 0 || r === 4 || isAutoEdge
            ? 0.0
            : (buffers.sideDepthTensor[rowOff + stx1] - 0.35) *
              sideReliefAmp *
              ringRelMask[r];

        const baseOut0 =
          r === 0 || r === 4
            ? 0.0
            : isTubularUnion
            ? tubeRadius0 * ringTubeSin[r]
            : ringOutBase[r];
        const baseOut1 =
          r === 0 || r === 4
            ? 0.0
            : isTubularUnion
            ? tubeRadius1 * ringTubeSin[r]
            : ringOutBase[r];

        const out0 = baseOut0 + sRel0;
        const out1 = baseOut1 + sRel1;

        const p0 = v0Idx * 3;
        const p1 = v1Idx * 3;

        posArr[p0] = x0 + outNx0 * out0;
        posArr[p0 + 1] = y0 + outNy0 * out0;
        posArr[p0 + 2] = r === 0 ? zf0 : r === 4 ? zb0 : zf0 * (1.0 - zFrac) + zb0 * zFrac;

        posArr[p1] = x1 + outNx1 * out1;
        posArr[p1 + 1] = y1 + outNy1 * out1;
        posArr[p1 + 2] = r === 0 ? zf1 : r === 4 ? zb1 : zf1 * (1.0 - zFrac) + zb1 * zFrac;

        // Smooth analytical normal for this ring using shared per-vertex outward normal
        const nOut = ringNormOut[r];
        const nZ = ringNormZ[r];
        const nInv = 1.0 / Math.max(1e-6, Math.hypot(nOut, nZ));

        analyticalSideNormals[p0] = outNx0 * nOut * nInv;
        analyticalSideNormals[p0 + 1] = outNy0 * nOut * nInv;
        analyticalSideNormals[p0 + 2] = nZ * nInv;
        analyticalSideNormals[p1] = outNx1 * nOut * nInv;
        analyticalSideNormals[p1 + 1] = outNy1 * nOut * nInv;
        analyticalSideNormals[p1 + 2] = nZ * nInv;

        if (!useCustomSideAtlas) {
          // Continuous UVs inside this boundary edge's dedicated 4x2 image-color patch in the Bottom Half of the 512x512 Atlas!
          const patchV = (cellY + 0.5 + zFrac) / ATLAS_TEX_SIZE;
          uvArr[v0Idx * 2] = patchU0;
          uvArr[v0Idx * 2 + 1] = patchV;
          uvArr[v1Idx * 2] = patchU1;
          uvArr[v1Idx * 2 + 1] = patchV;

          // Pure un-darkened interpolation of the Front and Back Image Colors at vStart and vEnd
          colArr[p0] = colArr[f0] * (1.0 - zFrac) + colArr[b0] * zFrac;
          colArr[p0 + 1] = colArr[f0 + 1] * (1.0 - zFrac) + colArr[b0 + 1] * zFrac;
          colArr[p0 + 2] = colArr[f0 + 2] * (1.0 - zFrac) + colArr[b0 + 2] * zFrac;

          colArr[p1] = colArr[f1] * (1.0 - zFrac) + colArr[b1] * zFrac;
          colArr[p1 + 1] = colArr[f1 + 1] * (1.0 - zFrac) + colArr[b1 + 1] * zFrac;
          colArr[p1 + 2] = colArr[f1 + 2] * (1.0 - zFrac) + colArr[b1 + 2] * zFrac;
        } else {
          // Custom uploaded side texture uses the Bottom Strip
          const vAtlas = 0.504 + (r / 4.0) * 0.492;
          uvArr[v0Idx * 2] = uAtlas0;
          uvArr[v0Idx * 2 + 1] = vAtlas;
          uvArr[v1Idx * 2] = uAtlas1;
          uvArr[v1Idx * 2 + 1] = vAtlas;

          const c0 = (rowOff + stx0) * 3;
          const c1 = (rowOff + stx1) * 3;
          colArr[p0] = buffers.sideRgbActive[c0];
          colArr[p0 + 1] = buffers.sideRgbActive[c0 + 1];
          colArr[p0 + 2] = buffers.sideRgbActive[c0 + 2];
          colArr[p1] = buffers.sideRgbActive[c1];
          colArr[p1 + 1] = buffers.sideRgbActive[c1 + 1];
          colArr[p1 + 2] = buffers.sideRgbActive[c1 + 2];
        }
      }

      // Connect the 5 rings with 4 quad strips (8 triangles)
      for (let r = 0; r < 4; r++) {
        const rA = baseV + r * 2;
        const rB = rA + 1;
        const rC = rA + 2;
        const rD = rA + 3;

        activeIndices[activeIndexPtr++] = rA;
        activeIndices[activeIndexPtr++] = rC;
        activeIndices[activeIndexPtr++] = rB;

        activeIndices[activeIndexPtr++] = rB;
        activeIndices[activeIndexPtr++] = rC;
        activeIndices[activeIndexPtr++] = rD;
      }

      stitchedSideFacesCount += 8;
    };

    for (let iy = 0; iy < segs; iy++) {
      for (let ix = 0; ix < segs; ix++) {
        if (quadActiveMask[iy * segs + ix] === 0) continue;
        const a = iy * cols + ix;
        const b = a + 1;
        const c = a + cols;
        const d = c + 1;

        // Top boundary edge: b -> a
        if (iy === 0 || quadActiveMask[(iy - 1) * segs + ix] === 0) {
          appendLateralSeamEdge(b, a);
        }
        // Bottom boundary edge: c -> d
        if (iy === segs - 1 || quadActiveMask[(iy + 1) * segs + ix] === 0) {
          appendLateralSeamEdge(c, d);
        }
        // Left boundary edge: a -> c
        if (ix === 0 || quadActiveMask[iy * segs + (ix - 1)] === 0) {
          appendLateralSeamEdge(a, c);
        }
        // Right boundary edge: d -> b
        if (ix === segs - 1 || quadActiveMask[iy * segs + (ix + 1)] === 0) {
          appendLateralSeamEdge(d, b);
        }
      }
    }
  }

  // Safety fallback if user sets an extreme threshold that removes 100% of triangles
  if (activeIndexPtr === 0) {
    const mid = Math.floor(segs / 2);
    const a = mid * cols + mid;
    activeIndices[0] = a;
    activeIndices[1] = a + cols;
    activeIndices[2] = a + 1;
    activeIndexPtr = 3;
  }

  const activeTriangleCount = activeIndexPtr / 3;
  const cutoutTrianglesRemoved = Math.max(
    0,
    (quadCount - activeFrontQuads) * 2
  );

  geometry.setIndex(
    new THREE.BufferAttribute(activeIndices.slice(0, activeIndexPtr), 1)
  );

  // Step 3: Stitch Spherical Poles if applicable
  if (hasSphericalPoles) {
    for (const poleRow of [0, segs]) {
      let avgX = 0;
      let avgY = 0;
      let avgZ = 0;
      for (let ix = 0; ix <= segs; ix++) {
        const p3 = (poleRow * cols + ix) * 3;
        avgX += posArr[p3];
        avgY += posArr[p3 + 1];
        avgZ += posArr[p3 + 2];
      }
      avgX /= cols;
      avgY /= cols;
      avgZ /= cols;
      for (let ix = 0; ix <= segs; ix++) {
        const p3 = (poleRow * cols + ix) * 3;
        posArr[p3] = avgX;
        posArr[p3 + 1] = avgY;
        posArr[p3 + 2] = avgZ;
      }
    }
  }

  // Step 4: Close-Point Spatial Vertex Welding ("Unión de Punto Cercano" + Auto-Soldadura de Unión Invisible)
  const effectiveWeldTolerance = Math.max(
    params.weldTolerance || 0,
    isDualSolid ? 0.006 * seamlessSmooth : 0
  );
  const weldedVerticesCount = weldCloseVerticesZeroGC(
    posArr,
    totalVertices,
    effectiveWeldTolerance,
    buffers
  );

  // Step 4B: Watertight Seam Ring Lock (Zero-GC)
  // Explicitly locks Ring 0 (r=0) of every lateral seam edge to the post-weld Front Shell boundary vertices (vStart, vEnd)
  // and Ring 4 (r=4) to the post-weld Back Shell boundary vertices (gridVerts + vStart, gridVerts + vEnd)
  // so it is mathematically impossible for any gap or crack to ever open along the joint!
  if (isDualSolid) {
    const { seamEdgeVStart, seamEdgeVEnd, seamEdgeBaseV } = buffers;
    const edgeCount = Math.min(
      boundaryEdgeCount,
      CNNReconstructionBuffers.MAX_SEAM_EDGES
    );
    for (let e = 0; e < edgeCount; e++) {
      const vStart = seamEdgeVStart[e];
      const vEnd = seamEdgeVEnd[e];
      const baseV = seamEdgeBaseV[e];
      if (baseV <= 0) continue;

      const f0 = vStart * 3;
      const f1 = vEnd * 3;
      const b0 = (gridVerts + vStart) * 3;
      const b1 = (gridVerts + vEnd) * 3;

      const r0_0 = baseV * 3;
      const r0_1 = (baseV + 1) * 3;
      posArr[r0_0] = posArr[f0];
      posArr[r0_0 + 1] = posArr[f0 + 1];
      posArr[r0_0 + 2] = posArr[f0 + 2];

      posArr[r0_1] = posArr[f1];
      posArr[r0_1 + 1] = posArr[f1 + 1];
      posArr[r0_1 + 2] = posArr[f1 + 2];

      const r4_0 = (baseV + 8) * 3;
      const r4_1 = (baseV + 9) * 3;
      posArr[r4_0] = posArr[b0];
      posArr[r4_0 + 1] = posArr[b0 + 1];
      posArr[r4_0 + 2] = posArr[b0 + 2];

      posArr[r4_1] = posArr[b1];
      posArr[r4_1 + 1] = posArr[b1 + 1];
      posArr[r4_1 + 2] = posArr[b1 + 2];
    }
  }

  // Step 5: Compute Vertex Normals + Unify Normals across Welded Points & Seams
  geometry.computeVertexNormals();

  if (isClosedSeamX) {
    for (let iy = 0; iy <= segs; iy++) {
      const i0 = (iy * cols + 0) * 3;
      const iS = (iy * cols + segs) * 3;
      const nx = normArr[i0] + normArr[iS];
      const ny = normArr[i0 + 1] + normArr[iS + 1];
      const nz = normArr[i0 + 2] + normArr[iS + 2];
      const invL = 1.0 / Math.max(1e-6, Math.sqrt(nx * nx + ny * ny + nz * nz));
      normArr[i0] = normArr[iS] = nx * invL;
      normArr[i0 + 1] = normArr[iS + 1] = ny * invL;
      normArr[i0 + 2] = normArr[iS + 2] = nz * invL;
    }
  }

  // When cornerSharpness < 0.55 (e.g. Seamless Union or Curved Fillet), unify normals across welded Front-Lateral-Back joints
  if (weldedVerticesCount > 0 && cornerSharpness < 0.55) {
    const { weldClusterRep } = buffers;
    for (let i = 0; i < totalVertices; i++) {
      const rep = weldClusterRep[i];
      if (rep !== i) {
        const i3 = i * 3;
        const r3 = rep * 3;
        normArr[r3] += normArr[i3];
        normArr[r3 + 1] += normArr[i3 + 1];
        normArr[r3 + 2] += normArr[i3 + 2];
      }
    }
    for (let i = 0; i < totalVertices; i++) {
      const rep = weldClusterRep[i];
      const i3 = i * 3;
      if (rep === i) {
        const nx = normArr[i3];
        const ny = normArr[i3 + 1];
        const nz = normArr[i3 + 2];
        const invL =
          1.0 / Math.max(1e-6, Math.sqrt(nx * nx + ny * ny + nz * nz));
        normArr[i3] = nx * invL;
        normArr[i3 + 1] = ny * invL;
        normArr[i3 + 2] = nz * invL;
      } else {
        const r3 = rep * 3;
        normArr[i3] = normArr[r3];
        normArr[i3 + 1] = normArr[r3 + 1];
        normArr[i3 + 2] = normArr[r3 + 2];
      }
    }
  }

  // Cross-Shell Seamless Border Normal Continuity Pass ("Desaparecer la Unión con Suavizado entre Ambas Partes"):
  // Smoothly rotates Front and Back normals into the 2D outward silhouette normal at the contour boundary (dist -> 0)
  // so diffuse and specular lighting wrap around the edge without any visible seam line!
  if (isDualSolid && seamlessSmooth > 0.01) {
    for (let iy = 0; iy <= segs; iy++) {
      const r0 = iy * cols;
      const iyDown = Math.max(0, iy - 1) * cols;
      const iyUp = Math.min(segs, iy + 1) * cols;
      for (let ix = 0; ix <= segs; ix++) {
        const frontVIdx = r0 + ix;
        if (vertexCutoutMask[frontVIdx] === 0) continue;
        const dist = borderDistField[frontVIdx];
        if (dist >= taperBandCells) continue;

        const t = Math.min(1.0, Math.max(0.0, dist / taperBandCells));
        const wSeam = seamlessSmooth * Math.pow(1.0 - t, 1.28) * 0.94;

        // Outward 2D silhouette gradient from borderDistField
        const dL = borderDistField[r0 + Math.max(0, ix - 1)];
        const dR = borderDistField[r0 + Math.min(segs, ix + 1)];
        const dD = borderDistField[iyDown + ix];
        const dU = borderDistField[iyUp + ix];
        let gx = dL - dR;
        let gy = dD - dU;
        const gLenSq = gx * gx + gy * gy;

        const f3 = frontVIdx * 3;
        const b3 = (gridVerts + frontVIdx) * 3;

        if (gLenSq > 1e-5) {
          const invG = 1.0 / Math.sqrt(gLenSq);
          gx *= invG;
          gy *= invG;
        } else {
          const px = posArr[f3];
          const py = posArr[f3 + 1];
          const pLen = Math.max(1e-5, Math.hypot(px, py));
          gx = px / pLen;
          gy = py / pLen;
        }

        const theta = t * Math.PI * 0.5;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);

        const targetNx = gx * cosT;
        const targetNy = gy * cosT;
        const targetFrontNz = sinT;
        const targetBackNz = -sinT;

        // Blend Front normal
        const fnx = normArr[f3] * (1.0 - wSeam) + targetNx * wSeam;
        const fny = normArr[f3 + 1] * (1.0 - wSeam) + targetNy * wSeam;
        const fnz = normArr[f3 + 2] * (1.0 - wSeam) + targetFrontNz * wSeam;
        const invF = 1.0 / Math.max(1e-6, Math.sqrt(fnx * fnx + fny * fny + fnz * fnz));
        normArr[f3] = fnx * invF;
        normArr[f3 + 1] = fny * invF;
        normArr[f3 + 2] = fnz * invF;

        // Blend Back normal symmetrically
        const bnx = normArr[b3] * (1.0 - wSeam) + targetNx * wSeam;
        const bny = normArr[b3 + 1] * (1.0 - wSeam) + targetNy * wSeam;
        const bnz = normArr[b3 + 2] * (1.0 - wSeam) + targetBackNz * wSeam;
        const invB = 1.0 / Math.max(1e-6, Math.sqrt(bnx * bnx + bny * bny + bnz * bnz));
        normArr[b3] = bnx * invB;
        normArr[b3 + 1] = bny * invB;
        normArr[b3 + 2] = bnz * invB;
      }
    }
  }

  const normSmoothSteps = Math.max(
    isDualSolid && seamlessSmooth > 0.2 ? 2 : 0,
    Math.min(10, params.normalSmoothingSteps || 0)
  );
  for (let step = 0; step < normSmoothSteps; step++) {
    applyMeshSmoothingPass(
      normArr,
      tempArr,
      segs,
      cols,
      0.55,
      false,
      isClosedSeamX,
      isClosedSeamY,
      vertexCutoutMask,
      0
    );
    if (isDualSolid && useSymmetricBackRules) {
      applyMeshSmoothingPass(
        normArr,
        tempArr,
        segs,
        cols,
        0.55,
        false,
        isClosedSeamX,
        isClosedSeamY,
        vertexCutoutMask,
        gridVerts
      );
    }
    const maxSmoothVerts = isDualSolid && useSymmetricBackRules ? gridVerts * 2 : gridVerts;
    for (let i = 0; i < maxSmoothVerts; i++) {
      if (vertexCutoutMask[i] === 0) continue;
      const i3 = i * 3;
      const nx = normArr[i3];
      const ny = normArr[i3 + 1];
      const nz = normArr[i3 + 2];
      const invL = 1.0 / Math.max(1e-6, Math.sqrt(nx * nx + ny * ny + nz * nz));
      normArr[i3] = nx * invL;
      normArr[i3 + 1] = ny * invL;
      normArr[i3 + 2] = nz * invL;
    }
  }

  // Blend lateral seam band normals with analytical profile normals (or smooth 180° arch normals when seamlessSmooth is high)
  if (isDualSolid) {
    const sideAnalyticalBlend = Math.max(cornerSharpness, seamlessSmooth * 0.88);
    if (sideAnalyticalBlend > 0.02) {
      const startSideV = gridVerts * 2;
      for (let v = startSideV; v < totalVertices; v++) {
        const v3 = v * 3;
        const nx =
          normArr[v3] * (1.0 - sideAnalyticalBlend) +
          analyticalSideNormals[v3] * sideAnalyticalBlend;
        const ny =
          normArr[v3 + 1] * (1.0 - sideAnalyticalBlend) +
          analyticalSideNormals[v3 + 1] * sideAnalyticalBlend;
        const nz =
          normArr[v3 + 2] * (1.0 - sideAnalyticalBlend) +
          analyticalSideNormals[v3 + 2] * sideAnalyticalBlend;
        const invL = 1.0 / Math.max(1e-6, Math.sqrt(nx * nx + ny * ny + nz * nz));
        normArr[v3] = nx * invL;
        normArr[v3 + 1] = ny * invL;
        normArr[v3 + 2] = nz * invL;
      }
    }

    // Unify Ring 0 normals with Front Shell boundary normals and Ring 4 normals with Back Shell boundary normals
    // when cornerSharpness < 0.65 so lighting across the Front->Joint->Back transition is 100% seamless!
    if (cornerSharpness < 0.65) {
      const { seamEdgeVStart, seamEdgeVEnd, seamEdgeBaseV } = buffers;
      const edgeCount = Math.min(
        boundaryEdgeCount,
        CNNReconstructionBuffers.MAX_SEAM_EDGES
      );
      for (let e = 0; e < edgeCount; e++) {
        const vStart = seamEdgeVStart[e];
        const vEnd = seamEdgeVEnd[e];
        const baseV = seamEdgeBaseV[e];
        if (baseV <= 0) continue;

        const f0 = vStart * 3;
        const f1 = vEnd * 3;
        const b0 = (gridVerts + vStart) * 3;
        const b1 = (gridVerts + vEnd) * 3;

        const r0_0 = baseV * 3;
        const r0_1 = (baseV + 1) * 3;
        normArr[r0_0] = normArr[f0];
        normArr[r0_0 + 1] = normArr[f0 + 1];
        normArr[r0_0 + 2] = normArr[f0 + 2];
        normArr[r0_1] = normArr[f1];
        normArr[r0_1 + 1] = normArr[f1 + 1];
        normArr[r0_1 + 2] = normArr[f1 + 2];

        const r4_0 = (baseV + 8) * 3;
        const r4_1 = (baseV + 9) * 3;
        normArr[r4_0] = normArr[b0];
        normArr[r4_0 + 1] = normArr[b0 + 1];
        normArr[r4_0 + 2] = normArr[b0 + 2];
        normArr[r4_1] = normArr[b1];
        normArr[r4_1 + 1] = normArr[b1 + 1];
        normArr[r4_1 + 2] = normArr[b1 + 2];
      }
    }
  }

  // ============================================================================
  // STEP 6: POPULATE 512x512 MULTI-VIEW PBR TEXTURE ATLAS
  // - Top-Left (0..255, 0..255): Cara Frontal (Inpainted Front Image + 4-Zone PBR)
  // - Top-Right (256..511, 0..255): Parte de Atrás (Inpainted Back Image + Back Normals)
  // - Bottom Strip (0..511, 256..511): Dedicated Per-Edge Image-Color Seam Cells ("La Junta")
  // ============================================================================
  const albData = albedoDataTex.image.data as Uint8Array;
  const rmData = roughMetalDataTex.image.data as Uint8Array;
  const normData = normalDataTex.image.data as Uint8Array;
  const atlasW = ATLAS_TEX_SIZE; // 512

  // 6A. Top-Left (Front Image) & Top-Right (Back Image) — 100% Opaque & Inpainted at Borders
  for (let ty = 0; ty < h; ty++) {
    const v = 1.0 - ty / (h - 1);
    const atlasRowOffset = ty * atlasW;

    for (let tx = 0; tx < w; tx++) {
      const u = tx / (w - 1);
      const texIdx = ty * w + tx;
      const i3 = texIdx * 3;

      const gx = Math.min(segs, Math.round(u * segs));
      const gy = Math.min(segs, Math.round(v * segs));
      const vIdx = gy * cols + gx;
      const overrideZ = buffers.customZoneOverrides[vIdx];
      const zId =
        overrideZ >= 0 && overrideZ < 4
          ? overrideZ
          : buffers.zoneMap[texIdx] & 3;
      const zone = zones[zId] || zones[0];
      const zCol = zoneColors[zId] || zoneColors[0];
      const pat = evaluateZonePattern(zone.pattern, u, v);

      const nx = buffers.normalTensor[i3];
      const ny = buffers.normalTensor[i3 + 1];
      const nz = buffers.normalTensor[i3 + 2];

      let frontR = 0;
      let frontG = 0;
      let frontB = 0;

      if (params.renderNormalShading) {
        frontR = nx * 0.5 + 0.5;
        frontG = ny * 0.5 + 0.5;
        frontB = nz * 0.5 + 0.5;
      } else {
        // Use frontInpaintedRgb so silhouette boundaries and joints NEVER bleed background color!
        const imgR = buffers.frontInpaintedRgb[i3];
        const imgG = buffers.frontInpaintedRgb[i3 + 1];
        const imgB = buffers.frontInpaintedRgb[i3 + 2];
        const mix = zone.zoneTintMix;
        frontR = (imgR * (1.0 - mix) + zCol.r * mix) * pat;
        frontG = (imgG * (1.0 - mix) + zCol.g * mix) * pat;
        frontB = (imgB * (1.0 - mix) + zCol.b * mix) * pat;
      }

      const roughByte = Math.round(
        Math.max(0.04, Math.min(1.0, zone.roughness * (2.0 - pat))) * 255
      );
      const metalByte = Math.round(
        Math.max(0.0, Math.min(1.0, zone.metalness)) * 255
      );

      // Write Top-Left Quadrant (tx: 0..255, ty: 0..255) -> Front (Alpha = 255 so no seam pixel is ever discarded!)
      const frontA4 = (atlasRowOffset + tx) * 4;
      albData[frontA4] = Math.min(255, Math.max(0, Math.round(frontR * 255)));
      albData[frontA4 + 1] = Math.min(255, Math.max(0, Math.round(frontG * 255)));
      albData[frontA4 + 2] = Math.min(255, Math.max(0, Math.round(frontB * 255)));
      albData[frontA4 + 3] = 255;

      rmData[frontA4] = 255;
      rmData[frontA4 + 1] = roughByte;
      rmData[frontA4 + 2] = metalByte;
      rmData[frontA4 + 3] = 255;

      normData[frontA4] = Math.min(255, Math.max(0, Math.round((nx * 0.5 + 0.5) * 255)));
      normData[frontA4 + 1] = Math.min(255, Math.max(0, Math.round((ny * 0.5 + 0.5) * 255)));
      normData[frontA4 + 2] = Math.min(255, Math.max(0, Math.round((nz * 0.5 + 0.5) * 255)));
      normData[frontA4 + 3] = 255;

      // Write Top-Right Quadrant (tx: 256..511, ty: 0..255) -> Back (with symmetric 4-Zone PBR & pattern rules)
      const bnx = buffers.backNormalTensor[i3];
      const bny = buffers.backNormalTensor[i3 + 1];
      const bnz = buffers.backNormalTensor[i3 + 2];

      const backZId =
        overrideZ >= 0 && overrideZ < 4
          ? overrideZ
          : buffers.backZoneMap[texIdx] & 3;
      const backZone = zones[backZId] || zones[0];
      const backZCol = zoneColors[backZId] || zoneColors[0];
      const backPat = useSymmetricBackRules
        ? evaluateZonePattern(backZone.pattern, u, v)
        : 1.0;

      let backR = buffers.backInpaintedRgb[i3];
      let backG = buffers.backInpaintedRgb[i3 + 1];
      let backB = buffers.backInpaintedRgb[i3 + 2];
      if (params.renderNormalShading) {
        backR = bnx * 0.5 + 0.5;
        backG = bny * 0.5 + 0.5;
        backB = bnz * 0.5 + 0.5;
      } else if (useSymmetricBackRules) {
        const bMix = backZone.zoneTintMix;
        backR = (backR * (1.0 - bMix) + backZCol.r * bMix) * backPat;
        backG = (backG * (1.0 - bMix) + backZCol.g * bMix) * backPat;
        backB = (backB * (1.0 - bMix) + backZCol.b * bMix) * backPat;
      }

      const backRoughByte = useSymmetricBackRules
        ? Math.round(
            Math.max(0.04, Math.min(1.0, backZone.roughness * (2.0 - backPat))) *
              255
          )
        : roughByte;
      const backMetalByte = useSymmetricBackRules
        ? Math.round(Math.max(0.0, Math.min(1.0, backZone.metalness)) * 255)
        : metalByte;

      const backA4 = (atlasRowOffset + 256 + tx) * 4;
      albData[backA4] = Math.min(255, Math.max(0, Math.round(backR * 255)));
      albData[backA4 + 1] = Math.min(255, Math.max(0, Math.round(backG * 255)));
      albData[backA4 + 2] = Math.min(255, Math.max(0, Math.round(backB * 255)));
      albData[backA4 + 3] = 255;

      rmData[backA4] = 255;
      rmData[backA4 + 1] = backRoughByte;
      rmData[backA4 + 2] = backMetalByte;
      rmData[backA4 + 3] = 255;

      normData[backA4] = Math.min(255, Math.max(0, Math.round((bnx * 0.5 + 0.5) * 255)));
      normData[backA4 + 1] = Math.min(255, Math.max(0, Math.round((bny * 0.5 + 0.5) * 255)));
      normData[backA4 + 2] = Math.min(255, Math.max(0, Math.round((bnz * 0.5 + 0.5) * 255)));
      normData[backA4 + 3] = 255;
    }
  }

  // 6B. Bottom Strip (tx: 0..511, ty: 256..511) -> Lateral Seam Band ("La Junta")
  const useCustomSideAtlas =
    params.sideSourceMode === SideImageSourceMode.CustomSideUpload &&
    buffers.hasCustomSideImage;

  if (useCustomSideAtlas) {
    for (let ty = 0; ty < h; ty++) {
      const atlasRowOffset = (256 + ty) * atlasW;
      const srcRowOffset = ty * w;

      for (let ax = 0; ax < atlasW; ax++) {
        const tx = ax >> 1;
        const sIdx = srcRowOffset + tx;
        const s3 = sIdx * 3;
        const dst4 = (atlasRowOffset + ax) * 4;

        const snx = buffers.sideNormalTensor[s3];
        const sny = buffers.sideNormalTensor[s3 + 1];
        const snz = buffers.sideNormalTensor[s3 + 2];

        const sR = params.renderNormalShading
          ? snx * 0.5 + 0.5
          : buffers.sideRgbActive[s3];
        const sG = params.renderNormalShading
          ? sny * 0.5 + 0.5
          : buffers.sideRgbActive[s3 + 1];
        const sB = params.renderNormalShading
          ? snz * 0.5 + 0.5
          : buffers.sideRgbActive[s3 + 2];

        albData[dst4] = Math.min(255, Math.max(0, Math.round(sR * 255)));
        albData[dst4 + 1] = Math.min(255, Math.max(0, Math.round(sG * 255)));
        albData[dst4 + 2] = Math.min(255, Math.max(0, Math.round(sB * 255)));
        albData[dst4 + 3] = 255;

        rmData[dst4] = 255;
        rmData[dst4 + 1] = 72;
        rmData[dst4 + 2] = 190;
        rmData[dst4 + 3] = 255;

        normData[dst4] = Math.min(255, Math.max(0, Math.round((snx * 0.5 + 0.5) * 255)));
        normData[dst4 + 1] = Math.min(255, Math.max(0, Math.round((sny * 0.5 + 0.5) * 255)));
        normData[dst4 + 2] = Math.min(255, Math.max(0, Math.round((snz * 0.5 + 0.5) * 255)));
        normData[dst4 + 3] = 255;
      }
    }
  } else if (isDualSolid) {
    // Write dedicated 4x2 per-boundary-edge color cells so every single edge of the joint ("la junta")
    // takes the EXACT local Front & Back color, roughness, and metalness of the image at (vStart, vEnd)!
    const { seamEdgeVStart, seamEdgeVEnd } = buffers;
    const edgeCount = Math.min(
      boundaryEdgeCount,
      CNNReconstructionBuffers.MAX_SEAM_EDGES
    );
    const isCoin = params.sideSourceMode === SideImageSourceMode.MilledCoinEdge;
    const isFiligree = params.sideSourceMode === SideImageSourceMode.FiligreeSeamBand;

    for (let e = 0; e < edgeCount; e++) {
      const vStart = seamEdgeVStart[e];
      const vEnd = seamEdgeVEnd[e];
      const f0 = vStart * 3;
      const f1 = vEnd * 3;
      const b0 = (gridVerts + vStart) * 3;
      const b1 = (gridVerts + vEnd) * 3;

      let mod0 = 1.0;
      let mod1 = 1.0;
      if (isCoin) {
        mod0 = 0.88 + 0.12 * Math.cos(e * 1.57);
        mod1 = 0.88 + 0.12 * Math.cos((e + 1) * 1.57);
      } else if (isFiligree) {
        mod0 = 0.86 + 0.14 * Math.sin(e * 0.78);
        mod1 = 0.86 + 0.14 * Math.sin((e + 1) * 0.78);
      }

      const fR0 = Math.min(255, Math.max(0, Math.round(colArr[f0] * mod0 * 255)));
      const fG0 = Math.min(255, Math.max(0, Math.round(colArr[f0 + 1] * mod0 * 255)));
      const fB0 = Math.min(255, Math.max(0, Math.round(colArr[f0 + 2] * mod0 * 255)));

      const fR1 = Math.min(255, Math.max(0, Math.round(colArr[f1] * mod1 * 255)));
      const fG1 = Math.min(255, Math.max(0, Math.round(colArr[f1 + 1] * mod1 * 255)));
      const fB1 = Math.min(255, Math.max(0, Math.round(colArr[f1 + 2] * mod1 * 255)));

      const bR0 = Math.min(255, Math.max(0, Math.round(colArr[b0] * mod0 * 255)));
      const bG0 = Math.min(255, Math.max(0, Math.round(colArr[b0 + 1] * mod0 * 255)));
      const bB0 = Math.min(255, Math.max(0, Math.round(colArr[b0 + 2] * mod0 * 255)));

      const bR1 = Math.min(255, Math.max(0, Math.round(colArr[b1] * mod1 * 255)));
      const bG1 = Math.min(255, Math.max(0, Math.round(colArr[b1 + 1] * mod1 * 255)));
      const bB1 = Math.min(255, Math.max(0, Math.round(colArr[b1 + 2] * mod1 * 255)));

      // Match surface roughness and metalness of the adjacent Front zone so the joint has identical material response
      const ovZ = buffers.customZoneOverrides[vStart];
      const zIdx = ovZ >= 0 && ovZ < 4 ? ovZ : 0;
      const zCfg = zones[zIdx] || zones[0];
      const rByte = Math.round(Math.max(0.04, Math.min(1.0, zCfg.roughness)) * 255);
      const mByte = Math.round(Math.max(0.0, Math.min(1.0, zCfg.metalness)) * 255);

      const cellX = (e & 127) << 2;
      const cellY = 256 + ((e >> 7) << 1);
      const rowTop = cellY * atlasW + cellX;
      const rowBot = (cellY + 1) * atlasW + cellX;

      // Top Row of 4x2 cell (Front Image color at vStart [cols 0,1] and vEnd [cols 2,3])
      for (let c = 0; c < 4; c++) {
        const isEnd = c >= 2;
        const t4 = (rowTop + c) * 4;
        albData[t4] = isEnd ? fR1 : fR0;
        albData[t4 + 1] = isEnd ? fG1 : fG0;
        albData[t4 + 2] = isEnd ? fB1 : fB0;
        albData[t4 + 3] = 255;

        rmData[t4] = 255;
        rmData[t4 + 1] = rByte;
        rmData[t4 + 2] = mByte;
        rmData[t4 + 3] = 255;

        normData[t4] = 128;
        normData[t4 + 1] = 128;
        normData[t4 + 2] = 255;
        normData[t4 + 3] = 255;

        // Bottom Row of 4x2 cell (Back Image color at vStart [cols 0,1] and vEnd [cols 2,3])
        const b4 = (rowBot + c) * 4;
        albData[b4] = isEnd ? bR1 : bR0;
        albData[b4 + 1] = isEnd ? bG1 : bG0;
        albData[b4 + 2] = isEnd ? bB1 : bB0;
        albData[b4 + 3] = 255;

        rmData[b4] = 255;
        rmData[b4 + 1] = rByte;
        rmData[b4 + 2] = mByte;
        rmData[b4 + 3] = 255;

        normData[b4] = 128;
        normData[b4 + 1] = 128;
        normData[b4 + 2] = 255;
        normData[b4 + 3] = 255;
      }
    }
  }

  albedoDataTex.needsUpdate = true;
  roughMetalDataTex.needsUpdate = true;
  normalDataTex.needsUpdate = true;

  posAttr.needsUpdate = true;
  normAttr!.needsUpdate = true;
  uvAttr!.needsUpdate = true;
  colorAttr!.needsUpdate = true;

  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  // Human body template: replace the relief mesh with a closed human body fitted
  // to the silhouette (the atlas textures written above are reused as they are).
  if (
    params.humanBodyTemplate !== false &&
    presetShape === PresetBaseShape.HumanoidAPose &&
    params.topology === MathTopology.MonolithRelief
  ) {
    const body = buildHumanBodyTemplateGeometry(
      geometry,
      {
        mask: buffers.texelCutoutMask,
        w: CNN_TENSOR_SIZE,
        frontRgb: buffers.rgbInput,
        frontRelief: buffers.depthTensor,
        backRelief: buffers.backDepthTensor,
        vertexUsedMask: buffers.vertexCutoutMask,
        maxVertices: CNNReconstructionBuffers.MAX_VERTICES,
        atlas: {
          albedo: albData,
          roughMetal: rmData,
          atlasWidth: atlasW,
          background: [buffers.autoDetectedBgR, buffers.autoDetectedBgG, buffers.autoDetectedBgB],
          backBackground: buffers.hasCustomBackImage
            ? [buffers.backAutoBgR, buffers.backAutoBgG, buffers.backAutoBgB]
            : undefined,
        },
      },
      {
        depthScale: Math.max(0.2, Math.min(1.8, params.humanBodyDepth ?? 1.0)),
        reliefAmount: Math.max(0.0, Math.min(1.0, params.humanReliefDetail ?? 0.5)),
        // Finer than the relief grid so small features (face, fingers) have vertices.
        rows: Math.round(Math.max(32, Math.min(144, params.gridResolution)) * 2.2),
        faceDetail: params.humanFaceDetail ?? 1.0,
        rig: params.enableSkeleton === true,
      }
    );
    return {
      elapsedMs: performance.now() - t0,
      poissonSolveMs,
      vertexCount: body.vertexCount,
      weldedVerticesCount: 0,
      cutoutTrianglesRemoved: 0,
      stitchedSideFacesCount: 0,
      triangleCount: body.triangleCount,
      rig: body.rig,
    };
  }

  return {
    elapsedMs: performance.now() - t0,
    poissonSolveMs,
    vertexCount: totalVertices,
    weldedVerticesCount,
    cutoutTrianglesRemoved,
    stitchedSideFacesCount,
    triangleCount: activeTriangleCount,
  };
}
