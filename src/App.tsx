/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Archive,
  Camera,
  ChevronDown,
  ChevronUp,
  Code2,
  Compass,
  Download,
  FileBox,
  Image as ImageIcon,
  Layers,
  MousePointer,
  Paintbrush,
  RotateCcw,
  Sliders,
  Upload,
} from 'lucide-react';
import {
  CNN_TENSOR_SIZE,
  CNNReconstructionBuffers,
  DEFAULT_ZONES,
  getDetectedSilhouetteFitMetrics,
  PRESET_REFERENCE_IMAGES,
  processUploadedBackImage,
  processUploadedSideImage,
  renderBackPreviewToCanvas,
  renderCNNFeaturePreview,
  renderPresetImageToCanvas,
  renderSidePreviewToCanvas,
  runCNNReconstructionPipeline,
} from './engine/cnnReconstruction';
import {
  buildSinglePresetModelAssets,
  canvasToPNGBlob,
  createDepthMapCanvasFromBuffers,
  createNormalMapCanvasFromBuffers,
  createRGBCanvasFromBuffers,
  exportAllApplicationModelsToZip,
  triggerBlobDownload,
  uint8ToBlob,
} from './engine/exportEngine';
import { OPTIMIZED_LOOP_REFERENCE_CODE } from './engine/objectPool';
import {
  StudioViewport,
  StudioViewportHandle,
} from './components/StudioViewport';
import {
  BackImageSourceMode,
  BackgroundCutoutMode,
  CNNFeatureView,
  EngineTelemetry,
  LateralJointProfile,
  LightingConfig,
  LightingEnvironmentPreset,
  LimbUnionMode,
  MathTopology,
  NormalReconstructionMode,
  PresetBaseShape,
  ReconstructionParams,
  ShapeSilhouetteMode,
  SideImageSourceMode,
  SmoothingAlgorithm,
  TexturePattern,
  ViewportToolMode,
  ZoneConfig,
} from './types/studio';

type DockTab =
  | 'molds'
  | 'multiview'
  | 'reconstruction'
  | 'cutout'
  | 'smoothing'
  | 'normals'
  | 'zones'
  | 'lighting'
  | 'downloads';

const PRESET_BASE_SHAPE_LABELS: Record<
  PresetBaseShape,
  {
    name: string;
    shortLabel: string;
    tag: string;
    desc: string;
    recommendedMoldStrength: number;
    recommendedWrapUV: number;
  }
> = {
  [PresetBaseShape.None]: {
    name: 'Sin Molde Base (Relieve Libre)',
    shortLabel: 'Libre (Sin Molde)',
    tag: '2.5D / 3D Libre',
    desc: 'Reconstruye el volumen únicamente a partir de la imagen, normales y ecuaciones de Poisson sin forzar una anatomía preestablecida.',
    recommendedMoldStrength: 0,
    recommendedWrapUV: 0,
  },
  [PresetBaseShape.HumanoidAPose]: {
    name: 'Cuerpo Humano (Pose A / Pose T 15°–90°)',
    shortLabel: 'Humano Pose A/T',
    tag: 'Pose 15°–90°',
    desc: 'Maniquí anatómico completo con cabeza, cuello, torso pectoral, pelvis, brazos graduables de 15° a 90° (Pose A → Pose T) y piernas.',
    recommendedMoldStrength: 0.95,
    recommendedWrapUV: 0.45,
  },
  [PresetBaseShape.Sphere3D]: {
    name: 'Esfera Volumétrica 3D',
    shortLabel: 'Esfera 3D',
    tag: 'R = √(1 − r²)',
    desc: 'Esfera 3D completa (hemisferio frontal y trasero simétrico) con mapeo UV conforme esférico para amoldar cualquier textura.',
    recommendedMoldStrength: 0.96,
    recommendedWrapUV: 0.65,
  },
  [PresetBaseShape.QuadrupedAnimal]: {
    name: 'Animal Cuadrúpedo 3D',
    shortLabel: 'Animal Cuadrúpedo',
    tag: '4 Patas + Cabeza + Cola',
    desc: 'Morfología orgánica de cuadrúpedo con cabeza/hocico, orejas, cuello, tronco cilíndrico, grupa, cola y 4 patas articuladas.',
    recommendedMoldStrength: 0.95,
    recommendedWrapUV: 0.42,
  },
  [PresetBaseShape.AnatomicalBust]: {
    name: 'Busto Escultórico (Cabeza y Hombros)',
    shortLabel: 'Busto / Cabeza',
    tag: 'Cráneo + Hombros',
    desc: 'Molde de busto clásico con volumen craneal, arco superciliar/nasal, cuello anatómico y hombros escultóricos.',
    recommendedMoldStrength: 0.94,
    recommendedWrapUV: 0.5,
  },
  [PresetBaseShape.HeroicCuirass]: {
    name: 'Coraza / Torso Heroico',
    shortLabel: 'Coraza Torso',
    tag: 'Pectoral + Hombreras',
    desc: 'Torso de armadura con quilla esternal, placas pectorales, abdomen esculpido y hombreras (pauldrons) pronunciadas.',
    recommendedMoldStrength: 0.92,
    recommendedWrapUV: 0.4,
  },
  [PresetBaseShape.CylindricalTotem]: {
    name: 'Cilindro / Tótem Columna 3D',
    shortLabel: 'Cilindro 3D',
    tag: 'Envoltura ArcSin(X)',
    desc: 'Columna cilíndrica 3D de sección circular continua que curva la imagen horizontalmente alrededor del eje vertical.',
    recommendedMoldStrength: 0.95,
    recommendedWrapUV: 0.6,
  },
};

const SHAPE_SILHOUETTE_MODE_LABELS: Record<
  ShapeSilhouetteMode,
  { name: string; tag: string; desc: string }
> = {
  [ShapeSilhouetteMode.AdaptiveImageToShape]: {
    name: 'Amoldar a Silueta del Dibujo (Igual en Unión)',
    tag: 'Unión = Silueta',
    desc: 'El dibujo se amolda a la silueta real de la imagen y en la zona de la unión es 100% igual a la silueta con bordes suavizados.',
  },
  [ShapeSilhouetteMode.StrictShapeSilhouette]: {
    name: 'Silueta Exacta del Molde 3D',
    tag: '100% Molde',
    desc: 'Recorta el contorno exterior exactamente con la silueta del Cuerpo Pose A, Esfera o Cuadrúpedo y adapta la imagen a toda la figura.',
  },
  [ShapeSilhouetteMode.UnionImageAndShape]: {
    name: 'Fusión Completa (Imagen ∪ Molde)',
    tag: 'Foto ∪ Molde',
    desc: 'Combina toda la forma 3D base con cualquier elemento extra que sobresalga en la imagen original.',
  },
};

const LIMB_UNION_MODE_LABELS: Record<
  LimbUnionMode,
  { name: string; shortLabel: string; tag: string; desc: string }
> = {
  [LimbUnionMode.TubularCylindersFull]: {
    name: 'Tubos / Cilindros Curvos (Grosor Total)',
    shortLabel: 'Tubos / Cilindros',
    tag: 'Cilindro 180°',
    desc: 'Cada extremidad y torso forma un tubo o cilindro continuo del grosor total de la figura con uniones curvas entre partes.',
  },
  [LimbUnionMode.SmoothCurvedUnions]: {
    name: 'Uniones Curvas Orgánicas Suaves',
    shortLabel: 'Curvas Orgánicas',
    tag: 'smax Curvo',
    desc: 'Transiciones curvas amplias y continuas entre cabeza, cuello, pecho, brazos y piernas con perfil anatómico elíptico.',
  },
  [LimbUnionMode.AnatomicalArticulated]: {
    name: 'Uniones Anatómicas Articuladas',
    shortLabel: 'Articulación Definida',
    tag: 'Pliegue Muscular',
    desc: 'Marca con mayor definición las articulaciones de hombros, codos, cadera y rodillas manteniendo cilindros esbeltos.',
  },
  [LimbUnionMode.SlimBasRelief]: {
    name: 'Volumen Esbelto (Bajo-Relieve)',
    shortLabel: 'Esbelto / Bajo Vol.',
    tag: 'Sin Abultar',
    desc: 'Reduce el volumen base del molde para que los relieves de la foto se proyecten con máxima nitidez sin abultar.',
  },
};

const CUTOUT_MODE_LABELS: Record<
  BackgroundCutoutMode,
  { name: string; tag: string; desc: string }
> = {
  [BackgroundCutoutMode.FlatReliefAndBorder]: {
    name: 'Quitar Zona Plana y Marco Exterior',
    tag: 'Z ≤ ε_plano',
    desc: 'Hace desaparecer el marco plano alrededor de la imagen eliminando los triángulos de baja altura y pendiente nula.',
  },
  [BackgroundCutoutMode.AutoCornerColor]: {
    name: 'Color de Fondo Automático (Esquinas)',
    tag: 'Auto RGB/Hue',
    desc: 'Detecta automáticamente el color de las 4 esquinas del marco exterior y recorta todo el fondo coincidente.',
  },
  [BackgroundCutoutMode.CustomColorAndHue]: {
    name: 'Por Color o Tono de Color (Chroma / Hue)',
    tag: 'ΔColor / ΔTono',
    desc: 'Permite elegir un color o hacer clic en la imagen (Gotero) para eliminar o aislar cualquier color o gama tonal.',
  },
  [BackgroundCutoutMode.None]: {
    name: 'Conservar Marco Cuadrado Completo',
    tag: '100% Malla',
    desc: 'Mantiene la lámina cuadrada completa incluyendo el marco plano alrededor de la figura.',
  },
};

const JOINT_PROFILE_LABELS: Record<
  LateralJointProfile,
  {
    name: string;
    shortLabel: string;
    tag: string;
    defaultAngle: number;
    defaultSharpness: number;
    desc: string;
  }
> = {
  [LateralJointProfile.SeamlessSmoothUnion]: {
    name: 'Unión Invisible Suavizada (Igual a Silueta)',
    shortLabel: 'Unión Invisible',
    tag: 'Sin Costura',
    defaultAngle: 90,
    defaultSharpness: 0.0,
    desc: 'Hace desaparecer la unión en los bordes mediante suavizado continuo entre ambas partes manteniendo el contorno 100% igual a la silueta.',
  },
  [LateralJointProfile.TubularCylinderFull]: {
    name: 'Tubo / Cilindro Redondo (Grosor Total 180°)',
    shortLabel: 'Tubo / Cilindro',
    tag: 'Cilindro Total',
    defaultAngle: 90,
    defaultSharpness: 0.08,
    desc: 'La zona de unión forma un tubo o cilindro curvo continuo adaptado al grosor total de cada parte del cuerpo u objeto.',
  },
  [LateralJointProfile.FlatStraight]: {
    name: 'Unión Plana Directa (0°)',
    shortLabel: 'Plana 0°',
    tag: 'θ = 0°',
    defaultAngle: 0,
    defaultSharpness: 0.9,
    desc: 'Pared lateral completamente recta y plana entre la parte frontal y trasera sin protuberancia radial.',
  },
  [LateralJointProfile.Curved90Fillet]: {
    name: 'Curva Circular Tangente (0° → 90°)',
    shortLabel: 'Curva 90°',
    tag: 'Arco 90°',
    defaultAngle: 90,
    defaultSharpness: 0.15,
    desc: 'Unión curva continua desde 0° hasta arco de cuarto de círculo de 90° que envuelve orgánicamente el borde.',
  },
  [LateralJointProfile.RightAngle90]: {
    name: 'Ángulo Recto Ortogonal (90°)',
    shortLabel: 'Ángulo Recto 90°',
    tag: '∟ 90° Ortogonal',
    defaultAngle: 90,
    defaultSharpness: 1.0,
    desc: 'Unión en escuadra de 90° con escalón horizontal perpendicular y pared vertical a 90° con aristas vivas.',
  },
  [LateralJointProfile.ChamferBevel]: {
    name: 'Chaflán / Bisel Facetado (45°–90°)',
    shortLabel: 'Chaflán 45°',
    tag: '∠ 45°/90°',
    defaultAngle: 45,
    defaultSharpness: 0.95,
    desc: 'Unión biselada con planos inclinados en ángulo mecánico y facetas definidas entre Frente, Lateral y Atrás.',
  },
  [LateralJointProfile.ConcaveCove90]: {
    name: 'Media Caña Cóncava (90°)',
    shortLabel: 'Cóncava 90°',
    tag: '⌒ −90°',
    defaultAngle: 90,
    defaultSharpness: 0.4,
    desc: 'Curva invertida en media caña de hasta 90° hacia el interior de la banda de unión lateral.',
  },
};

const SMOOTHING_ALGO_LABELS: Record<
  SmoothingAlgorithm,
  { name: string; tag: string; desc: string }
> = {
  [SmoothingAlgorithm.TaubinVolumePreserving]: {
    name: 'Taubin λ/μ (Sin Contracción)',
    tag: 'Δ²(λ, −μ)',
    desc: 'Alterna un paso de suavizado positivo (+λ) con un paso de inflación (−μ) en 8 vecinos para alisar hasta 30 pasadas sin encoger el volumen 3D.',
  },
  [SmoothingAlgorithm.LaplacianIsotropic]: {
    name: 'Laplaciano Isótropo (8 Vecinos)',
    tag: '∇²_8P',
    desc: 'Promedia cada vértice con su anillo completo de 8 vecinos ortogonales y diagonales para máxima relajación orgánica de la malla.',
  },
  [SmoothingAlgorithm.BilateralEdgePreserving]: {
    name: 'Bilateral (Preserva Bordes)',
    tag: 'W(d²)',
    desc: 'Pondera los vecinos usando decaimiento espacial d² (sin Math.sqrt) para alisar superficies planas conservando aristas y relieves.',
  },
};

const TOPOLOGY_LABELS: Record<MathTopology, { name: string; formula: string }> = {
  [MathTopology.MonolithRelief]: {
    name: 'Relieve 3D Frontal',
    formula: 'z(u,v) = [α·Z_Poisson(u,v) + β·cos(lu)cos(mv)]·W(u,v)',
  },
  [MathTopology.SphericalHarmonics]: {
    name: 'Esfera Armónica 360°',
    formula: 'r(θ,φ) = R₀ + α·Z_Poisson(u,v) + β·Yₗᵐ(θ,φ)',
  },
  [MathTopology.AxialRevolution]: {
    name: 'Revolución Cilíndrica',
    formula: 'ρ(φ,z) = ρ₀(z) + α·Z_Poisson(u,v) + β·cos(mφ)',
  },
  [MathTopology.ImplicitGyroid]: {
    name: 'Giroide Biomimético',
    formula: 'sin(kx)cos(ky) + sin(ky)cos(kz) + α·Z_Poisson(u,v)',
  },
  [MathTopology.RiemannianTorus]: {
    name: 'Toroide de Riemann',
    formula: 'P(u,v) = (R + (r + α·Z_Poisson)·cos v)(cos u, sin u)',
  },
};

const NORMAL_MODE_LABELS: Record<
  NormalReconstructionMode,
  { name: string; equation: string; desc: string }
> = {
  [NormalReconstructionMode.HybridScreenedPoisson]: {
    name: 'Screened Poisson Híbrido',
    equation: '(∇² − λI)Z = ∂p/∂x + ∂q/∂y − λZ_CNN',
    desc: 'Integra las pendientes de las normales (p = −nx/nz, q = −ny/nz) estabilizadas con el volumen de baja frecuencia CNN.',
  },
  [NormalReconstructionMode.PurePoissonIntegration]: {
    name: 'Integración Pura de Normales',
    equation: '∇²Z = ∂(−nx/nz)/∂x + ∂(−ny/nz)/∂y',
    desc: 'Reconstruye el 100% de la geometría 3D integrando exclusivamente el campo vectorial de normales mediante Gauss-Seidel SOR.',
  },
  [NormalReconstructionMode.DirectNormalMapRGB]: {
    name: 'Mapa de Normales RGB Directo',
    equation: 'n = normalize(2R−1, ±(2G−1), 2B−1) → ∇²Z = ∇·g',
    desc: 'Decodifica la imagen de entrada como un Mapa de Normales en espacio tangente (R=X, G=Y, B=Z) e integra la malla 3D exacta.',
  },
};

const PATTERN_LABELS: Record<TexturePattern, string> = {
  [TexturePattern.OriginalPhoto]: 'Foto / Imagen Original (100%)',
  [TexturePattern.Solid]: 'Tinte Uniforme',
  [TexturePattern.NeuralCircuit]: 'Circuito Neuronal',
  [TexturePattern.BrushedMetal]: 'Metal Cepillado',
  [TexturePattern.CarbonFiber]: 'Fibra de Carbono',
  [TexturePattern.VeinedMarble]: 'Mármol Veteado',
  [TexturePattern.HexScales]: 'Escamas Hexagonales',
};

/**
 * Draws an uploaded image into the square CNN tensor canvas without stretching it.
 * The image is cropped to the character, scaled to fit (keeping its aspect ratio) and centred; the empty
 * margins are filled with the image's own corner colour so background detection
 * still sees one uniform background.
 */
function drawImageFittedToTensor(ctx: CanvasRenderingContext2D, img: HTMLImageElement) {
  const size = CNN_TENSOR_SIZE;
  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;

  // Average a small patch in each corner of the source image to get its background colour.
  const probe = document.createElement('canvas');
  probe.width = iw;
  probe.height = ih;
  const pctx = probe.getContext('2d', { willReadFrequently: true });
  let fill = '#0c1220';
  if (pctx) {
    pctx.drawImage(img, 0, 0);
    const patch = Math.max(1, Math.min(8, Math.floor(Math.min(iw, ih) / 20)));
    let r = 0, g = 0, b = 0, a = 0, n = 0;
    for (const [x, y] of [[0, 0], [iw - patch, 0], [0, ih - patch], [iw - patch, ih - patch]]) {
      const d = pctx.getImageData(x, y, patch, patch).data;
      for (let i = 0; i < d.length; i += 4) {
        r += d[i]; g += d[i + 1]; b += d[i + 2]; a += d[i + 3]; n++;
      }
    }
    // Transparent PNG backgrounds stay transparent so the alpha-based cutout keeps working.
    fill = a / n < 32 ? 'rgba(0,0,0,0)' : `rgb(${Math.round(r / n)},${Math.round(g / n)},${Math.round(b / n)})`;
  }

  // Crop to the character: bounding box of pixels that differ from the background.
  let cx0 = 0;
  let cy0 = 0;
  let cw = iw;
  let ch = ih;
  if (pctx) {
    const ps = Math.min(1, 512 / Math.max(iw, ih));
    const sw = Math.max(1, Math.round(iw * ps));
    const sh = Math.max(1, Math.round(ih * ps));
    const small = document.createElement('canvas');
    small.width = sw;
    small.height = sh;
    const sctx = small.getContext('2d', { willReadFrequently: true });
    if (sctx) {
      sctx.drawImage(img, 0, 0, sw, sh);
      const d = sctx.getImageData(0, 0, sw, sh).data;
      const m = /rgb\((\d+),(\d+),(\d+)\)/.exec(fill);
      const transparentBg = !m;
      const br = m ? +m[1] : 0;
      const bg = m ? +m[2] : 0;
      const bb = m ? +m[3] : 0;
      const colCount = new Int32Array(sw);
      const rowCount = new Int32Array(sh);
      for (let y = 0; y < sh; y++) {
        for (let x = 0; x < sw; x++) {
          const o = (y * sw + x) * 4;
          const fg = transparentBg
            ? d[o + 3] > 32
            : (d[o] - br) ** 2 + (d[o + 1] - bg) ** 2 + (d[o + 2] - bb) ** 2 > 30 * 30;
          if (fg) {
            colCount[x]++;
            rowCount[y]++;
          }
        }
      }
      // Ignore specks: a row/column must hold a few foreground pixels.
      const minHits = 2;
      let x0 = 0, x1 = sw - 1, y0 = 0, y1 = sh - 1;
      while (x0 < sw && colCount[x0] < minHits) x0++;
      while (x1 > x0 && colCount[x1] < minHits) x1--;
      while (y0 < sh && rowCount[y0] < minHits) y0++;
      while (y1 > y0 && rowCount[y1] < minHits) y1--;
      if (x1 - x0 > sw * 0.05 && y1 - y0 > sh * 0.05) {
        const margin = Math.round(Math.max(x1 - x0, y1 - y0) * 0.03);
        x0 = Math.max(0, x0 - margin);
        y0 = Math.max(0, y0 - margin);
        x1 = Math.min(sw - 1, x1 + margin);
        y1 = Math.min(sh - 1, y1 + margin);
        cx0 = x0 / ps;
        cy0 = y0 / ps;
        cw = (x1 - x0 + 1) / ps;
        ch = (y1 - y0 + 1) / ps;
      }
    }
  }

  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = fill;
  ctx.fillRect(0, 0, size, size);
  const scale = Math.min(size / cw, size / ch);
  const dw = cw * scale;
  const dh = ch * scale;
  ctx.drawImage(img, cx0, cy0, cw, ch, (size - dw) / 2, (size - dh) / 2, dw, dh);
}

export default function App() {
  const viewportRef = useRef<StudioViewportHandle | null>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cutoutPreviewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const multiviewFrontCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const multiviewBackCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const multiviewSideCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const backFileInputRef = useRef<HTMLInputElement | null>(null);
  const sideFileInputRef = useRef<HTMLInputElement | null>(null);

  const cnnBuffersRef = useRef<CNNReconstructionBuffers>(
    new CNNReconstructionBuffers()
  );

  const [activeTab, setActiveTab] = useState<DockTab>('molds');
  const [dockCollapsed, setDockCollapsed] = useState(false);
  const [showZeroGcModal, setShowZeroGcModal] = useState(false);

  const [selectedPresetId, setSelectedPresetId] = useState<string>(
    PRESET_REFERENCE_IMAGES[0].id
  );
  const [customImageName, setCustomImageName] = useState<string | null>(null);
  const [customBackImageName, setCustomBackImageName] = useState<string | null>(null);
  const [customSideImageName, setCustomSideImageName] = useState<string | null>(null);

  const [featureView, setFeatureView] = useState<CNNFeatureView>(
    CNNFeatureView.RGBInput
  );
  const [cnnVersion, setCnnVersion] = useState<number>(0);
  const [cnnInferenceMs, setCnnInferenceMs] = useState<number>(0);

  const [exportStatus, setExportStatus] = useState<string | null>(null);
  const [isExportingBatch, setIsExportingBatch] = useState<boolean>(false);

  const [params, setParams] = useState<ReconstructionParams>({
    topology: MathTopology.MonolithRelief,
    normalReconMode: NormalReconstructionMode.HybridScreenedPoisson,
    poissonIterations: 45,
    normalIntegrationWeight: 0.78,
    gradientSlopeGain: 1.45,
    screenedLambda: 0.12,
    invertNormalGreenY: false,
    showNormalVectors3D: false,
    renderNormalShading: false,
    gridResolution: 112,
    cnnDepthWeight: 0.95,
    mathModulationWeight: 0.14,
    normalMapStrength: 1.5,
    harmonicDegreeL: 3,
    harmonicOrderM: 2,
    smoothingAlgorithm: SmoothingAlgorithm.TaubinVolumePreserving,
    laplacianSmoothingSteps: 6,
    smoothingLambda: 0.58,
    tensorBlurPasses: 1,
    weldTolerance: 0.012,
    autoStitchSeams: true,
    normalSmoothingSteps: 2,
    cutoutMode: BackgroundCutoutMode.FlatReliefAndBorder,
    cutoutFlatThreshold: 0.14,
    cutoutBorderMargin: 0.06,
    cutoutTargetColorHex: '#0c1220',
    cutoutColorTolerance: 0.24,
    cutoutHueTolerance: 0.15,
    matchByHueTone: true,
    invertCutout: false,
    closeInternalHoles: true,
    holeFillRadius: 4,
    enableDualSideSolid: true,
    backSourceMode: BackImageSourceMode.SymmetricRelief,
    sideSourceMode: SideImageSourceMode.AutoEdgeExtension,
    backDepthWeight: 0.85,
    sideThicknessCm: 0.0,
    seamEdgeClosure: 0.98,
    sideJointProfile: LateralJointProfile.SeamlessSmoothUnion,
    sideJointAngleDeg: 90,
    sideCornerSharpness: 0.0,
    sideSeamBevel: 0.35,
    sideReliefAmplitude: 0.06,
    mirrorBackImageX: true,
    detectStraightLines: true,
    straightLineSensitivity: 0.78,
    flatFacetPlanarity: 0.68,
    symmetricBackRules: true,
    presetBaseShape: PresetBaseShape.HumanoidAPose,
    shapeSilhouetteMode: ShapeSilhouetteMode.AdaptiveImageToShape,
    autoFitMoldToSilhouette: true,
    silhouetteConformity: 0.94,
    limbUnionMode: LimbUnionMode.TubularCylindersFull,
    limbUnionCurvature: 0.68,
    reliefInflationSuppression: 0.85,
    poseArmAngleDeg: 35,
    autoFitLimbSizes: true,
    armLengthScale: 1.0,
    armThicknessScale: 1.0,
    legLengthScale: 1.0,
    legThicknessScale: 1.0,
    shapeMoldStrength: 0.72,
    shapeWrapUV: 0.42,
    shapeScaleX: 1.0,
    shapeScaleY: 1.0,
    shapeAspectRatio: 1.0,
    shapeOffsetX: 0.0,
    shapeOffsetY: 0.0,
    seamlessBorderSmoothing: 0.92,
    seamlessBorderWidth: 8.5,
    humanBodyTemplate: true,
    enableSkeleton: false,
    rigAnimation: 'idle',
    rigAnimationSpeed: 1.0,
    rigShowBones: true,
    humanBodyDepth: 1.0,
    humanReliefDetail: 0.5,
    silhouetteInflation: 0.75,
    contourSnapStrength: 1.0,
    reliefEdgeSmoothing: 0.7,
    wireframeOverlay: false,
    instancedVoxelMode: false,
    showTurntablePedestal: true,
  });

  const [zones, setZones] = useState<ZoneConfig[]>(DEFAULT_ZONES);
  const [selectedZoneIdx, setSelectedZoneIdx] = useState<number>(0);

  const [toolMode, setToolMode] = useState<ViewportToolMode>(
    ViewportToolMode.Orbit
  );
  const [brushRadius, setBrushRadius] = useState<number>(0.42);
  const [brushStrength, setBrushStrength] = useState<number>(0.65);

  const [lighting, setLighting] = useState<LightingConfig>({
    preset: LightingEnvironmentPreset.StudioWarm,
    azimuthDeg: 32,
    elevationDeg: 46,
    keyIntensity: 2.6,
    fillIntensity: 1.25,
    rimIntensity: 2.4,
    exposure: 1.15,
    autoRotateLight: false,
    autoRotateModel: true,
  });

  const [telemetry, setTelemetry] = useState<EngineTelemetry>({
    vertexCount: 12769,
    weldedVerticesCount: 0,
    cutoutTrianglesRemoved: 0,
    stitchedSideFacesCount: 0,
    triangleCount: 25088,
    activeInstances: 36,
    cnnInferenceMs: 3.8,
    poissonSolveMs: 4.2,
    meshRebuildMs: 5.4,
    pooledObjectsReused: 0,
    gcAllocationsInLoop: 0,
  });

  const getSourceCanvas = useCallback(() => {
    if (!sourceCanvasRef.current) {
      const c = document.createElement('canvas');
      c.width = CNN_TENSOR_SIZE;
      c.height = CNN_TENSOR_SIZE;
      sourceCanvasRef.current = c;
    }
    return sourceCanvasRef.current;
  }, []);

  const syncAllPreviewCanvases = useCallback(() => {
    if (previewCanvasRef.current) {
      renderCNNFeaturePreview(
        featureView,
        cnnBuffersRef.current,
        zones,
        previewCanvasRef.current
      );
    }
    if (cutoutPreviewCanvasRef.current) {
      renderCNNFeaturePreview(
        CNNFeatureView.RGBInput,
        cnnBuffersRef.current,
        zones,
        cutoutPreviewCanvasRef.current
      );
    }
    if (multiviewFrontCanvasRef.current) {
      renderCNNFeaturePreview(
        CNNFeatureView.RGBInput,
        cnnBuffersRef.current,
        zones,
        multiviewFrontCanvasRef.current
      );
    }
    if (multiviewBackCanvasRef.current) {
      renderBackPreviewToCanvas(
        cnnBuffersRef.current,
        multiviewBackCanvasRef.current
      );
    }
    if (multiviewSideCanvasRef.current) {
      renderSidePreviewToCanvas(
        cnnBuffersRef.current,
        multiviewSideCanvasRef.current
      );
    }
  }, [featureView, zones]);

  const executeCNNReconstruction = useCallback(
    (srcCanvas: HTMLCanvasElement) => {
      const elapsed = runCNNReconstructionPipeline(
        srcCanvas,
        cnnBuffersRef.current
      );
      setCnnInferenceMs(elapsed);
      setCnnVersion((v) => v + 1);
      syncAllPreviewCanvases();
    },
    [syncAllPreviewCanvases]
  );

  useEffect(() => {
    const srcCanvas = getSourceCanvas();
    renderPresetImageToCanvas(selectedPresetId, srcCanvas);
    cnnBuffersRef.current.resetSculptAndZoneOverrides();
    executeCNNReconstruction(srcCanvas);
  }, [selectedPresetId, getSourceCanvas]);

  useEffect(() => {
    syncAllPreviewCanvases();
  }, [featureView, zones, cnnVersion, params, activeTab, syncAllPreviewCanvases]);

  const handlePreviewCanvasEyedropper = (
    e: React.MouseEvent<HTMLCanvasElement>
  ) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const u = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const v = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    const tx = Math.min(
      CNN_TENSOR_SIZE - 1,
      Math.floor(u * (CNN_TENSOR_SIZE - 1))
    );
    const ty = Math.min(
      CNN_TENSOR_SIZE - 1,
      Math.floor(v * (CNN_TENSOR_SIZE - 1))
    );
    const idx3 = (ty * CNN_TENSOR_SIZE + tx) * 3;
    const rgb = cnnBuffersRef.current.rgbInput;
    const r = Math.min(255, Math.max(0, Math.round(rgb[idx3] * 255)));
    const g = Math.min(255, Math.max(0, Math.round(rgb[idx3 + 1] * 255)));
    const b = Math.min(255, Math.max(0, Math.round(rgb[idx3 + 2] * 255)));
    const pickedHex =
      '#' +
      r.toString(16).padStart(2, '0') +
      g.toString(16).padStart(2, '0') +
      b.toString(16).padStart(2, '0');

    setParams((prev) => ({
      ...prev,
      cutoutMode: BackgroundCutoutMode.CustomColorAndHue,
      cutoutTargetColorHex: pickedHex,
    }));
  };

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLSelectElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      if (e.key.toLowerCase() === 'h') {
        setDockCollapsed((prev) => !prev);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const handleSelectPreset = (presetId: string) => {
    setCustomImageName(null);
    setSelectedPresetId(presetId);
    setZones(DEFAULT_ZONES.map((z) => ({ ...z })));
    const found = PRESET_REFERENCE_IMAGES.find((p) => p.id === presetId);
    if (found) {
      setParams((prev) => ({
        ...prev,
        topology:
          prev.presetBaseShape !== PresetBaseShape.None || prev.enableDualSideSolid
            ? MathTopology.MonolithRelief
            : found.recommendedTopology,
        normalReconMode:
          found.recommendedNormalMode ||
          NormalReconstructionMode.HybridScreenedPoisson,
      }));
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = new Image();
      img.onload = () => {
        const srcCanvas = getSourceCanvas();
        const ctx = srcCanvas.getContext('2d');
        if (!ctx) return;
        drawImageFittedToTensor(ctx, img);
        setCustomImageName(file.name);
        setFeatureView(CNNFeatureView.RGBInput);
        setZones(DEFAULT_ZONES.map((z) => ({ ...z })));
        cnnBuffersRef.current.resetSculptAndZoneOverrides();
        executeCNNReconstruction(srcCanvas);
        setParams((prev) => ({
          ...prev,
          cutoutTargetColorHex: cnnBuffersRef.current.autoDetectedBgHex,
          closeInternalHoles: true,
        }));
      };
      if (typeof ev.target?.result === 'string') {
        img.src = ev.target.result;
      }
    };
    reader.readAsDataURL(file);
  };

  const handleBackImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = new Image();
      img.onload = () => {
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = CNN_TENSOR_SIZE;
        tmpCanvas.height = CNN_TENSOR_SIZE;
        const ctx = tmpCanvas.getContext('2d');
        if (!ctx) return;
        drawImageFittedToTensor(ctx, img);
        processUploadedBackImage(tmpCanvas, cnnBuffersRef.current);
        setCustomBackImageName(file.name);
        setParams((prev) => ({
          ...prev,
          enableDualSideSolid: true,
          topology: MathTopology.MonolithRelief,
          backSourceMode: BackImageSourceMode.CustomBackUpload,
        }));
        setCnnVersion((v) => v + 1);
        syncAllPreviewCanvases();
      };
      if (typeof ev.target?.result === 'string') {
        img.src = ev.target.result;
      }
    };
    reader.readAsDataURL(file);
  };

  const handleSideImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = new Image();
      img.onload = () => {
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = CNN_TENSOR_SIZE;
        tmpCanvas.height = CNN_TENSOR_SIZE;
        const ctx = tmpCanvas.getContext('2d');
        if (!ctx) return;
        drawImageFittedToTensor(ctx, img);
        processUploadedSideImage(tmpCanvas, cnnBuffersRef.current);
        setCustomSideImageName(file.name);
        setParams((prev) => ({
          ...prev,
          enableDualSideSolid: true,
          topology: MathTopology.MonolithRelief,
          sideSourceMode: SideImageSourceMode.CustomSideUpload,
        }));
        setCnnVersion((v) => v + 1);
      };
      if (typeof ev.target?.result === 'string') {
        img.src = ev.target.result;
      }
    };
    reader.readAsDataURL(file);
  };

  const updateSelectedZone = <K extends keyof ZoneConfig>(
    key: K,
    value: ZoneConfig[K]
  ) => {
    setZones((prev) =>
      prev.map((z, idx) => {
        if (idx !== selectedZoneIdx) return z;
        const updated = { ...z, [key]: value };
        if (key === 'colorHex' && z.zoneTintMix === 0) {
          updated.zoneTintMix = 0.65;
        }
        return updated;
      })
    );
  };

  const resetAllZonesToOriginalPhoto = () => {
    setZones((prev) =>
      prev.map((z) => ({
        ...z,
        zoneTintMix: 0.0,
        pattern: TexturePattern.OriginalPhoto,
      }))
    );
  };

  // Active model texture downloads
  const activeBaseName = customImageName
    ? customImageName.replace(/\.[^/.]+$/, '')
    : selectedPresetId;

  const handleDownloadActiveRGB = async () => {
    const canvas = createRGBCanvasFromBuffers(cnnBuffersRef.current);
    const blob = await canvasToPNGBlob(canvas);
    triggerBlobDownload(blob, `${activeBaseName}_textura_rgb.png`);
  };

  const handleDownloadActiveNormals = async () => {
    const canvas = createNormalMapCanvasFromBuffers(cnnBuffersRef.current);
    const blob = await canvasToPNGBlob(canvas);
    triggerBlobDownload(blob, `${activeBaseName}_mapa_normales.png`);
  };

  const handleDownloadActiveDepth = async () => {
    const canvas = createDepthMapCanvasFromBuffers(cnnBuffersRef.current);
    const blob = await canvasToPNGBlob(canvas);
    triggerBlobDownload(blob, `${activeBaseName}_profundidad_poisson.png`);
  };

  // Download specific preset asset (RGB, Normal Map, or 3D .GLB)
  const handleDownloadCatalogItem = async (
    presetId: string,
    type: 'rgb' | 'normal' | 'glb'
  ) => {
    setExportStatus(`Generando ${type.toUpperCase()} de ${presetId}...`);
    try {
      const assets = await buildSinglePresetModelAssets(presetId, params, zones);
      if (type === 'rgb') {
        triggerBlobDownload(
          uint8ToBlob(assets.rgbBytes, 'image/png'),
          `${presetId}_textura_rgb.png`
        );
      } else if (type === 'normal') {
        triggerBlobDownload(
          uint8ToBlob(assets.normalBytes, 'image/png'),
          `${presetId}_mapa_normales.png`
        );
      } else {
        triggerBlobDownload(
          uint8ToBlob(assets.glbBytes, 'model/gltf-binary'),
          `${presetId}_modelo_3d.glb`
        );
      }
    } finally {
      setTimeout(() => setExportStatus(null), 2000);
    }
  };

  // Download ALL 6 application models (3D .GLB + .OBJ + RGB + Normales + Profundidad) in a single .ZIP
  const handleDownloadAllModelsZip = async () => {
    if (isExportingBatch) return;
    setIsExportingBatch(true);
    try {
      const zipBlob = await exportAllApplicationModelsToZip(
        params,
        (current, total, title) => {
          setExportStatus(
            `Reconstruyendo modelo ${current}/${total}: ${title}...`
          );
        }
      );
      setExportStatus('Empaquetando archivo .ZIP con los 6 modelos...');
      triggerBlobDownload(
        zipBlob,
        'vortex3d_todos_los_modelos_3d_rgb_normales.zip'
      );
      setExportStatus('Colección completa descargada (.ZIP)');
    } finally {
      setIsExportingBatch(false);
      setTimeout(() => setExportStatus(null), 3500);
    }
  };

  const activeZone = zones[selectedZoneIdx] || zones[0];

  return (
    <div className="w-screen h-screen flex flex-col bg-[#090b10] text-slate-100 overflow-hidden select-none">
      {/* Top Bar Contract: 3 zones separated by gap-8 */}
      <header className="h-14 flex items-center justify-between gap-8 px-6 border-b border-slate-800/90 bg-[#0b0e16] z-20 shrink-0">
        {/* Zone 1: Single text element Brand wordmark */}
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            viewportRef.current?.resetCamera();
          }}
          className="text-lg font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0"
        >
          Vortex3D
        </a>

        {/* Zone 2: Concise single-line navigation links */}
        <nav className="flex items-center gap-3 text-sm font-medium text-slate-400">
          <button
            type="button"
            onClick={() => {
              setActiveTab('molds');
              setDockCollapsed(false);
            }}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeTab === 'molds'
                ? 'text-sky-300 border-sky-400 font-semibold'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            Moldes 3D (Pose A / Animal)
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTab('multiview');
              setDockCollapsed(false);
            }}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeTab === 'multiview'
                ? 'text-slate-100 border-sky-400'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            Frente, Atrás y Lateral
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTab('reconstruction');
              setDockCollapsed(false);
            }}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeTab === 'reconstruction'
                ? 'text-slate-100 border-sky-400'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            Geometría
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTab('cutout');
              setDockCollapsed(false);
            }}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeTab === 'cutout'
                ? 'text-slate-100 border-sky-400'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            Marco y Color
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTab('smoothing');
              setDockCollapsed(false);
            }}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeTab === 'smoothing'
                ? 'text-slate-100 border-sky-400'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            Suavizado
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTab('normals');
              setDockCollapsed(false);
            }}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeTab === 'normals'
                ? 'text-slate-100 border-sky-400'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            Normales
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTab('zones');
              setDockCollapsed(false);
            }}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeTab === 'zones'
                ? 'text-slate-100 border-sky-400'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            Zonas
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTab('lighting');
              setDockCollapsed(false);
            }}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeTab === 'lighting'
                ? 'text-slate-100 border-sky-400'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            Iluminación
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTab('downloads');
              setDockCollapsed(false);
            }}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeTab === 'downloads'
                ? 'text-slate-100 border-sky-400'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            Descargas
          </button>
        </nav>

        {/* Zone 3: 1 primary action */}
        <div className="flex items-center shrink-0">
          <button
            type="button"
            onClick={handleDownloadAllModelsZip}
            disabled={isExportingBatch}
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-sky-400 rounded-lg hover:bg-sky-300 disabled:opacity-60 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Archive className="w-3.5 h-3.5" />
            {isExportingBatch
              ? 'Empaquetando Modelos...'
              : 'Descargar Todo (.ZIP)'}
          </button>
        </div>
      </header>

      {/* Main 3D Viewport Area (100% Unobstructed Center) */}
      <main className="relative flex-1 w-full overflow-hidden">
        <StudioViewport
          ref={viewportRef}
          buffers={cnnBuffersRef.current}
          cnnVersion={cnnVersion}
          cnnInferenceMs={cnnInferenceMs}
          params={params}
          zones={zones}
          lighting={lighting}
          toolMode={toolMode}
          activePaintZone={selectedZoneIdx}
          brushRadius={brushRadius}
          brushStrength={brushStrength}
          onTelemetryUpdate={setTelemetry}
        />

        {/* Minimal Spec Bar (Top-Left, Clean Unboxed Metadata with · separators) */}
        <div className="absolute top-3 left-6 pointer-events-none flex items-center gap-2 text-xs text-slate-400 font-mono tabular-nums z-10">
          {params.presetBaseShape !== PresetBaseShape.None && (
            <>
              <span className="text-sky-300 font-semibold">
                Molde: {PRESET_BASE_SHAPE_LABELS[params.presetBaseShape].shortLabel}
                {params.presetBaseShape === PresetBaseShape.HumanoidAPose ||
                params.presetBaseShape === PresetBaseShape.HeroicCuirass
                  ? ` (${params.poseArmAngleDeg}°)`
                  : ''}
              </span>
              <span aria-hidden="true">·</span>
            </>
          )}
          <span className="text-slate-200 font-semibold">
            {params.enableDualSideSolid
              ? `Sólido 3 Caras (Unión Lateral ${params.sideThicknessCm.toFixed(1)} cm)`
              : CUTOUT_MODE_LABELS[params.cutoutMode].name}
          </span>
          <span aria-hidden="true">·</span>
          <span>{telemetry.triangleCount.toLocaleString()} tris</span>
          {telemetry.stitchedSideFacesCount > 0 && (
            <>
              <span aria-hidden="true">·</span>
              <span className="text-sky-300">
                {telemetry.stitchedSideFacesCount.toLocaleString()} caras laterales cosidas
              </span>
            </>
          )}
          <span aria-hidden="true">·</span>
          <span className="text-emerald-300">
            {telemetry.weldedVerticesCount.toLocaleString()} puntos unidos
          </span>
          {exportStatus && (
            <>
              <span aria-hidden="true">·</span>
              <span className="text-amber-300 font-semibold">{exportStatus}</span>
            </>
          )}
        </div>

        {/* Optional Zero-GC Code Inspector Modal */}
        {showZeroGcModal && (
          <div className="fixed inset-x-0 top-16 flex justify-center z-30 px-4">
            <div className="bg-[#0b0f17] border border-slate-700 rounded-xl p-4 w-[min(920px,95vw)] shadow-2xl">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-slate-200">
                  Arquitectura de Bucle Principal Zero-GC (Object Pooling + InstancedMesh + distSq)
                </span>
                <button
                  type="button"
                  onClick={() => setShowZeroGcModal(false)}
                  className="px-2.5 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 rounded cursor-pointer"
                >
                  Cerrar
                </button>
              </div>
              <pre className="h-56 overflow-y-auto bg-[#07090e] border border-slate-800 rounded-lg p-3 text-[11px] leading-relaxed text-slate-300 font-mono select-text">
                <code>{OPTIMIZED_LOOP_REFERENCE_CODE}</code>
              </pre>
            </div>
          </div>
        )}

        {/* Unified Collapsible Bottom Control Dock */}
        <section
          aria-label="Panel de control unificado de reconstrucción 3D"
          className="fixed left-1/2 -translate-x-1/2 bottom-3 z-20 w-[min(1100px,96vw)] transition-transform duration-200 ease-out"
        >
          {/* Collapse / Expand Handle Bar */}
          <div className="flex items-center justify-between px-1 mb-1.5">
            <div className="flex items-center gap-1 p-1 bg-[#0c1018] border border-slate-800 rounded-lg">
              <button
                type="button"
                onClick={() => setToolMode(ViewportToolMode.Orbit)}
                className={`flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  toolMode === ViewportToolMode.Orbit
                    ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <MousePointer className="w-3.5 h-3.5" />
                Órbita 3D
              </button>
              <button
                type="button"
                onClick={() => {
                  setToolMode(ViewportToolMode.SculptNormals);
                  setActiveTab('normals');
                  setDockCollapsed(false);
                }}
                className={`flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  toolMode === ViewportToolMode.SculptNormals
                    ? 'bg-indigo-500/25 text-indigo-300 border border-indigo-400/50'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Compass className="w-3.5 h-3.5" />
                Peinar Normales 3D
              </button>
              <button
                type="button"
                onClick={() => setToolMode(ViewportToolMode.SculptInflate)}
                className={`flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  toolMode === ViewportToolMode.SculptInflate
                    ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Sliders className="w-3.5 h-3.5" />
                Esculpir +
              </button>
              <button
                type="button"
                onClick={() => setToolMode(ViewportToolMode.SculptDeflate)}
                className={`flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  toolMode === ViewportToolMode.SculptDeflate
                    ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Sliders className="w-3.5 h-3.5" />
                Cavidad −
              </button>
              <button
                type="button"
                onClick={() => {
                  setToolMode(ViewportToolMode.PaintZone);
                  setActiveTab('zones');
                  setDockCollapsed(false);
                }}
                className={`flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  toolMode === ViewportToolMode.PaintZone
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Paintbrush className="w-3.5 h-3.5" />
                Pintar {zones[selectedZoneIdx]?.name.split('·')[0]}
              </button>
              <button
                type="button"
                onClick={() => viewportRef.current?.resetSculptEdits()}
                title="Restaurar deformaciones de malla y normales"
                className="flex items-center gap-1 px-2.5 py-1 text-xs text-slate-400 hover:text-slate-200 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Limpiar trazo
              </button>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setShowZeroGcModal((prev) => !prev)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium bg-[#0c1018] border border-slate-800 rounded-lg text-slate-400 hover:text-slate-200 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <Code2 className="w-3.5 h-3.5 text-emerald-400" />
                Código Zero-GC
              </button>

              <button
                type="button"
                onClick={() => setDockCollapsed((prev) => !prev)}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#0c1018] border border-slate-800 rounded-lg text-slate-300 hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                {dockCollapsed ? (
                  <>
                    <ChevronUp className="w-3.5 h-3.5 text-sky-400" />
                    Mostrar Controles (H)
                  </>
                ) : (
                  <>
                    <ChevronDown className="w-3.5 h-3.5 text-sky-400" />
                    Ocultar Controles (H)
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Dock Drawer Content */}
          {!dockCollapsed && (
            <div className="bg-[#0c1018] border border-slate-800 rounded-xl p-4 shadow-2xl">
              {/* Hidden file inputs so they are always mounted regardless of active tab */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
              <input
                ref={backFileInputRef}
                type="file"
                accept="image/*"
                onChange={handleBackImageUpload}
                className="hidden"
              />
              <input
                ref={sideFileInputRef}
                type="file"
                accept="image/*"
                onChange={handleSideImageUpload}
                className="hidden"
              />

              {/* TAB 0A: MOLDES Y FORMAS PREESTABLECIDAS 3D (POSE A, ESFERA, CUADRÚPEDO, BUSTO, ETC.) */}
              {activeTab === 'molds' && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
                  {/* Left 7 Cols: Preset 3D Base Shape Selector Grid */}
                  <div className="lg:col-span-7 flex flex-col justify-between border-r border-slate-800/80 pr-4">
                    {/* Human body template fitted to the drawing's silhouette */}
                    <div className="mb-3 rounded-lg border border-emerald-500/40 bg-emerald-500/5 p-2.5">
                      <label className="flex items-center gap-2 text-xs font-semibold text-emerald-200 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={params.humanBodyTemplate !== false}
                          onChange={(e) =>
                            setParams((p) => ({ ...p, humanBodyTemplate: e.target.checked }))
                          }
                        />
                        Cuerpo humano ajustado a la silueta (con molde Humano Pose A)
                      </label>
                      <p className="text-[11px] text-slate-400 mt-0.5 mb-2">
                        Construye un cuerpo cerrado (cabeza, torso, brazos y piernas con proporciones humanas) que sigue
                        exactamente el contorno del dibujo, y le aplica la imagen de frente y de espalda. Sin juntas laterales.
                      </p>
                      <div className="flex flex-wrap gap-4">
                        <div className="flex-1 min-w-[160px]">
                          <div className="flex justify-between text-[11px] mb-0.5">
                            <span className="text-slate-400">Profundidad del cuerpo</span>
                            <span className="font-mono text-slate-200 tabular-nums">
                              {(params.humanBodyDepth ?? 1.0).toFixed(2)}
                            </span>
                          </div>
                          <input
                            type="range"
                            min={0.4}
                            max={1.6}
                            step={0.05}
                            value={params.humanBodyDepth ?? 1.0}
                            disabled={params.humanBodyTemplate === false}
                            onChange={(e) =>
                              setParams((p) => ({ ...p, humanBodyDepth: parseFloat(e.target.value) }))
                            }
                            className="w-full accent-emerald-400 cursor-pointer disabled:opacity-40"
                          />
                        </div>
                        <div className="flex-1 min-w-[160px]">
                          <div className="flex justify-between text-[11px] mb-0.5">
                            <span className="text-slate-400">Relieve de la imagen</span>
                            <span className="font-mono text-slate-200 tabular-nums">
                              {(params.humanReliefDetail ?? 0.5).toFixed(2)}
                            </span>
                          </div>
                          <input
                            type="range"
                            min={0}
                            max={1}
                            step={0.05}
                            value={params.humanReliefDetail ?? 0.5}
                            disabled={params.humanBodyTemplate === false}
                            onChange={(e) =>
                              setParams((p) => ({ ...p, humanReliefDetail: parseFloat(e.target.value) }))
                            }
                            className="w-full accent-emerald-400 cursor-pointer disabled:opacity-40"
                          />
                        </div>
                      </div>
                      {/* Skeleton + animations */}
                      <div className="mt-2.5 pt-2.5 border-t border-emerald-500/20">
                        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
                          <label className="flex items-center gap-2 text-xs font-semibold text-emerald-200 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={params.enableSkeleton === true}
                              disabled={params.humanBodyTemplate === false}
                              onChange={(e) =>
                                setParams((p) => ({ ...p, enableSkeleton: e.target.checked }))
                              }
                            />
                            Esqueleto y animaciones
                          </label>
                          <label className="flex items-center gap-1.5 text-[11px] text-slate-300 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={params.rigShowBones !== false}
                              disabled={params.enableSkeleton !== true}
                              onChange={(e) =>
                                setParams((p) => ({ ...p, rigShowBones: e.target.checked }))
                              }
                            />
                            Mostrar huesos
                          </label>
                        </div>
                        <p className="text-[11px] text-slate-400 mt-0.5 mb-1.5">
                          Coloca automáticamente un esqueleto humano (Hips, Spine, Head, brazos y piernas) dentro de la
                          silueta. El .GLB exportado incluye los huesos y todas las animaciones.
                        </p>
                        <div className="flex flex-wrap items-center gap-1.5">
                          {[
                            ['none', 'Pose de reposo'],
                            ['idle', 'Reposo'],
                            ['walk', 'Caminar'],
                            ['run', 'Correr'],
                            ['wave', 'Saludar'],
                          ].map(([id, label]) => (
                            <button
                              key={id}
                              type="button"
                              disabled={params.enableSkeleton !== true}
                              onClick={() => setParams((p) => ({ ...p, rigAnimation: id }))}
                              className={`px-2 py-1 text-[11px] font-semibold rounded border cursor-pointer transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                                (params.rigAnimation ?? 'idle') === id
                                  ? 'bg-emerald-500/25 border-emerald-400 text-emerald-100'
                                  : 'bg-slate-900 border-slate-700 text-slate-300 hover:text-slate-100'
                              }`}
                            >
                              {label}
                            </button>
                          ))}
                          <div className="flex items-center gap-2 ml-2 min-w-[180px] flex-1">
                            <span className="text-[11px] text-slate-400 whitespace-nowrap">Velocidad</span>
                            <input
                              type="range"
                              min={0.1}
                              max={2.5}
                              step={0.05}
                              value={params.rigAnimationSpeed ?? 1}
                              disabled={params.enableSkeleton !== true}
                              onChange={(e) =>
                                setParams((p) => ({ ...p, rigAnimationSpeed: parseFloat(e.target.value) }))
                              }
                              className="w-full accent-emerald-400 cursor-pointer disabled:opacity-40"
                            />
                            <span className="font-mono text-[11px] text-slate-200 tabular-nums">
                              {(params.rigAnimationSpeed ?? 1).toFixed(2)}x
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div>
                      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                        <div>
                          <span className="text-xs font-semibold text-slate-100 block">
                            Formas Base 3D Preestablecidas (Amoldado Sincronizado Frente + Atrás)
                          </span>
                          <span className="text-[11px] text-slate-400">
                            Tanto la imagen Frontal como la Trasera se conforman automáticamente a la misma silueta y molde 3D:
                          </span>
                        </div>
                        <div className="flex flex-wrap items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            className="flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-semibold bg-sky-500 hover:bg-sky-400 text-slate-950 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                          >
                            <Upload className="w-3 h-3" />
                            Subir Frente
                          </button>
                          <button
                            type="button"
                            onClick={() => backFileInputRef.current?.click()}
                            className="flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-semibold bg-amber-500 hover:bg-amber-400 text-slate-950 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                          >
                            <Upload className="w-3 h-3" />
                            {customBackImageName ? `Atrás: ${customBackImageName}` : 'Subir Atrás a Amoldar'}
                          </button>
                          <button
                            type="button"
                            onClick={() => viewportRef.current?.setCameraAngle(0)}
                            className="px-2 py-1 text-[10px] font-mono bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-700 cursor-pointer"
                            title="Ver cara frontal (0°)"
                          >
                            Ver Frente 0°
                          </button>
                          <button
                            type="button"
                            onClick={() => viewportRef.current?.setCameraAngle(180)}
                            className="px-2 py-1 text-[10px] font-mono bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 rounded border border-amber-500/40 cursor-pointer"
                            title="Girar cámara a la espalda (180°) para ver el molde trasero"
                          >
                            Ver Atrás 180°
                          </button>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
                        {(
                          [
                            PresetBaseShape.HumanoidAPose,
                            PresetBaseShape.Sphere3D,
                            PresetBaseShape.QuadrupedAnimal,
                            PresetBaseShape.AnatomicalBust,
                            PresetBaseShape.HeroicCuirass,
                            PresetBaseShape.CylindricalTotem,
                            PresetBaseShape.None,
                          ] as PresetBaseShape[]
                        ).map((shapeKey) => {
                          const info = PRESET_BASE_SHAPE_LABELS[shapeKey];
                          const isSelected = params.presetBaseShape === shapeKey;
                          return (
                            <button
                              key={shapeKey}
                              type="button"
                              onClick={() =>
                                setParams((p) => ({
                                  ...p,
                                  presetBaseShape: shapeKey,
                                  enableDualSideSolid: true,
                                  topology: MathTopology.MonolithRelief,
                                  shapeMoldStrength:
                                    shapeKey === PresetBaseShape.None
                                      ? p.shapeMoldStrength
                                      : Math.max(0.75, p.shapeMoldStrength),
                                }))
                              }
                              className={`p-2 rounded-lg border text-left transition-all cursor-pointer flex flex-col justify-between ${
                                isSelected
                                  ? 'bg-sky-500/20 border-sky-400 text-sky-100 shadow-sm'
                                  : 'bg-slate-900/80 border-slate-800 text-slate-300 hover:border-slate-700 hover:text-slate-100'
                              }`}
                            >
                              <div className="flex items-center justify-between gap-1 mb-1">
                                <span className="text-xs font-semibold truncate">
                                  {info.shortLabel}
                                </span>
                                <span
                                  className={`text-[9px] font-mono px-1 py-0.5 rounded whitespace-nowrap ${
                                    isSelected
                                      ? 'bg-sky-400/25 text-sky-200'
                                      : 'bg-slate-800 text-slate-400'
                                  }`}
                                >
                                  {shapeKey === PresetBaseShape.HumanoidAPose
                                    ? `Pose ${params.poseArmAngleDeg}°`
                                    : info.tag}
                                </span>
                              </div>
                              <p className="text-[10px] text-slate-400 line-clamp-2 leading-tight">
                                {info.desc}
                              </p>
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    {/* Silhouette Adaptation & Tubular / Cylindrical Limb Union Mode Selectors */}
                    <div className="pt-2 mt-1.5 border-t border-slate-800/80 flex flex-col gap-1.5">
                      <div className="flex flex-wrap items-center justify-between gap-1.5">
                        <span className="text-[11px] font-semibold text-sky-300">
                          Uniones y Sección (Tubos / Cilindros):
                        </span>
                        <div className="flex flex-wrap gap-1">
                          {(
                            Object.keys(
                              LIMB_UNION_MODE_LABELS
                            ) as LimbUnionMode[]
                          ).map((uMode) => {
                            const uInfo = LIMB_UNION_MODE_LABELS[uMode];
                            const active = params.limbUnionMode === uMode;
                            return (
                              <button
                                key={uMode}
                                type="button"
                                onClick={() =>
                                  setParams((p) => ({
                                    ...p,
                                    limbUnionMode: uMode,
                                    sideJointProfile:
                                      uMode === LimbUnionMode.TubularCylindersFull
                                        ? LateralJointProfile.TubularCylinderFull
                                        : p.sideJointProfile,
                                  }))
                                }
                                title={uInfo.desc}
                                className={`px-2 py-0.5 text-[10px] rounded border transition-colors cursor-pointer whitespace-nowrap ${
                                  active
                                    ? 'bg-sky-500/25 border-sky-400 text-sky-200 font-semibold'
                                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                                }`}
                              >
                                {uInfo.shortLabel}
                              </button>
                            );
                          })}
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center justify-between gap-1.5">
                        <span className="text-[11px] font-semibold text-slate-300">
                          Recorte de Silueta del Molde:
                        </span>
                        <div className="flex flex-wrap gap-1">
                          {(
                            Object.keys(
                              SHAPE_SILHOUETTE_MODE_LABELS
                            ) as ShapeSilhouetteMode[]
                          ).map((modeKey) => {
                            const modeInfo = SHAPE_SILHOUETTE_MODE_LABELS[modeKey];
                            const active = params.shapeSilhouetteMode === modeKey;
                            return (
                              <button
                                key={modeKey}
                                type="button"
                                onClick={() =>
                                  setParams((p) => ({
                                    ...p,
                                    shapeSilhouetteMode: modeKey,
                                  }))
                                }
                                title={modeInfo.desc}
                                className={`px-2 py-0.5 text-[10px] rounded border transition-colors cursor-pointer whitespace-nowrap ${
                                  active
                                    ? 'bg-emerald-500/20 border-emerald-400 text-emerald-200 font-semibold'
                                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                                }`}
                              >
                                {modeInfo.name}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Right 5 Cols: Auto-Silhouette Fit, Arm & Leg Sizing (Manual + Auto), Explicit Width/Height/Proportions, Seamless Border Union & Pose Angle */}
                  <div className="lg:col-span-5 flex flex-col justify-between gap-1.5">
                    {/* Auto-Fit Silhouette & Automatic Arm/Leg Sizing Bar */}
                    {(() => {
                      const detected = getDetectedSilhouetteFitMetrics(
                        cnnBuffersRef.current
                      );
                      return (
                        <div className="bg-slate-900/90 border border-emerald-500/35 rounded-lg p-2 flex flex-col gap-1">
                          <div className="flex items-center justify-between gap-1.5 flex-wrap">
                            <div className="flex items-center gap-2.5">
                              <label
                                className="flex items-center gap-1 text-[11px] font-semibold text-emerald-300 cursor-pointer"
                                title="Adapta automáticamente la altura, anchura, centrado y silueta del molde al tamaño exacto de tu dibujo"
                              >
                                <input
                                  type="checkbox"
                                  checked={params.autoFitMoldToSilhouette}
                                  onChange={(e) =>
                                    setParams((p) => ({
                                      ...p,
                                      autoFitMoldToSilhouette: e.target.checked,
                                      shapeSilhouetteMode: e.target.checked
                                        ? ShapeSilhouetteMode.AdaptiveImageToShape
                                        : p.shapeSilhouetteMode,
                                    }))
                                  }
                                  className="accent-emerald-400 cursor-pointer"
                                />
                                Auto-Silueta
                              </label>

                              <label
                                className="flex items-center gap-1 text-[11px] font-semibold text-sky-300 cursor-pointer"
                                title="Detecta y adapta automáticamente la longitud y el grosor de los brazos y piernas según la silueta de la imagen"
                              >
                                <input
                                  type="checkbox"
                                  checked={params.autoFitLimbSizes}
                                  onChange={(e) =>
                                    setParams((p) => ({
                                      ...p,
                                      autoFitLimbSizes: e.target.checked,
                                    }))
                                  }
                                  className="accent-sky-400 cursor-pointer"
                                />
                                Auto-Brazos/Piernas
                              </label>
                            </div>

                            <div className="flex items-center gap-1">
                              <button
                                type="button"
                                onClick={() => {
                                  const m = getDetectedSilhouetteFitMetrics(
                                    cnnBuffersRef.current
                                  );
                                  setParams((p) => ({
                                    ...p,
                                    autoFitMoldToSilhouette: true,
                                    autoFitLimbSizes: true,
                                    shapeSilhouetteMode:
                                      ShapeSilhouetteMode.AdaptiveImageToShape,
                                    silhouetteConformity: 0.96,
                                    seamlessBorderSmoothing: 0.95,
                                    sideJointProfile:
                                      LateralJointProfile.SeamlessSmoothUnion,
                                    sideThicknessCm: 0.0,
                                    shapeScaleX: 1.0,
                                    shapeScaleY: 1.0,
                                    shapeAspectRatio: m.aspectRatio,
                                    shapeOffsetX: 0.0,
                                    shapeOffsetY: 0.0,
                                    armLengthScale: m.armLengthScale,
                                    armThicknessScale: m.armThicknessScale,
                                    legLengthScale: m.legLengthScale,
                                    legThicknessScale: m.legThicknessScale,
                                    poseArmAngleDeg: m.poseArmAngleDeg,
                                  }));
                                }}
                                className="px-2 py-0.5 text-[10px] font-semibold bg-emerald-500/25 hover:bg-emerald-500/35 border border-emerald-400 text-emerald-200 rounded transition-colors cursor-pointer whitespace-nowrap"
                                title="Detecta y aplica automáticamente el tamaño de silueta, longitud/grosor de brazos y piernas y ángulo de pose"
                              >
                                Auto-Ajustar Todo ({detected.widthPct}%×{detected.heightPct}%)
                              </button>
                            </div>
                          </div>

                          <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                            <span>
                              Auto Brazos: L {detected.armLengthPct}% · G {detected.armThicknessPct}%
                            </span>
                            <span className="text-sky-300">
                              Auto Piernas: L {detected.legLengthPct}% · G {detected.legThicknessPct}%
                            </span>
                          </div>
                        </div>
                      );
                    })()}

                    {/* Control de Tamaño de Brazos y Piernas (Manual + Automático) */}
                    <div className="bg-slate-900/90 border border-indigo-500/35 rounded-lg p-1.5 flex flex-col gap-1">
                      <div className="flex items-center justify-between gap-1.5">
                        <span className="text-[11px] font-semibold text-indigo-300">
                          Tamaño Brazos y Piernas:
                        </span>
                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => {
                              const m = getDetectedSilhouetteFitMetrics(
                                cnnBuffersRef.current
                              );
                              setParams((p) => ({
                                ...p,
                                autoFitLimbSizes: true,
                                armLengthScale: m.armLengthScale,
                                armThicknessScale: m.armThicknessScale,
                                legLengthScale: m.legLengthScale,
                                legThicknessScale: m.legThicknessScale,
                              }));
                            }}
                            className="px-1.5 py-0.5 text-[10px] font-semibold bg-indigo-500/25 hover:bg-indigo-500/35 border border-indigo-400 text-indigo-200 rounded transition-colors cursor-pointer whitespace-nowrap"
                            title="Calcula automáticamente el tamaño óptimo de brazos y piernas a partir de la imagen"
                          >
                            Auto-Detectar
                          </button>
                          {(
                            [
                              [0.82, '82% Esbelto'],
                              [1.0, '100% Base'],
                              [1.18, '118% Atlético'],
                              [1.35, '135% Heroico'],
                            ] as const
                          ).map(([scaleVal, label]) => (
                            <button
                              key={scaleVal}
                              type="button"
                              onClick={() =>
                                setParams((p) => ({
                                  ...p,
                                  armLengthScale: scaleVal,
                                  armThicknessScale: scaleVal,
                                  legLengthScale: scaleVal,
                                  legThicknessScale: scaleVal,
                                }))
                              }
                              className={`px-1.5 py-0.5 text-[10px] font-mono rounded border transition-colors cursor-pointer whitespace-nowrap ${
                                Math.abs(params.armThicknessScale - scaleVal) < 0.03 &&
                                Math.abs(params.legThicknessScale - scaleVal) < 0.03
                                  ? 'bg-indigo-500/25 border-indigo-400 text-indigo-200 font-semibold'
                                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="grid grid-cols-4 gap-2">
                        <div>
                          <div className="flex justify-between text-[10px] mb-0.5">
                            <span
                              className="text-indigo-200 font-semibold truncate"
                              title="Aumenta o disminuye la longitud y alcance de los brazos"
                            >
                              Largo Brazos
                            </span>
                            <span className="font-mono text-indigo-300 tabular-nums">
                              {(params.armLengthScale * 100).toFixed(0)}%
                            </span>
                          </div>
                          <input
                            type="range"
                            min={0.5}
                            max={1.65}
                            step={0.02}
                            value={params.armLengthScale}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                armLengthScale: parseFloat(e.target.value),
                              }))
                            }
                            className="w-full accent-indigo-400 cursor-pointer"
                          />
                        </div>

                        <div>
                          <div className="flex justify-between text-[10px] mb-0.5">
                            <span
                              className="text-indigo-200 font-semibold truncate"
                              title="Aumenta o disminuye el grosor / volumen muscular de los brazos"
                            >
                              Grosor Brazos
                            </span>
                            <span className="font-mono text-indigo-300 tabular-nums">
                              {(params.armThicknessScale * 100).toFixed(0)}%
                            </span>
                          </div>
                          <input
                            type="range"
                            min={0.5}
                            max={1.65}
                            step={0.02}
                            value={params.armThicknessScale}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                armThicknessScale: parseFloat(e.target.value),
                              }))
                            }
                            className="w-full accent-indigo-400 cursor-pointer"
                          />
                        </div>

                        <div>
                          <div className="flex justify-between text-[10px] mb-0.5">
                            <span
                              className="text-sky-200 font-semibold truncate"
                              title="Aumenta o disminuye la longitud vertical de las piernas"
                            >
                              Largo Piernas
                            </span>
                            <span className="font-mono text-sky-300 tabular-nums">
                              {(params.legLengthScale * 100).toFixed(0)}%
                            </span>
                          </div>
                          <input
                            type="range"
                            min={0.5}
                            max={1.65}
                            step={0.02}
                            value={params.legLengthScale}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                legLengthScale: parseFloat(e.target.value),
                              }))
                            }
                            className="w-full accent-sky-400 cursor-pointer"
                          />
                        </div>

                        <div>
                          <div className="flex justify-between text-[10px] mb-0.5">
                            <span
                              className="text-sky-200 font-semibold truncate"
                              title="Aumenta o disminuye el grosor de muslos, gemelos y piernas"
                            >
                              Grosor Piernas
                            </span>
                            <span className="font-mono text-sky-300 tabular-nums">
                              {(params.legThicknessScale * 100).toFixed(0)}%
                            </span>
                          </div>
                          <input
                            type="range"
                            min={0.5}
                            max={1.65}
                            step={0.02}
                            value={params.legThicknessScale}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                legThicknessScale: parseFloat(e.target.value),
                              }))
                            }
                            className="w-full accent-sky-400 cursor-pointer"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Graduación del Ángulo de la Pose (15° hasta 90°: Pose A -> Pose T) */}
                    <div className="bg-slate-900/90 border border-sky-500/30 rounded-lg p-1.5">
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <span className="text-[11px] font-semibold text-sky-300">
                          Pose Brazos (15° a 90°):
                        </span>
                        <div className="flex items-center gap-1">
                          {(
                            [
                              [15, '15°'],
                              [35, '35° Pose A'],
                              [60, '60°'],
                              [90, '90° Pose T'],
                            ] as const
                          ).map(([degVal, label]) => (
                            <button
                              key={degVal}
                              type="button"
                              onClick={() =>
                                setParams((p) => ({
                                  ...p,
                                  poseArmAngleDeg: degVal,
                                }))
                              }
                              className={`px-1.5 py-0.5 text-[10px] font-mono rounded border transition-colors cursor-pointer whitespace-nowrap ${
                                Math.round(params.poseArmAngleDeg) === degVal
                                  ? 'bg-sky-500/25 border-sky-400 text-sky-200 font-semibold'
                                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-mono text-slate-400">
                          15°
                        </span>
                        <input
                          type="range"
                          min={15}
                          max={90}
                          step={1}
                          value={params.poseArmAngleDeg}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              poseArmAngleDeg: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-sky-400 cursor-pointer"
                        />
                        <span className="font-mono text-xs font-semibold text-sky-300 tabular-nums">
                          {params.poseArmAngleDeg.toFixed(0)}°
                        </span>
                      </div>
                    </div>

                    {/* Sliders: Anchura, Altura, Proporción, Unión = Silueta, Desaparecer Unión en Bordes y Anti-Inflado */}
                    <div className="grid grid-cols-3 gap-x-2.5 gap-y-1">
                      <div>
                        <div className="flex justify-between text-[10px] mb-0.5">
                          <span
                            className="text-sky-300 font-semibold truncate"
                            title="Define claramente la Anchura horizontal (X) de la silueta y el molde"
                          >
                            Anchura (Ancho X)
                          </span>
                          <span className="font-mono text-sky-200 tabular-nums">
                            {(params.shapeScaleX * 100).toFixed(0)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.45}
                          max={1.75}
                          step={0.02}
                          value={params.shapeScaleX}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              shapeScaleX: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-sky-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[10px] mb-0.5">
                          <span
                            className="text-sky-300 font-semibold truncate"
                            title="Define claramente la Altura vertical (Y) de la silueta y el molde"
                          >
                            Altura (Alto Y)
                          </span>
                          <span className="font-mono text-sky-200 tabular-nums">
                            {(params.shapeScaleY * 100).toFixed(0)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.45}
                          max={1.75}
                          step={0.02}
                          value={params.shapeScaleY}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              shapeScaleY: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-sky-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[10px] mb-0.5">
                          <span
                            className="text-sky-300 font-semibold truncate"
                            title="Relación de proporción entre anchura y altura de la figura"
                          >
                            Proporción X/Y
                          </span>
                          <span className="font-mono text-sky-200 tabular-nums">
                            {params.shapeAspectRatio.toFixed(2)}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.55}
                          max={1.55}
                          step={0.02}
                          value={params.shapeAspectRatio}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              shapeAspectRatio: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-sky-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[10px] mb-0.5">
                          <span
                            className="text-emerald-300 font-semibold truncate"
                            title="Hace que la unión en los bordes desaparezca con un suavizado continuo entre la parte frontal y trasera"
                          >
                            Suavizar Unión Borde
                          </span>
                          <span className="font-mono text-emerald-300 tabular-nums">
                            {(params.seamlessBorderSmoothing * 100).toFixed(0)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={params.seamlessBorderSmoothing}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              seamlessBorderSmoothing: parseFloat(
                                e.target.value
                              ),
                            }))
                          }
                          className="w-full accent-emerald-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[10px] mb-0.5">
                          <span
                            className="text-emerald-300 font-semibold truncate"
                            title="Fuerza a que el volumen y la zona de la unión sean idénticos a la silueta del dibujo"
                          >
                            Unión = Silueta
                          </span>
                          <span className="font-mono text-emerald-300 tabular-nums">
                            {(params.silhouetteConformity * 100).toFixed(0)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={params.silhouetteConformity}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              silhouetteConformity: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-emerald-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[10px] mb-0.5">
                          <span
                            className="text-amber-300 font-semibold truncate"
                            title="Elimina el abultamiento inicial de la foto para que los relieves se graben nítidos sin inflar al humano"
                          >
                            Anti-Inflado Relieve
                          </span>
                          <span className="font-mono text-amber-300 tabular-nums">
                            {(params.reliefInflationSuppression * 100).toFixed(0)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={params.reliefInflationSuppression}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              reliefInflationSuppression: parseFloat(
                                e.target.value
                              ),
                            }))
                          }
                          className="w-full accent-amber-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[10px] mb-0.5">
                          <span className="text-slate-300 truncate">
                            Grosor Molde 3D
                          </span>
                          <span className="font-mono text-slate-200 tabular-nums">
                            {(params.shapeMoldStrength * 100).toFixed(0)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.15}
                          max={1.2}
                          step={0.02}
                          value={params.shapeMoldStrength}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              shapeMoldStrength: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-sky-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[10px] mb-0.5">
                          <span className="text-slate-300 truncate">
                            Curvatura Tubos
                          </span>
                          <span className="font-mono text-slate-200 tabular-nums">
                            {(params.limbUnionCurvature * 100).toFixed(0)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.05}
                          max={1}
                          step={0.02}
                          value={params.limbUnionCurvature}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              limbUnionCurvature: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-emerald-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[10px] mb-0.5">
                          <span
                            className="text-slate-300 truncate"
                            title="Anchura de la banda de suavizado donde desaparece la unión del borde"
                          >
                            Banda Borde Suave
                          </span>
                          <span className="font-mono text-slate-200 tabular-nums">
                            {params.seamlessBorderWidth.toFixed(1)} cel
                          </span>
                        </div>
                        <input
                          type="range"
                          min={2}
                          max={18}
                          step={0.5}
                          value={params.seamlessBorderWidth}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              seamlessBorderWidth: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-emerald-400 cursor-pointer"
                        />
                      </div>
                    </div>

                    <div className="flex items-center justify-between gap-2 pt-1 border-t border-slate-800/80">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[11px] text-slate-400">
                          Detalle Relieve Foto:
                        </span>
                        <input
                          type="range"
                          min={0}
                          max={1.6}
                          step={0.05}
                          value={params.cnnDepthWeight}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              cnnDepthWeight: parseFloat(e.target.value),
                            }))
                          }
                          className="w-24 accent-amber-400 cursor-pointer"
                        />
                        <span className="font-mono text-[11px] text-amber-300 tabular-nums">
                          {params.cnnDepthWeight.toFixed(2)}
                        </span>
                      </div>

                      <button
                        type="button"
                        onClick={() =>
                          setParams((p) => ({
                            ...p,
                            poseArmAngleDeg: 35,
                            autoFitMoldToSilhouette: true,
                            autoFitLimbSizes: true,
                            armLengthScale: 1.0,
                            armThicknessScale: 1.0,
                            legLengthScale: 1.0,
                            legThicknessScale: 1.0,
                            silhouetteConformity: 0.94,
                            seamlessBorderSmoothing: 0.92,
                            seamlessBorderWidth: 8.5,
                            limbUnionMode: LimbUnionMode.TubularCylindersFull,
                            limbUnionCurvature: 0.68,
                            reliefInflationSuppression: 0.85,
                            shapeMoldStrength: 0.72,
                            shapeWrapUV: 0.42,
                            shapeScaleX: 1.0,
                            shapeScaleY: 1.0,
                            shapeAspectRatio: 1.0,
                            shapeOffsetX: 0.0,
                            shapeOffsetY: 0.0,
                            sideThicknessCm: 0.0,
                            sideJointProfile:
                              LateralJointProfile.SeamlessSmoothUnion,
                            shapeSilhouetteMode:
                              ShapeSilhouetteMode.AdaptiveImageToShape,
                          }))
                        }
                        className="px-2 py-1 text-[11px] bg-slate-800 hover:bg-slate-700 text-slate-200 rounded transition-colors cursor-pointer whitespace-nowrap"
                      >
                        Restaurar Molde y Extremidades
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 0: UNIÓN AUTOMÁTICA FRENTE + PARTE DE ATRÁS + LATERAL (EN CENTÍMETROS) */}
              {activeTab === 'multiview' && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
                  {/* Left: 3 Interactive Image Slots (1. Frente, 2. Parte de Atrás, 3. Lateral en cm) */}
                  <div className="lg:col-span-7 grid grid-cols-3 gap-3 border-r border-slate-800/80 pr-4">
                    {/* Card 1: Imagen Frontal */}
                    <div className="bg-slate-900/75 border border-slate-800 rounded-lg p-2.5 flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between gap-1 mb-1.5">
                          <span className="text-[11px] font-semibold text-sky-300">
                            1. Imagen Frontal
                          </span>
                          <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            className="flex items-center gap-1 px-2 py-0.5 text-[10px] font-semibold bg-sky-500 hover:bg-sky-400 text-slate-950 rounded transition-colors cursor-pointer whitespace-nowrap"
                          >
                            <Upload className="w-2.5 h-2.5" />
                            Subir Frente
                          </button>
                        </div>
                        <div className="relative w-full h-24 rounded overflow-hidden border border-slate-700 bg-slate-950 mb-1.5">
                          <canvas
                            ref={multiviewFrontCanvasRef}
                            onClick={handlePreviewCanvasEyedropper}
                            title="Clic para usar gotero de recorte de fondo"
                            className="w-full h-full object-cover cursor-crosshair"
                          />
                          <span className="absolute bottom-1 left-1 right-1 text-center text-[9px] bg-black/75 text-slate-200 py-0.5 rounded pointer-events-none truncate px-1">
                            {customImageName ||
                              PRESET_REFERENCE_IMAGES.find(
                                (p) => p.id === selectedPresetId
                              )?.title}
                          </span>
                        </div>
                      </div>
                      <div className="grid grid-cols-3 gap-1">
                        {PRESET_REFERENCE_IMAGES.slice(0, 3).map((preset, idx) => (
                          <button
                            key={preset.id}
                            type="button"
                            onClick={() => handleSelectPreset(preset.id)}
                            className={`py-0.5 text-[10px] rounded border transition-colors cursor-pointer truncate px-1 ${
                              !customImageName && selectedPresetId === preset.id
                                ? 'bg-sky-500/20 border-sky-400 text-sky-200 font-semibold'
                                : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                            }`}
                          >
                            0{idx + 1}. {preset.title.split(' ')[0]}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Card 2: Imagen de la Parte de Atrás */}
                    <div className="bg-slate-900/75 border border-slate-800 rounded-lg p-2.5 flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between gap-1 mb-1.5">
                          <span className="text-[11px] font-semibold text-emerald-300">
                            2. Parte de Atrás
                          </span>
                          <button
                            type="button"
                            onClick={() => backFileInputRef.current?.click()}
                            className="flex items-center gap-1 px-2 py-0.5 text-[10px] font-semibold bg-emerald-400 hover:bg-emerald-300 text-slate-950 rounded transition-colors cursor-pointer whitespace-nowrap"
                          >
                            <Upload className="w-2.5 h-2.5" />
                            Subir Atrás
                          </button>
                        </div>
                        <div className="relative w-full h-24 rounded overflow-hidden border border-slate-700 bg-slate-950 mb-1.5">
                          <canvas
                            ref={multiviewBackCanvasRef}
                            className="w-full h-full object-cover"
                          />
                          <span className="absolute bottom-1 left-1 right-1 text-center text-[9px] bg-black/75 text-emerald-200 py-0.5 rounded pointer-events-none truncate px-1">
                            {params.backSourceMode ===
                              BackImageSourceMode.CustomBackUpload &&
                            customBackImageName
                              ? customBackImageName
                              : params.backSourceMode ===
                                BackImageSourceMode.EngravedBackplate
                              ? 'Reverso Grabado Metálico'
                              : params.backSourceMode ===
                                BackImageSourceMode.SmoothCurvedBack
                              ? 'Dorso Curvo Liso'
                              : 'Simétrico al Frente'}
                          </span>
                        </div>
                      </div>
                      <select
                        value={params.backSourceMode}
                        onChange={(e) =>
                          setParams((prev) => ({
                            ...prev,
                            enableDualSideSolid: true,
                            backSourceMode: e.target
                              .value as BackImageSourceMode,
                          }))
                        }
                        className="w-full bg-slate-950 border border-slate-800 rounded px-1.5 py-1 text-[10px] text-slate-200 focus:outline-none focus:border-emerald-400 cursor-pointer"
                      >
                        <option value={BackImageSourceMode.SymmetricRelief}>
                          Simétrico al Frente (Espejo)
                        </option>
                        <option value={BackImageSourceMode.EngravedBackplate}>
                          Reverso Grabado de Medalla
                        </option>
                        <option value={BackImageSourceMode.SmoothCurvedBack}>
                          Dorso Convexo Pulido
                        </option>
                        <option value={BackImageSourceMode.CustomBackUpload}>
                          {customBackImageName
                            ? `Foto Atrás: ${customBackImageName}`
                            : 'Imagen Subida (Parte de Atrás)'}
                        </option>
                      </select>
                    </div>

                    {/* Card 3: Imagen del Lateral (Unos Centímetros para las Uniones) */}
                    <div className="bg-slate-900/75 border border-slate-800 rounded-lg p-2.5 flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between gap-1 mb-1.5">
                          <span className="text-[11px] font-semibold text-amber-300">
                            3. Banda Lateral ({params.sideThicknessCm.toFixed(1)} cm)
                          </span>
                          <button
                            type="button"
                            onClick={() => sideFileInputRef.current?.click()}
                            className="flex items-center gap-1 px-2 py-0.5 text-[10px] font-semibold bg-amber-400 hover:bg-amber-300 text-slate-950 rounded transition-colors cursor-pointer whitespace-nowrap"
                          >
                            <Upload className="w-2.5 h-2.5" />
                            Subir Lateral
                          </button>
                        </div>
                        <div className="relative w-full h-24 rounded overflow-hidden border border-slate-700 bg-slate-950 mb-1.5">
                          <canvas
                            ref={multiviewSideCanvasRef}
                            className="w-full h-full object-cover"
                          />
                          <span className="absolute bottom-1 left-1 right-1 text-center text-[9px] bg-black/75 text-amber-200 py-0.5 rounded pointer-events-none truncate px-1">
                            Unión de {params.sideThicknessCm.toFixed(1)} cm (Canto 360°)
                          </span>
                        </div>
                      </div>
                      <select
                        value={params.sideSourceMode}
                        onChange={(e) =>
                          setParams((prev) => ({
                            ...prev,
                            enableDualSideSolid: true,
                            sideSourceMode: e.target
                              .value as SideImageSourceMode,
                          }))
                        }
                        className="w-full bg-slate-950 border border-slate-800 rounded px-1.5 py-1 text-[10px] text-slate-200 focus:outline-none focus:border-amber-400 cursor-pointer"
                      >
                        <option value={SideImageSourceMode.AutoEdgeExtension}>
                          Fusión Automática Frente ↔ Atrás
                        </option>
                        <option value={SideImageSourceMode.MilledCoinEdge}>
                          Canto Estriado de Moneda
                        </option>
                        <option value={SideImageSourceMode.FiligreeSeamBand}>
                          Costura con Filigrana en Relieve
                        </option>
                        <option value={SideImageSourceMode.CustomSideUpload}>
                          {customSideImageName
                            ? `Foto Lateral: ${customSideImageName}`
                            : 'Imagen Subida (Banda Lateral)'}
                        </option>
                      </select>
                    </div>
                  </div>

                  {/* Right: Automatic Stitching Controls in Centimeters, Joint Profile & Invisible Seam Smoothing */}
                  <div className="lg:col-span-5 flex flex-col justify-between gap-1.5">
                    <div className="flex items-center justify-between gap-2 pb-1 border-b border-slate-800/80">
                      <label className="flex items-center gap-1.5 text-xs font-semibold text-slate-100 cursor-pointer truncate">
                        <input
                          type="checkbox"
                          checked={params.enableDualSideSolid}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              enableDualSideSolid: e.target.checked,
                              topology: e.target.checked
                                ? MathTopology.MonolithRelief
                                : p.topology,
                            }))
                          }
                          className="accent-sky-400 cursor-pointer"
                        />
                        Unir Frente + Lateral + Atrás
                      </label>

                      <div className="flex items-center gap-1.5 shrink-0">
                        <button
                          type="button"
                          title="Hace desaparecer la unión en los bordes suavizando entre ambas partes y manteniendo el contorno igual a la silueta"
                          onClick={() =>
                            setParams((p) => ({
                              ...p,
                              enableDualSideSolid: true,
                              sideJointProfile:
                                LateralJointProfile.SeamlessSmoothUnion,
                              seamlessBorderSmoothing: 0.96,
                              seamlessBorderWidth: 10.0,
                              silhouetteConformity: 0.96,
                              sideThicknessCm: 0.0,
                              seamEdgeClosure: 1.0,
                              closeInternalHoles: true,
                              holeFillRadius: Math.max(5, p.holeFillRadius),
                              weldTolerance: Math.max(0.018, p.weldTolerance),
                            }))
                          }
                          className="px-2 py-0.5 text-[10px] font-semibold bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/50 rounded transition-colors cursor-pointer whitespace-nowrap"
                        >
                          Desaparecer Unión (Suavizado en Silueta)
                        </button>

                        <label className="flex items-center gap-1 text-[11px] text-slate-300 cursor-pointer whitespace-nowrap">
                          <input
                            type="checkbox"
                            checked={params.closeInternalHoles}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                closeInternalHoles: e.target.checked,
                              }))
                            }
                            className="accent-emerald-400 cursor-pointer"
                          />
                          Sin Huecos Int.
                        </label>
                      </div>
                    </div>

                    {/* Joint Geometry Profile Selector */}
                    <div className="bg-slate-900/80 border border-slate-800 rounded-md p-1.5 flex flex-col gap-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[11px] font-semibold text-slate-200">
                          Geometría de Unión en los Bordes:
                        </span>
                        <span className="text-[10px] font-mono text-amber-300 bg-amber-500/10 border border-amber-500/30 px-1.5 py-0.5 rounded">
                          {JOINT_PROFILE_LABELS[params.sideJointProfile].tag} ·{' '}
                          {params.sideJointAngleDeg.toFixed(0)}°
                        </span>
                      </div>

                      <div className="grid grid-cols-4 gap-1">
                        {(
                          Object.values(
                            LateralJointProfile
                          ) as LateralJointProfile[]
                        ).map((profile) => {
                          const meta = JOINT_PROFILE_LABELS[profile];
                          const active = params.sideJointProfile === profile;
                          return (
                            <button
                              key={profile}
                              type="button"
                              title={meta.desc}
                              onClick={() =>
                                setParams((p) => ({
                                  ...p,
                                  enableDualSideSolid: true,
                                  sideJointProfile: profile,
                                  sideJointAngleDeg: meta.defaultAngle,
                                  sideCornerSharpness: meta.defaultSharpness,
                                  seamlessBorderSmoothing:
                                    profile ===
                                    LateralJointProfile.SeamlessSmoothUnion
                                      ? Math.max(0.92, p.seamlessBorderSmoothing)
                                      : p.seamlessBorderSmoothing,
                                }))
                              }
                              className={`px-1.5 py-1 text-[10px] font-semibold rounded border transition-colors cursor-pointer truncate ${
                                active
                                  ? 'bg-amber-500/20 border-amber-400 text-amber-200 shadow-sm'
                                  : 'bg-slate-950/90 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                              }`}
                            >
                              {meta.shortLabel}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-x-3 gap-y-1">
                      <div>
                        <div className="flex justify-between text-[11px] mb-0.5">
                          <span
                            className="text-emerald-300 font-semibold truncate"
                            title="Hace desaparecer la unión en los bordes suavizando entre ambas partes"
                          >
                            Suavizar Unión Borde
                          </span>
                          <span className="font-mono text-emerald-300 font-semibold tabular-nums">
                            {(params.seamlessBorderSmoothing * 100).toFixed(0)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.0}
                          max={1.0}
                          step={0.02}
                          value={params.seamlessBorderSmoothing}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              enableDualSideSolid: true,
                              seamlessBorderSmoothing: parseFloat(
                                e.target.value
                              ),
                            }))
                          }
                          className="w-full accent-emerald-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] mb-0.5">
                          <span className="text-emerald-300 font-medium truncate">
                            Cerrar Hueco Borde
                          </span>
                          <span className="font-mono text-emerald-300 font-semibold tabular-nums">
                            {(params.seamEdgeClosure * 100).toFixed(0)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.0}
                          max={1.0}
                          step={0.02}
                          value={params.seamEdgeClosure}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              enableDualSideSolid: true,
                              seamEdgeClosure: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-emerald-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] mb-0.5">
                          <span className="text-amber-200 font-medium truncate">
                            Separación Lateral
                          </span>
                          <span className="font-mono text-amber-300 font-semibold tabular-nums">
                            {params.sideThicknessCm.toFixed(2)} cm
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.0}
                          max={4.0}
                          step={0.05}
                          value={params.sideThicknessCm}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              enableDualSideSolid: true,
                              sideThicknessCm: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-amber-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] mb-0.5">
                          <span className="text-sky-300 font-medium truncate">
                            Unir Zonas / Huecos
                          </span>
                          <span className="font-mono text-sky-300 font-semibold tabular-nums">
                            {params.holeFillRadius} px
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={12}
                          step={1}
                          value={params.holeFillRadius}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              closeInternalHoles: true,
                              holeFillRadius: parseInt(e.target.value, 10),
                            }))
                          }
                          className="w-full accent-sky-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] mb-0.5">
                          <span className="text-amber-300 font-medium truncate">
                            Ángulo (0°→90°)
                          </span>
                          <span className="font-mono text-amber-300 font-semibold tabular-nums">
                            {params.sideJointAngleDeg.toFixed(0)}°
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={90}
                          step={1}
                          value={params.sideJointAngleDeg}
                          onChange={(e) => {
                            const angle = parseFloat(e.target.value);
                            setParams((p) => ({
                              ...p,
                              enableDualSideSolid: true,
                              sideJointAngleDeg: angle,
                              sideJointProfile:
                                angle === 0
                                  ? LateralJointProfile.FlatStraight
                                  : p.sideJointProfile ===
                                      LateralJointProfile.FlatStraight
                                    ? LateralJointProfile.Curved90Fillet
                                    : p.sideJointProfile,
                            }));
                          }}
                          className="w-full accent-amber-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] mb-0.5">
                          <span className="text-sky-300 font-medium truncate">
                            Arista (Curva↔90°)
                          </span>
                          <span className="font-mono text-sky-300 font-semibold tabular-nums">
                            {(params.sideCornerSharpness * 100).toFixed(0)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.05}
                          value={params.sideCornerSharpness}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              enableDualSideSolid: true,
                              sideCornerSharpness: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-sky-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] mb-0.5">
                          <span className="text-emerald-200 font-medium truncate">
                            Relieve Atrás
                          </span>
                          <span className="font-mono text-emerald-300 tabular-nums">
                            {(params.backDepthWeight * 100).toFixed(0)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={1.6}
                          step={0.05}
                          value={params.backDepthWeight}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              enableDualSideSolid: true,
                              backDepthWeight: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-emerald-400 cursor-pointer"
                        />
                      </div>
                    </div>

                    {/* Straight Lines (Anti-Mound) + Symmetric Depth-Aware Back Rules Bar */}
                    <div className="bg-slate-900/90 border border-slate-800 rounded-md px-2 py-1.5 flex flex-col gap-1">
                      <div className="flex items-center justify-between gap-2">
                        <label
                          title="Detecta líneas rectas mediante Tensor de Estructura J y aplana facetas para que no se curven ni formen montículos"
                          className="flex items-center gap-1.5 text-[11px] font-semibold text-sky-300 cursor-pointer truncate"
                        >
                          <input
                            type="checkbox"
                            checked={params.detectStraightLines}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                detectStraightLines: e.target.checked,
                              }))
                            }
                            className="accent-sky-400 cursor-pointer"
                          />
                          Detectar Líneas Rectas (Sin Montículos)
                        </label>

                        <label
                          title="La parte de atrás aplica las mismas reglas (4 zonas, Poisson, líneas rectas, anti-montículos y suavizado) de forma simétrica y según sus propias profundidades"
                          className="flex items-center gap-1.5 text-[11px] font-semibold text-emerald-300 cursor-pointer whitespace-nowrap"
                        >
                          <input
                            type="checkbox"
                            checked={params.symmetricBackRules}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                enableDualSideSolid: true,
                                symmetricBackRules: e.target.checked,
                              }))
                            }
                            className="accent-emerald-400 cursor-pointer"
                          />
                          Atrás Simétrico (Mismas Reglas + Prof.)
                        </label>
                      </div>

                      {params.detectStraightLines && (
                        <div className="grid grid-cols-2 gap-3 pt-0.5">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-slate-400 whitespace-nowrap">
                              Rectitud:
                            </span>
                            <input
                              type="range"
                              min={0.1}
                              max={1.0}
                              step={0.02}
                              value={params.straightLineSensitivity}
                              onChange={(e) =>
                                setParams((p) => ({
                                  ...p,
                                  straightLineSensitivity: parseFloat(
                                    e.target.value
                                  ),
                                }))
                              }
                              className="w-full accent-sky-400 cursor-pointer"
                            />
                            <span className="font-mono text-[10px] text-sky-300 tabular-nums">
                              {(params.straightLineSensitivity * 100).toFixed(0)}%
                            </span>
                          </div>

                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-slate-400 whitespace-nowrap">
                              Sin Montículos:
                            </span>
                            <input
                              type="range"
                              min={0.0}
                              max={1.0}
                              step={0.02}
                              value={params.flatFacetPlanarity}
                              onChange={(e) =>
                                setParams((p) => ({
                                  ...p,
                                  flatFacetPlanarity: parseFloat(
                                    e.target.value
                                  ),
                                }))
                              }
                              className="w-full accent-emerald-400 cursor-pointer"
                            />
                            <span className="font-mono text-[10px] text-emerald-300 tabular-nums">
                              {(params.flatFacetPlanarity * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Quick Inspection Camera Orientation Buttons */}
                    <div className="flex items-center justify-between gap-1.5 pt-1 border-t border-slate-800/80">
                      <span className="text-[11px] text-slate-400 whitespace-nowrap">
                        Inspeccionar Unión:
                      </span>
                      <div className="flex items-center gap-1">
                        <button
                          type="button"
                          onClick={() => {
                            setLighting((l) => ({ ...l, autoRotateModel: false }));
                            viewportRef.current?.setCameraAngle(0);
                          }}
                          className="px-2 py-1 text-[11px] font-medium bg-slate-900 hover:bg-slate-800 text-sky-300 border border-slate-700 rounded transition-colors cursor-pointer whitespace-nowrap"
                        >
                          Frente (0°)
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setLighting((l) => ({ ...l, autoRotateModel: false }));
                            viewportRef.current?.setCameraAngle(82);
                          }}
                          className="px-2 py-1 text-[11px] font-medium bg-slate-900 hover:bg-slate-800 text-amber-300 border border-slate-700 rounded transition-colors cursor-pointer whitespace-nowrap"
                        >
                          Unión Lateral (90°)
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setLighting((l) => ({ ...l, autoRotateModel: false }));
                            viewportRef.current?.setCameraAngle(180);
                          }}
                          className="px-2 py-1 text-[11px] font-medium bg-slate-900 hover:bg-slate-800 text-emerald-300 border border-slate-700 rounded transition-colors cursor-pointer whitespace-nowrap"
                        >
                          Atrás (180°)
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            setLighting((l) => ({
                              ...l,
                              autoRotateModel: !l.autoRotateModel,
                            }))
                          }
                          className={`px-2 py-1 text-[11px] font-medium border rounded transition-colors cursor-pointer whitespace-nowrap ${
                            lighting.autoRotateModel
                              ? 'bg-sky-500/20 border-sky-400 text-sky-200'
                              : 'bg-slate-900 border-slate-700 text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          Giro 360°
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 1: IMAGEN DE ENTRADA Y TOPOLOGÍA MATEMÁTICA */}
              {activeTab === 'reconstruction' && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
                  <div className="lg:col-span-5 flex gap-3.5 items-center border-r border-slate-800/80 pr-4">
                    <div
                      title="Haz clic sobre cualquier zona de la imagen para capturar su color y recortarlo"
                      className="relative w-32 h-32 rounded-lg overflow-hidden border-2 border-sky-500/50 shrink-0 bg-slate-950 shadow-inner cursor-crosshair"
                    >
                      <canvas
                        ref={previewCanvasRef}
                        onClick={handlePreviewCanvasEyedropper}
                        className="w-full h-full object-cover"
                      />
                      <span className="absolute bottom-1 left-1 right-1 text-center text-[10px] font-medium bg-black/75 text-sky-200 py-0.5 rounded pointer-events-none">
                        {featureView === CNNFeatureView.RGBInput
                          ? 'Clic = Gotero Color'
                          : featureView === CNNFeatureView.NormalTensor
                          ? 'Campo Normales'
                          : featureView === CNNFeatureView.PoissonSurface
                          ? 'Superficie Poisson'
                          : featureView === CNNFeatureView.DivergenceCurvature
                          ? 'Divergencia ∇·n'
                          : featureView === CNNFeatureView.StraightLines
                          ? 'Líneas Rectas (J)'
                          : '4 Zonas CNN'}
                      </span>
                    </div>

                    <div className="flex flex-col justify-between h-full flex-1 min-w-0">
                      <div>
                        <div className="flex items-center justify-between gap-2 mb-1.5">
                          <span className="text-xs font-semibold text-slate-100 truncate">
                            {customImageName
                              ? customImageName
                              : PRESET_REFERENCE_IMAGES.find(
                                  (p) => p.id === selectedPresetId
                                )?.title}
                          </span>
                          <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            className="flex items-center gap-1 px-2.5 py-1 text-xs font-semibold bg-sky-500 hover:bg-sky-400 text-slate-950 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                          >
                            <Upload className="w-3 h-3" />
                            Subir Foto / Normal
                          </button>
                        </div>

                        <div className="flex flex-wrap gap-1 mb-2">
                          {PRESET_REFERENCE_IMAGES.map((preset, idx) => (
                            <button
                              key={preset.id}
                              type="button"
                              onClick={() => handleSelectPreset(preset.id)}
                              className={`px-2 py-0.5 text-[11px] font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                                !customImageName &&
                                selectedPresetId === preset.id
                                  ? 'bg-sky-500/25 border border-sky-400 text-sky-200 font-semibold'
                                  : 'bg-slate-900 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              0{idx + 1}. {preset.title.split(' ')[0]}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div>
                        <label className="block text-[11px] text-slate-400 mb-1">
                          Inspeccionar Tensores (256×256):
                        </label>
                        <div className="grid grid-cols-6 gap-1 bg-slate-900 p-0.5 rounded-md">
                          {(
                            [
                              [CNNFeatureView.RGBInput, 'RGB'],
                              [CNNFeatureView.NormalTensor, 'Normales'],
                              [CNNFeatureView.PoissonSurface, 'Poisson'],
                              [CNNFeatureView.StraightLines, 'Rectas'],
                              [CNNFeatureView.DivergenceCurvature, 'Curvatura'],
                              [CNNFeatureView.ZoneSegmentation, 'Zonas'],
                            ] as const
                          ).map(([viewKey, label]) => (
                            <button
                              key={viewKey}
                              type="button"
                              onClick={() => setFeatureView(viewKey)}
                              className={`py-1 text-[10px] font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                                featureView === viewKey
                                  ? 'bg-slate-800 text-sky-300 font-semibold'
                                  : 'text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="lg:col-span-3 flex flex-col justify-between border-r border-slate-800/80 pr-4">
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold text-slate-200">
                          Molde Base 3D Activo
                        </span>
                        <button
                          type="button"
                          onClick={() => setActiveTab('molds')}
                          className="text-[10px] font-semibold text-sky-400 hover:text-sky-300 cursor-pointer"
                        >
                          Configurar Molde →
                        </button>
                      </div>
                      <select
                        value={params.presetBaseShape}
                        onChange={(e) =>
                          setParams((prev) => ({
                            ...prev,
                            presetBaseShape: e.target.value as PresetBaseShape,
                          }))
                        }
                        className="w-full px-2 py-1 mb-2 text-xs bg-slate-900 border border-sky-500/40 rounded text-sky-200 font-semibold cursor-pointer"
                      >
                        {(
                          Object.keys(
                            PRESET_BASE_SHAPE_LABELS
                          ) as PresetBaseShape[]
                        ).map((shapeKey) => (
                          <option key={shapeKey} value={shapeKey}>
                            {PRESET_BASE_SHAPE_LABELS[shapeKey].name}
                          </option>
                        ))}
                      </select>

                      <span className="text-xs font-semibold text-slate-200 block mb-1">
                        Variedad Geométrica 3D
                      </span>
                      <div className="grid grid-cols-1 gap-1">
                        {(Object.keys(TOPOLOGY_LABELS) as MathTopology[]).map(
                          (topo) => (
                            <button
                              key={topo}
                              type="button"
                              onClick={() =>
                                setParams((prev) => ({
                                  ...prev,
                                  topology: topo,
                                }))
                              }
                              className={`px-2.5 py-0.5 text-left text-xs rounded-md border transition-colors whitespace-nowrap truncate cursor-pointer ${
                                params.topology === topo
                                  ? 'bg-sky-500/15 border-sky-500/50 text-sky-300 font-semibold'
                                  : 'bg-slate-900/70 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              {TOPOLOGY_LABELS[topo].name}
                            </button>
                          )
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="lg:col-span-4 grid grid-cols-2 gap-x-4 gap-y-2 items-center">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Amplitud 3D (α)</span>
                        <span className="font-mono text-sky-300 tabular-nums">
                          {params.cnnDepthWeight.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1.8}
                        step={0.02}
                        value={params.cnnDepthWeight}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            cnnDepthWeight: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Micro-Normales</span>
                        <span className="font-mono text-sky-300 tabular-nums">
                          {params.normalMapStrength.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={2.5}
                        step={0.05}
                        value={params.normalMapStrength}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            normalMapStrength: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Armónicos (β)</span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {params.mathModulationWeight.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1.0}
                        step={0.02}
                        value={params.mathModulationWeight}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            mathModulationWeight: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Suavizado Malla</span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {params.laplacianSmoothingSteps} / 30 pasos
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={30}
                        step={1}
                        value={params.laplacianSmoothingSteps}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            laplacianSmoothingSteps: parseInt(
                              e.target.value,
                              10
                            ),
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div className="col-span-2 flex flex-col gap-1.5 pt-1.5 border-t border-slate-800/70">
                      <div className="flex items-center justify-between gap-2">
                        <label className="flex items-center gap-1.5 text-[11px] font-semibold text-sky-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={params.detectStraightLines}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                detectStraightLines: e.target.checked,
                              }))
                            }
                            className="accent-sky-400 cursor-pointer"
                          />
                          Líneas Rectas (Sin Montículos)
                        </label>

                        <label className="flex items-center gap-1.5 text-[11px] font-semibold text-emerald-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={params.symmetricBackRules}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                symmetricBackRules: e.target.checked,
                              }))
                            }
                            className="accent-emerald-400 cursor-pointer"
                          />
                          Atrás Simétrico (+Prof.)
                        </label>
                      </div>

                      <div className="flex items-center justify-between">
                        <button
                          type="button"
                          onClick={() => setActiveTab('cutout')}
                          className="text-xs font-semibold text-sky-400 hover:text-sky-300 underline cursor-pointer whitespace-nowrap"
                        >
                          Quitar Marco / Recortar por Color →
                        </button>

                        <button
                          type="button"
                          onClick={() => setActiveTab('smoothing')}
                          className="text-xs font-semibold text-emerald-400 hover:text-emerald-300 underline cursor-pointer whitespace-nowrap"
                        >
                          Suavizado y Líneas Rectas →
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 1A: ELIMINAR MARCO EXTERIOR, ZONA PLANA O RECORTAR POR COLOR / TONO DE COLOR */}
              {activeTab === 'cutout' && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
                  {/* Column 1: Interactive 2D Eyedropper Preview + Cutout Mode Selector */}
                  <div className="lg:col-span-5 flex gap-3.5 items-center border-r border-slate-800/80 pr-4">
                    <div
                      title="Haz clic en cualquier punto de la imagen para capturar ese color/tono y recortarlo"
                      className="relative w-32 h-32 rounded-lg overflow-hidden border-2 border-sky-500/60 shrink-0 bg-slate-950 shadow-inner cursor-crosshair"
                    >
                      <canvas
                        ref={cutoutPreviewCanvasRef}
                        onClick={handlePreviewCanvasEyedropper}
                        className="w-full h-full object-cover"
                      />
                      <span className="absolute bottom-1 left-1 right-1 text-center text-[10px] font-semibold bg-black/80 text-sky-300 py-0.5 rounded pointer-events-none">
                        Clic = Gotero Color
                      </span>
                    </div>

                    <div className="flex flex-col justify-between h-full flex-1 min-w-0">
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-semibold text-slate-100">
                            Modo de Eliminación de Marco y Fondo
                          </span>
                          <span className="px-1.5 py-0.5 text-[10px] font-mono font-semibold bg-sky-500/15 border border-sky-500/40 text-sky-300 rounded">
                            −{telemetry.cutoutTrianglesRemoved.toLocaleString()}{' '}
                            tris
                          </span>
                        </div>

                        <div className="grid grid-cols-1 gap-1">
                          {(
                            Object.keys(
                              CUTOUT_MODE_LABELS
                            ) as BackgroundCutoutMode[]
                          ).map((modeKey) => (
                            <button
                              key={modeKey}
                              type="button"
                              onClick={() =>
                                setParams((p) => ({
                                  ...p,
                                  cutoutMode: modeKey,
                                }))
                              }
                              className={`px-2.5 py-1 rounded-md border text-left transition-colors cursor-pointer ${
                                params.cutoutMode === modeKey
                                  ? 'bg-sky-500/15 border-sky-400 text-slate-100'
                                  : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              <div className="flex items-center justify-between gap-2">
                                <span className="text-xs font-semibold truncate">
                                  {CUTOUT_MODE_LABELS[modeKey].name}
                                </span>
                                <span className="text-[10px] font-mono text-sky-300 shrink-0">
                                  {CUTOUT_MODE_LABELS[modeKey].tag}
                                </span>
                              </div>
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Column 2: Sliders for Flat Area Threshold, Outer Frame Margin, Color Tolerance, and Hue Tone Tolerance */}
                  <div className="lg:col-span-4 grid grid-cols-2 gap-x-4 gap-y-2.5 border-r border-slate-800/80 pr-4 items-center">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-300 font-medium">
                          Quitar Zona Plana
                        </span>
                        <span className="font-mono text-sky-300 tabular-nums">
                          {Math.round(params.cutoutFlatThreshold * 100)}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.0}
                        max={0.55}
                        step={0.01}
                        value={params.cutoutFlatThreshold}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            cutoutFlatThreshold: parseFloat(e.target.value),
                            cutoutMode:
                              p.cutoutMode === BackgroundCutoutMode.None
                                ? BackgroundCutoutMode.FlatReliefAndBorder
                                : p.cutoutMode,
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-300 font-medium">
                          Recorte Marco Ext.
                        </span>
                        <span className="font-mono text-sky-300 tabular-nums">
                          {Math.round(params.cutoutBorderMargin * 100)}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.0}
                        max={0.45}
                        step={0.01}
                        value={params.cutoutBorderMargin}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            cutoutBorderMargin: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">
                          Tolerancia Color RGB
                        </span>
                        <span className="font-mono text-amber-300 tabular-nums">
                          {Math.round(params.cutoutColorTolerance * 100)}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.03}
                        max={0.75}
                        step={0.01}
                        value={params.cutoutColorTolerance}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            cutoutColorTolerance: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-amber-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">
                          Tolerancia de Tono
                        </span>
                        <span className="font-mono text-amber-300 tabular-nums">
                          {Math.round(params.cutoutHueTolerance * 100)}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.02}
                        max={0.48}
                        step={0.01}
                        value={params.cutoutHueTolerance}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            cutoutHueTolerance: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-amber-400 cursor-pointer"
                      />
                    </div>
                  </div>

                  {/* Column 3: Color Picker, Quick Tone Swatches, Match by Hue Tone & Invert Cutout */}
                  <div className="lg:col-span-3 flex flex-col justify-between gap-2">
                    <div>
                      <div className="flex items-center justify-between gap-2 mb-1.5">
                        <label className="flex items-center gap-2 text-xs font-semibold text-slate-200 cursor-pointer">
                          <span>Color / Tono:</span>
                          <input
                            type="color"
                            value={params.cutoutTargetColorHex}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                cutoutMode:
                                  BackgroundCutoutMode.CustomColorAndHue,
                                cutoutTargetColorHex: e.target.value,
                              }))
                            }
                            className="w-6 h-6 rounded bg-transparent border border-slate-600 cursor-pointer"
                          />
                          <span className="font-mono text-[11px] text-sky-300">
                            {params.cutoutTargetColorHex.toUpperCase()}
                          </span>
                        </label>

                        <button
                          type="button"
                          onClick={() =>
                            setParams((p) => ({
                              ...p,
                              cutoutMode: BackgroundCutoutMode.AutoCornerColor,
                              cutoutTargetColorHex:
                                cnnBuffersRef.current.autoDetectedBgHex,
                            }))
                          }
                          className="px-2 py-0.5 text-[10px] font-semibold bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sky-300 rounded cursor-pointer whitespace-nowrap"
                        >
                          Auto Esquinas
                        </button>
                      </div>

                      <div className="flex items-center gap-1.5 mb-2">
                        {(
                          [
                            ['Fondo Oscuro', '#0c1220'],
                            ['Azul Marino', '#142346'],
                            ['Negro', '#050508'],
                            ['Dorado', '#eab308'],
                            ['Rojo', '#dc2626'],
                            ['Esmeralda', '#10b981'],
                            ['Blanco', '#f8fafc'],
                          ] as const
                        ).map(([label, hex]) => (
                          <button
                            key={hex}
                            type="button"
                            title={`Recortar tono ${label} (${hex})`}
                            onClick={() =>
                              setParams((p) => ({
                                ...p,
                                cutoutMode:
                                  BackgroundCutoutMode.CustomColorAndHue,
                                cutoutTargetColorHex: hex,
                              }))
                            }
                            className={`w-5 h-5 rounded-md border transition-transform cursor-pointer ${
                              params.cutoutTargetColorHex.toLowerCase() === hex
                                ? 'border-sky-400 scale-110 ring-1 ring-sky-400'
                                : 'border-white/25 hover:scale-105'
                            }`}
                            style={{ backgroundColor: hex }}
                          />
                        ))}
                      </div>
                    </div>

                    <div className="space-y-1.5 pt-1 border-t border-slate-800/80">
                      <div className="flex items-center justify-between gap-2">
                        <label className="flex items-center gap-1.5 text-xs font-semibold text-emerald-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={params.closeInternalHoles}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                closeInternalHoles: e.target.checked,
                              }))
                            }
                            className="accent-emerald-400"
                          />
                          <span>Cerrar Huecos Internos y Unir Zonas</span>
                        </label>
                        <span className="font-mono text-[11px] text-emerald-300">
                          {params.holeFillRadius} px
                        </span>
                      </div>

                      <input
                        type="range"
                        min={0}
                        max={12}
                        step={1}
                        value={params.holeFillRadius}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            closeInternalHoles: true,
                            holeFillRadius: parseInt(e.target.value, 10),
                          }))
                        }
                        className="w-full accent-emerald-400 cursor-pointer"
                      />

                      <div className="flex items-center justify-between gap-2">
                        <label className="flex items-center gap-1.5 text-[11px] text-slate-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={params.matchByHueTone}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                matchByHueTone: e.target.checked,
                              }))
                            }
                            className="accent-amber-400"
                          />
                          <span>Por Gama / Tono</span>
                        </label>

                        <label className="flex items-center gap-1.5 text-[11px] text-slate-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={params.invertCutout}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                invertCutout: e.target.checked,
                              }))
                            }
                            className="accent-sky-400"
                          />
                          <span>Invertir Recorte</span>
                        </label>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 1B: SUAVIZADO AVANZADO (HASTA 30 PASOS) Y UNIÓN DE PUNTOS CERCANOS (SOLDADURA ESPACIAL ZERO-GC) */}
              {activeTab === 'smoothing' && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
                  {/* Volume, outline and relief quality */}
                  <div className="lg:col-span-12 grid grid-cols-1 md:grid-cols-3 gap-4 pb-3 border-b border-slate-800/80">
                    <div title="Da grosor redondeado a cada parte según su anchura (brazos como tubos, torso más profundo).">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Volumen redondeado (inflado de silueta)</span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {(params.silhouetteInflation ?? 0.75).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1.5}
                        step={0.05}
                        value={params.silhouetteInflation ?? 0.75}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            silhouetteInflation: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-emerald-400 cursor-pointer"
                      />
                    </div>
                    <div title="Ajusta los vértices del borde al contorno real del dibujo y suaviza la línea.">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Contorno suave (anti dientes de sierra)</span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {(params.contourSnapStrength ?? 1.0).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={params.contourSnapStrength ?? 1.0}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            contourSnapStrength: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-emerald-400 cursor-pointer"
                      />
                    </div>
                    <div title="Quita grumos del relieve de la foto sin borrar los bordes entre placas.">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Suavizado de relieve (respeta bordes)</span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {(params.reliefEdgeSmoothing ?? 0.7).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={params.reliefEdgeSmoothing ?? 0.7}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            reliefEdgeSmoothing: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-emerald-400 cursor-pointer"
                      />
                    </div>
                  </div>
                  {/* Column 1: Smoothing Algorithm Selector + Presets */}
                  <div className="lg:col-span-4 flex flex-col justify-between border-r border-slate-800/80 pr-4">
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-semibold text-slate-100">
                          Algoritmo de Suavizado de Malla (0–30 Pasos)
                        </span>
                        <div className="flex items-center gap-1">
                          {[
                            ['Suave', 4],
                            ['Alto', 14],
                            ['Máx 30', 30],
                          ].map(([label, val]) => (
                            <button
                              key={label}
                              type="button"
                              onClick={() =>
                                setParams((p) => ({
                                  ...p,
                                  laplacianSmoothingSteps: val as number,
                                }))
                              }
                              className={`px-1.5 py-0.5 text-[10px] font-semibold rounded border cursor-pointer transition-colors ${
                                params.laplacianSmoothingSteps === val
                                  ? 'bg-sky-500/25 border-sky-400 text-sky-200'
                                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="grid grid-cols-1 gap-1.5">
                        {(
                          Object.keys(
                            SMOOTHING_ALGO_LABELS
                          ) as SmoothingAlgorithm[]
                        ).map((algoKey) => (
                          <button
                            key={algoKey}
                            type="button"
                            onClick={() =>
                              setParams((p) => ({
                                ...p,
                                smoothingAlgorithm: algoKey,
                              }))
                            }
                            className={`px-2.5 py-1.5 rounded-lg border text-left transition-colors cursor-pointer ${
                              params.smoothingAlgorithm === algoKey
                                ? 'bg-sky-500/15 border-sky-400 text-slate-100'
                                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold">
                                {SMOOTHING_ALGO_LABELS[algoKey].name}
                              </span>
                              <span className="text-[10px] font-mono text-sky-300">
                                {SMOOTHING_ALGO_LABELS[algoKey].tag}
                              </span>
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>

                    <p className="text-[11px] text-slate-400 mt-1.5 leading-snug">
                      {SMOOTHING_ALGO_LABELS[params.smoothingAlgorithm].desc}
                    </p>
                  </div>

                  {/* Column 2: Multi-Stage Smoothing Sliders (Steps 0-30, Lambda, Tensor Pre-Blur, Normal Smoothing) */}
                  <div className="lg:col-span-4 grid grid-cols-2 gap-x-4 gap-y-2.5 border-r border-slate-800/80 pr-4 items-center">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-300 font-medium">
                          Pasos de Suavizado
                        </span>
                        <span className="font-mono text-sky-300 font-semibold tabular-nums">
                          {params.laplacianSmoothingSteps} pasos
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={30}
                        step={1}
                        value={params.laplacianSmoothingSteps}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            laplacianSmoothingSteps: parseInt(
                              e.target.value,
                              10
                            ),
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">
                          Intensidad Paso (λ)
                        </span>
                        <span className="font-mono text-sky-300 tabular-nums">
                          {params.smoothingLambda.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.15}
                        max={0.9}
                        step={0.02}
                        value={params.smoothingLambda}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            smoothingLambda: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">
                          Suavizado Normales
                        </span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {params.normalSmoothingSteps} pasadas
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={10}
                        step={1}
                        value={params.normalSmoothingSteps}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            normalSmoothingSteps: parseInt(e.target.value, 10),
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">
                          Pre-Filtro Tensor
                        </span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {params.tensorBlurPasses} pasadas
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={8}
                        step={1}
                        value={params.tensorBlurPasses}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            tensorBlurPasses: parseInt(e.target.value, 10),
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    {/* Straight-Line Anisotropic Lock & Anti-Mound Facet Planarity inside Smoothing Tab */}
                    <div className="col-span-2 bg-slate-900/85 border border-slate-800 rounded-lg p-2 space-y-1.5">
                      <div className="flex items-center justify-between gap-2">
                        <label className="flex items-center gap-1.5 text-[11px] font-semibold text-sky-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={params.detectStraightLines}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                detectStraightLines: e.target.checked,
                              }))
                            }
                            className="accent-sky-400 cursor-pointer"
                          />
                          Preservar Líneas Rectas (Sin Curvar / Sin Montículos)
                        </label>
                        <label className="flex items-center gap-1 text-[10px] font-semibold text-emerald-300 cursor-pointer whitespace-nowrap">
                          <input
                            type="checkbox"
                            checked={params.symmetricBackRules}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                symmetricBackRules: e.target.checked,
                              }))
                            }
                            className="accent-emerald-400 cursor-pointer"
                          />
                          Aplicar Simétrico Atrás
                        </label>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <div className="flex justify-between text-[10px] mb-0.5">
                            <span className="text-slate-400">
                              Rigidez Líneas Rectas
                            </span>
                            <span className="font-mono text-sky-300">
                              {(params.straightLineSensitivity * 100).toFixed(0)}%
                            </span>
                          </div>
                          <input
                            type="range"
                            min={0.0}
                            max={1.0}
                            step={0.02}
                            disabled={!params.detectStraightLines}
                            value={params.straightLineSensitivity}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                straightLineSensitivity: parseFloat(
                                  e.target.value
                                ),
                              }))
                            }
                            className="w-full accent-sky-400 cursor-pointer disabled:opacity-40"
                          />
                        </div>

                        <div>
                          <div className="flex justify-between text-[10px] mb-0.5">
                            <span className="text-slate-400">
                              Aplanar Montículos
                            </span>
                            <span className="font-mono text-emerald-300">
                              {(params.flatFacetPlanarity * 100).toFixed(0)}%
                            </span>
                          </div>
                          <input
                            type="range"
                            min={0.0}
                            max={1.0}
                            step={0.02}
                            disabled={!params.detectStraightLines}
                            value={params.flatFacetPlanarity}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                flatFacetPlanarity: parseFloat(e.target.value),
                              }))
                            }
                            className="w-full accent-emerald-400 cursor-pointer disabled:opacity-40"
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Column 3: Unión de Puntos Cercanos (Close-Point Vertex Welding + Seam Stitching + Mesh Resolution) */}
                  <div className="lg:col-span-4 flex flex-col justify-between gap-2">
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold text-slate-100">
                          Unión de Puntos Cercanos (Soldadura d² ≤ ε²)
                        </span>
                        <span className="px-2 py-0.5 text-[10px] font-mono font-semibold bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 rounded">
                          {telemetry.weldedVerticesCount.toLocaleString()}{' '}
                          unidos
                        </span>
                      </div>

                      <div className="mb-2">
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-400">
                            Radio de Unión de Vértices (ε)
                          </span>
                          <span className="font-mono text-emerald-300 tabular-nums">
                            {params.weldTolerance.toFixed(3)} u
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.0}
                          max={0.08}
                          step={0.002}
                          value={params.weldTolerance}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              weldTolerance: parseFloat(e.target.value),
                            }))
                          }
                          className="w-full accent-emerald-400 cursor-pointer"
                        />
                        <div className="flex items-center justify-between gap-1 mt-1">
                          {[
                            ['Desactivado (0)', 0.0],
                            ['Costuras (0.012)', 0.012],
                            ['Fusión Media (0.030)', 0.03],
                            ['Agresiva (0.065)', 0.065],
                          ].map(([label, tol]) => (
                            <button
                              key={label}
                              type="button"
                              onClick={() =>
                                setParams((p) => ({
                                  ...p,
                                  weldTolerance: tol as number,
                                }))
                              }
                              className={`px-1.5 py-0.5 text-[10px] font-medium rounded border cursor-pointer transition-colors whitespace-nowrap ${
                                Math.abs(
                                  params.weldTolerance - (tol as number)
                                ) < 0.001
                                  ? 'bg-emerald-500/20 border-emerald-400 text-emerald-200 font-semibold'
                                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 items-center pt-1.5 border-t border-slate-800/80">
                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-slate-400">
                            Densidad Malla
                          </span>
                          <span className="font-mono text-slate-200 tabular-nums">
                            {params.gridResolution}×{params.gridResolution}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={48}
                          max={128}
                          step={8}
                          value={params.gridResolution}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              gridResolution: parseInt(e.target.value, 10),
                            }))
                          }
                          className="w-full accent-sky-400 cursor-pointer"
                        />
                      </div>

                      <div className="space-y-1">
                        <label className="flex items-center gap-1.5 text-[11px] text-slate-300 cursor-pointer whitespace-nowrap">
                          <input
                            type="checkbox"
                            checked={params.autoStitchSeams}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                autoStitchSeams: e.target.checked,
                              }))
                            }
                            className="accent-emerald-400"
                          />
                          Unir Polos y Costuras 360°
                        </label>

                        <label className="flex items-center gap-1.5 text-[11px] text-slate-300 cursor-pointer whitespace-nowrap">
                          <input
                            type="checkbox"
                            checked={params.wireframeOverlay}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                wireframeOverlay: e.target.checked,
                              }))
                            }
                            className="accent-sky-400"
                          />
                          Ver Malla Alámbrica
                        </label>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 2: HERRAMIENTA DE RECONSTRUCCIÓN BASADA EN LAS NORMALES (POISSON SOLVER) */}
              {activeTab === 'normals' && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
                  <div className="lg:col-span-4 flex flex-col justify-between border-r border-slate-800/80 pr-4">
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-semibold text-slate-100">
                          Reconstrucción de Superficie desde Normales
                        </span>
                        <button
                          type="button"
                          onClick={() => {
                            handleSelectPreset('tangent_normal_shield');
                            setFeatureView(CNNFeatureView.NormalTensor);
                          }}
                          className="px-2 py-0.5 text-[11px] font-medium bg-indigo-500/20 border border-indigo-400/50 text-indigo-200 rounded hover:bg-indigo-500/30 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                        >
                          Probar Normal Map RGB
                        </button>
                      </div>

                      <div className="grid grid-cols-1 gap-1.5">
                        {(
                          Object.keys(
                            NORMAL_MODE_LABELS
                          ) as NormalReconstructionMode[]
                        ).map((modeKey) => (
                          <button
                            key={modeKey}
                            type="button"
                            onClick={() =>
                              setParams((p) => ({
                                ...p,
                                normalReconMode: modeKey,
                              }))
                            }
                            className={`px-2.5 py-1.5 rounded-lg border text-left transition-colors cursor-pointer ${
                              params.normalReconMode === modeKey
                                ? 'bg-indigo-500/15 border-indigo-400 text-slate-100'
                                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold">
                                {NORMAL_MODE_LABELS[modeKey].name}
                              </span>
                              <span className="text-[10px] font-mono text-indigo-300">
                                {NORMAL_MODE_LABELS[modeKey].equation.split('=')[0]}
                              </span>
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>

                    <p className="text-[11px] text-slate-400 mt-1.5 leading-snug">
                      {NORMAL_MODE_LABELS[params.normalReconMode].desc}
                    </p>
                  </div>

                  <div className="lg:col-span-4 grid grid-cols-2 gap-x-4 gap-y-2.5 border-r border-slate-800/80 pr-4 items-center">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Iteraciones SOR</span>
                        <span className="font-mono text-indigo-300 tabular-nums">
                          {params.poissonIterations} it
                        </span>
                      </div>
                      <input
                        type="range"
                        min={10}
                        max={110}
                        step={5}
                        value={params.poissonIterations}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            poissonIterations: parseInt(e.target.value, 10),
                          }))
                        }
                        className="w-full accent-indigo-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Pendiente (p, q)</span>
                        <span className="font-mono text-indigo-300 tabular-nums">
                          {params.gradientSlopeGain.toFixed(2)}×
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.4}
                        max={3.5}
                        step={0.05}
                        value={params.gradientSlopeGain}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            gradientSlopeGain: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-indigo-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Peso Normales (γ)</span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {Math.round(params.normalIntegrationWeight * 100)}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.0}
                        max={1.0}
                        step={0.05}
                        value={params.normalIntegrationWeight}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            normalIntegrationWeight: parseFloat(
                              e.target.value
                            ),
                          }))
                        }
                        className="w-full accent-indigo-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Regularización λ</span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {params.screenedLambda.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.01}
                        max={0.45}
                        step={0.01}
                        value={params.screenedLambda}
                        onChange={(e) =>
                          setParams((p) => ({
                            ...p,
                            screenedLambda: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-indigo-400 cursor-pointer"
                      />
                    </div>
                  </div>

                  <div className="lg:col-span-4 flex flex-col justify-between gap-2">
                    <div>
                      <span className="text-xs font-semibold text-slate-200 block mb-2">
                        Diagnóstico Vectorial y Pincel de Normales 3D
                      </span>
                      <div className="space-y-1.5">
                        <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={params.showNormalVectors3D}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                showNormalVectors3D: e.target.checked,
                              }))
                            }
                            className="accent-indigo-400"
                          />
                          <span>
                            Mostrar Agujas Vectoriales 3D (1,600 InstancedMesh)
                          </span>
                        </label>

                        <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={params.renderNormalShading}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                renderNormalShading: e.target.checked,
                              }))
                            }
                            className="accent-indigo-400"
                          />
                          <span>
                            Proyectar Mapa de Normales RGB (Nx, Ny, Nz) en Malla
                          </span>
                        </label>

                        <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={params.invertNormalGreenY}
                            onChange={(e) =>
                              setParams((p) => ({
                                ...p,
                                invertNormalGreenY: e.target.checked,
                              }))
                            }
                            className="accent-indigo-400"
                          />
                          <span>
                            Invertir Canal Verde Y (Convención DirectX −Y)
                          </span>
                        </label>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 items-center pt-2 border-t border-slate-800/80">
                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-slate-400">Radio Pincel</span>
                          <span className="font-mono text-indigo-300 tabular-nums">
                            {brushRadius.toFixed(2)}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.15}
                          max={0.95}
                          step={0.02}
                          value={brushRadius}
                          onChange={(e) =>
                            setBrushRadius(parseFloat(e.target.value))
                          }
                          className="w-full accent-indigo-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-slate-400">Inclinación n</span>
                          <span className="font-mono text-indigo-300 tabular-nums">
                            {brushStrength.toFixed(2)}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.15}
                          max={1.5}
                          step={0.05}
                          value={brushStrength}
                          onChange={(e) =>
                            setBrushStrength(parseFloat(e.target.value))
                          }
                          className="w-full accent-indigo-400 cursor-pointer"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 3: EDICIÓN DE MALLA Y TEXTURA POR ZONAS DIFERENTES */}
              {activeTab === 'zones' && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
                  <div className="lg:col-span-4 flex flex-col justify-between border-r border-slate-800/80 pr-4">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-semibold text-slate-200">
                          Zonas Semánticas por Divergencia de Normales
                        </span>
                        <button
                          type="button"
                          onClick={resetAllZonesToOriginalPhoto}
                          className="text-[11px] text-sky-400 hover:text-sky-300 underline cursor-pointer whitespace-nowrap"
                        >
                          Usar 100% Foto Original
                        </button>
                      </div>
                      <div className="grid grid-cols-2 gap-1.5">
                        {zones.map((z, idx) => (
                          <button
                            key={z.id}
                            type="button"
                            onClick={() => setSelectedZoneIdx(idx)}
                            className={`flex items-center gap-2 px-2.5 py-2 rounded-lg border text-left transition-colors cursor-pointer ${
                              selectedZoneIdx === idx
                                ? 'bg-slate-800/90 border-sky-400 text-slate-100'
                                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                            }`}
                          >
                            <span
                              className="w-3 h-3 rounded-sm shrink-0 border border-white/20"
                              style={{ backgroundColor: z.colorHex }}
                            />
                            <span className="text-xs font-medium truncate">
                              {z.name.split('·')[1] || z.name}
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>

                    <p className="text-[11px] text-slate-400 mt-2">
                      {activeZone.description}
                    </p>
                  </div>

                  <div className="lg:col-span-4 grid grid-cols-2 gap-x-4 gap-y-2.5 border-r border-slate-800/80 pr-4 items-center">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Extrusión Zonal</span>
                        <span className="font-mono text-sky-300 tabular-nums">
                          {activeZone.extrusion >= 0 ? '+' : ''}
                          {activeZone.extrusion.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={-0.5}
                        max={0.9}
                        step={0.02}
                        value={activeZone.extrusion}
                        onChange={(e) =>
                          updateSelectedZone(
                            'extrusion',
                            parseFloat(e.target.value)
                          )
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Escala Radial</span>
                        <span className="font-mono text-sky-300 tabular-nums">
                          {activeZone.radialScale.toFixed(2)}×
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.75}
                        max={1.35}
                        step={0.01}
                        value={activeZone.radialScale}
                        onChange={(e) =>
                          updateSelectedZone(
                            'radialScale',
                            parseFloat(e.target.value)
                          )
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Frecuencia Micro</span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {activeZone.microFrequency} Hz
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={20}
                        step={1}
                        value={activeZone.microFrequency}
                        onChange={(e) =>
                          updateSelectedZone(
                            'microFrequency',
                            parseInt(e.target.value, 10)
                          )
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Relieve Micro</span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {activeZone.microAmplitude.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={0.25}
                        step={0.01}
                        value={activeZone.microAmplitude}
                        onChange={(e) =>
                          updateSelectedZone(
                            'microAmplitude',
                            parseFloat(e.target.value)
                          )
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>
                  </div>

                  <div className="lg:col-span-4 flex flex-col justify-between gap-2">
                    <div className="flex items-center gap-3">
                      <label className="flex items-center gap-2 text-xs text-slate-300 shrink-0 cursor-pointer">
                        <span>Tinte:</span>
                        <input
                          type="color"
                          value={activeZone.colorHex}
                          onChange={(e) =>
                            updateSelectedZone('colorHex', e.target.value)
                          }
                          className="w-7 h-7 rounded bg-transparent border border-slate-700 cursor-pointer"
                        />
                      </label>

                      <div className="flex-1 min-w-0">
                        <select
                          value={activeZone.pattern}
                          onChange={(e) =>
                            updateSelectedZone(
                              'pattern',
                              e.target.value as TexturePattern
                            )
                          }
                          className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-sky-400"
                        >
                          {(
                            Object.keys(PATTERN_LABELS) as TexturePattern[]
                          ).map((pat) => (
                            <option key={pat} value={pat}>
                              {PATTERN_LABELS[pat]}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-3 items-center pt-1">
                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-slate-400">Mezcla Tinte</span>
                          <span className="font-mono text-sky-300 tabular-nums">
                            {Math.round(activeZone.zoneTintMix * 100)}%
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.0}
                          max={1.0}
                          step={0.05}
                          value={activeZone.zoneTintMix}
                          onChange={(e) =>
                            updateSelectedZone(
                              'zoneTintMix',
                              parseFloat(e.target.value)
                            )
                          }
                          className="w-full accent-sky-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-slate-400">Rugosidad</span>
                          <span className="font-mono text-slate-200 tabular-nums">
                            {activeZone.roughness.toFixed(2)}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.05}
                          max={1.0}
                          step={0.02}
                          value={activeZone.roughness}
                          onChange={(e) =>
                            updateSelectedZone(
                              'roughness',
                              parseFloat(e.target.value)
                            )
                          }
                          className="w-full accent-sky-400 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-slate-400">Metalicidad</span>
                          <span className="font-mono text-slate-200 tabular-nums">
                            {activeZone.metalness.toFixed(2)}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.0}
                          max={1.0}
                          step={0.02}
                          value={activeZone.metalness}
                          onChange={(e) =>
                            updateSelectedZone(
                              'metalness',
                              parseFloat(e.target.value)
                            )
                          }
                          className="w-full accent-sky-400 cursor-pointer"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 4: ILUMINACIÓN DINÁMICA REALISTA PBR */}
              {activeTab === 'lighting' && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-center">
                  <div className="lg:col-span-4 border-r border-slate-800/80 pr-4">
                    <span className="text-xs font-semibold text-slate-200 block mb-2">
                      Atmósfera e Iluminación de Estudio PBR
                    </span>
                    <div className="grid grid-cols-2 gap-1.5">
                      {(
                        [
                          [
                            LightingEnvironmentPreset.StudioWarm,
                            'Estudio Cálido',
                          ],
                          [
                            LightingEnvironmentPreset.CyberLabCool,
                            'Laboratorio Frío',
                          ],
                          [
                            LightingEnvironmentPreset.GoldenSunset,
                            'Hora Dorada',
                          ],
                          [
                            LightingEnvironmentPreset.DramaticRim,
                            'Contraluz Dramático',
                          ],
                        ] as const
                      ).map(([presetKey, label]) => (
                        <button
                          key={presetKey}
                          type="button"
                          onClick={() =>
                            setLighting((l) => ({ ...l, preset: presetKey }))
                          }
                          className={`px-2.5 py-1.5 text-xs rounded-lg border text-left transition-colors whitespace-nowrap truncate cursor-pointer ${
                            lighting.preset === presetKey
                              ? 'bg-sky-500/15 border-sky-500/50 text-sky-300 font-semibold'
                              : 'bg-slate-900/70 border-slate-800 text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="lg:col-span-4 grid grid-cols-2 gap-x-4 gap-y-2.5 border-r border-slate-800/80 pr-4">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Azimut Solar</span>
                        <span className="font-mono text-sky-300 tabular-nums">
                          {lighting.azimuthDeg}°
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={360}
                        step={1}
                        value={lighting.azimuthDeg}
                        onChange={(e) =>
                          setLighting((l) => ({
                            ...l,
                            azimuthDeg: parseInt(e.target.value, 10),
                            autoRotateLight: false,
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Elevación Cenital</span>
                        <span className="font-mono text-sky-300 tabular-nums">
                          {lighting.elevationDeg}°
                        </span>
                      </div>
                      <input
                        type="range"
                        min={10}
                        max={85}
                        step={1}
                        value={lighting.elevationDeg}
                        onChange={(e) =>
                          setLighting((l) => ({
                            ...l,
                            elevationDeg: parseInt(e.target.value, 10),
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Luz Principal</span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {lighting.keyIntensity.toFixed(1)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.4}
                        max={4.5}
                        step={0.1}
                        value={lighting.keyIntensity}
                        onChange={(e) =>
                          setLighting((l) => ({
                            ...l,
                            keyIntensity: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Exposición ACES</span>
                        <span className="font-mono text-slate-200 tabular-nums">
                          {lighting.exposure.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.5}
                        max={1.8}
                        step={0.02}
                        value={lighting.exposure}
                        onChange={(e) =>
                          setLighting((l) => ({
                            ...l,
                            exposure: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>
                  </div>

                  <div className="lg:col-span-4 flex flex-col justify-between gap-3">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">
                          Intensidad Contraluz (Rim Light)
                        </span>
                        <span className="font-mono text-amber-300 tabular-nums">
                          {lighting.rimIntensity.toFixed(1)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.0}
                        max={4.0}
                        step={0.1}
                        value={lighting.rimIntensity}
                        onChange={(e) =>
                          setLighting((l) => ({
                            ...l,
                            rimIntensity: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-amber-400 cursor-pointer"
                      />
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-800/80">
                      <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer whitespace-nowrap">
                        <input
                          type="checkbox"
                          checked={lighting.autoRotateModel}
                          onChange={(e) =>
                            setLighting((l) => ({
                              ...l,
                              autoRotateModel: e.target.checked,
                            }))
                          }
                          className="accent-sky-400"
                        />
                        Animar Vista
                      </label>

                      <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer whitespace-nowrap">
                        <input
                          type="checkbox"
                          checked={lighting.autoRotateLight}
                          onChange={(e) =>
                            setLighting((l) => ({
                              ...l,
                              autoRotateLight: e.target.checked,
                            }))
                          }
                          className="accent-sky-400"
                        />
                        Orbitar Luz
                      </label>

                      <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer whitespace-nowrap">
                        <input
                          type="checkbox"
                          checked={params.showTurntablePedestal}
                          onChange={(e) =>
                            setParams((p) => ({
                              ...p,
                              showTurntablePedestal: e.target.checked,
                            }))
                          }
                          className="accent-sky-400"
                        />
                        Pedestal
                      </label>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 5: CENTRO DE DESCARGAS (RGB, NORMALES, 3D .GLB/.OBJ Y TODOS LOS MODELOS DE LA APLICACIÓN) */}
              {activeTab === 'downloads' && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
                  {/* Column 1: Individual Downloads for Active Model (RGB, Normals, Depth, Render, GLB, OBJ) */}
                  <div className="lg:col-span-4 flex flex-col justify-between border-r border-slate-800/80 pr-4">
                    <div>
                      <span className="text-xs font-semibold text-slate-100 block mb-2">
                        1. Descargas del Modelo Activo ({activeBaseName})
                      </span>
                      <div className="grid grid-cols-2 gap-1.5">
                        <button
                          type="button"
                          onClick={handleDownloadActiveRGB}
                          className="flex items-center gap-1.5 px-2.5 py-2 text-xs font-medium bg-slate-900 hover:bg-slate-800 border border-slate-700/80 rounded-lg text-slate-200 transition-colors whitespace-nowrap truncate cursor-pointer"
                        >
                          <ImageIcon className="w-3.5 h-3.5 text-sky-400 shrink-0" />
                          Textura RGB (.PNG)
                        </button>

                        <button
                          type="button"
                          onClick={handleDownloadActiveNormals}
                          className="flex items-center gap-1.5 px-2.5 py-2 text-xs font-medium bg-slate-900 hover:bg-slate-800 border border-slate-700/80 rounded-lg text-slate-200 transition-colors whitespace-nowrap truncate cursor-pointer"
                        >
                          <Compass className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                          Mapa Normales (.PNG)
                        </button>

                        <button
                          type="button"
                          onClick={handleDownloadActiveDepth}
                          className="flex items-center gap-1.5 px-2.5 py-2 text-xs font-medium bg-slate-900 hover:bg-slate-800 border border-slate-700/80 rounded-lg text-slate-200 transition-colors whitespace-nowrap truncate cursor-pointer"
                        >
                          <Layers className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                          Altura Poisson (.PNG)
                        </button>

                        <button
                          type="button"
                          onClick={() =>
                            viewportRef.current?.captureViewportPNG(
                              `${activeBaseName}_render_3d.png`
                            )
                          }
                          className="flex items-center gap-1.5 px-2.5 py-2 text-xs font-medium bg-slate-900 hover:bg-slate-800 border border-slate-700/80 rounded-lg text-slate-200 transition-colors whitespace-nowrap truncate cursor-pointer"
                        >
                          <Camera className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                          Captura 3D (.PNG)
                        </button>

                        <button
                          type="button"
                          onClick={() =>
                            viewportRef.current?.exportGLB(
                              `${activeBaseName}_modelo_3d.glb`
                            )
                          }
                          className="flex items-center gap-1.5 px-2.5 py-2 text-xs font-semibold bg-sky-500/20 hover:bg-sky-500/30 border border-sky-400/50 rounded-lg text-sky-200 transition-colors whitespace-nowrap truncate cursor-pointer"
                        >
                          <FileBox className="w-3.5 h-3.5 text-sky-300 shrink-0" />
                          Modelo 3D (.GLB)
                        </button>

                        <button
                          type="button"
                          onClick={() =>
                            viewportRef.current?.exportOBJ(
                              `${activeBaseName}_malla_3d.obj`
                            )
                          }
                          className="flex items-center gap-1.5 px-2.5 py-2 text-xs font-semibold bg-indigo-500/20 hover:bg-indigo-500/30 border border-indigo-400/50 rounded-lg text-indigo-200 transition-colors whitespace-nowrap truncate cursor-pointer"
                        >
                          <FileBox className="w-3.5 h-3.5 text-indigo-300 shrink-0" />
                          Malla CAD (.OBJ)
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Column 2: All 6 Application Models Individual Quick Download Matrix */}
                  <div className="lg:col-span-5 flex flex-col justify-between border-r border-slate-800/80 pr-4">
                    <div>
                      <span className="text-xs font-semibold text-slate-100 block mb-1.5">
                        2. Catálogo de Todos los Modelos de la Aplicación (RGB · Normales · 3D)
                      </span>
                      <div className="max-h-32 overflow-y-auto space-y-1 pr-1">
                        {PRESET_REFERENCE_IMAGES.map((preset, idx) => (
                          <div
                            key={preset.id}
                            className="flex items-center justify-between gap-2 bg-slate-900/75 border border-slate-800/90 rounded-lg px-2.5 py-1.5"
                          >
                            <button
                              type="button"
                              onClick={() => handleSelectPreset(preset.id)}
                              className="text-xs font-medium text-slate-200 hover:text-sky-300 truncate text-left cursor-pointer"
                            >
                              0{idx + 1}. {preset.title}
                            </button>
                            <div className="flex items-center gap-1 shrink-0">
                              <button
                                type="button"
                                onClick={() =>
                                  handleDownloadCatalogItem(preset.id, 'rgb')
                                }
                                className="px-2 py-0.5 text-[10px] font-semibold bg-slate-800 hover:bg-slate-700 text-sky-300 rounded cursor-pointer whitespace-nowrap"
                              >
                                RGB
                              </button>
                              <button
                                type="button"
                                onClick={() =>
                                  handleDownloadCatalogItem(preset.id, 'normal')
                                }
                                className="px-2 py-0.5 text-[10px] font-semibold bg-slate-800 hover:bg-slate-700 text-indigo-300 rounded cursor-pointer whitespace-nowrap"
                              >
                                Normales
                              </button>
                              <button
                                type="button"
                                onClick={() =>
                                  handleDownloadCatalogItem(preset.id, 'glb')
                                }
                                className="px-2 py-0.5 text-[10px] font-semibold bg-sky-500 hover:bg-sky-400 text-slate-950 rounded cursor-pointer whitespace-nowrap"
                              >
                                3D .GLB
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Column 3: Batch ZIP Bundles */}
                  <div className="lg:col-span-3 flex flex-col justify-between gap-2">
                    <div>
                      <span className="text-xs font-semibold text-slate-100 block mb-1.5">
                        3. Paquetes Completos (.ZIP)
                      </span>
                      <p className="text-[11px] text-slate-400 leading-snug mb-2.5">
                        Descarga en un solo archivo .ZIP las texturas RGB, mapas
                        de normales, altura de Poisson y mallas 3D (.GLB y .OBJ).
                      </p>
                    </div>

                    <div className="flex flex-col gap-2">
                      <button
                        type="button"
                        onClick={() =>
                          viewportRef.current?.exportActivePackZip(
                            activeBaseName
                          )
                        }
                        className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs font-semibold bg-slate-800 hover:bg-slate-700 border border-slate-600 text-slate-100 rounded-lg transition-colors whitespace-nowrap cursor-pointer"
                      >
                        <Download className="w-3.5 h-3.5 text-sky-400" />
                        Pack Modelo Activo (.ZIP)
                      </button>

                      <button
                        type="button"
                        onClick={handleDownloadAllModelsZip}
                        disabled={isExportingBatch}
                        className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs font-semibold bg-sky-400 hover:bg-sky-300 disabled:opacity-60 text-slate-950 rounded-lg transition-colors whitespace-nowrap cursor-pointer"
                      >
                        <Archive className="w-3.5 h-3.5" />
                        {isExportingBatch
                          ? 'Generando 6 Modelos...'
                          : 'Todos los 6 Modelos (.ZIP)'}
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
