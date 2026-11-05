export enum MathTopology {
  MonolithRelief = 'monolith_relief',
  SphericalHarmonics = 'spherical_harmonics',
  AxialRevolution = 'axial_revolution',
  ImplicitGyroid = 'implicit_gyroid',
  RiemannianTorus = 'riemannian_torus',
}

export enum NormalReconstructionMode {
  HybridScreenedPoisson = 'hybrid_screened_poisson',
  PurePoissonIntegration = 'pure_poisson_integration',
  DirectNormalMapRGB = 'direct_normal_map_rgb',
}

export enum SmoothingAlgorithm {
  TaubinVolumePreserving = 'taubin_volume_preserving',
  LaplacianIsotropic = 'laplacian_isotropic',
  BilateralEdgePreserving = 'bilateral_edge_preserving',
}

export enum BackgroundCutoutMode {
  None = 'none',
  FlatReliefAndBorder = 'flat_relief_border',
  AutoCornerColor = 'auto_corner_color',
  CustomColorAndHue = 'custom_color_hue',
}

export enum BackImageSourceMode {
  SymmetricRelief = 'symmetric_relief',
  EngravedBackplate = 'engraved_backplate',
  SmoothCurvedBack = 'smooth_curved_back',
  CustomBackUpload = 'custom_back_upload',
}

export enum SideImageSourceMode {
  AutoEdgeExtension = 'auto_edge_extension',
  MilledCoinEdge = 'milled_coin_edge',
  FiligreeSeamBand = 'filigree_seam_band',
  CustomSideUpload = 'custom_side_upload',
}

export enum PresetBaseShape {
  None = 'none',
  HumanoidAPose = 'humanoid_a_pose',
  Sphere3D = 'sphere_3d',
  QuadrupedAnimal = 'quadruped_animal',
  AnatomicalBust = 'anatomical_bust',
  HeroicCuirass = 'heroic_cuirass',
  CylindricalTotem = 'cylindrical_totem',
}

export enum ShapeSilhouetteMode {
  AdaptiveImageToShape = 'adaptive_image_to_shape',
  StrictShapeSilhouette = 'strict_shape_silhouette',
  UnionImageAndShape = 'union_image_and_shape',
}

export enum LimbUnionMode {
  TubularCylindersFull = 'tubular_cylinders_full',
  SmoothCurvedUnions = 'smooth_curved_unions',
  AnatomicalArticulated = 'anatomical_articulated',
  SlimBasRelief = 'slim_bas_relief',
}

export enum LateralJointProfile {
  SeamlessSmoothUnion = 'seamless_smooth_union',
  TubularCylinderFull = 'tubular_cylinder_full',
  Curved90Fillet = 'curved_90_fillet',
  FlatStraight = 'flat_straight',
  RightAngle90 = 'right_angle_90',
  ChamferBevel = 'chamfer_bevel',
  ConcaveCove90 = 'concave_cove_90',
}

export enum CNNFeatureView {
  RGBInput = 'rgb_input',
  NormalTensor = 'normal_tensor',
  PoissonSurface = 'poisson_surface',
  DivergenceCurvature = 'divergence_curvature',
  ZoneSegmentation = 'zone_segmentation',
  StraightLines = 'straight_lines',
}

export enum TexturePattern {
  OriginalPhoto = 'original_photo',
  Solid = 'solid',
  CarbonFiber = 'carbon_fiber',
  VeinedMarble = 'veined_marble',
  BrushedMetal = 'brushed_metal',
  HexScales = 'hex_scales',
  NeuralCircuit = 'neural_circuit',
}

export enum ViewportToolMode {
  Orbit = 'orbit',
  SculptNormals = 'sculpt_normals',
  SculptInflate = 'sculpt_inflate',
  SculptDeflate = 'sculpt_deflate',
  SculptSmooth = 'sculpt_smooth',
  PaintZone = 'paint_zone',
}

export enum LightingEnvironmentPreset {
  StudioWarm = 'studio_warm',
  CyberLabCool = 'cyber_lab_cool',
  GoldenSunset = 'golden_sunset',
  DramaticRim = 'dramatic_rim',
}

export interface ZoneConfig {
  id: number;
  name: string;
  description: string;
  extrusion: number; // -0.6 to +1.0
  microFrequency: number; // 0 to 16
  microAmplitude: number; // 0 to 0.35
  radialScale: number; // 0.7 to 1.35
  colorHex: string;
  zoneTintMix: number; // 0.0 = 100% original image texture, 1.0 = 100% zone color tint
  pattern: TexturePattern;
  roughness: number; // 0.05 to 1.0
  metalness: number; // 0.0 to 1.0
}

export interface ReconstructionParams {
  topology: MathTopology;
  // Preset 3D Base Shape Molding & Silhouette Conformity
  presetBaseShape: PresetBaseShape;
  shapeSilhouetteMode: ShapeSilhouetteMode;
  autoFitMoldToSilhouette: boolean; // Automatically fits the 3D mold height, width, center & row spans to the image's active silhouette
  silhouetteConformity: number; // 0.0 to 1.0 forces the 3D volume and union zone to match the drawing's exact silhouette
  limbUnionMode: LimbUnionMode; // TubularCylindersFull, SmoothCurvedUnions, AnatomicalArticulated, SlimBasRelief
  limbUnionCurvature: number; // 0.05 to 1.0 curvature radius of unions between body parts (tubes/cylinders)
  reliefInflationSuppression: number; // 0.0 to 1.0 suppresses initial volume ballooning so photo relief carves cleanly without bloating
  poseArmAngleDeg: number; // 15° to 90° adjustable arm / pose angle (15° low A-pose, 35° standard A-pose, 90° full T-pose)
  autoFitLimbSizes: boolean; // Automatically detects and adapts arm & leg length and thickness from the image silhouette
  armLengthScale: number; // 0.45 to 1.75 scale factor for arm length / reach
  armThicknessScale: number; // 0.45 to 1.75 scale factor for arm thickness / caliber
  legLengthScale: number; // 0.45 to 1.75 scale factor for leg length / extension
  legThicknessScale: number; // 0.45 to 1.75 scale factor for leg thickness / caliber
  shapeMoldStrength: number; // 0.0 to 1.5 volumetric depth & curvature contribution of the preset 3D shape
  shapeWrapUV: number; // 0.0 to 1.0 conformal UV wrapping of the image around the 3D shape curvature
  shapeScaleX: number; // 0.4 to 1.8 explicit Width (Anchura) scale of the silhouette/mold
  shapeScaleY: number; // 0.4 to 1.8 explicit Height (Altura) scale of the silhouette/mold
  shapeAspectRatio: number; // 0.5 to 1.6 explicit Width/Height Proportion (Proporción)
  shapeOffsetX: number; // -0.35 to +0.35 horizontal center offset
  shapeOffsetY: number; // -0.35 to +0.35 vertical center offset
  seamlessBorderSmoothing: number; // 0.0 to 1.0 makes the border union between Front & Back disappear via cross-shell smoothing
  enableSkeleton?: boolean; // build a skeleton + skin weights on the human body template
  rigAnimation?: string; // 'none' | 'idle' | 'walk' | 'run' | 'wave'
  rigAnimationSpeed?: number; // playback speed multiplier
  rigShowBones?: boolean; // draw the bones in the viewport
  humanBodyTemplate?: boolean; // fit a closed human body to the silhouette (Humanoid A-pose mold)
  humanBodyDepth?: number; // 0.2..1.8 body depth multiplier for the human template
  humanReliefDetail?: number; // 0..1 photo relief added on top of the human template
  silhouetteInflation?: number; // 0..1.5 rounded volume from the outline (0.75 ≈ elliptical human cross-section)
  contourSnapStrength?: number; // 0..1 how strongly border vertices follow the real outline (anti saw-tooth)
  reliefEdgeSmoothing?: number; // 0..1 edge-aware smoothing of the photo relief (removes lumps, keeps plate edges)
  seamlessBorderWidth: number; // 2 to 18 grid cells width of the smooth border transition zone
  normalReconMode: NormalReconstructionMode;
  poissonIterations: number; // 10 to 120
  normalIntegrationWeight: number; // 0.0 to 1.0
  gradientSlopeGain: number; // 0.5 to 4.0
  screenedLambda: number; // 0.02 to 0.50
  invertNormalGreenY: boolean;
  showNormalVectors3D: boolean;
  renderNormalShading: boolean;
  gridResolution: number; // e.g., 112
  cnnDepthWeight: number; // alpha: 0 to 1.8
  mathModulationWeight: number; // beta: 0 to 1.0
  normalMapStrength: number; // 0.0 to 2.5
  harmonicDegreeL: number; // 1 to 6
  harmonicOrderM: number; // 0 to 6
  // Straight-Line & Planar Facet Detection ("Detectar Líneas Rectas · Evitar Montículos")
  detectStraightLines: boolean; // Preserves collinear straight edges and prevents parabolic mound doming
  straightLineSensitivity: number; // 0.0 to 1.0 anisotropy & collinear tangent lock strength
  flatFacetPlanarity: number; // 0.0 to 1.0 flattens pillow/mound domes into crisp planar facets
  // Advanced Mesh Smoothing & Close-Point Vertex Welding (Unión de Punto Cercano)
  smoothingAlgorithm: SmoothingAlgorithm;
  laplacianSmoothingSteps: number; // 0 to 30 steps
  smoothingLambda: number; // 0.15 to 0.90 intensity per step
  tensorBlurPasses: number; // 0 to 8 Gaussian pre-smoothing passes on height/normals
  weldTolerance: number; // 0.000 to 0.080 spatial distance threshold for close-point welding
  autoStitchSeams: boolean; // Welds 360° UV seams (u=0 / u=1) and poles (v=0 / v=1)
  normalSmoothingSteps: number; // 0 to 10 passes of vertex normal relaxation
  // Outer Frame & Color / Hue Cutout ("Eliminar Marco Exterior / Zona Plana / Por Color o Tono")
  cutoutMode: BackgroundCutoutMode;
  cutoutFlatThreshold: number; // 0.00 to 0.50 flat height threshold
  cutoutBorderMargin: number; // 0.00 to 0.45 outer frame margin trim
  cutoutTargetColorHex: string; // Target color hex for Chroma/Tone cutout
  cutoutColorTolerance: number; // 0.02 to 0.70 RGB distance tolerance
  cutoutHueTolerance: number; // 0.02 to 0.50 Hue/Tone angular tolerance
  matchByHueTone: boolean; // True = filter by color tone/hue; False = filter by RGB color distance
  invertCutout: boolean; // True = keep only the matched color/tone
  closeInternalHoles: boolean; // True = only cut exterior background & seal internal holes inside the figure
  holeFillRadius: number; // 0 to 12 morphological closing radius to bridge separated zones and close gaps
  // Multi-View Solid Reconstruction ("Frente + Parte de Atrás + Unión Lateral en cm")
  enableDualSideSolid: boolean;
  symmetricBackRules: boolean; // True = Back shell uses the exact same 4-zone, Poisson, straight-line & smoothing rules symmetrically with its own depths
  backSourceMode: BackImageSourceMode;
  sideSourceMode: SideImageSourceMode;
  backDepthWeight: number; // 0.0 to 1.8 relief amplitude of the back shell
  sideThicknessCm: number; // 0.0 to 6.0 cm lateral seam band width (0.0 cm = direct flush union without gap)
  seamEdgeClosure: number; // 0.0 to 1.0 tapers front & back relief at contour edge so both shells join with zero gap
  sideJointProfile: LateralJointProfile; // SeamlessSmoothUnion, TubularCylinderFull, FlatStraight, Curved90Fillet, RightAngle90, ChamferBevel, ConcaveCove90
  sideJointAngleDeg: number; // 0° (Plana) to 90° (Curva de 90° / Ángulo Recto de 90°)
  sideCornerSharpness: number; // 0.0 (Suave continuo) to 1.0 (Arista viva de 90° ortogonal)
  sideSeamBevel: number; // 0.0 to 1.2 radial extension / depth of the joint profile
  sideReliefAmplitude: number; // 0.0 to 0.6 micro-relief depth along the lateral band
  mirrorBackImageX: boolean; // Mirrors the back image horizontally so silhouettes align 100%
  wireframeOverlay: boolean;
  instancedVoxelMode: boolean;
  showTurntablePedestal: boolean;
}

export interface LightingConfig {
  preset: LightingEnvironmentPreset;
  azimuthDeg: number; // 0 to 360
  elevationDeg: number; // 10 to 85
  keyIntensity: number; // 0.5 to 4.5
  fillIntensity: number; // 0.2 to 2.5
  rimIntensity: number; // 0.2 to 3.5
  exposure: number; // 0.6 to 1.8
  autoRotateLight: boolean;
  autoRotateModel: boolean;
}

export interface EngineTelemetry {
  vertexCount: number;
  weldedVerticesCount: number;
  cutoutTrianglesRemoved: number;
  stitchedSideFacesCount: number;
  triangleCount: number;
  activeInstances: number;
  cnnInferenceMs: number;
  poissonSolveMs: number;
  meshRebuildMs: number;
  pooledObjectsReused: number;
  gcAllocationsInLoop: number;
}

export interface PresetReferenceImage {
  id: string;
  title: string;
  subtitle: string;
  recommendedTopology: MathTopology;
  recommendedNormalMode?: NormalReconstructionMode;
}
