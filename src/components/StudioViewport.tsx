import React, { useEffect, useImperativeHandle, useRef, forwardRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import {
  ATLAS_TEX_SIZE,
  CNN_TENSOR_SIZE,
  CNNReconstructionBuffers,
  buildReconstructedMeshGeometry,
} from '../engine/cnnReconstruction';
import {
  canvasToPNGBytes,
  createDepthMapCanvasFromBuffers,
  createNormalMapCanvasFromBuffers,
  createRGBCanvasFromBuffers,
  createZipArchiveBlob,
  exportGeometryToWavefrontOBJ,
  exportMeshToGLBBytes,
  triggerBlobDownload,
  uint8ToBlob,
} from '../engine/exportEngine';
import {
  VertexDeltaPool,
  _colorA,
  _mat4A,
  _normalA,
  _quatA,
  _scaleVec,
  _vec3A,
  _vec3B,
  _vec3C,
  distSq3D,
  wyvillFalloffFromDistSq,
} from '../engine/objectPool';
import {
  EngineTelemetry,
  LightingConfig,
  LightingEnvironmentPreset,
  ReconstructionParams,
  ViewportToolMode,
  ZoneConfig,
} from '../types/studio';

export interface StudioViewportHandle {
  exportGLB: (filename?: string) => void;
  exportOBJ: (filename?: string) => void;
  captureViewportPNG: (filename?: string) => void;
  exportActivePackZip: (baseName?: string) => Promise<void>;
  resetCamera: () => void;
  setCameraAngle: (azimuthDeg: number) => void;
  resetSculptEdits: () => void;
}

interface StudioViewportProps {
  buffers: CNNReconstructionBuffers;
  cnnVersion: number;
  cnnInferenceMs: number;
  params: ReconstructionParams;
  zones: ZoneConfig[];
  lighting: LightingConfig;
  toolMode: ViewportToolMode;
  activePaintZone: number;
  brushRadius: number;
  brushStrength: number;
  onTelemetryUpdate: (telemetry: EngineTelemetry) => void;
}

const LOD_NEAR_DIST_SQ = 14.0 * 14.0;
const LOD_FAR_DIST_SQ = 36.0 * 36.0;
const MAX_INSTANCES = 4096;
const MAX_NORMAL_NEEDLES = 1600;
const _upAxis = new THREE.Vector3(0, 1, 0);

export const StudioViewport = forwardRef<StudioViewportHandle, StudioViewportProps>(
  (
    {
      buffers,
      cnnVersion,
      cnnInferenceMs,
      params,
      zones,
      lighting,
      toolMode,
      activePaintZone,
      brushRadius,
      brushStrength,
      onTelemetryUpdate,
    },
    ref
  ) => {
    const canvasMountRef = useRef<HTMLDivElement | null>(null);

    const paramsRef = useRef(params);
    paramsRef.current = params;
    const zonesRef = useRef(zones);
    zonesRef.current = zones;
    const lightingRef = useRef(lighting);
    lightingRef.current = lighting;
    const toolModeRef = useRef(toolMode);
    toolModeRef.current = toolMode;
    const activePaintZoneRef = useRef(activePaintZone);
    activePaintZoneRef.current = activePaintZone;
    const brushRadiusRef = useRef(brushRadius);
    brushRadiusRef.current = brushRadius;
    const brushStrengthRef = useRef(brushStrength);
    brushStrengthRef.current = brushStrength;
    const cnnInferenceMsRef = useRef(cnnInferenceMs);
    cnnInferenceMsRef.current = cnnInferenceMs;

    const heroMeshRef = useRef<THREE.Mesh | null>(null);
    const heroMatRef = useRef<THREE.MeshPhysicalMaterial | null>(null);
    const wireframeMeshRef = useRef<THREE.LineSegments | null>(null);
    const voxelInstancedRef = useRef<THREE.InstancedMesh | null>(null);
    const normalNeedlesRef = useRef<THREE.InstancedMesh | null>(null);
    const pedestalGroupRef = useRef<THREE.Group | null>(null);
    const controlsRef = useRef<OrbitControls | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const keyLightRef = useRef<THREE.DirectionalLight | null>(null);
    const fillLightRef = useRef<THREE.DirectionalLight | null>(null);
    const rimLightRef = useRef<THREE.SpotLight | null>(null);
    const hemiLightRef = useRef<THREE.HemisphereLight | null>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const vertexDeltaPoolRef = useRef<VertexDeltaPool>(new VertexDeltaPool(16384));
    const rebuildGeometryFnRef = useRef<(() => void) | null>(null);

    useImperativeHandle(ref, () => ({
      exportGLB: (filename = 'vortex3d_modelo_neuronal.glb') => {
        const heroMesh = heroMeshRef.current;
        if (!heroMesh) return;

        exportMeshToGLBBytes(heroMesh)
          .then((bytes) => {
            triggerBlobDownload(uint8ToBlob(bytes, 'model/gltf-binary'), filename);
          })
          .catch((error) => {
            console.error('Error al exportar archivo .GLB:', error);
          });
      },
      exportOBJ: (filename = 'vortex3d_malla.obj') => {
        const heroMesh = heroMeshRef.current;
        if (!heroMesh) return;
        const objBytes = exportGeometryToWavefrontOBJ(
          heroMesh.geometry,
          filename.replace(/\.obj$/i, '')
        );
        const blob = uint8ToBlob(objBytes, 'text/plain;charset=utf-8');
        triggerBlobDownload(blob, filename);
      },
      captureViewportPNG: (filename = 'vortex3d_render_3d.png') => {
        const renderer = rendererRef.current;
        const scene = sceneRef.current;
        const camera = cameraRef.current;
        if (!renderer || !scene || !camera) return;

        // Render synchronously right before toBlob() so the drawing buffer is fresh
        renderer.render(scene, camera);
        renderer.domElement.toBlob((blob) => {
          if (blob) {
            triggerBlobDownload(blob, filename);
          }
        }, 'image/png');
      },
      exportActivePackZip: async (baseName = 'vortex3d_modelo_activo') => {
        const heroMesh = heroMeshRef.current;
        const renderer = rendererRef.current;
        const scene = sceneRef.current;
        const camera = cameraRef.current;
        if (!heroMesh || !renderer || !scene || !camera) return;

        renderer.render(scene, camera);
        const renderBlobPromise = new Promise<Uint8Array>((resolve) => {
          renderer.domElement.toBlob(async (b) => {
            if (b) {
              resolve(new Uint8Array(await b.arrayBuffer()));
            } else {
              resolve(new Uint8Array(0));
            }
          }, 'image/png');
        });

        const [rgbBytes, normalBytes, depthBytes, glbBytes, renderBytes] =
          await Promise.all([
            canvasToPNGBytes(createRGBCanvasFromBuffers(buffers)),
            canvasToPNGBytes(createNormalMapCanvasFromBuffers(buffers)),
            canvasToPNGBytes(createDepthMapCanvasFromBuffers(buffers)),
            exportMeshToGLBBytes(heroMesh),
            renderBlobPromise,
          ]);
        const objBytes = exportGeometryToWavefrontOBJ(heroMesh.geometry, baseName);

        const zipBlob = createZipArchiveBlob([
          { filename: `${baseName}_modelo_3d.glb`, data: glbBytes },
          { filename: `${baseName}_malla_3d.obj`, data: objBytes },
          { filename: `${baseName}_textura_rgb.png`, data: rgbBytes },
          { filename: `${baseName}_mapa_normales.png`, data: normalBytes },
          { filename: `${baseName}_profundidad_poisson.png`, data: depthBytes },
          { filename: `${baseName}_captura_render_3d.png`, data: renderBytes },
        ]);

        triggerBlobDownload(zipBlob, `${baseName}_pack_completo.zip`);
      },
      resetCamera: () => {
        if (cameraRef.current && controlsRef.current) {
          cameraRef.current.position.set(0, 0.65, 4.1);
          controlsRef.current.target.set(0, 0.15, 0);
          controlsRef.current.update();
        }
        if (heroMeshRef.current && wireframeMeshRef.current && voxelInstancedRef.current && normalNeedlesRef.current) {
          heroMeshRef.current.rotation.y = 0;
          wireframeMeshRef.current.rotation.y = 0;
          voxelInstancedRef.current.rotation.y = 0;
          normalNeedlesRef.current.rotation.y = 0;
        }
      },
      setCameraAngle: (azimuthDeg: number) => {
        if (cameraRef.current && controlsRef.current) {
          const rad = THREE.MathUtils.degToRad(azimuthDeg);
          const dist = 4.1;
          cameraRef.current.position.set(
            Math.sin(rad) * dist,
            0.45,
            Math.cos(rad) * dist
          );
          controlsRef.current.target.set(0, 0.15, 0);
          controlsRef.current.update();
        }
        if (heroMeshRef.current && wireframeMeshRef.current && voxelInstancedRef.current && normalNeedlesRef.current) {
          heroMeshRef.current.rotation.y = 0;
          wireframeMeshRef.current.rotation.y = 0;
          voxelInstancedRef.current.rotation.y = 0;
          normalNeedlesRef.current.rotation.y = 0;
        }
      },
      resetSculptEdits: () => {
        buffers.resetSculptAndZoneOverrides();
        rebuildGeometryFnRef.current?.();
      },
    }));

    useEffect(() => {
      const container = canvasMountRef.current;
      if (!container) return;

      const isSoftwareGl = (() => {
        const probeCanvas = document.createElement('canvas');
        const gl = probeCanvas.getContext('webgl2');
        if (!gl) return true;
        const ext = gl.getExtension('WEBGL_debug_renderer_info');
        const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
        const gpu = String(gl.getParameter(param));
        gl.getExtension('WEBGL_lose_context')?.loseContext();
        return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
      })();

      const width = container.clientWidth || window.innerWidth;
      const height = container.clientHeight || window.innerHeight;

      const renderer = new THREE.WebGLRenderer({
        antialias: !isSoftwareGl,
        powerPreference: 'high-performance',
      });
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = lightingRef.current.exposure;
      renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
      renderer.setSize(width, height);
      renderer.shadowMap.enabled = !isSoftwareGl;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      container.appendChild(renderer.domElement);
      rendererRef.current = renderer;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x090b10);
      scene.fog = new THREE.FogExp2(0x090b10, 0.042);
      sceneRef.current = scene;

      const pmremGenerator = new THREE.PMREMGenerator(renderer);
      scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
      pmremGenerator.dispose();

      const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
      camera.position.set(0, 0.65, 4.1);
      cameraRef.current = camera;

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = !isSoftwareGl;
      controls.dampingFactor = 0.06;
      controls.maxDistance = 16;
      controls.minDistance = 1.4;
      controls.target.set(0, 0.15, 0);
      controls.update();
      controlsRef.current = controls;

      const hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0x0f172a, 0.95);
      scene.add(hemiLight);
      hemiLightRef.current = hemiLight;

      const keyLight = new THREE.DirectionalLight(0xfff7ed, 2.6);
      keyLight.position.set(3.2, 4.2, 4.2);
      keyLight.castShadow = !isSoftwareGl;
      keyLight.shadow.mapSize.set(1024, 1024);
      keyLight.shadow.camera.near = 0.5;
      keyLight.shadow.camera.far = 16;
      const d = 3.2;
      keyLight.shadow.camera.left = -d;
      keyLight.shadow.camera.right = d;
      keyLight.shadow.camera.top = d;
      keyLight.shadow.camera.bottom = -d;
      keyLight.shadow.bias = -0.0005;
      keyLight.shadow.normalBias = 0.02;
      scene.add(keyLight);
      keyLightRef.current = keyLight;

      const fillLight = new THREE.DirectionalLight(0xbae6fd, 1.25);
      fillLight.position.set(-4.0, 2.2, 3.0);
      scene.add(fillLight);
      fillLightRef.current = fillLight;

      const rimLight = new THREE.SpotLight(0xf59e0b, 2.4, 18, Math.PI * 0.35, 0.5, 1.2);
      rimLight.position.set(0, 3.8, -4.5);
      scene.add(rimLight);
      rimLightRef.current = rimLight;

      if (!isSoftwareGl) {
        RectAreaLightUniformsLib.init();
        const softboxLeft = new THREE.RectAreaLight(0xe0f2fe, 2.4, 3.5, 3.5);
        softboxLeft.position.set(-4.2, 2.5, 3.2);
        softboxLeft.lookAt(0, 0.2, 0);
        scene.add(softboxLeft);

        const softboxRight = new THREE.RectAreaLight(0xfef3c7, 2.0, 3.5, 3.5);
        softboxRight.position.set(4.2, 2.8, 2.2);
        softboxRight.lookAt(0, 0.2, 0);
        scene.add(softboxRight);
      }

      const pedestalGroup = new THREE.Group();
      pedestalGroup.position.set(0, -1.55, 0);

      const baseGeo = new THREE.CylinderGeometry(2.35, 2.55, 0.18, 64);
      const baseMat = new THREE.MeshStandardMaterial({
        color: 0x111827,
        roughness: 0.42,
        metalness: 0.78,
      });
      const baseMesh = new THREE.Mesh(baseGeo, baseMat);
      baseMesh.position.y = -0.09;
      baseMesh.receiveShadow = true;
      pedestalGroup.add(baseMesh);

      const ringGeo = new THREE.TorusGeometry(2.32, 0.035, 16, 96);
      ringGeo.rotateX(Math.PI * 0.5);
      const ringMat = new THREE.MeshStandardMaterial({
        color: 0x38bdf8,
        roughness: 0.14,
        metalness: 0.94,
      });
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.position.y = 0.01;
      pedestalGroup.add(ringMesh);

      const stageGeo = new THREE.CylinderGeometry(1.95, 2.02, 0.06, 64);
      const stageMat = new THREE.MeshStandardMaterial({
        color: 0x0f172a,
        roughness: 0.75,
        metalness: 0.25,
      });
      const stageMesh = new THREE.Mesh(stageGeo, stageMat);
      stageMesh.position.y = 0.03;
      stageMesh.receiveShadow = true;
      pedestalGroup.add(stageMesh);

      const lugGeo = new THREE.BoxGeometry(0.03, 0.02, 0.14);
      const lugMat = new THREE.MeshStandardMaterial({
        color: 0x94a3b8,
        roughness: 0.25,
        metalness: 0.9,
      });
      const lugInstances = new THREE.InstancedMesh(lugGeo, lugMat, 36);
      for (let i = 0; i < 36; i++) {
        const angle = (i * Math.PI * 2) / 36;
        _vec3A.set(Math.cos(angle) * 2.14, 0.02, Math.sin(angle) * 2.14);
        _quatA.setFromAxisAngle(_vec3B.set(0, 1, 0), -angle);
        _scaleVec.set(1, 1, i % 3 === 0 ? 1.45 : 0.85);
        _mat4A.compose(_vec3A, _quatA, _scaleVec);
        lugInstances.setMatrixAt(i, _mat4A);
      }
      lugInstances.instanceMatrix.needsUpdate = true;
      lugInstances.computeBoundingSphere();
      pedestalGroup.add(lugInstances);

      const shadowCanvas = document.createElement('canvas');
      shadowCanvas.width = 128;
      shadowCanvas.height = 128;
      const sCtx = shadowCanvas.getContext('2d');
      if (sCtx) {
        const grad = sCtx.createRadialGradient(64, 64, 6, 64, 64, 62);
        grad.addColorStop(0, 'rgba(0, 0, 0, 0.72)');
        grad.addColorStop(0.55, 'rgba(0, 0, 0, 0.28)');
        grad.addColorStop(1, 'rgba(0, 0, 0, 0.0)');
        sCtx.fillStyle = grad;
        sCtx.fillRect(0, 0, 128, 128);
      }
      const contactTex = new THREE.CanvasTexture(shadowCanvas);
      const contactGeo = new THREE.PlaneGeometry(3.4, 3.4);
      contactGeo.rotateX(-Math.PI * 0.5);
      const contactMat = new THREE.MeshBasicMaterial({
        map: contactTex,
        transparent: true,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1,
      });
      const contactPlane = new THREE.Mesh(contactGeo, contactMat);
      contactPlane.position.y = 0.065;
      pedestalGroup.add(contactPlane);

      scene.add(pedestalGroup);
      pedestalGroupRef.current = pedestalGroup;

      const maxAniso = renderer.capabilities.getMaxAnisotropy();

      const albedoTex = new THREE.DataTexture(
        new Uint8Array(ATLAS_TEX_SIZE * ATLAS_TEX_SIZE * 4),
        ATLAS_TEX_SIZE,
        ATLAS_TEX_SIZE,
        THREE.RGBAFormat
      );
      albedoTex.colorSpace = THREE.SRGBColorSpace;
      albedoTex.wrapS = THREE.ClampToEdgeWrapping;
      albedoTex.wrapT = THREE.ClampToEdgeWrapping;
      albedoTex.minFilter = THREE.LinearFilter;
      albedoTex.magFilter = THREE.LinearFilter;
      albedoTex.generateMipmaps = false;
      albedoTex.anisotropy = maxAniso;

      const roughMetalTex = new THREE.DataTexture(
        new Uint8Array(ATLAS_TEX_SIZE * ATLAS_TEX_SIZE * 4),
        ATLAS_TEX_SIZE,
        ATLAS_TEX_SIZE,
        THREE.RGBAFormat
      );
      roughMetalTex.colorSpace = THREE.NoColorSpace;
      roughMetalTex.wrapS = THREE.ClampToEdgeWrapping;
      roughMetalTex.wrapT = THREE.ClampToEdgeWrapping;
      roughMetalTex.minFilter = THREE.LinearFilter;
      roughMetalTex.magFilter = THREE.LinearFilter;
      roughMetalTex.anisotropy = maxAniso;

      const normalTex = new THREE.DataTexture(
        new Uint8Array(ATLAS_TEX_SIZE * ATLAS_TEX_SIZE * 4),
        ATLAS_TEX_SIZE,
        ATLAS_TEX_SIZE,
        THREE.RGBAFormat
      );
      normalTex.colorSpace = THREE.NoColorSpace;
      normalTex.wrapS = THREE.ClampToEdgeWrapping;
      normalTex.wrapT = THREE.ClampToEdgeWrapping;
      normalTex.minFilter = THREE.LinearFilter;
      normalTex.magFilter = THREE.LinearFilter;
      normalTex.anisotropy = maxAniso;

      const heroGeo = new THREE.BufferGeometry();
      const nStrength = paramsRef.current.normalMapStrength;
      const heroMat = new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        map: albedoTex,
        normalMap: normalTex,
        normalScale: new THREE.Vector2(nStrength, nStrength),
        roughnessMap: roughMetalTex,
        metalnessMap: roughMetalTex,
        roughness: 1.0,
        metalness: 1.0,
        clearcoat: 0.22,
        clearcoatRoughness: 0.18,
        alphaTest: 0,
        vertexColors: false,
        side: THREE.DoubleSide,
      });
      heroMatRef.current = heroMat;

      const heroMesh = new THREE.Mesh(heroGeo, heroMat);
      heroMesh.castShadow = !isSoftwareGl;
      heroMesh.receiveShadow = !isSoftwareGl;
      heroMesh.position.set(0, 0.18, 0);
      scene.add(heroMesh);
      heroMeshRef.current = heroMesh;

      const wireMat = new THREE.LineBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.32,
      });
      const wireMesh = new THREE.LineSegments(new THREE.BufferGeometry(), wireMat);
      wireMesh.position.copy(heroMesh.position);
      wireMesh.visible = false;
      scene.add(wireMesh);
      wireframeMeshRef.current = wireMesh;

      const voxelGeo = new THREE.OctahedronGeometry(0.036, 1);
      const voxelMat = new THREE.MeshStandardMaterial({
        roughness: 0.25,
        metalness: 0.82,
      });
      const voxelInstanced = new THREE.InstancedMesh(voxelGeo, voxelMat, MAX_INSTANCES);
      voxelInstanced.instanceColor = new THREE.InstancedBufferAttribute(
        new Float32Array(MAX_INSTANCES * 3),
        3
      );
      voxelInstanced.position.copy(heroMesh.position);
      voxelInstanced.castShadow = !isSoftwareGl;
      voxelInstanced.receiveShadow = !isSoftwareGl;
      voxelInstanced.visible = false;
      voxelInstanced.frustumCulled = false;
      scene.add(voxelInstanced);
      voxelInstancedRef.current = voxelInstanced;

      const needleGeo = new THREE.ConeGeometry(0.014, 0.16, 6);
      needleGeo.translate(0, 0.08, 0);
      const needleMat = new THREE.MeshBasicMaterial({
        color: 0xffffff,
      });
      const normalNeedles = new THREE.InstancedMesh(needleGeo, needleMat, MAX_NORMAL_NEEDLES);
      normalNeedles.instanceColor = new THREE.InstancedBufferAttribute(
        new Float32Array(MAX_NORMAL_NEEDLES * 3),
        3
      );
      normalNeedles.position.copy(heroMesh.position);
      normalNeedles.visible = false;
      normalNeedles.frustumCulled = false;
      scene.add(normalNeedles);
      normalNeedlesRef.current = normalNeedles;

      const cursorRingGeo = new THREE.RingGeometry(0.85, 1.0, 32);
      const cursorRingMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.85,
        depthTest: false,
      });
      const brushCursorMesh = new THREE.Mesh(cursorRingGeo, cursorRingMat);
      brushCursorMesh.visible = false;
      brushCursorMesh.renderOrder = 10;
      scene.add(brushCursorMesh);

      const syncVoxelInstancedMesh = () => {
        const posAttr = heroGeo.getAttribute('position') as THREE.BufferAttribute | undefined;
        const colAttr = heroGeo.getAttribute('color') as THREE.BufferAttribute | undefined;
        if (!posAttr || !colAttr) return 0;

        const totalVerts = posAttr.count;
        const stride = Math.max(1, Math.ceil(totalVerts / MAX_INSTANCES));
        const posArr = posAttr.array as Float32Array;
        const colArr = colAttr.array as Float32Array;

        _vec3C.copy(camera.position);
        let instanceIdx = 0;

        for (let v = 0; v < totalVerts && instanceIdx < MAX_INSTANCES; v += stride) {
          if (buffers.vertexCutoutMask[v] === 0) continue;
          const v3 = v * 3;
          const px = posArr[v3];
          const py = posArr[v3 + 1];
          const pz = posArr[v3 + 2];

          const dSq = distSq3D(px, py, pz, _vec3C.x, _vec3C.y, _vec3C.z);
          let lodScale = 1.0;
          if (dSq > LOD_FAR_DIST_SQ) {
            lodScale = 0.0;
          } else if (dSq > LOD_NEAR_DIST_SQ) {
            lodScale = 1.0 - (dSq - LOD_NEAR_DIST_SQ) / (LOD_FAR_DIST_SQ - LOD_NEAR_DIST_SQ);
          }

          _vec3A.set(px, py, pz);
          _quatA.identity();
          _scaleVec.setScalar(lodScale);
          _mat4A.compose(_vec3A, _quatA, _scaleVec);
          voxelInstanced.setMatrixAt(instanceIdx, _mat4A);

          _colorA.setRGB(colArr[v3], colArr[v3 + 1], colArr[v3 + 2]);
          voxelInstanced.setColorAt(instanceIdx, _colorA);
          instanceIdx++;
        }

        voxelInstanced.count = instanceIdx;
        voxelInstanced.instanceMatrix.needsUpdate = true;
        if (voxelInstanced.instanceColor) {
          voxelInstanced.instanceColor.needsUpdate = true;
        }
        voxelInstanced.computeBoundingSphere();
        return instanceIdx;
      };

      const syncNormalNeedlesInstanced = () => {
        const posAttr = heroGeo.getAttribute('position') as THREE.BufferAttribute | undefined;
        const normAttr = heroGeo.getAttribute('normal') as THREE.BufferAttribute | undefined;
        if (!posAttr || !normAttr) return 0;

        const totalVerts = posAttr.count;
        const stride = Math.max(1, Math.ceil(totalVerts / MAX_NORMAL_NEEDLES));
        const posArr = posAttr.array as Float32Array;
        const normArr = normAttr.array as Float32Array;

        let instanceIdx = 0;
        for (let v = 0; v < totalVerts && instanceIdx < MAX_NORMAL_NEEDLES; v += stride) {
          if (buffers.vertexCutoutMask[v] === 0) continue;
          const v3 = v * 3;
          const px = posArr[v3];
          const py = posArr[v3 + 1];
          const pz = posArr[v3 + 2];

          const nx = normArr[v3];
          const ny = normArr[v3 + 1];
          const nz = normArr[v3 + 2];

          _vec3A.set(px, py, pz);
          _normalA.set(nx, ny, nz).normalize();
          _quatA.setFromUnitVectors(_upAxis, _normalA);
          _scaleVec.setScalar(1.0);
          _mat4A.compose(_vec3A, _quatA, _scaleVec);
          normalNeedles.setMatrixAt(instanceIdx, _mat4A);

          _colorA.setRGB(nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5);
          normalNeedles.setColorAt(instanceIdx, _colorA);
          instanceIdx++;
        }

        normalNeedles.count = instanceIdx;
        normalNeedles.instanceMatrix.needsUpdate = true;
        if (normalNeedles.instanceColor) {
          normalNeedles.instanceColor.needsUpdate = true;
        }
        normalNeedles.computeBoundingSphere();
        return instanceIdx;
      };

      const rebuildAll = () => {
        const stats = buildReconstructedMeshGeometry(
          heroGeo,
          buffers,
          paramsRef.current,
          zonesRef.current,
          albedoTex,
          roughMetalTex,
          normalTex
        );

        const ns = paramsRef.current.normalMapStrength;
        heroMat.normalScale.set(ns, ns);

        if (paramsRef.current.wireframeOverlay) {
          wireMesh.geometry.dispose();
          wireMesh.geometry = new THREE.WireframeGeometry(heroGeo);
          wireMesh.visible = true;
        } else {
          wireMesh.visible = false;
        }

        const isVoxel = paramsRef.current.instancedVoxelMode;
        heroMesh.visible = !isVoxel;
        voxelInstanced.visible = isVoxel;
        let activeInstances = 36;
        if (isVoxel) {
          activeInstances += syncVoxelInstancedMesh();
        }

        if (paramsRef.current.showNormalVectors3D) {
          normalNeedles.visible = true;
          activeInstances += syncNormalNeedlesInstanced();
        } else {
          normalNeedles.visible = false;
        }

        onTelemetryUpdate({
          vertexCount: stats.vertexCount,
          weldedVerticesCount: stats.weldedVerticesCount,
          cutoutTrianglesRemoved: stats.cutoutTrianglesRemoved,
          stitchedSideFacesCount: stats.stitchedSideFacesCount,
          triangleCount: stats.triangleCount,
          activeInstances,
          cnnInferenceMs: cnnInferenceMsRef.current,
          poissonSolveMs: stats.poissonSolveMs,
          meshRebuildMs: stats.elapsedMs,
          pooledObjectsReused: vertexDeltaPoolRef.current.totalReused,
          gcAllocationsInLoop: 0,
        });
      };

      rebuildGeometryFnRef.current = rebuildAll;
      rebuildAll();

      if (!isSoftwareGl) {
        renderer.compile(scene, camera);
      }

      const raycaster = new THREE.Raycaster();
      const pointerNDC = new THREE.Vector2();
      const prevPointerNDC = new THREE.Vector2();
      let isPointerDown = false;
      let lastRaycastTime = 0;

      const applyBrushAtIntersection = (
        hit: THREE.Intersection,
        dragDeltaX: number,
        dragDeltaY: number
      ) => {
        const mode = toolModeRef.current;
        if (mode === ViewportToolMode.Orbit) return;

        if (mode === ViewportToolMode.SculptNormals && hit.uv) {
          const centerU = hit.uv.x;
          const centerV = hit.uv.y;
          const w = CNN_TENSOR_SIZE;
          const h = CNN_TENSOR_SIZE;
          const cx = Math.floor(centerU * (w - 1));
          const cy = Math.floor(centerV * (h - 1));

          const texRadius = Math.max(4, Math.floor(brushRadiusRef.current * 44));
          const texRadiusSq = texRadius * texRadius;
          const invRadSq = 1.0 / texRadiusSq;

          const tiltStrength = brushStrengthRef.current * 1.15;
          const strokeNx =
            Math.abs(dragDeltaX) + Math.abs(dragDeltaY) > 0.001
              ? dragDeltaX * 35.0
              : 0.0;
          const strokeNy =
            Math.abs(dragDeltaX) + Math.abs(dragDeltaY) > 0.001
              ? dragDeltaY * 35.0
              : 0.0;

          const yMin = Math.max(1, cy - texRadius);
          const yMax = Math.min(h - 2, cy + texRadius);
          const xMin = Math.max(1, cx - texRadius);
          const xMax = Math.min(w - 2, cx + texRadius);

          for (let ty = yMin; ty <= yMax; ty++) {
            const dy = ty - cy;
            const yOffset = ty * w;
            for (let tx = xMin; tx <= xMax; tx++) {
              const dx = tx - cx;
              const dSq = dx * dx + dy * dy;
              if (dSq < texRadiusSq) {
                const t = 1.0 - dSq * invRadSq;
                const weight = t * t * tiltStrength;
                const t2 = (yOffset + tx) * 2;
                if (strokeNx !== 0 || strokeNy !== 0) {
                  buffers.customNormalTilt[t2] += strokeNx * weight;
                  buffers.customNormalTilt[t2 + 1] += strokeNy * weight;
                } else {
                  buffers.customNormalTilt[t2] += (dx / texRadius) * weight * 0.65;
                  buffers.customNormalTilt[t2 + 1] -= (dy / texRadius) * weight * 0.65;
                }
              }
            }
          }

          rebuildAll();
          return;
        }

        _vec3A.copy(hit.point);
        heroMesh.worldToLocal(_vec3A);

        const radius = brushRadiusRef.current;
        const radiusSq = radius * radius;
        const strength = brushStrengthRef.current * 0.045;

        const posAttr = heroGeo.getAttribute('position') as THREE.BufferAttribute;
        const normAttr = heroGeo.getAttribute('normal') as THREE.BufferAttribute;
        const posArr = posAttr.array as Float32Array;
        const normArr = normAttr.array as Float32Array;
        const count = posAttr.count;

        const pool = vertexDeltaPoolRef.current;
        pool.releaseAll();

        for (let i = 0; i < count; i++) {
          const i3 = i * 3;
          const vx = posArr[i3];
          const vy = posArr[i3 + 1];
          const vz = posArr[i3 + 2];

          const dSq = distSq3D(vx, vy, vz, _vec3A.x, _vec3A.y, _vec3A.z);
          if (dSq < radiusSq) {
            const weight = wyvillFalloffFromDistSq(dSq, radiusSq);
            if (mode === ViewportToolMode.PaintZone) {
              if (weight > 0.15) {
                buffers.customZoneOverrides[i] = activePaintZoneRef.current;
                if (zonesRef.current[activePaintZoneRef.current]?.zoneTintMix === 0) {
                  zonesRef.current[activePaintZoneRef.current].zoneTintMix = 0.65;
                }
              }
            } else {
              const sign = mode === ViewportToolMode.SculptDeflate ? -1.0 : 1.0;
              if (mode === ViewportToolMode.SculptSmooth) {
                pool.acquire(
                  i,
                  -buffers.sculptOffsets[i3] * weight * 0.25,
                  -buffers.sculptOffsets[i3 + 1] * weight * 0.25,
                  -buffers.sculptOffsets[i3 + 2] * weight * 0.25,
                  weight
                );
              } else {
                pool.acquire(
                  i,
                  normArr[i3] * strength * weight * sign,
                  normArr[i3 + 1] * strength * weight * sign,
                  normArr[i3 + 2] * strength * weight * sign,
                  weight
                );
              }
            }
          }
        }

        const activeLen = pool.getActiveLength();
        const activeBuf = pool.getActiveBuffer();
        for (let k = 0; k < activeLen; k++) {
          const item = activeBuf[k];
          const v3 = item.vertexIndex * 3;
          buffers.sculptOffsets[v3] += item.dx;
          buffers.sculptOffsets[v3 + 1] += item.dy;
          buffers.sculptOffsets[v3 + 2] += item.dz;
        }

        rebuildAll();
      };

      const updatePointerRay = (clientX: number, clientY: number, applyStroke: boolean) => {
        const mode = toolModeRef.current;
        if (mode === ViewportToolMode.Orbit) {
          brushCursorMesh.visible = false;
          return;
        }

        const rect = renderer.domElement.getBoundingClientRect();
        pointerNDC.x = ((clientX - rect.left) / rect.width) * 2 - 1;
        pointerNDC.y = -((clientY - rect.top) / rect.height) * 2 + 1;

        const dragDx = pointerNDC.x - prevPointerNDC.x;
        const dragDy = pointerNDC.y - prevPointerNDC.y;
        prevPointerNDC.copy(pointerNDC);

        raycaster.setFromCamera(pointerNDC, camera);
        const hits = raycaster.intersectObject(heroMesh, false);
        if (hits.length > 0) {
          const hit = hits[0];
          brushCursorMesh.visible = true;
          brushCursorMesh.position.copy(hit.point);
          if (hit.face) {
            _normalA.copy(hit.face.normal).transformDirection(heroMesh.matrixWorld);
            _vec3B.copy(hit.point).add(_normalA);
            brushCursorMesh.lookAt(_vec3B);
          }
          const r = brushRadiusRef.current;
          brushCursorMesh.scale.set(r, r, r);

          if (applyStroke) {
            applyBrushAtIntersection(hit, dragDx, dragDy);
          }
        } else {
          brushCursorMesh.visible = false;
        }
      };

      const onPointerDown = (e: PointerEvent) => {
        if (e.button !== 0) return;
        const rect = renderer.domElement.getBoundingClientRect();
        prevPointerNDC.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        prevPointerNDC.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        if (toolModeRef.current !== ViewportToolMode.Orbit) {
          isPointerDown = true;
          controls.enabled = false;
          updatePointerRay(e.clientX, e.clientY, true);
        } else {
          controls.enabled = true;
        }
      };

      const onPointerMove = (e: PointerEvent) => {
        const now = performance.now();
        if (now - lastRaycastTime < 24) return;
        lastRaycastTime = now;
        updatePointerRay(e.clientX, e.clientY, isPointerDown);
      };

      const onPointerUp = () => {
        isPointerDown = false;
        controls.enabled = toolModeRef.current === ViewportToolMode.Orbit;
      };

      const domElem = renderer.domElement;
      domElem.addEventListener('pointerdown', onPointerDown);
      domElem.addEventListener('pointermove', onPointerMove);
      window.addEventListener('pointerup', onPointerUp);

      const onResize = () => {
        if (!container) return;
        const w = container.clientWidth || window.innerWidth;
        const h = container.clientHeight || window.innerHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      };
      window.addEventListener('resize', onResize);

      let frameId = 0;
      const clock = new THREE.Clock();

      function afterDraw(cb: () => void) {
        const gl = renderer.getContext() as WebGL2RenderingContext;
        const sync = gl.fenceSync ? gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0) : null;
        gl.flush();
        const poll = () => {
          const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
          if (state === gl.UNSIGNALED) {
            frameId = window.setTimeout(poll, 5);
          } else {
            if (sync) gl.deleteSync(sync);
            frameId = window.setTimeout(cb, 80);
          }
        };
        frameId = window.setTimeout(poll, 5);
      }

      let lightOrbitAngle = 0;
      let swayTime = 0;

      function tick() {
        const dt = Math.min(clock.getDelta(), 0.1);
        const lCfg = lightingRef.current;

        if (lCfg.autoRotateModel) {
          swayTime += dt;
          if (
            paramsRef.current.topology === 'monolith_relief' &&
            !paramsRef.current.enableDualSideSolid
          ) {
            heroMesh.rotation.y = Math.sin(swayTime * 0.75) * 0.45;
          } else {
            heroMesh.rotation.y += dt * 0.42;
          }
          wireMesh.rotation.y = heroMesh.rotation.y;
          voxelInstanced.rotation.y = heroMesh.rotation.y;
          normalNeedles.rotation.y = heroMesh.rotation.y;
          pedestalGroup.rotation.y = heroMesh.rotation.y * 0.5;
        }

        const elevRad = THREE.MathUtils.degToRad(lCfg.elevationDeg);
        if (lCfg.autoRotateLight) {
          lightOrbitAngle += dt * 0.65;
        } else {
          lightOrbitAngle = THREE.MathUtils.degToRad(lCfg.azimuthDeg);
        }

        const lightRadius = 5.4;
        const lx = Math.cos(elevRad) * Math.sin(lightOrbitAngle) * lightRadius;
        const ly = Math.sin(elevRad) * lightRadius;
        const lz = Math.cos(elevRad) * Math.cos(lightOrbitAngle) * lightRadius;

        keyLight.position.set(lx, ly, lz);
        fillLight.position.set(-lx * 0.85, ly * 0.55, Math.abs(lz) * 0.75);

        controls.update();
        renderer.render(scene, camera);

        if (isSoftwareGl) {
          afterDraw(tick);
        } else {
          frameId = requestAnimationFrame(tick);
        }
      }

      tick();

      return () => {
        cancelAnimationFrame(frameId);
        clearTimeout(frameId);
        domElem.removeEventListener('pointerdown', onPointerDown);
        domElem.removeEventListener('pointermove', onPointerMove);
        window.removeEventListener('pointerup', onPointerUp);
        window.removeEventListener('resize', onResize);
        heroGeo.dispose();
        heroMat.dispose();
        wireMat.dispose();
        voxelGeo.dispose();
        voxelMat.dispose();
        needleGeo.dispose();
        needleMat.dispose();
        albedoTex.dispose();
        roughMetalTex.dispose();
        normalTex.dispose();
        renderer.dispose();
        if (container.contains(domElem)) {
          container.removeChild(domElem);
        }
      };
    }, []);

    useEffect(() => {
      rebuildGeometryFnRef.current?.();
      if (pedestalGroupRef.current) {
        pedestalGroupRef.current.visible = params.showTurntablePedestal;
      }
      if (params.topology === 'monolith_relief' && heroMeshRef.current && !lighting.autoRotateModel) {
        heroMeshRef.current.rotation.y = 0;
      }
    }, [cnnVersion, params, zones]);

    useEffect(() => {
      const renderer = rendererRef.current;
      const keyLight = keyLightRef.current;
      const fillLight = fillLightRef.current;
      const rimLight = rimLightRef.current;
      const hemiLight = hemiLightRef.current;
      if (!renderer || !keyLight || !fillLight || !rimLight || !hemiLight) return;

      renderer.toneMappingExposure = lighting.exposure;
      keyLight.intensity = lighting.keyIntensity;
      fillLight.intensity = lighting.fillIntensity;
      rimLight.intensity = lighting.rimIntensity;

      if (lighting.preset === LightingEnvironmentPreset.StudioWarm) {
        keyLight.color.setHex(0xfff7ed);
        fillLight.color.setHex(0x38bdf8);
        rimLight.color.setHex(0xf59e0b);
        hemiLight.color.setHex(0xdbeafe);
      } else if (lighting.preset === LightingEnvironmentPreset.CyberLabCool) {
        keyLight.color.setHex(0xe0f2fe);
        fillLight.color.setHex(0x6366f1);
        rimLight.color.setHex(0x06b6d4);
        hemiLight.color.setHex(0xc7d2fe);
      } else if (lighting.preset === LightingEnvironmentPreset.GoldenSunset) {
        keyLight.color.setHex(0xfde68a);
        fillLight.color.setHex(0x818cf8);
        rimLight.color.setHex(0xf97316);
        hemiLight.color.setHex(0xfef3c7);
      } else {
        keyLight.color.setHex(0xf8fafc);
        fillLight.color.setHex(0x1e293b);
        rimLight.color.setHex(0x38bdf8);
        hemiLight.color.setHex(0x475569);
      }
    }, [lighting]);

    useEffect(() => {
      if (controlsRef.current) {
        controlsRef.current.enabled = toolMode === ViewportToolMode.Orbit;
      }
    }, [toolMode]);

    return (
      <div
        ref={canvasMountRef}
        className="w-full h-full relative overflow-hidden cursor-crosshair"
        aria-label="Visor 3D interactivo con reconstrucción de Poisson basada en normales"
      />
    );
  }
);
