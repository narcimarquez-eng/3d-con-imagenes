// ============================================================================
// LAYER MAP ("mapa de capas")
// An optional image, painted over a copy of the front/back picture with a few
// flat colours, that says which parts stick out from the body and by how much.
// The original picture is still used for the texture; the layer map only
// drives the relief, so it replaces guessing depth from image brightness.
// ============================================================================

export interface LayerClass {
  id: string;
  label: string;
  /** Colour the user paints with. */
  rgb: [number, number, number];
  /** Height above the body surface, in "units" (1 unit ≈ 1.2% of the figure height). */
  offset: number;
  /** Extra softening of the edges (fur, hair). */
  soft: boolean;
}

export const LAYER_CLASSES: LayerClass[] = [
  { id: 'skin', label: 'Piel / cuerpo', rgb: [0, 255, 0], offset: 0, soft: false },
  { id: 'armor', label: 'Armadura', rgb: [255, 0, 0], offset: 1, soft: false },
  { id: 'cloth', label: 'Tela / ropa', rgb: [0, 0, 255], offset: 0.35, soft: false },
  { id: 'fur', label: 'Pelo / pelaje', rgb: [255, 255, 0], offset: 0.8, soft: true },
  { id: 'spikes', label: 'Pinchos / adornos', rgb: [255, 0, 255], offset: 1.8, soft: false },
  { id: 'hollow', label: 'Huecos / hendiduras', rgb: [0, 255, 255], offset: -0.6, soft: false },
];

/** Index into LAYER_CLASSES, or -1 for "no change" (any other colour). */
export function classifyLayerColor(r: number, g: number, b: number): number {
  let best = -1;
  let bestD = 110 * 110; // colours further than this from every swatch are ignored
  for (let k = 0; k < LAYER_CLASSES.length; k++) {
    const [cr, cg, cb] = LAYER_CLASSES[k].rgb;
    const d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2;
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  }
  return best;
}

/**
 * Converts RGBA pixels of the (already cropped and fitted) layer map into
 * per-pixel offsets in units and a "softness" flag. Pixels with no class get
 * NaN, meaning "leave the relief as it is".
 */
export function layerMapToOffsets(rgba: Uint8ClampedArray, offsets: Float32Array, soft: Uint8Array) {
  const n = offsets.length;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    if (rgba[o + 3] < 128) {
      offsets[i] = NaN;
      soft[i] = 0;
      continue;
    }
    const k = classifyLayerColor(rgba[o], rgba[o + 1], rgba[o + 2]);
    if (k < 0) {
      offsets[i] = NaN;
      soft[i] = 0;
    } else {
      offsets[i] = LAYER_CLASSES[k].offset;
      soft[i] = LAYER_CLASSES[k].soft ? 1 : 0;
    }
  }
}
