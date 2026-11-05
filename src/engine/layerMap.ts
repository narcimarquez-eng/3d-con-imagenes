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
  { id: 'leather', label: 'Cuero / correas (gris o verde oscuro)', rgb: [110, 110, 100], offset: 0.5, soft: false },
];

const classIndex = (id: string) => LAYER_CLASSES.findIndex((c) => c.id === id);

/**
 * Index into LAYER_CLASSES, or -1 for "no change".
 * Classified by hue rather than exact colour, so any red counts as armour,
 * any bright green as skin, etc. (and JPEG noise does not matter):
 *   red → armour, yellow → fur/hair, bright green → skin, cyan → hollows,
 *   blue → cloth, magenta → spikes, dark green or mid grey → leather/straps.
 * Light greys/whites (typical backgrounds) and black mean no change.
 */
export function classifyLayerColor(r: number, g: number, b: number): number {
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  const v = mx / 255;
  const sat = mx > 0 ? (mx - mn) / mx : 0;
  if (sat < 0.18) {
    // Greys: mid grey = leather/straps, light grey/white/black = nothing.
    return v > 0.25 && v < 0.58 ? classIndex('leather') : -1;
  }
  if (v < 0.2) return -1;
  const d = mx - mn;
  let hue = 0;
  if (mx === r) hue = (((g - b) / d) % 6) * 60;
  else if (mx === g) hue = ((b - r) / d + 2) * 60;
  else hue = ((r - g) / d + 4) * 60;
  if (hue < 0) hue += 360;
  if (hue < 25 || hue >= 335) return classIndex('armor');
  if (hue < 75) return sat > 0.35 ? classIndex('fur') : classIndex('leather');
  if (hue < 165) return v > 0.45 && sat > 0.35 ? classIndex('skin') : classIndex('leather');
  if (hue < 200) return classIndex('hollow');
  if (hue < 265) return classIndex('cloth');
  return classIndex('spikes');
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
