#!/usr/bin/env python3
"""
Makes a ROUGH layer map ("mapa de capas") from a picture by colour, as a
starting point or for quick tests. A map painted by hand is always better.

Classes written (same colours the app understands):
  green  (0,255,0)   skin      -> hue in --skin-hue range, saturated
  blue   (0,0,255)   cloth     -> optional --cloth-hue range
  gray   (110,110,100) leather -> brown, mid saturation
  red    (255,0,0)   armour    -> everything else on the character
  background stays light grey (= no change)

Usage:
  python3 make_layer_map.py foto.png capas.png [--skin-hue 60 150] [--cloth-hue 200 250]
Skin hue examples: green orc 60-150, human skin 5-35 (then leather/armour
detection is less reliable).
"""
import argparse
import numpy as np
from PIL import Image, ImageFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--skin-hue', nargs=2, type=float, default=[60, 150])
    ap.add_argument('--cloth-hue', nargs=2, type=float, default=None)
    args = ap.parse_args()

    im = Image.open(args.src).convert('RGB')
    a = np.asarray(im).astype(float) / 255
    corners = np.concatenate([a[:10, :10].reshape(-1, 3), a[-10:, -10:].reshape(-1, 3),
                              a[:10, -10:].reshape(-1, 3), a[-10:, :10].reshape(-1, 3)])
    bg = np.median(corners, 0)
    is_bg = np.sqrt(((a - bg) ** 2).sum(2)) < 0.09
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(2), a.min(2)
    d = np.maximum(mx - mn, 1e-6)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60

    out = np.zeros(a.shape, dtype=np.uint8)
    out[...] = (200, 200, 200)
    fig = ~is_bg
    skin = fig & (hue >= args.skin_hue[0]) & (hue <= args.skin_hue[1]) & (sat > 0.18) & (mx > 0.12)
    leather = fig & ~skin & (hue > 10) & (hue < 40) & (sat > 0.25) & (mx > 0.12) & (mx < 0.5)
    cloth = np.zeros_like(skin)
    if args.cloth_hue:
        cloth = fig & ~skin & (hue >= args.cloth_hue[0]) & (hue <= args.cloth_hue[1]) & (sat > 0.2)
    armor = fig & ~skin & ~leather & ~cloth
    out[armor] = (255, 0, 0)
    out[leather] = (110, 110, 100)
    out[cloth] = (0, 0, 255)
    out[skin] = (0, 255, 0)
    Image.fromarray(out).filter(ImageFilter.ModeFilter(5)).save(args.dst)
    print('Guardado', args.dst)


if __name__ == '__main__':
    main()
