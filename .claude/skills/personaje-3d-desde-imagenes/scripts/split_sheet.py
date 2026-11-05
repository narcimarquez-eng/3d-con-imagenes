#!/usr/bin/env python3
"""
Splits a reference sheet with four panels laid out left to right:
    [foto frente] [foto espalda] [mapa capas frente] [mapa capas espalda]
into four images with exactly the same framing per pair, so each layer map
lines up pixel by pixel with its picture (the app crops a map with the same
rectangle as its picture).

- Photos are found on the left half, maps on the right half.
- The horizontal offset between photos and maps is searched by maximising
  the overlap (IoU) of the silhouettes (photo: dark or saturated pixels,
  map: saturated pixels).
- Title/label rows (e.g. "Pose A", "Vista de frente") are trimmed by taking
  the rows that contain the figures.

Usage:
  python3 split_sheet.py hoja.jpg salida/ [--prefix personaje]
Writes salida/<prefix>_frente.png, _espalda.png, _capas_frente.png, _capas_espalda.png
"""
import sys
import os
import argparse
import numpy as np
from PIL import Image


def fg_photo(a):
    sat = a.max(2) - a.min(2)
    lum = a.mean(2)
    return (sat > 45) | (lum < 95)


def fg_map(a):
    return (a.max(2) - a.min(2)) > 80


def empty_runs(cols, min_len=10):
    runs, start = [], None
    for x, c in enumerate(cols):
        if c < 3 and start is None:
            start = x
        if c >= 3 and start is not None:
            if x - start >= min_len:
                runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, len(cols) - 1))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sheet')
    ap.add_argument('outdir')
    ap.add_argument('--prefix', default='personaje')
    args = ap.parse_args()

    im = Image.open(args.sheet).convert('RGB')
    a = np.asarray(im).astype(float)
    H, W = a.shape[:2]
    half = W // 2

    # Rows holding the figures: the tallest block of rows with foreground.
    # Titles ("Pose A") and labels ("Vista de frente") are short blocks
    # separated by empty bands, so they are left out.
    pf = fg_photo(a[:, :half])
    on = pf.sum(1) > max(3, W * 0.004)
    blocks, start, gap = [], None, 0
    for y, v in enumerate(on):
        if v:
            if start is None:
                start = y
            gap = 0
            last = y
        elif start is not None:
            gap += 1
            if gap > 12:
                blocks.append((start, last))
                start, gap = None, 0
    if start is not None:
        blocks.append((start, last))
    b0, b1 = max(blocks, key=lambda r: r[1] - r[0])
    y0 = max(0, b0 - 8)
    y1 = min(H, b1 + 8)
    p = a[y0:y1]
    photo = fg_photo(p)
    mapm = fg_map(p)

    # Split between the two photos: the widest empty column run near the quarter.
    runs = empty_runs(photo[:, :half].sum(0))
    mid = half // 2
    split = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - mid)) if runs else (mid, mid)
    sx = (split[0] + split[1]) // 2

    # Offset photos -> maps.
    best = (-1, half, 0)
    P = photo[:, :sx]
    for dx in range(half - 60, half + 61):
        if dx + sx > W:
            continue
        M = mapm[:, dx:dx + sx]
        for dy in range(-6, 7):
            Ms = np.roll(M, dy, 0)
            iou = (P & Ms).sum() / max(1, (P | Ms).sum())
            if iou > best[0]:
                best = (iou, dx, dy)
    iou, dx, dy = best
    print(f'filas {y0}-{y1}, corte fotos x={sx}, desplazamiento mapas dx={dx} dy={dy} (IoU {iou:.2f})')

    os.makedirs(args.outdir, exist_ok=True)
    pre = os.path.join(args.outdir, args.prefix)
    im.crop((0, y0, sx, y1)).save(pre + '_frente.png')
    im.crop((sx, y0, half, y1)).save(pre + '_espalda.png')
    im.crop((dx, y0 - dy, dx + sx, y1 - dy)).save(pre + '_capas_frente.png')
    im.crop((dx + sx, y0 - dy, dx + half, y1 - dy)).save(pre + '_capas_espalda.png')
    print('Guardado', pre + '_{frente,espalda,capas_frente,capas_espalda}.png')


if __name__ == '__main__':
    main()
