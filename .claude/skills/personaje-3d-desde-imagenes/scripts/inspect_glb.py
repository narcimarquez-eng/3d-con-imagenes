#!/usr/bin/env python3
"""
Quick checks on an exported .glb:
  - size, skins/joints, animation clips and channel counts, mesh attributes
  - connected mesh pieces (front/back halves of the body and each boot show
    up as separate pieces; tiny extra pieces are floating specks = a bug)
  - saves the colour atlas (front quadrant | back quadrant) with --atlas out.png

Usage:
  python3 inspect_glb.py modelo.glb [--atlas atlas.png]
"""
import sys
import io
import json
import struct
import numpy as np
from PIL import Image


def load(path):
    b = open(path, 'rb').read()
    L = struct.unpack('<I', b[12:16])[0]
    j = json.loads(b[20:20 + L])
    return j, b[20 + L + 8:]


def accessor(j, bin_, i):
    a = j['accessors'][i]
    bv = j['bufferViews'][a['bufferView']]
    dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a['componentType']]
    n = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[a['type']]
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    return np.frombuffer(bin_[off:off + a['count'] * n * np.dtype(dt).itemsize], dtype=dt).reshape(-1, n)


def main():
    path = sys.argv[1]
    j, bin_ = load(path)
    prim = j['meshes'][0]['primitives'][0]
    pos = accessor(j, bin_, prim['attributes']['POSITION'])
    idx = accessor(j, bin_, prim['indices']).reshape(-1, 3)
    print('vertices', len(pos), 'triangulos', len(idx))
    print('tamaño', (pos.max(0) - pos.min(0)).round(3))
    print('atributos', list(prim['attributes'].keys()))
    skins = j.get('skins', [])
    print('esqueleto', f"{len(skins[0]['joints'])} huesos" if skins else 'no')
    print('animaciones', [(a.get('name'), len(a['channels'])) for a in j.get('animations', [])])

    parent = np.arange(len(pos))

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t in idx:
        a, b, c = f(t[0]), f(t[1]), f(t[2])
        parent[b] = a
        parent[c] = a
    used = np.unique(idx)
    roots = np.array([f(i) for i in used])
    _, counts = np.unique(roots, return_counts=True)
    print('piezas (vertices):', sorted(counts.tolist(), reverse=True)[:10])

    if '--atlas' in sys.argv:
        out = sys.argv[sys.argv.index('--atlas') + 1]
        m = j['materials'][0]['pbrMetallicRoughness']
        src = j['textures'][m['baseColorTexture']['index']]['source']
        bv = j['bufferViews'][j['images'][src]['bufferView']]
        im = Image.open(io.BytesIO(bin_[bv.get('byteOffset', 0):bv.get('byteOffset', 0) + bv['byteLength']]))
        im.convert('RGB').crop((0, 0, im.width, im.height // 2)).save(out)
        print('atlas guardado en', out)


if __name__ == '__main__':
    main()
