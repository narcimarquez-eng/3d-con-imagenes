"""Quita de cada panel los restos del panel vecino pegados al borde.

Se queda con los trozos de dibujo que tocan la franja central (donde está el
personaje) y pinta el resto del color del fondo.
"""
import sys
import numpy as np
from PIL import Image

def etiquetar(fg):
    """Etiqueta trozos conectados (8-vecinos) sin scipy."""
    h, w = fg.shape
    lab = np.zeros((h, w), np.int32)
    tam = {}
    cur = 0
    pila = []
    for sy in range(h):
        for sx in range(w):
            if not fg[sy, sx] or lab[sy, sx]:
                continue
            cur += 1
            n = 0
            pila.append((sy, sx))
            lab[sy, sx] = cur
            while pila:
                y, x = pila.pop()
                n += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = cur
                            pila.append((ny, nx))
            tam[cur] = n
    return lab, cur, tam


def limpiar(path, out, margen=0.62):
    im = Image.open(path).convert('RGB')
    a = np.asarray(im).astype(int)
    # fondo = color de las esquinas
    esq = np.concatenate([a[:6, :6].reshape(-1, 3), a[:6, -6:].reshape(-1, 3),
                          a[-6:, :6].reshape(-1, 3), a[-6:, -6:].reshape(-1, 3)])
    bg = np.median(esq, axis=0)
    fg = (np.abs(a - bg).sum(axis=2) > 40)
    lab, n, tam = etiquetar(fg)
    h, w = fg.shape
    x0, x1 = int(w * (1 - margen) / 2), int(w * (1 + margen) / 2)
    centro = set(np.unique(lab[:, x0:x1])) - {0}
    buenos = {i for i in centro if tam[i] > fg.sum() * 0.01}
    keep = np.isin(lab, list(buenos))
    limpio = a.copy()
    limpio[~keep] = bg
    quitado = int((fg & ~keep).sum())
    Image.fromarray(limpio.astype('uint8')).save(out)
    print(f'{out}: {n} trozos, {len(buenos)} conservados, {quitado} px limpiados')

for p in sys.argv[1:]:
    limpiar(p, p.replace('.png', '_limpio.png'))
