"""Separa una hoja de 2 filas x 4 columnas (natural arriba, codificada abajo).

Busca los huecos de fondo entre columnas y entre filas, recorta cada panel al
dibujo y quita los rótulos de debajo.
"""
import sys
import numpy as np
from PIL import Image

def componentes(fg):
    h, w = fg.shape
    lab = np.zeros((h, w), np.int32); cur = 0; tam = {}
    for sy in range(h):
        for sx in range(w):
            if not fg[sy, sx] or lab[sy, sx]:
                continue
            cur += 1; n = 0; pila = [(sy, sx)]; lab[sy, sx] = cur
            while pila:
                y, x = pila.pop(); n += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = cur; pila.append((ny, nx))
            tam[cur] = n
    return lab, tam

src = sys.argv[1]; out = sys.argv[2]; pref = sys.argv[3]
im = Image.open(src).convert('RGB')
a = np.asarray(im).astype(int)
H, W, _ = a.shape
bg = np.median(np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3)]), axis=0)
fg = np.abs(a - bg).sum(axis=2) > 45
print('hoja', im.size, 'fondo', bg)

# Filas: dos bandas separadas por una franja vacía
filas_llenas = fg.sum(axis=1) > W * 0.004
bandas = []
y = 0
while y < H:
    if filas_llenas[y]:
        y0 = y
        while y < H and filas_llenas[y]:
            y += 1
        if y - y0 > H * 0.05:
            bandas.append((y0, y))
    else:
        y += 1
print('bandas horizontales:', bandas)

lab, tam = componentes(fg)
grandes = [i for i, n in tam.items() if n > fg.sum() * 0.012]
cajas = []
for i in grandes:
    ys, xs = np.where(lab == i)
    cajas.append((xs.min(), ys.min(), xs.max(), ys.max(), len(xs)))
cajas.sort(key=lambda c: (c[1], c[0]))
print('trozos grandes:', len(cajas))

paneles = []
for (y0, y1) in bandas[:2]:
    # dentro de la banda, agrupa los trozos por columnas
    dentro = [c for c in cajas if c[1] >= y0 - 5 and c[3] <= y1 + 5]
    dentro.sort(key=lambda c: c[0])
    grupos = []
    for c in dentro:
        if grupos and c[0] - grupos[-1][2] < W * 0.03:
            g = grupos[-1]
            grupos[-1] = (min(g[0], c[0]), min(g[1], c[1]), max(g[2], c[2]), max(g[3], c[3]))
        else:
            grupos.append((c[0], c[1], c[2], c[3]))
    grupos = [g for g in grupos if (g[2] - g[0]) > W * 0.03 and (g[3] - g[1]) > H * 0.08]
    print(f'  banda {y0}-{y1}: {len(grupos)} paneles')
    paneles.append(grupos)

nombres = ['frente', 'perfil_izq', 'perfil_der', 'espalda']
for fila, (grupos, suf) in enumerate(zip(paneles, ['', '_capas'])):
    for i, (x0, y0, x1, y1) in enumerate(grupos):
        m = 6
        rec = im.crop((max(0, x0 - m), max(0, y0 - m), min(W, x1 + m + 1), min(H, y1 + m + 1)))
        nom = f'{out}/{pref}_{nombres[i]}{suf}.png'
        rec.save(nom)
        print('  ', nom, rec.size)
