---
name: personaje-3d-desde-imagenes
description: Convierte imágenes de un personaje (frente + espalda, opcionalmente con mapas de capas pintados) en un modelo 3D .glb con la app Vortex3D de este repositorio, incluyendo cuerpo humano ajustado a la silueta, cara, botas, esqueleto y animaciones. Úsala cuando el usuario quiera generar, probar o mejorar un .glb a partir de imágenes, procesar una hoja de referencia, pintar o interpretar un mapa de capas, o tocar el motor de reconstrucción (humanBodyTemplate, humanRig, layerMap, exportación GLB).
---

# Personaje 3D desde imágenes (Vortex3D)

El usuario habla español y suele trabajar con personajes de fantasía (guerreros, orcos) en pose A, con vista de frente y de espalda. Respóndele en español, con poca jerga.

## Flujo rápido (de imágenes a .glb)

Todo se ejecuta desde la raíz del repo. Los scripts están en `.claude/skills/personaje-3d-desde-imagenes/scripts/`.

1. `npm install` (o `bun install`) si falta `node_modules`.
2. Arranca la app en segundo plano: `npx vite --port=3000 --host=127.0.0.1`.
3. Si te dan **una hoja con cuatro paneles** (foto frente, foto espalda, mapa frente, mapa espalda, de izquierda a derecha):
   `python3 scripts/split_sheet.py hoja.jpg salida/ --prefix orco`
   Separa los paneles, quita títulos y etiquetas y alinea cada mapa con su foto píxel a píxel (busca el desplazamiento por solapamiento de siluetas; revisa el IoU que imprime, > 0,7 es bueno).
   Después, **siempre**: `python3 scripts/limpiar_paneles.py salida/orco_*.png`, que deja en cada panel solo el personaje y borra los trozos del panel vecino que se cuelan por el borde. Un trozo de la mano del dibujo de al lado basta para que esa fila cuente como "cuerpo" de lado a lado y el modelo salga con una repisa plana atravesándolo.
4. Exporta (usa los `_limpio.png`):
   `node scripts/export_glb.cjs --front f.png --back b.png [--layer-front lf.png --layer-back lb.png] [--rig --anim walk] --out modelo.glb`
5. Revisa el resultado (siempre, antes de enseñarlo):
   `node scripts/render_glb.cjs --glb modelo.glb --out renders/m --clay --head --anims`
   Mira `*_grid.png` (con textura), `*_clay_grid.png` (geometría en gris: es la que delata los fallos de forma), `*_head.png` y `*_anims.png`.
6. `python3 scripts/inspect_glb.py modelo.glb --atlas atlas.png` da huesos, animaciones y piezas de malla. Lo normal son 4 piezas: frente y espalda del cuerpo (se cuentan aparte porque los vértices del contorno están duplicados por las UV) y una por bota. Una pieza pequeña extra es una isla flotante, o sea, un fallo.
7. Para enseñar resultados: compón imágenes antes/después y envíalas junto con el .glb.

El visor neutro no tiene mapa de entorno, así que el metal sale más oscuro que en la app. Para juzgar la forma, usa la vista en gris.

## Mapa de capas (relieve pintado)

Es una copia de cada foto (mismo encuadre; la resolución puede ser otra) pintada con colores planos. **La textura sigue saliendo de la foto**; el mapa solo dice qué sobresale. Se clasifica **por tono**, no por color exacto (`src/engine/layerMap.ts`):

| Color | Clase | Altura (1 unidad ≈ 1,2% de la altura del personaje) |
|---|---|---|
| Verde vivo | Piel / cuerpo | 0 |
| Rojo | Armadura | +1 |
| Azul | Tela / ropa | +0,35 |
| Amarillo | Pelo / pelaje (borde suave) | +0,8 |
| Magenta | Pinchos / adornos | +1,8 |
| Cian | Huecos / juntas (se hunden) | −0,6 |
| Gris medio o verde oscuro | Cuero / correas | +0,5 |
| Gris claro, blanco o negro | Sin cambio (fondo) | — |

Además, lo pintado **define la silueta**: se une al recorte automático (recupera cuernos y pinchos finos) y el recorte solo se acepta a 2 px o menos de lo pintado (quita sombras y suelo). El mapa trasero se refleja en horizontal para encajar con la vista trasera. Hay que subir la foto **antes** que su mapa, porque el mapa se recorta con el mismo rectángulo que la foto. Si no hay mapa pintado, `scripts/make_layer_map.py` genera uno aproximado por color (útil para pruebas; uno pintado a mano es mucho mejor).

En la app: pestaña **Moldes**, panel "Cuerpo humano ajustado a la silueta", sección "Mapa de capas"; el deslizador "Altura de capas" lo escala.

## Arquitectura (qué toca qué)

- `src/App.tsx`: interfaz. `drawImageFittedToTensor` recorta la imagen al personaje y la encaja **sin deformar** en el cuadrado de 256×256 (rellena con el color de fondo de la imagen) y devuelve el recorte; `loadLayerMapIntoBuffers` aplica ese mismo recorte al mapa, sin suavizado. Los paneles nuevos están en la pestaña Moldes (cuerpo humano, rasgos de la cara, mapa de capas, esqueleto y animaciones) y en Suavizado (inflado, contorno suave, suavizado de relieve: solo afectan al modo antiguo).
- `src/engine/cnnReconstruction.ts`: motor original (relieve a partir del brillo, Poisson, atlas de texturas 512: frente en el cuadrante de arriba a la izquierda, espalda arriba a la derecha). Al final de `buildReconstructedMeshGeometry`, si el molde es **Humano Pose A** y `humanBodyTemplate !== false`, sustituye la malla por la del cuerpo humano (reutiliza el atlas).
- `src/engine/humanBodyTemplate.ts`: **cuerpo humano** cerrado ajustado a la silueta. Ahí están los puntos de referencia, el mapa de profundidad por partes, la cara, las botas, la reparación del borde del atlas, la malla y las islas.
- `src/engine/humanRig.ts`: esqueleto de 19 huesos con nombres estándar (Hips, Spine, Chest, Neck, Head, Left/Right Shoulder, Arm, ForeArm, Hand, UpLeg, Leg, Foot; izquierda y derecha son las del personaje), pesos de piel y animaciones procedurales (Reposo, Caminar, Correr, Saludar).
- `src/engine/layerMap.ts`: paleta y clasificación de los mapas de capas.
- `src/engine/exportEngine.ts`: exportación a GLB (convierte las DataTextures a canvas, quita la rotación de exhibición del visor, clona con SkeletonUtils, deja los huesos en pose de reposo e incluye las animaciones).
- `src/components/StudioViewport.tsx`: visor (crea una SkinnedMesh con AnimationMixer cuando el esqueleto está activo; cambiar de animación o velocidad no regenera la malla).

## Cómo se construye el cuerpo (y por qué)

1. **Puntos de referencia** (en píxeles de la imagen de 256):
   - Cráneo: anchura mediana de la cabeza entre el 2,5% y el 7% de la altura (por encima de orejas y cuellos de piel).
   - Barbilla estimada: arriba + 1,35 × anchura del cráneo.
   - Hombros: no pueden empezar antes de una anchura de cráneo desde arriba, y exigen que la anchura sea más de 1,6 veces la máxima de la cabeza durante 3 filas. Así las puntas de las hombreras junto a la cabeza no engañan.
   - Axila: primera fila con brazos separados a ambos lados y torso de menos del 80% del ancho de hombros.
   - Entrepierna: hueco entre las piernas, como mucho al 56% de la altura (la tela entre las piernas no cuenta como torso).
2. **Cara detectada en la imagen** (`detectFace`):
   - Ojos: la fila con dos manchas oscuras simétricas, más oscuras que la ceja de arriba y el pómulo de abajo. Da la altura, el centro y la separación de los ojos.
   - Barbilla: se muestrea el color de la piel bajo los ojos y se baja hasta que ese color deja de llenar la cara.
   - Si no hay confianza, se usan proporciones (ojos al 56% de la cabeza, porque pelo o casco ocupan la parte de arriba).
3. **Profundidad por partes**:
   - Cabeza: óvalo 1,08 veces más profundo que ancho, limitado al ancho del cráneo.
   - Torso: sección superelíptica con relación profundidad/ancho de 0,8 en los hombros, 0,74 en el pecho, 0,64 en la cintura y 0,7 en la cadera, con tope de 0,85 × semiancho del pecho (para faldas que se abren).
   - Piernas: dos cilindros, aunque haya tela entre ellas.
   - Brazos y hombreras: redondos según **su propia forma** (silueta menos el núcleo del cuerpo), unidos al torso con un máximo suave.
   - Todo se difumina un poco para que no haya escalones entre partes.
4. **Detalle**: relieve de la foto (filtrado paso alto, poco), rasgos faciales sutiles colocados según ojos y barbilla (cejas, cuencas, nariz, pómulos, boca, mentón) y mapa de capas.
5. **Malla**:
   - Rejilla 2,2 veces más fina que la de la app.
   - Los vértices del borde se ajustan al contorno real con precisión de subpíxel y se suaviza la línea del contorno.
   - Frente y espalda se unen **directamente en el contorno**, sin tira de junta, con normales compartidas.
   - Solo se conserva el trozo conectado más grande.
6. **Botas**: largo del 15,5% de la altura, con el tobillo a un cuarto de la longitud desde el talón y la puntera más baja y algo afilada que siempre sobresale de la pierna. El **ancho es el de la pierna**, medido en la espinilla por encima del tobillo (no en las filas de abajo, donde sandalias, dedos y sombras lo ensanchan). Van unidas al hueso del pie.
7. **Textura**: el borde del atlas se repara (fondo y grises de mezcla dentro de una banda de 6 px se rellenan con el color del cuerpo; la espalda es más estricta) y las UV del borde se desplazan un poco hacia dentro. El mapa de metal y rugosidad se difumina para quitar chispas.
8. **Sombras del suelo**: sin mapa, las franjas finas pegadas al suelo entre las piernas (de menos del 3,5% de la altura) se eliminan, porque si no se estiran al caminar.

## Lecciones (errores ya resueltos; no reintroducirlos)

- **Exportación GLB:** `GLTFExporter` falla con las `DataTexture` cuando tiene que redibujar el mapa de normales; hay que convertirlas a canvas antes. La rotación de exhibición del visor no debe ir al GLB.
- **Imágenes verticales:** no se estiran a un cuadrado; se encajan conservando la proporción, recortadas al personaje.
- **Gris en las juntas:** venía de la tira lateral y de la imagen de espalda reajustada, que mete fondo y trazo en el borde. La solución es no tener tira lateral y reparar el borde del atlas.
- **Escalón de las hombreras:** salía de un detector de axila engañado por las puntas de las hombreras. Para depurar la profundidad, vuelca el mapa (`debugHumanDepthMap`, ejecutable con `npx tsx`) y míralo como imagen.
- **Barbilla:** los cuellos de piel tapan la silueta del cuello; la barbilla se saca del cráneo o de la cara detectada, no de la silueta.
- **Mapas pintados en JPEG:** los colores no son puros; el verde de piel del usuario era (12,156,12). Por eso se clasifica por tono.
- **Índices de fila:** redondéalos siempre (un `neckY` con decimales rompió `runsByRow`).
- **Pruebas sin navegador:** pueden engañar si el recorte no es idéntico al de la app; para datos fiables, lee los resultados desde la app.
- **Restos del panel vecino:** al separar una hoja de cuatro paneles, un trozo del dibujo de al lado pegado al borde se cuela en el recorte. Si ese trozo es gris medio o de color, el mapa de capas lo cuenta como cuerpo, la fila pasa a ocupar todo el ancho y el modelo sale con una repisa plana a esa altura. Pásale siempre `limpiar_paneles.py` y comprueba que ninguna fila del mapa ocupa más del 95% del ancho.
- **Animación Saludar:** la hombrera se queda en su sitio y la axila se estira. Es una limitación conocida.

## Pruebas y entrega

- Antes de subir: `npx tsc --noEmit` y `npx vite build`.
- Para validar cambios del motor, prueba al menos dos personajes distintos (por ejemplo, el guerrero de armadura dorada y un orco) y revisa la vista en gris de frente, 3/4, perfil y espalda.
- Si tocas el esqueleto, comprueba que el GLB trae 19 huesos y 4 animaciones (`inspect_glb.py`) y renderiza fotogramas con `--anims`.
- La resolución de trabajo es de 256 px por vista; es el límite principal del detalle y la mejora pendiente más grande.
