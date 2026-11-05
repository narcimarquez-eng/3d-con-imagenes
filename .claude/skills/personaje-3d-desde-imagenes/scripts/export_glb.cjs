#!/usr/bin/env node
// Drives the Vortex3D app headless: uploads the front/back pictures (and the
// optional painted layer maps), optionally enables the skeleton, and saves the
// exported .glb.
//
// Requires the dev server running (from the repo root):
//   npx vite --port=3000 --host=127.0.0.1
//
// Usage:
//   node export_glb.cjs --front frente.png --back espalda.png --out modelo.glb
//        [--layer-front capas_frente.png] [--layer-back capas_espalda.png]
//        [--rig] [--anim idle|walk|run|wave|none] [--url http://127.0.0.1:3000]
//        [--shots dir]   (also saves screenshots of the app)
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0) return def;
  const v = process.argv[i + 1];
  return v === undefined || v.startsWith('--') ? true : v;
}

(async () => {
  const front = arg('front');
  const back = arg('back');
  const out = arg('out', 'modelo.glb');
  const url = arg('url', 'http://127.0.0.1:3000');
  if (!front) {
    console.error('Falta --front');
    process.exit(2);
  }
  const abs = (p) => path.resolve(p);
  const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 }, acceptDownloads: true });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text().slice(0, 300));
  });
  await page.goto(url);
  await page.waitForTimeout(3000);

  // Hidden file inputs, in DOM order: 0 front, 1 back, 2 layer front, 3 layer back, 4 side.
  const inputs = page.locator('input[type=file]');
  await inputs.nth(0).setInputFiles(abs(front));
  await page.waitForTimeout(3000);
  if (back) {
    await inputs.nth(1).setInputFiles(abs(back));
    await page.waitForTimeout(4000);
  }
  const lf = arg('layer-front');
  const lb = arg('layer-back');
  if (lf) {
    await inputs.nth(2).setInputFiles(abs(lf));
    await page.waitForTimeout(3000);
  }
  if (lb) {
    await inputs.nth(3).setInputFiles(abs(lb));
    await page.waitForTimeout(3000);
  }
  if (arg('rig')) {
    await page.locator('label:has-text("Esqueleto y animaciones") input').click();
    await page.waitForTimeout(4000);
    const anim = arg('anim');
    const labels = { none: 'Pose de reposo', idle: 'Reposo', walk: 'Caminar', run: 'Correr', wave: 'Saludar' };
    if (anim && labels[anim]) {
      await page.locator('button', { hasText: new RegExp(`^${labels[anim]}$`) }).first().click();
      await page.waitForTimeout(1000);
    }
  }
  const shots = arg('shots');
  if (shots) {
    fs.mkdirSync(shots, { recursive: true });
    await page.screenshot({ path: path.join(shots, 'app.png') });
  }

  await page.locator('button', { hasText: /^\s*Descargas\s*$/ }).first().click();
  await page.waitForTimeout(800);
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.locator('button', { hasText: 'Modelo 3D (.GLB)' }).first().click();
  const d = await dl;
  await d.saveAs(abs(out));
  console.log('GLB guardado en', abs(out));
  if (errors.length) console.log('Errores de consola:', JSON.stringify(errors));
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
