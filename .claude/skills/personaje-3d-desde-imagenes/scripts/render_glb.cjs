#!/usr/bin/env node
// Renders a .glb from several angles (textured and "clay" grey) and, if it has
// animations, a few frames of each clip. Serves the model and three.js itself,
// so no other server is needed. Run from the repo root (needs node_modules/three).
//
// Usage:
//   node render_glb.cjs --glb modelo.glb --out renders/prefijo [--clay] [--anims] [--head]
// Outputs: <prefijo>_grid.png (frente, 3/4, perfil, espalda), _clay_grid.png,
//          _anims.png, _head.png
const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0) return def;
  const v = process.argv[i + 1];
  return v === undefined || v.startsWith('--') ? true : v;
}

const glb = path.resolve(arg('glb', 'modelo.glb'));
const out = path.resolve(arg('out', 'render'));
const threeDir = path.resolve(process.cwd(), 'node_modules/three');
const viewer = path.join(__dirname, 'viewer.html');
const types = { '.js': 'text/javascript', '.html': 'text/html', '.glb': 'model/gltf-binary' };

const server = http.createServer((req, res) => {
  const u = decodeURIComponent(req.url.split('?')[0]);
  let file = null;
  if (u === '/' || u === '/viewer.html') file = viewer;
  else if (u === '/model.glb') file = glb;
  else if (u.startsWith('/three/')) file = path.join(threeDir, u.slice('/three/'.length));
  if (!file || !file.startsWith(threeDir) && file !== viewer && file !== glb) {
    res.writeHead(404);
    return res.end();
  }
  fs.readFile(file, (err, data) => {
    if (err) {
      res.writeHead(404);
      return res.end();
    }
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
});

(async () => {
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  fs.mkdirSync(path.dirname(out), { recursive: true });
  const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 600, height: 800 } });
  await page.goto(`http://127.0.0.1:${port}/viewer.html`);
  await page.waitForFunction(() => window.ready, null, { timeout: 60000 });
  const info = await page.evaluate(() => window.info);
  console.log('Tamaño', info.size.map((v) => v.toFixed(3)).join(' x '), '| animaciones:', info.clips.join(', ') || 'ninguna');

  const views = [0, Math.PI / 4, Math.PI / 2, Math.PI];
  const grab = async () => page.screenshot({ type: 'png' });
  const sheet = async (bufs, file, cols) => {
    // Compose with the page itself (no image library needed).
    const urls = bufs.map((b) => 'data:image/png;base64,' + b.toString('base64'));
    const composed = await page.evaluate(async ({ urls, cols }) => {
      const imgs = await Promise.all(urls.map((u) => new Promise((r) => { const i = new Image(); i.onload = () => r(i); i.src = u; })));
      const w = imgs[0].width, h = imgs[0].height;
      const c = document.createElement('canvas');
      c.width = w * cols;
      c.height = h * Math.ceil(imgs.length / cols);
      const ctx = c.getContext('2d');
      imgs.forEach((im, k) => ctx.drawImage(im, (k % cols) * w, Math.floor(k / cols) * h));
      return c.toDataURL('image/png');
    }, { urls, cols });
    fs.writeFileSync(file, Buffer.from(composed.split(',')[1], 'base64'));
    console.log('Guardado', file);
  };

  const textured = [];
  for (const a of views) { await page.evaluate((a) => window.shot(a), a); textured.push(await grab()); }
  await sheet(textured, `${out}_grid.png`, 4);

  if (arg('clay')) {
    await page.evaluate(() => window.clay(true));
    const clay = [];
    for (const a of views) { await page.evaluate((a) => window.shot(a), a); clay.push(await grab()); }
    await sheet(clay, `${out}_clay_grid.png`, 4);
    await page.evaluate(() => window.clay(false));
  }

  if (arg('head')) {
    const hy = info.size[1] * 0.4;
    const heads = [];
    for (const clayOn of [true, false]) {
      await page.evaluate((on) => window.clay(on), clayOn);
      for (const a of [0, 0.75, Math.PI / 2]) {
        await page.evaluate(([a, hy]) => window.shot(a, 0, hy, 6), [a, hy]);
        heads.push(await grab());
      }
    }
    await page.evaluate(() => window.clay(false));
    await sheet(heads, `${out}_head.png`, 3);
  }

  if (arg('anims') && info.clips.length) {
    const frames = [];
    for (const clip of info.clips) {
      for (const t of [0.15, 0.45, 0.8]) {
        await page.evaluate(([n, t]) => window.anim(n, t), [clip, t]);
        await page.evaluate(() => window.shot(0.35));
        frames.push(await grab());
      }
    }
    await page.evaluate(() => window.anim('', 0));
    await sheet(frames, `${out}_anims.png`, 3);
  }

  await browser.close();
  server.close();
})().catch((e) => {
  console.error(e);
  server.close();
  process.exit(1);
});
