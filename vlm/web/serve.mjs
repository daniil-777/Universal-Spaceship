// vlm/web/serve.mjs — the demo server: the harness serve() for the site (this repo), plus /__vlm/{models,datasets,raw}/…
// read-only from LaCie (dataset records reference frames under raw/<run>/…; §9: frames are not copied). Everything else under
// the LaCie root (logs, venv, hf cache) stays private, as do exFAT ._* sidecars.
//   node vlm/web/serve.mjs [--port 0]   (never 8788, 8790 or 8791)
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
export const VLM_ROOT = '/Volumes/LaCie/astro-pilot/vlm';
const PUBLIC = ['models', 'datasets', 'raw'], RESERVED = [8788, 8790, 8791];
const TYPES = { '.json': 'application/json', '.jsonl': 'application/json', '.onnx': 'application/octet-stream', '.png': 'image/png', '.jpg': 'image/jpeg', '.txt': 'text/plain; charset=utf-8', '.md': 'text/markdown; charset=utf-8', '.u8': 'application/octet-stream' };
// the file a /__vlm/… path names, or null when it falls outside the allow-list
export function vlmFile(pathname, root = VLM_ROOT) {
  if (!pathname.startsWith('/__vlm/')) return null;
  const f = path.normalize(path.join(root, pathname.slice('/__vlm/'.length)));
  if (!PUBLIC.some((d) => f.startsWith(`${root}/${d}/`)) || path.basename(f).startsWith('._')) return null;
  return f;
}
export function checkPort(port) {
  if (RESERVED.includes(port)) throw new Error(`port ${port} is reserved`);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error(`bad port ${port}`);
  return port;
}
export async function startServer({ port = 0, root = VLM_ROOT } = {}) {
  checkPort(port);
  process.env.AP_SITE ||= fileURLToPath(new URL('../../', import.meta.url)).replace(/\/$/, '');
  const { serve } = await import('/Volumes/LaCie/astro-pilot/test/shot.mjs'), { server: inner } = await serve();
  const outer = http.createServer((req, res) => {
    let u; try { u = decodeURIComponent(new URL(req.url, 'http://x').pathname); } catch { res.writeHead(400); return res.end('bad path'); }
    if (!u.startsWith('/__vlm/')) return inner.emit('request', req, res);
    const f = vlmFile(u, root);
    if (req.method !== 'GET' || !f || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end('not found'); }
    res.writeHead(200, { 'content-type': TYPES[path.extname(f)] || 'application/octet-stream', 'content-length': fs.statSync(f).size, 'cache-control': 'no-store' });
    fs.createReadStream(f).pipe(res);
  });
  await new Promise((ok) => outer.listen(port, '127.0.0.1', ok));
  return { outer, inner, port: outer.address().port, close: () => { outer.close(); inner.close(); } };
}
if (process.argv[1] && import.meta.url === pathToFileURL(fs.realpathSync(process.argv[1])).href) {
  const pi = process.argv.indexOf('--port'), { port } = await startServer({ port: pi > 0 ? +process.argv[pi + 1] : 0 });
  console.log(`demo: http://127.0.0.1:${port}/vlm/web/demo.html`);
}
