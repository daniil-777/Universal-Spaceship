// chat/web/serve.mjs — CAPCOM's local static server: the repo (worktree) root, plus /__capcom/{models,kb,data}/… read-only from
// LaCie (chat/models, chat/kb, chat/data). Everything else under the LaCie root (logs, venv, hf cache, runs) stays private, as do
// dotfiles and exFAT ._* sidecars. --isolate adds COOP/COEP (crossOriginIsolated → multi-threaded WASM; GitHub Pages cannot).
//   node chat/web/serve.mjs [--port 0] [--isolate]   → http://127.0.0.1:<port>/chat/web/demo.html   (never 8788/8790/8791/8792/8766)
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
export const CHAT_ROOT = '/Volumes/LaCie/astro-pilot/chat';
export const SITE_ROOT = fileURLToPath(new URL('../../', import.meta.url)).replace(/\/$/, '');
const PUBLIC = ['models', 'kb', 'data'], RESERVED = [8788, 8790, 8791, 8792, 8766], PREFIX = '/__capcom/';
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json', '.jsonl': 'application/json', '.jinja': 'text/plain; charset=utf-8', '.css': 'text/css', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.onnx_data': 'application/octet-stream',
  '.txt': 'text/plain; charset=utf-8', '.md': 'text/markdown; charset=utf-8' };
const hidden = (f, root) => path.relative(root, f).split(path.sep).some((p) => p.startsWith('.'));

// the file a request path names, or null when it falls outside the allow-list
export function resolveFile(pathname, { site = SITE_ROOT, chat = CHAT_ROOT } = {}) {
  const cap = pathname.startsWith(PREFIX), root = cap ? chat : site;
  const f = path.normalize(path.join(root, cap ? pathname.slice(PREFIX.length) : pathname));
  if ((f !== root && !f.startsWith(root + path.sep)) || hidden(f, root)) return null;
  if (cap && !PUBLIC.includes(path.relative(root, f).split(path.sep)[0])) return null;
  return f;
}
export function checkPort(port) {
  if (RESERVED.includes(port)) throw new Error(`port ${port} is reserved`);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error(`bad port ${port}`);
  return port;
}
export async function startServer({ port = 0, isolate = false, site = SITE_ROOT, chat = CHAT_ROOT } = {}) {
  checkPort(port);
  const server = http.createServer((req, res) => {
    let u; try { u = decodeURIComponent(new URL(req.url, 'http://x').pathname); } catch { res.writeHead(400); return res.end('bad path'); }
    if (u.endsWith('/')) u += 'index.html';
    const f = resolveFile(u, { site, chat });
    let st = null; try { st = f && fs.statSync(f); } catch { /* missing */ }
    if (!['GET', 'HEAD'].includes(req.method) || !st || st.isDirectory()) { res.writeHead(404); return res.end('not found'); }
    const h = { 'content-type': TYPES[path.extname(f)] || 'application/octet-stream', 'content-length': st.size, 'cache-control': 'no-store' };
    if (isolate) Object.assign(h, { 'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'credentialless' });
    res.writeHead(200, h);
    if (req.method === 'HEAD') return res.end();
    fs.createReadStream(f).pipe(res);
  });
  await new Promise((ok) => server.listen(port, '127.0.0.1', ok));
  return { server, port: server.address().port, close: () => new Promise((ok) => server.close(ok)) };
}
if (process.argv[1] && import.meta.url === pathToFileURL(fs.realpathSync(process.argv[1])).href) {
  const pi = process.argv.indexOf('--port'), { port } = await startServer({ port: pi > 0 ? +process.argv[pi + 1] : 0, isolate: process.argv.includes('--isolate') });
  console.log(`CAPCOM demo: http://127.0.0.1:${port}/chat/web/demo.html`);
}
