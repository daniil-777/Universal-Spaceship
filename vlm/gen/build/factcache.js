// vlm/gen/build/factcache.js — the per-record derived facts of the build (image facts, the dHash, the Z geo facts and tags),
// cached so a rebuild only computes new or changed records. An entry is keyed by the record key and valid for one record
// stamp (the raw record file's size and mtime: a re-captured record misses) under one code sha (codeShaOf the modules that
// compute the facts: new code, new cache folder). One append-only JSONL per code sha; a later line wins.
//   <dir>/<codeSha>/derived.jsonl   {k: key, s: stamp, v: value}
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

export function codeShaOf(files) {
  const h = crypto.createHash('sha256');
  for (const f of [...files].sort()) { h.update(path.basename(f)); h.update('\0'); h.update(fs.readFileSync(f)); h.update('\0'); }
  return h.digest('hex').slice(0, 12);
}
export const stampOf = (file) => { const s = fs.statSync(file); return `${s.size}:${Math.round(s.mtimeMs)}`; };
export function openFactCache(dir, codeSha) {
  const folder = path.join(dir, codeSha), file = path.join(folder, 'derived.jsonl'), map = new Map();
  if (fs.existsSync(file)) {
    for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
      if (!line.trim()) continue;
      try { const e = JSON.parse(line); map.set(e.k, e); } catch { /* a torn last line from an interrupted build */ }
    }
  }
  let buf = [];
  return {
    get(key, stamp) { const e = map.get(key); return e && e.s === stamp ? e.v : undefined; },
    put(key, stamp, value) { const e = { k: key, s: stamp, v: value }; map.set(key, e); buf.push(JSON.stringify(e)); if (buf.length >= 500) this.flush(); },
    flush() { if (!buf.length) return; fs.mkdirSync(folder, { recursive: true }); fs.appendFileSync(file, buf.join('\n') + '\n'); buf = []; },
    get size() { return map.size; },
    file,
  };
}
