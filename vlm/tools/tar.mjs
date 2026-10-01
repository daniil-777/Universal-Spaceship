// vlm/tools/tar.mjs — a minimal ustar writer for WebDataset shards (flat members <key>.<ext>, regular files only) that also
// SHA-256s everything it writes, and a reader for the tests. Python's tarfile, GNU tar and bsdtar read what it writes.
import crypto from 'node:crypto';
import fs from 'node:fs';

const BLOCK = 512;
function header(name, size, mtime) {
  if (Buffer.byteLength(name) > 100 || name.includes('/')) throw new Error(`tar member name ${JSON.stringify(name)}: flat names of at most 100 bytes only`);
  const h = Buffer.alloc(BLOCK), oct = (n, w) => n.toString(8).padStart(w - 1, '0') + '\0';
  h.write(name, 0); h.write(oct(0o644, 8), 100); h.write(oct(0, 8), 108); h.write(oct(0, 8), 116); h.write(oct(size, 12), 124); h.write(oct(mtime, 12), 136);
  h.write('        ', 148); h.write('0', 156); h.write('ustar\0', 257); h.write('00', 263);
  let sum = 0; for (const b of h) sum += b;
  h.write(sum.toString(8).padStart(6, '0') + '\0 ', 148);
  return h;
}
// a shard being written: add(name, bytes) appends one member; close() writes the two zero blocks and returns {bytes, sha256}
export function createTar(file, { mtime = 0 } = {}) {
  const fd = fs.openSync(file, 'w'), hash = crypto.createHash('sha256');
  let bytes = 0;
  const put = (b) => { fs.writeSync(fd, b); hash.update(b); bytes += b.length; };
  return {
    add(name, data) {
      const buf = Buffer.isBuffer(data) ? data : Buffer.from(data);
      put(header(name, buf.length, mtime)); put(buf);
      const pad = (BLOCK - (buf.length % BLOCK)) % BLOCK; if (pad) put(Buffer.alloc(pad));
    },
    get bytes() { return bytes; },
    close() { put(Buffer.alloc(2 * BLOCK)); fs.closeSync(fd); return { bytes, sha256: hash.digest('hex') }; },
  };
}
// [{name, data}] of a tar written by createTar (checksums verified)
export function readTar(file) {
  const b = fs.readFileSync(file), out = [];
  for (let o = 0; o + BLOCK <= b.length;) {
    const h = b.subarray(o, o + BLOCK); if (h.every((x) => x === 0)) break;
    const name = h.subarray(0, 100).toString().replace(/\0.*$/s, ''), size = parseInt(h.subarray(124, 136).toString().replace(/\0.*$/s, '').trim(), 8);
    let sum = 0; for (let i = 0; i < BLOCK; i++) sum += i >= 148 && i < 156 ? 32 : h[i];
    if (sum !== parseInt(h.subarray(148, 156).toString().replace(/\0.*$/s, '').trim(), 8)) throw new Error(`${file}: bad header checksum at ${o}`);
    out.push({ name, data: b.subarray(o + BLOCK, o + BLOCK + size) }); o += BLOCK + Math.ceil(size / BLOCK) * BLOCK;
  }
  return out;
}
