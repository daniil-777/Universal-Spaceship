"""vlm/train/colab/pack_io.py — the Colab package on the local SSD: copied from Drive once (every file checked against
SHA256SUMS), the Pilot Eye cache parts joined back into cache/pilot_eye_<split>.u8, and the WebDataset shards unpacked to
frames/<shard>/<key>.jpg, which is what the rewritten narrator rows name (images[0] = '<shard>/<key>.jpg'). Plus the
stratified Narrator epoch: family x task strata, sampled so that every prefix of the order is balanced."""
import hashlib, json, os, random, shutil, tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

BUF = 8 << 20

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(BUF), b''): h.update(b)
    return h.hexdigest()

def read_sums(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip(): h, name = line.rstrip('\n').split(None, 1); out[name.lstrip('*')] = h
    return out

def _copy(src, dst, want=None):
    os.makedirs(os.path.dirname(dst), exist_ok=True); tmp = dst + '.part'; h = hashlib.sha256()
    with open(src, 'rb') as a, open(tmp, 'wb') as b:
        for buf in iter(lambda: a.read(BUF), b''): h.update(buf); b.write(buf)
    if want and h.hexdigest() != want: os.remove(tmp); raise IOError(f'{src}: sha256 {h.hexdigest()} is not {want} (a partial upload? copy it to Drive again)')
    os.replace(tmp, dst)

def _join(parts, dst, sums, data_dir):
    tmp = dst + '.part'
    with open(tmp, 'wb') as out:
        for p in parts:
            h = hashlib.sha256()
            with open(os.path.join(data_dir, p), 'rb') as f:
                for buf in iter(lambda: f.read(BUF), b''): h.update(buf); out.write(buf)
            if sums.get(p) and h.hexdigest() != sums[p]: os.remove(tmp); raise IOError(f'{p}: sha256 mismatch')
    os.replace(tmp, dst)

def safe_extract(tar_path, dest):
    """Plain files only, names without '/' or '..' (a WebDataset shard is flat: <key>.<ext>)."""
    os.makedirs(dest, exist_ok=True); n = 0
    with tarfile.open(tar_path) as t:
        for m in t:
            if not m.isfile() or '/' in m.name or m.name.startswith('.') or '..' in m.name: continue
            with t.extractfile(m) as f, open(os.path.join(dest, m.name), 'wb') as o: shutil.copyfileobj(f, o, BUF); n += 1
    return n

def _json(p):
    with open(p) as f: return json.load(f)

def sync_package(data_dir, local_dir, threads=4, log=print):
    """Drive -> local SSD, once per package id (the marker local_dir/.synced.json skips a finished copy on a re-run)."""
    man, marker = _json(os.path.join(data_dir, 'manifest.json')), os.path.join(local_dir, '.synced.json')
    if os.path.exists(marker) and _json(marker).get('id') == man['id']: log(f'package {man["id"]} already on {local_dir}'); return man
    sums = read_sums(os.path.join(data_dir, 'SHA256SUMS')); os.makedirs(local_dir, exist_ok=True)
    files = [f for f in man['files'] if not f.startswith(('shards/', 'cache/parts/'))]
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda f: _copy(os.path.join(data_dir, f), os.path.join(local_dir, f), sums.get(f)), files))
    log(f'copied {len(files)} files')
    os.makedirs(os.path.join(local_dir, 'cache'), exist_ok=True)
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda kv: _join(kv[1], os.path.join(local_dir, 'cache', f'pilot_eye_{kv[0]}.u8'), sums, data_dir), man['eye_parts'].items()))
    log(f'joined the Pilot Eye cache: {", ".join(f"{s} {len(p)} part(s)" for s, p in man["eye_parts"].items())}')
    def shard(s):
        tmp = os.path.join(local_dir, 'shards_tmp', os.path.basename(s['path'])); _copy(os.path.join(data_dir, s['path']), tmp, sums.get(s['path']))
        n = safe_extract(tmp, os.path.join(local_dir, 'frames', s['name'])); os.remove(tmp); return n
    with ThreadPoolExecutor(threads) as ex: n = sum(ex.map(shard, man['shards']))
    shutil.rmtree(os.path.join(local_dir, 'shards_tmp'), ignore_errors=True)
    log(f'unpacked {len(man["shards"])} shards ({n} files) to {local_dir}/frames')
    with open(marker, 'w') as f: json.dump({'id': man['id']}, f)
    return man

def allocate(sizes, n, power=0.5, max_repeat=4):
    """Water-filling: shares proportional to size**power, each capped at max_repeat * size; the caps' leftovers go to the
    other strata; integer counts by largest remainders. Returns {stratum: count}, summing to min(n, total cap)."""
    cap = {k: max_repeat * v for k, v in sizes.items() if v}; alloc = {k: 0 for k in cap}; left = min(n, sum(cap.values())); free = set(cap)
    while left > 0 and free:
        w = {k: sizes[k] ** power for k in free}; tot = sum(w.values()); share = {k: left * w[k] / tot for k in free}
        hit = [k for k in sorted(free) if alloc[k] + share[k] >= cap[k]]
        if hit:
            for k in hit: left -= cap[k] - alloc[k]; alloc[k] = cap[k]; free.discard(k)
            continue
        base = {k: int(share[k]) for k in free}; rest = left - sum(base.values())
        for k in sorted(free, key=lambda k: (-(share[k] - base[k]), str(k)))[:rest]: base[k] += 1
        for k in free: alloc[k] += base[k]
        left = 0
    return alloc

def stratified_order(rows, n, seed=0, key=lambda r: (r['family'], r['task']), power=0.5, max_repeat=4):
    """n row indices over family x task strata (allocate): within a stratum full shuffled passes then a partial one; the
    strata are interleaved by relative position, so any prefix of the order is itself close to the target mix."""
    rng, by = random.Random(seed), defaultdict(list)
    for i, r in enumerate(rows): by[key(r)].append(i)
    alloc, picks = allocate({k: len(v) for k, v in by.items()}, n, power, max_repeat), []
    for k in sorted(by, key=str):
        ix, got = by[k], []
        while len(got) < alloc[k]: p = ix[:]; rng.shuffle(p); got.extend(p[:alloc[k] - len(got)])
        picks.extend(((j + rng.random()) / len(got), i) for j, i in enumerate(got))
    picks.sort()
    return [i for _, i in picks]

def narrator_frame_root(local_dir): return os.path.join(local_dir, 'frames')
