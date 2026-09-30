"""vlm/train/common.py — shared helpers: JSONL, the seed-group halves for cross-fitting, the LaCie root; the Narrator loss mask
is added in Task 15."""
import json, zlib
LACIE = '/Volumes/LaCie/astro-pilot/vlm'
def read_jsonl(path):
    with open(path) as f: return [json.loads(l) for l in f if l.strip()]
def group_half(group): return zlib.crc32(group.encode()) % 2
