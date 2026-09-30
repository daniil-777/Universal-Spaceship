"""vlm/train/common.py — shared helpers: JSONL, the seed-group halves for cross-fitting, the LaCie root; the Narrator loss mask
is added in Task 15."""
import json, zlib
LACIE = '/Volumes/LaCie/astro-pilot/vlm'
def read_jsonl(path):
    with open(path) as f: return [json.loads(l) for l in f if l.strip()]
def group_half(group): return zlib.crc32(group.encode()) % 2

# ---- Narrator loss mask (Task 15) ----
# SmolVLM ids: <end_of_utterance>, '\n', <|im_end|> (the pad token), and the image tokens <fake_token_around_image>,
# <global-img>, <image>
EOU, NL, PAD = 49279, 198, 2
IMAGE_IDS = {49189, 49152, 49190}

def build_labels(full_ids, prompt_ids):
    """Labels by prefix length (spec §10.2): -100 everywhere except P .. the last <end_of_utterance>, inclusive."""
    P = len(prompt_ids); assert full_ids[:P] == prompt_ids, 'the prompt ids are not a prefix of the full row'
    last = max((i for i, t in enumerate(full_ids) if t == EOU), default=-1); assert last >= P, 'no closing <end_of_utterance> after the prompt'
    return [-100] * P + full_ids[P:last + 1] + [-100] * (len(full_ids) - last - 1)

def check_row(full_ids, labels, special_ids):
    """The §10.2 row check: exactly two <end_of_utterance>, only the second unmasked and it is the last unmasked label; the
    '\n' after it masked; no other special or image token unmasked."""
    assert len(labels) == len(full_ids), 'labels and ids differ in length'
    eous = [i for i, t in enumerate(full_ids) if t == EOU]; un = [i for i, l in enumerate(labels) if l != -100]
    assert len(eous) == 2, f'{len(eous)} <end_of_utterance> tokens'
    assert labels[eous[0]] == -100 and labels[eous[1]] == EOU and un and un[-1] == eous[1], 'only the closing <end_of_utterance> is unmasked, and it is last'
    if eous[1] + 1 < len(full_ids): assert full_ids[eous[1] + 1] == NL and labels[eous[1] + 1] == -100, 'the newline after the closing <end_of_utterance> is masked'
    assert all(full_ids[i] not in special_ids or i == eous[1] for i in un), 'a special or image token is unmasked'
    assert all(labels[i] == full_ids[i] for i in un), 'an unmasked label differs from its id'
