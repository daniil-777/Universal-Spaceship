"""chat/prompt.py — CAPCOM's prompt contract v1 (spec 2026-09-30-astro-pilot-capcom-design.md §4), the same text as chat/web/prompt.js:
a fixed preamble, up to MAX_NOTES retrieved notes, a state line (scene + <= 3 not-yet-seen highlights in tour order), then the last
HISTORY messages (two exchanges) and the visitor's message. Training rows, the notebook, eval and the page all build prompts here."""

PREAMBLE = ('You are CAPCOM, the guide inside Astro Pilot, a spaceflight demo running in this browser. '
            'Answer briefly from the notes, then offer one next step.')
MAX_NOTES, HISTORY, MAX_UNSEEN = 3, 4, 3
# the demo's highlights in tour order: (key, the name the state line uses)
HIGHLIGHTS = (
    ('belt', 'the asteroid belt'), ('comets', 'comets'), ('orbit', 'Earth orbit'), ('atmo', 'atmospheric flight'),
    ('cities', 'city skylines'), ('alps', 'the Alps'), ('pillars', 'the Zhangjiajie pillars'), ('weather', 'storm weather'),
    ('landing', 'the airliner landing'), ('docking', 'station docking'), ('moon', 'lunar orbit'), ('zoom', 'the Earth-zoom telescope'),
    ('training', 'live training'), ('narrator', 'the Narrator'),
)
NAMES = dict(HIGHLIGHTS)

def unseen(scene, seen=()):
    """The first MAX_UNSEEN highlight names in tour order that are neither the current scene nor already seen (keys)."""
    skip = {scene, *seen}
    return [name for key, name in HIGHLIGHTS if key not in skip][:MAX_UNSEEN]

def state_line(state):
    """'State: <scene name>; not seen yet: a, b, c.' from {'scene': key, 'seen': [keys]} (None: no state line)."""
    if not state: return None
    left = unseen(state['scene'], state.get('seen', ()))
    tail = f"; not seen yet: {', '.join(left)}." if left else '; everything seen.'
    return f"State: {NAMES.get(state['scene'], state['scene'])}{tail}"

def system_text(notes, state=None):
    """notes: fact texts, all rendered (the retriever returns at most MAX_NOTES; training rows may show a 4th oracle fact)."""
    lines = [PREAMBLE]
    lines += ['Notes:', *[f'- {n}' for n in notes]] if notes else ['Notes: none.']
    s = state_line(state)
    if s: lines.append(s)
    return '\n'.join(lines)

def messages(history, user, notes, state=None):
    """[system, <= HISTORY previous messages, the user message]; history: [{'role','content'}] alternating, oldest first."""
    h = [{'role': m['role'], 'content': m['content']} for m in history][-HISTORY:]
    return [{'role': 'system', 'content': system_text(notes, state)}, *h, {'role': 'user', 'content': user}]

def prev_user(history):
    """The previous visitor message (the retriever's follow-up context), or ''."""
    return next((m['content'] for m in reversed(history) if m['role'] == 'user'), '')
