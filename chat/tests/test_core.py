"""chat/tests/test_core.py — the shared contracts: prompt (spec §4), scorer (R6), dialog validator and row building (§5).
  python3 -m unittest chat.tests.test_core   (from the repo root; stdlib only)"""
import unittest
from chat import prompt, rewards
from chat.data.build import must_terms, turn_rows
from chat.data.validate import check
from chat.retrieval import BM25

FACTS = {f['id']: f for f in [
    {'id': 'LANDING-001', 'title': 'Landing flag', 'text': 'Open ?scenario=landing to watch the airliner autoland; its base approach speed is 157 kt.', 'keywords': ['landing', 'autoland']},
    {'id': 'HOWTO-001', 'title': 'Zoom key', 'text': "Press Z or the Playbox 'Zoom in' button to open the Earth-zoom telescope.", 'keywords': ['zoom', 'telescope']},
    {'id': 'RL-001', 'title': 'PPO pilot', 'text': 'The Astro Pilot pilot is a PPO policy that commands body rates at 15 Hz.', 'keywords': ['ppo', 'policy', 'rates']},
]}

def dialog(**kw):
    d = {'id': 'D00001', 'brief': 'B00001', 'persona': 'kid', 'archetype': 'qa_chain', 'state': {'scene': 'belt', 'seen': []}, 'goal': 'g',
         'turns': [{'role': 'user', 'content': 'how fast does the plane land'},
                   {'role': 'assistant', 'content': 'It approaches at 157 kt. Want to watch it with ?scenario=landing?', 'facts': ['LANDING-001'], 'lead': 'question',
                    'abstain': "That's not in my flight notes, sorry. Want to hear how the pilot learned instead?"},
                   {'role': 'user', 'content': 'cool and the telescope'},
                   {'role': 'assistant', 'content': "Press Z to open the Earth-zoom telescope.", 'facts': ['HOWTO-001'], 'lead': 'suggestion'}]}
    d.update(kw); return d

class TestPrompt(unittest.TestCase):
    def test_system_text_and_state(self):
        s = prompt.system_text(['a', 'b'], {'scene': 'belt', 'seen': ['comets']})
        self.assertEqual(s.split('\n'), [prompt.PREAMBLE, 'Notes:', '- a', '- b', 'State: the asteroid belt; not seen yet: Earth orbit, atmospheric flight, city skylines.'])
        self.assertIn('Notes: none.', prompt.system_text([], None)); self.assertNotIn('State:', prompt.system_text([], None))
    def test_history_window(self):
        h = [{'role': r, 'content': str(i)} for i, r in enumerate(['user', 'assistant'] * 4)]
        m = prompt.messages(h, 'now', ['n'])
        self.assertEqual([x['content'] for x in m[1:]], ['4', '5', '6', '7', 'now']); self.assertEqual(m[0]['role'], 'system')
        self.assertEqual(prompt.prev_user(h), '6')
    def test_everything_seen(self):
        self.assertTrue(prompt.state_line({'scene': 'belt', 'seen': [k for k, _ in prompt.HIGHLIGHTS]}).endswith('everything seen.'))

class TestRewards(unittest.TestCase):
    ROW = {'notes': [FACTS['LANDING-001']['text']], 'user': 'how fast does it land', 'must': ['157', 'approach'], 'abstain': False, 'lead': True}
    def test_good_answer_scores_high(self):
        c = rewards.score('It approaches at 157 kt. Want to watch it with ?scenario=landing?', self.ROW)
        self.assertEqual(c['grounded'], 1.0); self.assertEqual(c['lead'], 1.0); self.assertEqual(c['recall'], 1.0); self.assertGreater(c['total'], 0.8)
    def test_invented_number_and_flag(self):
        self.assertLess(rewards.score('It lands at 999 kt. Try ?scenario=moonbase next.', self.ROW)['grounded'], 0.5)
    def test_writer_gotchas(self):
        notes = ["Open ?atmo=1 for atmospheric flight. The Narrator's buttons are 'Is it safe' and 'What now'."]
        self.assertEqual(rewards.comp_grounded('Try ?atmo=1, then fly low.', notes, ''), 1.0)      # trailing comma is punctuation
        self.assertEqual(rewards.comp_grounded('Open ?atmo=1.', notes, ''), 1.0)
        self.assertEqual(rewards.questions('Tap "Is it safe?" on the card. Want to try?'), 1)     # a quoted label is not a question
        self.assertEqual(rewards.comp_grounded('Press the key - then wait.', notes, ''), 1.0)     # an unquoted dash is not a key
        self.assertTrue(rewards.ABSTAIN.search("Sorry, that isn't in my flight notes."))
    def test_keys(self):
        notes = ['The Z key or the Playbox button opens the telescope.']
        self.assertEqual(rewards.comp_grounded('Press Z to zoom.', notes, ''), 1.0)
        self.assertEqual(rewards.comp_grounded('Press Q to zoom.', notes, ''), 0.5)
    def test_nagging_and_no_lead(self):
        self.assertEqual(rewards.comp_lead('It is fast. Want more? Or the moon?'), 0.0)
        self.assertEqual(rewards.comp_lead('It is fast at 157 kt.'), 0.0)
        self.assertEqual(rewards.comp_lead('Bye, safe flight!', expected=False), 1.0)
    def test_abstain(self):
        row = {**self.ROW, 'abstain': True, 'must': []}
        self.assertEqual(rewards.score("That's not in my flight notes. Want to hear about the pilot?", row)['abstain'], 1.0)
        self.assertEqual(rewards.score('It lands at 157 kt.', row)['abstain'], 0.0)
    def test_leak_and_empty(self):
        self.assertEqual(rewards.comp_leak('Notes: - secret'), 0.0); self.assertEqual(rewards.score('', self.ROW)['total'], 0.0)
    def test_answer_first(self):
        self.assertEqual(rewards.comp_first('Why? Because it is trained.'), 0.0); self.assertEqual(rewards.comp_first('Want the landing?'), 1.0)

class TestValidate(unittest.TestCase):
    def test_valid(self): self.assertEqual(check(dialog(), FACTS), [])
    def test_ungrounded_number(self):
        d = dialog(); d['turns'][1]['content'] = 'It approaches at 170 kt. Want to see it?'
        self.assertTrue(any('unsupported' in e for e in check(d, FACTS)))
    def test_lead_mismatch_and_shape(self):
        d = dialog(); d['turns'][3]['lead'] = 'question'
        self.assertTrue(any('question marks' in e for e in check(d, FACTS)))
        d = dialog(); d['turns'] = d['turns'][:3]
        self.assertTrue(any('last turn' in e for e in check(d, FACTS)))
    def test_abstain_rules(self):
        d = dialog(); d['turns'][1]['abstain'] = 'Sure, it is 157 kt!'
        e = check(d, FACTS); self.assertTrue(any('not in the notes' in x for x in e)); self.assertTrue(any('abstain cites' in x for x in e))

class TestRows(unittest.TestCase):
    def test_rows_completion_only_and_notes(self):
        bm = BM25(list(FACTS.values())); rows = turn_rows(dialog(), FACTS, bm, 0.0)
        self.assertEqual(len(rows), 2)
        (sft, pr), (sft2, pr2) = rows
        self.assertEqual(sft['prompt'][-1], {'role': 'user', 'content': 'how fast does the plane land'})
        self.assertEqual(sft['completion'][0]['role'], 'assistant'); self.assertEqual(len(sft2['prompt']), 4)
        if pr['meta']['kind'] == 'answer':
            self.assertIn(FACTS['LANDING-001']['text'], pr['notes']); self.assertFalse(pr['abstain']); self.assertIn('157', pr['must'])
        else:
            self.assertNotIn(FACTS['LANDING-001']['text'], pr['notes']); self.assertTrue(pr['abstain']); self.assertEqual(pr['must'], [])
    def test_must_terms_rank_rare_first(self):
        idf = {'157': 3.0, 'approach': 1.0, 'speed': 2.0}
        self.assertEqual(must_terms('approach speed 157', ['base approach speed is 157 kt'], idf)[:2], ['157', 'speed'])

if __name__ == '__main__':
    unittest.main()
