import os, unittest
from PIL import Image
from transformers import AutoProcessor
from vlm.train.common import build_labels, check_row, EOU, NL, PAD, IMAGE_IDS
SMOL = os.environ.get('SMOLVLM_DIR', '/Volumes/LaCie/astro-pilot/test/out/vlm_explore/hfcache/HuggingFaceTB/SmolVLM-256M-Instruct')

class TestLossMask(unittest.TestCase):
    def test_mask_by_prefix_length(self):
        # the stock folder keeps its chat template only in tokenizer_config.json, which the 4.57 processor does not read; the call-time
        # do_image_splitting kwarg is ignored in 4.57, images_kwargs is honoured (1 crop, 64 image tokens, like the exported folder)
        p = AutoProcessor.from_pretrained(SMOL); p.chat_template = p.tokenizer.chat_template; img = Image.new('RGB', (896, 504), (40, 80, 120))
        user = {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'Context: telemetry: speed 270 m/s.\nIs it safe?'}]}
        asst = {'role': 'assistant', 'content': [{'type': 'text', 'text': 'The monitor rates this SAFE.'}]}
        full = p(text=p.apply_chat_template([user, asst]), images=[img], return_tensors='pt', images_kwargs={'do_image_splitting': False})['input_ids'][0].tolist()
        prompt = p(text=p.apply_chat_template([user], add_generation_prompt=True), images=[img], return_tensors='pt', images_kwargs={'do_image_splitting': False})['input_ids'][0].tolist()
        labels = build_labels(full, prompt)
        check_row(full, labels, set(p.tokenizer.all_special_ids) | IMAGE_IDS)
        self.assertEqual(sum(t == EOU for t in full), 2); self.assertEqual([l for l in labels if l != -100][-1], EOU)
        with self.assertRaises(AssertionError): check_row(full, [t for t in full], set(p.tokenizer.all_special_ids) | IMAGE_IDS)

class TestLossMaskSynthetic(unittest.TestCase):
    """The mask on hand-made id rows: [prompt with the user EOU] + answer + closing EOU + NL."""
    SPECIAL = {EOU, PAD, 1} | IMAGE_IDS
    def row(self):
        prompt = [1, 49189, 49190, 49190, 49189, 500, 501, EOU, NL, 600, 601]
        return prompt + [700, 701, 702, EOU, NL], prompt

    def test_labels_are_the_answer_and_the_closing_eou(self):
        full, prompt = self.row(); labels = build_labels(full, prompt)
        self.assertEqual(labels, [-100] * len(prompt) + [700, 701, 702, EOU, -100]); check_row(full, labels, self.SPECIAL)

    def test_prompt_must_be_a_prefix(self):
        full, prompt = self.row()
        with self.assertRaises(AssertionError): build_labels(full, prompt[:-1] + [999])

    def test_a_closing_eou_is_required(self):
        full, prompt = self.row()
        with self.assertRaises(AssertionError): build_labels(full[:-2], full[:-2])

    def test_check_row_rejects_bad_masks(self):
        full, prompt = self.row(); labels = build_labels(full, prompt)
        for bad in ([l if i != 2 else full[2] for i, l in enumerate(labels)],   # an image token unmasked
                    labels[:-1] + [NL],                                          # the trailing newline unmasked
                    [l if i != 7 else EOU for i, l in enumerate(labels)],        # the user turn's EOU unmasked
                    labels[:-2] + [-100, -100]):                                 # the closing EOU masked
            with self.assertRaises(AssertionError): check_row(full, bad, self.SPECIAL)
        with self.assertRaises(AssertionError): check_row(full + [EOU], labels + [EOU], self.SPECIAL)   # three EOU tokens

if __name__ == '__main__':
    unittest.main()
