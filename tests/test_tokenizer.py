"""Unit tests for BPETokenizer."""

import unittest
from thoughtformula.tokenizer import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    def test_train_and_encode_decode(self):
        tok = BPETokenizer()
        text = "deep learning in pure python. deep learning learns fast."
        tok.train(text, max_merges=10, min_freq=2)

        encoded = tok.encode("deep learning")
        decoded = tok.decode(encoded)
        self.assertEqual(decoded, "deep learning")

    def test_special_tokens(self):
        tok = BPETokenizer()
        tok.train("hello world")
        prompt = tok.format_rag_prompt("context fact", "user question")
        self.assertIn(BPETokenizer.CONTEXT_TOKEN, prompt)
        self.assertIn(BPETokenizer.USER_TOKEN, prompt)
        self.assertIn(BPETokenizer.ASSISTANT_TOKEN, prompt)

        encoded = tok.encode(prompt)
        decoded = tok.decode(encoded, skip_special=True)
        self.assertNotIn("<|", decoded)

    def test_serialization(self):
        tok = BPETokenizer()
        tok.train("sample text for merges", max_merges=4)
        d = tok.to_dict()
        tok2 = BPETokenizer.from_dict(d)
        self.assertEqual(tok.vocab, tok2.vocab)
        self.assertEqual(tok.merges, tok2.merges)


if __name__ == "__main__":
    unittest.main()
