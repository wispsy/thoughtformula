"""Unit tests for ModernMicroGPT."""

import os
import tempfile
import unittest

from thoughtformula.model import ModernMicroGPT
from thoughtformula.tensor import FlatTensor
from thoughtformula.tokenizer import BPETokenizer
from thoughtformula.rag import PurePythonBM25RAG
from thoughtformula.draft import NGramDraftModel


class TestModernMicroGPT(unittest.TestCase):
    def setUp(self):
        self.model = ModernMicroGPT(
            vocab_size=32,
            n_embd=16,
            n_heads=2,
            n_kv_heads=1,
            n_layers=1,
            n_experts=2,
            moe_top_k=1,
            expert_hidden=16,
            lora_rank=2,
            lora_alpha=4.0,
        )

    def test_rope_cache_precomputed(self):
        self.assertIsNotNone(self.model.rope_cache)
        cos_buf, sin_buf = self.model.rope_cache
        half_d = self.model.head_dim // 2
        self.assertEqual(len(cos_buf), self.model.max_seq_len * half_d)
        self.assertEqual(len(sin_buf), self.model.max_seq_len * half_d)
        # Position 0 angle is 0 -> cos is 1, sin is 0
        for m in range(half_d):
            self.assertAlmostEqual(cos_buf[m], 1.0)
            self.assertAlmostEqual(sin_buf[m], 0.0)

    def test_forward_shape(self):
        seq = [1, 2, 3, 4]
        logits = self.model.forward(seq, use_int8=False)
        self.assertEqual(logits.rows, 4)
        self.assertEqual(logits.cols, 32)

    def test_forward_sparse_moe_training_grad(self):
        model = ModernMicroGPT(
            vocab_size=32,
            n_embd=16,
            n_heads=2,
            n_kv_heads=1,
            n_layers=1,
            n_experts=4,
            moe_top_k=2,
            expert_hidden=16,
        )
        seq = [1, 2, 3]
        logits = model.forward(seq, use_int8=False)
        loss = FlatTensor.cross_entropy(logits, [2, 3, 4])
        loss.backward()
        router = model.weights["blocks.0.w_router"]
        self.assertTrue(any(abs(g) > 1e-12 for g in router.grad))
        expert_grad_sum = sum(
            sum(abs(g) for g in model.weights[f"blocks.0.experts.{e}.w_down"].grad)
            for e in range(model.n_experts)
        )
        self.assertTrue(expert_grad_sum > 0.0)

    def test_forward_int8(self):
        seq = [1, 2, 3]
        logits = self.model.forward(seq, use_int8=True)
        self.assertEqual(logits.rows, 3)
        self.assertEqual(logits.cols, 32)

    def test_generate_kv_cache(self):
        prompt = [1, 2]
        out = self.model.generate(prompt, max_new_tokens=3, use_kv_cache=True, use_int8=True)
        self.assertEqual(len(out), 5)
        self.assertEqual(out[:2], prompt)

    def test_speculative_decoding(self):
        drafter = NGramDraftModel(max_n=3)
        drafter.fit([[1, 2, 3, 4, 5]])
        prompt = [1, 2]
        out, acc, drf, passes = self.model.generate_speculative(
            prompt, drafter, max_new_tokens=3, gamma=2, use_int8=True
        )
        self.assertTrue(len(out) >= 3)
        self.assertEqual(out[:2], prompt)

    def test_checkpoint_save_and_load(self):
        tok = BPETokenizer()
        tok.train("test text")
        rag = PurePythonBM25RAG()
        rag.add_document("q", "a")
        corpus = ["<|context|> a <|user|> q <|assistant|> a <|endoftext|> "]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            self.model.save_checkpoint(temp_path, tok, rag, corpus)
            loaded_model, l_tok, l_rag, l_corpus, migrated = ModernMicroGPT.load_checkpoint(temp_path)
            self.assertEqual(loaded_model.vocab_size, self.model.vocab_size)
            self.assertEqual(len(l_rag.documents), 1)
            self.assertEqual(l_corpus, corpus)
            self.assertFalse(migrated)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
