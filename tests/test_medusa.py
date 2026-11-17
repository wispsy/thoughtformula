"""Unit tests for Medusa multi-head speculative drafter."""

from array import array
import unittest

from thoughtformula.tensor import FlatTensor
from thoughtformula.medusa import MedusaHead, MedusaDrafter
from thoughtformula.model import ModernMicroGPT


class TestMedusa(unittest.TestCase):
    def test_medusa_head_forward(self):
        head = MedusaHead(head_idx=0, n_embd=8, vocab_size=12)
        h = FlatTensor(2, 8, array('d', [0.5] * 16))
        logits = head.forward(h)
        self.assertEqual(logits.rows, 2)
        self.assertEqual(logits.cols, 12)

    def test_medusa_drafter_draft_candidates(self):
        drafter = MedusaDrafter(n_embd=8, vocab_size=16, num_heads=3)
        h_last = FlatTensor(1, 8, array('d', [0.2] * 8))

        candidates = drafter.draft(h_last, base_token=5)
        # Should return base token + 3 medusa head predictions
        self.assertEqual(len(candidates), 4)
        self.assertEqual(candidates[0], 5)
        for tok in candidates:
            self.assertTrue(0 <= tok < 16)

    def test_medusa_train_step(self):
        drafter = MedusaDrafter(n_embd=8, vocab_size=16, num_heads=2)
        T = 8
        h_seq = FlatTensor(T, 8, array('d', [0.1 * i for i in range(T * 8)]))
        targets = [1, 2, 3, 4, 5, 6, 7, 8]

        loss1 = drafter.train_step(h_seq, targets, lr=0.05)
        self.assertTrue(loss1 > 0.0)

        for _ in range(5):
            loss_final = drafter.train_step(h_seq, targets, lr=0.05)

        self.assertLess(loss_final, loss1)

    def test_medusa_serialization_roundtrip(self):
        drafter = MedusaDrafter(n_embd=8, vocab_size=16, num_heads=2)
        d = drafter.to_dict()
        self.assertEqual(d["n_embd"], 8)
        self.assertEqual(d["vocab_size"], 16)
        self.assertEqual(d["num_heads"], 2)

        restored = MedusaDrafter.from_dict(d)
        self.assertEqual(restored.n_embd, 8)
        self.assertEqual(restored.num_heads, 2)
        self.assertEqual(len(restored.heads), 2)

    def test_model_generate_medusa_and_stream(self):
        model = ModernMicroGPT(vocab_size=16, n_embd=8, n_heads=2, n_kv_heads=1, n_layers=1, n_experts=2, moe_top_k=1, expert_hidden=8)
        model.init_medusa(num_heads=2)

        prompt = [1, 2, 3]
        out_ids, acc, drf, passes = model.generate_medusa(prompt, max_new_tokens=4, temperature=0.0)
        self.assertEqual(len(out_ids), len(prompt) + 4)
        self.assertGreaterEqual(passes, 1)

        streamed = list(model.generate_medusa_stream(prompt, max_new_tokens=4, temperature=0.0))
        self.assertEqual(len(streamed), 4)


if __name__ == "__main__":
    unittest.main()
