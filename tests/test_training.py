"""Unit tests for training utilities and continual learning."""

import unittest
from thoughtformula.training import (
    get_cosine_lr,
    build_training_sequences,
    continual_learn_lora_turn,
    train_dpo_turn,
    train_kto_turn,
)
from thoughtformula.tokenizer import BPETokenizer
from thoughtformula.model import ModernMicroGPT
from thoughtformula.rag import PurePythonBM25RAG
from thoughtformula.draft import NGramDraftModel


class TestTraining(unittest.TestCase):
    def test_get_cosine_lr(self):
        lr_0 = get_cosine_lr(0, 100, max_lr=0.01, min_lr=0.001, warmup_steps=10)
        self.assertAlmostEqual(lr_0, 0.0)

        lr_warmup = get_cosine_lr(10, 100, max_lr=0.01, min_lr=0.001, warmup_steps=10)
        self.assertAlmostEqual(lr_warmup, 0.01)

        lr_end = get_cosine_lr(100, 100, max_lr=0.01, min_lr=0.001, warmup_steps=10)
        self.assertAlmostEqual(lr_end, 0.001)

    def test_build_training_sequences(self):
        tok = BPETokenizer()
        tok.train("deep learning fast patterns", max_merges=4)
        corpus = ["deep learning fast ", "small models learn fast "]
        seqs = build_training_sequences(tok, corpus, block_size=10)
        self.assertTrue(len(seqs) > 0)
        for x, y in seqs:
            self.assertEqual(len(x), len(y))

    def test_train_dpo_turn(self):
        tok = BPETokenizer()
        tok.train("what is this model? deep learning in pure python. it is a bad model.", max_merges=6)
        model = ModernMicroGPT(
            vocab_size=len(tok.vocab),
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
        dpo_loss, margin = train_dpo_turn(
            model,
            tok,
            question="what is this model?",
            chosen_answer="deep learning in pure python.",
            rejected_answer="it is a bad model.",
            beta=0.1,
            lr=0.01,
            epochs=3,
        )
        self.assertTrue(dpo_loss > 0.0)

    def test_continual_learn_lora_turn(self):
        tok = BPETokenizer()
        tok.train("what is this model? deep learning in pure python.", max_merges=6)
        model = ModernMicroGPT(
            vocab_size=len(tok.vocab),
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
        rag = PurePythonBM25RAG()
        drafter = NGramDraftModel()
        corpus = [tok.format_rag_qa("deep learning in pure python.", "what is this model?", "deep learning in pure python.")]

        loss, minted = continual_learn_lora_turn(
            model,
            tok,
            rag,
            drafter,
            corpus,
            question="what is this model?",
            answer="deep learning in pure python.",
            ckpt_path=None,
            block_size=12,
        )
        self.assertTrue(loss >= 0.0)

    def test_train_kto_turn_desirable(self):
        tok = BPETokenizer()
        tok.train("what is this model? deep learning in pure python.", max_merges=6)
        model = ModernMicroGPT(
            vocab_size=len(tok.vocab),
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
        kto_loss, reward = train_kto_turn(
            model,
            tok,
            question="what is this model?",
            answer="deep learning in pure python.",
            is_desirable=True,
            beta=0.1,
            lr=0.01,
            epochs=3,
        )
        self.assertTrue(kto_loss > 0.0)

    def test_train_kto_turn_undesirable(self):
        tok = BPETokenizer()
        tok.train("what is this model? bad nonsense.", max_merges=6)
        model = ModernMicroGPT(
            vocab_size=len(tok.vocab),
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
        kto_loss, reward = train_kto_turn(
            model,
            tok,
            question="what is this model?",
            answer="bad nonsense.",
            is_desirable=False,
            beta=0.1,
            lr=0.01,
            epochs=3,
        )
        self.assertTrue(kto_loss > 0.0)


if __name__ == "__main__":
    unittest.main()
