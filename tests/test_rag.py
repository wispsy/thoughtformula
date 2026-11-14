"""Unit tests for PurePythonBM25RAG."""

import unittest
from thoughtformula.rag import PurePythonBM25RAG


class TestPurePythonBM25RAG(unittest.TestCase):
    def test_term_tokenization(self):
        terms = PurePythonBM25RAG.tokenize_terms("What is this deep learning model?")
        self.assertIn("deep", terms)
        self.assertIn("learn", terms)  # stemmed from learning
        self.assertIn("model", terms)
        self.assertNotIn("what", terms)  # stopword
        self.assertNotIn("is", terms)    # stopword

    def test_retrieval(self):
        rag = PurePythonBM25RAG()
        rag.add_document("what is this model?", "deep learning in pure python.")
        rag.add_document("how does gqa work?", "grouped query attention shares key value heads.")

        results = rag.retrieve("tell me about gqa", top_k=1)
        self.assertEqual(len(results), 1)
        top_doc, score = results[0]
        self.assertEqual(top_doc["question"], "how does gqa work?")
        self.assertTrue(score > 0.0)

    def test_hybrid_dense_retrieval(self):
        from thoughtformula.tokenizer import BPETokenizer
        from thoughtformula.model import ModernMicroGPT

        tok = BPETokenizer()
        tok.train("what is this model? deep learning in pure python. grouped query attention shares key value heads.")
        model = ModernMicroGPT(vocab_size=len(tok.vocab), n_embd=16, n_heads=2, n_kv_heads=1, n_layers=1, n_experts=2, moe_top_k=1)

        rag = PurePythonBM25RAG()
        rag.add_document("what is this model?", "deep learning in pure python.")
        rag.add_document("how does gqa work?", "grouped query attention shares key value heads.")

        dense_vec = rag.compute_dense_embedding("deep learning", model, tok)
        self.assertEqual(len(dense_vec), 16)
        # Verify L2 norm is ~1.0
        norm = sum(v * v for v in dense_vec)
        self.assertAlmostEqual(norm, 1.0, places=4)

        # Test hybrid retrieval with RRF
        hybrid_res = rag.retrieve_hybrid("deep learning without numpy", model, tok, top_k=1)
        self.assertEqual(len(hybrid_res), 1)
        top_doc, rrf_score = hybrid_res[0]
        self.assertTrue(rrf_score > 0.0)


if __name__ == "__main__":
    unittest.main()
