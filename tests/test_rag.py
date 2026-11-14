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


if __name__ == "__main__":
    unittest.main()
