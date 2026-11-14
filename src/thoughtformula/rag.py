"""Pure-Python Okapi BM25 and Hybrid Dense + Lexical RAG retriever."""

import math


class PurePythonBM25RAG:
    STOPWORDS = {
        "what", "is", "the", "a", "an", "how", "do", "does", "are",
        "this", "in", "to", "of", "for", "on", "with", "it", "tell", "me", "about"
    }

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []

    @classmethod
    def tokenize_terms(cls, text):
        clean = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        terms = []
        for word in clean.split():
            if word in cls.STOPWORDS:
                continue
            if len(word) > 5 and word.endswith("ing"):
                word = word[:-3]
            elif len(word) > 4 and word.endswith("ed"):
                word = word[:-2]
            elif len(word) > 4 and word.endswith("es"):
                word = word[:-2]
            elif len(word) > 3 and word.endswith("s"):
                word = word[:-1]
            terms.append(word)
        return terms

    def add_document(self, question, fact):
        q_clean, f_clean = question.strip(), fact.strip()
        for doc in self.documents:
            if doc["question"].lower() == q_clean.lower():
                doc["fact"] = f_clean
                return
            if doc["fact"].lower() == f_clean.lower():
                doc["question"] = q_clean
                return
        self.documents.append({"question": q_clean, "fact": f_clean})

    @staticmethod
    def compute_dense_embedding(text, model, tokenizer):
        """Computes mean-pooled, L2-normalized dense sentence embedding from model tok_emb."""
        raw_ids = tokenizer.encode(text)
        token_ids = [tid for tid in raw_ids if tid not in getattr(tokenizer, "special_ids", ())]
        D = model.n_embd
        if not token_ids:
            return [0.0] * D
        tok_data = model.weights["tok_emb"].data
        vec = [sum(tok_data[tid * D + d] for tid in token_ids) / len(token_ids) for d in range(D)]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-12:
            vec = [v / norm for v in vec]
        return vec

    def retrieve_dense(self, query, model, tokenizer, top_k=None):
        if not self.documents:
            return []
        q_vec = self.compute_dense_embedding(query, model, tokenizer)
        scored = []
        for doc in self.documents:
            d_text = doc["question"] + " " + doc["fact"]
            d_vec = self.compute_dense_embedding(d_text, model, tokenizer)
            cos_sim = math.sumprod(q_vec, d_vec)
            scored.append((doc, cos_sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k] if top_k is not None else scored

    def retrieve_hybrid(self, query, model, tokenizer, top_k=1, rrf_k=60):
        """
        Hybrid Dense + Lexical RAG via Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = 1 / (rrf_k + rank_bm25(d)) + 1 / (rrf_k + rank_dense(d))
        Matches exact terms while retrieving semantic synonyms.
        """
        if not self.documents:
            return []
        bm25_results = self.retrieve(query, top_k=len(self.documents))
        dense_results = self.retrieve_dense(query, model, tokenizer, top_k=len(self.documents))

        bm25_ranks = {id(doc): rank for rank, (doc, _) in enumerate(bm25_results, 1)}
        dense_ranks = {id(doc): rank for rank, (doc, _) in enumerate(dense_results, 1)}

        fused = []
        for doc in self.documents:
            doc_id = id(doc)
            r_bm25 = bm25_ranks.get(doc_id, len(self.documents) + 1)
            r_dense = dense_ranks.get(doc_id, len(self.documents) + 1)
            rrf_score = (1.0 / (rrf_k + r_bm25)) + (1.0 / (rrf_k + r_dense))
            fused.append((doc, rrf_score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    def retrieve(self, query, top_k=1, model=None, tokenizer=None):
        if model is not None and tokenizer is not None:
            return self.retrieve_hybrid(query, model, tokenizer, top_k=top_k)

        if not self.documents:
            return []
        q_terms = self.tokenize_terms(query)
        if not q_terms:
            return [(self.documents[0], 0.0)]

        N = len(self.documents)
        doc_tokens = [
            self.tokenize_terms(d["question"] + " " + d["fact"])
            for d in self.documents
        ]
        avgdl = sum(len(toks) for toks in doc_tokens) / max(1, N)

        scores = []
        for idx, toks in enumerate(doc_tokens):
            dl = len(toks)
            tf_map = {}
            for t in toks:
                tf_map[t] = tf_map.get(t, 0) + 1

            score = 0.0
            for qt in q_terms:
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                df = sum(1 for dt in doc_tokens if qt in dt)
                idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(1e-6, avgdl)))
                score += idf * (num / den)
            scores.append((self.documents[idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def to_list(self):
        return self.documents

    @classmethod
    def from_list(cls, docs):
        rag = cls()
        for d in docs:
            rag.add_document(d["question"], d["fact"])
        return rag


HybridDenseBM25RAG = PurePythonBM25RAG
