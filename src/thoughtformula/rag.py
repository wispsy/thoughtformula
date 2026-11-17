"""Pure-Python Okapi BM25 and Hybrid Dense + Lexical RAG retriever with persistent binary index (.tfidx)."""

from array import array
import json
import math
import struct
import zlib


class PurePythonBM25RAG:
    MAGIC_TFIDX = b"TFIDX\x01\x00"

    STOPWORDS = {
        "what", "is", "the", "a", "an", "how", "do", "does", "are",
        "this", "in", "to", "of", "for", "on", "with", "it", "tell", "me", "about"
    }

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.is_indexed = False
        self.doc_lens = []
        self.avgdl = 0.0
        self.inverted_index = {}
        self.dense_embeddings = None
        self.embedding_dim = 0

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

        if self.is_indexed and self.dense_embeddings is not None and self.embedding_dim > 0:
            D = self.embedding_dim
            for doc_id, doc in enumerate(self.documents):
                base = doc_id * D
                d_vec = memoryview(self.dense_embeddings)[base : base + D]
                cos_sim = math.sumprod(q_vec, d_vec)
                scored.append((doc, cos_sim))
        else:
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

        # Fast path: use precomputed inverted postings index
        if self.is_indexed and self.inverted_index:
            scores_by_doc = [0.0] * N
            avgdl = max(1e-6, self.avgdl)
            matched_any = False

            for qt in q_terms:
                postings = self.inverted_index.get(qt)
                if not postings:
                    continue
                matched_any = True
                df = len(postings)
                idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
                for doc_id, tf in postings:
                    dl = self.doc_lens[doc_id]
                    num = tf * (self.k1 + 1.0)
                    den = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                    scores_by_doc[doc_id] += idf * (num / den)

            if not matched_any:
                return [(self.documents[0], 0.0)]

            scored = [(self.documents[i], scores_by_doc[i]) for i in range(N)]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

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

    def build_index(self, model=None, tokenizer=None):
        """Builds inverted term index and precomputes dense embeddings for fast retrieval."""
        N = len(self.documents)
        if N == 0:
            return

        doc_tokens = [
            self.tokenize_terms(d["question"] + " " + d["fact"])
            for d in self.documents
        ]
        self.doc_lens = [len(toks) for toks in doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, N)

        # Inverted index: term -> [(doc_id, tf), ...]
        self.inverted_index = {}
        for doc_id, toks in enumerate(doc_tokens):
            tf_map = {}
            for t in toks:
                tf_map[t] = tf_map.get(t, 0) + 1
            for t, tf in tf_map.items():
                if t not in self.inverted_index:
                    self.inverted_index[t] = []
                self.inverted_index[t].append((doc_id, tf))

        if model is not None and tokenizer is not None:
            D = model.n_embd
            self.embedding_dim = D
            emb_flat = array('d', [0.0] * (N * D))
            for doc_id, doc in enumerate(self.documents):
                d_text = doc["question"] + " " + doc["fact"]
                vec = self.compute_dense_embedding(d_text, model, tokenizer)
                base = doc_id * D
                for d in range(D):
                    emb_flat[base + d] = vec[d]
            self.dense_embeddings = emb_flat
        else:
            self.dense_embeddings = None
            self.embedding_dim = 0

        self.is_indexed = True

    def save_index(self, filepath: str, model=None, tokenizer=None):
        """Serializes precomputed document store and inverted index into binary .tfidx container."""
        if not self.is_indexed or (model is not None and self.dense_embeddings is None):
            self.build_index(model=model, tokenizer=tokenizer)

        N = len(self.documents)
        meta = {
            "num_docs": N,
            "embedding_dim": self.embedding_dim,
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "doc_lens": self.doc_lens,
        }
        meta_bytes = json.dumps(meta).encode("utf-8")
        docs_bytes = zlib.compress(json.dumps(self.documents).encode("utf-8"), level=9)

        if self.dense_embeddings is not None and len(self.dense_embeddings) > 0:
            raw_emb = struct.pack(f"<{len(self.dense_embeddings)}d", *self.dense_embeddings)
            emb_bytes = zlib.compress(raw_emb, level=9)
        else:
            emb_bytes = b""

        inv_bytes = zlib.compress(json.dumps(self.inverted_index).encode("utf-8"), level=9)

        header = struct.pack(
            "<7sIIII",
            self.MAGIC_TFIDX,
            len(meta_bytes),
            len(docs_bytes),
            len(emb_bytes),
            len(inv_bytes),
        )

        with open(filepath, "wb") as f:
            f.write(header)
            f.write(meta_bytes)
            f.write(docs_bytes)
            f.write(emb_bytes)
            f.write(inv_bytes)

    @classmethod
    def load_index(cls, filepath: str) -> "PurePythonBM25RAG":
        """Loads precomputed binary index (.tfidx) with zero recomputation."""
        with open(filepath, "rb") as f:
            header_bytes = f.read(23)
            if len(header_bytes) < 23:
                raise ValueError("Corrupt .tfidx header")
            magic, m_len, d_len, e_len, i_len = struct.unpack("<7sIIII", header_bytes)
            if magic != cls.MAGIC_TFIDX:
                raise ValueError(f"Invalid .tfidx magic: {magic}")

            meta_bytes = f.read(m_len)
            docs_bytes = f.read(d_len)
            emb_bytes = f.read(e_len)
            inv_bytes = f.read(i_len)

        meta = json.loads(meta_bytes.decode("utf-8"))
        documents = json.loads(zlib.decompress(docs_bytes).decode("utf-8"))
        inverted_index = json.loads(zlib.decompress(inv_bytes).decode("utf-8"))

        rag = cls(k1=meta.get("k1", 1.5), b=meta.get("b", 0.75))
        rag.documents = documents
        rag.doc_lens = meta.get("doc_lens", [])
        rag.avgdl = meta.get("avgdl", 0.0)
        rag.inverted_index = {
            term: [tuple(p) for p in postings]
            for term, postings in inverted_index.items()
        }
        rag.embedding_dim = meta.get("embedding_dim", 0)

        if e_len > 0 and rag.embedding_dim > 0:
            raw_emb = zlib.decompress(emb_bytes)
            n_floats = meta["num_docs"] * rag.embedding_dim
            unpacked = struct.unpack(f"<{n_floats}d", raw_emb)
            rag.dense_embeddings = array('d', unpacked)
        else:
            rag.dense_embeddings = None

        rag.is_indexed = True
        return rag

    def to_list(self):
        return self.documents

    @classmethod
    def from_list(cls, docs):
        rag = cls()
        for d in docs:
            rag.add_document(d["question"], d["fact"])
        return rag


HybridDenseBM25RAG = PurePythonBM25RAG
