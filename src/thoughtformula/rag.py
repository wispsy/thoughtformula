"""Pure-Python Okapi BM25 retriever for RAG."""

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

    def retrieve(self, query, top_k=1):
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
