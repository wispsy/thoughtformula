"""Dynamic Byte Pair Encoding (BPE) Tokenizer with conversation special tokens."""


class BPETokenizer:
    EOS_TOKEN = "<|endoftext|>"
    CONTEXT_TOKEN = "<|context|>"
    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    SPECIAL_TOKENS = [EOS_TOKEN, CONTEXT_TOKEN, USER_TOKEN, ASSISTANT_TOKEN]

    BASE_ALPHABET = (
        " \n.,!?'\"-:;()0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    def __init__(self):
        self.vocab = {}
        self.stoi = {}
        self.merges = []
        self.special_ids = set()

    def train(self, text, max_merges=18, min_freq=4):
        self.vocab = {}
        self.stoi = {}
        self.merges = []
        self.special_ids = set()

        for sp in self.SPECIAL_TOKENS:
            idx = len(self.vocab)
            self.vocab[idx] = sp
            self.stoi[sp] = idx
            self.special_ids.add(idx)

        cleaned = text
        for sp in self.SPECIAL_TOKENS:
            cleaned = cleaned.replace(sp, "")
        chars = sorted(list(set(self.BASE_ALPHABET + cleaned)))
        for ch in chars:
            if ch not in self.stoi:
                idx = len(self.vocab)
                self.vocab[idx] = ch
                self.stoi[ch] = idx

        self.expand_vocab(text, max_new_merges=max_merges, min_freq=min_freq)
        return self.encode(text)

    def ensure_special_tokens(self):
        new_tokens_info = []
        for sp in self.SPECIAL_TOKENS:
            if sp not in self.stoi:
                new_id = len(self.vocab)
                self.vocab[new_id] = sp
                self.stoi[sp] = new_id
                new_tokens_info.append((new_id, None, None))
            self.special_ids.add(self.stoi[sp])
        return new_tokens_info

    def expand_vocab(self, text, max_new_merges=3, min_freq=4):
        new_tokens_info = self.ensure_special_tokens()

        stripped = text
        for sp in self.SPECIAL_TOKENS:
            stripped = stripped.replace(sp, "")
        for ch in stripped:
            if ch not in self.stoi:
                new_id = len(self.vocab)
                self.vocab[new_id] = ch
                self.stoi[ch] = new_id
                new_tokens_info.append((new_id, None, None))

        ids = self.encode(text)
        for _ in range(max_new_merges):
            counts = {}
            for a, b in zip(ids, ids[1:]):
                if a in self.special_ids or b in self.special_ids:
                    continue
                va, vb = self.vocab[a], self.vocab[b]
                if ' ' in va or ' ' in vb or '\n' in va or '\n' in vb:
                    continue
                counts[(a, b)] = counts.get((a, b), 0) + 1

            if not counts:
                break
            best_pair, freq = max(counts.items(), key=lambda x: x[1])
            if freq < min_freq:
                break

            merged_str = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            if merged_str in self.stoi:
                break

            new_id = len(self.vocab)
            self.vocab[new_id] = merged_str
            self.stoi[merged_str] = new_id
            self.merges.append([best_pair[0], best_pair[1], new_id])
            new_tokens_info.append((new_id, best_pair[0], best_pair[1]))
            ids = self._apply_merge(ids, best_pair[0], best_pair[1], new_id)

        return new_tokens_info

    @staticmethod
    def _apply_merge(ids, a, b, new_id):
        out, i, n = [], 0, len(ids)
        while i < n:
            if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def _encode_plain_chunk(self, chunk):
        ids = [self.stoi[c] for c in chunk if c in self.stoi]
        for a, b, new_id in self.merges:
            ids = self._apply_merge(ids, a, b, new_id)
        return ids

    def encode(self, text):
        segments = [text]
        for sp in self.SPECIAL_TOKENS:
            next_segments = []
            for seg in segments:
                if seg in self.SPECIAL_TOKENS:
                    next_segments.append(seg)
                else:
                    parts = seg.split(sp)
                    for idx, part in enumerate(parts):
                        if part:
                            next_segments.append(part)
                        if idx < len(parts) - 1:
                            next_segments.append(sp)
            segments = next_segments

        out_ids = []
        for seg in segments:
            if seg in self.SPECIAL_TOKENS and seg in self.stoi:
                out_ids.append(self.stoi[seg])
            else:
                out_ids.extend(self._encode_plain_chunk(seg))
        return out_ids

    def decode(self, ids, skip_special=False):
        if skip_special:
            return "".join(self.vocab.get(i, "") for i in ids if i not in self.special_ids)
        return "".join(self.vocab.get(i, "") for i in ids)

    @property
    def eos_id(self):
        return self.stoi[self.EOS_TOKEN]

    def format_rag_qa(self, context, question, answer):
        c, q, a = context.strip(), question.strip(), answer.strip()
        return f"{self.CONTEXT_TOKEN} {c} {self.USER_TOKEN} {q} {self.ASSISTANT_TOKEN} {a} {self.EOS_TOKEN} "

    def format_rag_prompt(self, context, question):
        c, q = context.strip(), question.strip()
        return f"{self.CONTEXT_TOKEN} {c} {self.USER_TOKEN} {q} {self.ASSISTANT_TOKEN} "

    def to_dict(self):
        return {"vocab": {str(k): v for k, v in self.vocab.items()}, "merges": self.merges}

    @classmethod
    def from_dict(cls, data):
        tok = cls()
        tok.vocab = {int(k): v for k, v in data["vocab"].items()}
        tok.stoi = {v: k for k, v in tok.vocab.items()}
        tok.merges = data["merges"]
        tok.special_ids = {tok.stoi[sp] for sp in cls.SPECIAL_TOKENS if sp in tok.stoi}
        return tok
