"""Pure-Python N-Gram Draft Model for speculative decoding."""

class NGramDraftModel:
    """
    Hierarchical 4-gram -> 3-gram -> 2-gram backoff drafter over BPE token IDs.
    Generates gamma candidate tokens in <0.02 ms for single-pass Transformer verification.
    """
    def __init__(self, max_n=4):
        self.max_n = max_n
        self.tables = {n: {} for n in range(1, max_n)}  # context_tuple -> {next_id: count}

    def fit(self, token_sequences):
        self.tables = {n: {} for n in range(1, self.max_n)}
        for seq in token_sequences:
            L = len(seq)
            for n in range(1, self.max_n):
                tbl = self.tables[n]
                for i in range(L - n):
                    ctx = tuple(seq[i : i + n])
                    nxt = seq[i + n]
                    bucket = tbl.get(ctx)
                    if bucket is None:
                        tbl[ctx] = {nxt: 1}
                    else:
                        bucket[nxt] = bucket.get(nxt, 0) + 1

    def draft(self, prefix_ids, gamma=4, stop_token_id=None):
        """Autoregressively drafts up to `gamma` tokens using longest-suffix N-gram match."""
        drafted = []
        cur = list(prefix_ids)
        for _ in range(gamma):
            next_tok = None
            for n in range(self.max_n - 1, 0, -1):
                if len(cur) >= n:
                    ctx = tuple(cur[-n:])
                    bucket = self.tables[n].get(ctx)
                    if bucket:
                        next_tok = max(bucket.items(), key=lambda kv: kv[1])[0]
                        break
            if next_tok is None:
                break
            drafted.append(next_tok)
            cur.append(next_tok)
            if stop_token_id is not None and next_tok == stop_token_id:
                break
        return drafted
