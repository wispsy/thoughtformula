import json
import math
import os
import random

random.seed(42)

# =====================================================================
# 1. VECTORIZED 2D TENSOR AUTOGRAD ENGINE
# =====================================================================
class Tensor:
    """2D Matrix Autograd Node: data and grad have shape [rows, cols]."""
    __slots__ = ('data', 'grad', 'rows', 'cols', '_backward', '_prev', 'm', 'v')

    def __init__(self, data, _children=()):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
        self.grad = [[0.0] * self.cols for _ in range(self.rows)]
        self._backward = lambda: None
        self._prev = _children
        # Adam optimizer state (allocated lazily on parameters)
        self.m = None
        self.v = None

    @staticmethod
    def randn(rows, cols, std=0.08):
        return Tensor([[random.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)])

    @staticmethod
    def zeros(rows, cols):
        return Tensor([[0.0] * cols for _ in range(rows)])

    def zero_grad(self):
        for r in range(self.rows):
            row_g = self.grad[r]
            for c in range(self.cols):
                row_g[c] = 0.0

    def __add__(self, other):
        """Supports [T, D] + [T, D] or broadcasted bias [T, D] + [1, D]."""
        T, D = self.rows, self.cols
        if other.rows == 1:
            b = other.data[0]
            out_data = [[self.data[t][d] + b[d] for d in range(D)] for t in range(T)]
        else:
            out_data = [[self.data[t][d] + other.data[t][d] for d in range(D)] for t in range(T)]

        out = Tensor(out_data, (self, other))

        def _backward():
            for t in range(T):
                sg, og_out = self.grad[t], out.grad[t]
                for d in range(D):
                    sg[d] += og_out[d]
            if other.rows == 1:
                bg = other.grad[0]
                for t in range(T):
                    og_out = out.grad[t]
                    for d in range(D):
                        bg[d] += og_out[d]
            else:
                for t in range(T):
                    og, og_out = other.grad[t], out.grad[t]
                    for d in range(D):
                        og[d] += og_out[d]

        out._backward = _backward
        return out

    def matmul(self, other):
        """Matrix multiply: [M, K] @ [K, N] -> [M, N]."""
        M, K, N = self.rows, self.cols, other.cols
        A, B = self.data, other.data
        out_data = [[0.0] * N for _ in range(M)]
        for i in range(M):
            Ai, out_i = A[i], out_data[i]
            for k in range(K):
                 a_ik = Ai[k]
                 Bk = B[k]
                 for j in range(N):
                     out_i[j] += a_ik * Bk[j]

        out = Tensor(out_data, (self, other))

        def _backward():
            # dA += dOut @ B^T, dB += A^T @ dOut
            for i in range(M):
                dOut_i = out.grad[i]
                dA_i = self.grad[i]
                Ai = A[i]
                for k in range(K):
                    Bk = B[k]
                    dB_k = other.grad[k]
                    a_ik = Ai[k]
                    s = 0.0
                    for j in range(N):
                        dout_ij = dOut_i[j]
                        s += dout_ij * Bk[j]
                        dB_k[j] += a_ik * dout_ij
                    dA_i[k] += s

        out._backward = _backward
        return out

    def relu(self):
        T, D = self.rows, self.cols
        out_data = [[x if x > 0.0 else 0.0 for x in row] for row in self.data]
        out = Tensor(out_data, (self,))

        def _backward():
            for t in range(T):
                row_d, row_g, out_g = self.data[t], self.grad[t], out.grad[t]
                for d in range(D):
                    if row_d[d] > 0.0:
                        row_g[d] += out_g[d]

        out._backward = _backward
        return out

    def rmsnorm(self, eps=1e-5):
        """Fused Row-wise Root Mean Square Layer Normalization."""
        T, D = self.rows, self.cols
        out_data = []
        inv_rms_list = []
        for t in range(T):
            row = self.data[t]
            ms = sum(x * x for x in row) / D
            inv_rms = 1.0 / math.sqrt(ms + eps)
            inv_rms_list.append(inv_rms)
            out_data.append([x * inv_rms for x in row])

        out = Tensor(out_data, (self,))

        def _backward():
            for t in range(T):
                r = inv_rms_list[t]
                y_row = out.data[t]
                g_row = out.grad[t]
                dx_row = self.grad[t]
                dot_gy = sum(g_row[d] * y_row[d] for d in range(D)) / D
                for d in range(D):
                    dx_row[d] += r * (g_row[d] - y_row[d] * dot_gy)

        out._backward = _backward
        return out

    def slice_rows(self, start, end):
        out = Tensor([row[:] for row in self.data[start:end]], (self,))

        def _backward():
            for i, r in enumerate(range(start, end)):
                sg, og = self.grad[r], out.grad[i]
                for c in range(self.cols):
                    sg[c] += og[c]

        out._backward = _backward
        return out

    @staticmethod
    def embedding(weight, indices):
        """Looks up rows from weight matrix [V, D] -> [T, D]."""
        D = weight.cols
        out_data = [weight.data[idx][:] for idx in indices]
        out = Tensor(out_data, (weight,))

        def _backward():
            for t, idx in enumerate(indices):
                wg, og = weight.grad[idx], out.grad[t]
                for d in range(D):
                    wg[d] += og[d]

        out._backward = _backward
        return out

    @staticmethod
    def causal_attention(Q, K, V):
        """Fused Causal Self-Attention over [T, D] matrices."""
        T, D = Q.rows, Q.cols
        scale = 1.0 / math.sqrt(D)
        probs_matrix = []
        out_data = [[0.0] * D for _ in range(T)]

        for t in range(T):
            Qt = Q.data[t]
            scores = [
                sum(Qt[d] * K.data[k][d] for d in range(D)) * scale
                for k in range(t + 1)
            ]
            max_s = max(scores)
            exps = [math.exp(s - max_s) for s in scores]
            sum_e = sum(exps)
            probs = [e / sum_e for e in exps]
            probs_matrix.append(probs)

            out_t = out_data[t]
            for k, p in enumerate(probs):
                Vk = V.data[k]
                for d in range(D):
                    out_t[d] += p * Vk[d]

        out = Tensor(out_data, (Q, K, V))

        def _backward():
            for t in range(T):
                probs = probs_matrix[t]
                dOut_t = out.grad[t]
                Qt = Q.data[t]
                dQt = Q.grad[t]

                # 1. dV and dP
                dP = [0.0] * (t + 1)
                for k in range(t + 1):
                    pk = probs[k]
                    Vk = V.data[k]
                    dVk = V.grad[k]
                    dp_k = 0.0
                    for d in range(D):
                        do_d = dOut_t[d]
                        dVk[d] += pk * do_d
                        dp_k += do_d * Vk[d]
                    dP[k] = dp_k

                # 2. Softmax backward -> dScores
                dot_p_dp = sum(probs[k] * dP[k] for k in range(t + 1))
                for k in range(t + 1):
                    ds = probs[k] * (dP[k] - dot_p_dp) * scale
                    Kk = K.data[k]
                    dKk = K.grad[k]
                    for d in range(D):
                        dQt[d] += ds * Kk[d]
                        dKk[d] += ds * Qt[d]

        out._backward = _backward
        return out

    @staticmethod
    def cross_entropy(logits, targets):
        """Fused Softmax + Cross-Entropy Loss over [T, V] logits."""
        T, V = logits.rows, logits.cols
        probs_all = []
        total_loss = 0.0

        for t in range(T):
            row = logits.data[t]
            max_l = max(row)
            exps = [math.exp(x - max_l) for x in row]
            sum_e = sum(exps)
            probs = [e / sum_e for e in exps]
            probs_all.append(probs)
            total_loss -= math.log(max(probs[targets[t]], 1e-12))

        out = Tensor([[total_loss / T]], (logits,))

        def _backward():
            grad_scale = out.grad[0][0] / T
            for t in range(T):
                probs = probs_all[t]
                target = targets[t]
                l_grad = logits.grad[t]
                for v in range(V):
                    indicator = 1.0 if v == target else 0.0
                    l_grad[v] += (probs[v] - indicator) * grad_scale

        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if node in visited:
                continue
            if processed:
                visited.add(node)
                topo.append(node)
            else:
                stack.append((node, True))
                for child in node._prev:
                    if child not in visited:
                        stack.append((child, False))
        self.grad[0][0] = 1.0
        for v in reversed(topo):
            v._backward()


# =====================================================================
# 2. PURE-PYTHON BYTE-PAIR ENCODING (BPE) TOKENIZER
# =====================================================================
class BPETokenizer:
    """Subword BPE Tokenizer built from scratch without regex/external libs."""
    def __init__(self):
        self.vocab = {}   # int -> str
        self.stoi = {}    # str -> int
        self.merges = []  # list of [id1, id2, new_id]

    def train(self, text, max_merges=12, min_freq=6):
        chars = sorted(list(set(text)))
        self.vocab = {i: ch for i, ch in enumerate(chars)}
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        ids = [self.stoi[c] for c in text]

        for _ in range(max_merges):
            counts = {}
            for a, b in zip(ids, ids[1:]):
                # Do not merge across whitespace so word prompts align cleanly
                if ' ' in self.vocab[a] or ' ' in self.vocab[b]:
                    continue
                counts[(a, b)] = counts.get((a, b), 0) + 1

            if not counts:
                break
            best_pair, freq = max(counts.items(), key=lambda x: x[1])
            if freq < min_freq:
                break

            new_id = len(self.vocab)
            merged_str = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = merged_str
            self.stoi[merged_str] = new_id
            self.merges.append([best_pair[0], best_pair[1], new_id])
            ids = self._apply_merge(ids, best_pair[0], best_pair[1], new_id)

        return ids

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

    def encode(self, text):
        ids = [self.stoi[c] for c in text if c in self.stoi]
        for a, b, new_id in self.merges:
            ids = self._apply_merge(ids, a, b, new_id)
        return ids

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def to_dict(self):
        return {"vocab": {str(k): v for k, v in self.vocab.items()}, "merges": self.merges}

    @classmethod
    def from_dict(cls, data):
        tok = cls()
        tok.vocab = {int(k): v for k, v in data["vocab"].items()}
        tok.stoi = {v: k for k, v in tok.vocab.items()}
        tok.merges = data["merges"]
        return tok


# =====================================================================
# 3. VECTORIZED MICRO-GPT & JSON CHECKPOINTING
# =====================================================================
class VectorizedGPT:
    def __init__(self, vocab_size, n_embd=24, block_size=16):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        self.weights = {
            "tok_emb": Tensor.randn(vocab_size, n_embd),
            "pos_emb": Tensor.randn(block_size, n_embd),
            "wq": Tensor.randn(n_embd, n_embd),
            "wk": Tensor.randn(n_embd, n_embd),
            "wv": Tensor.randn(n_embd, n_embd),
            "wo": Tensor.randn(n_embd, n_embd),
            "w_fc1": Tensor.randn(n_embd, 2 * n_embd),
            "b_fc1": Tensor.zeros(1, 2 * n_embd),
            "w_fc2": Tensor.randn(2 * n_embd, n_embd),
            "b_fc2": Tensor.zeros(1, n_embd),
            "lm_head": Tensor.randn(n_embd, vocab_size),
            "b_head": Tensor.zeros(1, vocab_size),
        }

    def forward(self, idx_seq):
        T = len(idx_seq)
        w = self.weights

        # 1. Token + Positional Embeddings ([T, D])
        x = Tensor.embedding(w["tok_emb"], idx_seq) + w["pos_emb"].slice_rows(0, T)

        # 2. Causal Self-Attention + Residual
        x_norm = x.rmsnorm()
        Q = x_norm.matmul(w["wq"])
        K = x_norm.matmul(w["wk"])
        V = x_norm.matmul(w["wv"])
        attn_out = Tensor.causal_attention(Q, K, V).matmul(w["wo"])
        x = x + attn_out

        # 3. Feed-Forward MLP + Residual
        hidden = (x.rmsnorm().matmul(w["w_fc1"]) + w["b_fc1"]).relu()
        mlp_out = hidden.matmul(w["w_fc2"]) + w["b_fc2"]
        x = x + mlp_out

        # 4. Final Norm & Output Logits ([T, V])
        logits = x.rmsnorm().matmul(w["lm_head"]) + w["b_head"]
        return logits

    def step_adam(self, step_num, lr=0.015):
        b1_corr = 1.0 - 0.9 ** step_num
        b2_corr = 1.0 - 0.999 ** step_num
        for param in self.weights.values():
            if param.m is None:
                param.m = [[0.0] * param.cols for _ in range(param.rows)]
                param.v = [[0.0] * param.cols for _ in range(param.rows)]
            for r in range(param.rows):
                pd, pg, pm, pv = param.data[r], param.grad[r], param.m[r], param.v[r]
                for c in range(param.cols):
                    g = pg[c]
                    pm[c] = 0.9 * pm[c] + 0.1 * g
                    pv[c] = 0.999 * pv[c] + 0.001 * (g * g)
                    m_hat = pm[c] / b1_corr
                    v_hat = pv[c] / b2_corr
                    pd[c] -= lr * m_hat / (math.sqrt(v_hat) + 1e-8)
                    pg[c] = 0.0

    def generate(self, start_tokens, max_new_tokens, temperature=0.1):
        idx = list(start_tokens)
        for _ in range(max_new_tokens):
            context = idx[-self.block_size:]
            last_logits = self.forward(context).data[-1]
            scaled = [l / max(temperature, 1e-4) for l in last_logits]
            max_l = max(scaled)
            probs = [math.exp(s - max_l) for s in scaled]
            next_id = random.choices(range(self.vocab_size), weights=probs)[0]
            idx.append(next_id)
        return idx

    def save_checkpoint(self, filepath, tokenizer):
        payload = {
            "config": {
                "vocab_size": self.vocab_size,
                "n_embd": self.n_embd,
                "block_size": self.block_size,
            },
            "tokenizer": tokenizer.to_dict(),
            "weights": {name: tensor.data for name, tensor in self.weights.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load_checkpoint(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg = payload["config"]
        model = cls(cfg["vocab_size"], cfg["n_embd"], cfg["block_size"])
        for name, matrix in payload["weights"].items():
            model.weights[name] = Tensor(matrix)
        tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
        return model, tokenizer


# =====================================================================
# 4. TRAINING, SAVING/LOADING, & INFERENCE
# =====================================================================
if __name__ == "__main__":
    base_text = (
        "deep learning in pure python. "
        "small models learn patterns fast. "
        "attention is all you need. "
    )
    corpus = base_text * 4

    # 1. Train BPE Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, max_merges=12, min_freq=6)
    data = tokenizer.encode(corpus)

    learned_subwords = [tokenizer.vocab[new_id] for _, _, new_id in tokenizer.merges]
    print(f"BPE Vocab Size: {len(tokenizer.vocab)} tokens | Learned subwords: {learned_subwords}")
    print(f"Corpus compressed from {len(corpus)} chars -> {len(data)} BPE tokens\n")

    # 2. Initialize Vectorized Model
    block_size = 16
    model = VectorizedGPT(vocab_size=len(tokenizer.vocab), n_embd=24, block_size=block_size)
    total_params = sum(t.rows * t.cols for t in model.weights.values())
    print(f"Total Parameters: {total_params} | Training Vectorized GPT...")

    # Pre-build all unique stride-1 sliding windows over one full cycle of the corpus
    cycle_len = len(tokenizer.encode(base_text))
    windows = [
        (data[i : i + block_size], data[i + 1 : i + block_size + 1])
        for i in range(cycle_len)
    ]

    epochs = 35
    step_num = 0
    for epoch in range(1, epochs + 1):
        random.shuffle(windows)
        epoch_loss = 0.0
        for x_seq, y_seq in windows:
            step_num += 1
            logits = model.forward(x_seq)
            loss = Tensor.cross_entropy(logits, y_seq)
            loss.backward()
            model.step_adam(step_num, lr=0.012)
            epoch_loss += loss.data[0][0]

        if epoch % 7 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Avg Loss: {epoch_loss / len(windows):.4f}")

    # 3. Save weights & tokenizer to JSON, then reload into a fresh instance
    ckpt_path = "micro_gpt_checkpoint.json"
    model.save_checkpoint(ckpt_path, tokenizer)
    print(f"\nSaved checkpoint to '{ckpt_path}' ({os.path.getsize(ckpt_path) // 1024} KB)")

    loaded_model, loaded_tok = VectorizedGPT.load_checkpoint(ckpt_path)
    print("Reloaded model and BPE tokenizer from JSON successfully!\n")

    # 4. Generate from the reloaded model
    print("--- Generated Output (from Reloaded Checkpoint) ---")
    for prompt in ["deep ", "small ", "atten"]:
        prompt_ids = loaded_tok.encode(prompt)
        out_ids = loaded_model.generate(prompt_ids, max_new_tokens=18, temperature=0.05)
        print(f"Prompt '{prompt}' -> {loaded_tok.decode(out_ids)}")
