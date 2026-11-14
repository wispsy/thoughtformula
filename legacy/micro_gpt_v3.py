from array import array
import json
import math
import os
import random
import time

random.seed(42)

# =====================================================================
# 1. FLAT 1D array('d') VECTORIZED AUTOGRAD ENGINE
# =====================================================================
class FlatTensor:
    """2D Matrix stored in a contiguous 1D C-double array('d') of size rows * cols."""
    __slots__ = ('rows', 'cols', 'data', 'grad', '_backward', '_prev', 'm', 'v')

    def __init__(self, rows, cols, data=None, _children=()):
        self.rows = rows
        self.cols = cols
        size = rows * cols
        self.data = data if data is not None else array('d', [0.0] * size)
        self.grad = array('d', [0.0] * size)
        self._backward = lambda: None
        self._prev = _children
        self.m = None
        self.v = None

    @staticmethod
    def randn(rows, cols, std=0.08):
        size = rows * cols
        return FlatTensor(rows, cols, array('d', (random.gauss(0.0, std) for _ in range(size))))

    @staticmethod
    def zeros(rows, cols):
        return FlatTensor(rows, cols)

    def to_list_2d(self):
        C = self.cols
        return [list(self.data[r * C : (r + 1) * C]) for r in range(self.rows)]

    @classmethod
    def from_list_2d(cls, mat):
        rows = len(mat)
        cols = len(mat[0]) if rows > 0 else 0
        flat = array('d', (x for row in mat for x in row))
        return cls(rows, cols, flat)

    def __add__(self, other):
        """Elementwise [T, D] + [T, D] or broadcasted row bias [T, D] + [1, D]."""
        T, D = self.rows, self.cols
        N_el = T * D
        A, B = self.data, other.data
        out_data = array('d', [0.0] * N_el)

        if other.rows == 1:
            for t in range(T):
                tD = t * D
                for d in range(D):
                    out_data[tD + d] = A[tD + d] + B[d]
        else:
            for i in range(N_el):
                out_data[i] = A[i] + B[i]

        out = FlatTensor(T, D, out_data, (self, other))

        def _backward():
            dOut = out.grad
            dA = self.grad
            dB = other.grad
            for i in range(N_el):
                dA[i] += dOut[i]
            if other.rows == 1:
                for t in range(T):
                    tD = t * D
                    for d in range(D):
                        dB[d] += dOut[tD + d]
            else:
                for i in range(N_el):
                    dB[i] += dOut[i]

        out._backward = _backward
        return out

    def matmul(self, other):
        """Flat 1D Matrix Multiply: [M, K] @ [K, N] -> [M, N]."""
        M, K, N = self.rows, self.cols, other.cols
        A, B = self.data, other.data
        out_data = array('d', [0.0] * (M * N))

        for i in range(M):
            iK = i * K
            iN = i * N
            for k in range(K):
                a_ik = A[iK + k]
                if a_ik == 0.0:
                    continue
                kN = k * N
                for j in range(N):
                    out_data[iN + j] += a_ik * B[kN + j]

        out = FlatTensor(M, N, out_data, (self, other))

        def _backward():
            dOut = out.grad
            dA = self.grad
            dB = other.grad
            for i in range(M):
                iK = i * K
                iN = i * N
                for k in range(K):
                    a_ik = A[iK + k]
                    kN = k * N
                    s = 0.0
                    for j in range(N):
                        dout_ij = dOut[iN + j]
                        s += dout_ij * B[kN + j]
                        dB[kN + j] += a_ik * dout_ij
                    dA[iK + k] += s

        out._backward = _backward
        return out

    def rmsnorm(self, eps=1e-5):
        """Fused Row-wise Root Mean Square Layer Normalization."""
        T, D = self.rows, self.cols
        X = self.data
        out_data = array('d', [0.0] * (T * D))
        inv_rms = array('d', [0.0] * T)

        for t in range(T):
            tD = t * D
            ms = 0.0
            for d in range(D):
                val = X[tD + d]
                ms += val * val
            r = 1.0 / math.sqrt(ms / D + eps)
            inv_rms[t] = r
            for d in range(D):
                out_data[tD + d] = X[tD + d] * r

        out = FlatTensor(T, D, out_data, (self,))

        def _backward():
            Y = out.data
            dOut = out.grad
            dX = self.grad
            for t in range(T):
                tD = t * D
                r = inv_rms[t]
                dot_gy = 0.0
                for d in range(D):
                    dot_gy += dOut[tD + d] * Y[tD + d]
                dot_gy /= D
                for d in range(D):
                    dX[tD + d] += r * (dOut[tD + d] - Y[tD + d] * dot_gy)

        out._backward = _backward
        return out

    @staticmethod
    def swiglu_act(gate, up):
        """Fused SwiGLU nonlinearity: SiLU(gate) * up = (gate * sigmoid(gate)) * up."""
        T, D = gate.rows, gate.cols
        N_el = T * D
        G, U = gate.data, up.data
        out_data = array('d', [0.0] * N_el)
        sig_cache = array('d', [0.0] * N_el)

        for i in range(N_el):
            g = G[i]
            if g >= 0.0:
                z = math.exp(-min(g, 20.0))
                sig = 1.0 / (1.0 + z)
            else:
                z = math.exp(max(g, -20.0))
                sig = z / (1.0 + z)
            sig_cache[i] = sig
            out_data[i] = (g * sig) * U[i]

        out = FlatTensor(T, D, out_data, (gate, up))

        def _backward():
            dOut = out.grad
            dG = gate.grad
            dU = up.grad
            for i in range(N_el):
                dout = dOut[i]
                g = G[i]
                u = U[i]
                sig = sig_cache[i]
                silu = g * sig
                dU[i] += dout * silu
                dsilu = sig * (1.0 + g * (1.0 - sig))
                dG[i] += dout * u * dsilu

        out._backward = _backward
        return out

    @staticmethod
    def embedding(weight, indices):
        """Row lookup from [V, D] -> [T, D]."""
        T = len(indices)
        D = weight.cols
        W = weight.data
        out_data = array('d', [0.0] * (T * D))

        for t, idx in enumerate(indices):
            tD = t * D
            idxD = idx * D
            for d in range(D):
                out_data[tD + d] = W[idxD + d]

        out = FlatTensor(T, D, out_data, (weight,))

        def _backward():
            dW = weight.grad
            dOut = out.grad
            for t, idx in enumerate(indices):
                tD = t * D
                idxD = idx * D
                for d in range(D):
                    dW[idxD + d] += dOut[tD + d]

        out._backward = _backward
        return out

    @staticmethod
    def rope_mha(Q, K, V, n_heads, start_pos=0, kv_cache=None):
        """
        Fused Multi-Head Causal Self-Attention with Rotary Position Embeddings (RoPE)
        and optional KV-Caching for O(1) incremental generation.
        """
        T, D = Q.rows, Q.cols
        head_dim = D // n_heads
        half_d = head_dim // 2
        scale = 1.0 / math.sqrt(head_dim)

        inv_freq = [1.0 / (10000.0 ** ((2 * m) / head_dim)) for m in range(half_d)]

        # 1. Apply RoPE rotation to Q and K
        Q_rot = array('d', [0.0] * (T * D))
        K_rot = array('d', [0.0] * (T * D))
        cos_table = array('d', [0.0] * (T * half_d))
        sin_table = array('d', [0.0] * (T * half_d))

        Qd, Kd = Q.data, K.data
        for t in range(T):
            pos = start_pos + t
            tD = t * D
            t_half = t * half_d
            for m in range(half_d):
                angle = pos * inv_freq[m]
                c = math.cos(angle)
                s = math.sin(angle)
                cos_table[t_half + m] = c
                sin_table[t_half + m] = s
                for h in range(n_heads):
                    base = tD + h * head_dim + 2 * m
                    q0, q1 = Qd[base], Qd[base + 1]
                    k0, k1 = Kd[base], Kd[base + 1]
                    Q_rot[base] = q0 * c - q1 * s
                    Q_rot[base + 1] = q0 * s + q1 * c
                    K_rot[base] = k0 * c - k1 * s
                    K_rot[base + 1] = k0 * s + k1 * c

        # 2. Update or bypass KV-Cache
        if kv_cache is not None:
            kv_cache["k"].extend(K_rot)
            kv_cache["v"].extend(V.data)
            kv_cache["seq_len"] += T
            K_all = kv_cache["k"]
            V_all = kv_cache["v"]
        else:
            K_all = K_rot
            V_all = V.data

        # 3. Multi-Head Causal Attention over [0 .. start_pos + t]
        out_data = array('d', [0.0] * (T * D))
        probs_store = []  # Indexed by [h * T + t]

        for h in range(n_heads):
            h_offset = h * head_dim
            for t in range(T):
                max_k = start_pos + t + 1
                q_base = t * D + h_offset
                scores = [0.0] * max_k
                for k in range(max_k):
                    k_base = k * D + h_offset
                    s_val = 0.0
                    for d in range(head_dim):
                        s_val += Q_rot[q_base + d] * K_all[k_base + d]
                    scores[k] = s_val * scale

                max_s = max(scores)
                exps = [math.exp(s - max_s) for s in scores]
                sum_e = sum(exps)
                probs = [e / sum_e for e in exps]
                probs_store.append(probs)

                out_base = t * D + h_offset
                for k, p in enumerate(probs):
                    v_base = k * D + h_offset
                    for d in range(head_dim):
                        out_data[out_base + d] += p * V_all[v_base + d]

        out = FlatTensor(T, D, out_data, (Q, K, V))

        def _backward():
            # Used during training (start_pos == 0, kv_cache is None)
            dOut = out.grad
            dV = V.grad
            dQ_rot = array('d', [0.0] * (T * D))
            dK_rot = array('d', [0.0] * (T * D))

            for h in range(n_heads):
                h_offset = h * head_dim
                for t in range(T):
                    probs = probs_store[h * T + t]
                    max_k = t + 1
                    q_base = t * D + h_offset
                    out_base = q_base

                    dP = [0.0] * max_k
                    for k in range(max_k):
                        pk = probs[k]
                        v_base = k * D + h_offset
                        dp_k = 0.0
                        for d in range(head_dim):
                            do_d = dOut[out_base + d]
                            dV[v_base + d] += pk * do_d
                            dp_k += do_d * V_all[v_base + d]
                        dP[k] = dp_k

                    dot_p_dp = sum(probs[k] * dP[k] for k in range(max_k))
                    for k in range(max_k):
                        ds = probs[k] * (dP[k] - dot_p_dp) * scale
                        k_base = k * D + h_offset
                        for d in range(head_dim):
                            dQ_rot[q_base + d] += ds * K_rot[k_base + d]
                            dK_rot[k_base + d] += ds * Q_rot[q_base + d]

            # Inverse RoPE rotation R(-phi) back to Q.grad and K.grad
            dQ, dK = Q.grad, K.grad
            for t in range(T):
                tD = t * D
                t_half = t * half_d
                for m in range(half_d):
                    c = cos_table[t_half + m]
                    s = sin_table[t_half + m]
                    for h in range(n_heads):
                        base = tD + h * head_dim + 2 * m
                        dq0, dq1 = dQ_rot[base], dQ_rot[base + 1]
                        dk0, dk1 = dK_rot[base], dK_rot[base + 1]
                        dQ[base] += dq0 * c + dq1 * s
                        dQ[base + 1] += -dq0 * s + dq1 * c
                        dK[base] += dk0 * c + dk1 * s
                        dK[base + 1] += -dk0 * s + dk1 * c

        out._backward = _backward
        return out

    @staticmethod
    def cross_entropy(logits, targets):
        """Fused Softmax + Cross-Entropy over [T, V] flat logits."""
        T, V = logits.rows, logits.cols
        L = logits.data
        probs_all = []
        total_loss = 0.0

        for t in range(T):
            tV = t * V
            row = L[tV : tV + V]
            max_l = max(row)
            exps = [math.exp(x - max_l) for x in row]
            sum_e = sum(exps)
            probs = [e / sum_e for e in exps]
            probs_all.append(probs)
            total_loss -= math.log(max(probs[targets[t]], 1e-12))

        out = FlatTensor(1, 1, array('d', [total_loss / T]), (logits,))

        def _backward():
            grad_scale = out.grad[0] / T
            dL = logits.grad
            for t in range(T):
                tV = t * V
                probs = probs_all[t]
                target = targets[t]
                for v in range(V):
                    indicator = 1.0 if v == target else 0.0
                    dL[tV + v] += (probs[v] - indicator) * grad_scale

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
        self.grad[0] = 1.0
        for v in reversed(topo):
            v._backward()


# =====================================================================
# 2. BPE TOKENIZER (Pre-Seeded with Printable ASCII for Continual Chat)
# =====================================================================
class BPETokenizer:
    BASE_ALPHABET = (
        " \n.,!?'\"-:;()0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    def __init__(self):
        self.vocab = {}
        self.stoi = {}
        self.merges = []

    def train(self, text, max_merges=14, min_freq=5):
        chars = sorted(list(set(self.BASE_ALPHABET + text)))
        self.vocab = {i: ch for i, ch in enumerate(chars)}
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        ids = [self.stoi[c] for c in text]

        for _ in range(max_merges):
            counts = {}
            for a, b in zip(ids, ids[1:]):
                if ' ' in self.vocab[a] or ' ' in self.vocab[b] or '\n' in self.vocab[a] or '\n' in self.vocab[b]:
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
        return "".join(self.vocab.get(i, "") for i in ids)

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
# 3. MULTI-LAYER TRANSFORMER (RoPE + MHA + SwiGLU + KV-Cache)
# =====================================================================
class ModernMicroGPT:
    def __init__(self, vocab_size, n_embd=24, n_heads=4, n_layers=2, mlp_hidden=48):
        assert n_embd % n_heads == 0 and (n_embd // n_heads) % 2 == 0, "head_dim must be even for RoPE"
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_hidden = mlp_hidden
        self.total_steps = 0

        # Scale residual output projections by 1 / sqrt(2 * n_layers)
        res_std = 0.08 / math.sqrt(2.0 * n_layers)

        self.weights = {
            "tok_emb": FlatTensor.randn(vocab_size, n_embd, std=0.08),
            "lm_head": FlatTensor.randn(n_embd, vocab_size, std=0.08),
            "b_head": FlatTensor.zeros(1, vocab_size),
        }

        for l in range(n_layers):
            p = f"blocks.{l}."
            # Multi-Head Attention projections
            self.weights[p + "wq"] = FlatTensor.randn(n_embd, n_embd, std=0.08)
            self.weights[p + "wk"] = FlatTensor.randn(n_embd, n_embd, std=0.08)
            self.weights[p + "wv"] = FlatTensor.randn(n_embd, n_embd, std=0.08)
            self.weights[p + "wo"] = FlatTensor.randn(n_embd, n_embd, std=res_std)
            # SwiGLU Feed-Forward Network projections
            self.weights[p + "w_gate"] = FlatTensor.randn(n_embd, mlp_hidden, std=0.08)
            self.weights[p + "w_up"] = FlatTensor.randn(n_embd, mlp_hidden, std=0.08)
            self.weights[p + "w_down"] = FlatTensor.randn(mlp_hidden, n_embd, std=res_std)

    def init_kv_cache(self):
        return [{"k": array('d'), "v": array('d'), "seq_len": 0} for _ in range(self.n_layers)]

    def forward(self, idx_seq, start_pos=0, kv_caches=None):
        w = self.weights
        x = FlatTensor.embedding(w["tok_emb"], idx_seq)

        for l in range(self.n_layers):
            p = f"blocks.{l}."
            layer_cache = kv_caches[l] if kv_caches is not None else None

            # 1. Pre-Norm Multi-Head Causal RoPE Attention
            x_norm = x.rmsnorm()
            Q = x_norm.matmul(w[p + "wq"])
            K = x_norm.matmul(w[p + "wk"])
            V = x_norm.matmul(w[p + "wv"])
            attn_out = FlatTensor.rope_mha(Q, K, V, self.n_heads, start_pos=start_pos, kv_cache=layer_cache)
            x = x + attn_out.matmul(w[p + "wo"])

            # 2. Pre-Norm SwiGLU Feed-Forward Block
            x_norm2 = x.rmsnorm()
            gate = x_norm2.matmul(w[p + "w_gate"])
            up = x_norm2.matmul(w[p + "w_up"])
            hidden = FlatTensor.swiglu_act(gate, up)
            x = x + hidden.matmul(w[p + "w_down"])

        logits = x.rmsnorm().matmul(w["lm_head"]) + w["b_head"]
        return logits

    def step_adam(self, lr):
        self.total_steps += 1
        b1_corr = 1.0 - 0.9 ** self.total_steps
        b2_corr = 1.0 - 0.999 ** self.total_steps

        for param in self.weights.values():
            N_el = param.rows * param.cols
            if param.m is None:
                param.m = array('d', [0.0] * N_el)
                param.v = array('d', [0.0] * N_el)
            pd, pg, pm, pv = param.data, param.grad, param.m, param.v
            for i in range(N_el):
                g = pg[i]
                # Gradient clipping at [-1.0, 1.0] for multi-layer stability
                if g > 1.0:
                    g = 1.0
                elif g < -1.0:
                    g = -1.0
                pm[i] = 0.9 * pm[i] + 0.1 * g
                pv[i] = 0.999 * pv[i] + 0.001 * (g * g)
                m_hat = pm[i] / b1_corr
                v_hat = pv[i] / b2_corr
                pd[i] -= lr * m_hat / (math.sqrt(v_hat) + 1e-8)
                pg[i] = 0.0

    @staticmethod
    def sample_top_k_top_p(logits_row, temperature=0.2, top_k=8, top_p=0.9):
        if temperature <= 1e-4:
            return max(range(len(logits_row)), key=lambda i: logits_row[i])

        scaled = [(val / temperature, idx) for idx, val in enumerate(logits_row)]
        scaled.sort(key=lambda x: x[0], reverse=True)

        if top_k > 0:
            scaled = scaled[:min(top_k, len(scaled))]

        max_l = scaled[0][0]
        exps = [math.exp(l - max_l) for l, _ in scaled]
        total = sum(exps)
        probs = [e / total for e in exps]

        # Top-P (Nucleus) filtering
        filtered_indices, filtered_probs = [], []
        cum_prob = 0.0
        for (_, idx), p in zip(scaled, probs):
            filtered_indices.append(idx)
            filtered_probs.append(p)
            cum_prob += p
            if cum_prob >= top_p:
                break

        return random.choices(filtered_indices, weights=filtered_probs)[0]

    def generate(self, prompt_ids, max_new_tokens=20, temperature=0.1, top_k=8, top_p=0.9, use_kv_cache=True):
        if not prompt_ids:
            return []
        idx = list(prompt_ids)
        V = self.vocab_size

        if use_kv_cache:
            kv_caches = self.init_kv_cache()
            # Prefill prompt in one pass
            logits = self.forward(idx, start_pos=0, kv_caches=kv_caches)
            last_row = logits.data[(logits.rows - 1) * V : logits.rows * V]
            next_id = self.sample_top_k_top_p(last_row, temperature, top_k, top_p)
            idx.append(next_id)

            # Decode subsequent tokens in O(1) incremental steps ([1, D] input)
            for _ in range(max_new_tokens - 1):
                cur_pos = kv_caches[0]["seq_len"]
                logits = self.forward([idx[-1]], start_pos=cur_pos, kv_caches=kv_caches)
                last_row = logits.data[0 : V]
                next_id = self.sample_top_k_top_p(last_row, temperature, top_k, top_p)
                idx.append(next_id)
        else:
            for _ in range(max_new_tokens):
                logits = self.forward(idx, start_pos=0, kv_caches=None)
                last_row = logits.data[(logits.rows - 1) * V : logits.rows * V]
                next_id = self.sample_top_k_top_p(last_row, temperature, top_k, top_p)
                idx.append(next_id)

        return idx

    def save_checkpoint(self, filepath, tokenizer, corpus_segments):
        payload = {
            "config": {
                "vocab_size": self.vocab_size,
                "n_embd": self.n_embd,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "mlp_hidden": self.mlp_hidden,
                "total_steps": self.total_steps,
            },
            "tokenizer": tokenizer.to_dict(),
            "corpus_segments": corpus_segments,
            "weights": {name: tensor.to_list_2d() for name, tensor in self.weights.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load_checkpoint(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg = payload["config"]
        model = cls(
            vocab_size=cfg["vocab_size"],
            n_embd=cfg["n_embd"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            mlp_hidden=cfg["mlp_hidden"],
        )
        model.total_steps = cfg.get("total_steps", 0)
        for name, mat in payload["weights"].items():
            model.weights[name] = FlatTensor.from_list_2d(mat)
        tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
        corpus_segments = payload.get("corpus_segments", [])
        return model, tokenizer, corpus_segments


# =====================================================================
# 4. COSINE LR SCHEDULE & ONLINE CONTINUAL LEARNING HELPERS
# =====================================================================
def get_cosine_lr(step, total_steps, max_lr=0.012, min_lr=0.001, warmup_steps=15):
    if step <= warmup_steps:
        return max_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_sliding_windows(token_ids, block_size=16):
    if len(token_ids) <= block_size:
        padded = (token_ids * ((block_size // max(1, len(token_ids))) + 2))[: block_size + 1]
        return [(padded[:-1], padded[1:])]
    return [
        (token_ids[i : i + block_size], token_ids[i + 1 : i + block_size + 1])
        for i in range(len(token_ids) - block_size)
    ]


def continual_learn_on_turn(model, tokenizer, corpus_segments, new_text, ckpt_path, block_size=16):
    """
    Adds new_text to the persistent corpus, runs online continual gradient updates
    mixing the new text with replay windows from prior segments, and saves to JSON.
    """
    clean_text = new_text.strip()
    if not clean_text.endswith(" "):
        clean_text += " "

    if clean_text not in corpus_segments:
        corpus_segments.append(clean_text)

    # Build windows focused on the new text + replay windows from existing knowledge
    new_ids = tokenizer.encode(clean_text * 3)
    new_windows = build_sliding_windows(new_ids, block_size=min(block_size, max(4, len(new_ids) - 1)))

    full_ids = tokenizer.encode("".join(corpus_segments) * 2)
    replay_pool = build_sliding_windows(full_ids, block_size=block_size)

    online_epochs = 12
    total_online_steps = online_epochs * (len(new_windows) + min(6, len(replay_pool)))
    local_step = 0
    last_loss = 0.0

    for _ in range(online_epochs):
        replay_sample = random.sample(replay_pool, k=min(6, len(replay_pool)))
        batch_windows = new_windows + replay_sample
        random.shuffle(batch_windows)
        epoch_loss = 0.0
        for x_seq, y_seq in batch_windows:
            local_step += 1
            lr = get_cosine_lr(local_step, total_online_steps, max_lr=0.008, min_lr=0.001, warmup_steps=4)
            logits = model.forward(x_seq)
            loss = FlatTensor.cross_entropy(logits, y_seq)
            loss.backward()
            model.step_adam(lr)
            epoch_loss += loss.data[0]
        last_loss = epoch_loss / max(1, len(batch_windows))

    model.save_checkpoint(ckpt_path, tokenizer, corpus_segments)
    return last_loss


# =====================================================================
# 5. MAIN ENTRYPOINT: AUTO-RESUME CHECKPOINT + INTERACTIVE REPL
# =====================================================================
if __name__ == "__main__":
    ckpt_path = "micro_gpt_checkpoint.json"
    block_size = 16

    if os.path.exists(ckpt_path):
        print(f"Found existing checkpoint '{ckpt_path}'! Loading model & BPE state...")
        model, tokenizer, corpus_segments = ModernMicroGPT.load_checkpoint(ckpt_path)
        total_params = sum(t.rows * t.cols for t in model.weights.values())
        print(
            f"Loaded {model.n_layers}-Layer RoPE-MHA-SwiGLU GPT | "
            f"Params: {total_params} | Accumulated Steps: {model.total_steps} | "
            f"Learned Segments: {len(corpus_segments)}\n"
        )
    else:
        print("No checkpoint found. Initializing & training 2-Layer RoPE-MHA-SwiGLU GPT from scratch...")
        corpus_segments = [
            "deep learning in pure python. ",
            "small models learn patterns fast. ",
            "attention is all you need. ",
        ]
        base_text = "".join(corpus_segments)
        full_corpus = base_text * 4

        tokenizer = BPETokenizer()
        tokenizer.train(full_corpus, max_merges=14, min_freq=5)
        data = tokenizer.encode(full_corpus)

        model = ModernMicroGPT(
            vocab_size=len(tokenizer.vocab),
            n_embd=24,
            n_heads=4,
            n_layers=2,
            mlp_hidden=48,
        )
        total_params = sum(t.rows * t.cols for t in model.weights.values())
        print(f"BPE Vocab: {len(tokenizer.vocab)} tokens | Model Parameters: {total_params}")

        cycle_len = len(tokenizer.encode(base_text))
        windows = [
            (data[i : i + block_size], data[i + 1 : i + block_size + 1])
            for i in range(cycle_len)
        ]

        epochs = 32
        total_train_steps = epochs * len(windows)
        local_step = 0

        for epoch in range(1, epochs + 1):
            random.shuffle(windows)
            epoch_loss = 0.0
            for x_seq, y_seq in windows:
                local_step += 1
                lr = get_cosine_lr(local_step, total_train_steps, max_lr=0.012, min_lr=0.0008, warmup_steps=20)
                logits = model.forward(x_seq)
                loss = FlatTensor.cross_entropy(logits, y_seq)
                loss.backward()
                model.step_adam(lr)
                epoch_loss += loss.data[0]

            if epoch % 8 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:2d}/{epochs} | "
                    f"Avg Loss: {epoch_loss / len(windows):.4f} | "
                    f"Cosine LR: {lr:.5f}"
                )

        model.save_checkpoint(ckpt_path, tokenizer, corpus_segments)
        print(f"\nSaved initial checkpoint to '{ckpt_path}' ({os.path.getsize(ckpt_path) // 1024} KB)\n")

    # Verify generation & KV-Cache speedup
    print("--- Inference Verification (Top-K=6, Top-P=0.9, KV-Cached) ---")
    t0 = time.perf_counter()
    for prompt in ["deep ", "small ", "atten"]:
        p_ids = tokenizer.encode(prompt)
        out_ids = model.generate(p_ids, max_new_tokens=18, temperature=0.1, top_k=6, top_p=0.9, use_kv_cache=True)
        print(f"Prompt '{prompt}' -> {tokenizer.decode(out_ids)}")
    dt_cached = (time.perf_counter() - t0) * 1000.0
    print(f"(Generated 3 sequences with KV-Cache in {dt_cached:.1f} ms)\n")

    # Interactive Continual-Learning Chat / REPL
    print("=====================================================================")
    print("Interactive Continual-Learning REPL (Top-K / Top-P Sampling)")
    print("Commands:")
    print("  - Type any prompt/sentence: generates a response, trains on your")
    print("    input with replay, and updates 'micro_gpt_checkpoint.json'!")
    print("  - /set k=8 p=0.9 temp=0.3 : adjust Top-K, Top-P, and temperature")
    print("  - /quit or exit           : exit the REPL")
    print("=====================================================================")

    top_k, top_p, temperature = 6, 0.9, 0.15

    while True:
        try:
            user_in = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting REPL. Checkpoint is up to date.")
            break

        if not user_in:
            continue
        if user_in.lower() in ("/quit", "exit", "quit"):
            print("Goodbye! All learned updates are saved in", ckpt_path)
            break

        if user_in.startswith("/set"):
            for part in user_in.split()[1:]:
                if part.startswith("k="):
                    top_k = int(part.split("=")[1])
                elif part.startswith("p="):
                    top_p = float(part.split("=")[1])
                elif part.startswith("temp="):
                    temperature = float(part.split("=")[1])
            print(f"Updated sampling -> top_k={top_k}, top_p={top_p}, temperature={temperature}")
            continue

        # 1. Generate completion from current weights using KV-cache
        prompt_ids = tokenizer.encode(user_in)
        gen_ids = model.generate(
            prompt_ids,
            max_new_tokens=18,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_kv_cache=True,
        )
        print(f"Model (before update) > {tokenizer.decode(gen_ids)}")

        # 2. If the user entered a statement/phrase (>= 8 chars), learn it & update JSON checkpoint
        if len(user_in) >= 8:
            loss_after = continual_learn_on_turn(model, tokenizer, corpus_segments, user_in, ckpt_path)
            # Show how the model completes the start of the user's phrase now that it learned it
            prefix = user_in[: max(4, len(user_in) // 3)]
            verify_ids = model.generate(
                tokenizer.encode(prefix),
                max_new_tokens=18,
                temperature=0.05,
                top_k=top_k,
                top_p=top_p,
                use_kv_cache=True,
            )
            print(
                f"[Learned & Saved to '{ckpt_path}' | Loss: {loss_after:.4f} | Total Steps: {model.total_steps}]\n"
                f"Model (after learning '{prefix}...') > {tokenizer.decode(verify_ids)}"
            )
