from array import array
import base64
import json
import math
import os
import random
import struct
import time
import zlib

random.seed(42)

# =====================================================================
# 1. FLAT 1D array('d') AUTOGRAD ENGINE + BINARY COMPRESSION
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

    def to_compressed_dict(self):
        """Packs flat doubles into IEEE-754 float32 binary, compresses with zlib, and encodes in base64."""
        n = self.rows * self.cols
        raw_bytes = struct.pack(f"<{n}f", *self.data)
        compressed = zlib.compress(raw_bytes, level=9)
        return {
            "rows": self.rows,
            "cols": self.cols,
            "encoding": "fp32_zlib_b64",
            "blob": base64.b64encode(compressed).decode("ascii"),
        }

    @classmethod
    def from_serialized(cls, obj):
        """Loads either a compressed binary dict (v3) or a legacy 2D float list (v2)."""
        if isinstance(obj, list):
            rows = len(obj)
            cols = len(obj[0]) if rows > 0 else 0
            return cls(rows, cols, array('d', (x for row in obj for x in row)))

        rows, cols = obj["rows"], obj["cols"]
        n = rows * cols
        raw_bytes = zlib.decompress(base64.b64decode(obj["blob"].encode("ascii")))
        floats = struct.unpack(f"<{n}f", raw_bytes)
        return cls(rows, cols, array('d', floats))

    def __add__(self, other):
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
            dOut, dA, dB = out.grad, self.grad, other.grad
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
            dOut, dA, dB = out.grad, self.grad, other.grad
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
            Y, dOut, dX = out.data, out.grad, self.grad
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
            dOut, dG, dU = out.grad, gate.grad, up.grad
            for i in range(N_el):
                dout = dOut[i]
                g, u, sig = G[i], U[i], sig_cache[i]
                dU[i] += dout * (g * sig)
                dG[i] += dout * u * (sig * (1.0 + g * (1.0 - sig)))

        out._backward = _backward
        return out

    @staticmethod
    def embedding(weight, indices):
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
            dW, dOut = weight.grad, out.grad
            for t, idx in enumerate(indices):
                tD = t * D
                idxD = idx * D
                for d in range(D):
                    dW[idxD + d] += dOut[tD + d]

        out._backward = _backward
        return out

    @staticmethod
    def rope_mha(Q, K, V, n_heads, start_pos=0, kv_cache=None):
        T, D = Q.rows, Q.cols
        head_dim = D // n_heads
        half_d = head_dim // 2
        scale = 1.0 / math.sqrt(head_dim)
        inv_freq = [1.0 / (10000.0 ** ((2 * m) / head_dim)) for m in range(half_d)]

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
                c, s = math.cos(angle), math.sin(angle)
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

        if kv_cache is not None:
            kv_cache["k"].extend(K_rot)
            kv_cache["v"].extend(V.data)
            kv_cache["seq_len"] += T
            K_all, V_all = kv_cache["k"], kv_cache["v"]
        else:
            K_all, V_all = K_rot, V.data

        out_data = array('d', [0.0] * (T * D))
        probs_store = []

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
            dOut, dV = out.grad, V.grad
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
# 2. DYNAMIC BPE TOKENIZER (Supports Live Vocabulary Expansion)
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
        self.merges = []
        self.expand_vocab(text, max_new_merges=max_merges, min_freq=min_freq)
        return self.encode(text)

    def expand_vocab(self, text, max_new_merges=3, min_freq=4):
        """
        Detects unseen characters and high-frequency unmerged token pairs in `text`,
        mints new BPE token IDs, and returns [(new_id, parent_a, parent_b), ...].
        """
        new_tokens_info = []

        # 1. Register any unseen raw Unicode characters first
        for ch in text:
            if ch not in self.stoi:
                new_id = len(self.vocab)
                self.vocab[new_id] = ch
                self.stoi[ch] = new_id
                new_tokens_info.append((new_id, None, None))

        # 2. Encode text with current merges and search for frequent new pairs
        ids = self.encode(text)
        for _ in range(max_new_merges):
            counts = {}
            for a, b in zip(ids, ids[1:]):
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
# 3. MULTI-LAYER TRANSFORMER WITH DYNAMIC VOCAB SURGERY & BINARY IO
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

        res_std = 0.08 / math.sqrt(2.0 * n_layers)

        self.weights = {
            "tok_emb": FlatTensor.randn(vocab_size, n_embd, std=0.08),
            "lm_head": FlatTensor.randn(n_embd, vocab_size, std=0.08),
            "b_head": FlatTensor.zeros(1, vocab_size),
        }

        for l in range(n_layers):
            p = f"blocks.{l}."
            self.weights[p + "wq"] = FlatTensor.randn(n_embd, n_embd, std=0.08)
            self.weights[p + "wk"] = FlatTensor.randn(n_embd, n_embd, std=0.08)
            self.weights[p + "wv"] = FlatTensor.randn(n_embd, n_embd, std=0.08)
            self.weights[p + "wo"] = FlatTensor.randn(n_embd, n_embd, std=res_std)
            self.weights[p + "w_gate"] = FlatTensor.randn(n_embd, mlp_hidden, std=0.08)
            self.weights[p + "w_up"] = FlatTensor.randn(n_embd, mlp_hidden, std=0.08)
            self.weights[p + "w_down"] = FlatTensor.randn(mlp_hidden, n_embd, std=res_std)

    def expand_vocab(self, new_tokens_info):
        """
        Dynamically expands tok_emb [V, D], lm_head [D, V], and b_head [1, V]
        when BPE mints new subwords, initializing each new token from the average
        of its parent token embeddings/logits while preserving Adam states.
        """
        if not new_tokens_info:
            return

        D = self.n_embd
        tok_emb = self.weights["tok_emb"]
        lm_head = self.weights["lm_head"]
        b_head = self.weights["b_head"]

        for new_id, pa, pb in new_tokens_info:
            assert new_id == self.vocab_size, "Token IDs must grow sequentially"
            V_old = self.vocab_size
            V_new = V_old + 1

            # 1. Expand tok_emb: [V_old, D] -> [V_new, D]
            for d in range(D):
                if pa is not None and pb is not None:
                    val = 0.5 * (tok_emb.data[pa * D + d] + tok_emb.data[pb * D + d]) + random.gauss(0.0, 0.005)
                else:
                    val = random.gauss(0.0, 0.08)
                tok_emb.data.append(val)
                tok_emb.grad.append(0.0)
                if tok_emb.m is not None:
                    tok_emb.m.append(0.0)
                    tok_emb.v.append(0.0)
            tok_emb.rows = V_new

            # 2. Expand lm_head: [D, V_old] -> [D, V_new] (insert new column per row)
            new_head_data = array('d', [0.0] * (D * V_new))
            new_head_m = array('d', [0.0] * (D * V_new)) if lm_head.m is not None else None
            new_head_v = array('d', [0.0] * (D * V_new)) if lm_head.v is not None else None

            for d in range(D):
                old_row_start = d * V_old
                new_row_start = d * V_new
                for v in range(V_old):
                    new_head_data[new_row_start + v] = lm_head.data[old_row_start + v]
                    if new_head_m is not None:
                        new_head_m[new_row_start + v] = lm_head.m[old_row_start + v]
                        new_head_v[new_row_start + v] = lm_head.v[old_row_start + v]

                if pa is not None and pb is not None:
                    col_val = 0.5 * (lm_head.data[old_row_start + pa] + lm_head.data[old_row_start + pb]) + random.gauss(0.0, 0.005)
                else:
                    col_val = random.gauss(0.0, 0.08)
                new_head_data[new_row_start + V_old] = col_val

            lm_head.cols = V_new
            lm_head.data = new_head_data
            lm_head.grad = array('d', [0.0] * (D * V_new))
            lm_head.m = new_head_m
            lm_head.v = new_head_v

            # 3. Expand b_head: [1, V_old] -> [1, V_new]
            b_val = 0.5 * (b_head.data[pa] + b_head.data[pb]) if (pa is not None and pb is not None) else 0.0
            b_head.data.append(b_val)
            b_head.grad.append(0.0)
            if b_head.m is not None:
                b_head.m.append(0.0)
                b_head.v.append(0.0)
            b_head.cols = V_new

            self.vocab_size = V_new

    def init_kv_cache(self):
        return [{"k": array('d'), "v": array('d'), "seq_len": 0} for _ in range(self.n_layers)]

    def forward(self, idx_seq, start_pos=0, kv_caches=None):
        w = self.weights
        x = FlatTensor.embedding(w["tok_emb"], idx_seq)

        for l in range(self.n_layers):
            p = f"blocks.{l}."
            layer_cache = kv_caches[l] if kv_caches is not None else None

            x_norm = x.rmsnorm()
            Q = x_norm.matmul(w[p + "wq"])
            K = x_norm.matmul(w[p + "wk"])
            V = x_norm.matmul(w[p + "wv"])
            attn_out = FlatTensor.rope_mha(Q, K, V, self.n_heads, start_pos=start_pos, kv_cache=layer_cache)
            x = x + attn_out.matmul(w[p + "wo"])

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
            logits = self.forward(idx, start_pos=0, kv_caches=kv_caches)
            last_row = logits.data[(logits.rows - 1) * V : logits.rows * V]
            next_id = self.sample_top_k_top_p(last_row, temperature, top_k, top_p)
            idx.append(next_id)

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
                "compression": "fp32_zlib_b64",
            },
            "tokenizer": tokenizer.to_dict(),
            "corpus_segments": corpus_segments,
            "weights": {name: tensor.to_compressed_dict() for name, tensor in self.weights.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

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
        for name, serialized in payload["weights"].items():
            model.weights[name] = FlatTensor.from_serialized(serialized)
        tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
        corpus_segments = payload.get("corpus_segments", [])
        return model, tokenizer, corpus_segments


# =====================================================================
# 4. COSINE LR & CONTINUAL LEARNING WITH DYNAMIC VOCAB GROWTH
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
    1. Expands the BPE vocabulary if new subword patterns cross the frequency threshold.
    2. Dynamically resizes the model's embedding & output head matrices using parent-averaging.
    3. Trains on the new text + replay windows and saves compressed binary weights to JSON.
    """
    clean_text = new_text.strip()
    if not clean_text.endswith(" "):
        clean_text += " "

    if clean_text not in corpus_segments:
        corpus_segments.append(clean_text)

    # 1. Dynamic BPE Vocabulary Expansion on combined corpus + emphasized new input
    candidate_corpus = "".join(corpus_segments) + (clean_text * 4)
    new_tokens_info = tokenizer.expand_vocab(candidate_corpus, max_new_merges=3, min_freq=4)
    if new_tokens_info:
        model.expand_vocab(new_tokens_info)

    # 2. Build sliding windows with the newly expanded vocabulary
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
    minted_labels = [f"'{tokenizer.vocab[nid]}' (id={nid})" for nid, _, _ in new_tokens_info]
    return last_loss, minted_labels


# =====================================================================
# 5. MAIN ENTRYPOINT: AUTO-RESUME + CONTINUAL LEARNING REPL
# =====================================================================
if __name__ == "__main__":
    ckpt_path = "micro_gpt_checkpoint.json"
    block_size = 16

    if os.path.exists(ckpt_path):
        old_size_kb = os.path.getsize(ckpt_path) / 1024.0
        print(f"Found existing checkpoint '{ckpt_path}' ({old_size_kb:.1f} KB)! Loading...")
        model, tokenizer, corpus_segments = ModernMicroGPT.load_checkpoint(ckpt_path)
        # Re-save immediately to upgrade any legacy v2 uncompressed checkpoint to binary v3
        model.save_checkpoint(ckpt_path, tokenizer, corpus_segments)
        new_size_kb = os.path.getsize(ckpt_path) / 1024.0
        total_params = sum(t.rows * t.cols for t in model.weights.values())
        print(
            f"Loaded {model.n_layers}-Layer RoPE-MHA-SwiGLU GPT | "
            f"Vocab: {model.vocab_size} tokens | Params: {total_params} | "
            f"Compressed File Size: {new_size_kb:.1f} KB\n"
        )
    else:
        print("No checkpoint found. Training 2-Layer RoPE-MHA-SwiGLU GPT from scratch...")
        corpus_segments = [
            "deep learning in pure python. ",
            "small models learn patterns fast. ",
            "attention is all you need. ",
        ]
        base_text = "".join(corpus_segments)
        full_corpus = base_text * 4

        tokenizer = BPETokenizer()
        data = tokenizer.train(full_corpus, max_merges=14, min_freq=5)

        model = ModernMicroGPT(
            vocab_size=len(tokenizer.vocab),
            n_embd=24,
            n_heads=4,
            n_layers=2,
            mlp_hidden=48,
        )
        total_params = sum(t.rows * t.cols for t in model.weights.values())
        print(f"Initial BPE Vocab: {len(tokenizer.vocab)} tokens | Model Parameters: {total_params}")

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
        size_kb = os.path.getsize(ckpt_path) / 1024.0
        print(f"\nSaved compressed binary checkpoint to '{ckpt_path}' ({size_kb:.1f} KB — down from ~332 KB!)\n")

    # Verify generation
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
    print("Interactive Continual-Learning REPL (Dynamic BPE + Binary Checkpoints)")
    print("Commands:")
    print("  - Type any sentence: generates a reply, dynamically mints new BPE")
    print("    subwords, expands model matrices, trains, & saves compressed JSON")
    print("  - /stats                  : inspect vocab, merges, & compression")
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

        if user_in.startswith("/stats"):
            total_params = sum(t.rows * t.cols for t in model.weights.values())
            raw_fp64_kb = (total_params * 8) / 1024.0
            file_kb = os.path.getsize(ckpt_path) / 1024.0 if os.path.exists(ckpt_path) else 0.0
            subwords = [tokenizer.vocab[nid] for _, _, nid in tokenizer.merges]
            print(
                f"Vocab Size: {model.vocab_size} tokens ({len(subwords)} BPE merges: {subwords})\n"
                f"Parameters: {total_params} floats ({raw_fp64_kb:.1f} KB raw fp64 -> {file_kb:.1f} KB compressed JSON)"
            )
            continue

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

        # 1. Generate completion from current weights
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

        # 2. Dynamically expand BPE + model weights, run continual learning, and save compressed JSON
        if len(user_in) >= 8:
            loss_after, minted = continual_learn_on_turn(
                model, tokenizer, corpus_segments, user_in, ckpt_path
            )
            total_params = sum(t.rows * t.cols for t in model.weights.values())
            file_kb = os.path.getsize(ckpt_path) / 1024.0
            mint_msg = f"Minted new BPE tokens: {', '.join(minted)} | " if minted else ""

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
                f"[{mint_msg}Vocab: {model.vocab_size} | Params: {total_params} | "
                f"Loss: {loss_after:.4f} | Saved '{ckpt_path}' ({file_kb:.1f} KB)]\n"
                f"Model (after learning '{prefix}...') > {tokenizer.decode(verify_ids)}"
            )
