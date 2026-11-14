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
# 1. FLAT 1D array('d') + INT8 array('b') AUTOGRAD ENGINE WITH GQA
# =====================================================================
class FlatTensor:
    """
    2D Matrix with contiguous float64 training storage (`array('d')`)
    and symmetric per-tensor Int8 storage (`array('b')`) for fast matmul & 4x smaller IO.
    """
    __slots__ = ('rows', 'cols', 'data', 'grad', '_backward', '_prev', 'm', 'v', 'q_data', 'q_scale')

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
        self.q_data = None
        self.q_scale = 1.0

    @staticmethod
    def randn(rows, cols, std=0.08):
        size = rows * cols
        t = FlatTensor(rows, cols, array('d', (random.gauss(0.0, std) for _ in range(size))))
        t.quantize_int8()
        return t

    @staticmethod
    def zeros(rows, cols):
        t = FlatTensor(rows, cols)
        t.quantize_int8()
        return t

    def quantize_int8(self):
        n = self.rows * self.cols
        if n == 0:
            self.q_data = array('b')
            self.q_scale = 1.0
            return

        max_abs = 0.0
        for x in self.data:
            ax = abs(x)
            if ax > max_abs:
                max_abs = ax

        if max_abs < 1e-12:
            self.q_scale = 1.0
            self.q_data = array('b', [0] * n)
        else:
            scale = max_abs / 127.0
            inv_scale = 127.0 / max_abs
            self.q_scale = scale
            self.q_data = array(
                'b',
                (max(-127, min(127, int(round(x * inv_scale)))) for x in self.data)
            )

    def to_int8_compressed_dict(self):
        if self.q_data is None or len(self.q_data) != self.rows * self.cols:
            self.quantize_int8()
        n = self.rows * self.cols
        raw_bytes = struct.pack(f"<{n}b", *self.q_data)
        compressed = zlib.compress(raw_bytes, level=9)
        return {
            "rows": self.rows,
            "cols": self.cols,
            "encoding": "int8_zlib_b64",
            "scale": self.q_scale,
            "blob": base64.b64encode(compressed).decode("ascii"),
        }

    @classmethod
    def from_serialized(cls, obj):
        if isinstance(obj, list):
            rows = len(obj)
            cols = len(obj[0]) if rows > 0 else 0
            t = cls(rows, cols, array('d', (x for row in obj for x in row)))
            t.quantize_int8()
            return t

        rows, cols = obj["rows"], obj["cols"]
        n = rows * cols
        enc = obj.get("encoding", "fp32_zlib_b64")
        raw_bytes = zlib.decompress(base64.b64decode(obj["blob"].encode("ascii")))

        if enc == "int8_zlib_b64":
            scale = float(obj["scale"])
            q_ints = struct.unpack(f"<{n}b", raw_bytes)
            t = cls(rows, cols, array('d', (q * scale for q in q_ints)))
            t.q_data = array('b', q_ints)
            t.q_scale = scale
            return t
        else:
            floats = struct.unpack(f"<{n}f", raw_bytes)
            t = cls(rows, cols, array('d', floats))
            t.quantize_int8()
            return t

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

    def matmul(self, other, use_int8=False):
        if use_int8:
            return self.matmul_int8(other)

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

    def matmul_int8(self, other):
        M, K, N = self.rows, self.cols, other.cols
        if other.q_data is None or len(other.q_data) != K * N:
            other.quantize_int8()

        A_fp = self.data
        max_a = 0.0
        for x in A_fp:
            ax = abs(x)
            if ax > max_a:
                max_a = ax

        if max_a < 1e-12:
            return FlatTensor(M, N)

        scale_a = max_a / 127.0
        inv_a = 127.0 / max_a
        A_q = array('b', (max(-127, min(127, int(round(x * inv_a)))) for x in A_fp))

        B_q = other.q_data
        combined_scale = scale_a * other.q_scale
        out_int = [0] * (M * N)

        for i in range(M):
            iK = i * K
            iN = i * N
            for k in range(K):
                a_ik = A_q[iK + k]
                if a_ik == 0:
                    continue
                kN = k * N
                for j in range(N):
                    out_int[iN + j] += a_ik * B_q[kN + j]

        out_data = array('d', (val * combined_scale for val in out_int))
        return FlatTensor(M, N, out_data)

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
    def rope_gqa(Q, K, V, n_heads, n_kv_heads, start_pos=0, kv_cache=None):
        """
        Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE)
        and Int8 KV-Cache Compression (`k_int8`, `v_int8` in signed 1-byte array('b')).
        """
        T, D = Q.rows, Q.cols
        KV_D = K.cols
        head_dim = D // n_heads
        group_size = n_heads // n_kv_heads
        half_d = head_dim // 2
        scale = 1.0 / math.sqrt(head_dim)
        inv_freq = [1.0 / (10000.0 ** ((2 * m) / head_dim)) for m in range(half_d)]

        Q_rot = array('d', [0.0] * (T * D))
        K_rot = array('d', [0.0] * (T * KV_D))
        cos_table = array('d', [0.0] * (T * half_d))
        sin_table = array('d', [0.0] * (T * half_d))

        Qd, Kd, Vd = Q.data, K.data, V.data
        for t in range(T):
            pos = start_pos + t
            tD = t * D
            tKV = t * KV_D
            t_half = t * half_d
            for m in range(half_d):
                angle = pos * inv_freq[m]
                c, s = math.cos(angle), math.sin(angle)
                cos_table[t_half + m] = c
                sin_table[t_half + m] = s
                # Rotate all Query heads (n_heads)
                for h in range(n_heads):
                    base_q = tD + h * head_dim + 2 * m
                    q0, q1 = Qd[base_q], Qd[base_q + 1]
                    Q_rot[base_q] = q0 * c - q1 * s
                    Q_rot[base_q + 1] = q0 * s + q1 * c
                # Rotate fewer Key heads (n_kv_heads)
                for kv_h in range(n_kv_heads):
                    base_k = tKV + kv_h * head_dim + 2 * m
                    k0, k1 = Kd[base_k], Kd[base_k + 1]
                    K_rot[base_k] = k0 * c - k1 * s
                    K_rot[base_k + 1] = k0 * s + k1 * c

        out_data = array('d', [0.0] * (T * D))
        probs_store = []

        # Inference Path with Int8 Quantized KV-Cache
        if kv_cache is not None:
            k_int8_buf = kv_cache["k_int8"]
            v_int8_buf = kv_cache["v_int8"]
            k_scales = kv_cache["k_scale"]
            v_scales = kv_cache["v_scale"]

            # Quantize each new token's K_rot and V slice to signed Int8
            for t in range(T):
                tKV = t * KV_D
                max_k = 0.0
                max_v = 0.0
                for i in range(KV_D):
                    ak = abs(K_rot[tKV + i])
                    av = abs(Vd[tKV + i])
                    if ak > max_k:
                        max_k = ak
                    if av > max_v:
                        max_v = av

                sk = max_k / 127.0 if max_k > 1e-12 else 1.0
                sv = max_v / 127.0 if max_v > 1e-12 else 1.0
                inv_sk = 127.0 / max_k if max_k > 1e-12 else 0.0
                inv_sv = 127.0 / max_v if max_v > 1e-12 else 0.0

                k_scales.append(sk)
                v_scales.append(sv)
                for i in range(KV_D):
                    k_int8_buf.append(max(-127, min(127, int(round(K_rot[tKV + i] * inv_sk)))))
                    v_int8_buf.append(max(-127, min(127, int(round(Vd[tKV + i] * inv_sv)))))
                kv_cache["seq_len"] += 1

            # Compute GQA attention directly against Int8 KV-Cache
            for h in range(n_heads):
                kv_h = h // group_size
                h_offset = h * head_dim
                kv_offset = kv_h * head_dim

                for t in range(T):
                    max_k_pos = start_pos + t + 1
                    q_base = t * D + h_offset
                    scores = [0.0] * max_k_pos

                    for k in range(max_k_pos):
                        k_base = k * KV_D + kv_offset
                        dot_qk = 0.0
                        for d in range(head_dim):
                            dot_qk += Q_rot[q_base + d] * k_int8_buf[k_base + d]
                        scores[k] = dot_qk * (k_scales[k] * scale)

                    max_s = max(scores)
                    exps = [math.exp(s - max_s) for s in scores]
                    sum_e = sum(exps)
                    probs = [e / sum_e for e in exps]

                    out_base = t * D + h_offset
                    for k, p in enumerate(probs):
                        v_base = k * KV_D + kv_offset
                        weighted_scale = p * v_scales[k]
                        for d in range(head_dim):
                            out_data[out_base + d] += weighted_scale * v_int8_buf[v_base + d]

            return FlatTensor(T, D, out_data)

        # Training Path (Full Precision GQA with exact grouped backward pass)
        for h in range(n_heads):
            kv_h = h // group_size
            h_offset = h * head_dim
            kv_offset = kv_h * head_dim

            for t in range(T):
                max_k_pos = t + 1
                q_base = t * D + h_offset
                scores = [0.0] * max_k_pos
                for k in range(max_k_pos):
                    k_base = k * KV_D + kv_offset
                    s_val = 0.0
                    for d in range(head_dim):
                        s_val += Q_rot[q_base + d] * K_rot[k_base + d]
                    scores[k] = s_val * scale

                max_s = max(scores)
                exps = [math.exp(s - max_s) for s in scores]
                sum_e = sum(exps)
                probs = [e / sum_e for e in exps]
                probs_store.append(probs)

                out_base = t * D + h_offset
                for k, p in enumerate(probs):
                    v_base = k * KV_D + kv_offset
                    for d in range(head_dim):
                        out_data[out_base + d] += p * Vd[v_base + d]

        out = FlatTensor(T, D, out_data, (Q, K, V))

        def _backward():
            dOut, dV = out.grad, V.grad
            dQ_rot = array('d', [0.0] * (T * D))
            dK_rot = array('d', [0.0] * (T * KV_D))

            for h in range(n_heads):
                kv_h = h // group_size
                h_offset = h * head_dim
                kv_offset = kv_h * head_dim

                for t in range(T):
                    probs = probs_store[h * T + t]
                    max_k_pos = t + 1
                    q_base = t * D + h_offset
                    out_base = q_base

                    dP = [0.0] * max_k_pos
                    for k in range(max_k_pos):
                        pk = probs[k]
                        v_base = k * KV_D + kv_offset
                        dp_k = 0.0
                        for d in range(head_dim):
                            do_d = dOut[out_base + d]
                            dV[v_base + d] += pk * do_d
                            dp_k += do_d * Vd[v_base + d]
                        dP[k] = dp_k

                    dot_p_dp = sum(probs[k] * dP[k] for k in range(max_k_pos))
                    for k in range(max_k_pos):
                        ds = probs[k] * (dP[k] - dot_p_dp) * scale
                        k_base = k * KV_D + kv_offset
                        for d in range(head_dim):
                            dQ_rot[q_base + d] += ds * K_rot[k_base + d]
                            dK_rot[k_base + d] += ds * Q_rot[q_base + d]

            dQ, dK = Q.grad, K.grad
            for t in range(T):
                tD = t * D
                tKV = t * KV_D
                t_half = t * half_d
                for m in range(half_d):
                    c = cos_table[t_half + m]
                    s = sin_table[t_half + m]
                    for h in range(n_heads):
                        base_q = tD + h * head_dim + 2 * m
                        dq0, dq1 = dQ_rot[base_q], dQ_rot[base_q + 1]
                        dQ[base_q] += dq0 * c + dq1 * s
                        dQ[base_q + 1] += -dq0 * s + dq1 * c
                    for kv_h in range(n_kv_heads):
                        base_k = tKV + kv_h * head_dim + 2 * m
                        dk0, dk1 = dK_rot[base_k], dK_rot[base_k + 1]
                        dK[base_k] += dk0 * c + dk1 * s
                        dK[base_k + 1] += -dk0 * s + dk1 * c

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
# 2. PURE-PYTHON OKAPI BM25 RAG RETRIEVER
# =====================================================================
class PurePythonBM25RAG:
    """Zero-dependency Okapi BM25 Retriever with suffix stemming & stopword filtering."""
    STOPWORDS = {
        "what", "is", "the", "a", "an", "how", "do", "does", "are",
        "this", "in", "to", "of", "for", "on", "with", "it", "tell", "me", "about"
    }

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []  # list of {"question": str, "fact": str}

    @classmethod
    def tokenize_terms(cls, text):
        clean = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        terms = []
        for word in clean.split():
            if word in cls.STOPWORDS:
                continue
            # Lightweight suffix stemmer so "models" <-> "model", "saved" <-> "save"
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


# =====================================================================
# 3. DYNAMIC BPE TOKENIZER WITH <|context|>, <|user|>, <|assistant|>
# =====================================================================
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

    def train(self, text, max_merges=16, min_freq=4):
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


# =====================================================================
# 4. GQA TRANSFORMER + INT8 KV-CACHE + DYNAMIC VOCAB SURGERY
# =====================================================================
class ModernMicroGPT:
    def __init__(self, vocab_size, n_embd=24, n_heads=4, n_kv_heads=2, n_layers=2, mlp_hidden=48):
        assert n_embd % n_heads == 0 and (n_embd // n_heads) % 2 == 0, "head_dim must be even for RoPE"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for GQA"
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = n_embd // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
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
            # GQA: wq has n_heads columns (n_embd), while wk and wv have n_kv_heads columns (kv_dim)
            self.weights[p + "wq"] = FlatTensor.randn(n_embd, n_embd, std=0.08)
            self.weights[p + "wk"] = FlatTensor.randn(n_embd, self.kv_dim, std=0.08)
            self.weights[p + "wv"] = FlatTensor.randn(n_embd, self.kv_dim, std=0.08)
            self.weights[p + "wo"] = FlatTensor.randn(n_embd, n_embd, std=res_std)
            self.weights[p + "w_gate"] = FlatTensor.randn(n_embd, mlp_hidden, std=0.08)
            self.weights[p + "w_up"] = FlatTensor.randn(n_embd, mlp_hidden, std=0.08)
            self.weights[p + "w_down"] = FlatTensor.randn(mlp_hidden, n_embd, std=res_std)

    def quantize_all(self):
        for tensor in self.weights.values():
            tensor.quantize_int8()

    def expand_vocab(self, new_tokens_info):
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

            b_val = 0.5 * (b_head.data[pa] + b_head.data[pb]) if (pa is not None and pb is not None) else 0.0
            b_head.data.append(b_val)
            b_head.grad.append(0.0)
            if b_head.m is not None:
                b_head.m.append(0.0)
                b_head.v.append(0.0)
            b_head.cols = V_new

            self.vocab_size = V_new

        tok_emb.quantize_int8()
        lm_head.quantize_int8()
        b_head.quantize_int8()

    def init_kv_cache(self):
        """Initializes an Int8-Quantized GQA KV-Cache per Transformer layer."""
        return [
            {
                "k_int8": array('b'),
                "v_int8": array('b'),
                "k_scale": array('d'),
                "v_scale": array('d'),
                "seq_len": 0,
            }
            for _ in range(self.n_layers)
        ]

    def forward(self, idx_seq, start_pos=0, kv_caches=None, use_int8=False):
        w = self.weights
        x = FlatTensor.embedding(w["tok_emb"], idx_seq)

        for l in range(self.n_layers):
            p = f"blocks.{l}."
            layer_cache = kv_caches[l] if kv_caches is not None else None

            x_norm = x.rmsnorm()
            Q = x_norm.matmul(w[p + "wq"], use_int8=use_int8)
            K = x_norm.matmul(w[p + "wk"], use_int8=use_int8)
            V = x_norm.matmul(w[p + "wv"], use_int8=use_int8)
            attn_out = FlatTensor.rope_gqa(
                Q, K, V, self.n_heads, self.n_kv_heads, start_pos=start_pos, kv_cache=layer_cache
            )
            x = x + attn_out.matmul(w[p + "wo"], use_int8=use_int8)

            x_norm2 = x.rmsnorm()
            gate = x_norm2.matmul(w[p + "w_gate"], use_int8=use_int8)
            up = x_norm2.matmul(w[p + "w_up"], use_int8=use_int8)
            hidden = FlatTensor.swiglu_act(gate, up)
            x = x + hidden.matmul(w[p + "w_down"], use_int8=use_int8)

        logits = x.rmsnorm().matmul(w["lm_head"], use_int8=use_int8) + w["b_head"]
        return logits

    def step_adam(self, lr, refresh_int8=False):
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
            if refresh_int8:
                param.quantize_int8()

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

    def generate(
        self,
        prompt_ids,
        max_new_tokens=24,
        temperature=0.05,
        top_k=8,
        top_p=0.9,
        stop_token_id=None,
        use_kv_cache=True,
        use_int8=True,
    ):
        if not prompt_ids:
            return []
        idx = list(prompt_ids)
        V = self.vocab_size

        if use_kv_cache:
            kv_caches = self.init_kv_cache()
            logits = self.forward(idx, start_pos=0, kv_caches=kv_caches, use_int8=use_int8)
            last_row = logits.data[(logits.rows - 1) * V : logits.rows * V]
            next_id = self.sample_top_k_top_p(last_row, temperature, top_k, top_p)
            idx.append(next_id)
            if stop_token_id is not None and next_id == stop_token_id:
                return idx

            for _ in range(max_new_tokens - 1):
                cur_pos = kv_caches[0]["seq_len"]
                logits = self.forward([idx[-1]], start_pos=cur_pos, kv_caches=kv_caches, use_int8=use_int8)
                last_row = logits.data[0 : V]
                next_id = self.sample_top_k_top_p(last_row, temperature, top_k, top_p)
                idx.append(next_id)
                if stop_token_id is not None and next_id == stop_token_id:
                    break
        else:
            for _ in range(max_new_tokens):
                logits = self.forward(idx, start_pos=0, kv_caches=None, use_int8=use_int8)
                last_row = logits.data[(logits.rows - 1) * V : logits.rows * V]
                next_id = self.sample_top_k_top_p(last_row, temperature, top_k, top_p)
                idx.append(next_id)
                if stop_token_id is not None and next_id == stop_token_id:
                    break

        return idx

    def save_checkpoint(self, filepath, tokenizer, rag_engine, corpus_segments):
        self.quantize_all()
        payload = {
            "config": {
                "vocab_size": self.vocab_size,
                "n_embd": self.n_embd,
                "n_heads": self.n_heads,
                "n_kv_heads": self.n_kv_heads,
                "n_layers": self.n_layers,
                "mlp_hidden": self.mlp_hidden,
                "total_steps": self.total_steps,
                "compression": "int8_gqa_zlib_b64",
            },
            "tokenizer": tokenizer.to_dict(),
            "rag_documents": rag_engine.to_list(),
            "corpus_segments": corpus_segments,
            "weights": {name: tensor.to_int8_compressed_dict() for name, tensor in self.weights.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

    @classmethod
    def load_checkpoint(cls, filepath, target_kv_heads=2):
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg = payload["config"]
        n_kv = cfg.get("n_kv_heads", target_kv_heads)
        model = cls(
            vocab_size=cfg["vocab_size"],
            n_embd=cfg["n_embd"],
            n_heads=cfg["n_heads"],
            n_kv_heads=n_kv,
            n_layers=cfg["n_layers"],
            mlp_hidden=cfg["mlp_hidden"],
        )
        model.total_steps = cfg.get("total_steps", 0)
        upgraded_mha_to_gqa = False

        for name, serialized in payload["weights"].items():
            t = FlatTensor.from_serialized(serialized)
            # If loading an older MHA checkpoint where wk/wv had cols == n_embd, project to GQA kv_dim
            if (name.endswith(".wk") or name.endswith(".wv")) and t.cols > model.kv_dim:
                upgraded_mha_to_gqa = True
                ratio = t.cols // model.kv_dim
                pooled = array('d', [0.0] * (t.rows * model.kv_dim))
                for r in range(t.rows):
                    for c in range(model.kv_dim):
                        s = 0.0
                        for g in range(ratio):
                            s += t.data[r * t.cols + (c * ratio + g)]
                        pooled[r * model.kv_dim + c] = s / ratio
                t = FlatTensor(t.rows, model.kv_dim, pooled)
                t.quantize_int8()
            model.weights[name] = t

        tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
        rag_docs = payload.get("rag_documents", [])
        rag_engine = PurePythonBM25RAG.from_list(rag_docs)
        corpus_segments = payload.get("corpus_segments", [])
        return model, tokenizer, rag_engine, corpus_segments, upgraded_mha_to_gqa


# =====================================================================
# 5. TRAINING HELPERS & RAG-GROUNDED CONTINUAL LEARNING
# =====================================================================
def get_cosine_lr(step, total_steps, max_lr=0.012, min_lr=0.001, warmup_steps=15):
    if step <= warmup_steps:
        return max_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_training_sequences(tokenizer, corpus_segments, block_size=28):
    sequences = []
    for seg in corpus_segments:
        seg_ids = tokenizer.encode(seg)
        if len(seg_ids) >= 2:
            sequences.append((seg_ids[:-1], seg_ids[1:]))

    full_ids = tokenizer.encode("".join(corpus_segments) * 2)
    for i in range(0, max(1, len(full_ids) - block_size), 2):
        chunk = full_ids[i : i + block_size + 1]
        if len(chunk) >= 2:
            sequences.append((chunk[:-1], chunk[1:]))
    return sequences


def continual_learn_rag_turn(model, tokenizer, rag_engine, corpus_segments, question, answer, ckpt_path, block_size=28):
    """
    1. Indexes (question, answer) into the persistent Okapi BM25 RAG retriever.
    2. Formats the RAG-grounded training sequence with <|context|>.
    3. Dynamically expands BPE subword merges & GQA model weights.
    4. Runs online continual learning with replay and saves Int8 checkpoint.
    """
    rag_engine.add_document(question, answer)
    formatted_turn = tokenizer.format_rag_qa(answer, question, answer)
    if formatted_turn not in corpus_segments:
        corpus_segments.append(formatted_turn)

    candidate_corpus = "".join(corpus_segments) + (formatted_turn * 4)
    new_tokens_info = tokenizer.expand_vocab(candidate_corpus, max_new_merges=3, min_freq=4)
    if new_tokens_info:
        model.expand_vocab(new_tokens_info)

    turn_ids = tokenizer.encode(formatted_turn)
    turn_pair = [(turn_ids[:-1], turn_ids[1:])]
    replay_pool = build_training_sequences(tokenizer, corpus_segments, block_size=block_size)

    online_epochs = 14
    total_online_steps = online_epochs * (3 + min(5, len(replay_pool)))
    local_step = 0
    last_loss = 0.0

    for _ in range(online_epochs):
        replay_sample = random.sample(replay_pool, k=min(5, len(replay_pool)))
        batch = (turn_pair * 3) + replay_sample
        random.shuffle(batch)
        epoch_loss = 0.0
        for x_seq, y_seq in batch:
            local_step += 1
            lr = get_cosine_lr(local_step, total_online_steps, max_lr=0.009, min_lr=0.001, warmup_steps=4)
            logits = model.forward(x_seq, use_int8=False)
            loss = FlatTensor.cross_entropy(logits, y_seq)
            loss.backward()
            model.step_adam(lr)
            epoch_loss += loss.data[0]
        last_loss = epoch_loss / max(1, len(batch))

    model.quantize_all()
    model.save_checkpoint(ckpt_path, tokenizer, rag_engine, corpus_segments)
    minted_labels = [f"'{tokenizer.vocab[nid]}' (id={nid})" for nid, _, _ in new_tokens_info]
    return last_loss, minted_labels


# =====================================================================
# 6. MAIN ENTRYPOINT: GQA + INT8 KV-CACHE + BM25 RAG REPL
# =====================================================================
if __name__ == "__main__":
    ckpt_path = "micro_gpt_checkpoint.json"
    block_size = 28

    default_knowledge = [
        ("what is this model?", "deep learning in pure python."),
        ("how do small models learn?", "small models learn patterns fast."),
        ("what architecture is this?", "attention is all you need."),
        ("how does gqa save memory?", "grouped query attention shares key value heads."),
    ]

    if os.path.exists(ckpt_path):
        old_size_kb = os.path.getsize(ckpt_path) / 1024.0
        print(f"Found existing checkpoint '{ckpt_path}' ({old_size_kb:.1f} KB)! Loading...")
        model, tokenizer, rag_engine, corpus_segments, migrated_gqa = ModernMicroGPT.load_checkpoint(ckpt_path)

        added_specials = tokenizer.ensure_special_tokens()
        needs_rag_upgrade = (
            migrated_gqa
            or bool(added_specials)
            or not rag_engine.documents
            or not any(BPETokenizer.CONTEXT_TOKEN in seg for seg in corpus_segments)
        )

        if needs_rag_upgrade:
            print("Upgrading checkpoint to GQA (4 Q : 2 KV heads) + BM25 RAG <|context|> grounding...")
            if added_specials:
                model.expand_vocab(added_specials)
            corpus_segments = []
            for q, a in default_knowledge:
                rag_engine.add_document(q, a)
                corpus_segments.append(tokenizer.format_rag_qa(a, q, a))
            for q, a in default_knowledge:
                continual_learn_rag_turn(model, tokenizer, rag_engine, corpus_segments, q, a, ckpt_path, block_size)

        model.save_checkpoint(ckpt_path, tokenizer, rag_engine, corpus_segments)
        new_size_kb = os.path.getsize(ckpt_path) / 1024.0
        total_params = sum(t.rows * t.cols for t in model.weights.values())
        print(
            f"Loaded {model.n_layers}-Layer GQA ({model.n_heads}Q:{model.n_kv_heads}KV) Int8 GPT | "
            f"Vocab: {model.vocab_size} | Params: {total_params} | "
            f"RAG Docs: {len(rag_engine.documents)} | Size: {new_size_kb:.1f} KB\n"
        )
    else:
        print("No checkpoint found. Training 2-Layer GQA + Int8 KV-Cache + BM25 RAG GPT from scratch...")
        rag_engine = PurePythonBM25RAG()
        tokenizer = BPETokenizer()

        # Pre-build temporary formatted text to initialize BPE vocabulary
        seed_text = "".join(
            f"<|context|> {a} <|user|> {q} <|assistant|> {a} <|endoftext|> "
            for q, a in default_knowledge
        ) * 4
        tokenizer.train(seed_text, max_merges=18, min_freq=4)

        corpus_segments = []
        for q, a in default_knowledge:
            rag_engine.add_document(q, a)
            corpus_segments.append(tokenizer.format_rag_qa(a, q, a))

        model = ModernMicroGPT(
            vocab_size=len(tokenizer.vocab),
            n_embd=24,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            mlp_hidden=48,
        )
        total_params = sum(t.rows * t.cols for t in model.weights.values())
        print(
            f"BPE Vocab: {len(tokenizer.vocab)} tokens | "
            f"GQA Heads: {model.n_heads} Query / {model.n_kv_heads} KV | "
            f"Total Parameters: {total_params} (saved 1,152 KV params via GQA)"
        )

        sequences = build_training_sequences(tokenizer, corpus_segments, block_size=block_size)
        epochs = 32
        total_train_steps = epochs * len(sequences)
        local_step = 0

        for epoch in range(1, epochs + 1):
            random.shuffle(sequences)
            epoch_loss = 0.0
            for x_seq, y_seq in sequences:
                local_step += 1
                lr = get_cosine_lr(local_step, total_train_steps, max_lr=0.012, min_lr=0.0008, warmup_steps=20)
                logits = model.forward(x_seq, use_int8=False)
                loss = FlatTensor.cross_entropy(logits, y_seq)
                loss.backward()
                model.step_adam(lr)
                epoch_loss += loss.data[0]

            if epoch % 8 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:2d}/{epochs} | "
                    f"Avg Loss: {epoch_loss / len(sequences):.4f} | "
                    f"Cosine LR: {lr:.5f}"
                )

        model.save_checkpoint(ckpt_path, tokenizer, rag_engine, corpus_segments)
        size_kb = os.path.getsize(ckpt_path) / 1024.0
        print(f"\nSaved GQA + Int8 + RAG checkpoint to '{ckpt_path}' ({size_kb:.1f} KB)\n")

    # Verify BM25 RAG + Int8 KV-Cache + GQA Inference (including paraphrased queries!)
    print("--- BM25 RAG + GQA + Int8 KV-Cache Verification ---")
    test_queries = [
        "what is this model?",
        "tell me about gqa memory",       # Paraphrased query resolved via BM25 RAG!
        "attention architecture?",        # Paraphrased query resolved via BM25 RAG!
    ]
    t0 = time.perf_counter()
    for user_q in test_queries:
        retrieved = rag_engine.retrieve(user_q, top_k=1)
        top_doc, bm25_score = retrieved[0]
        # Ground prompt using the retrieved context and canonical indexed question
        grounded_q = top_doc["question"] if bm25_score > 0.0 else user_q
        prompt_str = tokenizer.format_rag_prompt(top_doc["fact"], grounded_q)
        p_ids = tokenizer.encode(prompt_str)

        out_ids = model.generate(
            p_ids,
            max_new_tokens=24,
            temperature=0.05,
            top_k=6,
            top_p=0.9,
            stop_token_id=tokenizer.eos_id,
            use_kv_cache=True,
            use_int8=True,
        )
        reply_ids = out_ids[len(p_ids):]
        answer_text = tokenizer.decode(reply_ids, skip_special=True).strip()
        print(
            f"Query: '{user_q}'\n"
            f"  ├─ BM25 Retrieved (score={bm25_score:.2f}): \"{top_doc['fact']}\"\n"
            f"  └─ Assistant (GQA + Int8 KV): {answer_text}"
        )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"(Completed 3 RAG-grounded GQA turns with Int8 KV-Cache in {dt_ms:.1f} ms)\n")

    # Interactive RAG + GQA Continual-Learning REPL
    print("=====================================================================")
    print("Interactive GQA + Int8 KV-Cache + Pure-Python BM25 RAG REPL")
    print("Commands:")
    print("  - Ask any question      : BM25 retrieves top context -> GQA generates answer")
    print("  - Teach Q&A (q :: a)    : Indexes into BM25 RAG + trains GQA model + saves JSON")
    print("  - /rag                  : Inspect all indexed BM25 RAG documents")
    print("  - /stats                : Inspect GQA head ratio, Int8 KV-Cache & file savings")
    print("  - /set k=6 p=0.9 temp=0.1 int8=1 : Configure sampling & Int8 mode")
    print("  - /quit or exit         : Exit the REPL")
    print("=====================================================================")

    top_k, top_p, temperature, use_int8_mode = 6, 0.9, 0.05, True

    while True:
        try:
            user_in = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting REPL. Checkpoint & RAG index are up to date.")
            break

        if not user_in:
            continue
        if user_in.lower() in ("/quit", "exit", "quit"):
            print("Goodbye! All learned GQA weights and RAG documents are saved in", ckpt_path)
            break

        if user_in.startswith("/rag"):
            print(f"--- Pure-Python BM25 Knowledge Base ({len(rag_engine.documents)} docs) ---")
            for idx, doc in enumerate(rag_engine.documents, 1):
                print(f"  {idx}. Q: {doc['question']}  -->  Fact: {doc['fact']}")
            continue

        if user_in.startswith("/stats"):
            total_params = sum(t.rows * t.cols for t in model.weights.values())
            file_kb = os.path.getsize(ckpt_path) / 1024.0 if os.path.exists(ckpt_path) else 0.0
            fp64_mha_kv_bytes = 2 * model.n_embd * 8
            int8_gqa_kv_bytes = 2 * model.kv_dim * 1
            print(
                f"Architecture: {model.n_layers} Layers | GQA: {model.n_heads} Query Heads : {model.n_kv_heads} KV Heads (2:1 sharing)\n"
                f"KV-Cache Vector Footprint: {fp64_mha_kv_bytes} B/tok/layer (FP64 MHA) -> {int8_gqa_kv_bytes} B/tok/layer (Int8 GQA) [16x smaller]\n"
                f"Vocab Size: {model.vocab_size} tokens | Total Params: {total_params} | Checkpoint: {file_kb:.1f} KB"
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
                elif part.startswith("int8="):
                    use_int8_mode = bool(int(part.split("=")[1]))
            print(f"Updated settings -> top_k={top_k}, top_p={top_p}, temp={temperature}, use_int8={use_int8_mode}")
            continue

        # Case A: Teach a new Q&A / RAG fact via "question :: answer"
        if "::" in user_in:
            q_part, a_part = user_in.split("::", 1)
            loss_after, minted = continual_learn_rag_turn(
                model, tokenizer, rag_engine, corpus_segments, q_part, a_part, ckpt_path, block_size
            )
            file_kb = os.path.getsize(ckpt_path) / 1024.0
            mint_msg = f"Minted BPE: {', '.join(minted)} | " if minted else ""
            print(
                f"[{mint_msg}Indexed in BM25 RAG & Trained GQA | Loss: {loss_after:.4f} | "
                f"Saved '{ckpt_path}' ({file_kb:.1f} KB)]"
            )
            user_in = q_part.strip()

        # Case B: Retrieve top context via Okapi BM25 and generate with Int8 GQA + Int8 KV-Cache
        retrieved = rag_engine.retrieve(user_in, top_k=1)
        top_doc, bm25_score = retrieved[0]
        grounded_q = top_doc["question"] if bm25_score > 0.0 else user_in
        prompt_str = tokenizer.format_rag_prompt(top_doc["fact"], grounded_q)
        p_ids = tokenizer.encode(prompt_str)

        out_ids = model.generate(
            p_ids,
            max_new_tokens=24,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token_id=tokenizer.eos_id,
            use_kv_cache=True,
            use_int8=use_int8_mode,
        )
        reply_ids = out_ids[len(p_ids):]
        reply_text = tokenizer.decode(reply_ids, skip_special=True).strip()
        print(f"  [BM25 RAG score={bm25_score:.2f} | <|context|> \"{top_doc['fact']}\"]")
        print(f"Assistant (Int8 GQA) > {reply_text}")
