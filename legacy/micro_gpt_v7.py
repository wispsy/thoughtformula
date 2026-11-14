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
# 1. FLAT 1D array('d') + INT8 array('b') AUTOGRAD ENGINE (GQA + MoE + LoRA)
# =====================================================================
class FlatTensor:
    """
    2D Matrix with contiguous float64 storage (`array('d')`), symmetric Int8 storage (`array('b')`),
    and a `requires_grad` flag so base weights can be frozen during LoRA continual learning.
    """
    __slots__ = (
        'rows', 'cols', 'data', 'grad', '_backward', '_prev',
        'm', 'v', 'q_data', 'q_scale', 'requires_grad'
    )

    def __init__(self, rows, cols, data=None, _children=(), requires_grad=True):
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
        self.requires_grad = requires_grad

    @staticmethod
    def randn(rows, cols, std=0.08, requires_grad=True):
        size = rows * cols
        t = FlatTensor(
            rows, cols,
            array('d', (random.gauss(0.0, std) for _ in range(size))),
            requires_grad=requires_grad,
        )
        t.quantize_int8()
        return t

    @staticmethod
    def zeros(rows, cols, requires_grad=True):
        t = FlatTensor(rows, cols, requires_grad=requires_grad)
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

    def to_fp32_compressed_dict(self):
        """Used for saving high-precision low-rank LoRA adapter matrices."""
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

    def scale(self, factor):
        T, D = self.rows, self.cols
        N_el = T * D
        A = self.data
        out_data = array('d', (A[i] * factor for i in range(N_el)))
        out = FlatTensor(T, D, out_data, (self,))

        def _backward():
            if not self.requires_grad:
                return
            dOut, dA = out.grad, self.grad
            for i in range(N_el):
                dA[i] += dOut[i] * factor

        out._backward = _backward
        return out

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
            dOut = out.grad
            if self.requires_grad:
                dA = self.grad
                for i in range(N_el):
                    dA[i] += dOut[i]
            if other.requires_grad:
                dB = other.grad
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
        if use_int8 and not self.requires_grad and not other.requires_grad:
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
            dOut = out.grad
            A_req = self.requires_grad
            B_req = other.requires_grad
            dA = self.grad
            dB = other.grad

            if A_req and B_req:
                for i in range(M):
                    iK, iN = i * K, i * N
                    for k in range(K):
                        a_ik = A[iK + k]
                        kN = k * N
                        s = 0.0
                        for j in range(N):
                            dout_ij = dOut[iN + j]
                            s += dout_ij * B[kN + j]
                            dB[kN + j] += a_ik * dout_ij
                        dA[iK + k] += s
            elif A_req:
                # Fast path when weight matrix `other` is frozen (e.g., LoRA continual learning!)
                for i in range(M):
                    iK, iN = i * K, i * N
                    for k in range(K):
                        kN = k * N
                        s = 0.0
                        for j in range(N):
                            s += dOut[iN + j] * B[kN + j]
                        dA[iK + k] += s
            elif B_req:
                for i in range(M):
                    iK, iN = i * K, i * N
                    for k in range(K):
                        a_ik = A[iK + k]
                        if a_ik == 0.0:
                            continue
                        kN = k * N
                        for j in range(N):
                            dB[kN + j] += a_ik * dOut[iN + j]

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
            return FlatTensor(M, N, requires_grad=False)

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
        return FlatTensor(M, N, out_data, requires_grad=False)

    @staticmethod
    def linear_lora(x, W, lora_A=None, lora_B=None, lora_scale=2.0, use_int8=False):
        """
        Computes y = x @ W + lora_scale * ((x @ A) @ B).
        During inference (use_int8=True), x @ W uses Int8 matmul while LoRA uses rank-r fast matmul.
        """
        if use_int8 and not W.requires_grad:
            base_out = x.matmul_int8(W)
        else:
            base_out = x.matmul(W, use_int8=False)

        if lora_A is not None and lora_B is not None:
            # Quick check if lora_B is completely zero (e.g., before any LoRA training or after merge)
            has_nonzero_b = any(val != 0.0 for val in lora_B.data)
            if has_nonzero_b or lora_A.requires_grad or lora_B.requires_grad:
                lora_delta = x.matmul(lora_A).matmul(lora_B).scale(lora_scale)
                return base_out + lora_delta
        return base_out

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
            if not self.requires_grad:
                return
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
    def moe_swiglu(x, w_router, experts, top_k=2, expert_counts=None):
        """
        Sparse Mixture-of-Experts (MoE) SwiGLU Layer with Top-K Softmax Routing.
        For each token t, only the top_k highest-scoring experts execute!
        """
        T, D = x.rows, x.cols
        E = w_router.cols
        H = experts[0]["w_gate"].cols
        X = x.data
        WR = w_router.data

        out_data = array('d', [0.0] * (T * D))
        # Per-token cache for backward pass
        routing_cache = []

        for t in range(T):
            tD = t * D
            xt = X[tD : tD + D]

            # 1. Router logits: xt @ w_router -> [E]
            r_logits = [0.0] * E
            for e in range(E):
                s = 0.0
                for d in range(D):
                    s += xt[d] * WR[d * E + e]
                r_logits[e] = s

            # 2. Top-K expert selection & Softmax gating weights
            ranked = sorted(range(E), key=lambda e: r_logits[e], reverse=True)[:top_k]
            max_r = r_logits[ranked[0]]
            exps = [math.exp(r_logits[e] - max_r) for e in ranked]
            sum_e = sum(exps)
            gates = [val / sum_e for val in exps]

            if expert_counts is not None:
                for e in ranked:
                    expert_counts[e] += 1

            # 3. Execute ONLY the selected Top-K SwiGLU experts for token t
            tok_expert_cache = []
            for idx_k, e in enumerate(ranked):
                g_weight = gates[idx_k]
                WG = experts[e]["w_gate"].data
                WU = experts[e]["w_up"].data
                WD = experts[e]["w_down"].data

                gate_pre = [0.0] * H
                up_pre = [0.0] * H
                sig_vec = [0.0] * H
                act_vec = [0.0] * H

                for h in range(H):
                    sg = 0.0
                    su = 0.0
                    for d in range(D):
                        x_d = xt[d]
                        dH = d * H + h
                        sg += x_d * WG[dH]
                        su += x_d * WU[dH]
                    gate_pre[h] = sg
                    up_pre[h] = su
                    if sg >= 0.0:
                        z = math.exp(-min(sg, 20.0))
                        sig = 1.0 / (1.0 + z)
                    else:
                        z = math.exp(max(sg, -20.0))
                        sig = z / (1.0 + z)
                    sig_vec[h] = sig
                    act_vec[h] = (sg * sig) * su

                exp_out = [0.0] * D
                for h in range(H):
                    a_h = act_vec[h]
                    hD = h * D
                    for d in range(D):
                        exp_out[d] += a_h * WD[hD + d]

                for d in range(D):
                    out_data[tD + d] += g_weight * exp_out[d]

                tok_expert_cache.append((e, g_weight, gate_pre, up_pre, sig_vec, act_vec, exp_out))

            routing_cache.append(tok_expert_cache)

        children = [x, w_router]
        for exp in experts:
            children.extend([exp["w_gate"], exp["w_up"], exp["w_down"]])
        out = FlatTensor(T, D, out_data, tuple(children))

        def _backward():
            dOut = out.grad
            dX = x.grad
            dWR = w_router.grad
            router_req = w_router.requires_grad
            experts_req = experts[0]["w_gate"].requires_grad

            for t in range(T):
                tD = t * D
                xt = X[tD : tD + D]
                dout_t = dOut[tD : tD + D]
                tok_cache = routing_cache[t]

                d_gates = [0.0] * len(tok_cache)

                for idx_k, (e, g_weight, gate_pre, up_pre, sig_vec, act_vec, exp_out) in enumerate(tok_cache):
                    # Gradient w.r.t. router gate weight g_weight
                    dg = 0.0
                    for d in range(D):
                        dg += dout_t[d] * exp_out[d]
                    d_gates[idx_k] = dg

                    WG_t = experts[e]["w_gate"]
                    WU_t = experts[e]["w_up"]
                    WD_t = experts[e]["w_down"]
                    WG, WU, WD = WG_t.data, WU_t.data, WD_t.data
                    dWG, dWU, dWD = WG_t.grad, WU_t.grad, WD_t.grad

                    # 1. Backprop through w_down: act_vec [H] @ WD [H, D]
                    d_act = [0.0] * H
                    for h in range(H):
                        hD = h * D
                        a_h = act_vec[h]
                        s_act = 0.0
                        for d in range(D):
                            scaled_dout = g_weight * dout_t[d]
                            s_act += scaled_dout * WD[hD + d]
                            if experts_req:
                                dWD[hD + d] += a_h * scaled_dout
                        d_act[h] = s_act

                    # 2. Backprop through SwiGLU activation
                    d_gate_pre = [0.0] * H
                    d_up_pre = [0.0] * H
                    for h in range(H):
                        da = d_act[h]
                        g_val = gate_pre[h]
                        u_val = up_pre[h]
                        sig = sig_vec[h]
                        d_up_pre[h] = da * (g_val * sig)
                        d_gate_pre[h] = da * u_val * (sig * (1.0 + g_val * (1.0 - sig)))

                    # 3. Backprop through w_gate and w_up into dX (and dWG/dWU if not frozen)
                    for d in range(D):
                        dH = d * H
                        x_d = xt[d]
                        dx_acc = 0.0
                        for h in range(H):
                            dg_h = d_gate_pre[h]
                            du_h = d_up_pre[h]
                            dx_acc += dg_h * WG[dH + h] + du_h * WU[dH + h]
                            if experts_req:
                                dWG[dH + h] += x_d * dg_h
                                dWU[dH + h] += x_d * du_h
                        dX[tD + d] += dx_acc

                # 4. Backprop through Top-K Softmax into w_router and dX
                dot_g_dg = sum(tok_cache[k][1] * d_gates[k] for k in range(len(tok_cache)))
                for idx_k, (e, g_weight, _, _, _, _, _) in enumerate(tok_cache):
                    dr_e = g_weight * (d_gates[idx_k] - dot_g_dg)
                    for d in range(D):
                        if router_req:
                            dWR[d * E + e] += xt[d] * dr_e
                        dX[tD + d] += dr_e * WR[d * E + e]

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
            if not weight.requires_grad:
                return
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
                for h in range(n_heads):
                    base_q = tD + h * head_dim + 2 * m
                    q0, q1 = Qd[base_q], Qd[base_q + 1]
                    Q_rot[base_q] = q0 * c - q1 * s
                    Q_rot[base_q + 1] = q0 * s + q1 * c
                for kv_h in range(n_kv_heads):
                    base_k = tKV + kv_h * head_dim + 2 * m
                    k0, k1 = Kd[base_k], Kd[base_k + 1]
                    K_rot[base_k] = k0 * c - k1 * s
                    K_rot[base_k + 1] = k0 * s + k1 * c

        out_data = array('d', [0.0] * (T * D))
        probs_store = []

        if kv_cache is not None:
            k_int8_buf = kv_cache["k_int8"]
            v_int8_buf = kv_cache["v_int8"]
            k_scales = kv_cache["k_scale"]
            v_scales = kv_cache["v_scale"]

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

            return FlatTensor(T, D, out_data, requires_grad=False)

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
# 2. PURE-PYTHON N-GRAM DRAFT MODEL FOR SPECULATIVE DECODING
# =====================================================================
class NGramDraftModel:
    """
    Zero-latency backoff N-gram (4-gram -> 3-gram -> 2-gram -> 1-gram) draft model.
    Proposes gamma tokens in O(1) dictionary lookups for Speculative Decoding.
    """
    def __init__(self, max_n=4):
        self.max_n = max_n
        self.tables = {n: {} for n in range(1, max_n)}  # context_tuple -> {next_id: count}

    def fit(self, token_sequences):
        self.tables = {n: {} for n in range(1, self.max_n)}
        for seq in token_sequences:
            L = len(seq)
            for n in range(1, self.max_n):
                table = self.tables[n]
                for i in range(L - n):
                    ctx = tuple(seq[i : i + n])
                    nxt = seq[i + n]
                    bucket = table.get(ctx)
                    if bucket is None:
                        table[ctx] = {nxt: 1}
                    else:
                        bucket[nxt] = bucket.get(nxt, 0) + 1

    def draft_next(self, context_ids):
        for n in range(self.max_n - 1, 0, -1):
            if len(context_ids) >= n:
                ctx = tuple(context_ids[-n:])
                bucket = self.tables[n].get(ctx)
                if bucket:
                    return max(bucket.items(), key=lambda kv: kv[1])[0]
        return None

    def draft_sequence(self, context_ids, gamma=4, stop_token_id=None):
        drafted = []
        sim_ctx = list(context_ids)
        for _ in range(gamma):
            nxt = self.draft_next(sim_ctx)
            if nxt is None:
                break
            drafted.append(nxt)
            sim_ctx.append(nxt)
            if stop_token_id is not None and nxt == stop_token_id:
                break
        return drafted


# =====================================================================
# 3. PURE-PYTHON OKAPI BM25 RAG RETRIEVER
# =====================================================================
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
# 4. DYNAMIC BPE TOKENIZER WITH SPECIAL TOKENS
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


# =====================================================================
# 5. SPARSE MoE + GQA + LoRA + SPECULATIVE DECODING TRANSFORMER
# =====================================================================
class ModernMicroGPT:
    def __init__(
        self,
        vocab_size,
        n_embd=24,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        n_experts=4,
        moe_top_k=2,
        expert_hidden=32,
        lora_rank=6,
        lora_alpha=12.0,
    ):
        assert n_embd % n_heads == 0 and (n_embd // n_heads) % 2 == 0, "head_dim must be even for RoPE"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for GQA"
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = n_embd // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.moe_top_k = moe_top_k
        self.expert_hidden = expert_hidden
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_alpha / max(1, lora_rank)
        self.total_steps = 0
        self.expert_load_counts = [0] * n_experts

        res_std = 0.08 / math.sqrt(2.0 * n_layers)

        # Base Weights (Quantized to Int8 on save / inference)
        self.weights = {
            "tok_emb": FlatTensor.randn(vocab_size, n_embd, std=0.08),
            "lm_head": FlatTensor.randn(n_embd, vocab_size, std=0.08),
            "b_head": FlatTensor.zeros(1, vocab_size),
        }

        for l in range(n_layers):
            p = f"blocks.{l}."
            # Grouped-Query Attention projections
            self.weights[p + "wq"] = FlatTensor.randn(n_embd, n_embd, std=0.08)
            self.weights[p + "wk"] = FlatTensor.randn(n_embd, self.kv_dim, std=0.08)
            self.weights[p + "wv"] = FlatTensor.randn(n_embd, self.kv_dim, std=0.08)
            self.weights[p + "wo"] = FlatTensor.randn(n_embd, n_embd, std=res_std)
            # Sparse Mixture-of-Experts (MoE) Router + E SwiGLU Experts
            self.weights[p + "w_router"] = FlatTensor.randn(n_embd, n_experts, std=0.08)
            for e in range(n_experts):
                ep = f"{p}experts.{e}."
                self.weights[ep + "w_gate"] = FlatTensor.randn(n_embd, expert_hidden, std=0.08)
                self.weights[ep + "w_up"] = FlatTensor.randn(n_embd, expert_hidden, std=0.08)
                self.weights[ep + "w_down"] = FlatTensor.randn(expert_hidden, n_embd, std=res_std)

        # Low-Rank Adaptation (LoRA) Matrices: A ~ N(0, 0.05), B = 0
        self.lora_weights = {}
        self._init_lora_adapters()

    def _add_lora_pair(self, name, in_dim, out_dim):
        r = self.lora_rank
        self.lora_weights[name + ".lora_A"] = FlatTensor.randn(in_dim, r, std=0.05, requires_grad=True)
        self.lora_weights[name + ".lora_B"] = FlatTensor.zeros(r, out_dim, requires_grad=True)

    def _init_lora_adapters(self):
        for l in range(self.n_layers):
            p = f"blocks.{l}."
            self._add_lora_pair(p + "wq", self.n_embd, self.n_embd)
            self._add_lora_pair(p + "wk", self.n_embd, self.kv_dim)
            self._add_lora_pair(p + "wv", self.n_embd, self.kv_dim)
            self._add_lora_pair(p + "wo", self.n_embd, self.n_embd)
            # Shared low-rank adapter across the MoE residual path
            self._add_lora_pair(p + "moe_shared", self.n_embd, self.n_embd)
        self._add_lora_pair("lm_head", self.n_embd, self.vocab_size)

    def set_lora_training_mode(self, lora_only=True):
        """
        If lora_only=True, freezes all base Transformer/MoE weights (requires_grad=False)
        and only enables gradients on LoRA A/B adapters (plus tok_emb/b_head for new tokens).
        """
        for name, param in self.weights.items():
            if lora_only:
                param.requires_grad = name in ("tok_emb", "b_head")
            else:
                param.requires_grad = True
        for l_param in self.lora_weights.values():
            l_param.requires_grad = lora_only

    def merge_lora_into_base(self):
        """
        Folds learned LoRA adapters (scale * A @ B) directly into the base weights,
        re-quantizes base weights to Int8, and resets LoRA B matrices to 0.
        """
        r = self.lora_rank
        scale = self.lora_scale
        merged_count = 0

        direct_targets = ["lm_head"]
        for l in range(self.n_layers):
            p = f"blocks.{l}."
            direct_targets.extend([p + "wq", p + "wk", p + "wv", p + "wo"])

        for target in direct_targets:
            A = self.lora_weights[target + ".lora_A"]
            B = self.lora_weights[target + ".lora_B"]
            if not any(v != 0.0 for v in B.data):
                continue
            W = self.weights[target]
            M, N = W.rows, W.cols
            for i in range(M):
                for j in range(N):
                    delta = 0.0
                    for k in range(r):
                        delta += A.data[i * r + k] * B.data[k * N + j]
                    W.data[i * N + j] += scale * delta
            W.quantize_int8()
            B.data = array('d', [0.0] * (r * N))
            B.grad = array('d', [0.0] * (r * N))
            merged_count += 1

        return merged_count

    def quantize_all(self):
        for tensor in self.weights.values():
            tensor.quantize_int8()

    def expand_vocab(self, new_tokens_info):
        if not new_tokens_info:
            return

        D = self.n_embd
        r = self.lora_rank
        tok_emb = self.weights["tok_emb"]
        lm_head = self.weights["lm_head"]
        b_head = self.weights["b_head"]
        head_lora_B = self.lora_weights["lm_head.lora_B"]

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

            # 2. Expand lm_head: [D, V_old] -> [D, V_new]
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

            # 3. Expand lm_head.lora_B: [r, V_old] -> [r, V_new]
            new_lb_data = array('d', [0.0] * (r * V_new))
            new_lb_m = array('d', [0.0] * (r * V_new)) if head_lora_B.m is not None else None
            new_lb_v = array('d', [0.0] * (r * V_new)) if head_lora_B.v is not None else None
            for k in range(r):
                old_rs = k * V_old
                new_rs = k * V_new
                for v in range(V_old):
                    new_lb_data[new_rs + v] = head_lora_B.data[old_rs + v]
                    if new_lb_m is not None:
                        new_lb_m[new_rs + v] = head_lora_B.m[old_rs + v]
                        new_lb_v[new_rs + v] = head_lora_B.v[old_rs + v]
                if pa is not None and pb is not None:
                    new_lb_data[new_rs + V_old] = 0.5 * (head_lora_B.data[old_rs + pa] + head_lora_B.data[old_rs + pb])
            head_lora_B.cols = V_new
            head_lora_B.data = new_lb_data
            head_lora_B.grad = array('d', [0.0] * (r * V_new))
            head_lora_B.m = new_lb_m
            head_lora_B.v = new_lb_v

            # 4. Expand b_head: [1, V_old] -> [1, V_new]
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

    def rollback_kv_cache(self, kv_caches, target_len):
        """Truncates the Int8 GQA KV-Cache back to `target_len` when speculative draft tokens are rejected."""
        KV_D = self.kv_dim
        for cache in kv_caches:
            if cache["seq_len"] > target_len:
                del cache["k_int8"][target_len * KV_D :]
                del cache["v_int8"][target_len * KV_D :]
                del cache["k_scale"][target_len :]
                del cache["v_scale"][target_len :]
                cache["seq_len"] = target_len

    def forward(self, idx_seq, start_pos=0, kv_caches=None, use_int8=False):
        w = self.weights
        lw = self.lora_weights
        sc = self.lora_scale
        x = FlatTensor.embedding(w["tok_emb"], idx_seq)

        for l in range(self.n_layers):
            p = f"blocks.{l}."
            layer_cache = kv_caches[l] if kv_caches is not None else None

            # 1. Pre-Norm Grouped-Query Attention + LoRA Adapters
            x_norm = x.rmsnorm()
            Q = FlatTensor.linear_lora(x_norm, w[p + "wq"], lw[p + "wq.lora_A"], lw[p + "wq.lora_B"], sc, use_int8)
            K = FlatTensor.linear_lora(x_norm, w[p + "wk"], lw[p + "wk.lora_A"], lw[p + "wk.lora_B"], sc, use_int8)
            V = FlatTensor.linear_lora(x_norm, w[p + "wv"], lw[p + "wv.lora_A"], lw[p + "wv.lora_B"], sc, use_int8)

            attn_out = FlatTensor.rope_gqa(
                Q, K, V, self.n_heads, self.n_kv_heads, start_pos=start_pos, kv_cache=layer_cache
            )
            proj_out = FlatTensor.linear_lora(
                attn_out, w[p + "wo"], lw[p + "wo.lora_A"], lw[p + "wo.lora_B"], sc, use_int8
            )
            x = x + proj_out

            # 2. Pre-Norm Sparse Mixture-of-Experts (MoE) SwiGLU + Shared LoRA Adapter
            x_norm2 = x.rmsnorm()
            experts_list = [
                {
                    "w_gate": w[f"{p}experts.{e}.w_gate"],
                    "w_up": w[f"{p}experts.{e}.w_up"],
                    "w_down": w[f"{p}experts.{e}.w_down"],
                }
                for e in range(self.n_experts)
            ]
            moe_out = FlatTensor.moe_swiglu(
                x_norm2,
                w[p + "w_router"],
                experts_list,
                top_k=self.moe_top_k,
                expert_counts=self.expert_load_counts,
            )

            # Apply LoRA residual path for rapid continual adaptation without modifying frozen MoE experts
            moe_A = lw[p + "moe_shared.lora_A"]
            moe_B = lw[p + "moe_shared.lora_B"]
            if any(v != 0.0 for v in moe_B.data) or moe_A.requires_grad or moe_B.requires_grad:
                moe_out = moe_out + x_norm2.matmul(moe_A).matmul(moe_B).scale(sc)

            x = x + moe_out

        # 3. Final RMSNorm + Output Head (with LoRA)
        logits = FlatTensor.linear_lora(
            x.rmsnorm(), w["lm_head"], lw["lm_head.lora_A"], lw["lm_head.lora_B"], sc, use_int8
        ) + w["b_head"]
        return logits

    def step_adam(self, lr, active_params=None):
        self.total_steps += 1
        b1_corr = 1.0 - 0.9 ** self.total_steps
        b2_corr = 1.0 - 0.999 ** self.total_steps

        if active_params is None:
            active_params = [
                p for p in list(self.weights.values()) + list(self.lora_weights.values())
                if p.requires_grad
            ]

        for param in active_params:
            if not param.requires_grad:
                continue
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
    def sample_top_k_top_p(logits_row, temperature=0.05, top_k=8, top_p=0.9):
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

    def generate_speculative(
        self,
        prompt_ids,
        draft_model,
        max_new_tokens=24,
        gamma=4,
        temperature=0.05,
        top_k=6,
        top_p=0.9,
        stop_token_id=None,
        use_int8=True,
    ):
        """
        Speculative Decoding with an N-Gram Draft Model and Int8 KV-Cache Rollback.
        Returns (output_token_ids, spec_stats_dict).
        """
        if not prompt_ids:
            return [], {"target_passes": 0, "drafted": 0, "accepted": 0, "tokens_gen": 0}

        idx = list(prompt_ids)
        V = self.vocab_size
        kv_caches = self.init_kv_cache()

        target_passes = 0
        total_drafted = 0
        total_accepted = 0
        tokens_generated = 0

        # 1. Prefill prompt [0 .. P-1] in a single pass -> sample first token idx[P]
        logits = self.forward(idx, start_pos=0, kv_caches=kv_caches, use_int8=use_int8)
        target_passes += 1
        last_row = logits.data[(logits.rows - 1) * V : logits.rows * V]
        next_id = self.sample_top_k_top_p(last_row, temperature, top_k, top_p)
        idx.append(next_id)
        tokens_generated += 1

        if stop_token_id is not None and next_id == stop_token_id:
            return idx, {
                "target_passes": target_passes,
                "drafted": 0,
                "accepted": 0,
                "tokens_gen": tokens_generated,
            }

        # 2. Speculative Draft-and-Verify Loop
        # Invariant: kv_caches[0]["seq_len"] == len(idx) - 1, and idx[-1] is the unprocessed anchor token
        while tokens_generated < max_new_tokens:
            rem = max_new_tokens - tokens_generated
            cur_gamma = min(gamma, rem - 1) if rem > 1 else 0
            anchor_token = idx[-1]
            cur_pos = kv_caches[0]["seq_len"]

            draft_tokens = (
                draft_model.draft_sequence(idx, gamma=cur_gamma, stop_token_id=stop_token_id)
                if (draft_model is not None and cur_gamma > 0)
                else []
            )
            K = len(draft_tokens)
            total_drafted += K

            # Verify [anchor_token, d_1, ..., d_K] in ONE batched Transformer forward pass
            verify_batch = [anchor_token] + draft_tokens
            logits = self.forward(verify_batch, start_pos=cur_pos, kv_caches=kv_caches, use_int8=use_int8)
            target_passes += 1

            all_accepted = True
            hit_eos = False

            for i in range(K):
                row_i = logits.data[i * V : (i + 1) * V]
                target_tok = self.sample_top_k_top_p(row_i, temperature, top_k, top_p)
                if target_tok == draft_tokens[i]:
                    idx.append(target_tok)
                    total_accepted += 1
                    tokens_generated += 1
                    if stop_token_id is not None and target_tok == stop_token_id:
                        self.rollback_kv_cache(kv_caches, cur_pos + 1 + i)
                        hit_eos = True
                        break
                else:
                    # Mismatch at draft step i: accept target model's correction & rollback KV-Cache!
                    idx.append(target_tok)
                    tokens_generated += 1
                    self.rollback_kv_cache(kv_caches, cur_pos + 1 + i)
                    all_accepted = False
                    if stop_token_id is not None and target_tok == stop_token_id:
                        hit_eos = True
                    break

            if hit_eos or tokens_generated >= max_new_tokens:
                break

            # If all K draft tokens were accepted, sample the bonus (K+1)-th token from row K for free!
            if all_accepted:
                row_k = logits.data[K * V : (K + 1) * V]
                bonus_tok = self.sample_top_k_top_p(row_k, temperature, top_k, top_p)
                idx.append(bonus_tok)
                tokens_generated += 1
                if stop_token_id is not None and bonus_tok == stop_token_id:
                    break

        return idx, {
            "target_passes": target_passes,
            "drafted": total_drafted,
            "accepted": total_accepted,
            "tokens_gen": tokens_generated,
        }

    def save_checkpoint(self, filepath, tokenizer, rag_engine, corpus_segments):
        self.quantize_all()
        payload = {
            "config": {
                "vocab_size": self.vocab_size,
                "n_embd": self.n_embd,
                "n_heads": self.n_heads,
                "n_kv_heads": self.n_kv_heads,
                "n_layers": self.n_layers,
                "n_experts": self.n_experts,
                "moe_top_k": self.moe_top_k,
                "expert_hidden": self.expert_hidden,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "total_steps": self.total_steps,
                "compression": "int8_moe_gqa_lora_zlib_b64",
            },
            "tokenizer": tokenizer.to_dict(),
            "rag_documents": rag_engine.to_list(),
            "corpus_segments": corpus_segments,
            "weights": {name: tensor.to_int8_compressed_dict() for name, tensor in self.weights.items()},
            "lora_weights": {name: tensor.to_fp32_compressed_dict() for name, tensor in self.lora_weights.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

    @classmethod
    def load_checkpoint(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg = payload["config"]
        is_moe_ckpt = "n_experts" in cfg

        model = cls(
            vocab_size=cfg["vocab_size"],
            n_embd=cfg["n_embd"],
            n_heads=cfg["n_heads"],
            n_kv_heads=cfg.get("n_kv_heads", 2),
            n_layers=cfg["n_layers"],
            n_experts=cfg.get("n_experts", 4),
            moe_top_k=cfg.get("moe_top_k", 2),
            expert_hidden=cfg.get("expert_hidden", 32),
            lora_rank=cfg.get("lora_rank", 6),
            lora_alpha=cfg.get("lora_alpha", 12.0),
        )
        model.total_steps = cfg.get("total_steps", 0)

        for name, serialized in payload["weights"].items():
            if name in model.weights:
                t = FlatTensor.from_serialized(serialized)
                if (name.endswith(".wk") or name.endswith(".wv")) and t.cols > model.kv_dim:
                    ratio = t.cols // model.kv_dim
                    pooled = array('d', [0.0] * (t.rows * model.kv_dim))
                    for r in range(t.rows):
                        for c in range(model.kv_dim):
                            s = sum(t.data[r * t.cols + (c * ratio + g)] for g in range(ratio))
                            pooled[r * model.kv_dim + c] = s / ratio
                    t = FlatTensor(t.rows, model.kv_dim, pooled)
                    t.quantize_int8()
                model.weights[name] = t

        if "lora_weights" in payload:
            for name, serialized in payload["lora_weights"].items():
                if name in model.lora_weights:
                    model.lora_weights[name] = FlatTensor.from_serialized(serialized)

        tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
        rag_docs = payload.get("rag_documents", [])
        rag_engine = PurePythonBM25RAG.from_list(rag_docs)
        corpus_segments = payload.get("corpus_segments", [])
        return model, tokenizer, rag_engine, corpus_segments, (not is_moe_ckpt)


# =====================================================================
# 6. TRAINING, LoRA CONTINUAL LEARNING & N-GRAM SYNC
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


def rebuild_ngram_drafter(tokenizer, corpus_segments):
    drafter = NGramDraftModel(max_n=4)
    encoded_segments = [tokenizer.encode(seg) for seg in corpus_segments]
    drafter.fit(encoded_segments)
    return drafter


def continual_learn_lora_rag_turn(
    model, tokenizer, rag_engine, corpus_segments, question, answer, ckpt_path, block_size=28
):
    """
    Instant Continual Learning via Low-Rank Adaptation (LoRA):
    1. Freezes all 24 MoE expert matrices and base GQA matrices in Int8.
    2. Updates ONLY the low-rank LoRA (A, B) matrices (plus embeddings for any newly minted BPE tokens).
    3. Rebuilds the N-Gram Speculative Draft model and saves the checkpoint.
    """
    t0 = time.perf_counter()
    rag_engine.add_document(question, answer)
    formatted_turn = tokenizer.format_rag_qa(answer, question, answer)
    if formatted_turn not in corpus_segments:
        corpus_segments.append(formatted_turn)

    candidate_corpus = "".join(corpus_segments) + (formatted_turn * 4)
    new_tokens_info = tokenizer.expand_vocab(candidate_corpus, max_new_merges=3, min_freq=4)
    if new_tokens_info:
        model.expand_vocab(new_tokens_info)

    # Freeze base MoE & GQA matrices; train ONLY LoRA adapters (+ tok_emb/b_head)
    model.set_lora_training_mode(lora_only=True)
    active_lora_params = [
        p for p in list(model.lora_weights.values()) + [model.weights["tok_emb"], model.weights["b_head"]]
        if p.requires_grad
    ]

    turn_ids = tokenizer.encode(formatted_turn)
    turn_pair = [(turn_ids[:-1], turn_ids[1:])]
    replay_pool = build_training_sequences(tokenizer, corpus_segments, block_size=block_size)

    online_epochs = 14
    total_online_steps = online_epochs * (3 + min(4, len(replay_pool)))
    local_step = 0
    last_loss = 0.0

    for _ in range(online_epochs):
        replay_sample = random.sample(replay_pool, k=min(4, len(replay_pool)))
        batch = (turn_pair * 3) + replay_sample
        random.shuffle(batch)
        epoch_loss = 0.0
        for x_seq, y_seq in batch:
            local_step += 1
            lr = get_cosine_lr(local_step, total_online_steps, max_lr=0.015, min_lr=0.001, warmup_steps=3)
            logits = model.forward(x_seq, use_int8=False)
            loss = FlatTensor.cross_entropy(logits, y_seq)
            loss.backward()
            model.step_adam(lr, active_params=active_lora_params)
            epoch_loss += loss.data[0]
        last_loss = epoch_loss / max(1, len(batch))

    # Freeze all parameters back for fast Int8 inference
    for p in list(model.weights.values()) + list(model.lora_weights.values()):
        p.requires_grad = False

    model.save_checkpoint(ckpt_path, tokenizer, rag_engine, corpus_segments)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    minted_labels = [f"'{tokenizer.vocab[nid]}' (id={nid})" for nid, _, _ in new_tokens_info]
    return last_loss, minted_labels, dt_ms


# =====================================================================
# 7. MAIN ENTRYPOINT: SPARSE MoE + LoRA + SPECULATIVE DECODING REPL
# =====================================================================
if __name__ == "__main__":
    ckpt_path = "micro_gpt_checkpoint.json"
    block_size = 28

    default_knowledge = [
        ("what is this model?", "deep learning in pure python."),
        ("how do small models learn?", "small models learn patterns fast."),
        ("what architecture is this?", "attention is all you need."),
        ("how does gqa save memory?", "grouped query attention shares key value heads."),
        ("how does moe scale capacity?", "sparse mixture of experts routes tokens to top experts."),
    ]

    if os.path.exists(ckpt_path):
        old_size_kb = os.path.getsize(ckpt_path) / 1024.0
        print(f"Found existing checkpoint '{ckpt_path}' ({old_size_kb:.1f} KB)! Loading...")
        model, tokenizer, rag_engine, corpus_segments, needs_moe_pretrain = ModernMicroGPT.load_checkpoint(ckpt_path)
    else:
        needs_moe_pretrain = True

    if needs_moe_pretrain:
        print("Initializing & pre-training 2-Layer Sparse MoE (4 Experts, Top-2) + GQA GPT...")
        rag_engine = PurePythonBM25RAG()
        tokenizer = BPETokenizer()

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
            n_experts=4,
            moe_top_k=2,
            expert_hidden=32,
            lora_rank=6,
            lora_alpha=12.0,
        )
        # Train full base weights (GQA + MoE experts + Router) during initial pre-training
        model.set_lora_training_mode(lora_only=False)
        for lp in model.lora_weights.values():
            lp.requires_grad = False

        base_params = sum(t.rows * t.cols for t in model.weights.values())
        lora_params = sum(t.rows * t.cols for t in model.lora_weights.values())
        print(
            f"Vocab: {len(tokenizer.vocab)} tokens | "
            f"MoE: {model.n_experts} SwiGLU Experts (Top-{model.moe_top_k} Active) | "
            f"Base Params: {base_params} | LoRA Params: {lora_params} ({100.0 * lora_params / base_params:.1f}%)"
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
                    f"Expert Routing Counts: {model.expert_load_counts}"
                )

        for p in list(model.weights.values()) + list(model.lora_weights.values()):
            p.requires_grad = False
        model.save_checkpoint(ckpt_path, tokenizer, rag_engine, corpus_segments)
        size_kb = os.path.getsize(ckpt_path) / 1024.0
        print(f"\nSaved Sparse MoE + GQA + LoRA Int8 checkpoint to '{ckpt_path}' ({size_kb:.1f} KB)\n")
    else:
        for p in list(model.weights.values()) + list(model.lora_weights.values()):
            p.requires_grad = False
        base_params = sum(t.rows * t.cols for t in model.weights.values())
        lora_params = sum(t.rows * t.cols for t in model.lora_weights.values())
        size_kb = os.path.getsize(ckpt_path) / 1024.0
        print(
            f"Loaded {model.n_layers}-Layer Sparse MoE ({model.n_experts} Experts, Top-{model.moe_top_k}) GPT | "
            f"Base Params: {base_params} | LoRA Params: {lora_params} | File Size: {size_kb:.1f} KB\n"
        )

    # Build the N-Gram Draft Model for Speculative Decoding
    ngram_drafter = rebuild_ngram_drafter(tokenizer, corpus_segments)

    # Benchmark Standard Autoregressive vs. Speculative Decoding (gamma = 4)
    print("--- Speculative Decoding (N-Gram Drafter, gamma=4) vs. Standard Decoding ---")
    test_queries = [
        "what is this model?",
        "tell me about moe capacity",
        "how does gqa save memory?",
    ]

    for user_q in test_queries:
        top_doc, bm25_score = rag_engine.retrieve(user_q, top_k=1)[0]
        grounded_q = top_doc["question"] if bm25_score > 0.0 else user_q
        p_ids = tokenizer.encode(tokenizer.format_rag_prompt(top_doc["fact"], grounded_q))

        t0 = time.perf_counter()
        out_ids, spec_stats = model.generate_speculative(
            p_ids,
            draft_model=ngram_drafter,
            max_new_tokens=24,
            gamma=4,
            temperature=0.05,
            top_k=6,
            top_p=0.9,
            stop_token_id=tokenizer.eos_id,
            use_int8=True,
        )
        dt_spec = (time.perf_counter() - t0) * 1000.0
        reply_text = tokenizer.decode(out_ids[len(p_ids):], skip_special=True).strip()
        toks_per_pass = spec_stats["tokens_gen"] / max(1, spec_stats["target_passes"])
        acc_rate = (100.0 * spec_stats["accepted"] / max(1, spec_stats["drafted"])) if spec_stats["drafted"] else 0.0
        print(
            f"Query: '{user_q}' -> {reply_text}\n"
            f"  └─ Speculative Stats: {spec_stats['tokens_gen']} tokens in ONLY {spec_stats['target_passes']} target passes "
            f"({toks_per_pass:.1f} tok/pass | Draft Accept: {acc_rate:.0f}% | {dt_spec:.1f} ms)"
        )

    # Interactive REPL with LoRA Continual Learning, Speculative Decoding & MoE Inspection
    print("\n=====================================================================")
    print("Interactive Sparse MoE + LoRA + Speculative Decoding REPL")
    print("Commands:")
    print("  - Ask any question      : BM25 RAG + Speculative Decoding (gamma=4)")
    print("  - Teach Q&A (q :: a)    : Instant LoRA Continual Learning (base MoE frozen!)")
    print("  - /lora merge           : Fold learned LoRA A/B weights into Int8 base matrices")
    print("  - /stats                : Inspect MoE expert utilization, LoRA params, & Speculative stats")
    print("  - /rag                  : List indexed BM25 RAG documents")
    print("  - /set gamma=4 temp=0.05: Adjust speculative lookahead gamma & sampling")
    print("  - /quit or exit         : Exit the REPL")
    print("=====================================================================")

    gamma, top_k, top_p, temperature = 4, 6, 0.9, 0.05

    while True:
        try:
            user_in = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting REPL. Checkpoint is up to date.")
            break

        if not user_in:
            continue
        if user_in.lower() in ("/quit", "exit", "quit"):
            print("Goodbye! All MoE, LoRA, and RAG updates are saved in", ckpt_path)
            break

        if user_in.startswith("/lora"):
            merged_n = model.merge_lora_into_base()
            model.save_checkpoint(ckpt_path, tokenizer, rag_engine, corpus_segments)
            print(f"Merged {merged_n} LoRA adapter pairs into Int8 base weights and saved '{ckpt_path}'!")
            continue

        if user_in.startswith("/rag"):
            print(f"--- Pure-Python BM25 Knowledge Base ({len(rag_engine.documents)} docs) ---")
            for idx, doc in enumerate(rag_engine.documents, 1):
                print(f"  {idx}. Q: {doc['question']}  -->  Fact: {doc['fact']}")
            continue

        if user_in.startswith("/stats"):
            base_params = sum(t.rows * t.cols for t in model.weights.values())
            lora_params = sum(t.rows * t.cols for t in model.lora_weights.values())
            lora_l2 = math.sqrt(sum(v * v for t in model.lora_weights.values() for v in t.data))
            file_kb = os.path.getsize(ckpt_path) / 1024.0 if os.path.exists(ckpt_path) else 0.0
            total_routed = max(1, sum(model.expert_load_counts))
            load_pcts = [f"E{i}: {100.0 * c / total_routed:.1f}%" for i, c in enumerate(model.expert_load_counts)]
            print(
                f"MoE Routing ({model.n_experts} SwiGLU Experts, Top-{model.moe_top_k}): {', '.join(load_pcts)}\n"
                f"LoRA Rank r={model.lora_rank} (alpha={model.lora_alpha}): {lora_params} trainable LoRA params vs {base_params} base params "
                f"({100.0 * lora_params / base_params:.1f}%) | LoRA L2 Norm: {lora_l2:.3f}\n"
                f"Speculative Drafter: 4-gram backoff ({len(ngram_drafter.tables[3])} trigrams) | Checkpoint: {file_kb:.1f} KB"
            )
            continue

        if user_in.startswith("/set"):
            for part in user_in.split()[1:]:
                if part.startswith("gamma="):
                    gamma = int(part.split("=")[1])
                elif part.startswith("k="):
                    top_k = int(part.split("=")[1])
                elif part.startswith("p="):
                    top_p = float(part.split("=")[1])
                elif part.startswith("temp="):
                    temperature = float(part.split("=")[1])
            print(f"Updated settings -> gamma={gamma}, top_k={top_k}, top_p={top_p}, temp={temperature}")
            continue

        # Case A: Teach a new Q&A pair via instant LoRA continual learning ("question :: answer")
        if "::" in user_in:
            q_part, a_part = user_in.split("::", 1)
            loss_after, minted, dt_lora = continual_learn_lora_rag_turn(
                model, tokenizer, rag_engine, corpus_segments, q_part, a_part, ckpt_path, block_size
            )
            ngram_drafter = rebuild_ngram_drafter(tokenizer, corpus_segments)
            file_kb = os.path.getsize(ckpt_path) / 1024.0
            mint_msg = f"Minted BPE: {', '.join(minted)} | " if minted else ""
            print(
                f"[{mint_msg}LoRA Continual Update (Base MoE Frozen) in {dt_lora:.0f} ms | "
                f"Loss: {loss_after:.4f} | Saved '{ckpt_path}' ({file_kb:.1f} KB)]"
            )
            user_in = q_part.strip()

        # Case B: Answer via BM25 RAG + Speculative Decoding
        top_doc, bm25_score = rag_engine.retrieve(user_in, top_k=1)[0]
        grounded_q = top_doc["question"] if bm25_score > 0.0 else user_in
        p_ids = tokenizer.encode(tokenizer.format_rag_prompt(top_doc["fact"], grounded_q))

        t0 = time.perf_counter()
        out_ids, spec_stats = model.generate_speculative(
            p_ids,
            draft_model=ngram_drafter,
            max_new_tokens=24,
            gamma=gamma,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token_id=tokenizer.eos_id,
            use_int8=True,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        reply_text = tokenizer.decode(out_ids[len(p_ids):], skip_special=True).strip()
        toks_per_pass = spec_stats["tokens_gen"] / max(1, spec_stats["target_passes"])
        acc_rate = (100.0 * spec_stats["accepted"] / max(1, spec_stats["drafted"])) if spec_stats["drafted"] else 0.0
        print(
            f"  [BM25 score={bm25_score:.2f} | Speculative: {spec_stats['tokens_gen']} toks / "
            f"{spec_stats['target_passes']} passes ({toks_per_pass:.1f} tok/pass, {acc_rate:.0f}% accept) in {dt_ms:.1f} ms]"
        )
        print(f"Assistant (MoE+LoRA) > {reply_text}")
