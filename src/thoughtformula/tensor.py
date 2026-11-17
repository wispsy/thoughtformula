"""Flat 1D array('d') + Int8 array('b') autograd engine for GQA, MoE, and LoRA."""

from array import array
import base64
import math
import random
import struct
import zlib


class FlatTensor:
    """
    2D Matrix with contiguous float64 training storage (`array('d')`),
    symmetric Int8 storage (`array('b')`), and `requires_grad` control for LoRA freezing.
    """
    __slots__ = (
        'rows', 'cols', 'data', 'grad', '_backward', '_prev',
        'm', 'v', 'q_data', 'q_scale', 'requires_grad', 'aux_loss'
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
        self.aux_loss = 0.0

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

    def scale(self, factor):
        N_el = self.rows * self.cols
        A = self.data
        out_data = array('d', (x * factor for x in A))
        out = FlatTensor(self.rows, self.cols, out_data, (self,))

        def _backward():
            if self.requires_grad:
                dOut, dA = out.grad, self.grad
                for i in range(N_el):
                    dA[i] += dOut[i] * factor

        out._backward = _backward
        return out

    def transpose(self):
        """Returns a transposed FlatTensor [cols, rows]."""
        R, C = self.rows, self.cols
        transposed_data = array('d', [0.0] * (R * C))
        for r in range(R):
            rC = r * C
            for c in range(C):
                transposed_data[c * R + r] = self.data[rC + c]
        t = FlatTensor(C, R, transposed_data, requires_grad=self.requires_grad)
        if self.q_data is not None and len(self.q_data) == R * C:
            q_trans = array('b', [0] * (R * C))
            for r in range(R):
                rC = r * C
                for c in range(C):
                    q_trans[c * R + r] = self.q_data[rC + c]
            t.q_data = q_trans
            t.q_scale = self.q_scale
        return t

    def matmul(self, other, use_int8=False):
        if use_int8:
            return self.matmul_int8(other)

        M, K, N = self.rows, self.cols, other.cols
        A, B = self.data, other.data

        # Transpose B into B_T of shape [N, K] for contiguous C-level math.sumprod slices
        B_T = array('d', [0.0] * (K * N))
        for k in range(K):
            kN = k * N
            for j in range(N):
                B_T[j * K + k] = B[kN + j]

        A_rows = [memoryview(A)[i * K : (i + 1) * K] for i in range(M)]
        B_cols = [memoryview(B_T)[j * K : (j + 1) * K] for j in range(N)]
        out_data = array('d', (math.sumprod(a_row, b_col) for a_row in A_rows for b_col in B_cols))

        out = FlatTensor(M, N, out_data, (self, other))

        def _backward():
            dOut = out.grad
            dA = self.grad
            dB = other.grad
            comp_A = self.requires_grad
            comp_B = other.requires_grad

            dOut_rows = [memoryview(dOut)[i * N : (i + 1) * N] for i in range(M)]

            if comp_A:
                # dA = dOut @ B.T where B is [K, N], so row k of B is col k of B.T
                B_rows = [memoryview(B)[k * N : (k + 1) * N] for k in range(K)]
                for i in range(M):
                    iK = i * K
                    d_row = dOut_rows[i]
                    for k in range(K):
                        dA[iK + k] += math.sumprod(d_row, B_rows[k])

            if comp_B:
                # dB = A.T @ dOut. A.T is [K, M], dOut is [M, N]
                A_T = array('d', [0.0] * (M * K))
                for i in range(M):
                    iK = i * K
                    for k in range(K):
                        A_T[k * M + i] = A[iK + k]
                dOut_T = array('d', [0.0] * (M * N))
                for i in range(M):
                    iN = i * N
                    for j in range(N):
                        dOut_T[j * M + i] = dOut[iN + j]

                A_T_rows = [memoryview(A_T)[k * M : (k + 1) * M] for k in range(K)]
                dOut_T_rows = [memoryview(dOut_T)[j * M : (j + 1) * M] for j in range(N)]

                for k in range(K):
                    kN = k * N
                    a_t_row = A_T_rows[k]
                    for j in range(N):
                        dB[kN + j] += math.sumprod(a_t_row, dOut_T_rows[j])

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

        # Transpose B_q into B_T_q for contiguous C-level math.sumprod
        B_T_q = array('b', [0] * (K * N))
        for k in range(K):
            kN = k * N
            for j in range(N):
                B_T_q[j * K + k] = B_q[kN + j]

        A_rows = [memoryview(A_q)[i * K : (i + 1) * K] for i in range(M)]
        B_cols = [memoryview(B_T_q)[j * K : (j + 1) * K] for j in range(N)]

        out_data = array('d', (math.sumprod(a_row, b_col) * combined_scale for a_row in A_rows for b_col in B_cols))
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
                if up.requires_grad:
                    dU[i] += dout * (g * sig)
                if gate.requires_grad:
                    dG[i] += dout * u * (sig * (1.0 + g * (1.0 - sig)))

        out._backward = _backward
        return out

    @staticmethod
    def sparse_moe_combine(
        router_logits,
        expert_outputs_map,
        expert_token_positions,
        selected_experts,
        selected_probs,
        top_k=2,
        aux_loss_coeff=0.01,
    ):
        """
        Combines sparse expert outputs with Switch Transformer auxiliary load-balancing loss
        to prevent expert collapse.
        """
        T, N_E = router_logits.rows, router_logits.cols
        if not expert_outputs_map:
            return FlatTensor(T, router_logits.cols)
        first_out = next(iter(expert_outputs_map.values()))
        D = first_out.cols
        out_data = array('d', [0.0] * (T * D))

        for t in range(T):
            tD = t * D
            ranked = selected_experts[t]
            probs = selected_probs[t]
            for m, e in enumerate(ranked):
                p = probs[m]
                pos = expert_token_positions[(t, e)]
                e_data = expert_outputs_map[e].data
                posD = pos * D
                for d in range(D):
                    out_data[tD + d] += p * e_data[posD + d]

        # Calculate Switch Transformer Auxiliary Load-Balancing Loss
        P_full = []
        f = [0.0] * N_E
        for t in range(T):
            tE = t * N_E
            row = router_logits.data[tE : tE + N_E]
            max_r = max(row)
            exps = [math.exp(x - max_r) for x in row]
            s = sum(exps)
            P_full.append([e / s for e in exps])
            for e in selected_experts[t]:
                f[e] += 1.0 / T

        P_mean = [sum(P_full[t][e] for t in range(T)) / T for e in range(N_E)]
        aux_loss = aux_loss_coeff * N_E * sum(f[e] * P_mean[e] for e in range(N_E)) if aux_loss_coeff > 0.0 else 0.0

        children = (router_logits,) + tuple(expert_outputs_map.values())
        out = FlatTensor(T, D, out_data, children)
        out.aux_loss = aux_loss

        def _backward():
            dOut = out.grad
            dR = router_logits.grad
            comp_R = router_logits.requires_grad

            for t in range(T):
                tD = t * D
                tE = t * N_E
                ranked = selected_experts[t]
                probs = selected_probs[t]
                c_dots = [0.0] * top_k

                for m, e in enumerate(ranked):
                    p = probs[m]
                    pos = expert_token_positions[(t, e)]
                    exp_t = expert_outputs_map[e]
                    posD = pos * D
                    dot_val = 0.0
                    for d in range(D):
                        dout_d = dOut[tD + d]
                        dot_val += dout_d * exp_t.data[posD + d]
                        if exp_t.requires_grad:
                            exp_t.grad[posD + d] += p * dout_d
                    c_dots[m] = dot_val

                if comp_R:
                    weighted_c = sum(probs[m] * c_dots[m] for m in range(top_k))
                    for m, e in enumerate(ranked):
                        dR[tE + e] += probs[m] * (c_dots[m] - weighted_c)

                    if aux_loss_coeff > 0.0:
                        scale_aux = (aux_loss_coeff * N_E) / T
                        p_row = P_full[t]
                        dot_pf = sum(p_row[j] * f[j] for j in range(N_E))
                        for e in range(N_E):
                            dR[tE + e] += scale_aux * p_row[e] * (f[e] - dot_pf)

        out._backward = _backward
        return out

    @staticmethod
    def moe_router_combine(router_logits, expert_outputs, top_k=2):
        """
        Combines outputs from N_E experts using Top-K Softmax routing per token,
        with exact analytical gradients back to both the winning experts and router logits.
        """
        T, N_E = router_logits.rows, router_logits.cols
        D = expert_outputs[0].cols
        R = router_logits.data
        out_data = array('d', [0.0] * (T * D))

        selected_experts = []
        selected_probs = []

        for t in range(T):
            tE = t * N_E
            ranked = sorted(range(N_E), key=lambda e: R[tE + e], reverse=True)[:top_k]
            max_r = R[tE + ranked[0]]
            exps = [math.exp(R[tE + e] - max_r) for e in ranked]
            sum_e = sum(exps)
            probs = [ex / sum_e for ex in exps]

            selected_experts.append(ranked)
            selected_probs.append(probs)

            tD = t * D
            for m, e_idx in enumerate(ranked):
                p = probs[m]
                E_data = expert_outputs[e_idx].data
                for d in range(D):
                    out_data[tD + d] += p * E_data[tD + d]

        children = (router_logits,) + tuple(expert_outputs)
        out = FlatTensor(T, D, out_data, children)

        def _backward():
            dOut = out.grad
            dR = router_logits.grad
            comp_R = router_logits.requires_grad

            for t in range(T):
                tD = t * D
                tE = t * N_E
                ranked = selected_experts[t]
                probs = selected_probs[t]
                c_dots = [0.0] * top_k

                for m, e_idx in enumerate(ranked):
                    p = probs[m]
                    exp_tensor = expert_outputs[e_idx]
                    E_data = exp_tensor.data
                    E_grad = exp_tensor.grad
                    dot_val = 0.0
                    for d in range(D):
                        dout_d = dOut[tD + d]
                        dot_val += dout_d * E_data[tD + d]
                        if exp_tensor.requires_grad:
                            E_grad[tD + d] += p * dout_d
                    c_dots[m] = dot_val

                if comp_R:
                    weighted_c = sum(probs[m] * c_dots[m] for m in range(top_k))
                    for m, e_idx in enumerate(ranked):
                        dR[tE + e_idx] += probs[m] * (c_dots[m] - weighted_c)

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
    def rope_gqa(Q, K, V, n_heads, n_kv_heads, start_pos=0, kv_cache=None, rope_cache=None, use_tiling=False, block_size=16):
        T, D = Q.rows, Q.cols
        KV_D = K.cols
        head_dim = D // n_heads
        group_size = n_heads // n_kv_heads
        half_d = head_dim // 2
        scale = 1.0 / math.sqrt(head_dim)

        Q_rot = array('d', [0.0] * (T * D))
        K_rot = array('d', [0.0] * (T * KV_D))
        cos_table = array('d', [0.0] * (T * half_d))
        sin_table = array('d', [0.0] * (T * half_d))

        Qd, Kd, Vd = Q.data, K.data, V.data

        if rope_cache is not None and (start_pos + T) * half_d <= len(rope_cache[0]):
            cos_buf, sin_buf = rope_cache
            for t in range(T):
                pos = start_pos + t
                tD = t * D
                tKV = t * KV_D
                t_half = t * half_d
                pos_half = pos * half_d
                for m in range(half_d):
                    c = cos_buf[pos_half + m]
                    s = sin_buf[pos_half + m]
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
        else:
            inv_freq = [1.0 / (10000.0 ** ((2 * m) / head_dim)) for m in range(half_d)]
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

        if kv_cache is not None:
            if getattr(kv_cache, "is_paged", False):
                # vLLM-style paged KV-cache non-contiguous block allocation
                for t in range(T):
                    tKV = t * KV_D
                    kv_cache.append_token(
                        K_rot[tKV : tKV + KV_D],
                        Vd[tKV : tKV + KV_D]
                    )

                page_size = kv_cache.page_size
                for h in range(n_heads):
                    kv_h = h // group_size
                    h_offset = h * head_dim
                    kv_offset = kv_h * head_dim

                    for t in range(T):
                        max_k_pos = start_pos + t + 1
                        q_base = t * D + h_offset
                        out_base = t * D + h_offset
                        m = -float('inf')
                        l = 0.0
                        o = [0.0] * head_dim

                        num_pages = (max_k_pos + page_size - 1) // page_size
                        for p_idx in range(num_pages):
                            block = kv_cache.get_block(p_idx)
                            b_start = p_idx * page_size
                            b_len = min(page_size, max_k_pos - b_start)
                            if b_len <= 0:
                                break

                            scores = [0.0] * b_len
                            for slot in range(b_len):
                                slot_base = slot * KV_D + kv_offset
                                dot_qk = 0.0
                                for d in range(head_dim):
                                    dot_qk += Q_rot[q_base + d] * block.k_int8[slot_base + d]
                                scores[slot] = dot_qk * (block.k_scale[slot] * scale)

                            m_b = max(scores)
                            m_new = max(m, m_b)
                            alpha = math.exp(m - m_new) if m != -float('inf') else 0.0
                            l *= alpha
                            for d in range(head_dim):
                                o[d] *= alpha
                            for slot in range(b_len):
                                w = math.exp(scores[slot] - m_new)
                                l += w
                                slot_base = slot * KV_D + kv_offset
                                weighted_scale = w * block.v_scale[slot]
                                for d in range(head_dim):
                                    o[d] += weighted_scale * block.v_int8[slot_base + d]
                            m = m_new

                        inv_l = 1.0 / l if l > 0 else 0.0
                        for d in range(head_dim):
                            out_data[out_base + d] = o[d] * inv_l

                return FlatTensor(T, D, out_data)

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

            if use_tiling:
                # Online softmax tiling for KV-cache inference (Dao et al. FlashAttention streaming)
                for h in range(n_heads):
                    kv_h = h // group_size
                    h_offset = h * head_dim
                    kv_offset = kv_h * head_dim

                    for t in range(T):
                        max_k_pos = start_pos + t + 1
                        q_base = t * D + h_offset
                        out_base = t * D + h_offset
                        m = -float('inf')
                        l = 0.0
                        o = [0.0] * head_dim

                        for b_start in range(0, max_k_pos, block_size):
                            b_end = min(b_start + block_size, max_k_pos)
                            b_len = b_end - b_start
                            scores = [0.0] * b_len
                            for idx, k in enumerate(range(b_start, b_end)):
                                k_base = k * KV_D + kv_offset
                                dot_qk = 0.0
                                for d in range(head_dim):
                                    dot_qk += Q_rot[q_base + d] * k_int8_buf[k_base + d]
                                scores[idx] = dot_qk * (k_scales[k] * scale)

                            m_b = max(scores)
                            m_new = max(m, m_b)
                            alpha = math.exp(m - m_new) if m != -float('inf') else 0.0
                            l *= alpha
                            for d in range(head_dim):
                                o[d] *= alpha
                            for idx, k in enumerate(range(b_start, b_end)):
                                w = math.exp(scores[idx] - m_new)
                                l += w
                                v_base = k * KV_D + kv_offset
                                weighted_scale = w * v_scales[k]
                                for d in range(head_dim):
                                    o[d] += weighted_scale * v_int8_buf[v_base + d]
                            m = m_new

                        inv_l = 1.0 / l if l > 0 else 0.0
                        for d in range(head_dim):
                            out_data[out_base + d] = o[d] * inv_l
            else:
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

        if use_tiling:
            # FlashAttention online softmax tiling (Dao et al. 2022) with O(N) memory
            m_stats = array('d', [0.0] * (n_heads * T))
            l_stats = array('d', [0.0] * (n_heads * T))

            for h in range(n_heads):
                kv_h = h // group_size
                h_offset = h * head_dim
                kv_offset = kv_h * head_dim

                for t in range(T):
                    max_k_pos = t + 1
                    q_base = t * D + h_offset
                    out_base = q_base
                    m = -float('inf')
                    l = 0.0
                    o = [0.0] * head_dim

                    for b_start in range(0, max_k_pos, block_size):
                        b_end = min(b_start + block_size, max_k_pos)
                        b_len = b_end - b_start
                        scores = [0.0] * b_len
                        for idx, k in enumerate(range(b_start, b_end)):
                            k_base = k * KV_D + kv_offset
                            s_val = 0.0
                            for d in range(head_dim):
                                s_val += Q_rot[q_base + d] * K_rot[k_base + d]
                            scores[idx] = s_val * scale

                        m_b = max(scores)
                        m_new = max(m, m_b)
                        alpha = math.exp(m - m_new) if m != -float('inf') else 0.0
                        l *= alpha
                        for d in range(head_dim):
                            o[d] *= alpha
                        for idx, k in enumerate(range(b_start, b_end)):
                            w = math.exp(scores[idx] - m_new)
                            l += w
                            v_base = k * KV_D + kv_offset
                            for d in range(head_dim):
                                o[d] += w * Vd[v_base + d]
                        m = m_new

                    inv_l = 1.0 / l if l > 0 else 0.0
                    for d in range(head_dim):
                        out_data[out_base + d] = o[d] * inv_l
                    ht_idx = h * T + t
                    m_stats[ht_idx] = m
                    l_stats[ht_idx] = l

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
                        max_k_pos = t + 1
                        q_base = t * D + h_offset
                        out_base = q_base
                        ht_idx = h * T + t
                        m = m_stats[ht_idx]
                        l = l_stats[ht_idx]

                        # D_t = sum_d (dOut[t, d] * out[t, d]) eliminates full matrix backward
                        D_t = 0.0
                        for d in range(head_dim):
                            D_t += dOut[out_base + d] * out_data[out_base + d]

                        for b_start in range(0, max_k_pos, block_size):
                            b_end = min(b_start + block_size, max_k_pos)
                            for k in range(b_start, b_end):
                                k_base = k * KV_D + kv_offset
                                s_val = 0.0
                                for d in range(head_dim):
                                    s_val += Q_rot[q_base + d] * K_rot[k_base + d]
                                s_val *= scale
                                p_k = math.exp(s_val - m) / l
                                dp_k = 0.0
                                for d in range(head_dim):
                                    dp_k += dOut[out_base + d] * Vd[k_base + d]
                                ds = p_k * (dp_k - D_t) * scale
                                for d in range(head_dim):
                                    dV[k_base + d] += p_k * dOut[out_base + d]
                                    dQ_rot[q_base + d] += ds * K_rot[k_base + d]
                                    dK_rot[k_base + d] += ds * Q_rot[q_base + d]

                dQ, dK = Q.grad, K.grad
                for t in range(T):
                    tD = t * D
                    tKV = t * KV_D
                    t_half = t * half_d
                    for m_idx in range(half_d):
                        c = cos_table[t_half + m_idx]
                        s = sin_table[t_half + m_idx]
                        for h in range(n_heads):
                            base_q = tD + h * head_dim + 2 * m_idx
                            dq0, dq1 = dQ_rot[base_q], dQ_rot[base_q + 1]
                            dQ[base_q] += dq0 * c + dq1 * s
                            dQ[base_q + 1] += -dq0 * s + dq1 * c
                        for kv_h in range(n_kv_heads):
                            base_k = tKV + kv_h * head_dim + 2 * m_idx
                            dk0, dk1 = dK_rot[base_k], dK_rot[base_k + 1]
                            dK[base_k] += dk0 * c + dk1 * s
                            dK[base_k + 1] += -dk0 * s + dk1 * c

            out._backward = _backward
            return out

        probs_store = []
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
