"""ModernMicroGPT: GQA + Sparse MoE SwiGLU + LoRA + Speculative Decoding."""

from array import array
import json
import math
import os
import random
import struct
import zipfile

from .tensor import FlatTensor
from .tokenizer import BPETokenizer
from .rag import PurePythonBM25RAG
from .draft import NGramDraftModel


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
        expert_hidden=24,
        lora_rank=4,
        lora_alpha=8.0,
        moe_aux_loss_coeff=0.01,
    ):
        assert n_embd % n_heads == 0 and (n_embd // n_heads) % 2 == 0
        assert n_heads % n_kv_heads == 0
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
        self.moe_aux_loss_coeff = moe_aux_loss_coeff
        self.total_steps = 0
        self.lora_steps = 0
        self.last_aux_loss = 0.0
        self.max_seq_len = 256

        # Precomputed RoPE (cos, sin) lookup tables up to max_seq_len
        half_d = self.head_dim // 2
        inv_freq = [1.0 / (10000.0 ** ((2 * m) / self.head_dim)) for m in range(half_d)]
        self.cos_cache = array('d', [0.0] * (self.max_seq_len * half_d))
        self.sin_cache = array('d', [0.0] * (self.max_seq_len * half_d))
        for p in range(self.max_seq_len):
            p_offset = p * half_d
            for m in range(half_d):
                ang = p * inv_freq[m]
                self.cos_cache[p_offset + m] = math.cos(ang)
                self.sin_cache[p_offset + m] = math.sin(ang)
        self.rope_cache = (self.cos_cache, self.sin_cache)

        res_std = 0.08 / math.sqrt(2.0 * n_layers)

        # Base Weights
        self.weights = {
            "tok_emb": FlatTensor.randn(vocab_size, n_embd, std=0.08),
            "lm_head": FlatTensor.randn(n_embd, vocab_size, std=0.08),
            "b_head": FlatTensor.zeros(1, vocab_size),
        }

        # Low-Rank Adaptation (LoRA) Weights: {target_key: (lora_A, lora_B)}
        self.lora = {}
        self._init_lora_pair("lm_head", n_embd, vocab_size, rank=max(6, lora_rank))

        for l in range(n_layers):
            p = f"blocks.{l}."
            # 1. GQA Projections
            self.weights[p + "wq"] = FlatTensor.randn(n_embd, n_embd, std=0.08)
            self.weights[p + "wk"] = FlatTensor.randn(n_embd, self.kv_dim, std=0.08)
            self.weights[p + "wv"] = FlatTensor.randn(n_embd, self.kv_dim, std=0.08)
            self.weights[p + "wo"] = FlatTensor.randn(n_embd, n_embd, std=res_std)

            self._init_lora_pair(p + "wq", n_embd, n_embd, rank=lora_rank)
            self._init_lora_pair(p + "wv", n_embd, self.kv_dim, rank=lora_rank)
            self._init_lora_pair(p + "wo", n_embd, n_embd, rank=lora_rank)

            # 2. Sparse Mixture-of-Experts (MoE) Router + N_E SwiGLU Experts
            self.weights[p + "w_router"] = FlatTensor.randn(n_embd, n_experts, std=0.08)
            for e in range(n_experts):
                ep = f"{p}experts.{e}."
                self.weights[ep + "w_gate"] = FlatTensor.randn(n_embd, expert_hidden, std=0.08)
                self.weights[ep + "w_up"] = FlatTensor.randn(n_embd, expert_hidden, std=0.08)
                self.weights[ep + "w_down"] = FlatTensor.randn(expert_hidden, n_embd, std=res_std)

            # Shared Post-MoE LoRA Adapter for parameter-efficient MLP adaptation
            self._init_lora_pair(p + "moe_adapt", n_embd, n_embd, rank=lora_rank)

    def _init_lora_pair(self, name, in_dim, out_dim, rank):
        A = FlatTensor.randn(in_dim, rank, std=0.05)
        B = FlatTensor.zeros(rank, out_dim)
        self.lora[name] = (A, B)

    def set_training_mode(self, lora_only=False):
        for name, w in self.weights.items():
            if lora_only:
                w.requires_grad = name in ("tok_emb", "b_head")
            else:
                w.requires_grad = True
        for A, B in self.lora.values():
            A.requires_grad = lora_only
            B.requires_grad = lora_only

    def apply_linear_with_lora(self, x, weight_name, use_int8=False, disable_lora=False):
        W = self.weights[weight_name]
        out = x.matmul(W, use_int8=use_int8)
        if not disable_lora and weight_name in self.lora:
            A, B = self.lora[weight_name]
            if A.requires_grad or any(v != 0.0 for v in B.data):
                delta = x.matmul(A).matmul(B).scale(self.lora_scale)
                out = out + delta
        return out

    def merge_lora_into_base(self):
        merged_count = 0
        for name, (A, B) in self.lora.items():
            if name not in self.weights:
                continue
            if not any(abs(v) > 1e-12 for v in B.data):
                continue
            W = self.weights[name]
            M, R, N = A.rows, A.cols, B.cols
            for i in range(M):
                iR = i * R
                iN = i * N
                for r in range(R):
                    a_ir = A.data[iR + r] * self.lora_scale
                    rN = r * N
                    for j in range(N):
                        W.data[iN + j] += a_ir * B.data[rN + j]
            W.quantize_int8()
            for idx in range(len(B.data)):
                B.data[idx] = 0.0
                B.grad[idx] = 0.0
            merged_count += 1
        return merged_count

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
        lora_A_head, lora_B_head = self.lora["lm_head"]
        R_head = lora_B_head.rows

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

            # 3. Expand lm_head LoRA B matrix: [R_head, V_old] -> [R_head, V_new]
            new_lb_data = array('d', [0.0] * (R_head * V_new))
            new_lb_m = array('d', [0.0] * (R_head * V_new)) if lora_B_head.m is not None else None
            new_lb_v = array('d', [0.0] * (R_head * V_new)) if lora_B_head.v is not None else None
            for r in range(R_head):
                old_rs = r * V_old
                new_rs = r * V_new
                for v in range(V_old):
                    new_lb_data[new_rs + v] = lora_B_head.data[old_rs + v]
                    if new_lb_m is not None:
                        new_lb_m[new_rs + v] = lora_B_head.m[old_rs + v]
                        new_lb_v[new_rs + v] = lora_B_head.v[old_rs + v]
            lora_B_head.cols = V_new
            lora_B_head.data = new_lb_data
            lora_B_head.grad = array('d', [0.0] * (R_head * V_new))
            lora_B_head.m = new_lb_m
            lora_B_head.v = new_lb_v

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

    def truncate_kv_cache(self, kv_caches, target_len):
        kv_d = self.kv_dim
        for cache in kv_caches:
            if cache["seq_len"] > target_len:
                del cache["k_int8"][target_len * kv_d :]
                del cache["v_int8"][target_len * kv_d :]
                del cache["k_scale"][target_len :]
                del cache["v_scale"][target_len :]
                cache["seq_len"] = target_len

    def _forward_moe_block(self, x_norm2, layer_idx, use_int8=False, disable_lora=False):
        p = f"blocks.{layer_idx}."
        w = self.weights
        router_logits = x_norm2.matmul(w[p + "w_router"], use_int8=False)

        T, D = x_norm2.rows, x_norm2.cols
        N_E = self.n_experts
        top_k = self.moe_top_k
        R = router_logits.data

        selected_experts = []
        selected_probs = []
        token_indices_per_expert = {e: [] for e in range(N_E)}
        expert_token_positions = {}

        for t in range(T):
            tE = t * N_E
            ranked = sorted(range(N_E), key=lambda e: R[tE + e], reverse=True)[:top_k]
            max_r = R[tE + ranked[0]]
            exps = [math.exp(R[tE + e] - max_r) for e in ranked]
            sum_e = sum(exps)
            probs = [ex / sum_e for ex in exps]

            selected_experts.append(ranked)
            selected_probs.append(probs)

            for e in ranked:
                pos = len(token_indices_per_expert[e])
                token_indices_per_expert[e].append(t)
                expert_token_positions[(t, e)] = pos

        if use_int8:
            out_data = array('d', [0.0] * (T * D))
            for e, tok_indices in token_indices_per_expert.items():
                if not tok_indices:
                    continue
                ep = f"{p}experts.{e}."
                for t in tok_indices:
                    ranked = selected_experts[t]
                    m = ranked.index(e)
                    prob = selected_probs[t][m]

                    x_row = FlatTensor(1, D, array('d', x_norm2.data[t * D : (t + 1) * D]))
                    gate = x_row.matmul(w[ep + "w_gate"], use_int8=True)
                    up = x_row.matmul(w[ep + "w_up"], use_int8=True)
                    hidden = FlatTensor.swiglu_act(gate, up)
                    exp_out = hidden.matmul(w[ep + "w_down"], use_int8=True).data

                    tD = t * D
                    for d in range(D):
                        out_data[tD + d] += prob * exp_out[d]
            moe_out = FlatTensor(T, D, out_data)
        else:
            # True sparse token dispatch: each expert evaluates ONLY its routed tokens
            expert_outputs_map = {}
            for e, tok_indices in token_indices_per_expert.items():
                if not tok_indices:
                    continue
                ep = f"{p}experts.{e}."
                x_e = FlatTensor.embedding(x_norm2, tok_indices)
                gate_e = x_e.matmul(w[ep + "w_gate"], use_int8=False)
                up_e = x_e.matmul(w[ep + "w_up"], use_int8=False)
                hidden_e = FlatTensor.swiglu_act(gate_e, up_e)
                expert_outputs_map[e] = hidden_e.matmul(w[ep + "w_down"], use_int8=False)

            moe_out = FlatTensor.sparse_moe_combine(
                router_logits,
                expert_outputs_map,
                expert_token_positions,
                selected_experts,
                selected_probs,
                top_k=top_k,
                aux_loss_coeff=self.moe_aux_loss_coeff,
            )

        aux_loss = getattr(moe_out, "aux_loss", 0.0)
        if not disable_lora:
            A, B = self.lora[p + "moe_adapt"]
            if A.requires_grad or any(v != 0.0 for v in B.data):
                moe_out = moe_out + x_norm2.matmul(A).matmul(B).scale(self.lora_scale)
        moe_out.aux_loss = aux_loss

        return moe_out

    def forward(self, idx_seq, start_pos=0, kv_caches=None, use_int8=False, disable_lora=False):
        self.last_aux_loss = 0.0
        w = self.weights
        x = FlatTensor.embedding(w["tok_emb"], idx_seq)

        for l in range(self.n_layers):
            p = f"blocks.{l}."
            layer_cache = kv_caches[l] if kv_caches is not None else None

            # 1. Pre-Norm GQA with RoPE + LoRA
            x_norm = x.rmsnorm()
            Q = self.apply_linear_with_lora(x_norm, p + "wq", use_int8=use_int8, disable_lora=disable_lora)
            K = x_norm.matmul(w[p + "wk"], use_int8=use_int8)
            V = self.apply_linear_with_lora(x_norm, p + "wv", use_int8=use_int8, disable_lora=disable_lora)
            attn_out = FlatTensor.rope_gqa(
                Q, K, V, self.n_heads, self.n_kv_heads, start_pos=start_pos, kv_cache=layer_cache, rope_cache=self.rope_cache
            )
            x = x + self.apply_linear_with_lora(attn_out, p + "wo", use_int8=use_int8, disable_lora=disable_lora)

            # 2. Pre-Norm Sparse Mixture-of-Experts (MoE) SwiGLU + LoRA
            x_norm2 = x.rmsnorm()
            moe_out = self._forward_moe_block(x_norm2, l, use_int8=use_int8, disable_lora=disable_lora)
            self.last_aux_loss += getattr(moe_out, "aux_loss", 0.0)
            x = x + moe_out

        logits = self.apply_linear_with_lora(x.rmsnorm(), "lm_head", use_int8=use_int8, disable_lora=disable_lora) + w["b_head"]
        return logits

    def compute_sequence_log_probs(self, prompt_len, seq_ids, disable_lora=False):
        """
        Computes total log-probability of target tokens seq_ids[prompt_len:] conditioned on seq_ids[:prompt_len].
        Returns (sum_log_prob, logits, probs_per_pos).
        """
        logits = self.forward(seq_ids, disable_lora=disable_lora, use_int8=False)
        T, V = logits.rows, logits.cols
        sum_log_prob = 0.0
        probs_per_pos = []

        for t in range(T):
            row = logits.data[t * V : (t + 1) * V]
            max_l = max(row)
            exps = [math.exp(x - max_l) for x in row]
            sum_e = sum(exps)
            probs = [e / sum_e for e in exps]
            probs_per_pos.append(probs)

            # Prediction at position t predicts token at t + 1
            if t >= prompt_len - 1 and (t + 1) < len(seq_ids):
                target_token = seq_ids[t + 1]
                sum_log_prob += math.log(max(1e-12, probs[target_token]))

        return sum_log_prob, logits, probs_per_pos

    def _adam_update_tensor(self, param, lr, b1_corr, b2_corr, weight_decay=0.0):
        N_el = param.rows * param.cols
        if param.m is None or len(param.m) != N_el:
            param.m = array('d', [0.0] * N_el)
            param.v = array('d', [0.0] * N_el)
        pd, pg, pm, pv = param.data, param.grad, param.m, param.v
        decay_factor = (1.0 - lr * weight_decay) if weight_decay > 0.0 else 1.0

        for i in range(N_el):
            if weight_decay > 0.0:
                pd[i] *= decay_factor
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

    def step_adam(self, lr, lora_only=False, weight_decay=0.01):
        """AdamW step with decoupled weight decay to bound weight scales and preserve Int8 dynamic range."""
        if lora_only:
            self.lora_steps += 1
            b1_corr = 1.0 - 0.9 ** self.lora_steps
            b2_corr = 1.0 - 0.999 ** self.lora_steps
            for A, B in self.lora.values():
                self._adam_update_tensor(A, lr, b1_corr, b2_corr, weight_decay=weight_decay)
                self._adam_update_tensor(B, lr, b1_corr, b2_corr, weight_decay=weight_decay)
            self._adam_update_tensor(self.weights["tok_emb"], lr * 0.5, b1_corr, b2_corr, weight_decay=weight_decay)
            self._adam_update_tensor(self.weights["b_head"], lr * 0.5, b1_corr, b2_corr, weight_decay=0.0)
        else:
            self.total_steps += 1
            b1_corr = 1.0 - 0.9 ** self.total_steps
            b2_corr = 1.0 - 0.999 ** self.total_steps
            for name, param in self.weights.items():
                if param.requires_grad:
                    wd = 0.0 if ("b_head" in name or param.rows == 1) else weight_decay
                    self._adam_update_tensor(param, lr, b1_corr, b2_corr, weight_decay=wd)

    @staticmethod
    def get_probs(logits_row, temperature=0.05, top_k=6, top_p=0.9):
        V = len(logits_row)
        if temperature <= 1e-4:
            probs = [0.0] * V
            best_idx = max(range(V), key=lambda i: logits_row[i])
            probs[best_idx] = 1.0
            return probs

        scaled = [(logits_row[i] / temperature, i) for i in range(V)]
        scaled.sort(key=lambda x: x[0], reverse=True)
        if top_k > 0:
            scaled = scaled[:min(top_k, len(scaled))]

        max_l = scaled[0][0]
        exps = [math.exp(l - max_l) for l, _ in scaled]
        total = sum(exps)
        norm_p = [e / total for e in exps]

        probs = [0.0] * V
        cum_prob = 0.0
        active_sum = 0.0
        for (_, idx), p in zip(scaled, norm_p):
            probs[idx] = p
            active_sum += p
            cum_prob += p
            if cum_prob >= top_p:
                break

        if active_sum > 0:
            for idx in range(V):
                probs[idx] /= active_sum
        return probs

    @staticmethod
    def sample_top_k_top_p(logits_row, temperature=0.05, top_k=6, top_p=0.9):
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
        top_k=6,
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

    def generate_speculative(
        self,
        prompt_ids,
        ngram_drafter,
        max_new_tokens=24,
        gamma=4,
        temperature=0.05,
        top_k=6,
        top_p=0.9,
        stop_token_id=None,
        use_int8=True,
    ):
        """
        Lossless Speculative Decoding (Leviathan et al., 2023):
        1. Drafts up to gamma tokens with proposal probabilities q(x_i).
        2. Computes target distribution p(x) in a single Transformer pass.
        3. Accepts draft token with probability min(1, p(x_i) / q(x_i)).
        4. On rejection, samples replacement from normalized residual max(0, p(x) - q(x))
           and rolls back Int8 KV-cache.
        """
        if not prompt_ids:
            return [], 0, 0, 0

        idx = list(prompt_ids)
        prompt_len = len(idx)
        V = self.vocab_size
        kv_caches = self.init_kv_cache()

        if len(idx) > 1:
            self.forward(idx[:-1], start_pos=0, kv_caches=kv_caches, use_int8=use_int8)

        total_drafted = 0
        total_accepted = 0
        forward_passes = 0

        while (len(idx) - prompt_len) < max_new_tokens:
            base_len = kv_caches[0]["seq_len"]
            remaining = max_new_tokens - (len(idx) - prompt_len)
            cur_gamma = min(gamma, max(1, remaining - 1))

            draft_info = (
                ngram_drafter.draft_with_probs(idx, gamma=cur_gamma, stop_token_id=stop_token_id)
                if cur_gamma > 0 and hasattr(ngram_drafter, "draft_with_probs")
                else [(t, 1.0, {t: 1.0}) for t in (ngram_drafter.draft(idx, gamma=cur_gamma, stop_token_id=stop_token_id) if cur_gamma > 0 else [])]
            )
            K = len(draft_info)
            total_drafted += K

            verify_chunk = [idx[-1]] + [t for t, _, _ in draft_info]
            logits = self.forward(verify_chunk, start_pos=base_len, kv_caches=kv_caches, use_int8=use_int8)
            forward_passes += 1

            all_accepted = True
            hit_eos = False

            for i in range(K):
                draft_tok, q_prob, q_dist = draft_info[i]
                row = logits.data[i * V : (i + 1) * V]
                p_dist = self.get_probs(row, temperature=temperature, top_k=top_k, top_p=top_p)

                if temperature <= 1e-4:
                    target_tok = max(range(V), key=lambda v: p_dist[v])
                    accepted = (target_tok == draft_tok)
                    replacement_tok = target_tok
                else:
                    p_tok = p_dist[draft_tok]
                    accept_prob = min(1.0, p_tok / max(1e-12, q_prob))
                    accepted = (random.random() < accept_prob)
                    if not accepted:
                        res = [max(0.0, p_dist[v] - q_dist.get(v, 0.0)) for v in range(V)]
                        res_sum = sum(res)
                        if res_sum > 1e-12:
                            norm_res = [r / res_sum for r in res]
                            replacement_tok = random.choices(range(V), weights=norm_res)[0]
                        else:
                            replacement_tok = random.choices(range(V), weights=p_dist)[0]

                if accepted:
                    idx.append(draft_tok)
                    total_accepted += 1
                    if (stop_token_id is not None and draft_tok == stop_token_id) or (len(idx) - prompt_len) >= max_new_tokens:
                        hit_eos = True
                        break
                else:
                    idx.append(replacement_tok)
                    self.truncate_kv_cache(kv_caches, base_len + 1 + i)
                    all_accepted = False
                    if stop_token_id is not None and replacement_tok == stop_token_id:
                        hit_eos = True
                    break

            if hit_eos or (len(idx) - prompt_len) >= max_new_tokens:
                break

            if all_accepted:
                bonus_row = logits.data[K * V : (K + 1) * V]
                bonus_tok = self.sample_top_k_top_p(bonus_row, temperature, top_k, top_p)
                idx.append(bonus_tok)
                if stop_token_id is not None and bonus_tok == stop_token_id:
                    break

        return idx, total_accepted, total_drafted, forward_passes

    def save_checkpoint(self, filepath, tokenizer, rag_engine, corpus_segments):
        self.quantize_all()
        cfg = {
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
            "lora_steps": self.lora_steps,
            "compression": "binary_mgpt_zlib" if filepath.endswith(".mgpt") else "int8_gqa_moe_lora_zlib_b64",
        }

        if filepath.endswith(".mgpt"):
            # Binary .mgpt container using standard zipfile (zero Base64 overhead)
            manifest = {
                "config": cfg,
                "tokenizer": tokenizer.to_dict(),
                "rag_documents": rag_engine.to_list(),
                "corpus_segments": corpus_segments,
                "weights_manifest": {
                    name: {"rows": t.rows, "cols": t.cols, "scale": t.q_scale}
                    for name, t in self.weights.items()
                },
                "lora_manifest": {
                    name: {
                        "A": {"rows": A.rows, "cols": A.cols, "scale": A.q_scale},
                        "B": {"rows": B.rows, "cols": B.cols, "scale": B.q_scale},
                    }
                    for name, (A, B) in self.lora.items()
                },
            }
            with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
                for name, t in self.weights.items():
                    n = t.rows * t.cols
                    zf.writestr(f"weights/{name}.bin", struct.pack(f"<{n}b", *t.q_data))
                for name, (A, B) in self.lora.items():
                    zf.writestr(f"lora/{name}_A.bin", struct.pack(f"<{A.rows * A.cols}b", *A.q_data))
                    zf.writestr(f"lora/{name}_B.bin", struct.pack(f"<{B.rows * B.cols}b", *B.q_data))
            return

        # Legacy JSON + Base64 format
        payload = {
            "config": cfg,
            "tokenizer": tokenizer.to_dict(),
            "rag_documents": rag_engine.to_list(),
            "corpus_segments": corpus_segments,
            "weights": {name: tensor.to_int8_compressed_dict() for name, tensor in self.weights.items()},
            "lora": {
                name: {
                    "A": A.to_int8_compressed_dict(),
                    "B": B.to_int8_compressed_dict(),
                }
                for name, (A, B) in self.lora.items()
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

    @classmethod
    def load_checkpoint(cls, filepath):
        if zipfile.is_zipfile(filepath):
            # Load binary .mgpt container
            with zipfile.ZipFile(filepath, "r") as zf:
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                cfg = manifest["config"]
                model = cls(
                    vocab_size=cfg["vocab_size"],
                    n_embd=cfg.get("n_embd", 24),
                    n_heads=cfg.get("n_heads", 4),
                    n_kv_heads=cfg.get("n_kv_heads", 2),
                    n_layers=cfg.get("n_layers", 2),
                    n_experts=cfg.get("n_experts", 4),
                    moe_top_k=cfg.get("moe_top_k", 2),
                    expert_hidden=cfg.get("expert_hidden", 24),
                    lora_rank=cfg.get("lora_rank", 4),
                    lora_alpha=cfg.get("lora_alpha", 8.0),
                )
                model.total_steps = cfg.get("total_steps", 0)
                model.lora_steps = cfg.get("lora_steps", 0)

                for name, meta in manifest.get("weights_manifest", {}).items():
                    rows, cols, scale = meta["rows"], meta["cols"], float(meta["scale"])
                    n = rows * cols
                    raw = zf.read(f"weights/{name}.bin")
                    q_ints = struct.unpack(f"<{n}b", raw)
                    t = FlatTensor(rows, cols, array('d', (q * scale for q in q_ints)))
                    t.q_data = array('b', q_ints)
                    t.q_scale = scale
                    model.weights[name] = t

                for name, pair_meta in manifest.get("lora_manifest", {}).items():
                    meta_a, meta_b = pair_meta["A"], pair_meta["B"]
                    ra, ca, sa = meta_a["rows"], meta_a["cols"], float(meta_a["scale"])
                    rb, cb, sb = meta_b["rows"], meta_b["cols"], float(meta_b["scale"])
                    na, nb = ra * ca, rb * cb

                    raw_a = zf.read(f"lora/{name}_A.bin")
                    q_a = struct.unpack(f"<{na}b", raw_a)
                    A = FlatTensor(ra, ca, array('d', (q * sa for q in q_a)))
                    A.q_data = array('b', q_a)
                    A.q_scale = sa

                    raw_b = zf.read(f"lora/{name}_B.bin")
                    q_b = struct.unpack(f"<{nb}b", raw_b)
                    B = FlatTensor(rb, cb, array('d', (q * sb for q in q_b)))
                    B.q_data = array('b', q_b)
                    B.q_scale = sb

                    model.lora[name] = (A, B)

                tokenizer = BPETokenizer.from_dict(manifest["tokenizer"])
                rag_engine = PurePythonBM25RAG.from_list(manifest.get("rag_documents", []))
                corpus_segments = manifest.get("corpus_segments", [])
                return model, tokenizer, rag_engine, corpus_segments, False

        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg = payload["config"]
        model = cls(
            vocab_size=cfg["vocab_size"],
            n_embd=cfg.get("n_embd", 24),
            n_heads=cfg.get("n_heads", 4),
            n_kv_heads=cfg.get("n_kv_heads", 2),
            n_layers=cfg.get("n_layers", 2),
            n_experts=cfg.get("n_experts", 4),
            moe_top_k=cfg.get("moe_top_k", 2),
            expert_hidden=cfg.get("expert_hidden", 24),
            lora_rank=cfg.get("lora_rank", 4),
            lora_alpha=cfg.get("lora_alpha", 8.0),
        )
        model.total_steps = cfg.get("total_steps", 0)
        model.lora_steps = cfg.get("lora_steps", 0)

        saved_weights = payload.get("weights", {})
        migrated_legacy = False

        for name, serialized in saved_weights.items():
            t = FlatTensor.from_serialized(serialized)
            if name in model.weights:
                if t.rows == model.weights[name].rows and t.cols == model.weights[name].cols:
                    model.weights[name] = t
                else:
                    migrated_legacy = True
            else:
                migrated_legacy = True

        saved_lora = payload.get("lora", {})
        for name, pair in saved_lora.items():
            if name in model.lora:
                A = FlatTensor.from_serialized(pair["A"])
                B = FlatTensor.from_serialized(pair["B"])
                if A.rows == model.lora[name][0].rows and B.cols == model.lora[name][1].cols:
                    model.lora[name] = (A, B)

        tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
        rag_engine = PurePythonBM25RAG.from_list(payload.get("rag_documents", []))
        corpus_segments = payload.get("corpus_segments", [])
        return model, tokenizer, rag_engine, corpus_segments, migrated_legacy
