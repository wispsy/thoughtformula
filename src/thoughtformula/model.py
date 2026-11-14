"""ModernMicroGPT: GQA + Sparse MoE SwiGLU + LoRA + Speculative Decoding."""

from array import array
import json
import math
import os
import random

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
        self.total_steps = 0
        self.lora_steps = 0

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

    def apply_linear_with_lora(self, x, weight_name, use_int8=False):
        W = self.weights[weight_name]
        out = x.matmul(W, use_int8=use_int8)
        if weight_name in self.lora:
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

    def _forward_moe_block(self, x_norm2, layer_idx, use_int8=False):
        p = f"blocks.{layer_idx}."
        w = self.weights
        router_logits = x_norm2.matmul(w[p + "w_router"], use_int8=False)

        if use_int8:
            T, D = x_norm2.rows, x_norm2.cols
            N_E = self.n_experts
            top_k = self.moe_top_k
            R = router_logits.data
            out_data = array('d', [0.0] * (T * D))

            for t in range(T):
                tE = t * N_E
                ranked = sorted(range(N_E), key=lambda e: R[tE + e], reverse=True)[:top_k]
                max_r = R[tE + ranked[0]]
                exps = [math.exp(R[tE + e] - max_r) for e in ranked]
                sum_e = sum(exps)
                probs = [ex / sum_e for ex in exps]

                x_row = FlatTensor(1, D, array('d', x_norm2.data[t * D : (t + 1) * D]))
                tD = t * D
                for m, e_idx in enumerate(ranked):
                    ep = f"{p}experts.{e_idx}."
                    gate = x_row.matmul(w[ep + "w_gate"], use_int8=True)
                    up = x_row.matmul(w[ep + "w_up"], use_int8=True)
                    hidden = FlatTensor.swiglu_act(gate, up)
                    exp_out = hidden.matmul(w[ep + "w_down"], use_int8=True).data
                    prob = probs[m]
                    for d in range(D):
                        out_data[tD + d] += prob * exp_out[d]

            moe_out = FlatTensor(T, D, out_data)
        else:
            expert_outs = []
            for e in range(self.n_experts):
                ep = f"{p}experts.{e}."
                gate = x_norm2.matmul(w[ep + "w_gate"], use_int8=False)
                up = x_norm2.matmul(w[ep + "w_up"], use_int8=False)
                hidden = FlatTensor.swiglu_act(gate, up)
                expert_outs.append(hidden.matmul(w[ep + "w_down"], use_int8=False))
            moe_out = FlatTensor.moe_router_combine(router_logits, expert_outs, top_k=self.moe_top_k)

        A, B = self.lora[p + "moe_adapt"]
        if A.requires_grad or any(v != 0.0 for v in B.data):
            moe_out = moe_out + x_norm2.matmul(A).matmul(B).scale(self.lora_scale)

        return moe_out

    def forward(self, idx_seq, start_pos=0, kv_caches=None, use_int8=False):
        w = self.weights
        x = FlatTensor.embedding(w["tok_emb"], idx_seq)

        for l in range(self.n_layers):
            p = f"blocks.{l}."
            layer_cache = kv_caches[l] if kv_caches is not None else None

            # 1. Pre-Norm GQA with RoPE + LoRA
            x_norm = x.rmsnorm()
            Q = self.apply_linear_with_lora(x_norm, p + "wq", use_int8=use_int8)
            K = x_norm.matmul(w[p + "wk"], use_int8=use_int8)
            V = self.apply_linear_with_lora(x_norm, p + "wv", use_int8=use_int8)
            attn_out = FlatTensor.rope_gqa(
                Q, K, V, self.n_heads, self.n_kv_heads, start_pos=start_pos, kv_cache=layer_cache
            )
            x = x + self.apply_linear_with_lora(attn_out, p + "wo", use_int8=use_int8)

            # 2. Pre-Norm Sparse Mixture-of-Experts (MoE) SwiGLU + LoRA
            x_norm2 = x.rmsnorm()
            moe_out = self._forward_moe_block(x_norm2, l, use_int8=use_int8)
            x = x + moe_out

        logits = self.apply_linear_with_lora(x.rmsnorm(), "lm_head", use_int8=use_int8) + w["b_head"]
        return logits

    def _adam_update_tensor(self, param, lr, b1_corr, b2_corr):
        N_el = param.rows * param.cols
        if param.m is None or len(param.m) != N_el:
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

    def step_adam(self, lr, lora_only=False):
        if lora_only:
            self.lora_steps += 1
            b1_corr = 1.0 - 0.9 ** self.lora_steps
            b2_corr = 1.0 - 0.999 ** self.lora_steps
            for A, B in self.lora.values():
                self._adam_update_tensor(A, lr, b1_corr, b2_corr)
                self._adam_update_tensor(B, lr, b1_corr, b2_corr)
            self._adam_update_tensor(self.weights["tok_emb"], lr * 0.5, b1_corr, b2_corr)
            self._adam_update_tensor(self.weights["b_head"], lr * 0.5, b1_corr, b2_corr)
        else:
            self.total_steps += 1
            b1_corr = 1.0 - 0.9 ** self.total_steps
            b2_corr = 1.0 - 0.999 ** self.total_steps
            for param in self.weights.values():
                if param.requires_grad:
                    self._adam_update_tensor(param, lr, b1_corr, b2_corr)

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

            drafted = ngram_drafter.draft(idx, gamma=cur_gamma, stop_token_id=stop_token_id) if cur_gamma > 0 else []
            K = len(drafted)
            total_drafted += K

            verify_chunk = [idx[-1]] + drafted
            logits = self.forward(verify_chunk, start_pos=base_len, kv_caches=kv_caches, use_int8=use_int8)
            forward_passes += 1

            all_accepted = True
            hit_eos = False

            for i in range(K):
                row = logits.data[i * V : (i + 1) * V]
                target_tok = self.sample_top_k_top_p(row, temperature, top_k, top_p)
                draft_tok = drafted[i]

                if target_tok == draft_tok:
                    idx.append(draft_tok)
                    total_accepted += 1
                    if (stop_token_id is not None and draft_tok == stop_token_id) or (len(idx) - prompt_len) >= max_new_tokens:
                        hit_eos = True
                        break
                else:
                    idx.append(target_tok)
                    self.truncate_kv_cache(kv_caches, base_len + 1 + i)
                    all_accepted = False
                    if stop_token_id is not None and target_tok == stop_token_id:
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
                "lora_steps": self.lora_steps,
                "compression": "int8_gqa_moe_lora_zlib_b64",
            },
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
