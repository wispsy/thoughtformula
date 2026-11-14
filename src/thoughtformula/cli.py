"""Command line interface and interactive REPL for thoughtformula."""

import os
from pathlib import Path
import random
import sys
import time

from .tensor import FlatTensor
from .tokenizer import BPETokenizer
from .rag import PurePythonBM25RAG
from .draft import NGramDraftModel
from .model import ModernMicroGPT
from .training import get_cosine_lr, build_training_sequences, continual_learn_lora_turn


DEFAULT_KNOWLEDGE = [
    ("what is this model?", "deep learning in pure python."),
    ("how do small models learn?", "small models learn patterns fast."),
    ("what architecture is this?", "attention is all you need."),
    ("how does gqa save memory?", "grouped query attention shares key value heads."),
    ("how does moe work?", "mixture of experts routes tokens to top swiglu experts."),
]


def resolve_checkpoint_path(default_name="micro_gpt_checkpoint.json") -> str:
    # 1. Environment variable override
    env_path = os.environ.get("THOUGHTFORMULA_CHECKPOINT")
    if env_path and os.path.exists(env_path):
        return env_path

    # 2. Current working directory
    if os.path.exists(default_name):
        return default_name

    # 3. Packaged checkpoint in src/thoughtformula
    pkg_dir = Path(__file__).resolve().parent
    pkg_ckpt = pkg_dir / default_name
    if pkg_ckpt.exists():
        return str(pkg_ckpt)

    # 4. Check Desktop dir if present from original environment
    desktop_dir = os.path.expanduser("~/Desktop/gemini-code-1794514134973")
    if os.path.isdir(desktop_dir):
        dt_path = os.path.join(desktop_dir, default_name)
        if os.path.exists(dt_path):
            return dt_path

    return default_name


def benchmark_speculative(model, tokenizer, rag_engine, ngram_drafter, test_queries=None):
    if test_queries is None:
        test_queries = [
            "what is this model?",
            "tell me about gqa memory",
            "how does moe work?",
        ]

    print("--- Speculative Decoding Benchmark (N-Gram Drafter gamma=4 vs. Standard Decoding) ---")
    t_std_start = time.perf_counter()
    std_passes = 0
    for q in test_queries:
        retrieved = rag_engine.retrieve(q, top_k=1)
        top_doc = retrieved[0][0] if retrieved else {"fact": "", "question": q}
        p_ids = tokenizer.encode(tokenizer.format_rag_prompt(top_doc["fact"], top_doc["question"]))
        out_std = model.generate(
            p_ids, max_new_tokens=24, temperature=0.05, stop_token_id=tokenizer.eos_id, use_kv_cache=True, use_int8=True
        )
        std_passes += len(out_std) - len(p_ids)
    dt_std = (time.perf_counter() - t_std_start) * 1000.0

    t_spec_start = time.perf_counter()
    spec_passes = 0
    total_acc, total_drf = 0, 0
    for q in test_queries:
        retrieved = rag_engine.retrieve(q, top_k=1)
        top_doc = retrieved[0][0] if retrieved else {"fact": "", "question": q}
        p_ids = tokenizer.encode(tokenizer.format_rag_prompt(top_doc["fact"], top_doc["question"]))
        out_spec, acc, drf, passes = model.generate_speculative(
            p_ids,
            ngram_drafter,
            max_new_tokens=24,
            gamma=4,
            temperature=0.05,
            stop_token_id=tokenizer.eos_id,
            use_int8=True,
        )
        spec_passes += passes
        total_acc += acc
        total_drf += drf
        reply = tokenizer.decode(out_spec[len(p_ids):], skip_special=True).strip()
        print(f"Q: '{q}' -> {reply} [accepted {acc}/{drf} drafted tokens in {passes} forward passes]")
    dt_spec = (time.perf_counter() - t_spec_start) * 1000.0
    acc_rate = (100.0 * total_acc / max(1, total_drf))
    print(
        f"Summary: Standard={dt_std:.1f} ms ({std_passes} passes) | "
        f"Speculative={dt_spec:.1f} ms ({spec_passes} passes) | "
        f"Draft Acceptance={acc_rate:.1f}% | Speedup={dt_std / max(1e-6, dt_spec):.2f}x\n"
    )


def run_repl(model, tokenizer, rag_engine, ngram_drafter, corpus_segments, ckpt_path, block_size=28):
    print("=====================================================================")
    print("Interactive GQA + Sparse MoE + LoRA + Speculative Decoding REPL")
    print("Commands:")
    print("  - Ask any question   : BM25 RAG + Speculative Decoding generates reply")
    print("  - Teach (q :: a)     : Instant LoRA fine-tuning (freezes MoE base weights!)")
    print("  - /merge             : Merge learned LoRA adapters (A @ B) into Int8 base weights")
    print("  - /rag               : List all indexed BM25 RAG documents")
    print("  - /stats             : Inspect MoE experts, LoRA rank, & Speculative stats")
    print("  - /quit or exit      : Exit the REPL")
    print("=====================================================================")

    top_k, top_p, temperature, gamma = 6, 0.9, 0.05, 4

    while True:
        try:
            user_in = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\nExiting REPL. Checkpoint is saved at {ckpt_path}")
            break

        if not user_in:
            continue
        if user_in.lower() in ("/quit", "exit", "quit"):
            print(f"Goodbye! All LoRA, MoE, and RAG updates are saved in {ckpt_path}")
            break

        if user_in.startswith("/merge"):
            merged = model.merge_lora_into_base()
            model.save_checkpoint(ckpt_path, tokenizer, rag_engine, corpus_segments)
            print(f"Merged {merged} active LoRA adapter pairs into Int8 base weights and saved '{ckpt_path}'!")
            continue

        if user_in.startswith("/rag"):
            print(f"--- Pure-Python BM25 Knowledge Base ({len(rag_engine.documents)} docs) ---")
            for idx_d, doc in enumerate(rag_engine.documents, 1):
                print(f"  {idx_d}. Q: {doc['question']}  -->  Fact: {doc['fact']}")
            continue

        if user_in.startswith("/stats"):
            base_params = sum(t.rows * t.cols for t in model.weights.values())
            lora_params = sum(A.rows * A.cols + B.rows * B.cols for A, B in model.lora.values())
            file_kb = os.path.getsize(ckpt_path) / 1024.0 if os.path.exists(ckpt_path) else 0.0
            print(
                f"Architecture: {model.n_layers} Layers | GQA ({model.n_heads}Q:{model.n_kv_heads}KV) | "
                f"Sparse MoE: {model.n_experts} SwiGLU Experts (Top-{model.moe_top_k} active per token)\n"
                f"Parameters: {base_params} Base (frozen during REPL chat) | {lora_params} LoRA (rank={model.lora_rank})\n"
                f"Checkpoint: '{ckpt_path}' ({file_kb:.1f} KB) | Full Steps: {model.total_steps} | LoRA Steps: {model.lora_steps}"
            )
            continue

        # Case A: Teach a new Q&A turn using instant LoRA Continual Learning
        if "::" in user_in:
            q_part, a_part = user_in.split("::", 1)
            t_lora = time.perf_counter()
            loss_after, minted = continual_learn_lora_turn(
                model, tokenizer, rag_engine, ngram_drafter, corpus_segments, q_part, a_part, ckpt_path, block_size
            )
            dt_lora = (time.perf_counter() - t_lora) * 1000.0
            file_kb = os.path.getsize(ckpt_path) / 1024.0
            mint_msg = f"Minted BPE: {', '.join(minted)} | " if minted else ""
            print(
                f"[{mint_msg}LoRA Adapted in {dt_lora:.0f} ms | Loss: {loss_after:.4f} | "
                f"Saved '{ckpt_path}' ({file_kb:.1f} KB)]"
            )
            user_in = q_part.strip()

        # Case B: Retrieve context via BM25 RAG and answer via Speculative Decoding
        retrieved = rag_engine.retrieve(user_in, top_k=1)
        if retrieved:
            top_doc, bm25_score = retrieved[0]
            grounded_q = top_doc["question"] if bm25_score > 0.0 else user_in
            fact = top_doc["fact"]
        else:
            fact, grounded_q, bm25_score = "", user_in, 0.0

        prompt_str = tokenizer.format_rag_prompt(fact, grounded_q)
        p_ids = tokenizer.encode(prompt_str)

        t0 = time.perf_counter()
        out_ids, acc, drf, passes = model.generate_speculative(
            p_ids,
            ngram_drafter,
            max_new_tokens=24,
            gamma=gamma,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token_id=tokenizer.eos_id,
            use_int8=True,
        )
        dt_gen = (time.perf_counter() - t0) * 1000.0
        reply_ids = out_ids[len(p_ids):]
        reply_text = tokenizer.decode(reply_ids, skip_special=True).strip()
        print(
            f"  [BM25 score={bm25_score:.2f} | Speculative: {acc}/{drf} drafts accepted in {passes} passes ({dt_gen:.1f} ms)]\n"
            f"Assistant > {reply_text}"
        )


def main():
    ckpt_path = resolve_checkpoint_path("micro_gpt_checkpoint.json")
    block_size = 28

    need_full_train = True
    if os.path.exists(ckpt_path):
        old_size_kb = os.path.getsize(ckpt_path) / 1024.0
        print(f"Found existing checkpoint '{ckpt_path}' ({old_size_kb:.1f} KB)! Loading...")
        model, tokenizer, rag_engine, corpus_segments, migrated_legacy = ModernMicroGPT.load_checkpoint(ckpt_path)
        added_specials = tokenizer.ensure_special_tokens()
        if not migrated_legacy and not added_specials and len(rag_engine.documents) >= len(DEFAULT_KNOWLEDGE):
            need_full_train = False
            base_params = sum(t.rows * t.cols for t in model.weights.values())
            lora_params = sum(A.rows * A.cols + B.rows * B.cols for A, B in model.lora.values())
            print(
                f"Loaded {model.n_layers}-Layer GQA + Sparse MoE ({model.n_experts}E, Top-{model.moe_top_k}) + LoRA GPT!\n"
                f"Vocab: {model.vocab_size} | Base Params: {base_params} | LoRA Params: {lora_params} | "
                f"Size: {old_size_kb:.1f} KB\n"
            )
        else:
            print("Upgrading earlier checkpoint to Sparse MoE (4 Experts, Top-2) + LoRA architecture...")

    if need_full_train:
        print("Training 2-Layer GQA + Sparse MoE SwiGLU (4 Experts, Top-2) + LoRA GPT...")
        rag_engine = PurePythonBM25RAG()
        tokenizer = BPETokenizer()

        seed_text = "".join(
            f"<|context|> {a} <|user|> {q} <|assistant|> {a} <|endoftext|> "
            for q, a in DEFAULT_KNOWLEDGE
        ) * 4
        tokenizer.train(seed_text, max_merges=18, min_freq=4)

        corpus_segments = []
        for q, a in DEFAULT_KNOWLEDGE:
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
            expert_hidden=24,
            lora_rank=4,
            lora_alpha=8.0,
        )
        model.set_training_mode(lora_only=False)

        base_params = sum(t.rows * t.cols for t in model.weights.values())
        lora_params = sum(A.rows * A.cols + B.rows * B.cols for A, B in model.lora.values())
        print(
            f"BPE Vocab: {len(tokenizer.vocab)} tokens | Base MoE Params: {base_params} | "
            f"LoRA Adapter Params: {lora_params} ({100.0 * lora_params / base_params:.1f}% of base)"
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
                model.step_adam(lr, lora_only=False)
                epoch_loss += loss.data[0]

            if epoch % 8 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:2d}/{epochs} | "
                    f"Avg Loss: {epoch_loss / len(sequences):.4f} | "
                    f"Cosine LR: {lr:.5f}"
                )

        model.save_checkpoint(ckpt_path, tokenizer, rag_engine, corpus_segments)
        size_kb = os.path.getsize(ckpt_path) / 1024.0
        print(f"\nSaved Int8 GQA + MoE + LoRA checkpoint to '{ckpt_path}' ({size_kb:.1f} KB)\n")

    # Fit N-Gram Draft Model for Speculative Decoding
    ngram_drafter = NGramDraftModel(max_n=4)
    ngram_drafter.fit([tokenizer.encode(seg) for seg in corpus_segments])

    # If --benchmark or non-interactive flag passed
    if "--benchmark" in sys.argv:
        benchmark_speculative(model, tokenizer, rag_engine, ngram_drafter)
        return

    # Benchmark Standard Autoregressive vs. Speculative Decoding (gamma=4)
    benchmark_speculative(model, tokenizer, rag_engine, ngram_drafter)

    # Interactive REPL
    if sys.stdin.isatty():
        run_repl(model, tokenizer, rag_engine, ngram_drafter, corpus_segments, ckpt_path, block_size=block_size)
    else:
        print("Non-interactive mode detected. Exiting.")


if __name__ == "__main__":
    main()
