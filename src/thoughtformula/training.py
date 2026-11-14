"""Training routines, learning rate scheduling, and online LoRA adaptation."""

import math
import random
from .tensor import FlatTensor


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


def continual_learn_lora_turn(
    model,
    tokenizer,
    rag_engine,
    ngram_drafter,
    corpus_segments,
    question,
    answer,
    ckpt_path,
    block_size=28,
):
    """
    Parameter-Efficient Fine-Tuning (LoRA) in Pure Python:
    1. Freezes >85% of heavy GQA + MoE base weights (`model.set_training_mode(lora_only=True)`).
    2. Expands BPE vocabulary if new subwords appear.
    3. Trains ONLY the low-rank LoRA adapters (A, B) + embeddings in milliseconds.
    4. Refits the N-Gram speculative draft model and saves the checkpoint.
    """
    rag_engine.add_document(question, answer)
    formatted_turn = tokenizer.format_rag_qa(answer, question, answer)
    if formatted_turn not in corpus_segments:
        corpus_segments.append(formatted_turn)

    candidate_corpus = "".join(corpus_segments) + (formatted_turn * 4)
    new_tokens_info = tokenizer.expand_vocab(candidate_corpus, max_new_merges=3, min_freq=4)
    if new_tokens_info:
        model.expand_vocab(new_tokens_info)

    # Freeze base Transformer & MoE weights -> enable gradients ONLY on LoRA adapters
    model.set_training_mode(lora_only=True)

    turn_ids = tokenizer.encode(formatted_turn)
    turn_pair = [(turn_ids[:-1], turn_ids[1:])]
    replay_pool = build_training_sequences(tokenizer, corpus_segments, block_size=block_size)

    online_epochs = 12
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
            lr = get_cosine_lr(local_step, total_online_steps, max_lr=0.015, min_lr=0.002, warmup_steps=3)
            logits = model.forward(x_seq, use_int8=False)
            loss = FlatTensor.cross_entropy(logits, y_seq)
            loss.backward()
            model.step_adam(lr, lora_only=True)
            epoch_loss += loss.data[0]
        last_loss = epoch_loss / max(1, len(batch))

    # Update N-Gram Speculative Drafter with the newly learned sequence
    ngram_drafter.fit([tokenizer.encode(seg) for seg in corpus_segments])

    if ckpt_path:
        model.save_checkpoint(ckpt_path, tokenizer, rag_engine, corpus_segments)
    minted_labels = [f"'{tokenizer.vocab[nid]}' (id={nid})" for nid, _, _ in new_tokens_info]
    return last_loss, minted_labels


def train_dpo_turn(
    model,
    tokenizer,
    question,
    chosen_answer,
    rejected_answer,
    beta=0.1,
    lr=0.01,
    epochs=10,
    weight_decay=0.01,
):
    """
    Zero-Memory Direct Preference Optimization (DPO) via LoRA:
    Uses frozen base weights (disable_lora=True) as reference model π_ref with 0 extra RAM,
    optimizing active LoRA adapters to maximize preference margin between chosen and rejected answers.
    """
    prompt = tokenizer.format_rag_prompt(chosen_answer, question)
    p_ids = tokenizer.encode(prompt)
    prompt_len = len(p_ids)

    w_text = prompt + chosen_answer.strip() + " " + tokenizer.EOS_TOKEN + " "
    l_text = prompt + rejected_answer.strip() + " " + tokenizer.EOS_TOKEN + " "

    s_w_ids = tokenizer.encode(w_text)
    s_l_ids = tokenizer.encode(l_text)

    model.set_training_mode(lora_only=True)

    last_dpo_loss = 0.0
    last_margin = 0.0

    for _ in range(epochs):
        # 1. Reference log-probs with LoRA disabled (frozen Int8 base weights, zero extra RAM)
        ref_log_w, _, _ = model.compute_sequence_log_probs(prompt_len, s_w_ids, disable_lora=True)
        ref_log_l, _, _ = model.compute_sequence_log_probs(prompt_len, s_l_ids, disable_lora=True)

        # 2. Policy log-probs with LoRA enabled
        pi_log_w, logits_w, probs_w = model.compute_sequence_log_probs(prompt_len, s_w_ids, disable_lora=False)
        pi_log_l, logits_l, probs_l = model.compute_sequence_log_probs(prompt_len, s_l_ids, disable_lora=False)

        # 3. DPO Loss & reward margin
        delta_w = pi_log_w - ref_log_w
        delta_l = pi_log_l - ref_log_l
        margin = delta_w - delta_l
        u = beta * margin
        u_clipped = max(-30.0, min(30.0, u))
        dpo_loss = math.log(1.0 + math.exp(-u_clipped))
        sig_neg_u = 1.0 / (1.0 + math.exp(u_clipped))

        last_dpo_loss = dpo_loss
        last_margin = margin

        # 4. Analytical policy gradients on chosen and rejected logits
        grad_scale = beta * sig_neg_u
        V = model.vocab_size

        for t in range(prompt_len - 1, len(s_w_ids) - 1):
            target_token = s_w_ids[t + 1]
            p_row = probs_w[t]
            tV = t * V
            for v in range(V):
                indicator = 1.0 if v == target_token else 0.0
                logits_w.grad[tV + v] += grad_scale * (p_row[v] - indicator)

        for t in range(prompt_len - 1, len(s_l_ids) - 1):
            target_token = s_l_ids[t + 1]
            p_row = probs_l[t]
            tV = t * V
            for v in range(V):
                indicator = 1.0 if v == target_token else 0.0
                logits_l.grad[tV + v] -= grad_scale * (p_row[v] - indicator)

        logits_w.backward()
        logits_l.backward()
        model.step_adam(lr, lora_only=True, weight_decay=weight_decay)

    return last_dpo_loss, last_margin
