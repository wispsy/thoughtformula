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
