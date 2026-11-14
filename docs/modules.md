# Thoughtformula Modules Reference

Thoughtformula is organized into clean, single-responsibility modules:

| Module | Primary Class / Functions | Description |
|---|---|---|
| `thoughtformula.tensor` | `FlatTensor` | Autograd tensor engine with float64 storage, Int8 quantization, math.sumprod C-speed matmul, GQA, MoE router, and backward pass. |
| `thoughtformula.tokenizer` | `BPETokenizer` | Dynamic Byte Pair Encoding tokenizer with special tokens and online merge training. |
| `thoughtformula.rag` | `PurePythonBM25RAG`, `HybridDenseBM25RAG` | Okapi BM25 and dense semantic retrieval with Reciprocal Rank Fusion. |
| `thoughtformula.draft` | `NGramDraftModel` | Fast N-gram draft model with proposal distribution output for speculative decoding. |
| `thoughtformula.model` | `ModernMicroGPT` | Full transformer model uniting GQA, MoE SwiGLU, Int8 KV-cache, LoRA, and binary .mgpt container. |
| `thoughtformula.training` | `get_cosine_lr`, `build_training_sequences`, `continual_learn_lora_turn`, `train_dpo_turn` | Learning rate schedules, batch sequences, online LoRA adaptation, and DPO. |
| `thoughtformula.gradcheck` | `run_gradient_checks` | Finite-difference numerical gradient verification (< 1e-5 error tolerance). |
| `thoughtformula.cli` | `main`, `run_repl`, `benchmark_speculative`, `resolve_checkpoint_path` | CLI runner, benchmark harness, and interactive REPL. |

## Module Details

### `thoughtformula.tensor`
- **`FlatTensor(rows, cols, data, _children, requires_grad)`**: Underlying tensor data structure.
- **`FlatTensor.randn(rows, cols, std, requires_grad)`**: Random Gaussian initialization.
- **`FlatTensor.zeros(rows, cols, requires_grad)`**: Zero initialization.
- **`FlatTensor.matmul(other, use_int8)`**: Matrix multiplication with autograd.
- **`FlatTensor.matmul_int8(other)`**: Symmetric Int8 quantized matrix multiplication.
- **`FlatTensor.rmsnorm(eps)`**: Root Mean Square Normalization.
- **`FlatTensor.swiglu_act(gate, up)`**: SwiGLU activation.
- **`FlatTensor.moe_router_combine(router_logits, expert_outputs, top_k)`**: Top-K routing.
- **`FlatTensor.rope_gqa(Q, K, V, n_heads, n_kv_heads, start_pos, kv_cache)`**: Grouped Query Attention with RoPE.
- **`FlatTensor.cross_entropy(logits, targets)`**: Cross-entropy classification loss.

### `thoughtformula.tokenizer`
- **`BPETokenizer.train(text, max_merges, min_freq)`**: Trains BPE merges on raw text.
- **`BPETokenizer.encode(text)`**: Encodes text string to token IDs.
- **`BPETokenizer.decode(ids, skip_special)`**: Decodes token IDs back to string.
- **`BPETokenizer.expand_vocab(text)`**: Dynamically registers new subwords on the fly.

### `thoughtformula.draft`
- **`NGramDraftModel(max_n)`**: Instantiates multi-order N-gram tables.
- **`fit(token_sequences)`**: Compiles transition counts across sequence corpus.
- **`draft(prefix_ids, gamma, stop_token_id)`**: Proposes candidate draft tokens.

### `thoughtformula.rag`
- **`PurePythonBM25RAG(k1, b)`**: BM25 document store.
- **`add_document(question, fact)`**: Indexes knowledge fact.
- **`retrieve(query, top_k)`**: Returns ranked relevant documents.

### `thoughtformula.model`
- **`ModernMicroGPT(...)`**: Transformer model definition.
- **`forward(idx_seq, start_pos, kv_caches, use_int8)`**: Forward pass.
- **`generate(...)`**: Standard autoregressive generation.
- **`generate_speculative(...)`**: Speculative decoding generation.
- **`save_checkpoint(filepath, ...)`** / **`load_checkpoint(filepath)`**: Serialization.
