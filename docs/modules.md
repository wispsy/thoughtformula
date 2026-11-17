# Thoughtformula Modules Reference

Thoughtformula is organized into clean, single-responsibility modules:

| Module | Primary Class / Functions | Description |
|---|---|---|
| `thoughtformula.tensor` | `FlatTensor` | Autograd tensor engine with float64 storage, Int8 & BitNet b1.58 ternary quantization, FlashAttention online softmax tiling, math.sumprod C-speed matmul, GQA, MoE router, and backward pass. |
| `thoughtformula.cache` | `PhysicalBlock`, `BlockAllocator`, `PagedLayerCache`, `PagedKVCache`, `PagedKVCacheManager` | vLLM-style paged KV-cache with fixed-size block pages, reference counting, and prompt prefix sharing. |
| `thoughtformula.batching` | `ContinuousBatchScheduler`, `InferenceRequest` | Synchronous continuous batching scheduler for concurrent multi-request inference. |
| `thoughtformula.medusa` | `MedusaHead`, `MedusaDrafter` | Lightweight multi-head speculative drafter predicting multiple future tokens simultaneously. |
| `thoughtformula.tokenizer` | `BPETokenizer` | Dynamic Byte Pair Encoding tokenizer with special tokens and online merge training. |
| `thoughtformula.rag` | `PurePythonBM25RAG`, `HybridDenseBM25RAG` | Okapi BM25, dense semantic retrieval with Reciprocal Rank Fusion, and persistent `.tfidx` binary index. |
| `thoughtformula.draft` | `NGramDraftModel` | Fast N-gram draft model with proposal distribution output for speculative decoding. |
| `thoughtformula.model` | `ModernMicroGPT` | Full transformer model uniting FlashAttention, GQA, MoE SwiGLU, Paged KV-cache, LoRA, BitNet, and streaming APIs. |
| `thoughtformula.training` | `get_cosine_lr`, `build_training_sequences`, `continual_learn_lora_turn`, `train_dpo_turn`, `train_kto_turn` | Learning rate schedules, batch sequences, online LoRA adaptation, DPO, and KTO. |
| `thoughtformula.gradcheck` | `run_gradient_checks` | Finite-difference numerical gradient verification (< 1e-5 error tolerance). |
| `thoughtformula.cli` | `main`, `run_repl`, `benchmark_speculative`, `resolve_checkpoint_path` | CLI runner, typewriter streaming REPL, and benchmark harness. |

## Module Details

### `thoughtformula.tensor`
- **`FlatTensor(rows, cols, data, _children, requires_grad)`**: Underlying tensor data structure.
- **`FlatTensor.matmul(other, use_int8, use_ternary)`**: Matrix multiplication with autograd, Int8, or BitNet b1.58 addition-only mode.
- **`FlatTensor.quantize_ternary()`**: Quantizes weights to $\{-1, 0, 1\}$ with scale $\gamma = \text{mean}(|W|)$ and 2-bit bitpacking.
- **`FlatTensor.matmul_ternary(other)`**: BitNet b1.58 addition-only matrix multiplication.
- **`FlatTensor.rope_gqa(...)`**: RoPE Grouped Query Attention with FlashAttention online softmax tiling ($O(N)$ memory).

### `thoughtformula.cache`
- **`BlockAllocator(page_size, kv_dim, initial_blocks)`**: Global physical memory page pool with reference counting.
- **`PagedKVCache(n_layers, page_size, kv_dim, allocator)`**: Multi-layer paged cache supporting $O(1)$ prefix forks.
- **`PagedKVCacheManager`**: High-level coordinator managing sequence allocations and memory deduplication telemetry.

### `thoughtformula.batching`
- **`ContinuousBatchScheduler(model, max_batch_size, use_paged_cache, use_int8)`**: Iteration-level continuous batching scheduler.
- **`step()`**: Executes one synchronous decode step across active requests.
- **`stream()`**: Yields token dictionaries per iteration until all requests finish.

### `thoughtformula.medusa`
- **`MedusaHead(head_idx, n_embd, vocab_size)`**: Auxiliary linear head predicting token $t + 2 + k$.
- **`MedusaDrafter(n_embd, vocab_size, num_heads)`**: Multi-head auxiliary drafter for speculative tree decoding.
- **`train_step(h_seq, targets, lr)`**: Cross-entropy training on shifted hidden state sequence.

### `thoughtformula.rag`
- **`PurePythonBM25RAG(k1, b)`**: Lexical and dense hybrid document store.
- **`build_index(model, tokenizer)`**: Precomputes inverted index and dense embeddings.
- **`save_index(filepath)`** / **`load_index(filepath)`**: Persistent binary `.tfidx` container format.
- **`retrieve_hybrid(query, model, tokenizer, top_k, rrf_k)`**: RRF-fused retrieval.

### `thoughtformula.model`
- **`ModernMicroGPT(...)`**: Transformer model definition.
- **`forward(...)`**: Standard forward pass with FlashAttention, Int8, and BitNet b1.58 ternary options.
- **`forward_batched(token_list, req_caches, start_positions, ...)`**: Packed multi-request decode pass.
- **`generate_stream(...)`**: Streaming generator yielding tokens in real time.
- **`generate_medusa(...)`** / **`generate_medusa_stream(...)`**: Medusa speculative decoding and streaming.
