# Thoughtformula Architecture Specification

Thoughtformula is a complete, self-contained generative language model architecture written 100% in pure Python using only standard library built-ins (`array`, `math`, `struct`, `zlib`, `random`, `json`, `zipfile`).

```
                              [ User Query ]
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │ PurePythonBM25RAG Retriever  │
                     │  (Persistent .tfidx Index)   │
                     └──────────────┬───────────────┘
                                    │ Formatted RAG Prompt
                                    ▼
                     ┌──────────────────────────────┐
                     │   BPETokenizer (BPE Vocab)   │
                     └──────────────┬───────────────┘
                                    │ Token IDs
                        ┌───────────┴───────────┐
                        │                       │
                        ▼                       ▼
            ┌───────────────────────┐  ┌──────────────────┐
            │   Medusa / N-Gram     │  │   Target LLM     │
            │ Speculative Drafters  │  │ (ModernMicroGPT) │
            └───────────┬───────────┘  └────────┬─────────┘
                        │                       │
                        └───────────┬───────────┘
                                    │ Single-pass verify
                                    ▼
                     ┌──────────────────────────────┐
                     │ FlashAttention + Paged Cache │
                     │  (Continuous Batch Engine)   │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                         [ Interactive Stream ]
```

## Core Subsystems

### 1. Flat Autograd Tensor Engine (`FlatTensor`)
- Contiguous 1D flat memory layout using Python's native `array('d')` for float64 calculations.
- Integrated symmetric Int8 quantization (`array('b')`) with dynamic per-tensor scaling: `scale = max_abs / 127.0`.
- BitNet b1.58 ternary quantization ($\{-1, 0, 1\}$) with 2-bit packing and addition-only matrix multiplication.
- Native matrix multiplication accelerated with compiled C `math.sumprod` over contiguous transposed weight slices (5x-12x faster).
- Native Root Mean Square Normalization (`rmsnorm`), SwiGLU non-linear activation (`swiglu_act`), and cross-entropy loss.
- Dynamic Directed Acyclic Graph (DAG) reverse-mode automatic differentiation with analytical finite-difference verification to `< 1e-10`.
- `requires_grad` flag support for parameter freezing during Low-Rank Adaptation (LoRA).

### 2. FlashAttention Online Softmax Tiling
- Online softmax tiling (Dao et al. 2022) streaming across key-value blocks in fixed chunk sizes (e.g. 16 or 32 tokens).
- Eliminates materialization of the $T \times T$ attention matrix, cutting memory complexity from $O(T^2)$ to $O(T)$.
- Gradient backward pass reconstructs attention probabilities on-the-fly from $O(T)$ saved statistics ($m$ and $l$), matching standard attention analytical gradients within machine precision ($< 10^{-12}$).

### 3. vLLM-Style Paged KV-Cache (`PagedKVCache`)
- Fixed-size physical memory pages (`PhysicalBlock`) holding quantized keys, values, and scales.
- Global pool allocator (`BlockAllocator`) managing block allocation, reclamation, and reference counting.
- Logical page table mapping sequence positions to non-contiguous physical blocks.
- $O(1)$ prompt prefix sharing: forking active sequences increments page reference counts without copying underlying buffers, applying Copy-on-Write (CoW) on mutation.

### 4. BitNet b1.58 Ternary Quantization
- Weights quantized to $\{-1, 0, 1\}$ using mean absolute scale factor $\gamma = \frac{1}{R \cdot C} \sum |W|$.
- Bitpacking encodes 4 ternary weights per byte into standard library `array('B')`, reducing storage $4\times$ over bytes and $32\times$ over float64.
- Addition-only matrix multiplication: inner products perform additions and subtractions without floating-point multiplications, scaling by $\gamma$ once at output.

### 5. Preference Alignment: DPO and KTO
- **Zero-Memory DPO**: Direct Preference Optimization (`q :: good || bad`) using frozen Int8 base weights as reference policy ($\pi_{\text{ref}}$) alongside active LoRA adapters ($\pi_\theta$) with 0 extra memory allocation.
- **Kahneman-Tversky Optimization (KTO)**: Unpaired binary human preference alignment (`q :: a ++` and `q :: a --`) using prospect theory utility loss:
  $$\mathcal{L}_{\text{desirable}} = \log(1 + e^{-(z - z_{\text{ref}})})$$
  $$\mathcal{L}_{\text{undesirable}} = \log(1 + e^{\lambda_U (z - z_{\text{ref}})})$$

### 6. Speculative Decoding: Medusa & Hierarchical N-Gram
- **Medusa Drafter**: Lightweight multi-head prediction heads directly on top of the final transformer hidden state predicting multiple future tokens simultaneously.
- **N-Gram Drafter**: Hierarchical 4-gram $\to$ 3-gram $\to$ 2-gram backoff table with proposal probability distributions.
- **Lossless Verification**: Leviathan et al. (2023) rejection sampling with rollback support on rejection.

### 7. Persistent Binary Index Format (`.tfidx`)
- Binary container format packing metadata, documents, precomputed dense embeddings, and inverted postings lists using `struct` and `zlib`.
- Inverted index maps terms to postings `(doc_id, tf)` for instantaneous lexical BM25 lookup without scanning unmatching documents.
- Precomputed dense embeddings enable vector cosine similarities via `math.sumprod` with zero model forward passes during search.

### 8. Continuous Batching Engine (`ContinuousBatchScheduler`)
- Synchronous iteration-level continuous batcher packing active requests into unified forward passes.
- Amortizes linear parameter projections ($W_q, W_k, W_v, W_o, W_{\text{experts}}, W_{\text{head}}$) across concurrent requests while isolating attention per sequence KV-cache.
- As active requests finish, pending requests are immediately admitted on the next iteration without stalling.
