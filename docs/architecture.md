# Thoughtformula Architecture Specification

Thoughtformula is a complete, self-contained generative language model architecture written 100% in pure Python using only standard library built-ins (`array`, `math`, `struct`, `zlib`, `random`, `json`).

```
                              [ User Query ]
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │ PurePythonBM25RAG Retriever  │
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
            │ NGramDraftModel (O(1))│  │   Target LLM     │
            │   gamma candidates    │  │ (ModernMicroGPT) │
            └───────────┬───────────┘  └────────┬─────────┘
                        │                       │
                        └───────────┬───────────┘
                                    │ Single-pass verify
                                    ▼
                     ┌──────────────────────────────┐
                     │  Speculative Decoding Verify │
                     │   (Int8 KV-Cache Rollback)   │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                             [ Stream Output ]
```

## Core Subsystems

### 1. Flat Autograd Tensor Engine (`FlatTensor`)
- Contiguous 1D flat memory layout using Python's native `array('d')` for float64 calculations.
- Integrated symmetric Int8 quantization (`array('b')`) with dynamic per-tensor scaling: `scale = max_abs / 127.0`.
- Native matrix multiplication accelerated with compiled C `math.sumprod` over contiguous transposed weight slices (5x-12x faster).
- Native Root Mean Square Normalization (`rmsnorm`), SwiGLU non-linear activation (`swiglu_act`), and cross-entropy loss.
- Dynamic Directed Acyclic Graph (DAG) reverse-mode automatic differentiation with analytical finite-difference verification to `< 1e-10`.
- `requires_grad` flag support for parameter freezing during Low-Rank Adaptation (LoRA).

### 2. Grouped Query Attention (GQA) & Rotary Positional Embeddings (RoPE)
- Pre-norm transformer architecture.
- Grouped Query Attention sharing key-value heads: $N_{\text{heads}} = 4$, $N_{\text{kv\_heads}} = 2$.
- Rotary Positional Embeddings (RoPE) with precomputed `cos_cache` and `sin_cache` lookup tables eliminating redundant trigonometric calls.
- Quantized Int8 KV-cache for autoregressive decoding with rollback support.

### 3. Sparse Mixture of Experts (MoE) with SwiGLU & Switch Transformer Auxiliary Loss
- True sparse token dispatch during both inference and training: only active winning experts are evaluated, cutting MoE FLOPs in half.
- Switch Transformer auxiliary load-balancing loss ($\mathcal{L}_{\text{aux}} = \alpha \cdot N_E \sum_e f_e \cdot P_e$) to prevent expert collapse and ensure balanced specialization across all experts.
- SwiGLU feed-forward networks: $\text{SwiGLU}(x) = (x W_{\text{gate}} \cdot \sigma(x W_{\text{gate}})) \odot (x W_{\text{up}}) W_{\text{down}}$.
- Full analytical gradient backpropagation to both router logits and active expert weights.

### 4. Parameter-Efficient LoRA & Zero-Memory Direct Preference Optimization (DPO)
- Low-Rank Adaptation weights $W = W_0 + \frac{\alpha}{r} (A \cdot B)$, where $A \sim \mathcal{N}(0, 0.05)$ and $B = 0$.
- In REPL continual learning turns, heavy base weights (>85% of total parameters) are frozen while LoRA adapters train in milliseconds.
- Decoupled weight decay (AdamW) prevents weight drift and preserves Int8 dynamic range.
- Zero-Memory Direct Preference Optimization (DPO): uses frozen Int8 base weights as reference policy ($\pi_{\text{ref}}$) alongside trainable LoRA policy ($\pi_\theta$) with 0 extra memory allocation.
- Lossless merging of LoRA adapters back into Int8 base weights (`/merge`).

### 5. Speculative Decoding with N-Gram Drafter (Leviathan et al. 2023)
- Hierarchical N-gram drafter (4-gram $\to$ 3-gram $\to$ 2-gram backoff) generating $\gamma=4$ candidate tokens with zero neural overhead.
- Leviathan et al. (2023) lossless speculative rejection sampling at arbitrary decoding temperatures.
- Int8 KV-cache rollback on rejection with bonus token collection on full-draft acceptance.

### 6. Hybrid Dense + Lexical RAG (Reciprocal Rank Fusion)
- Okapi BM25 ranking algorithm with term stemming and stopword filtering.
- Dense semantic sentence embeddings computed via zero-parameter mean-pooling of model `tok_emb` vectors.
- Reciprocal Rank Fusion (RRF) blending lexical keyword precision with dense conceptual synonym matching.

### 7. Pure-Python Binary `.mgpt` Container Format
- Drop Base64 overhead: standard zipfile and binary struct packing eliminate the 33% Base64 storage expansion.
