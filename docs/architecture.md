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
- Native matrix multiplication (`matmul` and `matmul_int8`), Root Mean Square Normalization (`rmsnorm`), SwiGLU non-linear activation (`swiglu_act`), and cross-entropy loss.
- Dynamic Directed Acyclic Graph (DAG) reverse-mode automatic differentiation.
- `requires_grad` flag support for parameter freezing during Low-Rank Adaptation (LoRA).

### 2. Grouped Query Attention (GQA) & Rotary Positional Embeddings (RoPE)
- Pre-norm transformer architecture.
- Grouped Query Attention sharing key-value heads: $N_{\text{heads}} = 4$, $N_{\text{kv\_heads}} = 2$.
- Rotary Positional Embeddings (RoPE) applied to query and key vectors before dot-product attention.
- Quantized Int8 KV-cache for autoregressive decoding with rollback support.

### 3. Sparse Mixture of Experts (MoE) with SwiGLU Activation
- Dynamic routing across $N_E = 4$ experts, activating Top-2 experts per token.
- SwiGLU feed-forward networks: $\text{SwiGLU}(x) = (x W_{\text{gate}} \cdot \sigma(x W_{\text{gate}})) \odot (x W_{\text{up}}) W_{\text{down}}$.
- Full analytical gradient backpropagation to both router logits and active expert weights.

### 4. Parameter-Efficient Fine-Tuning (LoRA) Continual Learning
- Low-Rank Adaptation weights $W = W_0 + \frac{\alpha}{r} (A \cdot B)$, where $A \sim \mathcal{N}(0, 0.05)$ and $B = 0$.
- In REPL continual learning turns, heavy base weights (>85% of total parameters) are frozen while LoRA adapters train in milliseconds.
- Support for lossless merging of LoRA adapters back into Int8 base weights (`/merge`).

### 5. Speculative Decoding with N-Gram Drafter
- Hierarchical N-gram drafter (4-gram $\to$ 3-gram $\to$ 2-gram backoff) generating $\gamma=4$ candidate tokens with zero neural overhead.
- Single-pass Transformer verification with Int8 KV-cache rollback on rejection.
- Standard-compliant acceptance rate exceeding 80% on domain knowledge.

### 6. Retrieval-Augmented Generation (BM25 RAG)
- Okapi BM25 ranking algorithm with term stemming and stopword filtering.
- Automatic context injection into conversational turns.
