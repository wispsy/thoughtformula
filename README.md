# Thoughtformula 🧠

> Pure-Python generative Transformer architecture featuring FlashAttention online softmax tiling, vLLM-style Paged KV-Cache, BitNet b1.58 ternary quantization, Grouped Query Attention (GQA), Rotary Positional Embeddings (RoPE), Sparse Mixture of Experts (MoE SwiGLU), LoRA continual learning, Kahneman-Tversky Optimization (KTO), Medusa speculative decoding, persistent binary index (.tfidx), and continuous batching. Built exclusively with Python standard library built-ins.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12%2B%20%7C%203.14%2B-blue.svg)]()
[![Dependencies](https://img.shields.io/badge/dependencies-0%20(stdlib%20only)-success.svg)]()
[![Version](https://img.shields.io/badge/version-0.3.0-orange.svg)]()

---

## Key Features

- **Standard Library Autograd (`FlatTensor`)**: 1D contiguous float64 training buffer with symmetric Int8 inference quantization, BitNet b1.58 ternary quantization, and reverse-mode automatic differentiation.
- **FlashAttention Online Softmax Tiling**: Dao et al. streaming online softmax eliminates $T \times T$ attention matrix materialization, maintaining $O(N)$ memory across long context windows.
- **vLLM-Style Paged KV-Cache (`PagedKVCache`)**: Fixed-size physical memory pages (e.g., 16 tokens/block), block allocator, and page tables supporting non-contiguous allocation and $O(1)$ prompt prefix sharing.
- **BitNet b1.58 Ternary Quantization**: Weights in $\{-1, 0, 1\}$ with scale $\gamma = \text{mean}(|W|)$, 2-bit bitpacking (4 weights/byte), and addition-only matrix multiplication.
- **Kahneman-Tversky Optimization (KTO)**: Unpaired binary human preference alignment (`q :: a ++` and `q :: a --`) using prospect theory utility loss and frozen base Int8 weights as reference.
- **Interactive Streaming Token Generator**: Real-time typewriter output streaming via `generate_stream()`, `generate_speculative_stream()`, and `generate_medusa_stream()`.
- **Medusa Multi-Head Speculative Drafter**: Lightweight auxiliary heads predicting multiple future tokens simultaneously directly from the final hidden state without extra transformer layers.
- **Persistent Binary Index Format (`.tfidx`)**: High-performance binary container persisting precomputed dense embeddings and term inverted postings for large-scale hybrid RAG retrieval.
- **Continuous Batching Engine (`ContinuousBatchScheduler`)**: Synchronous iteration-level continuous batcher packing active requests into unified forward passes, amortizing parameter loading across concurrent sessions.
- **C-Speed Matrix Multiplication**: Accelerated with compiled C `math.sumprod` over contiguous transposed weight slices (5x-12x pure-Python speedup).
- **Grouped Query Attention (GQA) & Precomputed RoPE**: Memory-efficient query/key-value head grouping with precomputed `cos_cache` / `sin_cache` trigonometric lookup tables.
- **Sparse Mixture of Experts (MoE)**: 4 SwiGLU experts with true sparse token dispatch and Switch Transformer auxiliary load-balancing loss preventing expert collapse.
- **Zero-Memory Direct Preference Optimization (DPO)**: Align model responses (`q :: good || bad`) using frozen Int8 base weights as reference policy $\pi_{\text{ref}}$ with 0 extra RAM.
- **Decoupled Weight Decay (AdamW)**: Bounded weight magnitudes preventing Int8 scale drift.
- **Lossless Speculative Decoding**: Leviathan et al. (2023) rejection sampling with N-gram proposal distributions and Int8 KV-cache rollback.
- **Hybrid Dense + Lexical RAG**: Okapi BM25 combined with mean-pooled `tok_emb` dense embeddings via Reciprocal Rank Fusion (RRF).
- **Binary `.mgpt` Container**: Python standard library `zipfile` binary container eliminating 33% Base64 storage overhead.
- **Built-in Gradient Checker**: Finite-difference calculus test harness (`/test`) verifying all analytical gradients to $< 10^{-10}$.

---

## Quickstart

```bash
# Install via pip
pip install thoughtformula

# Run CLI directly
thoughtformula --benchmark

# Or run interactive typewriter REPL
thoughtformula
```

### Run from Source

```bash
git clone https://github.com/wispsy/thoughtformula.git
cd thoughtformula

# Run speculative decoding benchmark
PYTHONPATH=src python3 -m thoughtformula.cli --benchmark

# Run interactive typewriter REPL
PYTHONPATH=src python3 -m thoughtformula.cli
```

### REPL Commands & Syntax

| Syntax | Action |
|---|---|
| `question?` | RAG retrieval + speculative decoding typewriter output |
| `q :: answer` | Continual online LoRA adaptation |
| `q :: good \|\| bad` | Zero-Memory LoRA Direct Preference Optimization (DPO) |
| `q :: answer ++` | Kahneman-Tversky Optimization (KTO) Desirable feedback |
| `q :: answer --` | Kahneman-Tversky Optimization (KTO) Undesirable feedback |
| `/test` | Run finite-difference calculus gradient verification |
| `/merge` | Merge LoRA adapters into base weights |
| `/stats` | Display model parameter count and cache metrics |

---

## Project Structure

```
thoughtformula/
├── pyproject.toml
├── README.md
├── docs/
│   ├── architecture.md
│   ├── modules.md
│   └── quickstart.md
├── src/
│   └── thoughtformula/
│       ├── __init__.py
│       ├── batching.py
│       ├── cache.py
│       ├── cli.py
│       ├── draft.py
│       ├── gradcheck.py
│       ├── medusa.py
│       ├── micro_gpt_checkpoint.json
│       ├── micro_gpt_v8.py
│       ├── model.py
│       ├── rag.py
│       ├── tensor.py
│       ├── tokenizer.py
│       └── training.py
└── tests/
    ├── test_batching.py
    ├── test_cache.py
    ├── test_draft.py
    ├── test_gradcheck.py
    ├── test_medusa.py
    ├── test_model.py
    ├── test_rag.py
    ├── test_tensor.py
    ├── test_tokenizer.py
    └── test_training.py
```

---

## Running Tests

All 64 unit tests run using Python's built-in `unittest` framework with zero external dependencies:

```bash
PYTHONPATH=src python3 -m unittest discover tests
```

---

## License

MIT


## Generation request limits

All standard, speculative and Medusa generation methods accept nonnegative integer
token budgets, reject invalid vocabulary IDs, and require prompt plus output
budget within `model.max_seq_len`. Zero budget preserves the prompt in nonstream
results and yields nothing in streams, without cache allocation, Medusa setup or
a forward pass. Empty prompts retain their existing empty return shapes. Sampling
and return shapes otherwise remain compatible.


## Bounded homegrown training

`TrainingConfig`, `TrainingState`, `fit` and `evaluate` expose bounded next-token
training over the existing autograd/Adam implementation. Callers select token
sequences, context/step/time budgets and optional boundary stop/progress callbacks.
Evaluation reports token-weighted loss/perplexity and exact denominators.

`save_training_checkpoint` and `load_training_checkpoint` use versioned atomic
lossless float64 parameter/Adam state for exact fixed-order resume. These bounded
passive `.tft` artifacts are distinct from quantized inference exports, which
retain their existing API. Checkpoints preserve model/tokenizer/config/progress
and caller metadata; source/data identity remains the application's responsibility.
Save at completed optimizer boundaries. `reset_optimizer` explicitly starts a new
specialization trajectory while preserving learned weights. Auxiliary Medusa
heads are outside this initial training artifact profile. No data/model download,
pickle or implicit promotion occurs.
