# Thoughtformula 🧠

> Pure-Python generative Transformer architecture featuring Grouped Query Attention (GQA), Rotary Positional Embeddings (RoPE), Sparse Mixture of Experts (MoE SwiGLU), symmetric Int8 quantization, parameter-efficient LoRA continual learning, and speculative decoding. Built exclusively with Python standard library built-ins.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12%2B%20%7C%203.14%2B-blue.svg)]()
[![Dependencies](https://img.shields.io/badge/dependencies-0%20(stdlib%20only)-success.svg)]()

---

## Key Features

- **Standard Library Autograd (`FlatTensor`)**: 1D contiguous float64 training buffer with symmetric Int8 inference quantization and reverse-mode automatic differentiation.
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
# Clone the repository
git clone https://github.com/wispsy/thoughtformula.git
cd thoughtformula

# Run speculative decoding benchmark
PYTHONPATH=src python3 -m thoughtformula.cli --benchmark

# Run interactive REPL
PYTHONPATH=src python3 -m thoughtformula.cli
```

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
│       ├── cli.py
│       ├── draft.py
│       ├── micro_gpt_checkpoint.json
│       ├── micro_gpt_v8.py
│       ├── model.py
│       ├── rag.py
│       ├── tensor.py
│       ├── tokenizer.py
│       └── training.py
└── tests/
    ├── test_draft.py
    ├── test_model.py
    ├── test_rag.py
    ├── test_tensor.py
    ├── test_tokenizer.py
    └── test_training.py
```

---

## Running Tests

All unit tests run using Python's built-in `unittest` framework:

```bash
PYTHONPATH=src python3 -m unittest discover tests
```

---

## License

MIT
