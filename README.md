# Thoughtformula 🧠

> Pure-Python generative Transformer architecture featuring Grouped Query Attention (GQA), Rotary Positional Embeddings (RoPE), Sparse Mixture of Experts (MoE SwiGLU), symmetric Int8 quantization, parameter-efficient LoRA continual learning, and speculative decoding. Built exclusively with Python standard library built-ins.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12%2B%20%7C%203.14%2B-blue.svg)]()
[![Dependencies](https://img.shields.io/badge/dependencies-0%20(stdlib%20only)-success.svg)]()

---

## Key Features

- **Standard Library Autograd (`FlatTensor`)**: 1D contiguous float64 training buffer with symmetric Int8 inference quantization and reverse-mode automatic differentiation.
- **Grouped Query Attention (GQA) & RoPE**: Memory-efficient query/key-value head grouping with precomputed or rotational positional embeddings.
- **Sparse Mixture of Experts (MoE)**: 4 SwiGLU experts per layer with Top-2 token routing.
- **LoRA Continual Learning**: Instant parameter-efficient adaptation of frozen Int8 base weights during conversation turns.
- **Speculative Decoding**: Hierarchical N-gram drafter delivering single-pass candidate verification with Int8 KV-cache rollback.
- **Pure-Python BM25 RAG**: Okapi BM25 retrieval for grounding conversation context.

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
