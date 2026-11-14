# Thoughtformula Quickstart

## Installation

Thoughtformula requires Python 3.12+ (or Python 3.14+) and has **zero external dependencies**.

```bash
git clone https://github.com/wispsy/thoughtformula.git
cd thoughtformula
```

## Running the REPL

Launch the interactive REPL:

```bash
python3 -m thoughtformula.cli
```

Or using `uv`:

```bash
uv run thoughtformula
```

## Running the Benchmark

Run the speculative decoding benchmark comparing standard generation with N-gram drafting:

```bash
python3 -m thoughtformula.cli --benchmark
```

## Python API Usage

```python
from thoughtformula import ModernMicroGPT, BPETokenizer, PurePythonBM25RAG, NGramDraftModel

# Load the pretrained Int8 GQA + MoE checkpoint
model, tokenizer, rag_engine, corpus, migrated = ModernMicroGPT.load_checkpoint(
    "src/thoughtformula/micro_gpt_checkpoint.json"
)

# Prepare RAG prompt
prompt_text = tokenizer.format_rag_prompt(
    "deep learning in pure python.",
    "what is this model?"
)
prompt_ids = tokenizer.encode(prompt_text)

# Generate with Int8 KV-Cache
output_ids = model.generate(prompt_ids, max_new_tokens=20, use_int8=True)
response = tokenizer.decode(output_ids[len(prompt_ids):], skip_special=True)
print("Response:", response)
```

## Continual Learning (Teaching New Facts)

In the interactive REPL:

```
You > what is quantum computing? :: quantum computing uses qubits for superposition.
```

The model instantly adapts its LoRA adapters to incorporate the new knowledge in milliseconds.
