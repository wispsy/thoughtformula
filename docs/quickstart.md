# Thoughtformula Quickstart

## Installation

Thoughtformula requires Python 3.12+ (or Python 3.14+) and has **zero external dependencies**.

```bash
git clone https://github.com/wispsy/thoughtformula.git
cd thoughtformula
```

## Running the REPL

Launch the interactive typewriter streaming REPL:

```bash
PYTHONPATH=src python3 -m thoughtformula.cli
```

Or using `uv`:

```bash
uv run thoughtformula
```

### REPL Usage Examples

```text
User > what is this model?
Assistant > deep learning in pure python.

# Online Continual LoRA Learning
User > what is python? :: a modern programming language.
[Minted BPE: 'modern' | LoRA Adapted in 42 ms | Loss: 0.1234 | Saved 'micro_gpt_checkpoint.json']

# Kahneman-Tversky Optimization (KTO) Desirable feedback
User > what is python? :: an elegant and clean language. ++
[LoRA KTO Adapted (++ Desirable) in 38 ms | Loss: 0.2415 | Reward: +0.412 | Saved 'micro_gpt_checkpoint.json']

# Kahneman-Tversky Optimization (KTO) Undesirable feedback
User > what is python? :: an ancient snake. --
[LoRA KTO Adapted (-- Undesirable) in 36 ms | Loss: 0.1982 | Reward: -0.380 | Saved 'micro_gpt_checkpoint.json']

# Direct Preference Optimization (DPO)
User > what is python? :: a programming language. || an ancient snake.
[LoRA DPO Adapted in 55 ms | Loss: 0.0821 | Margin: +1.420 | Saved 'micro_gpt_checkpoint.json']
```

## Running the Benchmark

Run the speculative decoding benchmark:

```bash
PYTHONPATH=src python3 -m thoughtformula.cli --benchmark
```

## Python API Usage

### 1. Interactive Streaming Generation
```python
from thoughtformula import ModernMicroGPT, BPETokenizer

model, tokenizer, _, _, _ = ModernMicroGPT.load_checkpoint(
    "src/thoughtformula/micro_gpt_checkpoint.json"
)

prompt_ids = tokenizer.encode("what is this model?")
for tok_id in model.generate_stream(prompt_ids, max_new_tokens=16):
    print(tokenizer.decode([tok_id]), end="", flush=True)
print()
```

### 2. Continuous Batching Scheduler
```python
from thoughtformula import ModernMicroGPT, ContinuousBatchScheduler

model = ModernMicroGPT(vocab_size=32, n_embd=16, n_heads=2, n_kv_heads=1, n_layers=1)
scheduler = ContinuousBatchScheduler(model, max_batch_size=4, use_paged_cache=True)

# Add concurrent requests
scheduler.add_request([1, 2, 3], request_id="req1", max_new_tokens=10)
scheduler.add_request([4, 5], request_id="req2", max_new_tokens=12)

# Stream concurrent generation iterations
for step_out in scheduler.stream():
    print("Decoded step:", step_out)
```

### 3. Persistent Binary RAG Index (.tfidx)
```python
from thoughtformula import PurePythonBM25RAG

rag = PurePythonBM25RAG()
rag.add_document("what is python?", "python is a versatile programming language.")
rag.save_index("documents.tfidx")

loaded_rag = PurePythonBM25RAG.load_index("documents.tfidx")
results = loaded_rag.retrieve("python language", top_k=1)
print("Retrieved:", results[0][0])
```
