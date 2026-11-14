"""Backward-compatible entrypoint for micro_gpt_v8."""

from thoughtformula.tensor import FlatTensor
from thoughtformula.draft import NGramDraftModel
from thoughtformula.rag import PurePythonBM25RAG
from thoughtformula.tokenizer import BPETokenizer
from thoughtformula.model import ModernMicroGPT
from thoughtformula.training import get_cosine_lr, build_training_sequences, continual_learn_lora_turn
from thoughtformula.cli import main, resolve_checkpoint_path, run_repl, benchmark_speculative

if __name__ == "__main__":
    main()
