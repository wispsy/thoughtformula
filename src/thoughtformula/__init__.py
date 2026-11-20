"""Thoughtformula: Pure-Python Transformer with GQA, Sparse MoE SwiGLU, Int8 quantization, LoRA, and Speculative Decoding."""

from .tensor import FlatTensor
from .draft import NGramDraftModel
from .rag import PurePythonBM25RAG, HybridDenseBM25RAG
from .tokenizer import BPETokenizer
from .model import ModernMicroGPT
from .medusa import MedusaDrafter, MedusaHead
from .cache import PagedKVCache, PagedKVCacheManager, BlockAllocator
from .batching import ContinuousBatchScheduler, InferenceRequest
from .training import get_cosine_lr, build_training_sequences, continual_learn_lora_turn, train_dpo_turn, train_kto_turn
from .gradcheck import run_gradient_checks
from .cli import main

__version__ = "0.3.0"

__all__ = [
    "FlatTensor",
    "NGramDraftModel",
    "PurePythonBM25RAG",
    "HybridDenseBM25RAG",
    "BPETokenizer",
    "ModernMicroGPT",
    "MedusaDrafter",
    "MedusaHead",
    "PagedKVCache",
    "PagedKVCacheManager",
    "BlockAllocator",
    "ContinuousBatchScheduler",
    "InferenceRequest",
    "get_cosine_lr",
    "build_training_sequences",
    "continual_learn_lora_turn",
    "train_dpo_turn",
    "train_kto_turn",
    "run_gradient_checks",
    "main",
    "__version__",
]

from .pretraining import TrainingConfig, TrainingState, StepMetric, TrainingResult, fit, evaluate, parameter_count, reset_optimizer
from .training_checkpoint import save_training_checkpoint, load_training_checkpoint

__all__ += ["TrainingConfig", "TrainingState", "StepMetric", "TrainingResult", "fit", "evaluate", "parameter_count", "reset_optimizer", "save_training_checkpoint", "load_training_checkpoint"]

from .pretraining import token_sequences
from .training_checkpoint import validate_model_config
__all__ += ["token_sequences", "validate_model_config"]
