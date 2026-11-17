"""Medusa multi-head auxiliary speculative drafter.

Implements Cai et al. (2024) Medusa speculative decoding architecture:
Lightweight auxiliary heads predict multiple future tokens directly from
the final transformer hidden states without extra transformer backbone layers.
"""

from array import array
import math
import random

from .tensor import FlatTensor


class MedusaHead:
    """A single linear prediction head projecting hidden states to vocabulary logits."""

    __slots__ = ("head_idx", "w", "b")

    def __init__(self, head_idx: int, n_embd: int, vocab_size: int):
        self.head_idx = head_idx
        # Lightweight linear projection from hidden dimension to vocabulary
        self.w = FlatTensor.randn(n_embd, vocab_size, std=0.02, requires_grad=True)
        self.b = FlatTensor.zeros(1, vocab_size, requires_grad=True)

    def forward(self, h: FlatTensor) -> FlatTensor:
        """Projects hidden states h [T, D] -> logits [T, V]."""
        return h.matmul(self.w) + self.b

    def to_dict(self) -> dict:
        return {
            "head_idx": self.head_idx,
            "w": self.w.to_int8_compressed_dict(),
            "b": self.b.to_int8_compressed_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MedusaHead":
        w = FlatTensor.from_serialized(d["w"])
        b = FlatTensor.from_serialized(d["b"])
        head = cls.__new__(cls)
        head.head_idx = d["head_idx"]
        head.w = w
        head.b = b
        return head


class MedusaDrafter:
    """
    Multi-head speculative drafter holding K auxiliary heads.
    Head k predicts token (t + 2 + k), where base model head predicts (t + 1).
    """

    def __init__(self, n_embd: int, vocab_size: int, num_heads: int = 3):
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.heads = [MedusaHead(k, n_embd, vocab_size) for k in range(num_heads)]

    def forward(self, h: FlatTensor) -> list[FlatTensor]:
        """Runs all Medusa heads on hidden states h [T, D]."""
        return [head.forward(h) for head in self.heads]

    def draft(self, h_last: FlatTensor, base_token: int | None = None) -> list[int]:
        """
        Drafts multiple future tokens from the last hidden state h_last [1, D].
        Returns candidate token sequence [base_token, medusa_0, medusa_1, ...].
        """
        candidates = []
        if base_token is not None:
            candidates.append(base_token)

        V = self.vocab_size
        for head in self.heads:
            logits = head.forward(h_last)
            data = logits.data
            best_tok = 0
            best_val = -float('inf')
            for v in range(V):
                val = data[v]
                if val > best_val:
                    best_val = val
                    best_tok = v
            candidates.append(best_tok)

        return candidates

    def train_step(self, h_seq: FlatTensor, target_ids: list[int], lr: float = 0.01) -> float:
        """
        Trains all Medusa heads on hidden states h_seq [T, D] against shifted targets.
        Head k is trained to predict target_ids[t + 1 + (k + 1)].
        """
        T = h_seq.rows
        total_loss = 0.0
        heads_trained = 0

        for k, head in enumerate(self.heads):
            # Offset: head 0 predicts t + 2, head 1 predicts t + 3, etc.
            offset = k + 2
            valid_T = T - offset
            if valid_T <= 0:
                continue

            h_sub = FlatTensor(valid_T, self.n_embd, array('d', h_seq.data[: valid_T * self.n_embd]))
            logits = head.forward(h_sub)
            targets = target_ids[offset : offset + valid_T]

            loss = FlatTensor.cross_entropy(logits, targets)
            loss.backward()

            # SGD step with gradient zeroing on head parameters
            for p in (head.w, head.b):
                p_data, p_grad = p.data, p.grad
                for i in range(len(p_data)):
                    p_data[i] -= lr * p_grad[i]
                    p_grad[i] = 0.0

            total_loss += loss.data[0]
            heads_trained += 1

        return total_loss / max(1, heads_trained)

    def to_dict(self) -> dict:
        return {
            "n_embd": self.n_embd,
            "vocab_size": self.vocab_size,
            "num_heads": self.num_heads,
            "heads": [head.to_dict() for head in self.heads],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MedusaDrafter":
        drafter = cls.__new__(cls)
        drafter.n_embd = d["n_embd"]
        drafter.vocab_size = d["vocab_size"]
        drafter.num_heads = d["num_heads"]
        drafter.heads = [MedusaHead.from_dict(h) for h in d["heads"]]
        return drafter
