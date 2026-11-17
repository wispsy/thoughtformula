"""Pure-Python continuous batching scheduler for concurrent multi-request inference.

Implements iteration-level scheduling (Orca / vLLM style):
- Packs concurrent active decode tokens into unified forward passes.
- Dynamically admits pending requests as active requests finish.
- Isolates attention memory per sequence KV-cache while amortizing linear weight projections.
"""

from array import array
import uuid
from typing import Generator

from .model import ModernMicroGPT


class InferenceRequest:
    """Represents a single generation request managed by the continuous batcher."""

    __slots__ = (
        "request_id",
        "prompt_ids",
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "stop_token_id",
        "use_int8",
        "kv_cache",
        "output_tokens",
        "is_finished",
        "prefilled",
    )

    def __init__(
        self,
        prompt_ids: list[int],
        request_id: str | int | None = None,
        max_new_tokens: int = 24,
        temperature: float = 0.05,
        top_k: int = 6,
        top_p: float = 0.9,
        stop_token_id: int | None = None,
        use_int8: bool = True,
        kv_cache=None,
    ):
        self.request_id = str(uuid.uuid4())[:8] if request_id is None else request_id
        self.prompt_ids = list(prompt_ids)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.stop_token_id = stop_token_id
        self.use_int8 = use_int8
        self.kv_cache = kv_cache
        self.output_tokens: list[int] = []
        self.is_finished = False
        self.prefilled = False


class ContinuousBatchScheduler:
    """
    Synchronous continuous batching scheduler:
    Packs decode tokens from up to max_batch_size active requests into single forward passes,
    admitting new requests on the fly without waiting for previous requests to finish.
    """

    def __init__(
        self,
        model: ModernMicroGPT,
        max_batch_size: int = 4,
        use_paged_cache: bool = True,
        use_int8: bool = True,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.use_paged_cache = use_paged_cache
        self.use_int8 = use_int8
        self.pending_queue: list[InferenceRequest] = []
        self.active_requests: list[InferenceRequest] = []
        self.completed_requests: dict[str | int, list[int]] = {}

    def add_request(
        self,
        prompt_ids: list[int],
        request_id: str | int | None = None,
        max_new_tokens: int = 24,
        temperature: float = 0.05,
        top_k: int = 6,
        top_p: float = 0.9,
        stop_token_id: int | None = None,
    ) -> InferenceRequest:
        """Enqueues a new generation request into the pending queue."""
        cache = self.model.init_kv_cache(use_paged=self.use_paged_cache)
        req = InferenceRequest(
            prompt_ids=prompt_ids,
            request_id=request_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token_id=stop_token_id,
            use_int8=self.use_int8,
            kv_cache=cache,
        )
        self.pending_queue.append(req)
        return req

    def has_work(self) -> bool:
        """Returns True if there are pending or active requests being processed."""
        return bool(self.pending_queue or self.active_requests)

    def step(self) -> dict[str | int, int]:
        """
        Executes one continuous batching iteration:
        1. Admits pending requests up to max_batch_size.
        2. Runs prompt prefill for newly admitted requests.
        3. Packs active decode requests into a single forward pass.
        4. Evicts finished requests and returns {request_id: token_id} for this step.
        """
        if not self.has_work():
            return {}

        step_outputs: dict[str | int, int] = {}
        V = self.model.vocab_size

        # 1. Admit pending requests into active batch
        while len(self.active_requests) < self.max_batch_size and self.pending_queue:
            admitted = self.pending_queue.pop(0)
            self.active_requests.append(admitted)

        # 2. Prefill newly admitted requests (1 request at a time to populate its KV cache)
        decode_candidates: list[InferenceRequest] = []
        for req in self.active_requests:
            if not req.prefilled:
                if not req.prompt_ids:
                    req.is_finished = True
                    continue
                logits = self.model.forward(
                    req.prompt_ids,
                    start_pos=0,
                    kv_caches=req.kv_cache,
                    use_int8=req.use_int8,
                )
                last_row = logits.data[(logits.rows - 1) * V : logits.rows * V]
                next_id = self.model.sample_top_k_top_p(
                    last_row, req.temperature, req.top_k, req.top_p
                )
                req.output_tokens.append(next_id)
                req.prefilled = True
                step_outputs[req.request_id] = next_id

                if (req.stop_token_id is not None and next_id == req.stop_token_id) or len(
                    req.output_tokens
                ) >= req.max_new_tokens:
                    req.is_finished = True
            else:
                decode_candidates.append(req)

        # 3. Packed continuous decode forward pass for active requests
        if decode_candidates:
            tokens = [req.output_tokens[-1] for req in decode_candidates]
            caches = [req.kv_cache for req in decode_candidates]
            positions = [
                req.kv_cache.seq_len
                if getattr(req.kv_cache, "is_paged", False)
                else req.kv_cache[0]["seq_len"]
                for req in decode_candidates
            ]

            batched_logits = self.model.forward_batched(
                tokens, caches, positions, use_int8=self.use_int8
            )

            for idx, req in enumerate(decode_candidates):
                row = batched_logits.data[idx * V : (idx + 1) * V]
                next_id = self.model.sample_top_k_top_p(
                    row, req.temperature, req.top_k, req.top_p
                )
                req.output_tokens.append(next_id)
                step_outputs[req.request_id] = next_id

                if (req.stop_token_id is not None and next_id == req.stop_token_id) or len(
                    req.output_tokens
                ) >= req.max_new_tokens:
                    req.is_finished = True

        # 4. Evict finished requests and free their KV-caches
        remaining_active = []
        for req in self.active_requests:
            if req.is_finished:
                self.completed_requests[req.request_id] = list(req.output_tokens)
                if hasattr(req.kv_cache, "release"):
                    req.kv_cache.release()
            else:
                remaining_active.append(req)
        self.active_requests = remaining_active

        return step_outputs

    def run_all(self) -> dict[str | int, list[int]]:
        """Runs the continuous batcher to completion across all enqueued requests."""
        while self.has_work():
            self.step()
        return self.completed_requests

    def stream(self) -> Generator[dict[str | int, int], None, None]:
        """Yields step_outputs dict at each decoding iteration until all requests finish."""
        while self.has_work():
            outputs = self.step()
            if outputs:
                yield outputs

    def stats(self) -> dict:
        """Returns scheduler queues and active batch telemetry."""
        return {
            "pending_count": len(self.pending_queue),
            "active_count": len(self.active_requests),
            "completed_count": len(self.completed_requests),
            "max_batch_size": self.max_batch_size,
        }
