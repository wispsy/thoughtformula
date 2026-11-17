"""Unit tests for ContinuousBatchScheduler and multi-request concurrent inference."""

import unittest

from thoughtformula.model import ModernMicroGPT
from thoughtformula.batching import ContinuousBatchScheduler, InferenceRequest


class TestContinuousBatching(unittest.TestCase):
    def setUp(self):
        self.model = ModernMicroGPT(
            vocab_size=16,
            n_embd=8,
            n_heads=2,
            n_kv_heads=1,
            n_layers=1,
            n_experts=2,
            moe_top_k=1,
            expert_hidden=8,
        )

    def test_single_request_lifecycle(self):
        scheduler = ContinuousBatchScheduler(self.model, max_batch_size=2)
        req = scheduler.add_request([1, 2, 3], request_id="r1", max_new_tokens=4, temperature=0.0)
        self.assertEqual(req.status if hasattr(req, "status") else req.request_id, "r1")
        self.assertTrue(scheduler.has_work())

        results = scheduler.run_all()
        self.assertIn("r1", results)
        self.assertEqual(len(results["r1"]), 4)
        self.assertFalse(scheduler.has_work())

    def test_multi_request_concurrency(self):
        scheduler = ContinuousBatchScheduler(self.model, max_batch_size=3)
        scheduler.add_request([1, 2], request_id="r1", max_new_tokens=3, temperature=0.0)
        scheduler.add_request([3, 4, 5], request_id="r2", max_new_tokens=5, temperature=0.0)
        scheduler.add_request([2], request_id="r3", max_new_tokens=2, temperature=0.0)

        results = scheduler.run_all()
        self.assertEqual(len(results), 3)
        self.assertEqual(len(results["r1"]), 3)
        self.assertEqual(len(results["r2"]), 5)
        self.assertEqual(len(results["r3"]), 2)

    def test_dynamic_admission_on_the_fly(self):
        scheduler = ContinuousBatchScheduler(self.model, max_batch_size=2)
        scheduler.add_request([1, 2], request_id="r1", max_new_tokens=4, temperature=0.0)

        # Step twice so r1 is mid-flight in decode phase
        scheduler.step()
        scheduler.step()

        # Enqueue r2 mid-flight
        scheduler.add_request([3, 4], request_id="r2", max_new_tokens=3, temperature=0.0)

        results = scheduler.run_all()
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results["r1"]), 4)
        self.assertEqual(len(results["r2"]), 3)

    def test_streaming_generator(self):
        scheduler = ContinuousBatchScheduler(self.model, max_batch_size=2)
        scheduler.add_request([1, 2], request_id="r1", max_new_tokens=3, temperature=0.0)
        scheduler.add_request([3, 4], request_id="r2", max_new_tokens=2, temperature=0.0)

        step_counts = 0
        for step_out in scheduler.stream():
            self.assertIsInstance(step_out, dict)
            step_counts += 1

        self.assertGreaterEqual(step_counts, 3)
        self.assertFalse(scheduler.has_work())

    def test_batched_vs_independent_equivalence(self):
        prompt1 = [1, 2, 3]
        prompt2 = [4, 5]

        # Independent generation
        ind_tokens1 = self.model.generate(prompt1, max_new_tokens=4, temperature=0.0)[len(prompt1):]
        ind_tokens2 = self.model.generate(prompt2, max_new_tokens=4, temperature=0.0)[len(prompt2):]

        # Continuous batching generation
        scheduler = ContinuousBatchScheduler(self.model, max_batch_size=2, use_paged_cache=False)
        scheduler.add_request(prompt1, request_id="req1", max_new_tokens=4, temperature=0.0)
        scheduler.add_request(prompt2, request_id="req2", max_new_tokens=4, temperature=0.0)
        batched_results = scheduler.run_all()

        self.assertEqual(batched_results["req1"], ind_tokens1)
        self.assertEqual(batched_results["req2"], ind_tokens2)


if __name__ == "__main__":
    unittest.main()
