"""Unit tests for Paged KV-cache memory manager."""

from array import array
import unittest

from thoughtformula.cache import (
    BlockAllocator,
    PagedLayerCache,
    PagedKVCache,
    PagedKVCacheManager,
)
from thoughtformula.model import ModernMicroGPT


class TestPagedKVCache(unittest.TestCase):
    def test_block_allocator_pool_and_reuse(self):
        allocator = BlockAllocator(page_size=4, kv_dim=8, initial_blocks=2)
        self.assertEqual(allocator.free_blocks_count, 2)
        self.assertEqual(allocator.total_blocks_count, 2)

        b1 = allocator.allocate()
        b2 = allocator.allocate()
        self.assertEqual(allocator.active_blocks_count, 2)
        self.assertEqual(allocator.free_blocks_count, 0)

        # Allocate beyond initial pool -> auto grows
        b3 = allocator.allocate()
        self.assertEqual(allocator.total_blocks_count, 3)

        allocator.release(b1.block_id)
        self.assertEqual(allocator.free_blocks_count, 1)

        b_reused = allocator.allocate()
        self.assertEqual(b_reused.block_id, b1.block_id)

    def test_paged_layer_cache_append_and_pages(self):
        allocator = BlockAllocator(page_size=4, kv_dim=4, initial_blocks=8)
        layer = PagedLayerCache(layer_idx=0, page_size=4, kv_dim=4, allocator=allocator)

        # Append 10 tokens (should span 3 pages: 4 + 4 + 2)
        for i in range(10):
            k = [float(i)] * 4
            v = [float(i * 2)] * 4
            layer.append_token(k, v)

        self.assertEqual(layer.seq_len, 10)
        self.assertEqual(len(layer.page_table), 3)

        b0 = layer.get_block(0)
        self.assertEqual(b0.num_tokens, 4)
        b1 = layer.get_block(1)
        self.assertEqual(b1.num_tokens, 4)
        b2 = layer.get_block(2)
        self.assertEqual(b2.num_tokens, 2)

        # Truncate to 6 tokens (should drop page 2, keep 2 pages: 4 + 2)
        layer.truncate(6)
        self.assertEqual(layer.seq_len, 6)
        self.assertEqual(len(layer.page_table), 2)
        self.assertEqual(layer.get_block(1).num_tokens, 2)

        layer.release()
        self.assertEqual(layer.seq_len, 0)
        self.assertEqual(len(layer.page_table), 0)

    def test_prefix_sharing_and_copy_on_write(self):
        manager = PagedKVCacheManager(n_layers=2, kv_dim=4, page_size=4, initial_blocks=16)
        parent = manager.create_cache()

        # Append 4 tokens to parent (fills 1 block per layer)
        for i in range(4):
            for l in range(parent.n_layers):
                parent[l].append_token([1.0] * 4, [2.0] * 4)

        b0_id = parent[0].page_table[0]
        self.assertEqual(manager.allocator.blocks[b0_id].ref_count, 1)

        # Fork child from parent (prefix sharing)
        child = manager.fork_cache(parent)
        self.assertEqual(manager.allocator.blocks[b0_id].ref_count, 2)
        self.assertEqual(child[0].page_table[0], b0_id)

        stats = manager.stats()
        self.assertGreater(stats["sharing_ratio"], 1.0)
        self.assertEqual(stats["saved_blocks_via_sharing"], 2)

        # Append to child should trigger CoW if mutating shared block, or allocate new block when full
        for l in range(child.n_layers):
            child[l].append_token([5.0] * 4, [6.0] * 4)

        # Page 0 is still shared full block
        self.assertEqual(parent[0].page_table[0], b0_id)
        self.assertEqual(child[0].page_table[0], b0_id)
        # Page 1 is unique to child
        self.assertEqual(len(child[0].page_table), 2)
        self.assertEqual(len(parent[0].page_table), 1)

        manager.free_cache(child)
        self.assertEqual(manager.allocator.blocks[b0_id].ref_count, 1)
        manager.free_cache(parent)
        self.assertEqual(manager.allocator.blocks[b0_id].ref_count, 0)

    def test_model_autoregressive_paged_kv_cache_equivalence(self):
        model = ModernMicroGPT(vocab_size=16, n_embd=8, n_heads=2, n_kv_heads=1, n_layers=2)

        prompt = [1, 2, 3]
        # Run prompt on standard cache
        std_cache = model.init_kv_cache(use_paged=False)
        out_std = model.forward(prompt, kv_caches=std_cache)

        # Run prompt on paged cache
        paged_cache = model.init_kv_cache(use_paged=True, page_size=2)
        out_paged = model.forward(prompt, kv_caches=paged_cache)

        # Check prompt logits match
        max_diff = max(abs(a - b) for a, b in zip(out_std.data, out_paged.data))
        self.assertLess(max_diff, 1e-12)

        # Next token autoregressive step
        next_tok = [4]
        out_std_next = model.forward(next_tok, start_pos=3, kv_caches=std_cache)
        out_paged_next = model.forward(next_tok, start_pos=3, kv_caches=paged_cache)

        max_next_diff = max(abs(a - b) for a, b in zip(out_std_next.data, out_paged_next.data))
        self.assertLess(max_next_diff, 1e-12)


if __name__ == "__main__":
    unittest.main()
