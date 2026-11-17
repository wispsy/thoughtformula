"""Paged KV-cache memory manager for non-contiguous multi-turn allocation.

Implements vLLM-style paged attention memory management:
- PhysicalBlock: Fixed-size page holding quantized key/value states and scales.
- BlockAllocator: Global pool for physical block allocation and reference counting.
- PagedLayerCache: Logical page table mapping sequence positions to physical blocks.
- PagedKVCache: Multi-layer sequence container supporting O(1) prefix forks and copy-on-write.
- PagedKVCacheManager: High-level coordinator for multi-turn request allocation.
"""

from array import array
import math


class PhysicalBlock:
    """Fixed-size physical block (page) storing quantized KV token slots."""

    __slots__ = (
        "block_id",
        "page_size",
        "kv_dim",
        "ref_count",
        "num_tokens",
        "k_int8",
        "v_int8",
        "k_scale",
        "v_scale",
    )

    def __init__(self, block_id: int, page_size: int, kv_dim: int):
        self.block_id = block_id
        self.page_size = page_size
        self.kv_dim = kv_dim
        self.ref_count = 1
        self.num_tokens = 0
        self.k_int8 = array('b', [0] * (page_size * kv_dim))
        self.v_int8 = array('b', [0] * (page_size * kv_dim))
        self.k_scale = array('d', [0.0] * page_size)
        self.v_scale = array('d', [0.0] * page_size)

    def is_full(self) -> bool:
        return self.num_tokens >= self.page_size

    def append_slot(self, k_quant, v_quant, sk: float, sv: float):
        """Appends a single token's KV quant data and scale factors into the block."""
        idx = self.num_tokens
        base = idx * self.kv_dim
        for d in range(self.kv_dim):
            self.k_int8[base + d] = k_quant[d]
            self.v_int8[base + d] = v_quant[d]
        self.k_scale[idx] = sk
        self.v_scale[idx] = sv
        self.num_tokens += 1

    def copy(self, new_block_id: int) -> "PhysicalBlock":
        """Creates a deep copy of this physical block with ref_count=1 (Copy-On-Write)."""
        copied = PhysicalBlock(new_block_id, self.page_size, self.kv_dim)
        copied.num_tokens = self.num_tokens
        copied.k_int8 = array('b', self.k_int8)
        copied.v_int8 = array('b', self.v_int8)
        copied.k_scale = array('d', self.k_scale)
        copied.v_scale = array('d', self.v_scale)
        return copied

    def reset(self):
        """Resets block state to be reused in the allocator pool."""
        self.ref_count = 1
        self.num_tokens = 0


class BlockAllocator:
    """Manages a pool of physical blocks with reference counting and reclamation."""

    def __init__(self, page_size: int = 16, kv_dim: int = 24, initial_blocks: int = 64):
        self.page_size = page_size
        self.kv_dim = kv_dim
        self.blocks: dict[int, PhysicalBlock] = {}
        self.free_pool: list[int] = []
        self._next_id = 0

        for _ in range(initial_blocks):
            self._grow()

    def _grow(self) -> int:
        bid = self._next_id
        self._next_id += 1
        block = PhysicalBlock(bid, self.page_size, self.kv_dim)
        self.blocks[bid] = block
        self.free_pool.append(bid)
        return bid

    def allocate(self) -> PhysicalBlock:
        """Allocates a free block, growing the pool if empty."""
        if not self.free_pool:
            self._grow()
        bid = self.free_pool.pop()
        block = self.blocks[bid]
        block.reset()
        return block

    def retain(self, block_id: int):
        """Increments block reference count for shared prefix pages."""
        self.blocks[block_id].ref_count += 1

    def release(self, block_id: int):
        """Decrements reference count and returns block to free pool if ref_count == 0."""
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.ref_count = 0
            self.free_pool.append(block_id)

    @property
    def free_blocks_count(self) -> int:
        return len(self.free_pool)

    @property
    def total_blocks_count(self) -> int:
        return self._next_id

    @property
    def active_blocks_count(self) -> int:
        return self._next_id - len(self.free_pool)


class PagedLayerCache:
    """Logical page table for one transformer layer of an active sequence."""

    __slots__ = (
        "layer_idx",
        "page_size",
        "kv_dim",
        "allocator",
        "page_table",
        "seq_len",
        "is_paged",
    )

    def __init__(self, layer_idx: int, page_size: int, kv_dim: int, allocator: BlockAllocator):
        self.layer_idx = layer_idx
        self.page_size = page_size
        self.kv_dim = kv_dim
        self.allocator = allocator
        self.page_table: list[int] = []
        self.seq_len = 0
        self.is_paged = True

    def get_block(self, page_idx: int) -> PhysicalBlock:
        return self.allocator.blocks[self.page_table[page_idx]]

    def append_token(self, k_slice, v_slice):
        """Quantizes token KV to Int8 and appends to the current tail page with CoW support."""
        kv_d = self.kv_dim

        # Compute int8 dynamic scales
        max_k = 0.0
        max_v = 0.0
        for i in range(kv_d):
            ak = abs(k_slice[i])
            av = abs(v_slice[i])
            if ak > max_k:
                max_k = ak
            if av > max_v:
                max_v = av

        sk = max_k / 127.0 if max_k > 1e-12 else 1.0
        sv = max_v / 127.0 if max_v > 1e-12 else 1.0
        inv_sk = 127.0 / max_k if max_k > 1e-12 else 0.0
        inv_sv = 127.0 / max_v if max_v > 1e-12 else 0.0

        k_quant = [max(-127, min(127, int(round(k_slice[i] * inv_sk)))) for i in range(kv_d)]
        v_quant = [max(-127, min(127, int(round(v_slice[i] * inv_sv)))) for i in range(kv_d)]

        # Find or allocate active tail block
        if not self.page_table or self.get_block(len(self.page_table) - 1).is_full():
            new_block = self.allocator.allocate()
            self.page_table.append(new_block.block_id)
            tail_block = new_block
        else:
            tail_block = self.get_block(len(self.page_table) - 1)
            # Copy-on-Write: If block is shared with another sequence, clone it
            if tail_block.ref_count > 1:
                cloned = tail_block.copy(self.allocator.allocate().block_id)
                self.allocator.release(tail_block.block_id)
                self.page_table[-1] = cloned.block_id
                tail_block = cloned

        tail_block.append_slot(k_quant, v_quant, sk, sv)
        self.seq_len += 1

    def truncate(self, target_len: int):
        """Truncates layer sequence to target_len and reclaims unused trailing blocks."""
        if self.seq_len <= target_len:
            return

        needed_pages = (target_len + self.page_size - 1) // self.page_size if target_len > 0 else 0
        while len(self.page_table) > needed_pages:
            bid = self.page_table.pop()
            self.allocator.release(bid)

        if needed_pages > 0:
            tail = self.get_block(needed_pages - 1)
            # Copy on write if shared
            if tail.ref_count > 1:
                cloned = tail.copy(self.allocator.allocate().block_id)
                self.allocator.release(tail.block_id)
                self.page_table[-1] = cloned.block_id
                tail = cloned
            rem = target_len % self.page_size
            tail.num_tokens = rem if rem > 0 else self.page_size

        self.seq_len = target_len

    def fork(self) -> "PagedLayerCache":
        """Forks this layer cache, sharing all physical blocks via reference counting."""
        forked = PagedLayerCache(self.layer_idx, self.page_size, self.kv_dim, self.allocator)
        forked.page_table = list(self.page_table)
        forked.seq_len = self.seq_len
        for bid in self.page_table:
            self.allocator.retain(bid)
        return forked

    def release(self):
        """Releases all blocks allocated by this layer cache."""
        for bid in self.page_table:
            self.allocator.release(bid)
        self.page_table.clear()
        self.seq_len = 0


class PagedKVCache:
    """Full multi-layer paged KV-cache container for a single sequence."""

    __slots__ = ("n_layers", "page_size", "kv_dim", "allocator", "layers", "is_paged")

    def __init__(self, n_layers: int, page_size: int, kv_dim: int, allocator: BlockAllocator):
        self.n_layers = n_layers
        self.page_size = page_size
        self.kv_dim = kv_dim
        self.allocator = allocator
        self.layers = [
            PagedLayerCache(l, page_size, kv_dim, allocator) for l in range(n_layers)
        ]
        self.is_paged = True

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len if self.layers else 0

    def __getitem__(self, idx: int) -> PagedLayerCache:
        return self.layers[idx]

    def __len__(self) -> int:
        return self.n_layers

    def fork(self) -> "PagedKVCache":
        """O(1) prefix sharing: forks the entire multi-layer cache without copying buffers."""
        forked = PagedKVCache(self.n_layers, self.page_size, self.kv_dim, self.allocator)
        forked.layers = [layer.fork() for layer in self.layers]
        return forked

    def truncate(self, target_len: int):
        """Truncates all layer caches to target_len."""
        for layer in self.layers:
            layer.truncate(target_len)

    def release(self):
        """Releases all physical memory held by this cache."""
        for layer in self.layers:
            layer.release()


class PagedKVCacheManager:
    """High-level coordinator managing block allocation and sequence caches."""

    def __init__(self, n_layers: int, kv_dim: int, page_size: int = 16, initial_blocks: int = 64):
        self.n_layers = n_layers
        self.kv_dim = kv_dim
        self.page_size = page_size
        self.allocator = BlockAllocator(page_size=page_size, kv_dim=kv_dim, initial_blocks=initial_blocks)
        self.active_caches: list[PagedKVCache] = []

    def create_cache(self) -> PagedKVCache:
        cache = PagedKVCache(self.n_layers, self.page_size, self.kv_dim, self.allocator)
        self.active_caches.append(cache)
        return cache

    def fork_cache(self, cache: PagedKVCache) -> PagedKVCache:
        forked = cache.fork()
        self.active_caches.append(forked)
        return forked

    def free_cache(self, cache: PagedKVCache):
        if cache in self.active_caches:
            self.active_caches.remove(cache)
        cache.release()

    def stats(self) -> dict:
        """Returns memory manager statistics including prefix sharing deduplication."""
        total_tokens = sum(c.seq_len for c in self.active_caches)
        unpaged_blocks_needed = sum(
            math.ceil(c.seq_len / self.page_size) * self.n_layers for c in self.active_caches
        )
        active_blocks = self.allocator.active_blocks_count
        saved_blocks = max(0, unpaged_blocks_needed - active_blocks)
        sharing_ratio = (
            unpaged_blocks_needed / active_blocks if active_blocks > 0 else 1.0
        )
        return {
            "page_size": self.page_size,
            "total_sequences": len(self.active_caches),
            "total_tokens": total_tokens,
            "total_blocks": self.allocator.total_blocks_count,
            "free_blocks": self.allocator.free_blocks_count,
            "active_blocks": active_blocks,
            "saved_blocks_via_sharing": saved_blocks,
            "sharing_ratio": sharing_ratio,
        }
