"""Asynchronous DRAM Staging Buffer and Speculative Layer Prefetcher."""

from __future__ import annotations
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import threading
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from ai_ssd.storage.base import BaseStorageBackend


class PrefetchStorageAdapter(BaseStorageBackend):
    """
    Prefetch adapter wrapping a BaseStorageBackend (FileStorageBackend or NvmeStorageBackend).
    
    Maintains a bounded LRU staging buffer in Host DRAM.
    Overlaps host compute with speculative background retrieval of layer L+1 blocks.
    """

    def __init__(
        self,
        storage_backend: BaseStorageBackend,
        buffer_capacity_blocks: int = 512,
        num_layers: int = 36,
        enable_async_pipeline: bool = True,
        max_async_workers: int = 2,
    ):
        self.storage_backend = storage_backend
        self.buffer_capacity_blocks = buffer_capacity_blocks
        self.num_layers = num_layers
        self.enable_async_pipeline = enable_async_pipeline
        self.max_async_workers = max_async_workers

        self._staging_buffer: OrderedDict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._pending_futures: Dict[Tuple[int, int], Future[Tuple[np.ndarray, np.ndarray]]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_async_workers) if enable_async_pipeline else None

        # Telemetry
        self.demand_requests: int = 0
        self.prefetch_requests: int = 0
        self.prefetch_hits: int = 0
        self.prefetch_misses: int = 0

    @property
    def CLASSIFICATION(self) -> str:
        return getattr(self.storage_backend, "CLASSIFICATION", "ANALYTICAL")

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self.storage_backend.close()

    def write_block(
        self,
        layer_idx: int,
        block_id: int,
        k_block: np.ndarray,
        v_block: np.ndarray,
        head_id: int = 0,
    ) -> None:
        self.storage_backend.write_block(layer_idx, block_id, k_block, v_block, head_id=head_id)

    def read_key_page(self, layer_idx: int, block_id: int) -> np.ndarray:
        return self._fetch_block(layer_idx, block_id)[0]

    def read_value_page(self, layer_idx: int, block_id: int) -> np.ndarray:
        return self._fetch_block(layer_idx, block_id)[1]

    def read_block_batch(
        self,
        layer_idx: int,
        block_ids: List[int],
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        res = {}
        missing_bids = []

        with self._lock:
            for bid in block_ids:
                key = (layer_idx, bid)
                if key in self._staging_buffer:
                    res[bid] = self._staging_buffer[key]
                    self._staging_buffer.move_to_end(key)
                    self.prefetch_hits += 1
                else:
                    missing_bids.append(bid)
                    self.prefetch_misses += 1

        if missing_bids:
            fetched = self.storage_backend.read_block_batch(layer_idx, missing_bids)
            with self._lock:
                for bid, block_pair in fetched.items():
                    res[bid] = block_pair
                    self._stage_block(layer_idx, bid, block_pair)

        self.demand_requests += len(block_ids)
        return res

    def compute_topk_filter(
        self,
        layer_idx: int,
        cand_bids: List[Tuple[int, int]],
        query: np.ndarray,
        top_k: int,
        scale: float,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> List[Tuple[float, int, int]]:
        return self.storage_backend.compute_topk_filter(
            layer_idx=layer_idx,
            cand_bids=cand_bids,
            query=query,
            top_k=top_k,
            scale=scale,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )

    def predict_and_prefetch(
        self,
        current_layer_id: int,
        current_block_ids: List[int],
        async_mode: bool = True,
    ) -> None:
        if not self.enable_async_pipeline or self._executor is None or not current_block_ids:
            return

        next_layer_id = (current_layer_id + 1) % self.num_layers
        bids_to_prefetch = []

        with self._lock:
            for bid in current_block_ids:
                key = (next_layer_id, bid)
                if key not in self._staging_buffer and key not in self._pending_futures:
                    bids_to_prefetch.append(bid)

        if not bids_to_prefetch:
            return

        self.prefetch_requests += len(bids_to_prefetch)

        def _bg_fetch(layer: int, bids: List[int]):
            try:
                blocks = self.storage_backend.read_block_batch(layer, bids)
                with self._lock:
                    for b_id, pair in blocks.items():
                        self._stage_block(layer, b_id, pair)
            except Exception:
                pass

        self._executor.submit(_bg_fetch, next_layer_id, bids_to_prefetch)

    def _fetch_block(self, layer_idx: int, block_id: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (layer_idx, block_id)
        with self._lock:
            if key in self._staging_buffer:
                self.prefetch_hits += 1
                self._staging_buffer.move_to_end(key)
                return self._staging_buffer[key]
            self.prefetch_misses += 1

        k = self.storage_backend.read_key_page(layer_idx, block_id)
        v = self.storage_backend.read_value_page(layer_idx, block_id)
        pair = (k, v)
        with self._lock:
            self._stage_block(layer_idx, block_id, pair)
        return pair

    def _stage_block(self, layer_idx: int, block_id: int, pair: Tuple[np.ndarray, np.ndarray]) -> None:
        key = (layer_idx, block_id)
        self._staging_buffer[key] = pair
        if len(self._staging_buffer) > self.buffer_capacity_blocks:
            self._staging_buffer.popitem(last=False)

    def reset_stats(self) -> None:
        self.storage_backend.reset_stats()
        self.demand_requests = 0
        self.prefetch_requests = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0

    def get_telemetry(self) -> Dict[str, Any]:
        tel = self.storage_backend.get_telemetry()
        total_access = max(1, self.prefetch_hits + self.prefetch_misses)
        tel.update({
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "prefetch_hit_rate_pct": round((self.prefetch_hits / total_access) * 100.0, 2),
            "demand_requests": self.demand_requests,
            "staged_blocks": len(self._staging_buffer),
        })
        return tel
