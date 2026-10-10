"""Unit tests for FileStorageBackend."""

import os
import numpy as np
import pytest

from ai_ssd.storage.file_backend import FileStorageBackend


def test_file_backend_write_and_read():
    backend = FileStorageBackend(num_layers=2, num_heads=2, tokens_per_block=16, head_dim=64)
    try:
        k_data = np.random.randn(16, 2, 64).astype(np.float32)
        v_data = np.random.randn(16, 2, 64).astype(np.float32)

        backend.write_block(layer_idx=0, block_id=1, k_block=k_data, v_block=v_data)

        read_k = backend.read_key_page(layer_idx=0, block_id=1)
        read_v = backend.read_value_page(layer_idx=0, block_id=1)

        np.testing.assert_allclose(read_k, k_data, atol=1e-5)
        np.testing.assert_allclose(read_v, v_data, atol=1e-5)

        # Batch read
        batch = backend.read_block_batch(layer_idx=0, block_ids=[1])
        assert 1 in batch
        np.testing.assert_allclose(batch[1][0], k_data, atol=1e-5)
        np.testing.assert_allclose(batch[1][1], v_data, atol=1e-5)
    finally:
        backend.close()


def test_file_backend_topk_filtering():
    backend = FileStorageBackend(num_layers=1, num_heads=2, tokens_per_block=16, head_dim=64)
    try:
        cand_bids = []
        for bid in range(10):
            k_data = np.random.randn(16, 2, 64).astype(np.float32)
            v_data = np.random.randn(16, 2, 64).astype(np.float32)
            backend.write_block(layer_idx=0, block_id=bid, k_block=k_data, v_block=v_data)
            cand_bids.append((bid, 16))

        query = np.random.randn(14, 64).astype(np.float32)
        top_k = 3

        results = backend.compute_topk_filter(
            layer_idx=0,
            cand_bids=cand_bids,
            query=query,
            top_k=top_k,
            scale=1.0 / 8.0,
            q_heads=14,
            kv_heads=2,
            head_dim=64,
        )

        assert len(results) == top_k
        assert all(isinstance(score, float) for score, _, _ in results)
        # Verify scores are sorted descending
        for i in range(len(results) - 1):
            assert results[i][0] >= results[i + 1][0]
    finally:
        backend.close()
