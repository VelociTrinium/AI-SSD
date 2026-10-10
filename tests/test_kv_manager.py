"""Unit tests for AISSDKVManager."""

import pytest
import torch
import numpy as np

from ai_ssd.engine.registry import ModelConfig
from ai_ssd.engine.kv_manager import AISSDKVManager
from ai_ssd.storage.file_backend import FileStorageBackend


def test_kv_manager_partitioning():
    cfg = ModelConfig(
        model_id="test-model",
        architecture="qwen2",
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        hidden_size=256,
        vocab_size=1000,
    )

    backend = FileStorageBackend(num_layers=2, num_heads=2, tokens_per_block=16, head_dim=64)
    try:
        kv_mgr = AISSDKVManager(
            backend=backend,
            config=cfg,
            sink_tokens=4,
            recent_tokens=16,
            tokens_per_block=16,
            top_k_pct=10.0,
        )

        # Simulate prefill with 100 tokens
        seq_len = 100
        mock_pkv = {
            0: (torch.randn(1, 2, seq_len, 64), torch.randn(1, 2, seq_len, 64)),
            1: (torch.randn(1, 2, seq_len, 64), torch.randn(1, 2, seq_len, 64)),
        }

        kv_mgr.init_from_prefill(mock_pkv)

        assert kv_mgr.is_active
        ld0 = kv_mgr.layer_data[0]
        # Sinks: 4 tokens
        assert ld0["sink_k"].shape[2] == 4
        # Recent: 16 tokens
        assert ld0["recent_k"].shape[2] == 16
        # Historical tokens: 100 - (4 + 16) = 80 tokens -> 80 / 16 = 5 blocks
        assert len(ld0["candidate_blocks"]) == 5

        # Test select and fetch active KV
        q = torch.randn(1, 4, 1, 64)
        active_k, active_v = kv_mgr.select_and_fetch_active_kv(layer_idx=0, query=q)

        # Top 10% of 5 blocks = 1 winning block (16 tokens)
        # Total active tokens in DRAM = 4 (sinks) + 16 (1 winning block) + 16 (recent) = 36 tokens
        assert active_k.shape[2] == 36
        assert active_v.shape[2] == 36
        assert kv_mgr.candidate_k_bytes_to_host == 0, "Candidate keys must have 0 B transfer!"
    finally:
        backend.close()
