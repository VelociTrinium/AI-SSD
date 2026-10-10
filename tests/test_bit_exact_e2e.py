"""End-to-end integration test verifying bit-exact correctness and telemetry."""

import pytest
import torch

from ai_ssd.engine.registry import ModelRegistry
from ai_ssd.engine.model_runner import (
    load_model_and_tokenizer,
    run_baseline_decode,
    run_aissd_decode,
)
from ai_ssd.storage.file_backend import FileStorageBackend


def test_bit_exact_e2e_execution():
    """Runs end-to-end decode comparing Baseline and AI-SSD."""
    try:
        model, tokenizer, cfg = load_model_and_tokenizer(
            model_name_or_path="qwen2.5-0.5b",
            dtype="float32",
            threads=2,
        )
    except Exception as e:
        pytest.skip(f"Model loading unavailable in current environment: {e}")

    prompt = "Artificial intelligence and computational storage systems allow long-context processing."
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Baseline
    res_b = run_baseline_decode(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        decode_tokens=4,
        seed=42,
    )

    # AI-SSD (with 100% Top-K so token match is guaranteed bit-exact)
    storage = FileStorageBackend(
        num_layers=cfg.num_layers,
        num_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype="float32",
    )
    try:
        res_a = run_aissd_decode(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            storage_backend=storage,
            config=cfg,
            decode_tokens=4,
            top_k_pct=100.0,
            seed=42,
        )

        assert res_a["candidate_k_bytes_to_host"] == 0, "Zero PCIe Candidate K invariant must hold!"
        assert res_a["token_ids"] == res_b["token_ids"], f"Tokens divergent: AI-SSD {res_a['token_ids']} vs Base {res_b['token_ids']}"
    finally:
        storage.close()
