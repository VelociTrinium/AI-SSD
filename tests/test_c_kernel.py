"""Unit tests for native in-storage AVX2 C kernel."""

import math
import numpy as np
import pytest

from ai_ssd.c_kernel.binding import get_native_c_kernel


def test_c_kernel_availability():
    kernel = get_native_c_kernel()
    assert kernel.is_available(), "Native C kernel should be compiled and loaded"


def test_c_kernel_topk_gqa_parity():
    kernel = get_native_c_kernel()
    if not kernel.is_available():
        pytest.skip("Native C kernel not available")

    q_heads = 14
    kv_heads = 2
    head_dim = 64
    tokens_per_block = 16
    num_blocks = 20
    top_k = 4
    scale = 1.0 / math.sqrt(head_dim)

    np.random.seed(42)
    query = np.random.randn(q_heads, head_dim).astype(np.float32)
    k_blocks = [
        np.random.randn(tokens_per_block, kv_heads, head_dim).astype(np.float32)
        for _ in range(num_blocks)
    ]
    actual_tokens = [tokens_per_block] * num_blocks

    # Compute reference scores with numpy
    gqa_ratio = q_heads // kv_heads
    ref_scores = []
    for bid, k_blk in enumerate(k_blocks):
        dots = np.einsum("hd,thd->th", query, k_blk[:, [h // gqa_ratio for h in range(q_heads)], :]) * scale
        max_score = float(np.max(dots))
        ref_scores.append((max_score, bid))
    ref_scores.sort(key=lambda x: x[0], reverse=True)
    expected_top_indices = [bid for _, bid in ref_scores[:top_k]]

    # Compute with native C kernel
    c_indices, c_scores = kernel.compute_topk_gqa(
        query=query,
        k_blocks=k_blocks,
        actual_tokens=actual_tokens,
        top_k=top_k,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        scale=scale,
    )

    assert list(c_indices) == expected_top_indices, f"Kernel indices {list(c_indices)} != Ref {expected_top_indices}"
    for i in range(top_k):
        assert abs(c_scores[i] - ref_scores[i][0]) < 1e-4, f"Score mismatch at {i}: {c_scores[i]} vs {ref_scores[i][0]}"
