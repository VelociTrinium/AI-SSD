"""Hierarchical KV Cache Manager for Computational Storage Inference."""

from __future__ import annotations
import math
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch

from ai_ssd.constants import (
    KEY_PAGE_BYTES,
    VALUE_PAGE_BYTES,
    LOGICAL_BLOCK_BYTES,
    TOKENS_PER_BLOCK,
    DEFAULT_SINK_TOKENS,
    DEFAULT_RECENT_TOKENS,
    DEFAULT_TOP_K_PCT,
)
from ai_ssd.storage.base import BaseStorageBackend
from ai_ssd.engine.registry import ModelConfig


class AISSDKVManager:
    """
    Manages the 3-tier computational storage KV cache hierarchy:
    1. Attention Sinks (first 4 tokens) permanently in Host DRAM
    2. Recent Window (last 16 tokens) in Host DRAM
    3. Historical Blocks (tokens 4..S-16) offloaded to storage
    """

    def __init__(
        self,
        backend: BaseStorageBackend,
        config: ModelConfig,
        sink_tokens: int = DEFAULT_SINK_TOKENS,
        recent_tokens: int = DEFAULT_RECENT_TOKENS,
        tokens_per_block: int = TOKENS_PER_BLOCK,
        top_k_pct: float = DEFAULT_TOP_K_PCT,
        enable_computational_storage: bool = True,
        enable_prefetch: bool = True,
    ):
        self.backend = backend
        self.config = config
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.tokens_per_block = tokens_per_block
        self.top_k_pct = top_k_pct
        self.enable_computational_storage = enable_computational_storage
        self.enable_prefetch = enable_prefetch

        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.layer_data: Dict[int, Dict[str, Any]] = {}
        self.is_active: bool = False

        # Invariant counters
        self.candidate_k_bytes_to_host: int = 0
        self.winning_k_bytes_to_host: int = 0
        self.winning_v_bytes_to_host: int = 0
        self.topk_metadata_bytes_to_host: int = 0

    def reset_timings(self) -> None:
        self.candidate_k_bytes_to_host = 0
        self.winning_k_bytes_to_host = 0
        self.winning_v_bytes_to_host = 0
        self.topk_metadata_bytes_to_host = 0

    def init_from_prefill(self, past_key_values: Any) -> None:
        """Partitions prefill KV cache across all layers and offloads historical blocks."""
        if hasattr(past_key_values, "layers"):
            layers_dict = {i: (lyr.keys, lyr.values) for i, lyr in enumerate(past_key_values.layers) if hasattr(lyr, "keys")}
        elif hasattr(past_key_values, "key_cache"):
            layers_dict = {i: (past_key_values.key_cache[i], past_key_values.value_cache[i]) for i in range(len(past_key_values.key_cache))}
        elif isinstance(past_key_values, dict):
            layers_dict = past_key_values
        else:
            layers_dict = {i: (past_key_values[i][0], past_key_values[i][1]) for i in range(len(past_key_values))}

        first_k = next(iter(layers_dict.values()))[0]
        self.num_kv_heads = first_k.shape[1]
        self.head_dim = first_k.shape[3]
        elem_bytes = 2 if first_k.dtype in (torch.float16, torch.bfloat16) else 4
        np_dtype = np.float16 if elem_bytes == 2 else np.float32

        for l_idx, (k_tensor, v_tensor) in layers_dict.items():
            seq_len = k_tensor.shape[2]

            if seq_len <= (self.sink_tokens + self.recent_tokens):
                sink_k = k_tensor.detach().clone()
                sink_v = v_tensor.detach().clone()
                recent_k = torch.empty((k_tensor.shape[0], k_tensor.shape[1], 0, k_tensor.shape[3]), dtype=k_tensor.dtype, device=k_tensor.device)
                recent_v = torch.empty((v_tensor.shape[0], v_tensor.shape[1], 0, v_tensor.shape[3]), dtype=v_tensor.dtype, device=v_tensor.device)
                cand_bids = []
            else:
                sink_k = k_tensor[:, :, :self.sink_tokens, :].detach().clone()
                sink_v = v_tensor[:, :, :self.sink_tokens, :].detach().clone()
                recent_k = k_tensor[:, :, -self.recent_tokens:, :].detach().clone()
                recent_v = v_tensor[:, :, -self.recent_tokens:, :].detach().clone()

                hist_start = self.sink_tokens
                hist_end = seq_len - self.recent_tokens
                cand_bids = []

                if hist_end > hist_start:
                    k_np = k_tensor[0].detach().cpu().numpy()
                    v_np = v_tensor[0].detach().cpu().numpy()
                    k_t = np.transpose(k_np[:, hist_start:hist_end, :], (1, 0, 2))  # [hist_tokens, kv_heads, head_dim]
                    v_t = np.transpose(v_np[:, hist_start:hist_end, :], (1, 0, 2))
                    total_hist_tok = k_t.shape[0]

                    bid = 0
                    for b_start in range(0, total_hist_tok, self.tokens_per_block):
                        b_end = min(b_start + self.tokens_per_block, total_hist_tok)
                        tok_count = b_end - b_start
                        if tok_count == self.tokens_per_block:
                            k_blk = np.ascontiguousarray(k_t[b_start:b_end])
                            v_blk = np.ascontiguousarray(v_t[b_start:b_end])
                        else:
                            k_blk = np.zeros((self.tokens_per_block, self.num_kv_heads, self.head_dim), dtype=np_dtype)
                            v_blk = np.zeros((self.tokens_per_block, self.num_kv_heads, self.head_dim), dtype=np_dtype)
                            k_blk[:tok_count] = k_t[b_start:b_end]
                            v_blk[:tok_count] = v_t[b_start:b_end]
                        self.backend.write_block(l_idx, bid, k_blk, v_blk)
                        cand_bids.append((bid, tok_count))
                        bid += 1

            self.layer_data[l_idx] = {
                "sink_k": sink_k,
                "sink_v": sink_v,
                "recent_k": recent_k,
                "recent_v": recent_v,
                "candidate_blocks": cand_bids,
                "total_tokens": seq_len,
            }

        self.is_active = True

    def append_new_token(self, layer_idx: int, k_tok: torch.Tensor, v_tok: torch.Tensor) -> None:
        """Slides recent window with newly generated token; offloads historical chunk if full."""
        ld = self.layer_data[layer_idx]
        ld["recent_k"] = torch.cat([ld["recent_k"], k_tok], dim=2)
        ld["recent_v"] = torch.cat([ld["recent_v"], v_tok], dim=2)
        ld["total_tokens"] += 1

        if ld["recent_k"].shape[2] >= (self.recent_tokens + self.tokens_per_block):
            k_chunk = ld["recent_k"][:, :, :self.tokens_per_block, :]
            v_chunk = ld["recent_v"][:, :, :self.tokens_per_block, :]
            ld["recent_k"] = ld["recent_k"][:, :, self.tokens_per_block:, :].contiguous()
            ld["recent_v"] = ld["recent_v"][:, :, self.tokens_per_block:, :].contiguous()

            elem_bytes = 2 if k_chunk.dtype in (torch.float16, torch.bfloat16) else 4
            np_dtype = np.float16 if elem_bytes == 2 else np.float32
            k_blk_np = np.ascontiguousarray(k_chunk[0].detach().cpu().numpy().transpose(1, 0, 2).astype(np_dtype, copy=False))
            v_blk_np = np.ascontiguousarray(v_chunk[0].detach().cpu().numpy().transpose(1, 0, 2).astype(np_dtype, copy=False))

            bid = len(ld["candidate_blocks"])
            self.backend.write_block(layer_idx, bid, k_blk_np, v_blk_np)
            ld["candidate_blocks"].append((bid, self.tokens_per_block))

    def select_and_fetch_active_kv(
        self,
        layer_idx: int,
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes In-Storage Top-K filtering with Query vector.
        ZERO candidate Key vectors travel across PCIe bus.
        Fetches only winning Top-K blocks and concatenates [Sinks, Winning, Recent].
        """
        ld = self.layer_data[layer_idx]
        cand_bids = ld["candidate_blocks"]
        q_np = query[0, :, 0, :].detach().to(torch.float32).cpu().numpy()
        scale = 1.0 / math.sqrt(self.head_dim)

        selected_k_blocks = []
        selected_v_blocks = []

        if cand_bids:
            k_val = max(1, int(math.ceil(len(cand_bids) * (self.top_k_pct / 100.0))))

            # In-storage Top-K filtering: Query broadcast over PCIe -> winning IDs returned
            top_bids = self.backend.compute_topk_filter(
                layer_idx=layer_idx,
                cand_bids=cand_bids,
                query=q_np,
                top_k=k_val,
                scale=scale,
                q_heads=q_np.shape[0],
                kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
            self.topk_metadata_bytes_to_host += len(top_bids) * 12

            win_bids = [bid for _, bid, _ in top_bids]

            # Contiguous batch fetch of winning blocks
            loaded_blocks = self.backend.read_block_batch(layer_idx, win_bids)
            page_bytes = self.tokens_per_block * self.num_kv_heads * self.head_dim * (2 if ld["sink_k"].dtype in (torch.float16, torch.bfloat16) else 4)
            self.winning_k_bytes_to_host += len(win_bids) * page_bytes
            self.winning_v_bytes_to_host += len(win_bids) * page_bytes

            # Speculative prefetch for next layer
            if self.enable_prefetch:
                self.backend.predict_and_prefetch(layer_idx, win_bids, async_mode=True)

            # Reconstruct active winning tensors preserving temporal order
            sorted_top_bids = sorted(top_bids, key=lambda x: x[1])
            target_dtype = ld["sink_k"].dtype
            target_device = ld["sink_k"].device

            for _, bid, actual_tokens in sorted_top_bids:
                k_blk, v_blk = loaded_blocks[bid]
                k_t = torch.from_numpy(np.array(k_blk[:actual_tokens], copy=True)).to(dtype=target_dtype, device=target_device).permute(1, 0, 2).unsqueeze(0)
                v_t = torch.from_numpy(np.array(v_blk[:actual_tokens], copy=True)).to(dtype=target_dtype, device=target_device).permute(1, 0, 2).unsqueeze(0)
                selected_k_blocks.append(k_t)
                selected_v_blocks.append(v_t)

        # Assemble full active working set in Host DRAM
        parts_k = [ld["sink_k"]] + selected_k_blocks + [ld["recent_k"]]
        parts_v = [ld["sink_v"]] + selected_v_blocks + [ld["recent_v"]]
        active_k = torch.cat(parts_k, dim=2)
        active_v = torch.cat(parts_v, dim=2)

        return active_k, active_v
