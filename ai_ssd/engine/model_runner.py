"""Executable Inference Runner for Baseline (in-DRAM) and AI-SSD (Storage Offload)."""

from __future__ import annotations
import gc
import re
import sys
import time
import inspect
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ai_ssd.engine.registry import ModelRegistry, ModelConfig
from ai_ssd.engine.kv_manager import AISSDKVManager
from ai_ssd.engine.attention_hook import wrap_model_attention
from ai_ssd.storage.base import BaseStorageBackend
from ai_ssd.telemetry.memory_sampler import ProcessMemorySampler, get_current_rss_mb


def is_english_text(text: str) -> bool:
    """Checks whether text consists predominantly of English/Latin characters."""
    if not text.strip():
        return True
    eng_chars = sum(1 for ch in text if not ch.isspace() and (ord(ch) < 128 or ("\u00A0" <= ch <= "\u024F")))
    total_chars = sum(1 for ch in text if not ch.isspace())
    return (eng_chars / max(1, total_chars)) >= 0.85


def get_non_english_mask(tokenizer: Any, device: torch.device) -> Optional[torch.Tensor]:
    """Caches token IDs of non-Latin scripts to enforce strictly English output."""
    vocab = getattr(tokenizer, "get_vocab", lambda: {})() or getattr(tokenizer, "vocab", {})
    if not vocab:
        return None
    pattern = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0900-\u097f]")
    bad_ids = [idx for token, idx in vocab.items() if pattern.search(token)]
    return torch.tensor(bad_ids, dtype=torch.long, device=device) if bad_ids else None


def filter_english_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None and len(mask) > 0:
        valid_mask = mask[mask < logits.shape[-1]]
        if len(valid_mask) > 0:
            logits = logits.clone()
            logits[:, valid_mask] = -float("inf")
    return logits


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: str = "float32",
    threads: int = 4,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Any, ModelConfig]:
    """Loads model and tokenizer with strict thread pinning and evaluation mode."""
    torch.set_num_threads(threads)
    cfg = ModelRegistry.resolve_config(model_name_or_path)

    torch_dtype = torch.float16 if "16" in dtype.lower() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    return model, tokenizer, cfg


def run_baseline_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    decode_tokens: int = 16,
    seed: int = 42,
    enforce_english: bool = True,
) -> Dict[str, Any]:
    """Executes dense in-DRAM reference inference using standard PyTorch DynamicCache."""
    torch.manual_seed(seed)
    gc.collect()

    non_eng_mask = get_non_english_mask(tokenizer, input_ids.device) if enforce_english else None

    # Prefill step
    prefill_sampler = ProcessMemorySampler(sample_interval_s=0.005)
    prefill_sampler.start()
    t_pref_start = time.perf_counter()

    with torch.inference_mode():
        prefill_out = model(input_ids=input_ids, use_cache=True)
    t_pref_end = time.perf_counter()
    prefill_mem = prefill_sampler.stop()
    prefill_time = t_pref_end - t_pref_start
    pkv = prefill_out.past_key_values

    last_logits = prefill_out.logits[:, -1, :]
    if enforce_english:
        last_logits = filter_english_logits(last_logits, non_eng_mask)

    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
    generated_tokens = [next_token.item()]

    del prefill_out
    gc.collect()
    post_prefill_rss = get_current_rss_mb()

    # Decode interval
    decode_sampler = ProcessMemorySampler(sample_interval_s=0.002)
    decode_sampler.start()
    t_dec_start = time.perf_counter()

    for _ in range(1, decode_tokens):
        with torch.inference_mode():
            step_out = model(input_ids=next_token, past_key_values=pkv, use_cache=True)
        pkv = step_out.past_key_values

        step_logits = step_out.logits[:, -1, :]
        if enforce_english:
            step_logits = filter_english_logits(step_logits, non_eng_mask)

        next_token = torch.argmax(step_logits, dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())

    t_dec_end = time.perf_counter()
    decode_mem = decode_sampler.stop()
    decode_time = t_dec_end - t_dec_start

    # KV memory computation
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", 36)
    num_kv_heads = getattr(cfg, "num_key_value_heads", 8)
    head_dim = getattr(cfg, "head_dim", 128)
    bytes_per_elem = 2 if next(model.parameters()).dtype in (torch.float16, torch.bfloat16) else 4
    total_tokens = input_ids.shape[1] + decode_tokens
    active_kv_mb = (total_tokens * num_layers * num_kv_heads * head_dim * bytes_per_elem * 2) / (1024.0 * 1024.0)

    overall_peak_rss = max(prefill_mem["peak_rss_mb"], decode_mem["peak_rss_mb"])

    return {
        "mode": "BASELINE",
        "wall_time_s": decode_time,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "tokens_per_second": len(generated_tokens) / max(1e-6, decode_time),
        "decode_tokens": len(generated_tokens),
        "token_ids": generated_tokens,
        "generated_text": tokenizer.decode(generated_tokens),
        "active_kv_mb": active_kv_mb,
        "cold_kv_mb": 0.0,
        "prefill_peak_rss_mb": prefill_mem["peak_rss_mb"],
        "post_prefill_rss_mb": post_prefill_rss,
        "decode_peak_rss_mb": decode_mem["peak_rss_mb"],
        "overall_peak_rss_mb": overall_peak_rss,
        "min_rss_mb": decode_mem["min_rss_mb"],
        "avg_rss_mb": decode_mem["avg_rss_mb"],
        "candidate_k_bytes_to_host": 0,
        "winning_kv_bytes_to_host": 0,
    }


def run_aissd_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    storage_backend: BaseStorageBackend,
    config: ModelConfig,
    decode_tokens: int = 16,
    top_k_pct: float = 10.0,
    enable_prefetch: bool = True,
    seed: int = 42,
    enforce_english: bool = True,
) -> Dict[str, Any]:
    """Executes AI-SSD computational storage inference with KV cache offload."""
    torch.manual_seed(seed)
    gc.collect()

    non_eng_mask = get_non_english_mask(tokenizer, input_ids.device) if enforce_english else None

    kv_mgr = AISSDKVManager(
        backend=storage_backend,
        config=config,
        top_k_pct=top_k_pct,
        enable_prefetch=enable_prefetch,
    )

    orig_forwards, restore_func = wrap_model_attention(model=model, kv_mgr=kv_mgr)

    try:
        # 1. Prefill Step
        prefill_sampler = ProcessMemorySampler(sample_interval_s=0.005)
        prefill_sampler.start()
        t_pref_start = time.perf_counter()

        with torch.inference_mode():
            prefill_out = model(input_ids=input_ids, use_cache=True)
        t_pref_end = time.perf_counter()
        prefill_mem = prefill_sampler.stop()
        prefill_time = t_pref_end - t_pref_start

        # Initialize KV manager hierarchy & offload historical blocks
        pkv_prefill = prefill_out.past_key_values
        kv_mgr.init_from_prefill(pkv_prefill)

        storage_backend.reset_stats()
        kv_mgr.reset_timings()

        last_logits = prefill_out.logits[:, -1, :]
        if enforce_english:
            last_logits = filter_english_logits(last_logits, non_eng_mask)

        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
        generated_tokens = [next_token.item()]
        cur_seq_len = input_ids.shape[1]

        # 2. True Host-RAM Detach & GC: drop unpruned prefill tensors from DRAM!
        del prefill_out
        del pkv_prefill
        gc.collect()
        post_prefill_rss = get_current_rss_mb()

        # 3. Decode Step Interval
        decode_sampler = ProcessMemorySampler(sample_interval_s=0.002)
        decode_sampler.start()
        t_dec_start = time.perf_counter()

        for step in range(1, decode_tokens):
            pos_ids = torch.tensor([[cur_seq_len + step - 1]], device=input_ids.device)
            with torch.inference_mode():
                step_out = model(input_ids=next_token, position_ids=pos_ids, use_cache=False)

            step_logits = step_out.logits[:, -1, :]
            if enforce_english:
                step_logits = filter_english_logits(step_logits, non_eng_mask)

            next_token = torch.argmax(step_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())

        t_dec_end = time.perf_counter()
        decode_mem = decode_sampler.stop()
        decode_time = t_dec_end - t_dec_start

        # Calculate Active vs Cold KV DRAM footprints
        bytes_per_elem = 2 if next(model.parameters()).dtype in (torch.float16, torch.bfloat16) else 4
        sinks = kv_mgr.sink_tokens
        recent = kv_mgr.recent_tokens
        cand_bids_count = len(kv_mgr.layer_data[0]["candidate_blocks"])
        winning_blocks = max(1, int(cand_bids_count * (top_k_pct / 100.0))) if cand_bids_count > 0 else 0
        active_tokens = sinks + (winning_blocks * kv_mgr.tokens_per_block) + recent

        active_kv_mb = (active_tokens * config.num_layers * config.num_key_value_heads * config.head_dim * bytes_per_elem * 2) / (1024.0 * 1024.0)
        cold_blocks = max(0, cand_bids_count - winning_blocks)
        cold_kv_mb = (cold_blocks * kv_mgr.tokens_per_block * config.num_layers * config.num_key_value_heads * config.head_dim * bytes_per_elem * 2) / (1024.0 * 1024.0)

        overall_peak_rss = max(prefill_mem["peak_rss_mb"], decode_mem["peak_rss_mb"])

        return {
            "mode": "AI-SSD",
            "wall_time_s": decode_time,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "tokens_per_second": len(generated_tokens) / max(1e-6, decode_time),
            "decode_tokens": len(generated_tokens),
            "token_ids": generated_tokens,
            "generated_text": tokenizer.decode(generated_tokens),
            "active_kv_mb": active_kv_mb,
            "cold_kv_mb": cold_kv_mb,
            "prefill_peak_rss_mb": prefill_mem["peak_rss_mb"],
            "post_prefill_rss_mb": post_prefill_rss,
            "decode_peak_rss_mb": decode_mem["peak_rss_mb"],
            "overall_peak_rss_mb": overall_peak_rss,
            "min_rss_mb": decode_mem["min_rss_mb"],
            "avg_rss_mb": decode_mem["avg_rss_mb"],
            "candidate_k_bytes_to_host": kv_mgr.candidate_k_bytes_to_host,
            "winning_kv_bytes_to_host": kv_mgr.winning_k_bytes_to_host + kv_mgr.winning_v_bytes_to_host,
            "storage_telemetry": storage_backend.get_telemetry(),
        }

    finally:
        restore_func()
