"""AI-SSD Command Line Interface.

Executes baseline (in-DRAM) and AI-SSD (computational storage offload) inference,
providing side-by-side memory telemetry, throughput, and bit-exact token verification.
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional

import torch

from ai_ssd.constants import DEFAULT_TOP_K_PCT, TOKENS_PER_BLOCK
from ai_ssd.engine.model_runner import (
    load_model_and_tokenizer,
    run_baseline_decode,
    run_aissd_decode,
    is_english_text,
)
from ai_ssd.storage.file_backend import FileStorageBackend
from ai_ssd.storage.nvme_backend import NvmeStorageBackend
from ai_ssd.storage.prefetcher import PrefetchStorageAdapter


def build_synthetic_prompt(tokenizer: Any, target_tokens: int) -> str:
    """Builds an English text prompt approximately matching target_tokens."""
    base_sentence = (
        "In modern computational storage systems, offloading high-dimensional Key and Value cache "
        "tensors directly into non-volatile storage controllers drastically mitigates host memory pressure. "
        "Through hardware-accelerated SIMD kernels, sparse attention scoring identifies winning token candidates "
        "with strictly zero candidate key movement across the PCIe interface. "
    )
    tokens_per_repeat = len(tokenizer.encode(base_sentence, add_special_tokens=False))
    repeats = max(1, target_tokens // tokens_per_repeat)
    prompt = " ".join([base_sentence] * repeats)
    # Truncate precisely to target_tokens
    enc = tokenizer.encode(prompt, add_special_tokens=False)
    if len(enc) > target_tokens:
        prompt = tokenizer.decode(enc[:target_tokens])
    return prompt


def print_comparison_table(b: Dict[str, Any], a: Dict[str, Any], model_name: str) -> None:
    """Prints a clear side-by-side comparison between Baseline and AI-SSD."""
    b_act_kv = b.get("active_kv_mb", 0.0)
    a_act_kv = a.get("active_kv_mb", 0.0)
    kv_reduction = ((b_act_kv - a_act_kv) / max(1e-6, b_act_kv)) * 100.0 if b_act_kv > 0 else 0.0

    b_peak_rss = b.get("overall_peak_rss_mb", 0.0)
    a_peak_rss = a.get("overall_peak_rss_mb", 0.0)

    b_toks = b.get("token_ids", [])
    a_toks = a.get("token_ids", [])
    matches = sum(1 for x, y in zip(b_toks, a_toks) if x == y)
    total_toks = max(len(b_toks), len(a_toks))
    exact_pct = (matches / max(1, total_toks)) * 100.0

    print("\n" + "=" * 80)
    print(f"             AI-SSD vs BASELINE EXECUTION SUMMARY ({model_name})           ")
    print("=" * 80)
    print(f"{'Metric':<34} | {'Baseline (In-DRAM)':<20} | {'AI-SSD (Storage Offload)'}")
    print("-" * 80)
    print(f"--- Process Memory Telemetry ---")
    print(f"{'Prefill Peak RSS':<34} | {b.get('prefill_peak_rss_mb', 0.0):>17.1f} MB | {a.get('prefill_peak_rss_mb', 0.0):>16.1f} MB")
    print(f"{'Post-Prefill RSS':<34} | {b.get('post_prefill_rss_mb', 0.0):>17.1f} MB | {a.get('post_prefill_rss_mb', 0.0):>16.1f} MB")
    print(f"{'Decode Peak RSS':<34} | {b.get('decode_peak_rss_mb', 0.0):>17.1f} MB | {a.get('decode_peak_rss_mb', 0.0):>16.1f} MB")
    print(f"{'Overall Peak RSS':<34} | {b_peak_rss:>17.1f} MB | {a_peak_rss:>16.1f} MB")
    print("-" * 80)
    print(f"--- KV Cache Memory Hierarchy ---")
    print(f"{'Active KV in Host RAM':<34} | {b_act_kv:>17.2f} MB | {a_act_kv:>16.2f} MB")
    print(f"{'Cold KV on Storage':<34} | {'0.00':>17} MB | {a.get('cold_kv_mb', 0.0):>16.2f} MB")
    print(f"{'KV Cache DRAM Reduction':<34} | {'0.0%':>20} | {kv_reduction:>19.1f}%")
    print("-" * 80)
    print(f"--- Bus Transfer Invariants ---")
    print(f"{'Candidate K -> Host PCIe':<34} | {'N/A (All In RAM)':>20} | {a.get('candidate_k_bytes_to_host', 0):>17} B (0 B)")
    print(f"{'Winning KV -> Host PCIe':<34} | {'N/A (All In RAM)':>20} | {a.get('winning_kv_bytes_to_host', 0) / (1024*1024):>16.2f} MB")
    print("-" * 80)
    print(f"--- Throughput & Latency ---")
    print(f"{'Decode Wall Time':<34} | {b.get('decode_time_s', 0.0):>17.3f} s  | {a.get('decode_time_s', 0.0):>16.3f} s")
    print(f"{'Tokens per Second':<34} | {b.get('tokens_per_second', 0.0):>17.2f} t/s| {a.get('tokens_per_second', 0.0):>16.2f} t/s")
    print("-" * 80)
    print(f"--- Numerical Correctness ---")
    print(f"{'Token Match Count':<34} | {f'{len(b_toks)}/{len(b_toks)} Reference':>20} | {f'{matches}/{total_toks} ({exact_pct:.1f}%)'}")
    print("=" * 80 + "\n")


def create_storage_backend(
    storage_mode: str,
    config: Any,
    dtype: str,
    enable_prefetch: bool = True,
) -> Any:
    if storage_mode == "nvme":
        backend = NvmeStorageBackend(
            num_layers=config.num_layers,
            num_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=dtype,
        )
    else:
        backend = FileStorageBackend(
            num_layers=config.num_layers,
            num_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=dtype,
        )

    if enable_prefetch:
        backend = PrefetchStorageAdapter(
            storage_backend=backend,
            num_layers=config.num_layers,
            enable_async_pipeline=True,
        )
    return backend


def main():
    parser = argparse.ArgumentParser(description="AI-SSD Computational Storage CLI")
    parser.add_argument("--mode", type=str, default="compare", choices=["baseline", "aissd", "compare"])
    parser.add_argument("--storage-mode", type=str, default="file", choices=["file", "nvme"])
    parser.add_argument("--model", type=str, default="qwen2.5-0.5b")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "fp32", "float16", "fp16"])
    parser.add_argument("--context", type=int, default=512, help="Context length in tokens")
    parser.add_argument("--decode", type=int, default=16, help="Tokens to generate")
    parser.add_argument("--threads", type=int, default=4, help="CPU execution threads")
    parser.add_argument("--top-k-pct", type=float, default=DEFAULT_TOP_K_PCT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--disable-prefetch", action="store_true", help="Disable async layer prefetch")
    parser.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args()

    print(f"\n[AI-SSD] Loading {args.model} ({args.dtype}, {args.threads} CPU threads)...")
    model, tokenizer, cfg = load_model_and_tokenizer(
        model_name_or_path=args.model,
        dtype=args.dtype,
        threads=args.threads,
    )

    if args.prompt:
        prompt_text = args.prompt
    else:
        prompt_text = build_synthetic_prompt(tokenizer, args.context)

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    actual_tokens = input_ids.shape[1]
    print(f"[AI-SSD] Prompt encoded: {actual_tokens} tokens")

    res_b = None
    res_a = None

    if args.mode in ("baseline", "compare"):
        print("\n--- Running Baseline (In-DRAM DynamicCache) ---")
        res_b = run_baseline_decode(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            decode_tokens=args.decode,
            seed=args.seed,
        )
        print(f"[BASELINE] Finished: {res_b['tokens_per_second']:.2f} tok/s, Peak RSS: {res_b['overall_peak_rss_mb']:.1f} MB")

    if args.mode in ("aissd", "compare"):
        print(f"\n--- Running AI-SSD (Storage Mode: {args.storage_mode.upper()}, Top-K: {args.top_k_pct}%) ---")
        storage_backend = create_storage_backend(
            storage_mode=args.storage_mode,
            config=cfg,
            dtype=args.dtype,
            enable_prefetch=not args.disable_prefetch,
        )
        try:
            res_a = run_aissd_decode(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                storage_backend=storage_backend,
                config=cfg,
                decode_tokens=args.decode,
                top_k_pct=args.top_k_pct,
                enable_prefetch=not args.disable_prefetch,
                seed=args.seed,
            )
            print(f"[AI-SSD] Finished: {res_a['tokens_per_second']:.2f} tok/s, Peak RSS: {res_a['overall_peak_rss_mb']:.1f} MB")
        finally:
            storage_backend.close()

    if args.mode == "compare" and res_b and res_a:
        print_comparison_table(res_b, res_a, cfg.model_id)

    if args.output_json:
        out_payload = res_a if args.mode == "aissd" else (res_b if args.mode == "baseline" else {"baseline": res_b, "aissd": res_a})
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        print(f"[INFO] Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
