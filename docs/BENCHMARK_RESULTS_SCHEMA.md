# AI-SSD V2 — Benchmark Results Storage Schema

This document defines the standardized specification and layout for all benchmark telemetry results stored in the `benchmarks/live_inference/results/` directory.

---

## 1. Directory Structure

All benchmark executions—including CLI demos (`scripts/demo_inference.py`), automated comparisons (`scripts/run_quick_comparison.py`), multi-context scaling sweeps, and automated validation tests—store results in:

```text
benchmarks/live_inference/results/
├── benchmark_records.json                  <-- Unified indexed array of all benchmark executions
├── demo_qwen3_4b_fp32_ctx4096_baseline_*.json
├── demo_qwen3_4b_fp32_ctx4096_aissd_*.json
├── quick_comp_qwen2.5_0.5b_fp32_ctx512_*.json
├── context_scaling_results.json            <-- Context scaling matrix (4K -> 32K)
├── final_benchmark_results.json            <-- Canonical system evaluation snapshot
└── optimization3/                          <-- 8B FP16 scaling records
    ├── qwen3_8b_fp16_baseline.json
    └── qwen3_8b_fp16_dense_reference.json
```

---

## 2. Standard JSON Schema (`Schema Version 2.0`)

Each individual benchmark execution file adheres to the following specification:

```json
{
  "schema_version": "2.0",
  "timestamp_utc": "2026-10-07T18:00:00Z",
  "source": "scripts/demo_inference.py",
  "model": {
    "name": "Qwen3",
    "full_name": "Qwen3-4B",
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "weight": "4B",
    "precision": "fp32",
    "precision_display": "FP32"
  },
  "benchmark": {
    "mode": "AI-SSD",
    "context_length": 4096,
    "decode_tokens": 16,
    "threads": 4,
    "storage_mode": "nvme_qemu",
    "computational_storage": true
  },
  "metrics": {
    "ram_consumption": {
      "peak_rss_mb": 17554.62,
      "avg_rss_mb": 17120.30,
      "min_rss_mb": 16890.10,
      "vm_peak_mb": 22400.00,
      "final_rss_mb": 16995.30
    },
    "kv_cache": {
      "active_kv_mb": 122.62,
      "cold_kv_mb": 1147.50,
      "total_kv_mb": 1270.12
    },
    "performance": {
      "wall_time_s": 20.552,
      "tokens_per_second": 0.779,
      "latency_ms_per_token": 1284.5
    },
    "storage_bus": {
      "candidate_k_bytes_to_host": 0,
      "winning_kv_bytes_to_host": 60293120,
      "storage_read_bytes": 1205862400,
      "visible_storage_s": 7.91
    },
    "verification": {
      "exact_token_match": true,
      "token_ids": [11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13],
      "generated_text": " hardware accelerated attention scoring engine computes dot products between the query vector and candidate keys."
    }
  }
}
```

---

## 3. Key Fields & Data Types

| Field Path | Type | Unit / Description |
| :--- | :--- | :--- |
| `model.name` | `string` | Base model family (e.g. `Qwen3`, `Qwen2.5`, `Mistral`) |
| `model.weight` | `string` | Parameter size / weight (e.g. `0.5B`, `4B`, `8B`, `0.21B`) |
| `model.precision_display` | `string` | Numerical precision (e.g. `FP32`, `FP16`) |
| `benchmark.mode` | `string` | Inference mode: `BASELINE` or `AI-SSD` |
| `benchmark.context_length` | `integer` | Sequence token length (X-axis on charts) |
| `metrics.ram_consumption.peak_rss_mb` | `float` | Peak Host RAM RSS in Megabytes (Y-axis on charts) |
| `metrics.kv_cache.active_kv_mb` | `float` | Active KV cache resident in host DRAM |
| `metrics.performance.tokens_per_second` | `float` | Autoregressive decoding throughput |
| `metrics.storage_bus.candidate_k_bytes_to_host` | `integer` | Bytes transmitted across storage bus for scoring (0 in AI-SSD) |
| `metrics.verification.exact_token_match` | `boolean` | Verification against canonical baseline token stream |

---

## 4. Programmatic Ingestion & Saving API

All scripts utilize the shared utility module `benchmarks.live_inference.result_schema`:

```python
from benchmarks.live_inference.result_schema import build_benchmark_record, save_benchmark_result

record = build_benchmark_record(
    source="scripts/demo_inference.py",
    mode="aissd",
    model_name="qwen3-4b",
    precision="fp32",
    context_length=4096,
    decode_tokens=16,
    threads=4,
    raw_metrics=worker_output_dict,
    storage_mode="nvme_qemu",
    computational_storage=True,
)

saved_path = save_benchmark_result(record, prefix="demo_inference")
print(f"Saved benchmark result: {saved_path}")
```
