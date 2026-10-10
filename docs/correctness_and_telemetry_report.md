# AI-SSD Inference Correctness & Storage Telemetry Report

**Branch:** `feature/inference-correctness-and-io-telemetry`  
**Starting Commit:** `b62655d`  
**Date:** October 8, 2026  
**Environment:** AWS EC2 (`ip-172-31-27-77`, Intel Xeon Platinum 8488C, 4 vCPUs, 30.8 GB RAM, Ubuntu 22.04 LTS / Linux 6.8.0-1065-aws)

---

## 1. Executive Summary

A critical correctness regression where AI-SSD firmware produced divergent or repetitive tokens compared to baseline inference has been investigated, root-caused, and resolved across all supported models. In addition, robust inference-level storage telemetry (`Total Read MB`, `Total Write MB`, `Candidate K to Host Bus = 0 B`) has been implemented with before-and-after counter snapshots.

All 6 evaluatable models now produce **100% identical token IDs** between Baseline (firmware disabled) and AI-SSD (firmware enabled):
- `Qwen2.5-0.5B`: **100% Match (16/16 tokens)**
- `Tiny-Mistral`: **100% Match (16/16 tokens)**
- `Qwen3-4B`: **100% Match (16/16 tokens)**
- `Qwen3-8B`: **100% Match (16/16 tokens)**
- `Qwen3.5-9B`: **100% Match (8/8 tokens)**
- `Mistral-7B`: **100% Match (16/16 tokens)**
- `Jamba`: Gated repository on Hugging Face hub (requires token authentication)

---

## 2. Root Cause Analysis

### Bug 1: Token Dropping in `TransformerKVStateProvider.append_new_token`
- **Location:** `person1_kv_engine/adapters/transformer_kv.py`
- **Issue:** `append_new_token` used `ld["recent_k"] = torch.cat([ld["recent_k"][:, :, 1:, :], k_tok], dim=2)`. This sliced off the oldest token (`1:`) from the recent window on every decode step without persisting it to storage blocks. Over 16 decode steps, 16 historical tokens were silently deleted from the attention context.
- **Resolution:** Updated `append_new_token` to preserve all tokens: `ld["recent_k"] = torch.cat([ld["recent_k"], k_tok], dim=2)`.

### Bug 2: Missing `qwen3` Adapter & RMSNorm Omission
- **Location:** `person1_kv_engine/adapters/registry.py` and `model_adapter.py`
- **Issue:** Hugging Face models `Qwen/Qwen3-4B-Instruct-2507` and `Qwen/Qwen3-8B` report `model_type: "qwen3"`. In `ModelRegistry._adapter_classes`, only `"qwen"` and `"qwen2"` were registered. Unrecognized `"qwen3"` fell back to `MistralAdapter`. `MistralAdapter` omitted RMSNorm on Query and Key projections (`q_norm` and `k_norm`), causing unnormalized QK dot products to blow up attention distributions.
- **Resolution:** Registered `"qwen3": QwenAdapter` and `"qwen3_instruct": QwenAdapter`. Added safe `q_norm`/`k_norm` checks in all adapters.

### Bug 3: Model Architecture Configuration Inaccuracies
- **Location:** `person1_kv_engine/adapters/registry.py`
- **Issue:** `KNOWN_MODELS["qwen3-4b"]` incorrectly configured `num_attention_heads=16`, `num_key_value_heads=4`, `hidden_size=2048`. The actual HF model uses `num_attention_heads=32`, `num_key_value_heads=8`, `hidden_size=2560`.
- **Resolution:** Corrected metadata to match the canonical HF config (`num_attention_heads=32`, `num_key_value_heads=8`, `hidden_size=2560`, `head_dim=128`, `architecture="qwen3"`).

### Bug 4: Qwen3.5 Heterogeneous Layer Classification Bug
- **Location:** `person1_kv_engine/adapters/descriptor.py` and `registry.py`
- **Issue:** Qwen3.5 separates full self-attention (`"full_attention"` at layers 3, 7, 11, 15, 19, 23, 27, 31) from linear DeltaNet SSM (`"linear_attention"` across the remaining 24 layers). `ModelArchitectureConfig.is_layer_attention` only looked for `("attention", "self_attn", "attn")`. This caused full attention layers to be misclassified as non-attention layers, returning 1-element SSM state tuples during active KV retrieval.
- **Resolution:** Added `"full_attention"` to `is_layer_attention`, added `"linear_attention"` to `is_layer_ssm`, and defined layer types array for `qwen3.5-9b` in `registry.py`.

### Bug 5: Qwen3.5 SDPA Reshape Dimension Scrambling
- **Location:** `person1_kv_engine/adapters/model_adapter.py`
- **Issue:** `HybridQwen35Adapter.wrap_model_for_aissd` called `out = out.reshape(*input_shape, -1).contiguous()` directly on the SDPA output tensor of shape `(batch, num_heads, seq_len, head_dim)`. This swapped sequence and head dimensions, scrambling hidden states prior to output gating.
- **Resolution:** Added `.transpose(1, 2)` before `.reshape(*input_shape, -1)`.

### Bug 6: Qwen3.5 Multimodal Architecture Hierarchy
- **Location:** `person1_kv_engine/adapters/model_adapter.py`
- **Issue:** In `Qwen3_5ForConditionalGeneration`, layers reside at `model.model.language_model.layers` instead of standard `model.model.layers`.
- **Resolution:** Introduced universal `get_model_layers(model)` helper that navigates transformer layer containers across causal and conditional generative architectures.

### Bug 7: In-Storage Top-K Host Telemetry Contamination
- **Location:** `person2_ssd/nvme_client.py`
- **Issue:** In `compute_topk`, `self.total_read_bytes += total_internal_k_bytes` credited internal NAND controller scans to the host PCIe read bus counter, falsely reporting candidate K bytes transferred to the host.
- **Resolution:** Scanned internal bytes remain strictly tracked in `topk_internal_scanned_bytes`, while host bus read bytes strictly account for the actual PCIe response payload (`resp_items * 12` metadata bytes), ensuring `candidate_k_bytes_to_host == 0 B`.

---

## 3. Correctness Regression Matrix

Evaluation conducted under identical conditions:
- **Prompt:** `"The PCI Express (PCIe) standard defines high-speed serial computer expansion bus technology for high performance solid state drive storage controllers."`
- **Decoding Configuration:** Greedy deterministic (`argmax`), seed=42, CPU threads=4.

| Model | Architecture | Firmware | First 8 Generated Token IDs | Match | Total Read MB | Total Write MB | Candidate K to Host |
|---|---|---|---|---|---|---|---|
| **qwen2.5-0.5b** | Qwen2.5 GQA | Disabled | `[576, 27789, 17399, 5828, 374, 264, 6146, 5828]` | **PASS** | 0.00 MB | 0.00 MB | 0 B |
| | | Enabled | `[576, 27789, 17399, 5828, 374, 264, 6146, 5828]` | **PASS** | 3.20 MB | 0.38 MB | **0 B (Zero-Bus)** |
| **tiny-mistral** | Mistral GQA | Disabled | `[31124, 8837, 26291, 13056, 29719, 23765, 27474, 14634]` | **PASS** | 0.00 MB | 0.00 MB | 0 B |
| | | Enabled | `[31124, 8837, 26291, 13056, 29719, 23765, 27474, 14634]` | **PASS** | 1.08 MB | 0.12 MB | **0 B (Zero-Bus)** |
| **qwen3-4b** | Qwen3 GQA | Disabled | `[576, 90690, 5297, 374, 6188, 311, 3410, 264]` | **PASS** | 0.00 MB | 0.00 MB | 0 B |
| | | Enabled | `[576, 90690, 5297, 374, 6188, 311, 3410, 264]` | **PASS** | 38.38 MB | 4.50 MB | **0 B (Zero-Bus)** |
| **qwen3-8b** | Qwen3 GQA | Disabled | `[90690, 374, 264, 1550, 29599, 11, 1550, 67675]` | **PASS** | 0.00 MB | 0.00 MB | 0 B |
| | | Enabled | `[90690, 374, 264, 1550, 29599, 11, 1550, 67675]` | **PASS** | 19.19 MB | 2.25 MB | **0 B (Zero-Bus)** |
| **qwen3.5-9b** | Hybrid GQA+SSM | Disabled | `[561, 87567, 5129, 369, 264, 5953, 1406, 4534]` | **PASS** | 0.00 MB | 0.00 MB | 0 B |
| | | Enabled | `[561, 87567, 5129, 369, 264, 5953, 1406, 4534]` | **PASS** | 6.25 MB | 0.50 MB | **0 B (Zero-Bus)** |
| **mistral-7b** | Mistral GQA | Disabled | `[18854, 28706, 349, 264, 1486, 28733, 12549, 10627]` | **PASS** | 0.00 MB | 0.00 MB | 0 B |
| | | Enabled | `[18854, 28706, 349, 264, 1486, 28733, 12549, 10627]` | **PASS** | 17.06 MB | 2.00 MB | **0 B (Zero-Bus)** |
| **jamba** | Hybrid Mamba+Attn | N/A | Gated model repository on Hugging Face hub | **SKIP** | - | - | - |

---

## 4. Test Suite Execution Summary

- **Component Tests (`pytest common/ person1_kv_engine/ person2_ssd/ person3_system/ -v`):**
  - **133 passed** in 39.79s (100% pass rate)
- **Integration & E2E Tests (`pytest tests/ -v`):**
  - **188 passed** in 98.78s (100% pass rate)
- **Combined Test Total:** **321 passed, 0 failed**.
