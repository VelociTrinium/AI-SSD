# Architecture-Adaptive Model State Support in AI-SSD V2

## Overview

AI-SSD V2 decouples model architectures from underlying computational storage mechanisms. Instead of hard-coding assumptions that every language model uses standard Transformer Key-Value tensors structured as:

$$\text{layers} \times \text{KV heads} \times \text{sequence} \times \text{head\_dim}$$

the runtime provides an **Architecture-Adaptive State Management Layer** composed of two primary abstractions:

1. **`ModelAdapter`**: Encapsulates model-specific internal wiring, forward-pass hooks, rotary positional embeddings (RoPE), and layer wrapping (`QwenAdapter`, `MistralAdapter`, `HybridJambaAdapter`).
2. **`StateProvider`**: Encapsulates serialization, blockification, storage offload, Top-$K$ retrieval, and working-set reconstruction for reusable sequence states (`TransformerKVStateProvider`, `SlidingWindowKVStateProvider`, `HybridStateProvider`).

---

## 1. Supported Model Architectures

AI-SSD classifies model architectures into three strict compatibility tiers:

```text
       Hugging Face / PyTorch Model
                    │
                    ▼
            ModelAdapter
   (QwenAdapter, MistralAdapter, HybridJambaAdapter)
                    │
                    ▼
              StateProvider
   (TransformerKV, SlidingWindowKV, HybridState)
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
    Host DRAM Tier     AI-SSD NVMe Storage Tier
   • Attention Sinks   • 8 KiB Flash Blocks (4K K + 4K V)
   • Recent Window     • In-Storage SIMD Top-K Selection
   • SSM / Recurrent   • Zero Candidate Keys to Host PCIe
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
          State Reconstruction
                    │
                    ▼
         Layer Attention / SDPA
```

### Compatibility Tiers:
- **`FULL`**: The model's reusable state consists of attention Key-Value activations. Historical blocks are offloaded to AI-SSD storage, and Top-$K$ selection executes directly on the storage controller (`candidate_k_bytes_to_host = 0 B`).
  - Standard Multi-Head Attention (MHA)
  - Grouped-Query Attention (GQA)
  - Multi-Query Attention (MQA)
  - Sliding-Window Attention (Mistral-style local attention span)
- **`PARTIAL`**: Hybrid architectures (e.g., Attention + Mamba/SSM). Attention Key-Value tensors are offloaded to AI-SSD storage, while fixed-size recurrent/SSM hidden states remain resident in Host DRAM.
- **`UNSUPPORTED`**: Pure state-space or RNN architectures (e.g., Mamba, RWKV). These models maintain compact $O(1)$ recurrent hidden states rather than growing sequence-indexed Key-Value caches, offering no separable block structure for Top-$K$ attention offload.

---

## 2. Current Verified Models

The following models are verified and registered in `person1_kv_engine/adapters/registry.py`:

| Model Identifier | Hugging Face ID | Architecture Family | Precision | Attention Type | State Provider | Compatibility Level | Verified Status |
|---|---|---|:---:|:---:|---|:---:|:---:|
| `qwen3-4b` | `Qwen/Qwen3-4B-Instruct-2507` | Transformer (Qwen2) | FP32 | GQA (16:4) | `TransformerKVStateProvider` | **`FULL`** | Verified (4K–32K) |
| `qwen3-8b` | `Qwen/Qwen3-8B` | Transformer (Qwen2) | FP16 | GQA (32:8) | `TransformerKVStateProvider` | **`FULL`** | Verified (4K–16K) |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B` | Transformer (Qwen2) | FP32 | GQA (14:2) | `TransformerKVStateProvider` | **`FULL`** | Verified (512–4K) |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B-Instruct` | Transformer (Qwen2) | FP32 | GQA (16:4) | `TransformerKVStateProvider` | **`FULL`** | Architecture Ready |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | Hybrid (Linear SSM + GQA) | FP16 | Hybrid (24 SSM : 8 GQA) | `HybridStateProvider` (`HybridQwen35Adapter`) | **`PARTIAL`** | Verified (8 GQA offloaded, 24 SSM resident) |
| `tiny-mistral` | `openaccess-ai-collective/tiny-mistral` | Transformer (Mistral) | FP32 / FP16 | GQA (16:4) + Sliding | `SlidingWindowKVStateProvider` | **`FULL`** | Verified (Unit/E2E) |
| `mistral-7b` | `mistralai/Mistral-7B-v0.1` | Transformer (Mistral) | FP16 | GQA (32:8) + Sliding | `SlidingWindowKVStateProvider` | **`FULL`** | Architecture Ready |
| `jamba` | `ai21labs/AI21-Jamba-1.5-Mini` | Hybrid (Transformer + SSM) | FP16 | GQA (32:8) + Mamba | `HybridStateProvider` | **`PARTIAL`** | Architecture Ready |
| `mamba` | `state-spaces/mamba-130m-hf` | Pure State-Space (Mamba) | FP32 | None (SSM) | N/A | **`UNSUPPORTED`** | Rejected (Pure SSM) |

> **Note on Verification**: `qwen3-4b` (FP32) and `qwen3-8b` (FP16) are canonical released benchmarks with end-to-end hardware telemetry. `tiny-mistral` is verified via unit and adapter suites. Larger models (`mistral-7b`, `jamba`) are architecture-validated but require explicit model weight downloads and sufficient host RAM before full execution.

---

## 3. How to Download a Model

The AI-SSD execution pipeline loads models using Hugging Face's `transformers` library via `AutoModelForCausalLM.from_pretrained()`.

### Download Methods:
1. **Direct HF ID (Automatic Download)**: Passing a Hugging Face repository ID will download weights automatically into the standard Hugging Face cache (`~/.cache/huggingface/hub/`).
2. **Local Directory (Pre-downloaded Weights)**: You can pass a local filesystem path directly to `--model /path/to/weights`.

### Tested Models & Download Commands:

#### 1. Qwen3-4B-Instruct-2507 (Canonical FP32)
- **Hugging Face ID**: `Qwen/Qwen3-4B-Instruct-2507`
- **Download CLI**:
  ```bash
  python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Instruct-2507', torch_dtype='auto')"
  ```
- **Storage Required**: ~16.0 GB on disk (FP32 safetensors)
- **Host RAM Required**: ~18 GB RAM for CPU execution
- **Default Precision**: `fp32`

#### 2. Qwen3-8B (Canonical FP16)
- **Hugging Face ID**: `Qwen/Qwen3-8B`
- **Download CLI**:
  ```bash
  python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-8B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B', torch_dtype='float16')"
  ```
- **Storage Required**: ~16.5 GB on disk (FP16 safetensors)
- **Host RAM Required**: ~18 GB RAM for CPU execution
- **Default Precision**: `fp16`

#### 3. Tiny-Mistral (Lightweight Verification Model)
- **Hugging Face ID**: `openaccess-ai-collective/tiny-mistral`
- **Download CLI**:
  ```bash
  python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('openaccess-ai-collective/tiny-mistral'); AutoModelForCausalLM.from_pretrained('openaccess-ai-collective/tiny-mistral')"
  ```
- **Storage Required**: ~450 MB on disk
- **Host RAM Required**: ~1 GB RAM
- **Default Precision**: `fp32`

#### 4. Qwen3.5-9B (Hybrid Attention + DeltaNet SSM)
- **Hugging Face ID**: `Qwen/Qwen3.5-9B`
- **Compatibility Tier**: **`PARTIAL`**
- **Architecture**: `HybridQwen35Adapter`
- **Geometry**: 32 hidden layers (8 full GQA Attention layers + 24 linear attention / DeltaNet SSM recurrent layers), 16 query heads, 4 KV heads, head dimension 256, partial rotary factor 0.25 (first 64 dimensions rotated), sigmoid attention output gating.
- **Offload Mechanism**:
  - **Full Attention Layers (3, 7, 11, 15, 19, 23, 27, 31)**: Key-Value activation pages are serialized into 16-token flash blocks and offloaded to AI-SSD flash storage. Under computational storage, candidate Key filtering evaluates in storage (`candidate_k_bytes_to_host = 0 B`).
  - **Linear Attention / SSM Layers (0–2, 4–6, 8–10, 12–14, 16–18, 20–22, 24–26, 28–30)**: Compact recurrent and convolution hidden states remain resident in Host DRAM.
- **Download CLI**:
  ```bash
  python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-9B', torch_dtype='float16')"
  ```
- **Storage Required**: ~19.3 GB on disk (FP16 safetensors)
- **Host RAM Required**: ~18 GB RAM for CPU execution
- **Default Precision**: `fp16`

---

## 4. How to Add a New Model

### 4.1 Automated One-Command Model Onboarding (`scripts/add_model.py`)

AI-SSD V2 provides an automated CLI to onboard any Hugging Face model or local weights directory with zero manual code editing:

```bash
# Automated onboarding (checks validity, checks cache, downloads if needed, registers)
python scripts/add_model.py <model_name_or_hf_id>

# Or via demo_inference:
python scripts/demo_inference.py --add-model <model_name_or_hf_id>
```

#### What `scripts/add_model.py` Does Automatically:
1. **Validates Model Identifier**: Checks the repository on Hugging Face Hub (or verifies local folder).
2. **Local Cache Check (Zero Redundant Downloads)**:
   - If the model is already downloaded in `~/.cache/huggingface/hub/` or a local path, it **skips downloading**.
   - If weights are missing, it downloads safetensors and tokenizer weights automatically.
3. **Architecture & Geometry Extraction**:
   - Inspects number of layers, query heads, KV heads, head dimension, hidden size, and sliding window.
   - Calculates attention type (MHA, GQA with ratio, MQA).
   - Estimates model parameter count (e.g., `1.5B`, `9.0B`).
   - Determines AI-SSD compatibility tier (`FULL`, `PARTIAL`, `UNSUPPORTED`).
4. **Persistent Registration**:
   - Assigns a clean, exact `model-key` (e.g., `qwen2.5-1.5b`) and saves it persistently to `person1_kv_engine/adapters/custom_models.json`.
5. **Ready for Immediate Demo Inference**:
   - The registered model appears directly in `python scripts/demo_inference.py --list-models`.
   - Run inference using the exact key: `python scripts/demo_inference.py --model <model-key>`.

#### Examples:
```bash
# Onboard a model from Hugging Face
python scripts/add_model.py Qwen/Qwen2.5-0.5B

# Onboard with custom model key
python scripts/add_model.py mistralai/Mistral-7B-v0.1 --model-key my-mistral-7b

# View all registered models (built-in + user added)
python scripts/demo_inference.py --list-models

# Run live inference with the exact registered key
python scripts/demo_inference.py --model qwen2.5-0.5b --context 4096
```

---

### 4.2 Manual / Deep Architecture Adaptation (12 Steps)

To integrate fundamentally novel non-Transformer architectures into the AI-SSD architecture-adaptive pipeline, follow these 12 steps:

```text
Step 1: Identify architecture family (Transformer, Sliding-Window, Hybrid, SSM)
Step 2: Inspect AutoConfig via `AutoConfig.from_pretrained(model_id)`
Step 3: Determine state type (separable KV cache vs recurrent state)
Step 4: Determine KV tensor layout (num_layers, num_kv_heads, head_dim)
Step 5: Determine attention geometry (MHA, GQA, or MQA)
Step 6: Determine sliding window span (if applicable)
Step 7: Implement or reuse `ModelAdapter` (`person1_kv_engine/adapters/model_adapter.py`)
Step 8: Implement or reuse `StateProvider` (`person1_kv_engine/adapters/state_provider.py`)
Step 9: Register the model in `KNOWN_MODELS` (`person1_kv_engine/adapters/registry.py`)
Step 10: Run unit and adapter compatibility tests
Step 11: Run numerical correctness validation against dense baseline
Step 12: Run end-to-end performance benchmarks
```

#### Files to Modify for Custom Adapters:
1. **`person1_kv_engine/adapters/model_adapter.py`**: Add layer hooks if the model does not inherit standard Hugging Face attention patterns (`self_attn.forward`).
2. **`person1_kv_engine/adapters/registry.py`**: Add entry to `KNOWN_MODELS` with layer count, head count, head dimension, and compatibility classification.
3. **`tests/test_model_compatibility.py`**: Add unit tests validating config detection and adapter creation.

---

## 5. Model Inspection Before Inference

Before downloading heavy weights or launching hardware emulation, inspect the model config using the AI-SSD CLI:

### 1. List All Registered Models
```bash
python scripts/demo_inference.py --list-models
```

### 2. Inspect a Model Architecture & Compatibility
```bash
# Inspect a registered model key
python scripts/demo_inference.py --inspect-model qwen3-4b

# Inspect a sliding-window Mistral model
python scripts/demo_inference.py --inspect-model tiny-mistral

# Inspect an arbitrary Hugging Face model repository ID
python scripts/demo_inference.py --inspect-model mistralai/Mistral-7B-v0.1
```

Example Inspection Output:
```text
==================================================
         AI-SSD MODEL ARCHITECTURE INSPECTION
==================================================
Target Model:          tiny-mistral
Resolved Model ID:     openaccess-ai-collective/tiny-mistral
Architecture Family:   transformer
Architecture Type:     mistral
Parameters:            0.21B
Attention Type:        GQA
Layers:                8
Attention Heads:       16
KV Heads:              4
Head Dimension:        32
Hidden Size:           512
Sliding Window:        4096
Default Precision:     FP32
Compatibility Level:   FULL
Compatibility Reason:  Validated Mistral architecture with GQA and sliding-window support
==================================================
```

---

## 6. Pre-Inference Test Checklist

Before running full live inference benchmarks, ensure each verification gate passes:

- [ ] **Python Environment**: Dependencies installed (`torch`, `transformers`, `pytest`, `numpy`).
- [ ] **Model Inspectable**: `python scripts/demo_inference.py --inspect-model <model_id>` returns without errors.
- [ ] **Unit Tests**: Full regression test suite passes (`pytest tests/`).
- [ ] **Adapter Tests**: `pytest tests/test_model_compatibility.py` passes all 16 tests.
- [ ] **Firmware Enable Healthy**: `./enable_ai_ssd.sh` reports `AI-SSD Firmware: ENABLED` and `Status: READY`.
- [ ] **Firmware Disable Healthy**: `./disable_ai_ssd.sh` reports `AI-SSD Firmware: DISABLED` and `Status: STOPPED`.
- [ ] **Idempotency Validated**: Running `./disable_ai_ssd.sh` multiple times exits cleanly with code 0.
- [ ] **Device Present**: `/dev/nvme0n1` is accessible while enabled and unmounted while disabled.

---

## 7. Test Levels & Execution Hierarchy

Testing must follow strict chronological levels. Never run performance benchmarks before correctness is proven.

```text
Level 1: Unit & Schema Tests
         pytest person1_kv_engine/ person2_ssd/ person3_system/
         Verifies KV block slicing, FTL striping math, and I/O request schemas.

Level 2: Architecture & Adapter Compatibility Tests
         pytest tests/test_model_compatibility.py
         Verifies AutoConfig detection, GQA ratio calculation, and StateProviders.

Level 3: Storage Integration & Hardware Block Roundtrip
         pytest tests/test_nvme_block_roundtrip.py tests/test_nvme_backend_integration.py
         Verifies real NVMe block device I/O and zero-corruption block roundtrips.

Level 4: Bit-Exact Numerical Parity & Invariant Validation
         pytest tests/test_phase9_final_validation.py
         Verifies 16/16 exact token match and candidate_k_bytes_to_host == 0.

Level 5: Live Inference Benchmark & Telemetry Profiling
         ./enable_ai_ssd.sh
         python scripts/demo_inference.py --model qwen3-4b --context 4096
         ./disable_ai_ssd.sh
         Measures wall-time latency, peak RSS memory, and flash channel balance.
```

---

## 8. Canonical Pre-Inference Command Sequence

Copy-paste workflow for developers and benchmark evaluators:

```bash
# 1. Verify repository status
git status

# 2. Run unit and compatibility test suite
pytest tests/test_model_compatibility.py

# 3. Inspect target model
python scripts/demo_inference.py --inspect-model qwen3-4b

# 4. Enable AI-SSD Virtual Device Firmware
./enable_ai_ssd.sh

# 5. Execute Canonical Benchmark Inference (4 CPU threads)
python scripts/demo_inference.py --model qwen3-4b --context 4096 --decode-tokens 16 --threads 4

# 6. Verify Firmware Teardown
./disable_ai_ssd.sh
```

---

## 9. Adding a New Model — Acceptance Criteria

To designate a newly added model as **`FULL`** compatibility:

1. **Architecture Detection**: `ModelRegistry.detect_model_config(model_id)` must correctly extract layers, KV heads, and head dimensions.
2. **State Separation**: Reusable sequence states must be cleanly extracted into 16-token chunks.
3. **Storage Offload**: Key-Value blocks must be serialized into 8 KiB units (4 KiB Key + 4 KiB Value) and written to `/dev/nvme0n1`.
4. **Computational Filtering**: Query vectors must be sent to storage, and Candidate Keys must evaluate in-controller without transferring candidate key bytes across PCIe (`candidate_k_bytes_to_host = 0 B`).
5. **Exact Numerical Parity**: The generated token sequence across 16 decode steps must match the dense baseline 100%.
6. **Regression Tests**: A unit test validating the model configuration must be committed to `tests/test_model_compatibility.py`.

---

## 10. Resource Warnings & Hardware Requirements

Before running heavy models on host hardware:

- **Host RAM Consumption**: Standard CPU inference loads all model weights into host memory. Qwen3-8B FP16 requires ~16.5 GB of RAM; Qwen3-4B FP32 requires ~16.0 GB of RAM.
- **Precision Matters**: FP32 doubles weight memory and KV cache sizes compared to FP16/BF16.
- **Context Memory Scaling**: At 32,768 tokens, an uncompressed dense KV cache consumes >30 GB of RAM. AI-SSD offloads ~80% of this state to storage, lowering process peak RSS by up to 27.9 GB.
- **AWS Sapphire Rapids Constraints**: On memory-constrained cloud instances, verify available memory with `free -h` before loading 8B+ parameter models.
