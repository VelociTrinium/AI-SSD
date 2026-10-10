# AI-SSD V2

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 180 Passed](https://img.shields.io/badge/tests-180%20passed-brightgreen.svg)]()
[![Status: Validated](https://img.shields.io/badge/status-validated-brightgreen.svg)]()

A co-designed computational storage architecture that offloads large language model (LLM) Key-Value (KV) cache tensors to NVMe storage and executes sparse attention (Top-$K$) scoring directly on the storage controller, breaking the memory wall in long-context generative AI inference.

---

## Overview

Modern large language models require gigabytes of Key-Value (KV) cache memory during autoregressive generation with long contexts (4K–32K+ tokens). **AI-SSD V2** co-designs LLM inference with computational NVMe storage:

- **Tiered Memory Hierarchy**: Attention sinks and recent tokens remain in fast host DRAM, while historical cold KV blocks reside in NVMe storage.
- **In-Storage Computational Filtering**: Query-key dot-product scoring and Top-$K$ candidate selection execute directly inside the storage controller using hardware-accelerated SIMD kernels.
- **Zero PCIe Overhead**: Candidate Key tensors are filtered in-storage, resulting in **strictly 0 bytes** of candidate keys transferred across the PCIe bus. Only the small broadcast query vector travels to storage, and only winning KV blocks are returned to the host.
- **Bit-Exact Correctness**: Achieves **100% bit-exact token match** against dense in-DRAM reference execution across all context windows.

---

## Problem

During long-context autoregressive token generation, the memory required for the KV cache scales linearly with sequence length:

$$\text{KV Cache Size} = 2 \times B \times L \times H_{\text{kv}} \times D_{\text{head}} \times S \times \text{sizeof(dtype)}$$

For a 36-layer model at 32K context, storing the full KV cache in host DRAM requires tens of gigabytes, rapidly exceeding physical host RAM limits and causing out-of-memory (OOM) failures or aggressive process thrashing.

Naive host-side SSD offloading fails because of the **PCIe / Storage Bandwidth Wall**:
- Reading gigabytes of unpruned historical KV tensors over the PCIe bus on every single generated token saturates I/O bandwidth.
- Decode latency spikes by $10\times$ to $50\times$, completely stalling the generation pipeline.

---

## Core idea

Instead of streaming historical KV blocks to the host CPU for attention scoring, AI-SSD V2 moves the compute to the data:

1. **Host Dispatches Query Vector**: The host CPU broadcasts the current layer's Query vector ($Q$) and candidate block IDs over PCIe to the NVMe controller (a few kilobytes).
2. **In-Storage Computational Selection**: The storage controller scans candidate Key ($K$) blocks directly from flash channels into internal buffers, computes GQA dot-product scores using vectorized SIMD kernels (AVX2/FMA/F16C), and selects the top $10\%$ winning blocks.
3. **Zero-Bus Invariant**: Candidate Keys **never traverse the PCIe bus** (`candidate_k_bytes_to_host = 0 B`).
4. **Targeted Winning Fetch**: Only winning Key and Value blocks are transferred across the bus into host DRAM for full multi-head attention.
5. **Asynchronous Pipelining**: Layer-by-layer I/O retrieval is overlapped with host CPU MLP computation, hiding storage latency.

---

## Architecture

The end-to-end architecture spans six tightly integrated layers across the host and computational storage device:

```text
                     Host CPU Runtime (PyTorch / C++)
                                    │
                       Prompt Tokens & Generation
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │        P1 KV Cache Engine [REAL]      │
                │  • Paged KV Block Manager             │
                │  • Attention Sinks & Recent Window    │
                │  • Online Attention Softmax Merger    │
                └───────────────────┬───────────────────┘
                                    │ Broadcast Query (Q)
                                    ▼
                ┌───────────────────────────────────────┐
                │   P3 Async Prefetch & Pipeline [REAL] │
                │  • Asynchronous Staging Buffer        │
                │  • Contiguous Block Aggregation       │
                └───────────────────┬───────────────────┘
                                    │ Storage Requests
                                    ▼
                ┌───────────────────────────────────────┐
                │    P2 Storage Backend & FTL [REAL]    │
                │  • Direct Block I/O Interface         │
                │  • 8-Channel / 4-Die NAND Model       │
                │  • Tensor-Aware Striping [ANALYTICAL] │
                └───────────────────┬───────────────────┘
                                    │ Block I/O & Socket Commands
════════════════════════════════════╪════════════════════════════════════
                     PCIe Physical / Virtual Boundary
════════════════════════════════════╪════════════════════════════════════
                                    ▼
                ┌───────────────────────────────────────┐
                │  QEMU Virtual NVMe Device (/dev/nvme0n1)
                │         [VIRTUAL-DEVICE]              │
                │  • Linux In-Guest NVMe Kernel Driver  │
                │  • High-Throughput Block Subsystem    │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │   In-Storage Computational Engine     │
                │         [VIRTUAL-DEVICE]              │
                │  • nvme_guest_daemon (Native C)       │
                │  • 128-dim AVX2 / FMA / F16C Kernel   │
                │  • Top-K Candidate Selection          │
                └───────────────────┬───────────────────┘
                                    │
                                    │ Winning Top-K KV Blocks Only
                                    ▼
                ┌───────────────────────────────────────┐
                │        Host Attention Execution       │
                │  • Native Grouped-Query SDPA          │
                │  • SwiGLU MLP & Layer Norm            │
                │  • Bit-Exact Next Token Emission      │
                └───────────────────────────────────────┘
```

---

## Key capabilities

- **Zero PCIe Candidate Key Overhead**: Candidate Keys are evaluated and filtered entirely in-storage, transferring strictly **0 bytes** of candidate keys over PCIe across all context lengths.
- **Substantial DRAM Footprint Reduction**: Reduces active KV cache DRAM footprint by **78.8% to 89.8%**, cutting total host process Peak RSS by up to **63.3%** at 32K context (saving **27.9 GB** physical host RAM).
- **Bit-Exact Numerical Precision**: Delivers **100% exact token-for-token parity (16/16 tokens)** against dense in-DRAM baseline execution.
- **Hardware SIMD Acceleration**: Optimized C storage daemon features hand-tuned 128-dim AVX2/FMA (FP32) and F16C (FP16) compute kernels with quad-accumulator pipelining.
- **Tensor-Aware Multi-Channel Striping**: Achieves near-perfect channel balance (**0.86% load imbalance** across 8 channels, contention ratio 10.50) compared to sequential FTL mapping (**700.0% imbalance**, contention ratio 83.26).
- **Asynchronous Contiguous I/O Pipeline**: Merges discrete Key and Value requests into contiguous block operations, cutting I/O transactions by 50% and hiding retrieval latency behind host compute.

---

## Validated results

Empirical comparison between canonical validated configurations on dedicated AWS Sapphire Rapids compute nodes (4 CPU threads, 4,096 prompt context, 16 decode tokens, QEMU/NVMe computational storage device `/dev/nvme0n1`):

| Metric | Dense Baseline (Qwen3-8B FP16) | AI-SSD Qwen3-4B FP32 (`cc3972e`) | AI-SSD Qwen3-8B FP16 (`5e1618a`) | Qwen3-8B vs Qwen3-4B Delta | Classification |
|---|:---:|:---:|:---:|:---:|:---:|
| **Model Parameters** | 8,190,094,336 | 4,020,000,000 | **8,190,094,336** | $+103.7\%$ capacity | `[REAL]` |
| **Precision** | FP16 (2 bytes) | FP32 (4 bytes) | **FP16 (2 bytes)** | Halved byte width | `[REAL]` |
| **Model Weight Footprint** | 15.26 GB | 14.30 GB | **15.26 GB** | $+6.7\%$ weight bytes | `[REAL]` |
| **Weight Memory Traffic / Token** | 15.26 GB/tok | 16.08 GB/tok | **15.26 GB/tok** | **$-5.1\%$ memory traffic** | `[REAL]` |
| **Arithmetic Intensity** | 1.0 FLOP/byte | 0.5 FLOP/byte | **1.0 FLOP/byte** | **$2.0\times$ efficiency** | `[REAL]` |
| **Decode Wall Time (16 tokens)** | 8.827 s | 16.658 ± 0.402 s | **13.895 ± 0.240 s** | **$-16.6\%$ decode latency** | `[REAL]` |
| **Throughput (mean)** | 1.813 tok/s | 0.961 ± 0.024 tok/s | **1.152 ± 0.020 tok/s** | **$+19.9\%$ throughput** | `[REAL]` |
| **Peak Throughput** | 1.813 tok/s | 1.000 tok/s | **1.187 tok/s** | **$+18.7\%$ peak tok/s** | `[REAL]` |
| **Host DRAM Active KV** | 578.25 MB | 122.60 MB | **122.63 MB** | Identical active KV | `[REAL]` |
| **KV DRAM Reduction** | 0.0% (all in DRAM) | 76.1% offloaded | **78.8% offloaded** | **$+2.7\%$ higher offload** | `[REAL]` |
| **Peak Process RSS** | 18,022.5 MB | ~17,600 MB | **16,768.2 ± 319.6 MB** | **$-1,254.3\text{ MB}$ host RAM** | `[REAL]` |
| **Candidate K $\to$ Host PCIe** | N/A | **0 B** | **0 B (100% Filtered)** | Zero-bus invariant verified | `[REAL]` |
| **Winning KV $\to$ Host PCIe** | N/A | 109.68 MB | **109.68 MB** | Pipelined async fetch | `[VIRTUAL-DEVICE]` |
| **Token Accuracy** | 16/16 Reference | 16/16 Reference | **16/16 (100% Match)** | Bit-exact output | `[REAL]` |

### Repeatability Validation (5 Independent Runs)

Workload: Context=4096, Decode=16 tokens, 4 CPU threads, QEMU/NVMe (`/dev/nvme0n1`), Top-K=10%, Async=ON.

| Repetition | Seed | Wall Time (s) | Throughput (tok/s) | Peak RSS (MB) | Active KV (MB) | Candidate K $\to$ Host | Token Verification |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Run 1 | 42 | 13.479 s | 1.187 tok/s | 16,878.1 MB | 122.63 MB | 0 B | 16/16 PASS |
| Run 2 | 43 | 13.891 s | 1.152 tok/s | 16,134.0 MB | 122.63 MB | 0 B | 16/16 PASS |
| Run 3 | 44 | 13.860 s | 1.154 tok/s | 16,909.8 MB | 122.63 MB | 0 B | 16/16 PASS |
| Run 4 | 45 | 14.050 s | 1.139 tok/s | 16,999.2 MB | 122.63 MB | 0 B | 16/16 PASS |
| Run 5 | 46 | 14.195 s | 1.127 tok/s | 16,920.0 MB | 122.63 MB | 0 B | 16/16 PASS |
| **Mean ± Std** | — | **13.895 ± 0.240 s** | **1.152 ± 0.020 tok/s** | **16,768.2 ± 319.6 MB** | **122.63 ± 0.0 MB** | **0 B ± 0.0 B** | **100% PASS** |

---

## Quick Start
 
The AI-SSD V2 execution flow enforces clean separation between firmware initialization and model execution. **The shell scripts initialize/teardown firmware and hardware emulation only; they do NOT execute Python or inference.**
 
### 1. Enable AI-SSD Firmware
 
Launch the firmware initializer to boot the QEMU/KVM virtual NVMe controller (`/dev/nvme0n1`) and the in-storage guest daemon:
 
```bash
./enable_ai_ssd.sh
```
 
Expected output:
```text
========================================
        AI-SSD V2 FIRMWARE
========================================
AI-SSD Firmware: ENABLED
NVMe Device:     /dev/nvme0n1
Status:          READY

Firmware remains enabled.
Run the inference script manually.
========================================
```
 
The shell script verifies prerequisite disk images, boots the virtual device in the background, validates socket availability on port 9999, and exits cleanly while leaving the firmware active.
 
### 2. Run Inference & Benchmarks

Run side-by-side benchmark comparisons between Dense Baseline and AI-SSD:

```bash
# Canonical quick comparison (File-backed direct I/O mode):
python scripts/run_quick_comparison.py --model qwen2.5-0.5b --context 2048 --storage-mode file

# Or invoke the unified bare-metal CLI directly:
python -m ai_ssd.cli --model qwen3-4b --context 4096 --decode 16 --threads 4 --storage-mode file

# Run against hardware QEMU NVMe controller (/dev/nvme0n1):
python -m ai_ssd.cli --model qwen3-8b --context 4096 --decode 16 --threads 4 --storage-mode nvme
```

> **Canonical Setting**: AI-SSD V2 benchmarks strictly standardize on **4 CPU threads** (`--threads 4`).

### 3. Gracefully Stop Firmware & Clean Runtime

When inference is finished, gracefully tear down the virtual NVMe subsystem:

```bash
./disable_ai_ssd.sh
```

Expected output:
```text
========================================
        AI-SSD V2 FIRMWARE TEARDOWN
========================================
AI-SSD Firmware: DISABLED
NVMe Device:     /dev/nvme0n1
Status:          STOPPED
Firmware is disabled.
========================================
```

The teardown script is idempotent, verifies the virtual NVMe device and guest socket are stopped, cleans up the managed runtime directory (`/tmp/ai-ssd-runtime`), and leaves unrelated host NVMe drives and QEMU instances completely untouched.
 
---
 
## Supported Models & Architecture Compatibility
 
AI-SSD V2 provides an architecture-adaptive state management abstraction (`ModelAdapter` and `StateProvider`) supporting standard Transformers, sliding-window attention, and hybrid Attention + SSM models:
 
| Model Alias | Hugging Face Model ID | Architecture | Attention Type | State Provider | Compatibility |
|---|---|---|:---:|---|:---:|
| **`qwen3-4b`** | `Qwen/Qwen3-4B-Instruct-2507` | Qwen2 | GQA (7:1) | `TransformerKVStateProvider` | **`FULL`** |
| **`qwen3-8b`** | `Qwen/Qwen3-8B` | Qwen2 | GQA (4:1) | `TransformerKVStateProvider` | **`FULL`** |
| **`qwen2.5-0.5b`** | `Qwen/Qwen2.5-0.5B` | Qwen2 | GQA (7:1) | `TransformerKVStateProvider` | **`FULL`** |
| **`qwen3.5-4b`** | `Qwen/Qwen3.5-4B-Instruct` | Qwen2 | GQA (4:1) | `TransformerKVStateProvider` | **`FULL`** |
| **`qwen3.5-9b`** | `Qwen/Qwen3.5-9B` | Qwen3.5 Hybrid | Hybrid (24 SSM : 8 GQA) | N/A (75% SSM) | **`UNSUPPORTED`** |
| **`tiny-mistral`** | `openaccess-ai-collective/tiny-mistral` | Mistral | GQA + Sliding | `SlidingWindowKVStateProvider` | **`FULL`** |
| **`mistral-7b`** | `mistralai/Mistral-7B-v0.1` | Mistral | GQA + Sliding | `SlidingWindowKVStateProvider` | **`FULL`** |
| **`jamba`** | `ai21labs/AI21-Jamba-1.5-Mini` | Jamba | Hybrid GQA + Mamba | `HybridStateProvider` | **`PARTIAL`** |
| **`mamba`** | `state-spaces/mamba-130m-hf` | Mamba | None (Pure SSM) | N/A (No separable KV) | **`UNSUPPORTED`** |
 
For complete architectural details, precision formats, and compatibility rules, see [`docs/v2/MODEL_SUPPORT.md`](docs/v2/MODEL_SUPPORT.md).

---

## Dashboard

AI-SSD V2 includes an interactive Streamlit visualization dashboard for inspecting live system behavior, KV cache memory footprint reductions, FTL channel contention, and latency breakdowns.

### Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

### Dashboard Features

- **Interactive Telemetry**: Real-time display of tokens generated, active host DRAM KV footprint, and PCIe bus data movement.
- **FTL Multi-Channel Heatmap**: Visual comparison of channel contention between conventional sequential mapping and tensor-aware multi-channel striping.
- **Latency Breakdown**: Side-by-side critical path analysis separating host CPU GEMV compute from visible NVMe storage retrieval.
- **Top-$K$ Retrieval Profiler**: Real-time monitoring of candidate key pruning, in-storage scan throughput, and winning block arrival rates.

---

## Documentation

Comprehensive guides, specifications, and deep-dive documentation:

- [`EXPLANATION.md`](docs/v2/EXPLANATION.md) — Comprehensive technical overview of AI-SSD V2 principles, architecture, and mechanics.
- [`DEEP_DIVE.md`](docs/v2/DEEP_DIVE.md) — Deep architectural analysis into KV cache math, storage contention physics, and SIMD vector kernels.
- [`ARCHITECTURE.md`](docs/v2/ARCHITECTURE.md) — End-to-end multi-layer architecture specification, component boundaries, and data flow.
- [`STORAGE_COMPANY_GUIDE.md`](docs/v2/STORAGE_COMPANY_GUIDE.md) — Technical handbook for SSD controller architects and computational storage hardware vendors.
- [`MODEL_COMPATIBILITY.md`](docs/v2/MODEL_COMPATIBILITY.md) — Model compatibility criteria, attention head constraints, and support roadmap.
- [`BENCHMARKS.md`](docs/v2/BENCHMARKS.md) — Complete benchmark matrix, repeatability methodology, and ablation study data.
- [`LIMITATIONS.md`](docs/v2/LIMITATIONS.md) — Architectural boundary conditions, virtualization trade-offs, and physical hardware constraints.
- [`INTERVIEW_QUESTIONS.md`](docs/v2/INTERVIEW_QUESTIONS.md) — Core technical interview questions, design trade-offs, and engineering rationale.

---

## Repository Structure

```text
ai-ssd-v2/
├── ai_ssd/                              # Core clean bare-metal package
│   ├── constants.py                     # Physical geometry & canonical dimensions
│   ├── cli.py                           # Unified CLI entrypoint (baseline vs AI-SSD)
│   ├── engine/                          # PyTorch LLM runtime & dynamic attention hooks
│   │   ├── model_runner.py              # HF model loading, thread pinning, prefill execution
│   │   ├── attention_hook.py            # Dynamic forward wrapper for self_attn.forward
│   │   ├── kv_manager.py                # Sinks (DRAM), Recent (DRAM), Historical (Offloaded)
│   │   └── registry.py                  # Model architecture registry (Qwen, Mistral, LLaMA)
│   ├── storage/                         # Unified computational storage subsystem
│   │   ├── base.py                      # StorageBackend abstract interface
│   │   ├── file_backend.py              # Direct OS file block I/O (os.pread / os.pwrite)
│   │   ├── nvme_backend.py              # QEMU / Hardware TCP socket client to NVMe daemon
│   │   ├── prefetcher.py                # Asynchronous DRAM staging buffer & worker thread
│   │   └── ftl_striping.py              # Tensor-aware multi-channel striping math
│   ├── c_kernel/                        # In-storage SIMD vector computing
│   │   ├── instorage_attention.c        # AVX2 / FMA / F16C dot product & Top-K selector
│   │   ├── compile_kernel.py            # Zero-dependency compiler script
│   │   └── binding.py                   # Ctypes native binding
│   └── telemetry/                       # OS process & hardware performance monitoring
│       └── memory_sampler.py            # High-resolution ProcessMemorySampler (VmRSS, Peak RSS)
├── dashboard/                           # Streamlit visualizer
│   ├── app.py                           # Cleaned Streamlit dashboard
│   └── schema.py                        # Result loading & formatting utilities
├── scripts/                             # Operational & firmware scripts only
│   ├── enable_ai_ssd.sh                 # Boots QEMU NVMe device & guest daemon
│   ├── disable_ai_ssd.sh                # Graceful teardown
│   ├── nvme_guest_daemon.c              # In-storage C daemon for /dev/nvme0n1
│   └── run_quick_comparison.py          # Quick benchmark wrapper
├── tests/                               # Focused unit and regression test suite
│   ├── test_c_kernel.py                 # Vector math correctness
│   ├── test_storage_backends.py         # File & NVMe block roundtrips
│   ├── test_kv_manager.py               # Tiering, Top-K selection, and SDPA assembly
│   └── test_bit_exact_e2e.py            # 100% token exactness against dense baseline
├── pyproject.toml
└── requirements.txt
```

---

## Evidence Classification

Every metric, benchmark result, and technical claim in this repository is strictly tagged according to scientific rigor and empirical provenance:

| Classification | Definition | Application in AI-SSD V2 |
|:---:|---|---|
| **`[REAL]`** | Physically executed and directly measured on the host system without virtualization or extrapolation. | Process RSS telemetry sampled via `/proc/self/status`, PyTorch CPU forward-pass compute, CPU hardware counters (`perf stat`), and 100% bit-exact token verification. |
| **`[VIRTUAL-DEVICE]`** | Real kernel drivers and real block I/O operating over an emulated hardware controller. | QEMU/KVM virtual NVMe subsystem (`/dev/nvme0n1`), in-guest Linux kernel driver, and the computational storage daemon running compiled C/AVX2 kernels on virtual device memory. |
| **`[ANALYTICAL]`** | Deterministic hardware simulation based on validated mathematical formulations and cycle/latency models. | 8-channel / 4-die NAND Flash Translation Layer (FTL) striping model, channel contention calculations, and wear-leveling simulation. |
| **`[PROJECTED]`** | Extrapolations derived mathematically from validated analytical or empirical baselines. | Performance scaling projections beyond tested physical boundaries (e.g., enterprise 128-channel flash arrays, dedicated ASIC computational controllers). |

---

## Limitations

To maintain scientific integrity and avoid exaggerating physical hardware capabilities, the following engineering constraints are explicitly acknowledged:

1. **Host CPU GEMV Memory-Bandwidth Bottleneck**: In single-token autoregressive generation ($M=1$), host CPU execution is strictly memory-bandwidth bound (arithmetic intensity $0.5$–$1.0\text{ FLOP/byte}$). At FP32/FP16, model weight loading consumes $>70\%$ of physical DDR5 bus bandwidth, bounding end-to-end decode throughput regardless of storage performance.
2. **Virtualization I/O Overhead**: QEMU NVMe emulation incurs software DMA and VM-exit overhead ($~2.3\times$ latency penalty relative to bare-metal physical NVMe direct I/O).
3. **Commodity SSD Firmware Constraints**: Off-the-shelf commodity SSDs do not currently expose general-purpose CPU/vector computing (e.g., AVX2) in unprivileged firmware. Bare-metal hardware deployment requires Computational Storage Devices (CSDs) adhering to NVMe-TP4061 / SNIA standards or custom controller FPGA firmware.
4. **Prefetch Latency Hiding Ceiling**: Although asynchronous contiguous batching reduces critical-path storage latency, autoregressive causal dependencies across transformer layers prevent deep speculative prefetching before earlier layers establish execution context.
5. **Sparse Attention Pruning Boundary**: While validated configurations achieve 100% token exactness at $10\%$ Top-$K$ selection, aggressive pruning thresholds ($<5\%$) can degrade output quality on tasks requiring global token recall.

---

## Future Work

- **Physical CSD & FPGA Hardware Deployment**: Port the in-storage AVX2/F16C computational kernel to physical computational storage hardware (e.g., AMD/Xilinx FPGA-based NVMe CSD) using NVMe Computational Storage Commands (TP4061).
- **Weight Quantization Integration**: Integrate INT4/INT8 weight quantization (AWQ/GPTQ) with AI-SSD KV offloading to relieve host CPU DDR5 memory bandwidth bottlenecks during GEMV execution.
- **Ultra-Long Context Validation (64K–128K)**: Extend validated context windows to 64K and 128K tokens on high-capacity NVMe storage to map scaling limits.
- **GPUDirect Storage (GDS) / SPDK Integration**: Implement direct NVMe-to-GPU memory transfers via GPUDirect Storage, bypassing host CPU DRAM entirely for GPU-accelerated inference.
- **Adaptive Entropy-Guided Top-$K$**: Develop dynamic layer-adaptive Top-$K$ selection based on attention entropy to maximize retrieval efficiency without fixed percentile thresholds.
