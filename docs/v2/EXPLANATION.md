# AI-SSD V2: Intuitive Technical Explanation

**Audience**: Professors, technical interviewers, and engineers discovering the project for the first time.  
**Core Purpose**: Explain why AI-SSD exists, what problem it solves, how it works at a conceptual level, what was built, what was measured, and what its limitations are.

---

## 1. The Core Problem: The KV Cache Memory Wall

When you chat with a Large Language Model (like GPT-4, Llama 3, or Qwen 3), the model operates in two distinct phases:

1. **Prefill Phase**: The model reads your prompt all at once, calculating internal representations in parallel.
2. **Decode Phase**: The model generates the response **one token at a time** autoregressively.

To generate token $N+1$, the model's self-attention mechanism must examine token $N$ against **all previous tokens** ($1$ to $N$). 

```
Standard Attention Mechanism:
Query (Current Token) × Keys (All Past Tokens) ──► Attention Weights
Attention Weights     × Values (All Past Tokens) ──► New Token Representation
```

If the model recomputed the Key ($K$) and Value ($V$) vectors for all past tokens at every step, generation time would grow quadratically ($O(N^2)$), quickly becoming unusable. To prevent this, LLM inference engines maintain a **Key-Value (KV) Cache** in host memory: once a token's $K$ and $V$ vectors are computed, they are cached so future tokens can reuse them.

### Why Does the KV Cache Break Down?
Unlike model weights—which remain fixed in size regardless of conversation length—the **KV Cache grows linearly with every additional token**:

$$\text{KV Cache Footprint} = 2 \times L \times H_{KV} \times D \times N \times B$$

Where:
- $L$ is the number of transformer layers (e.g., 36 layers in Qwen3-4B).
- $H_{KV}$ is the number of Key/Value attention heads (e.g., 8 heads in GQA).
- $D$ is the head dimension (e.g., 128 elements).
- $N$ is the context length in tokens (e.g., 4,096 to 32,768 tokens).
- $B$ is the precision byte width (e.g., 4 bytes for FP32, 2 bytes for FP16).
- The factor of 2 accounts for storing both Keys and Values.

```
Context Window     KV Cache Size (Qwen3-4B FP32)     Total Host RAM (Dense Baseline)
    4,096 tokens             1.16 GB                           19.94 GB
    8,192 tokens             2.31 GB                           22.98 GB
   16,384 tokens             4.59 GB                           30.12 GB
   32,768 tokens             9.16 GB                           44.07 GB
```

At 32,768 tokens, the KV cache alone consumes over **9.16 GB**, and the total Python process memory swells to **44.07 GB**. On high-throughput inference servers or edge systems, physical DRAM is strictly limited. When DRAM runs out, the operating system crashes or starts swapping to disk unpredictably, bringing inference to a halt.

---

## 2. The Naive Solution and Why It Fails

Why not simply offload the KV cache to a standard Solid-State Drive (SSD)?

In a standard system, the CPU/GPU connects to the SSD over the **PCIe bus**:

```
[Host CPU / GPU DRAM] <======== PCIe Bus (Bandwidth Bottleneck) ========> [Passive NVMe SSD]
   Requires all 9 GB                                                        Stores 9 GB
   transferred per step!                                                    of KV blocks
```

In standard attention, to select which past tokens matter, the host must compute:

$$\text{Score}_i = Q \cdot K_i^T$$

To evaluate this equation, the host must fetch **all candidate Key vectors** from the SSD into host RAM across the PCIe bus on **every single decode step**.

At 32K context with 16 decode steps, streaming all candidate keys across the PCIe bus transfers gigabytes of data every second. The PCIe bus saturates, latency skyrockets by $50\times$, and host memory is still polluted by the incoming streams of data. **Passive SSD offloading simply replaces the DRAM capacity wall with a PCIe bandwidth wall.**

---

## 3. The AI-SSD Solution: Computational Storage

**AI-SSD** transforms the SSD from a passive storage bucket into an **active computational storage device**.

Instead of moving gigabytes of candidate Keys across the PCIe bus into host memory to find out what to attend to, **we move the computation to the storage controller itself**:

```
+-----------------------------------------------------------------------------------------+
|                                    HOST SYSTEM (CPU)                                    |
|                                                                                         |
|  1. Generate Query (Q) for current token                                                |
|  2. Retain only critical tokens in DRAM (Attention Sinks + Recent 16 Tokens)            |
|  3. Send small Query vector Q (~288 KiB) to SSD over PCIe                               |
+--------------------------------------------┬--------------------------------------------+
                                             │ PCIe Bus (Traffic reduced by 81.5%!)
                                             │ Send: Q Vector (288 KiB)
                                             │ Recv: Winning KV Blocks ONLY (54.8 MB)
+--------------------------------------------▼--------------------------------------------+
|                             COMPUTATIONAL STORAGE (AI-SSD)                              |
|                                                                                         |
|  1. In-Storage Top-K Filter (AVX2/FMA/F16C Vector Engines):                             |
|     - Read candidate Keys directly from internal NAND flash into internal controller    |
|     - Calculate dot products: Score = Q · K^T                                           |
|     - Identify Top-10% most relevant blocks in internal controller SRAM/DRAM            |
|  2. Transmit ONLY the winning 10% Key and Value blocks back to host                     |
|  3. Strict Invariant: Candidate Key bytes transferred to host = 0 BYTES                 |
+-----------------------------------------------------------------------------------------+
```

### The Three Fundamental Insights:
1. **Attention Sparsity (Top-$K$)**: In long-context LLMs, over 90% of past tokens have negligible attention weights on any given step. By keeping the initial "attention sinks" (the first 4 tokens) and the recent local window (last 16 tokens) resident in DRAM, we only need the top 10% most relevant historical tokens to produce exact outputs.
2. **Bandwidth Elimination**: Candidate Keys are scanned inside the drive. Only the winning 10% KV blocks cross the PCIe bus. Candidate Key transfers to the host drop to **strictly 0 bytes**.
3. **True Host-RAM Offload**: Unselected KV blocks never touch host DRAM. Process memory remains flat regardless of context length.

---

## 4. The Engineering Division: P1, P2, and P3

To build and validate this system realistically, the architecture is divided into three coordinated subsystems:

```
+-----------------------------------------------------------------------------------------+
| P1: KV Engine & Inference Runtime                                                       |
| - Executes real Hugging Face model weights (Qwen3-4B FP32 & Qwen3-8B FP16).             |
| - Maintains Attention Sinks & Recent Sliding Window in DRAM.                            |
| - Packages historical tokens into 16-token KV Blocks (4 KiB K + 4 KiB V = 8 KiB block).  |
| - Executes fused Grouped-Query Attention (SDPA) on retrieved active tokens.             |
+--------------------------------------------┬--------------------------------------------+
                                             │
+--------------------------------------------▼--------------------------------------------+
| P3: System Integration & Async Prefetch Adapter                                         |
| - Provides seamless storage abstraction between P1 and P2.                              |
| - Contiguous 8 KiB block packing (merges K and V reads to cut I/O transactions by 50%). |
| - Asynchronous pipelined prefetching: overlaps storage retrieval with host CPU MLP.     |
| - High-resolution process RSS telemetry sampler (2.0 ms continuous sampling).           |
+--------------------------------------------┬--------------------------------------------+
                                             │
+--------------------------------------------▼--------------------------------------------+
| P2: Storage Backend, Tensor-Aware FTL & NVMe Controller                                 |
| - Direct block device interface to Linux kernel NVMe driver (/dev/nvme0n1).             |
| - 8-Channel Tensor-Aware Flash Translation Layer (FTL) for multi-channel striping.      |
| - Guest daemon executing in-storage AVX2/FMA/F16C Top-K vector scoring.                 |
+-----------------------------------------------------------------------------------------+
```

---

## 5. Key Concepts Explained Simply

### What is KV Blockization?
Instead of managing individual tokens (which is fragmented and inefficient for storage drives), tokens are grouped into fixed **KV Blocks** of 16 tokens:
- In Qwen3-4B (FP32, 8 KV heads, head dimension 128):
  - 1 Key page = $16 \times 8 \times 128 \times 4\text{ B} = 65,536\text{ B}$ ($64\text{ KiB}$)
  - 1 Value page = $16 \times 8 \times 128 \times 4\text{ B} = 65,536\text{ B}$ ($64\text{ KiB}$)
  - Combined block = $128\text{ KiB}$
- In Qwen2.5-0.5B / Canonical Geometry (FP32, 2 KV heads, head dimension 64):
  - 1 Key page = $16 \times 2 \times 64 \times 4\text{ B} = 4,096\text{ B}$ ($4\text{ KiB}$)
  - 1 Value page = $16 \times 2 \times 64 \times 4\text{ B} = 4,096\text{ B}$ ($4\text{ KiB}$)
  - Combined block = $8,192\text{ B}$ ($8\text{ KiB}$)

$4\text{ KiB}$ and $8\text{ KiB}$ align cleanly with physical SSD flash page geometry, eliminating unaligned page overhead and write amplification.

### What is In-Storage Top-$K$?
On each decode step, the host sends the current token's Query vector $Q$ to the storage controller. The controller's internal processor reads the stored candidate Key pages from flash memory, calculates $Q \cdot K^T$ across all candidate blocks using vectorized SIMD instructions, maintains a running top-$K$ min-heap, and selects the top 10% scoring blocks. Only those winning blocks are read and transferred to the host.

### What is the Flash Translation Layer (FTL) and Why Does Tensor-Aware Placement Matter?
Solid-state drives do not read from one single chip; they have **multiple parallel flash channels** (typically 8 to 16 channels) reading flash dies concurrently.

- **Conventional Sequential Mapping**: When a conventional SSD writes blocks sequentially, an entire layer's Key data ends up on a single channel (Channel 0). When the model reads that layer, Channel 0 is overwhelmed while Channels 1 through 7 sit completely idle (**700% load imbalance**).
- **Tensor-Aware FTL**: Our FTL understands tensor coordinates `(layer, head, block)`. It stripes head dimensions and consecutive blocks evenly across all 8 flash channels. During retrieval, all 8 channels work simultaneously in parallel (**0.86% load imbalance**), eliminating flash channel bottlenecks.

```
Conventional Mapping:
Channel 0: [Block 0][Block 1][Block 2][Block 3] (100% Saturation, Bottleneck!)
Channel 1: [Idle]
Channel 2: [Idle]
...
Channel 7: [Idle]

Tensor-Aware FTL Mapping:
Channel 0: [Block 0]  Channel 1: [Block 1]  Channel 2: [Block 2]  Channel 3: [Block 3]
Channel 4: [Block 4]  Channel 5: [Block 5]  Channel 6: [Block 6]  Channel 7: [Block 7]
(Balanced load across all channels: 0.86% imbalance!)
```

### What is QEMU/NVMe in This Project?
Because physical commercial computational SSDs with open programmable firmware are not commercially accessible commodities, we evaluated the storage subsystem using a **hardware-accelerated QEMU/KVM virtual machine**:
- The host runs Linux with KVM hardware virtualization.
- QEMU emulates an NVMe PCI controller attached to a 16 GiB raw virtual disk (`v2_nvme.raw`).
- Inside the virtual machine, a minimal Linux kernel exposes the disk as `/dev/nvme0n1` through the actual Linux kernel NVMe driver.
- A compiled C daemon (`nvme_guest_daemon.c`) runs inside the guest environment, receiving commands over a high-speed socket, executing block I/O against `/dev/nvme0n1`, and running AVX2/FMA SIMD vector scoring inside the virtual controller.

This gives us an authentic virtual device pipeline executing real Linux kernel NVMe I/O operations.

---

## 6. What Was Measured & Key Results

The complete system was validated on a dedicated AWS compute instance (Intel Xeon Sapphire Rapids, 8 vCPUs, 61.8 GB RAM) across real models (`Qwen/Qwen3-4B-Instruct-2507` and `Qwen/Qwen3-8B`):

### 1. Massive Host Memory Reduction
Across context scaling from 4,096 to 32,768 tokens, AI-SSD keeps process memory strictly bounded:

```
Context Window     Dense Baseline Peak RSS     AI-SSD Peak RSS     Physical RAM Saved
 4,096 tokens             19.94 GB                17.55 GB          2.39 GB  (12.0%)
 8,192 tokens             22.98 GB                15.92 GB          7.05 GB  (30.7%)
16,384 tokens             30.12 GB                16.03 GB         14.08 GB  (46.8%)
32,768 tokens             44.07 GB                16.17 GB         27.90 GB  (63.3%)
```

- **Active KV DRAM reduction**: **89.4% to 89.9%** across all contexts.
- **Physical RAM saved at 32K context**: **27.90 GB** (process RSS stayed at 16.17 GB instead of exploding to 44.07 GB).

### 2. Elimination of PCIe Bus Bottlenecks
During 16 decode steps at 4,096 context:
- Host-side candidate streaming moved **564.0 MB** of candidate Keys across the bus.
- In-storage computational filtering transferred **0 BYTES of candidate Keys** to the host.
- Total PCIe bus data movement dropped from **592.7 MB down to 109.8 MB** (**81.5% reduction**).

### 3. Execution Speed
- By moving Top-$K$ into storage and optimizing vector kernels with AVX2/FMA SIMD, decode time dropped from **51.39 s down to 16.66 s** (throughput increased from 0.311 tok/s to **0.961 tok/s**, a **$3.09\times$ speedup** over host-side candidate streaming).
- In Qwen3-8B FP16, throughput reached **1.152 tok/s** on 4 CPU threads.

### 4. Output Correctness
- **100% Exact Token Match (16/16 tokens)** verified against the dense PyTorch baseline across all configurations and all context lengths ($4\text{K}, 8\text{K}, 16\text{K}, 32\text{K}$).

---

## 7. Honest Limitations

To maintain scientific integrity, we explicitly acknowledge what this project is and is not:

1. **Virtual Device Environment**: Measurements are conducted over a hardware-accelerated QEMU virtual NVMe device and guest socket daemon. Virtualization introduces hypervisor trap overhead ($2.3\times$ latency compared to bare direct-file I/O).
2. **No Physical ASIC Hardware**: We did not fabricate custom computational SSD silicon or measure physical ASIC chip power on an oscilloscope.
3. **No Power or NAND Wear Data**: We do not claim measured wattage savings or measured flash cell wear cycles.
4. **CPU Inference Bottleneck**: Model weights were executed on host CPU (4 threads). On host CPU, computation accounts for ~60% of decode wall time. In high-density GPU inference, storage retrieval latency would represent a proportionally larger factor.
5. **32K Scaling Latency**: At 32,768 context, scanning 73,728 blocks inside a software daemon on 2 virtual CPU cores drops throughput to 0.024 tok/s. Dedicated hardware ASIC parallel scanning pipelines would be required in silicon to sustain interactive speeds at 32K.
