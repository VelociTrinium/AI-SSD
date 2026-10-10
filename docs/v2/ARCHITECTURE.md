# AI-SSD V2: System Architecture

**Document Version**: 2.0 (Phase 9 Final Architecture)  
**Scope**: High-level architecture, detailed data path, control path, storage path, and model forward execution path.

---

## 1. High-Level System Architecture

AI-SSD V2 divides responsibilities across three co-designed layers: the Host Inference Engine (P1), the System Prefetch Adapter (P3), and the Computational Storage Subsystem (P2).

```mermaid
flowchart TB
    subgraph HOST["Host System (Sapphire Rapids CPU)"]
        subgraph P1["Person 1: KV Engine & Architecture Adapters"]
            LLM["Hugging Face Models\n(Qwen, Mistral, Jamba)"]
            ADAPTER_ARCH["ModelAdapter Layer\n(QwenAdapter / MistralAdapter / HybridJambaAdapter)"]
            STATE_PROV["StateProvider Abstraction\n(TransformerKV / SlidingWindowKV / HybridState)"]
            KVMGR["AISSDKVManager\n(Sink + Recent + Sparse Assembly)"]
            SDPA["Native Fused GQA SDPA\n(torch.nn.functional)"]
        end

        subgraph P3["Person 3: Prefetch Adapter & Async Pipeline"]
            ADAPTER["RealInferencePrefetchAdapter\n(Contiguous 8 KiB Merging)"]
            STAGING["DRAM Staging Buffer\n(Ephemeral LRU)"]
            PREDICTOR["NextLayerPredictor\n(Inter-Layer Speculation)"]
        end

        subgraph TELEMETRY["Continuous OS Telemetry"]
            RSS["ProcessMemorySampler\n(/proc/self/status 2.0ms)"]
        end
    end

    subgraph BUS["Host-Storage Interface (PCIe Bus / Socket IPC)"]
        Q_TRAFFIC["Query Vector Q + Descriptors (341 KiB)"]
        WIN_TRAFFIC["Winning KV Blocks (109.8 MB)"]
        ZERO_K["Strict Invariant: Candidate Keys = 0 Bytes"]
    end

    subgraph CONTROLLER["Computational Storage Subsystem (QEMU/NVMe)"]
        subgraph P2["Person 2: Backend & FTL"]
            BACKEND["RealInferenceStorageBackend\n(Metadata Catalog: 0.0 MB Payload)"]
            FTL["DeterministicTensorMapper\n(8-Channel Flash Striping)"]
        end

        subgraph GUEST["NVMe Guest Environment (/dev/nvme0n1)"]
            DAEMON["nvme_guest_daemon.c\n(In-Storage AVX2 / FMA / F16C Engine)"]
            TOPK["In-Storage Min-Heap\n(Top 10% Block Selection)"]
            KERNEL_NVME["Linux In-Kernel NVMe Driver\n(/dev/nvme0n1 direct block I/O)"]
        end

        subgraph FLASH["Flash Emulation (NAND Plane)"]
            CH0["Channel 0"]
            CH1["Channel 1"]
            CH2["Channel 2"]
            CH3["Channel 3"]
            CH4["Channel 4"]
            CH5["Channel 5"]
            CH6["Channel 6"]
            CH7["Channel 7"]
        end
    end

    LLM --> KVMGR
    KVMGR --> ADAPTER
    ADAPTER --> BACKEND
    BACKEND --> Q_TRAFFIC
    Q_TRAFFIC --> DAEMON
    DAEMON --> KERNEL_NVME
    KERNEL_NVME --> FTL
    FTL --> CH0 & CH1 & CH2 & CH3 & CH4 & CH5 & CH6 & CH7
    DAEMON --> TOPK
    TOPK --> WIN_TRAFFIC
    WIN_TRAFFIC --> ADAPTER
    ADAPTER --> SDPA
    SDPA --> LLM
```

---

## 2. Detailed Data Path

The data path traces the flow of tensors, vectors, and raw flash blocks through the system during a single autoregressive decode step.

```mermaid
sequenceDiagram
    autonumber
    participant M as Qwen Attention (Layer L)
    participant KVM as AISSDKVManager (P1)
    participant P3 as PrefetchAdapter (P3)
    participant P2 as StorageBackend (P2)
    participant NVMe as QEMU/NVMe Daemon (P2/Guest)
    participant NAND as Flash Channels (0-7)

    Note over M,KVM: Step 1: Query Generation & RoPE
    M->>KVM: Query Vector Q [1, H_Q, 1, D]

    Note over KVM,NVMe: Step 2: Computational Top-K Dispatch
    KVM->>P2: compute_topk_filter(Q, candidate_descriptors)
    Note over P2,NVMe: Zero Candidate Keys Traverse PCIe!
    P2->>NVMe: OP_COMPUTE_TOPK (Q Vector + Candidate Offsets)
    
    Note over NVMe,NAND: Step 3: In-Storage Flash Scanning
    loop For each candidate block
        NVMe->>NAND: pread(candidate_k_offset, length)
        NAND-->>NVMe: 4 KiB / 64 KiB Key Page
        NVMe->>NVMe: SIMD Dot Product: Q · K_block^T (AVX2/FMA/F16C)
        NVMe->>NVMe: Update Top-K Min-Heap
    end

    Note over NVMe,P2: Step 4: Top-K Result Return
    NVMe-->>P2: Top-K Descriptors [(block_id, score)] (168 KiB)
    P2-->>KVM: Winning Block IDs (Top 10%)

    Note over KVM,NVMe: Step 5: Contiguous Winning Block Retrieval
    KVM->>P3: read_block_batch(winning_bids)
    P3->>P2: Batch Contiguous Read [(offset, K_len + V_len)]
    P2->>NVMe: OP_BATCH_READ (Winning Blocks Only)
    NVMe->>NAND: pread(contiguous_kv_blocks)
    NAND-->>NVMe: Winning KV Payloads
    NVMe-->>P3: Winning KV Byte Buffers (54.8 MB K + 54.8 MB V)
    
    Note over P3,M: Step 6: Sparse Active Working Set Assembly
    P3-->>KVM: Unpacked Winning Key & Value Tensors
    KVM->>KVM: Concat: Sinks (4) + Winning (K_win) + Recent (16)
    KVM-->>M: active_k [1, H_KV, S_act, D], active_v [1, H_KV, S_act, D]
    
    Note over M: Step 7: Native Fused GQA SDPA Execution
    M->>M: torch.nn.functional.scaled_dot_product_attention(Q, active_k, active_v)
```

---

## 3. Control Path

The control path coordinates the lifecycle, synchronization primitives, thread concurrency, and telemetry tracking.

```mermaid
flowchart TD
    subgraph INVOCATION["Inference Execution Control"]
        INIT["1. Model Load & Weight Setup\n(low_cpu_mem_usage=True, torch.set_num_threads=4)"]
        PREFILL["2. Prefill Execution\n(Full prompt forwarded through model)"]
        DETACH["3. True Host-RAM Detach\n(Release prefill activations, del past_key_values, gc.collect())"]
        DECODE_LOOP["4. 16-Step Autoregressive Decode Loop"]
    end

    subgraph SYNC["Thread & I/O Synchronization"]
        SAMPLER_START["Start ProcessMemorySampler\n(Daemon thread, 2.0 ms sleep interval)"]
        RLOCK["QemuNvmeClient Reentrant Lock\n(threading.RLock for thread-safe socket I/O)"]
        ASYNC_POOL["ThreadPoolExecutor (2 Workers)\n(Handles background prefetch tasks)"]
        SAMPLER_STOP["Stop ProcessMemorySampler\n(Aggregate min, mean, peak, std RSS)"]
    end

    subgraph INTERCEPT["PyTorch Module Interception"]
        WRAP["Wrap self_attn.forward across all 36 layers"]
        HOOK["Intercept hidden_states, extract Q, K, V"]
        RESTORE["Restore original self_attn.forward in finally block"]
    end

    INIT --> PREFILL
    PREFILL --> DETACH
    DETACH --> WRAP
    WRAP --> SAMPLER_START
    SAMPLER_START --> DECODE_LOOP
    DECODE_LOOP --> RLOCK
    DECODE_LOOP --> ASYNC_POOL
    DECODE_LOOP --> SAMPLER_STOP
    SAMPLER_STOP --> RESTORE
```

---

## 4. Storage Path & Flash Translation Layer (FTL)

The storage path maps logical tensor addresses `(layer, head, token_offset)` into physical block addresses (LBAs) and distributes them across parallel NAND channels.

```mermaid
flowchart LR
    subgraph TENSOR_COORD["Logical Tensor Address"]
        L["Layer Index (0..35)"]
        H["Head Index (0..7)"]
        B["Block Index (0..N)"]
        T["Token Offset (0..15)"]
    end

    subgraph FTL_CORE["DeterministicTensorMapper (P2)"]
        T_HASH["Tensor Coordinate Striping Hash:\nchannel = (layer * H_KV + head + block) % 8"]
        DIE_MAP["Die Mapping: (block / 8) % 4"]
        PLANE_MAP["Plane Mapping: (block / 32) % 2"]
        BLOCK_MAP["NAND Physical Block Allocation"]
        PAGE_MAP["Physical Flash Page (4 KiB)"]
    end

    subgraph PHYSICAL_NAND["8 Parallel Flash Channels (/dev/nvme0n1)"]
        CH0["Channel 0 (12.6% Load)"]
        CH1["Channel 1 (12.5% Load)"]
        CH2["Channel 2 (12.5% Load)"]
        CH3["Channel 3 (12.4% Load)"]
        CH4["Channel 4 (12.4% Load)"]
        CH5["Channel 5 (12.5% Load)"]
        CH6["Channel 6 (12.5% Load)"]
        CH7["Channel 7 (12.6% Load)"]
    end

    L & H & B & T --> T_HASH
    T_HASH --> DIE_MAP --> PLANE_MAP --> BLOCK_MAP --> PAGE_MAP
    PAGE_MAP --> CH0 & CH1 & CH2 & CH3 & CH4 & CH5 & CH6 & CH7
```

### Contrast: Conventional Sequential Mapping vs Tensor-Aware FTL Mapping
- **Conventional Mapping**: Sequential blocks mapped monotonically to LBA offsets without tensor awareness. Maps an entire layer's Key blocks to a single flash channel, resulting in **700.0% load imbalance** and severe head-of-line blocking.
- **Tensor-Aware Mapping**: Strips dimensions across all 8 channels, achieving **0.86% load imbalance** and reducing the flash contention ratio from **83.26 down to 10.50**.

---

## 5. Model Execution Path

The model execution path details how a transformer decoder layer executes with sparse working-set KV retrieval.

```mermaid
flowchart TD
    subgraph TRANSFORMER_LAYER["Transformer Decoder Layer L (1 of 36)"]
        INPUT["Input Hidden States x [1, 1, D_model]"]
        LN1["Input LayerNorm (RMSNorm)"]
        QKV["QKV Projections: q_proj, k_proj, v_proj"]
        ROPE["Rotary Position Embeddings (RoPE)"]
        
        subgraph AISSD_ATTENTION["AI-SSD Intercepted Self-Attention"]
            SLIDE["Slide Recent Window: append new (k, v) token"]
            FILTER["In-Storage Top-K Filter Dispatch (Q vector only)"]
            FETCH["Retrieve Winning KV Blocks (8 KiB contiguous)"]
            CONCAT["Assemble Working Set:\n[Sinks (4)] + [Winning Top-K] + [Recent (16)]"]
            SDPA["Fused Scaled Dot-Product Attention\n(PyTorch native with enable_gqa=True)"]
            OUT_PROJ["Output Projection (o_proj)"]
        end

        RES1["Residual Add 1: x + attn_out"]
        LN2["Post-Attention LayerNorm (RMSNorm)"]
        MLP["MLP Block: gate_proj, up_proj, silu, down_proj"]
        RES2["Residual Add 2: x + mlp_out"]
        OUTPUT["Output Hidden States to Layer L+1"]
    end

    INPUT --> LN1 --> QKV --> ROPE --> SLIDE --> FILTER --> FETCH --> CONCAT --> SDPA --> OUT_PROJ --> RES1 --> LN2 --> MLP --> RES2 --> OUTPUT
```

### Overlap Opportunity During Model Forward:
While the host CPU executes the compute-heavy **Post-Attention LayerNorm** and **MLP Block** (which together consume $21.3\%$ to $22.9\%$ of total wall time), the background asynchronous thread in P3 issues the I/O read for **Layer $L+1$'s KV blocks**, effectively hiding storage latency off the critical path.
