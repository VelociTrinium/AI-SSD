# AI-SSD V2: Implementation Deep Dive

**Audience**: ML systems engineers, storage systems architects, and systems researchers.  
**Objective**: Provide an exhaustive, code-grounded engineering breakdown of the actual AI-SSD V2 implementation. Every subsystem is specified with concrete inputs, outputs, data structures, execution flow, performance role, and failure modes. Zero invented details.

---

## 1. Mathematical & Transformer Attention Foundations

### 1.1 QKV Generation & Grouped-Query Attention (GQA)
In standard Multi-Head Attention (MHA), the number of Query heads ($H_Q$) equals Key/Value heads ($H_{KV}$). In modern open-weight LLMs—such as `Qwen/Qwen3-4B-Instruct-2507` and `Qwen/Qwen3-8B`—Grouped-Query Attention (GQA) is employed to minimize memory bandwidth during autoregressive decoding.

#### Model Geometries in AI-SSD V2
| Parameter | `Qwen2.5-0.5B` (Canonical V1) | `Qwen3-4B-Instruct-2507` (Canonical V2) | `Qwen3-8B` (V2 Scaling) |
|---|:---:|:---:|:---:|
| **Layers ($L$)** | 24 | 36 | 36 |
| **Hidden Size ($D_{model}$)** | 896 | 2,560 | 4,096 |
| **Intermediate Size ($D_{FFN}$)** | 4,864 | 9,728 | 12,288 |
| **Query Heads ($H_Q$)** | 14 | 32 | 32 |
| **KV Heads ($H_{KV}$)** | 2 | 8 | 8 |
| **Head Dimension ($D$)** | 64 | 128 | 128 |
| **GQA Ratio ($H_Q / H_{KV}$)** | 7 | 4 | 4 |
| **Native Dtype** | FP32 | FP32 | FP16 |
| **KV Bytes / Token / Layer** | $1,024\text{ B}$ (FP32) | $8,192\text{ B}$ (FP32) | $4,096\text{ B}$ (FP16) |
| **KV Footprint / Token (All Layers)** | $24.58\text{ KiB}$ | $294.91\text{ KiB}$ | $147.46\text{ KiB}$ |

#### Rotary Position Embeddings (RoPE)
Prior to attention scoring, query states $Q$ and key states $K$ are transformed by RoPE using precomputed frequency tensors $\cos, \sin \in \mathbb{R}^{1 \times 1 \times S \times D}$:
$$Q_{\text{rot}} = (Q \odot \cos) + (\text{rotate\_half}(Q) \odot \sin)$$
$$K_{\text{rot}} = (K \odot \cos) + (\text{rotate\_half}(K) \odot \sin)$$
Values $V$ do not receive positional embeddings.

---

## 2. KV Blockization & Physical Geometry Mapping

### 2.1 Block Structure & Page Alignment
Host attention requires contiguous multidimensional arrays, but NAND flash operates on discrete physical sectors and pages (typically $4,096\text{ bytes}$). AI-SSD groups tokens into fixed $T=16$ token blocks:

```text
Logical KV Block (Block ID: b)
┌────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│                   Key Page (Offset 0)                  │                 Value Page (Offset +K_size)            │
│  Shape: [tokens_per_block, num_kv_heads, head_dim]     │  Shape: [tokens_per_block, num_kv_heads, head_dim]     │
│  Tokens: [b * 16 ... (b + 1) * 16 - 1]                 │  Tokens: [b * 16 ... (b + 1) * 16 - 1]                 │
└────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

#### Byte Allocations:
- **Canonical V1 Geometry** ($T=16, H_{KV}=2, D=64, \text{FP32}$):
  $$\text{Key Page} = 16 \times 2 \times 64 \times 4\text{ B} = 8,192\text{ B} \quad (2 \times 4\text{ KiB flash sectors})$$
  $$\text{Value Page} = 16 \times 2 \times 64 \times 4\text{ B} = 8,192\text{ B} \quad (2 \times 4\text{ KiB flash sectors})$$
  $$\text{Total Block} = 16,384\text{ B} \quad (16\text{ KiB})$$
  *(In 1-head partitioned layouts: $16 \times 1 \times 64 \times 4\text{ B} = 4,096\text{ B}$ exactly 1 physical 4 KiB flash page).*
- **Qwen3-4B FP32 Geometry** ($T=16, H_{KV}=8, D=128, \text{FP32}$):
  $$\text{Key Page} = 16 \times 8 \times 128 \times 4\text{ B} = 65,536\text{ B} \quad (64\text{ KiB})$$
  $$\text{Value Page} = 16 \times 8 \times 128 \times 4\text{ B} = 65,536\text{ B} \quad (64\text{ KiB})$$
  $$\text{Total Block} = 131,072\text{ B} \quad (128\text{ KiB})$$
- **Qwen3-8B FP16 Geometry** ($T=16, H_{KV}=8, D=128, \text{FP16}$):
  $$\text{Key Page} = 16 \times 8 \times 128 \times 2\text{ B} = 32,768\text{ B} \quad (32\text{ KiB})$$
  $$\text{Value Page} = 16 \times 8 \times 128 \times 2\text{ B} = 32,768\text{ B} \quad (32\text{ KiB})$$
  $$\text{Total Block} = 65,536\text{ B} \quad (64\text{ KiB})$$

---

## 3. Subsystem Breakdown

### 3.1 Subsystem 1: P1 KV Engine & Inference Runtime (`[REAL]`)
Located in `person1_kv_engine/real_llm/aissd_inference.py`.

- **Input**:
  - `model`: Hugging Face `AutoModelForCausalLM` instance (`Qwen3ForCausalLM`).
  - `input_ids`: Prompt token tensor `torch.LongTensor [1, S_prompt]`.
  - `decode_tokens`: Number of autoregressive decode steps ($M=16$).
  - `top_k_pct`: Active candidate selection ratio (canonical = $10.0\%$).
- **Output**:
  - `generated_tokens`: List of integer token IDs `[16]`.
  - `generated_text`: Decoded UTF-8 string.
  - `timings`: Critical-path component breakdown dictionary.
  - `proc_mem`: Real OS Resident Set Size (RSS) telemetry statistics (`min`, `avg`, `peak`).
- **Data Structures**:
  - `AISSDKVManager`: Master orchestration state object.
  - `layer_data: Dict[int, Dict[str, Any]]`:
    - `sink_k`, `sink_v`: `torch.Tensor [1, H_KV, 4, D]` (first 4 tokens kept resident in DRAM).
    - `recent_k`, `recent_v`: `torch.Tensor [1, H_KV, 16, D]` (last 16 tokens kept resident in DRAM).
    - `candidate_blocks: List[Tuple[int, int]]`: List of `(block_id, actual_tokens)` describing offloaded historical blocks.
- **Execution Flow**:
  1. Prefill step computes initial prompt KV tensors using PyTorch forward pass.
  2. Attention sinks (first 4 tokens) and recent window (last 16 tokens) are cloned into host DRAM.
  3. Historical tokens between token 4 and token $S-16$ are sliced, transposed into `[tokens_per_block, H_KV, D]`, and written to storage via `backend.write_block()`.
  4. Prefill output tensors (`prefill_out`, `past_key_values`) are explicitly deleted with `gc.collect()` to enforce **True Host-RAM Offload**.
  5. The model's `self_attn.forward` methods across all 36 layers are dynamically wrapped to intercept query projections and substitute sparse working-set attention during decode steps.
- **Performance Role**: Manages memory tiering, preventing $O(N)$ growth in host DRAM.
- **Failure Modes**:
  - Out of Memory (OOM) if prefill activation graph is not garbage-collected prior to decode loop.
  - Shape mismatch exception if prompt length is less than sink ($4$) + recent ($16$) tokens ($<20$ tokens).

---

### 3.2 Subsystem 2: P3 Prefetch Adapter & Async Pipeline (`[REAL]`)
Located in `person3_system/prefetch/inference_adapter.py`.

- **Input**:
  - Block read requests from P1 (`read_block_batch(layer_idx, block_ids)`).
  - Inter-layer prefetch hints (`predict_and_prefetch(current_layer_id, current_block_ids)`).
- **Output**:
  - Staged contiguous KV numpy arrays delivered to P1.
  - Storage telemetry counters (`demand_hits`, `demand_misses`, `staging_memory_mb`).
- **Data Structures**:
  - `StagedInferenceBlock`: Dataclass holding `block_id`, `layer_id`, `data_k`, `data_v`, `future`, and timing markers.
  - `_staging_buffer: OrderedDict[Tuple[int, int], StagedInferenceBlock]`: LRU cache with configurable capacity (default 512 blocks).
  - `ThreadPoolExecutor`: Worker pool with `max_async_workers=2` for non-blocking I/O.
- **Execution Flow**:
  1. **Contiguous 8 KiB Fetch**: Intercepts P1 requests and maps separate Key and Value requests into a single contiguous block read request `(offset, k_size + v_size, bid)`, cutting socket round-trips in half.
  2. **Inter-Layer Speculative Prefetching**: When layer $L$ retrieves its winning blocks, P3 predicts layer $L+1$'s winning blocks (using the spatial Markov predictor in `NextLayerPredictor`) and dispatches asynchronous background reads.
  3. **Consumption & Immediate Eviction**: Once a staged block is consumed by P1's attention module, it is unlinked from the staging buffer, maintaining an active staging footprint of **0.0 MB** between decode steps.
- **Performance Role**: Hides raw storage access latency behind concurrent host CPU execution (MLP layers and LayerNorms). In Phase 8 ablation Run F, P3 hid **9.94 seconds (68.3%)** of raw storage latency.
- **Failure Modes**:
  - Worker thread contention if `max_async_workers` exceeds available physical CPU cores.
  - Stale prefetch future deadlock if background read encounters socket error without setting future exception.

---

### 3.3 Subsystem 3: P2 Storage Backend & Tensor-Aware FTL (`[VIRTUAL-DEVICE]`)
Located in `person2_ssd/inference_backend.py` and `person2_ssd/kv_allocator/tensor_mapping.py`.

- **Input**:
  - Block write and read requests with coordinates `(layer_idx, block_id, head_id)`.
  - Top-$K$ filter dispatch request with query vector $Q$ and candidate metadata array.
- **Output**:
  - Raw binary payloads transferred to/from `/dev/nvme0n1`.
  - Top-$K$ index lists `[(score, block_id, actual_tokens)]`.
  - Channel balance telemetry (`_per_channel_reads`, `_per_channel_read_bytes`).
- **Data Structures**:
  - `_storage: Dict[Tuple[int, int], Dict[str, Any]]`: Fast in-memory metadata catalog storing:
    ```python
    {
        "k_offset": uint64, "k_size": uint32, "k_shape": tuple, "k_dtype": dtype,
        "v_offset": uint64, "v_size": uint32, "v_shape": tuple, "v_dtype": dtype,
        "channel": uint8, "lba": uint64, "timestamp": float
    }
    ```
    *Strict Invariant*: Zero payload tensor arrays are retained in `_storage`.
  - `DeterministicTensorMapper`: Implements mathematical flash mapping:
    - 8 flash channels, 4 dies per channel, 2 planes per die, 1024 blocks per plane, 256 pages per block.
- **Execution Flow**:
  - **Writing**: Computes physical flash coordinates via `tensor_to_nand_physical()`, assigns contiguous offset on `/dev/nvme0n1`, issues write via `QemuNvmeClient`, and saves metadata.
  - **Reading**: Looks up offset, dispatches direct NVMe block read via client, unpacks binary buffer into numpy array matching canonical shape and dtype.
  - **Top-$K$ Filter**: Delegates to `QemuNvmeClient.compute_topk()`.
- **Performance Role**: Guarantees zero host RAM leakage for offloaded blocks; ensures uniform striping across 8 flash channels to avoid single-channel serialization.
- **Failure Modes**:
  - File offset collision if multiple processes write without unique base offsets.
  - Metadata desynchronization if write fails in NVMe driver but updates metadata map.

---

### 3.4 Subsystem 4: NVMe Client & Socket Transport (`[VIRTUAL-DEVICE]`)
Located in `person2_ssd/nvme_client.py`.

- **Input**:
  - Python write/read/compute commands.
  - TCP target `127.0.0.1:9999`.
- **Output**:
  - Serialized binary network frames sent to guest VM; deserialized responses returned to Python.
- **Data Structures**:
  - Packed C structs (Little Endian, packed 1 byte):
    - `req_header`: `<IBBHQI` (20 bytes: `magic`, `op`, `flags`, `reserved`, `offset`, `length`).
    - `resp_header`: `<IBBHQI` (20 bytes: `magic`, `status`, `op`, `reserved`, `offset`, `length`).
    - `batch_read_item`: `<QII` (16 bytes: `offset`, `length`, `block_id`).
    - `topk_req_header`: `<IIIIIf` (24 bytes: `num_candidates`, `top_k`, `q_heads`, `kv_heads`, `head_dim`, `scale`).
    - `topk_cand_item`: `<QIII` (20 bytes: `offset`, `length`, `block_id`, `actual_tokens`).
    - `topk_resp_item`: `<IfI` (12 bytes: `block_id`, `score`, `actual_tokens`).
- **Execution Flow**:
  - Reentrant lock (`threading.RLock`) guarantees thread safety across concurrent prefetch and demand calls.
  - `TCP_NODELAY` disabled Nagle's algorithm for sub-millisecond response latency.
  - `_recv_exact` loops over `socket.recv` to guarantee complete buffer ingestion without packet truncation.
- **Performance Role**: Transports requests across host-guest virtual machine boundary.
- **Failure Modes**:
  - `ConnectionRefusedError` if QEMU guest VM has not finished booting.
  - `BrokenPipeError` if guest daemon crashes or exits unexpectedly.

---

### 3.5 Subsystem 5: Guest Daemon & In-Storage AVX2/FMA/F16C Scoring (`[VIRTUAL-DEVICE]`)
Located in `scripts/nvme_guest_daemon.c`.

- **Input**:
  - Socket commands on port 9999.
  - Raw block device `/dev/nvme0n1` opened with `O_RDWR | O_SYNC`.
- **Output**:
  - Direct kernel block I/O responses via `pread`/`pwrite`.
  - Top-$K$ sorted response items.
- **Data Structures**:
  - `static char io_buf[1024 * 1024]`: 1 MB static aligned I/O scratchpad.
  - In-place min-heap / sorted insertion buffer for top-$K$ candidates.
- **Execution Flow**:
  1. Reads request header and validates `MAGIC == 0x4E564D45`.
  2. For `OP_COMPUTE_TOPK`:
     - Reads Query vector $Q$ ($H_Q \times D$ floats) and candidate descriptors into memory.
     - Loops over candidate blocks:
       - Issues `pread(dev_fd, io_buf, item_len, offset)` directly to `/dev/nvme0n1`.
       - Evaluates vectorized dot products between Query and stored Key tokens.
       - Dispatches to `compute_block_score_gqa_fp16` (using `_mm256_cvtph_ps` + `_mm256_fmadd_ps`) for FP16 or `compute_block_score_gqa` for FP32.
       - Maintains a top-$K$ descending insertion list.
     - Writes response header and top-$K$ response items back across socket.
- **Performance Role**: Eliminates candidate Key PCIe bus traffic (reducing bus volume by 81.5%).
- **Failure Modes**:
  - `EINVAL` if offset or length is not 4096-byte sector aligned when using `O_DIRECT`.
  - Buffer overflow if candidate count exceeds socket buffer capacity.

---

## 4. In-Storage SIMD Scoring Kernels (AVX2 / FMA / F16C)

### 4.1 128-Dimensional AVX2/FMA Kernel (FP32)
Used for `Qwen3-4B` ($D=128$, FP32). Evaluates 128-element inner product using four parallel 256-bit accumulator registers (`acc0`, `acc1`, `acc2`, `acc3`), completely unrolling the 128-float loop into eight 256-bit FMA steps:

```c
static inline float dot_product_128_avx2(const float* a, const float* b) {
    __m256 acc0 = _mm256_mul_ps(_mm256_loadu_ps(a),      _mm256_loadu_ps(b));
    __m256 acc1 = _mm256_mul_ps(_mm256_loadu_ps(a + 8),  _mm256_loadu_ps(b + 8));
    __m256 acc2 = _mm256_mul_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
    __m256 acc3 = _mm256_mul_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));

    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56), acc3);

    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 64), _mm256_loadu_ps(b + 64), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 72), _mm256_loadu_ps(b + 72), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 80), _mm256_loadu_ps(b + 80), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 88), _mm256_loadu_ps(b + 88), acc3);

    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 96),  _mm256_loadu_ps(b + 96),  acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 104), _mm256_loadu_ps(b + 104), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 112), _mm256_loadu_ps(b + 112), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 120), _mm256_loadu_ps(b + 120), acc3);

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
```

### 4.2 128-Dimensional F16C / AVX2 / FMA Kernel (FP16)
Used for `Qwen3-8B` ($D=128$, FP16). The query vector $Q$ is FP32, while stored Keys $K$ are FP16 (`uint16_t`). It loads 128-bit blocks of 8 half-precision floats and converts them on the fly to 256-bit single-precision floats using hardware instruction `_mm256_cvtph_ps`:

```c
static inline float dot_product_128_fp16_avx2(const float* a, const uint16_t* b) {
    __m256 b0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)b));
    __m256 b1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 8)));
    __m256 b2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 16)));
    __m256 b3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 24)));

    __m256 acc0 = _mm256_mul_ps(_mm256_loadu_ps(a),      b0);
    __m256 acc1 = _mm256_mul_ps(_mm256_loadu_ps(a + 8),  b1);
    __m256 acc2 = _mm256_mul_ps(_mm256_loadu_ps(a + 16), b2);
    __m256 acc3 = _mm256_mul_ps(_mm256_loadu_ps(a + 24), b3);
    // ... continues with 12 unrolled _mm256_fmadd_ps steps ...
```

---

## 5. Native Grouped-Query Attention (SDPA) Fusion

In early phases, attention on retrieved tokens was computed via manual tensor manipulation:
```python
# Unoptimized (Manual GQA + MatMul + Softmax):
k_rep = k.repeat_interleave(gqa_ratio, dim=1) # High DRAM memory copy overhead
scores = torch.matmul(q, k_rep.transpose(-1, -2)) * scale
attn_weights = torch.softmax(scores, dim=-1)
out = torch.matmul(attn_weights, v_rep)
```

In Phase 8/9 optimization (OPT-001), this was replaced with PyTorch's native C++ fused Grouped-Query SDPA kernel:
```python
# Optimized (Fused GQA SDPA):
out = torch.nn.functional.scaled_dot_product_attention(
    q, act_k, act_v, scale=scaling, enable_gqa=True
)
```
- **Impact**: Reduced attention execution time (`attn_matmul_s`) from **0.4478 s to 0.1838 s (-58.9%)**.
- **Correctness Invariant**: Bit-exact 16/16 token ID match preserved.

---

## 6. Process RAM Telemetry & Invariant Accounting

Continuous memory telemetry is captured strictly via `/proc/self/status` high-frequency sampling (2.0 ms interval) by `ProcessMemorySampler`:
- **`peak_rss_mb`**: Maximum observed Resident Set Size of the host Python process.
- **`active_kv_mb`**: Exact allocated DRAM byte footprint of active attention tensors:
  $$\text{Active Tokens} = \text{Sinks } (4) + \text{Recent } (16) + (K_{\text{winning}} \times 16)$$
  $$\text{Active KV DRAM} = \text{Active Tokens} \times H_{KV} \times D \times \text{sizeof(dtype)} \times 2 \times L$$

### Empirical Invariants Across All Runs:
1. `candidate_k_bytes_to_host == 0` (Zero candidate Keys sent across bus).
2. `p2_resident_payload_mb == 0.0` (Zero payload tensors retained in P2 metadata tables).
3. `p3_resident_payload_mb == 0.0` (Zero payload tensors retained in P3 prefetch cache between steps).
4. `reconciliation_error_pct < 2.0%` (Critical path reconciles with total wall-clock time).

---

## 7. Critical-Path Latency Accounting

Total wall time is strictly reconciled against physical components:

$$\text{Critical Path} = T_{\text{visible storage}} + T_{\text{visible compute}}$$

Where:
- $T_{\text{visible storage}} = T_{\text{candidate\_k\_reads}} + T_{\text{winning\_v\_reads}}$
- $T_{\text{visible compute}} = T_{\text{QKV}} + T_{\text{RoPE}} + T_{\text{topk\_scoring}} + T_{\text{cand\_select}} + T_{\text{prefetch}} + T_{\text{recon}} + T_{\text{concat}} + T_{\text{SDPA}} + T_{\text{out\_proj}} + T_{\text{MLP}} + T_{\text{bookkeeping}}$

$$\text{Reconciliation Error (\%)} = \frac{|T_{\text{wall}} - T_{\text{critical path}}|}{T_{\text{wall}}} \times 100\%$$

In canonical runs:
- $T_{\text{wall}} = 20.92\text{ s}$
- $T_{\text{critical path}} = 20.79\text{ s}$
- $\text{Reconciliation Error} = 0.63\%$ ($< 2.0\%$).
- Host CPU Compute represents **60.4% ($12.64\text{ s}$)** of execution time, while visible storage represents **39.0% ($8.15\text{ s}$)**.
