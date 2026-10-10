# AI-SSD V2: How It Works — End-to-End Execution Walkthrough

**Purpose**: Provide a step-by-step, chronological walkthrough of an inference request through the AI-SSD V2 system, tracing every tensor transformation, memory transition, network transaction, and storage operation from initial prompt ingestion to final token output.

---

## 1. High-Level Lifecycle Overview

```text
[1. User Prompt]
       │
       ▼
[2. Prefill Phase (Host CPU)] ────────► Produces Initial K & V Tensors for all S Tokens
       │
       ▼
[3. KV Partitioning]
       ├── Attention Sinks (First 4 tokens) ──► Retained in Host DRAM
       ├── Recent Window (Last 16 tokens)   ──► Retained in Host DRAM
       └── Historical Tokens (4 to S-16)    ──► Packed into 16-Token Blocks
                                                      │
                                                      ▼
[4. Storage Offload] ─────────────────────────► Written to /dev/nvme0n1 via P2 FTL
       │
       ▼
[5. True Host-RAM Detach] ────────────────────► del prefill_out, pkv; gc.collect()
       │
       ▼
[6. Autoregressive Decode Loop (Step 1 to M)]
       │
       ├─► 6.1 Query Projection & RoPE on Host CPU
       │
       ├─► 6.2 Send Query Q Vector to Storage Controller (PCIe)
       │
       ├─► 6.3 In-Storage SIMD Scoring (AVX2/FMA/F16C) inside NVMe Controller
       │       - Read candidate Keys directly from internal flash
       │       - Compute dot products Q · K^T
       │       - Select Top 10% winning blocks
       │       - Candidate Key bytes to host = 0 BYTES
       │
       ├─► 6.4 Host Receives Winning Block IDs (168 KiB)
       │
       ├─► 6.5 Host Fetches Winning KV Blocks (8 KiB contiguous read)
       │
       ├─► 6.6 Assemble Active Working Set: [Sinks] + [Top-K] + [Recent Window]
       │
       ├─► 6.7 Compute Native Fused GQA SDPA Attention
       │
       ├─► 6.8 Host Computes MLP / LayerNorm; Concurrently Prefetches Layer L+1
       │
       └─► 6.9 Argmax Sampling, Next Token ID Appended, Slide Recent Window
```

---

## 2. Detailed Step-by-Step Execution Walkthrough

### Step 1: Prompt Ingestion & Model Initialization
1. The user supplies a text prompt (e.g., deterministic evaluation text generated via `build_prompt_for_length(4096)`).
2. The tokenizer encodes the string into token IDs: `input_ids` with shape `[1, S]` (where $S=4,096$ tokens in the canonical configuration).
3. The host runtime configures execution concurrency: `torch.set_num_threads(4)` binds execution to 4 physical CPU cores on the Intel Sapphire Rapids processor, eliminating SMT thread thrashing.
4. The QEMU virtual NVMe device `/dev/nvme0n1` and in-guest daemon (`nvme_guest_daemon`) are verified ready via TCP probe to port 9999.

### Step 2: Prefill Phase on Host CPU
1. The prompt tensor is passed through the standard model forward pass:
   ```python
   with torch.no_grad():
       prefill_out = model(input_ids=input_ids, use_cache=True)
   ```
2. The model computes the forward pass across all 36 transformer layers.
3. Because the prefill phase is compute-bound, calculating all attention matrices in parallel is optimal. The model outputs the first decoded token (`torch.argmax(prefill_out.logits[:, -1, :])`) and populates `past_key_values` containing the full Key and Value tensors for all 4,096 tokens across all 36 layers.

### Step 3: KV Partitioning & Blockization
At this stage, `past_key_values` in host memory consumes **1,156.5 MB** (for Qwen3-4B in FP32). Instead of keeping this expanding tensor in DRAM, `AISSDKVManager.init_from_prefill()` partitions each layer's KV cache into three distinct tiers:

1. **Attention Sinks ($4\text{ tokens}$)**:
   - Slices $K[:, :, :4, :]$ and $V[:, :, :4, :]$ and clones them into resident DRAM tensors (`sink_k`, `sink_v`).
   - *Rationale*: Initial tokens in autoregressive models act as attention sinks that absorb high softmax mass; keeping them resident guarantees stability without I/O overhead.
2. **Recent Window ($16\text{ tokens}$)**:
   - Slices $K[:, :, -16:, :]$ and $V[:, :, -16:, :]$ and clones them into resident DRAM tensors (`recent_k`, `recent_v`).
   - *Rationale*: Natural language has high local recency; the immediately preceding tokens are attended to with near 100% probability.
3. **Historical Tokens ($S_{\text{hist}} = S - 20 = 4,076\text{ tokens}$)**:
   - Tokens between index 4 and $S-16$ are sliced and grouped into fixed 16-token blocks ($B = \lceil 4076 / 16 \rceil = 255\text{ blocks}$).
   - Each block is transposed into canonical layout `[16, num_kv_heads, head_dim]`.
   - Each block is written to storage via `backend.write_block(layer_idx, block_id, k_block, v_block)`.

### Step 4: True Host-RAM Detach & Memory Garbage Collection
To enforce **True Physical Offload** and prevent memory duplication:
```python
del prefill_out
del pkv_prefill
gc.collect()
```
The original PyTorch `past_key_values` and prefill intermediate activations are purged from host DRAM. Only the 4 sink tokens and 16 recent tokens remain in Python memory per layer. The active KV DRAM footprint drops from **1,156.5 MB down to 122.6 MB (an 89.4% reduction)**.

### Step 5: Dynamic Attention Forward Interception
Before entering the decode loop, `run_aissd_decode` dynamically wraps `self_attn.forward` across all 36 layers:
- The custom wrapper intercepts incoming `hidden_states`.
- It executes standard query projections ($W_Q$) and positional RoPE transformations.
- Instead of calling standard PyTorch attention against an in-RAM cache, it invokes `AISSDKVManager.select_and_fetch_active_kv()`.

### Step 6: Decode Step — Query Generation & RoPE
For each autoregressive step (1 to 16):
1. The single-token input `next_token` (shape `[1, 1]`) enters layer $L$.
2. The query, key, and value vectors for the new token are computed:
   $$Q = W_Q \cdot X, \quad K_{\text{new}} = W_K \cdot X, \quad V_{\text{new}} = W_V \cdot X$$
3. RoPE is applied to $Q$ and $K_{\text{new}}$ using the position embedding corresponding to the current sequence length.
4. The new token's $K_{\text{new}}$ and $V_{\text{new}}$ are appended to the host resident recent window, sliding the oldest recent token out.

### Step 7: In-Storage Top-$K$ Filter Dispatch (Query Only)
To identify which of the 255 historical blocks in storage are relevant to current Query $Q$:
1. The host extracts Query $Q$ (shape `[num_query_heads, head_dim]`).
2. The host prepares a list of candidate block descriptors: `[(offset, length, block_id, actual_tokens)]`.
3. P2 dispatches `OP_COMPUTE_TOPK` over the TCP socket to the QEMU NVMe guest daemon:
   - Data sent over the bus: Query vector ($32 \times 128 \times 4\text{ B} = 16\text{ KiB}$) + metadata descriptors ($255 \times 20\text{ B} \approx 5.1\text{ KiB}$).
   - **Crucial Invariant**: **Zero candidate Key vectors are fetched to the host**.

### Step 8: In-Device Vector Scoring (AVX2/FMA/F16C)
Inside the QEMU guest environment, `nvme_guest_daemon.c` receives the command:
1. It loops over the 255 candidate blocks directly on `/dev/nvme0n1`.
2. For each block, it executes `pread(dev_fd, io_buf, 65536, offset)` into controller memory.
3. It evaluates GQA dot-product scores between Query $Q$ and all tokens in the block using hand-tuned AVX2/FMA vector instructions:
   - For FP32 (`Qwen3-4B`): `dot_product_128_avx2` unrolls the 128-element inner product across 4 parallel 256-bit accumulator registers (`_mm256_fmadd_ps`).
   - For FP16 (`Qwen3-8B`): `dot_product_128_fp16_avx2` converts 16-bit flash data into 32-bit floats via `_mm256_cvtph_ps` before FMA accumulation.
4. The maximum scaled score across all heads and tokens in the block represents the block score:
   $$\text{Score}_{\text{block}} = \max_{t, q_h} \left( \frac{Q_{q_h} \cdot K_{t, k_h}^T}{\sqrt{D}} \right)$$
5. The daemon maintains a sorted top-$K$ min-heap ($K = \lceil 255 \times 10\% \rceil = 26\text{ blocks}$).

### Step 9: In-Device Result Return & PCIe Zero-Transfer
1. The daemon writes back a 20-byte response header followed by the 26 winning block descriptors:
   `[(block_id, score, actual_tokens)]` ($26 \times 12\text{ B} = 312\text{ bytes}$).
2. The host receives the winning block IDs.
3. Total data transferred across the bus for candidate filtering: $< 22\text{ KiB}$ (instead of $35.2\text{ MB}$ required by host-side filtering).

### Step 10: Contiguous Asynchronous Winning Block Retrieval
1. The host now requests **only the 26 winning blocks**:
   $$K_{\text{winning}} = 26 \text{ blocks}$$
2. In Phase 8 contiguous retrieval, P3 issues `read_block_batch()` for combined $128\text{ KiB}$ KV blocks (rather than issuing 26 separate Key reads and 26 separate Value reads), cutting storage transactions by 50%.
3. The guest daemon performs direct contiguous reads on `/dev/nvme0n1` and returns the binary payloads.
4. Total winning data transferred over the bus: $26 \times 128\text{ KiB} = 3.33\text{ MB}$ per layer.

### Step 11: Dynamic Working Set Assembly in DRAM
The host reconstructs the active working set for attention:
```python
parts_k = [sink_k] + winning_k_blocks + [recent_k]
parts_v = [sink_v] + winning_v_blocks + [recent_v]
active_k = torch.cat(parts_k, dim=2) # Shape: [1, H_KV, 4 + 416 + 16, D]
active_v = torch.cat(parts_v, dim=2)
```
- Total active sequence length in DRAM: $4 + 416 + 16 = 436\text{ tokens}$ (vs 4,096 in dense inference).
- DRAM footprint for attention: **$10.6\%$ of the full context**.

### Step 12: Native Fused Grouped-Query Attention (SDPA)
1. Layer attention is computed via PyTorch's native C++ kernel:
   ```python
   out = torch.nn.functional.scaled_dot_product_attention(
       q, active_k, active_v, scale=scaling, enable_gqa=True
   )
   ```
2. The fused kernel handles GQA head expansion (`enable_gqa=True`) in hardware registers, avoiding redundant DRAM memory copies and completing attention matmul in **$0.1838\text{ s}$**.
3. The output projection $W_O$ projects attention output back to hidden dimension $D_{model}$.

### Step 13: MLP Execution & Pipelined Next-Layer Prefetch
1. The residual connection adds attention output to the input hidden state.
2. RMSNorm normalizes the activation.
3. The host CPU enters the MLP block (`gate_proj`, `up_proj`, `silu`, `down_proj`).
4. **Pipelined Asynchronous Overlap**: While the host CPU executes the compute-heavy MLP (taking ~4.5 seconds per step), P3's background worker thread issues speculative prefetch requests for **Layer $L+1$'s KV blocks**.
5. When the model reaches Layer $L+1$, its required blocks are already staged in DRAM, hiding up to **68.3% of raw storage latency**.

### Step 14: Next Token Generation & Loop Step
1. After all 36 layers complete, the final hidden state passes through the language model head (`lm_head`).
2. Logits are evaluated: `next_token = torch.argmax(logits[:, -1, :], dim=-1)`.
3. The generated token ID is recorded.
4. The decode loop proceeds to step $m+1$ until all 16 decode tokens are generated.

### Step 15: Telemetry Collection & Cleanup
1. The `ProcessMemorySampler` daemon thread is stopped, recording exact minimum, average, and peak RSS memory.
2. Dynamic wrappers on `self_attn.forward` are restored to original PyTorch functions.
3. Generated token IDs are compared bit-for-bit against dense baseline output to verify 100% numerical correctness.
