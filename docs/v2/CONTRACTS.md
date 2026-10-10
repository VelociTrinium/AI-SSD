# AI-SSD V2 Contracts

This document specifies the frozen canonical shared interfaces between P1 (Real LLM / KV), P2 (FTL / FEMU / NVMe), and P3 (System Integration / Storage API / Prefetch / Experiments).

---

## 1. Canonical Shared Trace Contract (`common/schemas/trace.py`)

The trace contract unifies P1 real LLM traces (e.g., Qwen2.5-0.5B), P2 FTL replayer, and P3 storage integration into a single loss-free schema.

### 1.1 Record Schema (`CanonicalTraceRecord`)

| Field | Type | Description / Canonical Aliases |
|---|---|---|
| `event_id` | `int` | Monotonically increasing event ID (aliased by `seq_id`). |
| `step` | `int` | Inference token generation step (aliased by `step_id`). |
| `layer_id` | `int` | Transformer model layer index (`0 .. num_layers - 1`). |
| `head_id` | `int` | KV attention head index (aliased by `kv_head_id`, `0 .. num_kv_heads - 1`). |
| `operation` | `str` | Explicit operation type (see Section 1.2). |
| `block_id` | `int` | Primary logical KV block ID. |
| `token_start` | `int` | Starting token index within the context sequence. |
| `token_end` | `Optional[int]` | Ending token index (if bounded). |
| `token_count` | `int` | Number of tokens represented by this block (default: 16). |
| `sub_page` | `str` | Tensor partition: `"KEY"`, `"VALUE"`, or `"BOTH"`. |
| `byte_size` | `int` | Request transfer byte size (aliased by `byte_length` and `bytes`). |
| `tier` | `str` | Storage tier: `"DRAM"` or `"SSD"`. |
| `hotness` | `float` | Attention/access weight score. |
| `candidate_blocks` | `List[int]` | Candidate block IDs for top-k filtering. |
| `selected_blocks` | `List[int]` | Winning block IDs for top-k value retrieval. |
| `timestamp_ns` | `int` | High-precision event timestamp in nanoseconds (aliased by `timestamp_us`). |
| `metadata` | `Dict[str, Any]`| Preserved context metadata. |

### 1.2 Operation Semantics (`TraceOperation`)

- **`PREFILL_WRITE`**: Host offloads initial KV cache block (Key + Value, 8,192 B) to SSD during prefill phase.
- **`DECODE_READ`**: Host unpruned/dense read of both Key and Value pages (8,192 B) during decode phase.
- **`TOPK_FILTER`**: In-storage computational scan across candidate Key pages (e.g., 82 blocks × 4,096 B = 335,872 B).
- **`TOPK_FETCH`**: Host sparse fetch of winning Value pages (4,096 B each) selected by top-k attention scores.
- **`KV_PREFETCH`**: Speculative staging from SSD to host DRAM buffer.
- **`KV_EVICT`**: Host or SSD controller cache eviction.

### 1.3 Trace Manifest Contract (`TraceManifest`)

Real LLM traces must be accompanied by a `<trace_stem>.manifest.json` metadata manifest containing:
- `model_name`, `model_architecture`, `num_layers`, `num_query_heads`, `num_kv_heads`, `head_dim`, `gqa_ratio`, `dtype`.
- Physical dimensions: `key_page_bytes: 4096`, `value_page_bytes: 4096`, `logical_block_bytes: 8192`.
- Execution counts: `total_events`, `total_steps`, `total_tokens`, `prompt_tokens`, `generated_tokens`.
- Validation checksum: `sha256_checksum`.

Trace readers MUST automatically discover the associated manifest, validate versioning, and reject interpreting a manifest as a trace file.

---

## 2. KV Physical Sizing Contract (`common/schemas/kv_block.py`)

- **Key Flash Page**: Exactly `4,096` bytes (4 KiB physical flash page).
- **Value Flash Page**: Exactly `4,096` bytes (4 KiB physical flash page).
- **Logical KV Block**: Exactly `8,192` bytes (combined K + V block).
- **Tokens Per Block**: 16 tokens (for FP32 head_dim=64 or FP16 head_dim=128).

Storage backends must NEVER hardcode 4096 bytes when requests specify explicit lengths (e.g., 8192 bytes for combined blocks, 335872 bytes for batch top-k candidate filtering).

---

## 3. Storage Subsystem Contract (`person3_system/storage/backend.py`)

All storage backends (`MockStorageBackend`, `FileStorageBackend`, `AnalyticalFTLBackend`, `NVMeStorageBackend`) adhere to:

```python
class StorageBackend(ABC):
    def read(self, block_id: int, offset: int = 0, length: int = 4096, layer_id: int = 0, head_id: int = 0, **kwargs) -> StorageResult: ...
    def write(self, block_id: int, offset: int = 0, data: bytes = b"", layer_id: int = 0, head_id: int = 0, length: Optional[int] = None, **kwargs) -> StorageResult: ...
    def submit(self, request: StorageRequest) -> StorageResult: ...
    def submit_batch(self, requests: List[StorageRequest]) -> List[StorageResult]: ...
    def get_stats(self) -> Dict[str, Any]: ...
    def get_telemetry(self) -> Dict[str, Any]: ...
    def close(self) -> None: ...
```

---

## 4. Deterministic Physical Mapping Contract (`person2_ssd/kv_allocator/tensor_mapping.py`)

P3 integrates directly with P2's canonical `DeterministicTensorMapper`:
- Physical channel mapping: `ch = (layer + head + b_idx + (b_idx // channels)) % channels`
- Flash page alignment: Key pages mapped to even flash pages, Value pages mapped to odd flash pages.
- Striped across 8 independent flash channels for maximal parallel throughput.
