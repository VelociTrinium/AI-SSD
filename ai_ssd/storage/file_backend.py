"""Direct OS File-Backed Block Storage Backend with In-Storage Top-K Filtering."""

from __future__ import annotations
import os
import sys
import tempfile
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

from ai_ssd.constants import KEY_PAGE_BYTES, VALUE_PAGE_BYTES, LOGICAL_BLOCK_BYTES, TOKENS_PER_BLOCK, DEFAULT_HEAD_DIM
from ai_ssd.storage.base import BaseStorageBackend
from ai_ssd.storage.ftl_striping import TensorAwareFTLMapper
from ai_ssd.c_kernel.binding import get_native_c_kernel


class FileStorageBackend(BaseStorageBackend):
    """
    Direct OS file-backed computational storage backend.
    
    Stores actual KV tensors in a local binary file.
    Executes in-storage Top-K filtering internally via native C SIMD vector kernels.
    Preserves Zero-PCIe Candidate Key offload invariant (strictly 0 B returned for unpruned keys).
    """

    CLASSIFICATION: str = "ANALYTICAL"

    def __init__(
        self,
        backing_path: Optional[str] = None,
        channels: int = 8,
        num_layers: int = 36,
        num_heads: int = 2,
        tokens_per_block: int = TOKENS_PER_BLOCK,
        head_dim: int = DEFAULT_HEAD_DIM,
        dtype: str = "float32",
        mapping_mode: str = "tensor_aware",
    ):
        self.channels = channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.tokens_per_block = tokens_per_block
        self.head_dim = head_dim
        self.dtype_str = str(dtype).lower()
        self.mapping_mode = mapping_mode

        self.mapper = TensorAwareFTLMapper(channels=channels)
        self.kernel = get_native_c_kernel()

        # Temporary backing file
        if backing_path:
            self._backing_path = backing_path
            self._is_temp = False
        else:
            self._backing_path = os.path.join(
                tempfile.gettempdir(), f"aissd_file_{os.getpid()}_{id(self)}.bin"
            )
            self._is_temp = True

        self._backing_fd = os.open(
            self._backing_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC
        )
        self._file_offset: int = 0

        # Metadata table: (layer_idx, block_id) -> descriptor
        self._storage: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Telemetry counters
        self.bytes_read: int = 0
        self.bytes_written: int = 0
        self.blocks_read: int = 0
        self.blocks_written: int = 0
        self.requests: int = 0
        self.storage_batches: int = 0
        self.storage_computational_topk_calls: int = 0
        self.storage_internal_k_bytes: int = 0
        self._channel_counters: Dict[int, int] = {c: 0 for c in range(self.channels)}
        self._per_channel_reads: Dict[int, int] = {c: 0 for c in range(self.channels)}
        self._per_channel_writes: Dict[int, int] = {c: 0 for c in range(self.channels)}

    def close(self) -> None:
        """Closes file descriptor and removes temp file if applicable."""
        if self._backing_fd is not None:
            try:
                os.close(self._backing_fd)
            except OSError:
                pass
            self._backing_fd = None

        if self._is_temp and self._backing_path and os.path.exists(self._backing_path):
            try:
                os.unlink(self._backing_path)
            except OSError:
                pass
            self._backing_path = None

    def __del__(self) -> None:
        self.close()

    def write_block(
        self,
        layer_idx: int,
        block_id: int,
        k_block: Union[np.ndarray, bytes],
        v_block: Union[np.ndarray, bytes],
        head_id: int = 0,
    ) -> None:
        if isinstance(k_block, np.ndarray):
            k_arr = np.ascontiguousarray(k_block)
            k_shape = k_arr.shape
            k_dtype = k_arr.dtype
            k_bytes = k_arr.tobytes()
        else:
            k_bytes = bytes(k_block)
            k_shape = (self.tokens_per_block, self.num_heads, self.head_dim)
            k_dtype = np.float32

        if isinstance(v_block, np.ndarray):
            v_arr = np.ascontiguousarray(v_block)
            v_shape = v_arr.shape
            v_dtype = v_arr.dtype
            v_bytes = v_arr.tobytes()
        else:
            v_bytes = bytes(v_block)
            v_shape = (self.tokens_per_block, self.num_heads, self.head_dim)
            v_dtype = np.float32

        k_size = len(k_bytes)
        v_size = len(v_bytes)

        k_offset = self._file_offset
        self._safe_pwrite(self._backing_fd, k_bytes, k_offset)
        self._file_offset += k_size

        v_offset = self._file_offset
        self._safe_pwrite(self._backing_fd, v_bytes, v_offset)
        self._file_offset += v_size

        coord = self.mapper.map_block(
            layer_id=layer_idx,
            head_id=head_id,
            block_id=block_id,
            mode=self.mapping_mode,
            channel_counters=self._channel_counters,
        )
        self._channel_counters[coord.channel] += 1
        self._per_channel_writes[coord.channel] += 1

        self._storage[(layer_idx, block_id)] = {
            "k_offset": k_offset,
            "k_size": k_size,
            "v_offset": v_offset,
            "v_size": v_size,
            "k_shape": k_shape,
            "v_shape": v_shape,
            "k_dtype": k_dtype,
            "v_dtype": v_dtype,
            "channel": coord.channel,
        }

        self.blocks_written += 1
        self.bytes_written += (k_size + v_size)

    def read_key_page(self, layer_idx: int, block_id: int) -> np.ndarray:
        entry = self._storage[(layer_idx, block_id)]
        raw = self._safe_pread(self._backing_fd, entry["k_size"], entry["k_offset"])
        self.requests += 1
        self.bytes_read += entry["k_size"]
        self._per_channel_reads[entry["channel"]] += 1
        return np.frombuffer(raw, dtype=entry["k_dtype"]).reshape(entry["k_shape"])

    def read_value_page(self, layer_idx: int, block_id: int) -> np.ndarray:
        entry = self._storage[(layer_idx, block_id)]
        raw = self._safe_pread(self._backing_fd, entry["v_size"], entry["v_offset"])
        self.requests += 1
        self.blocks_read += 1
        self.bytes_read += entry["v_size"]
        self._per_channel_reads[entry["channel"]] += 1
        return np.frombuffer(raw, dtype=entry["v_dtype"]).reshape(entry["v_shape"])

    def read_block_batch(
        self,
        layer_idx: int,
        block_ids: List[int],
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        res = {}
        for bid in block_ids:
            res[bid] = (self.read_key_page(layer_idx, bid), self.read_value_page(layer_idx, bid))
        self.storage_batches += 1
        return res

    def compute_topk_filter(
        self,
        layer_idx: int,
        cand_bids: List[Tuple[int, int]],
        query: np.ndarray,
        top_k: int,
        scale: float,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> List[Tuple[float, int, int]]:
        if not cand_bids:
            return []

        cands_info = []
        for bid, actual_tokens in cand_bids:
            entry = self._storage.get((layer_idx, bid))
            if entry is not None:
                cands_info.append((entry["k_offset"], entry["k_size"], bid, actual_tokens))
                self._per_channel_reads[entry["channel"]] += 1
            else:
                cands_info.append((0, 4096, bid, actual_tokens))

        # In-storage controller internal scan:
        k_blocks_list = []
        act_tokens_list = []
        for offset, length, bid, act_tok in cands_info:
            raw_bytes = self._safe_pread(self._backing_fd, length, offset)
            entry = self._storage.get((layer_idx, bid))
            shape = entry["k_shape"] if entry else (self.tokens_per_block, self.num_heads, self.head_dim)
            dtype = entry["k_dtype"] if entry else np.float32
            k_blk = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
            k_blocks_list.append(k_blk)
            act_tokens_list.append(act_tok)

        # Vectorized SIMD Top-K evaluation
        if self.kernel.is_available() and hasattr(self.kernel._lib, "instorage_topk_filter_gqa_avx2"):
            top_indices, top_scores = self.kernel.compute_topk_gqa(
                query=query,
                k_blocks=k_blocks_list,
                actual_tokens=act_tokens_list,
                top_k=top_k,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                scale=scale,
            )
            results = [(float(top_scores[i]), cand_bids[idx][0], cand_bids[idx][1]) for i, idx in enumerate(top_indices)]
        else:
            # Fallback Python / numpy scoring
            scores = []
            gqa_ratio = q_heads // kv_heads if kv_heads > 0 else 1
            for i, k_blk in enumerate(k_blocks_list):
                bid, act_tok = cand_bids[i]
                dots = np.einsum("hd,thd->th", query, k_blk[:, [h // gqa_ratio for h in range(q_heads)], :]) * scale
                max_score = float(np.max(dots[:act_tok]))
                scores.append((max_score, bid, act_tok))
            scores.sort(key=lambda x: x[0], reverse=True)
            results = scores[:top_k]

        total_scanned_bytes = sum(item[1] for item in cands_info)
        self.storage_batches += 1
        self.requests += 1
        self.storage_computational_topk_calls += 1
        self.storage_internal_k_bytes += total_scanned_bytes
        self.bytes_read += total_scanned_bytes

        return results

    def reset_stats(self) -> None:
        self.bytes_read = 0
        self.bytes_written = 0
        self.blocks_read = 0
        self.blocks_written = 0
        self.requests = 0
        self.storage_batches = 0
        self.storage_computational_topk_calls = 0
        self.storage_internal_k_bytes = 0

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "blocks_read": self.blocks_read,
            "blocks_written": self.blocks_written,
            "requests": self.requests,
            "storage_batches": self.storage_batches,
            "computational_topk_calls": self.storage_computational_topk_calls,
            "internal_k_bytes_scanned": self.storage_internal_k_bytes,
            "per_channel_reads": dict(self._per_channel_reads),
            "per_channel_writes": dict(self._per_channel_writes),
        }

    @staticmethod
    def _safe_pwrite(fd: int, data: bytes, offset: int) -> None:
        if hasattr(os, "pwrite"):
            os.pwrite(fd, data, offset)
        else:
            os.lseek(fd, offset, os.SEEK_SET)
            os.write(fd, data)

    @staticmethod
    def _safe_pread(fd: int, length: int, offset: int) -> bytes:
        if hasattr(os, "pread"):
            return os.pread(fd, length, offset)
        else:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, length)
