"""QEMU / Physical NVMe Computational Storage Daemon Client Backend."""

from __future__ import annotations
import os
import sys
import time
import socket
import struct
import threading
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

from ai_ssd.constants import (
    KEY_PAGE_BYTES,
    VALUE_PAGE_BYTES,
    LOGICAL_BLOCK_BYTES,
    DEFAULT_NVME_HOST,
    DEFAULT_NVME_PORT,
    MAGIC_NVME,
    OP_WRITE,
    OP_READ,
    OP_PING,
    OP_BATCH_READ,
    OP_COMPUTE_TOPK,
    TOKENS_PER_BLOCK,
    DEFAULT_HEAD_DIM,
)
from ai_ssd.storage.base import BaseStorageBackend
from ai_ssd.storage.ftl_striping import TensorAwareFTLMapper


HEADER_FMT = "<IBBHQI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

TOPK_REQ_FMT = "<IIIIIf"
TOPK_REQ_SIZE = struct.calcsize(TOPK_REQ_FMT)

TOPK_CAND_FMT = "<QQII"  # offset, length, block_id, actual_tokens
TOPK_CAND_SIZE = struct.calcsize(TOPK_CAND_FMT)

TOPK_RESP_FMT = "<IfI"   # block_id, score, actual_tokens
TOPK_RESP_SIZE = struct.calcsize(TOPK_RESP_FMT)


class NvmeStorageBackend(BaseStorageBackend):
    """
    Client backend connecting host Python processes directly to the NVMe guest daemon.
    Dispatches block I/O and in-storage Top-K vector scoring over low-latency TCP socket.
    """

    CLASSIFICATION: str = "VIRTUAL-DEVICE"

    def __init__(
        self,
        host: str = DEFAULT_NVME_HOST,
        port: int = DEFAULT_NVME_PORT,
        channels: int = 8,
        num_layers: int = 36,
        num_heads: int = 2,
        tokens_per_block: int = TOKENS_PER_BLOCK,
        head_dim: int = DEFAULT_HEAD_DIM,
        dtype: str = "float32",
        mapping_mode: str = "tensor_aware",
    ):
        self.host = host
        self.port = port
        self.channels = channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.tokens_per_block = tokens_per_block
        self.head_dim = head_dim
        self.dtype_str = str(dtype).lower()
        self.mapping_mode = mapping_mode

        self.mapper = TensorAwareFTLMapper(channels=channels)
        self.sock: Optional[socket.socket] = None
        self._lock = threading.RLock()

        self._storage: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._device_offset: int = 0

        # Telemetry
        self.bytes_read: int = 0
        self.bytes_written: int = 0
        self.blocks_read: int = 0
        self.blocks_written: int = 0
        self.requests: int = 0
        self.storage_batches: int = 0
        self.storage_computational_topk_calls: int = 0
        self.storage_internal_k_bytes: int = 0
        self._per_channel_reads: Dict[int, int] = {c: 0 for c in range(self.channels)}
        self._per_channel_writes: Dict[int, int] = {c: 0 for c in range(self.channels)}

        self._connect()

    def _connect(self) -> None:
        with self._lock:
            if self.sock is not None:
                try:
                    self.sock.close()
                except OSError:
                    pass
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(10.0)
            s.connect((self.host, self.port))
            self.sock = s

    def close(self) -> None:
        with self._lock:
            if self.sock is not None:
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None

    def write_block(
        self,
        layer_idx: int,
        block_id: int,
        k_block: Union[np.ndarray, bytes],
        v_block: Union[np.ndarray, bytes],
        head_id: int = 0,
    ) -> None:
        k_bytes = k_block.tobytes() if isinstance(k_block, np.ndarray) else bytes(k_block)
        v_bytes = v_block.tobytes() if isinstance(v_block, np.ndarray) else bytes(v_block)

        k_size = len(k_bytes)
        v_size = len(v_bytes)

        k_offset = self._device_offset
        self._write_raw(k_offset, k_bytes)
        self._device_offset += k_size

        v_offset = self._device_offset
        self._write_raw(v_offset, v_bytes)
        self._device_offset += v_size

        coord = self.mapper.map_block(
            layer_id=layer_idx, head_id=head_id, block_id=block_id, mode=self.mapping_mode
        )
        self._per_channel_writes[coord.channel] += 1

        k_shape = k_block.shape if isinstance(k_block, np.ndarray) else (self.tokens_per_block, self.num_heads, self.head_dim)
        v_shape = v_block.shape if isinstance(v_block, np.ndarray) else (self.tokens_per_block, self.num_heads, self.head_dim)
        k_dtype = k_block.dtype if isinstance(k_block, np.ndarray) else np.float32
        v_dtype = v_block.dtype if isinstance(v_block, np.ndarray) else np.float32

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
        raw = self._read_raw(entry["k_offset"], entry["k_size"])
        self.requests += 1
        self.bytes_read += entry["k_size"]
        self._per_channel_reads[entry["channel"]] += 1
        return np.frombuffer(raw, dtype=entry["k_dtype"]).reshape(entry["k_shape"])

    def read_value_page(self, layer_idx: int, block_id: int) -> np.ndarray:
        entry = self._storage[(layer_idx, block_id)]
        raw = self._read_raw(entry["v_offset"], entry["v_size"])
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

        cands_data = bytearray()
        total_scanned_bytes = 0
        for bid, actual_tokens in cand_bids:
            entry = self._storage.get((layer_idx, bid))
            if entry:
                offset, length = entry["k_offset"], entry["k_size"]
                self._per_channel_reads[entry["channel"]] += 1
            else:
                offset, length = 0, 4096
            total_scanned_bytes += length
            cands_data.extend(struct.pack(TOPK_CAND_FMT, offset, length, bid, actual_tokens))

        q_flat = np.ascontiguousarray(query, dtype=np.float32).tobytes()
        num_candidates = len(cand_bids)

        topk_hdr = struct.pack(
            TOPK_REQ_FMT,
            num_candidates,
            top_k,
            q_heads,
            kv_heads,
            head_dim,
            scale,
        )

        payload = topk_hdr + bytes(cands_data) + q_flat
        hdr = struct.pack(HEADER_FMT, MAGIC_NVME, OP_COMPUTE_TOPK, 0, 0, 0, len(payload))

        with self._lock:
            self._send_all(hdr + payload)
            resp_hdr = self._recv_exact(HEADER_SIZE)
            magic, status, op, _, _, resp_len = struct.unpack(HEADER_FMT, resp_hdr)
            if status != 0:
                raise RuntimeError(f"NVMe daemon OP_COMPUTE_TOPK error status: {status}")
            resp_body = self._recv_exact(resp_len)

        results = []
        for i in range(0, resp_len, TOPK_RESP_SIZE):
            bid, score, actual_tokens = struct.unpack(TOPK_RESP_FMT, resp_body[i:i + TOPK_RESP_SIZE])
            results.append((float(score), int(bid), int(actual_tokens)))

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

    def _write_raw(self, offset: int, data: bytes) -> None:
        hdr = struct.pack(HEADER_FMT, MAGIC_NVME, OP_WRITE, 0, 0, offset, len(data))
        with self._lock:
            self._send_all(hdr + data)
            resp_hdr = self._recv_exact(HEADER_SIZE)
            _, status, _, _, _, _ = struct.unpack(HEADER_FMT, resp_hdr)
            if status != 0:
                raise IOError(f"NVMe write failed at offset {offset} with status {status}")

    def _read_raw(self, offset: int, length: int) -> bytes:
        hdr = struct.pack(HEADER_FMT, MAGIC_NVME, OP_READ, 0, 0, offset, length)
        with self._lock:
            self._send_all(hdr)
            resp_hdr = self._recv_exact(HEADER_SIZE)
            _, status, _, _, _, resp_len = struct.unpack(HEADER_FMT, resp_hdr)
            if status != 0:
                raise IOError(f"NVMe read failed at offset {offset} with status {status}")
            return self._recv_exact(resp_len)

    def _send_all(self, data: bytes) -> None:
        if self.sock is None:
            self._connect()
        total_sent = 0
        while total_sent < len(data):
            sent = self.sock.send(data[total_sent:])
            if sent == 0:
                raise ConnectionResetError("NVMe guest daemon closed socket")
            total_sent += sent

    def _recv_exact(self, length: int) -> bytes:
        if self.sock is None:
            self._connect()
        buf = bytearray(length)
        view = memoryview(buf)
        total_recvd = 0
        while total_recvd < length:
            recvd = self.sock.recv_into(view[total_recvd:])
            if recvd == 0:
                raise ConnectionResetError("NVMe guest daemon closed socket prematurely")
            total_recvd += recvd
        return bytes(buf)
