"""Python ctypes interface to the compiled native C in-storage kernel."""

from __future__ import annotations
import ctypes
import os
import platform
import math
from pathlib import Path
from typing import List, Tuple, Any, Optional
import numpy as np


class NativeCKernel:
    """Wrapper loading and binding to instorage_attention.so (Linux) or .dll (Windows)."""

    def __init__(self, lib_path: Optional[Path] = None):
        kernel_dir = Path(__file__).parent.resolve()
        if lib_path is None:
            if platform.system() == "Windows":
                lib_path = kernel_dir / "instorage_attention.dll"
            else:
                lib_path = kernel_dir / "instorage_attention.so"
                if not lib_path.exists():
                    lib_path = kernel_dir / "instorage_attention.dll"

        self.dll_path = lib_path
        self.lib_path = lib_path
        self._loaded = False
        self._lib = None

        if self.lib_path.exists():
            try:
                self._lib = ctypes.CDLL(str(self.lib_path))
                self._setup_function_signatures()
                self._loaded = True
            except Exception as e:
                print(f"[WARNING] Failed to load native library {self.lib_path}: {e}")
                self._loaded = False

    def is_available(self) -> bool:
        return self._loaded

    def _setup_function_signatures(self) -> None:
        # compute_block_score
        self._lib.compute_block_score.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        self._lib.compute_block_score.restype = ctypes.c_float

        # instorage_topk_filter
        self._lib.instorage_topk_filter.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.instorage_topk_filter.restype = ctypes.c_int

        # compute_block_score_gqa_avx2
        if hasattr(self._lib, "compute_block_score_gqa_avx2"):
            self._lib.compute_block_score_gqa_avx2.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]
            self._lib.compute_block_score_gqa_avx2.restype = ctypes.c_float

        # instorage_topk_filter_gqa_avx2
        if hasattr(self._lib, "instorage_topk_filter_gqa_avx2"):
            self._lib.instorage_topk_filter_gqa_avx2.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
            ]
            self._lib.instorage_topk_filter_gqa_avx2.restype = ctypes.c_int

    def compute_topk(
        self,
        query: np.ndarray,
        k_blocks: np.ndarray,
        top_k: int,
        scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._loaded:
            raise RuntimeError("Native C kernel is not compiled or loaded.")

        q_flat = np.ascontiguousarray(query, dtype=np.float32).flatten()
        k_flat = np.ascontiguousarray(k_blocks, dtype=np.float32).flatten()

        num_blocks = k_blocks.shape[0]
        tokens = k_blocks.shape[1]
        heads = k_blocks.shape[2]
        head_dim = k_blocks.shape[3]

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        effective_k = min(top_k, num_blocks)
        out_indices = np.zeros(effective_k, dtype=np.int32)
        out_scores = np.zeros(effective_k, dtype=np.float32)

        q_ptr = q_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        k_ptr = k_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        idx_ptr = out_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        score_ptr = out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = self._lib.instorage_topk_filter(
            q_ptr, k_ptr, num_blocks, tokens, heads, head_dim,
            effective_k, ctypes.c_float(scale), idx_ptr, score_ptr
        )

        if ret != 0:
            raise RuntimeError(f"Native instorage_topk_filter failed with error code: {ret}")

        return out_indices, out_scores

    def compute_topk_gqa(
        self,
        query: np.ndarray,
        k_blocks: List[np.ndarray],
        actual_tokens: List[int],
        top_k: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._loaded or not hasattr(self._lib, "instorage_topk_filter_gqa_avx2"):
            raise RuntimeError("Native AVX2 GQA kernel is not available.")

        num_blocks = len(k_blocks)
        if num_blocks == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        effective_k = min(top_k, num_blocks)
        out_indices = np.zeros(effective_k, dtype=np.int32)
        out_scores = np.zeros(effective_k, dtype=np.float32)

        q_flat = np.ascontiguousarray(query, dtype=np.float32).flatten()
        q_ptr = q_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Array of pointers to each candidate block
        k_blocks_contiguous = [np.ascontiguousarray(b, dtype=np.float32) for b in k_blocks]
        block_ptrs_arr = (ctypes.POINTER(ctypes.c_float) * num_blocks)()
        for i, b_arr in enumerate(k_blocks_contiguous):
            block_ptrs_arr[i] = b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        tokens_arr = np.ascontiguousarray(actual_tokens, dtype=np.int32)
        tokens_ptr = tokens_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        idx_ptr = out_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        score_ptr = out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = self._lib.instorage_topk_filter_gqa_avx2(
            q_ptr, block_ptrs_arr, tokens_ptr, num_blocks,
            q_heads, kv_heads, head_dim, effective_k,
            ctypes.c_float(scale), idx_ptr, score_ptr
        )

        if ret != 0:
            raise RuntimeError(f"Native instorage_topk_filter_gqa_avx2 failed with code: {ret}")

        return out_indices, out_scores


_GLOBAL_KERNEL: Optional[NativeCKernel] = None


def get_native_c_kernel() -> NativeCKernel:
    """Returns singleton instance of NativeCKernel."""
    global _GLOBAL_KERNEL
    if _GLOBAL_KERNEL is None:
        _GLOBAL_KERNEL = NativeCKernel()
    return _GLOBAL_KERNEL
