"""Abstract Base Storage Backend and Data Contracts for AI-SSD."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class BaseStorageBackend(ABC):
    """Abstract interface for AI-SSD computational storage backends."""

    CLASSIFICATION: str = "ANALYTICAL"

    @abstractmethod
    def write_block(
        self,
        layer_idx: int,
        block_id: int,
        k_block: np.ndarray,
        v_block: np.ndarray,
    ) -> None:
        """Writes an 8 KiB logical KV block (4 KiB Key + 4 KiB Value)."""
        pass

    @abstractmethod
    def read_key_page(self, layer_idx: int, block_id: int) -> np.ndarray:
        """Reads a 4 KiB Key page."""
        pass

    @abstractmethod
    def read_value_page(self, layer_idx: int, block_id: int) -> np.ndarray:
        """Reads a 4 KiB Value page."""
        pass

    @abstractmethod
    def read_block_batch(
        self,
        layer_idx: int,
        block_ids: List[int],
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Contiguously batch-reads winning KV blocks (cuts I/O transactions by 50%)."""
        pass

    @abstractmethod
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
        """
        Executes In-Storage Top-K selection on candidate blocks.
        Invariant: Zero candidate Key blocks are transferred across PCIe.
        Returns: [(score, block_id, actual_tokens)]
        """
        pass

    def predict_and_prefetch(
        self,
        current_layer_id: int,
        current_block_ids: List[int],
        async_mode: bool = False,
    ) -> None:
        """Optional prefetch hook for next layer."""
        pass

    @abstractmethod
    def reset_stats(self) -> None:
        """Resets performance and telemetry counters."""
        pass

    @abstractmethod
    def get_telemetry(self) -> Dict[str, Any]:
        """Returns storage throughput and channel telemetry."""
        pass

    def close(self) -> None:
        """Closes resources, files, or connections."""
        pass

    def __enter__(self) -> BaseStorageBackend:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
