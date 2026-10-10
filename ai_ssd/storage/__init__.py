"""Storage backends and prefetch adapters."""

from ai_ssd.storage.base import BaseStorageBackend
from ai_ssd.storage.file_backend import FileStorageBackend
from ai_ssd.storage.nvme_backend import NvmeStorageBackend
from ai_ssd.storage.prefetcher import PrefetchStorageAdapter
from ai_ssd.storage.ftl_striping import TensorAwareFTLMapper, NANDCoordinate

__all__ = [
    "BaseStorageBackend",
    "FileStorageBackend",
    "NvmeStorageBackend",
    "PrefetchStorageAdapter",
    "TensorAwareFTLMapper",
    "NANDCoordinate",
]
