"""Physical and Architectural Constants for AI-SSD V2."""

from typing import Final

# Canonical Physical Geometry (Codified Contract)
# 16 tokens x 1 KV head x 64 dimensions x 4 bytes (FP32) = 4,096 bytes per page
KEY_PAGE_BYTES: Final[int] = 4096         # 4 KiB Key Page
VALUE_PAGE_BYTES: Final[int] = 4096       # 4 KiB Value Page
LOGICAL_BLOCK_BYTES: Final[int] = 8192    # 8 KiB Logical Block (K + V)

# Token Chunking & Cache Tiering Defaults
TOKENS_PER_BLOCK: Final[int] = 16
DEFAULT_SINK_TOKENS: Final[int] = 4       # Attention sinks retained permanently in Host DRAM
DEFAULT_RECENT_TOKENS: Final[int] = 16    # Sliding window recency cache in Host DRAM
DEFAULT_TOP_K_PCT: Final[float] = 10.0    # Top-K candidate percentage selected from storage
DEFAULT_HEAD_DIM: Final[int] = 64

# Multi-channel Striping Parameters
DEFAULT_CHANNELS: Final[int] = 8
DEFAULT_DIES_PER_CHANNEL: Final[int] = 4
DEFAULT_PLANES_PER_DIE: Final[int] = 2

# Hardware & Daemon Socket Defaults
DEFAULT_NVME_HOST: Final[str] = "127.0.0.1"
DEFAULT_NVME_PORT: Final[int] = 9999
MAGIC_NVME: Final[int] = 0x4E564D45       # 'NVME' ASCII magic

# Protocol Operations
OP_WRITE: Final[int] = 1
OP_READ: Final[int] = 2
OP_PING: Final[int] = 3
OP_FLUSH: Final[int] = 4
OP_SHUTDOWN: Final[int] = 5
OP_BATCH_READ: Final[int] = 6
OP_COMPUTE_TOPK: Final[int] = 7
