"""AI-SSD: Computational Storage Architecture for Long-Context LLM Inference."""

__version__ = "2.0.0"

from ai_ssd.constants import (
    KEY_PAGE_BYTES,
    VALUE_PAGE_BYTES,
    LOGICAL_BLOCK_BYTES,
    TOKENS_PER_BLOCK,
    DEFAULT_SINK_TOKENS,
    DEFAULT_RECENT_TOKENS,
    DEFAULT_TOP_K_PCT,
)

__all__ = [
    "KEY_PAGE_BYTES",
    "VALUE_PAGE_BYTES",
    "LOGICAL_BLOCK_BYTES",
    "TOKENS_PER_BLOCK",
    "DEFAULT_SINK_TOKENS",
    "DEFAULT_RECENT_TOKENS",
    "DEFAULT_TOP_K_PCT",
]
