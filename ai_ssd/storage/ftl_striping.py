"""Tensor-Aware Multi-Channel Flash Translation Layer (FTL) Striping Math."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
from ai_ssd.constants import DEFAULT_CHANNELS, DEFAULT_DIES_PER_CHANNEL, DEFAULT_PLANES_PER_DIE


@dataclass(frozen=True)
class NANDCoordinate:
    channel: int
    die: int
    plane: int
    block: int
    page: int

    def __str__(self) -> str:
        return f"ch{self.channel}_die{self.die}_pl{self.plane}_blk{self.block}_pg{self.page}"


class TensorAwareFTLMapper:
    """Calculates channel striping to eliminate serialization bottlenecks."""

    def __init__(
        self,
        channels: int = DEFAULT_CHANNELS,
        dies_per_channel: int = DEFAULT_DIES_PER_CHANNEL,
        planes_per_die: int = DEFAULT_PLANES_PER_DIE,
        pages_per_block: int = 256,
        blocks_per_plane: int = 1024,
    ):
        self.channels = channels
        self.dies_per_channel = dies_per_channel
        self.planes_per_die = planes_per_die
        self.pages_per_block = pages_per_block
        self.blocks_per_plane = blocks_per_plane

    def map_block(
        self,
        layer_id: int,
        head_id: int,
        block_id: int,
        mode: str = "tensor_aware",
        channel_counters: Optional[Dict[int, int]] = None,
    ) -> NANDCoordinate:
        """Maps (layer, head, block) coordinate to physical channel & NAND geometry."""
        if mode == "conventional":
            ch = 0
            die = 0
            pl = 0
        else:
            # Tensor-aware placement incorporates layer, head, and token block index
            ch = (layer_id + head_id + block_id + (block_id // self.channels)) % self.channels
            die = (layer_id + (head_id // self.channels) + (block_id // self.channels)) % self.dies_per_channel
            pl = (block_id // (self.channels * self.dies_per_channel)) % self.planes_per_die

        counter = 0 if channel_counters is None else channel_counters.get(ch, 0)
        pg = counter % self.pages_per_block
        blk = (counter // self.pages_per_block) % self.blocks_per_plane
        return NANDCoordinate(channel=ch, die=die, plane=pl, block=blk, page=pg)
