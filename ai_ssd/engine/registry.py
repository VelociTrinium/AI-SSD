"""Model Architecture Registry and Configuration Detector."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class ModelConfig:
    model_id: str
    architecture: str
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    default_precision: str = "fp32"
    supported_precisions: List[str] = None
    attention_type: str = "GQA"

    @property
    def gqa_ratio(self) -> int:
        return self.num_attention_heads // max(1, self.num_key_value_heads)


KNOWN_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen3-4b": {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "architecture": "qwen2",
        "num_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 2560,
        "vocab_size": 151936,
        "default_precision": "fp32",
        "supported_precisions": ["fp32", "float32"],
        "attention_type": "GQA",
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "architecture": "qwen2",
        "num_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "vocab_size": 151936,
        "default_precision": "fp16",
        "supported_precisions": ["fp16", "float16"],
        "attention_type": "GQA",
    },
    "qwen2.5-0.5b": {
        "model_id": "Qwen/Qwen2.5-0.5B",
        "architecture": "qwen2",
        "num_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "hidden_size": 896,
        "vocab_size": 151936,
        "default_precision": "fp32",
        "supported_precisions": ["fp32", "float32", "fp16", "float16"],
        "attention_type": "GQA",
    },
    "qwen2.5-1.5b": {
        "model_id": "Qwen/Qwen2.5-1.5B",
        "architecture": "qwen2",
        "num_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "hidden_size": 1536,
        "vocab_size": 151936,
        "default_precision": "fp32",
        "supported_precisions": ["fp32", "float32", "fp16", "float16"],
        "attention_type": "GQA",
    },
    "qwen2.5-3b": {
        "model_id": "Qwen/Qwen2.5-3B",
        "architecture": "qwen2",
        "num_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "hidden_size": 2048,
        "vocab_size": 151936,
        "default_precision": "fp32",
        "supported_precisions": ["fp32", "float32", "fp16", "float16"],
        "attention_type": "GQA",
    },
    "tiny-mistral": {
        "model_id": "openaccess-ai-collective/tiny-mistral",
        "architecture": "mistral",
        "num_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "hidden_size": 1024,
        "vocab_size": 32000,
        "default_precision": "fp32",
        "supported_precisions": ["fp32", "float32"],
        "attention_type": "GQA",
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "architecture": "mistral",
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "vocab_size": 32000,
        "default_precision": "fp16",
        "supported_precisions": ["fp16", "float16"],
        "attention_type": "GQA",
    },
}


class ModelRegistry:
    """Discovers and resolves ModelConfig from HF IDs or aliases."""

    @staticmethod
    def resolve_config(target: str, model_obj: Optional[Any] = None) -> ModelConfig:
        key = target.strip().lower()
        if key in KNOWN_MODELS:
            entry = KNOWN_MODELS[key]
            return ModelConfig(
                model_id=entry["model_id"],
                architecture=entry["architecture"],
                num_layers=entry["num_layers"],
                num_attention_heads=entry["num_attention_heads"],
                num_key_value_heads=entry["num_key_value_heads"],
                head_dim=entry["head_dim"],
                hidden_size=entry["hidden_size"],
                vocab_size=entry["vocab_size"],
                default_precision=entry["default_precision"],
                supported_precisions=entry.get("supported_precisions", ["fp32"]),
                attention_type=entry.get("attention_type", "GQA"),
            )

        # Inspect model object config directly if passed
        cfg = getattr(model_obj, "config", None) if model_obj is not None else None
        num_layers = getattr(cfg, "num_hidden_layers", 36) if cfg else 36
        num_attn_heads = getattr(cfg, "num_attention_heads", 32) if cfg else 32
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_attn_heads) if cfg else 8
        head_dim = getattr(cfg, "head_dim", 128) if cfg else 128
        hidden_size = getattr(cfg, "hidden_size", 2560) if cfg else 2560
        vocab_size = getattr(cfg, "vocab_size", 151936) if cfg else 151936
        arch = getattr(cfg, "model_type", "qwen2") if cfg else "qwen2"

        return ModelConfig(
            model_id=target,
            architecture=arch,
            num_layers=num_layers,
            num_attention_heads=num_attn_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            default_precision="fp32",
            supported_precisions=["fp32", "float32", "fp16", "float16"],
            attention_type="GQA" if num_kv_heads < num_attn_heads else "MHA",
        )
