"""Dynamic Self-Attention Interception Hook for AI-SSD Storage Offload."""

from __future__ import annotations
import time
from typing import Dict, Any, Tuple, Callable, Optional
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from ai_ssd.engine.kv_manager import AISSDKVManager


def wrap_model_attention(
    model: torch.nn.Module,
    kv_mgr: AISSDKVManager,
) -> Tuple[Dict[int, Callable], Callable[[], None]]:
    """
    Dynamically wraps self_attn.forward across all transformer layers.
    Routes historical KV access during decode through the AI-SSD storage backend.
    Returns (orig_forwards_dict, restore_function).
    """
    orig_forwards: Dict[int, Callable] = {}
    layers = getattr(model, "model", model).layers

    for i, layer in enumerate(layers):
        attn = layer.self_attn
        orig_forwards[i] = attn.forward

        def make_forward(layer_idx: int, original_fwd: Callable):
            def forward(
                hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Any] = None,
                **kwargs: Any,
            ):
                # If prefill (seq_len > 1) or manager inactive, use original forward
                if not kv_mgr.is_active or hidden_states.shape[1] > 1:
                    return original_fwd(
                        hidden_states, position_embeddings,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        **kwargs,
                    )

                attn_mod = layers[layer_idx].self_attn
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_mod.head_dim)

                # 1. Linear projections
                q_raw = attn_mod.q_proj(hidden_states).view(hidden_shape)
                k_raw = attn_mod.k_proj(hidden_states).view(hidden_shape)
                if hasattr(attn_mod, "q_norm") and attn_mod.q_norm is not None:
                    q_raw = attn_mod.q_norm(q_raw)
                if hasattr(attn_mod, "k_norm") and attn_mod.k_norm is not None:
                    k_raw = attn_mod.k_norm(k_raw)

                q = q_raw.transpose(1, 2)
                k = k_raw.transpose(1, 2)
                v = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                # 2. Rotary Position Embedding
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                # 3. Append new token to recent window
                kv_mgr.append_new_token(layer_idx, k, v)

                # 4. In-storage Top-K filtering & active set retrieval
                act_k, act_v = kv_mgr.select_and_fetch_active_kv(layer_idx, query=q)

                # 5. Native PyTorch fused Grouped-Query Attention (SDPA)
                scaling = getattr(attn_mod, "scaling", 1.0 / (attn_mod.head_dim ** 0.5))
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, act_k, act_v, scale=scaling, enable_gqa=True
                )
                out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()

                # 6. Output projection
                out = attn_mod.o_proj(out)
                return out, None

            return forward

        attn.forward = make_forward(i, orig_forwards[i])

    def restore_func():
        for i, layer in enumerate(layers):
            layer.self_attn.forward = orig_forwards[i]

    return orig_forwards, restore_func
