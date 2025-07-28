"""Attention layer."""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from TD_Pipe.logger import init_logger
from TD_Pipe.model_executor.input_metadata import InputMetadata
from TD_Pipe.utils import is_hip

logger = init_logger(__name__)


class Attention(nn.Module):
    """Attention layer.
    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:
    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Output the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        attention_backend: Optional[str] = "flashAttention",
    ) -> None:
        super().__init__()
        if attention_backend == "flashAttention":
            from TD_Pipe.model_executor.layers.attention.backends.flash_attn import FlashAttentionBackend
            self.backend = FlashAttentionBackend(num_heads, head_size, scale,
                                                 num_kv_heads, alibi_slopes,
                                                 sliding_window)
        elif attention_backend == "xformers":
            from TD_Pipe.model_executor.layers.attention.backends.xformers import XFormersBackend
            self.backend = XFormersBackend(num_heads, head_size, scale,
                                           num_kv_heads, alibi_slopes,
                                           sliding_window)
        elif attention_backend == "flashInfer":
            from TD_Pipe.model_executor.layers.attention.backends.flashinfer import FlashInferBackend
            self.backend = FlashInferBackend(num_heads, head_size, scale,
                                             num_kv_heads, alibi_slopes,
                                             sliding_window)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        return self.backend.forward(query, key, value, key_cache, value_cache,
                                    input_metadata)

    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        raise NotImplementedError
