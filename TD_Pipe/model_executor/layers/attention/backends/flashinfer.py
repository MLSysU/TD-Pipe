from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from vllm_flash_attn import flash_attn_varlen_func
import torch

from TD_Pipe.model_executor.input_metadata import InputMetadata
from TD_Pipe.model_executor.layers.attention.ops.paged_attn import (
    PagedAttentionImpl)
from TD_Pipe._C import cache_ops


class FlashInferBackend():

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        if sliding_window is not None:
            raise ValueError("Sliding window is not supported in FlashInfer.")
        self.sliding_window = (-1, -1)
        self.alibi_slopes = alibi_slopes
        self.scale = scale
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ):
        num_tokens, hidden_size = query.shape
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        if key_cache is not None and value_cache is not None:
            cache_ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                input_metadata.slot_mapping.flatten(),
            )

        if input_metadata.is_prompt:
            output = flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=input_metadata.seq_start_loc,
                cu_seqlens_k=input_metadata.seq_start_loc,
                max_seqlen_q=input_metadata.max_seq_len,
                max_seqlen_k=input_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
            )
        else:
            query = query.contiguous(
            )  # Flashinfer requires query to be contiguous
            output = input_metadata.decode_wrapper.forward(
                query,
                (key_cache, value_cache),
            )
        return output.view(num_tokens, hidden_size)
