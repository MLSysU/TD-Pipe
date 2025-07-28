"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple, Optional

import torch

from TD_Pipe._C import cache_ops
from TD_Pipe.config import CacheConfig, ModelConfig, ParallelConfig
from TD_Pipe.logger import init_logger
from TD_Pipe.utils import in_wsl

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)
        self.dtype = model_config.dtype

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, "cuda", self.model_config.attention_backend)
        self.cpu_cache = self._allocate_kv_cache(
            self.num_cpu_blocks, "cpu", self.model_config.attention_backend)

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    def get_key_block_shape(self, num_blocks: int,
                            attention_backend: str) -> Tuple[int, ...]:
        if attention_backend == "xformers":
            element_size = torch.tensor([], dtype=self.dtype).element_size()
            x = 16 // element_size
            return (
                num_blocks,
                self.num_heads,
                self.head_size // x,
                self.block_size,
                x,
            )
        elif attention_backend == "flashAttention":
            if self.block_size % 16 != 0:
                raise ValueError("Block size must be a multiple of 16.")
            return (num_blocks, self.block_size, self.num_heads,
                    self.head_size)
        elif attention_backend == "flashInfer":
            return (num_blocks, self.block_size, self.num_heads,
                    self.head_size)

    def get_value_block_shape(self, num_blocks: int,
                              attention_backend: str) -> Tuple[int, ...]:
        if attention_backend == "xformers":
            return (
                num_blocks,
                self.num_heads,
                self.head_size,
                self.block_size,
            )
        elif attention_backend == "flashAttention":
            if self.block_size % 16 != 0:
                raise ValueError("Block size must be a multiple of 16.")
            return (num_blocks, self.block_size, self.num_heads,
                    self.head_size)
        elif attention_backend == "flashInfer":
            return (num_blocks, self.block_size, self.num_heads,
                    self.head_size)

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
        attention_backend: Optional[str] = "flashAttention",
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        key_cache_shape = self.get_key_block_shape(num_blocks,
                                                   attention_backend)
        value_cache_shape = self.get_value_block_shape(num_blocks,
                                                       attention_backend)
        pin_memory = not in_wsl() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            key_blocks = torch.zeros(
                size=key_cache_shape,
                dtype=self.dtype,
                pin_memory=pin_memory,
                device=device,
            )
            value_blocks = torch.zeros(
                size=value_cache_shape,
                dtype=self.dtype,
                pin_memory=pin_memory,
                device=device,
            )
            kv_cache.append((key_blocks, value_blocks))
        return kv_cache

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(src_value_cache, dst_value_cache,
                                      src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
