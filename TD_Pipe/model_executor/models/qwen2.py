# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Qwen2Config
from TD_Pipe.config import ParallelConfig

from TD_Pipe.model_executor.layers.activation import SiluAndMul
from TD_Pipe.model_executor.layers.attention import Attention
from TD_Pipe.model_executor.layers.layernorm import RMSNorm
from TD_Pipe.model_executor.layers.linear import (LinearMethodBase,
                                                  MergedColumnParallelLinear,
                                                  QKVParallelLinear,
                                                  RowParallelLinear)
from TD_Pipe.model_executor.layers.rotary_embedding import get_rope
from TD_Pipe.model_executor.layers.sampler import Sampler
from TD_Pipe.model_executor.layers.projection import Projection
from TD_Pipe.sequence import SamplerOutput
from TD_Pipe.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from TD_Pipe.model_executor.sampling_metadata import SamplingMetadata

from TD_Pipe.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)

from TD_Pipe.model_executor.weight_utils import (default_weight_loader,
                                                 hf_model_weights_iterator)
from TD_Pipe.model_executor.input_metadata import InputMetadata

KVCache = Tuple[torch.Tensor, torch.Tensor]


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)

        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        attention_backend: str,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        rope_scaling: Optional[Tuple] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              attention_backend=attention_backend,
                              num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        key_cache, value_cache = kv_cache
        attn_output = self.attn(q, k, v, key_cache, value_cache,
                                input_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        attention_backend: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_backend=attention_backend,
            linear_method=linear_method,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# @support_torch_compile
class Qwen2Model(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: ParallelConfig,
        attention_backend: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList()
        for i in range(self.parallel_config.start,
                       self.parallel_config.end + 1):
            self.layers.add_module(
                f"{i}",
                Qwen2DecoderLayer(config, attention_backend, linear_method))

        if self.parallel_config.is_last:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: List[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> Tuple[torch.Tensor]:
        if self.parallel_config.is_first:
            hidden_states = self.embed_tokens(input_ids[0])
            residual = None
        else:
            assert input_ids is not None
            hidden_states = input_ids[0]
            residual = input_ids[1]

        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                residual,
            )

        if self.parallel_config.is_last:
            hidden_states, residual = self.norm(hidden_states, residual)
        return hidden_states, residual


class Qwen2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: ParallelConfig,
        attention_backend: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.parallel_config = parallel_config
        self.linear_method = linear_method
        self.model = Qwen2Model(config, parallel_config, attention_backend,
                                linear_method)
        self.lm_head_weight = self.model.embed_tokens.weight
        self.projection = Projection(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: List[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> List[torch.Tensor]:
        hidden_states, residual = self.model(input_ids, positions, kv_caches,
                                             input_metadata)
        return [hidden_states, residual]

    def get_tensor_count(self):
        return 2

    def project(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        is_prompt: bool,
    ) -> torch.Tensor:
        logits = self.projection(self.lm_head_weight, hidden_states,
                                 sampling_metadata, is_prompt)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if name.replace(weight_name, param_name) not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
