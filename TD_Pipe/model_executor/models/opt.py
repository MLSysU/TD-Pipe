# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/opt/modeling_opt.py
# Copyright 2023 The vLLM team.
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights
# reserved.
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
"""Inference-only OPT model compatible with HuggingFace weights."""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import OPTConfig
from TD_Pipe.config import ParallelConfig

from TD_Pipe.model_executor.input_metadata import InputMetadata
from TD_Pipe.model_executor.layers.activation import get_act_fn
from TD_Pipe.model_executor.layers.attention import Attention
from TD_Pipe.model_executor.layers.linear import (ColumnParallelLinear,
                                                  LinearMethodBase,
                                                  QKVParallelLinear,
                                                  ReplicatedLinear,
                                                  RowParallelLinear)
from TD_Pipe.model_executor.layers.sampler import Sampler
from TD_Pipe.model_executor.layers.projection import Projection
from TD_Pipe.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from TD_Pipe.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from TD_Pipe.model_executor.sampling_metadata import SamplingMetadata
from TD_Pipe.model_executor.weight_utils import (default_weight_loader,
                                                 hf_model_weights_iterator)
from TD_Pipe.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_backend: str,
        bias: bool = True,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        total_num_heads = num_heads
        assert num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = embed_dim // total_num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            total_num_heads,
            bias=bias,
            linear_method=linear_method,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            linear_method=linear_method,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              attention_backend=attention_backend,
                              scale=self.scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        key_cache, value_cache = kv_cache
        attn_output = self.attn(q, k, v, key_cache, value_cache,
                                input_metadata)
        output, _ = self.out_proj(attn_output)
        return output


class OPTDecoderLayer(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        attention_backend: str,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
            attention_backend=attention_backend,
            linear_method=linear_method,
        )
        self.do_layer_norm_before = config.do_layer_norm_before

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            linear_method=linear_method,
        )
        quant_config = getattr(linear_method, "quant_config", None)
        self.activation_fn = get_act_fn(config.activation_function,
                                        quant_config, config.ffn_dim)
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            linear_method=linear_method,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       kv_cache=kv_cache,
                                       input_metadata=input_metadata)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        parallel_config: ParallelConfig,
        attention_backend: str,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        self.parallel_config = parallel_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
        )
        # Positional embeddings are replicated (not sharded).
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size)

        if self.parallel_config.is_last:
            # Project out & in will be replicated if they exist.
            if config.word_embed_proj_dim != config.hidden_size:
                self.project_out = ReplicatedLinear(
                    config.hidden_size,
                    config.word_embed_proj_dim,
                    bias=False,
                    linear_method=linear_method)
            else:
                self.project_out = None

        if self.parallel_config.is_first:
            if config.word_embed_proj_dim != config.hidden_size:
                self.project_in = ReplicatedLinear(config.word_embed_proj_dim,
                                                   config.hidden_size,
                                                   bias=False,
                                                   linear_method=linear_method)
            else:
                self.project_in = None

        if self.parallel_config.is_last:
            # Note that the only purpose of `config._remove_final_layer_norm` is to
            # keep backward compatibility with checkpoints that have been fine-tuned
            # before transformers v4.20.1
            # see https://github.com/facebookresearch/metaseq/pull/164
            if config.do_layer_norm_before and not config._remove_final_layer_norm:
                self.final_layer_norm = nn.LayerNorm(
                    config.hidden_size,
                    elementwise_affine=config.layer_norm_elementwise_affine)
            else:
                self.final_layer_norm = None

        self.layers = nn.ModuleList()
        for i in range(self.parallel_config.start,
                       self.parallel_config.end + 1):
            self.layers.add_module(
                f"{i}",
                OPTDecoderLayer(config, attention_backend, linear_method))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        if self.parallel_config.is_first:
            inputs_embeds = self.embed_tokens(input_ids)
            pos_embeds = self.embed_positions(positions)
            if self.project_in is not None:
                inputs_embeds, _ = self.project_in(inputs_embeds)
            hidden_states = inputs_embeds + pos_embeds
        else:
            hidden_states = input_ids

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                kv_caches[i],
                input_metadata,
            )

        if self.parallel_config.is_last:
            if self.final_layer_norm is not None:
                hidden_states = self.final_layer_norm(hidden_states)
            if self.project_out is not None:
                hidden_states, _ = self.project_out(hidden_states)
        return hidden_states


class OPTModel(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        parallel_config: ParallelConfig,
        attention_backend: str,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.decoder = OPTDecoder(config, parallel_config, attention_backend,
                                  linear_method)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        return self.decoder(input_ids, positions, kv_caches, input_metadata)


class OPTForCausalLM(nn.Module):

    def __init__(
        self,
        config,
        parallel_config: ParallelConfig,
        attention_backend: str,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        self.linear_method = linear_method
        self.model = OPTModel(config, parallel_config, attention_backend,
                              linear_method)
        self.lm_head_weight = self.model.decoder.embed_tokens.weight
        self.projection = Projection(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: List[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids[0], positions, kv_caches,
                                   input_metadata)
        return [hidden_states]

    def get_tensor_count(self):
        return 1

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
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "lm_head.weight" in name:
                continue
            if name.startswith("decoder."):
                name = "model." + name

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
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
