from typing import Dict, List, Optional, Tuple, Union

import torch

from TD_Pipe.config import ModelConfig, ParallelConfig, SchedulerConfig
from TD_Pipe.logger import init_logger
from TD_Pipe.model_executor import get_model, InputMetadata, SamplingMetadata
from TD_Pipe.sampling_params import SamplingParams, SamplingType
from TD_Pipe.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from TD_Pipe.utils import in_wsl

from TD_Pipe.model_executor.parallel_utils.parallel_state import (
    is_pipeline_model_parallel_first_rank,
    is_pipeline_model_parallel_last_rank, is_tensor_model_parallel_first_rank)
from TD_Pipe.model_executor.parallel_utils.communication_op import (
    pipeline_model_parallel_send_tensor_list,
    pipeline_model_parallel_recv_tensor_list)

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]
_PAD_SLOT_ID = -1


class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        is_driver_worker: bool = False,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.is_driver_worker = is_driver_worker

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.model = None
        self.block_size = None  # Set after initial profiling.
        # cache in_wsl result
        self.in_wsl = in_wsl()

        # Set if the backend is flashinfer.
        self.flashinfer_workspace_buffer: torch.Tensor

    def load_model(self) -> None:
        # self.model_config.load_format = "dummy"
        self.model = get_model(self.model_config, self.parallel_config)
        self.tensor_count = self.model.get_tensor_count()

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(prompt_len)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                start_idx = max(0, prompt_len - self.sliding_window)
            for i in range(prompt_len):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        max_seq_len = max(prompt_lens)
        prompt_lens_tensor = torch.tensor(prompt_lens,
                                          dtype=torch.long,
                                          device="cuda")
        seq_start_loc = torch.zeros(prompt_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device="cuda")
        torch.cumsum(prompt_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])
        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device="cuda")
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device="cuda")
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device="cuda")

        input_metadata = InputMetadata(
            is_prompt=True,
            slot_mapping=slot_mapping,
            prompt_lens=prompt_lens,
            max_context_len=None,
            context_lens=None,
            block_tables=None,
            max_seq_len=max_seq_len,
            seq_start_loc=seq_start_loc,
        )
        return input_tokens, input_positions, input_metadata, prompt_lens

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []
        seq_lens: List[int] = []

        paged_kv_indices: List[int] = []
        paged_kv_indptr: List[int] = [0]
        paged_kv_last_page_len: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                seq_len = seq_data.get_len()
                seq_lens.append(seq_len)
                position = seq_len - 1
                input_positions.append(position)

                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                context_lens.append(context_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

                paged_kv_indices.extend(block_table)
                paged_kv_indptr.append(paged_kv_indptr[-1] + len(block_table))
                last_page_len = seq_data.get_len() % self.block_size
                if last_page_len == 0:
                    last_page_len = self.block_size
                paged_kv_last_page_len.append(last_page_len)

        max_context_len = max(context_lens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device="cuda")
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device="cuda")
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device="cuda")
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device="cuda")

        block_tables = _make_tensor_with_pad(
            block_tables,
            max_len=(max_context_len + self.block_size - 1) // self.block_size,
            pad=0,
            dtype=torch.int,
            device="cuda",
        )
        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device="cuda")
        if self.model_config.attention_backend == "flashInfer":
            if not hasattr(self, "flashinfer_workspace_buffer"):
                # Allocate 16MB workspace buffer
                # Follow the example of flashinfer: https://docs.flashinfer.ai/api/python/decode.html
                self.flashinfer_workspace_buffer = torch.empty(
                    16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            paged_kv_indptr = torch.tensor(paged_kv_indptr,
                                           dtype=torch.int,
                                           device="cuda")
            paged_kv_indices = torch.tensor(paged_kv_indices,
                                            dtype=torch.int,
                                            device="cuda")
            paged_kv_last_page_len = torch.tensor(paged_kv_last_page_len,
                                                  dtype=torch.int,
                                                  device="cuda")
        else:
            self.flashinfer_workspace_buffer = None
            paged_kv_indptr = None
            paged_kv_indices = None
            paged_kv_last_page_len = None

        input_metadata = InputMetadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            max_context_len=max_context_len,
            context_lens=context_lens,
            block_tables=block_tables,
            max_seq_len=None,
            seq_start_loc=None,
            decode_seq_lens=seq_lens_tensor,
            workspace_buffer=self.flashinfer_workspace_buffer,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.model_config.get_num_attention_heads(
                self.parallel_config),
            num_kv_heads=self.model_config.get_num_kv_heads(
                self.parallel_config),
            head_dim=self.model_config.get_head_size(),
            page_size=self.block_size,
            use_flashinfer=self.model_config.attention_backend == "flashInfer",
        )
        return input_tokens, input_positions, input_metadata

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
    ) -> SamplingMetadata:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0

        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            if seq_group_metadata.is_prompt:
                assert len(seq_ids) == 1
                prompt_len = prompt_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token positions do not need sample, skip
                    categorized_sample_indices_start_idx += prompt_len - 1

                categorized_sample_indices[
                    sampling_params.sampling_type].append(
                        categorized_sample_indices_start_idx)
                categorized_sample_indices_start_idx += 1

                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + prompt_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              prompt_len - 1)
                selected_token_start_idx += prompt_len
            else:
                num_seqs = len(seq_ids)
                selected_token_indices.extend(
                    range(selected_token_start_idx,
                          selected_token_start_idx + num_seqs))
                selected_token_start_idx += num_seqs

                categorized_sample_indices[
                    sampling_params.sampling_type].extend(
                        range(categorized_sample_indices_start_idx,
                              categorized_sample_indices_start_idx + num_seqs))
                categorized_sample_indices_start_idx += num_seqs

        selected_token_indices = _async_h2d(selected_token_indices,
                                            dtype=torch.long,
                                            pin_memory=not self.in_wsl)
        categorized_sample_indices = {
            t: _async_h2d(seq_ids, dtype=torch.int, pin_memory=not self.in_wsl)
            for t, seq_ids in categorized_sample_indices.items()
        }

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
        )
        return sampling_metadata

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, SamplingMetadata,
               int]:
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            input_tokens, input_positions, input_metadata, prompt_lens = self._prepare_prompt(
                seq_group_metadata_list)
        else:
            input_tokens, input_positions, input_metadata = self._prepare_decode(
                seq_group_metadata_list)
            prompt_lens = []
        sampling_metadata = self._prepare_sample(seq_group_metadata_list,
                                                 prompt_lens)
        return input_tokens, input_positions, input_metadata, sampling_metadata, (
            max(prompt_lens) if prompt_lens else 1)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[SamplerOutput]:
        input_tokens, input_positions, input_metadata, sampling_metadata, length = (
            self.prepare_input_tensors(seq_group_metadata_list))
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Execute the model.
        model_executable = self.model
        # Pipeline execution.
        if is_pipeline_model_parallel_first_rank():
            hidden_states = [input_tokens]
        else:
            hidden_states = pipeline_model_parallel_recv_tensor_list(
                len(input_tokens), self.model.config.hidden_size,
                self.model.config.torch_dtype, self.tensor_count)

        hidden_states = model_executable(
            input_ids=hidden_states,
            positions=input_positions,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
        )
        if is_pipeline_model_parallel_last_rank():
            # Sample the next token.
            logits = self.model.project(
                hidden_states=hidden_states[0],
                sampling_metadata=sampling_metadata,
                is_prompt=is_prompt,
            )
            if is_tensor_model_parallel_first_rank():
                output = self.model.sample(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
                return output
        else:
            pipeline_model_parallel_send_tensor_list(hidden_states,
                                                     self.tensor_count)

        return None

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model_config.get_vocab_size()
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [(None, None)] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize()
        return


def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return x + [pad] * (max_len - len(x))


def _make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Union[str, torch.device] = "cuda",
    pin_memory: bool = False,
) -> torch.Tensor:
    padded_x = [_pad_to_max(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x,
                        dtype=dtype,
                        device=device,
                        pin_memory=pin_memory and str(device) == "cpu")


def _async_h2d(data: list, dtype, pin_memory):
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory)
    return t.to(device="cuda", non_blocking=True)
