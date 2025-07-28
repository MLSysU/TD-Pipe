from collections import deque
import enum
import time
from typing import Deque, Dict, Iterable, List, Optional, Tuple, Union

from TD_Pipe.config import CacheConfig, SchedulerConfig
from TD_Pipe.core.block_manager import AllocStatus, BlockSpaceManager
from TD_Pipe.core.policy import PolicyFactory
from TD_Pipe.logger import init_logger
from TD_Pipe.sequence import (Sequence, SequenceData, SequenceGroup,
                              SequenceGroupMetadata, SequenceStatus)
import math
import numpy as np
import csv

logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: Iterable[SequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # # Swap in and swap out should never happen at the same time.
        # assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups
        self.prefill_mode = True

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window)

        # Sequence groups in the WAITING state.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        self.swapped: Deque[SequenceGroup] = deque()

        self.ready: Deque[SequenceGroup] = deque()

        self.num_pipeline_stages = self.scheduler_config.pipeline_parallel_size

        self.max_num_total_blocks = self.block_manager.num_total_gpu_blocks
        self.max_num_running_blocks = self.max_num_total_blocks

        self.prompt_mode = True
        self.decode_count = 0
        self.prefill_mode = True
        self.peak_batch_size = 192

        self.max_length = 512
        self.intervals = 32
        self.length_intervals = np.linspace(0, self.max_length,
                                            self.intervals + 1)
        self.midpoints = [5, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        self.kv_cache_usage = {midpoint: 0 for midpoint in self.midpoints}
        self.prefill_data = {}
        self._init_prefill_data()
        self.intensity = {}
        # Reference batch size used for normalization of compute intensity
        ref_batch_size = 256
        max_intensity = ref_batch_size / self.prefill_data[ref_batch_size]
        for i in range(1, ref_batch_size):
            self.intensity[i] = (i / self.prefill_data[i]) / max_intensity

    def _init_prefill_data(self):
        with open(self.scheduler_config.profiler_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                length = int(row['Total Length'])
                prefill_time = float(row['Execute Time'])
                self.prefill_data[length] = prefill_time

    def update_kv_cache(self, input_len, predicted_len):
        for midpoint in self.midpoints:
            if midpoint <= predicted_len:
                self.kv_cache_usage[midpoint] += math.ceil(
                    (input_len + midpoint) / self.block_manager.block_size)

    def clear_kv_cache(self):
        for midpoint in self.midpoints:
            self.kv_cache_usage[midpoint] = 0

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity .
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

    def unfreeze_seq_groups(
            self, scheduled_seq_groups: Iterable[SequenceGroup]) -> None:
        for seq_group in scheduled_seq_groups:
            for seq in seq_group.get_seqs():
                seq.status = SequenceStatus.READY
            self.running.remove(seq_group)
            self.ready.append(seq_group)

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped or self.ready

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped) + len(
            self.ready)

    def update_mode(self):
        batch_size = len(self.ready) + len(self.running)
        if self.prefill_mode:
            if len(self.waiting) == 0:
                self.prefill_mode = False
                return
            else:
                max_midpoint = max(self.kv_cache_usage,
                                   key=self.kv_cache_usage.get)
                if self.kv_cache_usage[
                        max_midpoint] >= self.max_num_total_blocks * 0.9:
                    self.prefill_mode = False
                else:
                    self.prefill_mode = True
                return
        else:
            if len(self.waiting) > 0:
                if batch_size // self.num_pipeline_stages > self.peak_batch_size:
                    return
                decode_intensity = self._get_compute_intensity(
                    batch_size // self.num_pipeline_stages)

                self.clear_kv_cache()
                for seq in self.ready + self.running:
                    seq_id = next(iter(seq.seqs_dict))
                    sequence = seq.seqs_dict[seq_id]
                    input_len = sequence.get_prompt_len()
                    output_len = sequence.get_output_len()
                    predict_output_len = seq.predict_output_len - output_len
                    if predict_output_len < 0:
                        predict_output_len = 0
                    self.update_kv_cache(input_len + output_len,
                                         predict_output_len)
                max_predict_time = 0
                prefill_time = 0
                decode_batch = batch_size
                for seq_group in self.waiting:
                    max_midpoint = max(self.kv_cache_usage,
                                       key=self.kv_cache_usage.get)
                    if self.kv_cache_usage[
                            max_midpoint] < self.max_num_total_blocks:
                        self.update_kv_cache(seq_group.get_seqs()[0].get_len(),
                                             seq_group.predict_output_len)
                        predict_time = self._get_prefill_time(
                            seq_group.get_seqs()[0].get_len())
                        prefill_time += predict_time / self.num_pipeline_stages
                        decode_batch += 1
                        if max_predict_time < predict_time:
                            max_predict_time = predict_time
                    else:
                        break
                prefill_time = prefill_time + max_predict_time / self.num_pipeline_stages * (
                    self.num_pipeline_stages - 1)
                decode_time = self._get_decode_batch_time(
                    decode_batch // self.num_pipeline_stages)
                bubble_time = (max_predict_time / self.num_pipeline_stages -
                               decode_time) * (self.num_pipeline_stages - 1)
                prefill_intensity = (prefill_time - bubble_time) / prefill_time

                if decode_intensity < prefill_intensity:
                    self.prefill_mode = True
                    self.clear_kv_cache()
                    for seq in self.ready + self.running:
                        seq_id = next(iter(seq.seqs_dict))
                        sequence = seq.seqs_dict[seq_id]
                        input_len = sequence.get_prompt_len()
                        output_len = sequence.get_output_len()
                        predict_output_len = seq.predict_output_len - output_len
                        if predict_output_len < 0:
                            predict_output_len = 0
                        self.update_kv_cache(input_len + output_len,
                                             predict_output_len)
            return

    def _get_compute_intensity(self, batch_size) -> float:
        if batch_size >= self.peak_batch_size:
            return 1
        return self.intensity[batch_size]

    def _get_prefill_time(self, seq_len) -> int:
        # 2048 is the maximum profiled sequence length; use linear extrapolation beyond this
        max_profile_len = 2048
        max_profile_time = self.prefill_data.get(max_profile_len, None)
        return int(
            self.prefill_data.get(seq_len, max_profile_time * seq_len /
                                  max_profile_len))

    def _get_decode_batch_time(self, batch_size) -> int:
        return self._get_prefill_time(batch_size)

    def _schedule(self) -> SchedulerOutputs:
        '''
        Sched:
            Three states: waiting, running, ready
            Beam search is not considered; each seq_group contains only one seq
        '''
        assert not self.swapped, "swapped is currently reserved"

        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.monotonic()

        num_running_seqs = 0
        num_running_blocks = 0

        ignored_seq_groups: List[SequenceGroup] = []
        scheduled: List[SequenceGroup] = []
        seq_lens: List[int] = []

        if self.decode_count % self.num_pipeline_stages == 0:
            self.update_mode()
        if self.prefill_mode:
            while self.waiting:
                seq_group = self.waiting[0]
                waiting_seqs = seq_group.get_seqs(
                    status=SequenceStatus.WAITING)
                assert len(waiting_seqs) == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")

                num_prompt_tokens = waiting_seqs[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.popleft()
                    continue

                can_allocate = self.block_manager.can_allocate(seq_group)
                if can_allocate == AllocStatus.LATER:
                    break
                elif can_allocate == AllocStatus.NEVER:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds the capacity of block_manager")
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.popleft()
                    continue

                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens = seq_lens + [num_prompt_tokens]
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)

                seq_group = self.waiting.popleft()
                self._allocate(seq_group)
                self.running.append(seq_group)

                scheduled.append(seq_group)
                seq_lens = new_seq_lens
                if num_batched_tokens >= self.peak_batch_size:
                    break

            for i in range(len(scheduled)):
                self.update_kv_cache(seq_lens[i],
                                     scheduled[i].predict_output_len)
            if scheduled or ignored_seq_groups:
                self.decode_count = 0
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=len(seq_lens) *
                    max(seq_lens) if seq_lens else 0,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs
            else:
                self.prefill_mode = False
        if self.prefill_mode == False:
            self.ready = self.policy.sort_by_priority(now, self.ready)
            total_requests = len(self.running) + len(self.ready)
            running: Deque[SequenceGroup] = deque()
            while self.ready:
                seq_group = self.ready[0]

                # Assume 1 SeqGroup 1 Seq, append require +1
                num_free_blocks = self.block_manager.get_num_free_gpu_blocks()
                if num_free_blocks < 1 or num_running_blocks + 1 > self.max_num_running_blocks:
                    if len(self.ready) > 1:
                        victim_seq_group = self.ready.pop()
                        self._preempt_by_recompute(victim_seq_group)
                        total_requests -= 1
                    else:
                        break

                self.ready.popleft()

                self._append_slot(seq_group, blocks_to_copy)
                for seq in seq_group.get_seqs(status=SequenceStatus.READY):
                    seq.status = SequenceStatus.RUNNING
                running.append(seq_group)
                scheduled.append(seq_group)
                num_running_seqs = num_running_seqs + 1
                if num_running_seqs > 1 and num_running_seqs >= self._get_current_batch_requests(
                        total_requests):
                    if len(self.ready) > 0 and len(
                            self.ready) <= self.num_pipeline_stages:
                        continue
                    break

            self.running.extend(running)
            # Assume 1 SeqGroup 1 Seq
            self.decode_count = self.decode_count + 1
            num_batched_tokens = len(scheduled)
            scheduler_outputs = SchedulerOutputs(
                scheduled_seq_groups=scheduled,
                prompt_run=False,
                num_batched_tokens=num_batched_tokens,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                ignored_seq_groups=[],
            )
            return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()
        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def _get_current_batch_requests(self, total_requests):
        return total_requests // self.num_pipeline_stages + (
            self.decode_count % self.num_pipeline_stages <
            total_requests % self.num_pipeline_stages)

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.ready = deque([
            seq_group for seq_group in self.ready
            if not seq_group.is_finished()
        ])

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.READY):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.READY)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.appendleft(seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(self,
                  seq_group: SequenceGroup,
                  blocks_to_swap_out: Dict[int, int],
                  new_status: SequenceStatus = SequenceStatus.SWAPPED) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = new_status
