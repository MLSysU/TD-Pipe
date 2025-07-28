"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from TD_Pipe.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_gather)
from TD_Pipe.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from TD_Pipe.sampling_params import SamplingParams, SamplingType
from TD_Pipe.sequence import (PromptLogprobs, SampleLogprobs, SamplerOutput,
                              SequenceData, SequenceGroupOutput,
                              SequenceOutput)


class Projection(nn.Module):
    '''
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    '''

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        embedding: torch.Tensor,  # LM head
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        is_prompt: bool,
        embedding_bias: Optional[torch.Tensor] = None,
    ):
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, sampling_metadata,
                                             is_prompt)

        # Get the logits for the next tokens.
        logits = _get_logits(hidden_states, embedding, embedding_bias,
                             self.vocab_size)
        return logits


def _get_logits(hidden_states: torch.Tensor, embedding: torch.Tensor,
                embedding_bias: Optional[torch.Tensor],
                vocab_size: int) -> Optional[torch.Tensor]:
    # Get the logits for the next tokens.
    logits = torch.matmul(hidden_states, embedding.t())
    if embedding_bias is not None:
        logits += embedding_bias
    logits = tensor_model_parallel_gather(logits)
    # Remove paddings in vocab (if any).
    if logits is not None:
        logits = logits[:, :vocab_size]
    return logits


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    is_prompt: bool,
) -> torch.Tensor:
    if is_prompt:
        return hidden_states.index_select(
            0, sampling_metadata.selected_token_indices)
    else:
        return hidden_states
