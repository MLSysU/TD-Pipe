import torch
from collections import namedtuple
from typing import Any, Dict, List, Optional, Union

from TD_Pipe.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_src_rank,
    get_gloabl_rank,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)

_dtype = [
    torch.float32,
    torch.float,
    torch.float64,
    torch.double,
    torch.float16,
    torch.bfloat16,
    torch.half,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.short,
    torch.int32,
    torch.int,
    torch.int64,
    torch.long,
    torch.bool,
]

_dtype2id = {dtype: idx for idx, dtype in enumerate(_dtype)}
_id2dtype = {idx: dtype for idx, dtype in enumerate(_dtype)}


def barrier(ignore_pure_tp=False):
    if get_tensor_model_parallel_world_size(
    ) == 1 and get_pipeline_model_parallel_world_size() == 1:
        return
    if ignore_pure_tp and get_tensor_model_parallel_world_size(
    ) > 1 and get_pipeline_model_parallel_world_size() == 1:
        return
    torch.distributed.barrier()


def pipeline_model_parallel_send_tensor_list(tensors, tensor_count):
    dst_rank = get_pipeline_model_parallel_next_rank()
    for i in range(tensor_count):
        torch.distributed.isend(tensors[i], dst_rank)


def pipeline_model_parallel_recv_tensor_list(batch_size, hidden_size, dtype,
                                             tensor_count):
    src_rank = get_pipeline_model_parallel_prev_rank()
    shape = (batch_size, hidden_size)
    tensors = []
    for i in range(tensor_count):
        tensor = torch.empty(shape,
                             dtype=dtype,
                             device=torch.cuda.current_device())
        torch.distributed.recv(tensor, src_rank)
        tensors.append(tensor)
    return tensors


def tensor_model_parallel_all_reduce(input_):
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation is applied in-place on the input tensor.
    """
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    # All-reduce.
    torch.distributed.all_reduce(input_,
                                 group=get_tensor_model_parallel_group())
    return input_


def tensor_model_parallel_all_gather(input_, dim=-1):
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size, ) + input_size,
                                dtype=input_.dtype,
                                device=input_.device)
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=get_tensor_model_parallel_group())
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor


def tensor_model_parallel_gather(input_, dst=0, dim=-1):
    """Gather the input tensor across model parallel group.

    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    """
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    # Allocate output tensor.
    if get_tensor_model_parallel_rank() == dst:
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        gather_list = None
    if dst == 0:
        dst = get_tensor_model_parallel_src_rank()
    # Gather.
    torch.distributed.gather(input_,
                             gather_list,
                             dst=dst,
                             group=get_tensor_model_parallel_group())
    if get_tensor_model_parallel_rank() == 0:
        output_tensor = torch.cat(gather_list, dim=dim)
    else:
        output_tensor = None
    return output_tensor


def broadcast(input_, src=0):
    """Broadcast the input tensor."""
    world_size = torch.distributed.get_world_size()
    assert 0 <= src < world_size, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    # Broadcast.
    torch.distributed.broadcast(input_, src=src)
    return input_


def broadcast_object_list(obj_list, src=0):
    """Broadcast the input object list."""
    world_size = torch.distributed.get_world_size()
    assert 0 <= src < world_size, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return obj_list
    # Broadcast.
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list


TensorMetadata = namedtuple("TensorMetadata", ["dtype", "size"])


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
) -> Dict[Any, Union[torch.Tensor, Any]]:
    """Broadcast the input tensor dictionary."""
    group = torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"
    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor_dict
    rank = torch.distributed.get_rank()
    if rank == src:
        assert isinstance(
            tensor_dict,
            dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
        metadata_list = []
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                assert value.is_cuda, (
                    f"Tensor {key}: {value} is not on cuda. Currently we only "
                    f"support broadcasting tensors on cuda.")
                metadata_list.append(
                    (key, TensorMetadata(value.dtype, value.size())))
            else:
                metadata_list.append((key, value))
        torch.distributed.broadcast_object_list([metadata_list],
                                                src=src,
                                                group=group)
        for key, value in metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = tensor_dict[key]
                torch.distributed.broadcast(tensor, src=src, group=group)
    else:
        recv_metadata_list = [None]
        torch.distributed.broadcast_object_list(recv_metadata_list,
                                                src=src,
                                                group=group)
        metadata_list = recv_metadata_list[0]
        tensor_dict = {}
        async_handles = []
        for key, value in metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size,
                                     dtype=value.dtype,
                                     device="cuda")
                async_handle = torch.distributed.broadcast(tensor,
                                                           src=src,
                                                           async_op=True,
                                                           group=group)
                async_handles.append(async_handle)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        for async_handle in async_handles:
            async_handle.wait()
    return tensor_dict
