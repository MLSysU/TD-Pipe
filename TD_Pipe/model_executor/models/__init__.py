import importlib
from typing import List, Optional, Type

import torch.nn as nn

from TD_Pipe.logger import init_logger
from TD_Pipe.utils import is_hip

logger = init_logger(__name__)

_MODEL_CLASSES_SUPPORT_PIPELINE_PARALLEL = [
    "LlamaForCausalLM",
    "OPTForCausalLM",
    "MixtralForCausalLM",
    "Qwen2ForCausalLM",
]

# Architecture -> (module, class).
_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    # For decapoda-research/llama-*
    "LLaMAForCausalLM": ("llama", "LlamaForCausalLM"),
    "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
    # transformers's mpt class has lower case
    "OPTForCausalLM": ("opt", "OPTForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
}

# Models not supported by ROCm.
_ROCM_UNSUPPORTED_MODELS = []

# Models partially supported by ROCm.
# Architecture -> Reason.
_ROCM_PARTIALLY_SUPPORTED_MODELS = {
    "MistralForCausalLM":
    "Sliding window attention is not yet supported in ROCm's flash attention",
    "MixtralForCausalLM":
    "Sliding window attention is not yet supported in ROCm's flash attention",
}


class ModelRegistry:

    @staticmethod
    def load_model_cls(model_arch: str,
                       is_pipeline: bool = False) -> Optional[Type[nn.Module]]:
        if model_arch not in _MODELS:
            return None
        if is_pipeline and model_arch not in _MODEL_CLASSES_SUPPORT_PIPELINE_PARALLEL:
            logger.warning(
                f"Model architecture {model_arch} does not support pipeline parallelism."
            )
            return None

        if is_hip():
            if model_arch in _ROCM_UNSUPPORTED_MODELS:
                raise ValueError(
                    f"Model architecture {model_arch} is not supported by "
                    "ROCm for now.")
            if model_arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
                logger.warning(
                    f"Model architecture {model_arch} is partially supported "
                    "by ROCm: " + _ROCM_PARTIALLY_SUPPORTED_MODELS[model_arch])

        module_name, model_cls_name = _MODELS[model_arch]
        module = importlib.import_module(
            f"TD_Pipe.model_executor.models.{module_name}")
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())


__all__ = [
    "ModelRegistry",
]
