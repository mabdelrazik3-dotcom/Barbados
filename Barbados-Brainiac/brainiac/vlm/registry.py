"""Model registry: ModelSpec from configs/models.yaml, processor and model loading (transformers)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Cfg


@dataclass
class ModelSpec:
    name: str
    hf_id: str
    unsloth_id: str | None = None
    local_path: str | None = None
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    load_in_4bit: bool = False
    min_pixels: int = 3136
    max_pixels: int = 1003520
    end_marker: str = "<|im_end|>"


def get_spec(cfg: Cfg, name: str) -> ModelSpec:
    if name not in cfg.models:
        raise KeyError(f"unknown model '{name}' (configs/models.yaml)")
    fields = ModelSpec.__dataclass_fields__
    return ModelSpec(name=name, **{k: v for k, v in dict(cfg.models[name]).items() if k in fields})


def model_source(spec: ModelSpec, unsloth: bool = False) -> str:
    if spec.local_path:
        return spec.local_path
    if unsloth and spec.unsloth_id:
        return spec.unsloth_id
    return spec.hf_id


def torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32}[name]


def _new_dtype_kwarg() -> bool:
    """transformers >= 4.56 calls the argument `dtype` (earlier: `torch_dtype`)."""
    import transformers
    from packaging.version import Version

    return Version(transformers.__version__.split("+")[0]) >= Version("4.56.0")


def configure_image_processor(processor: Any, spec: ModelSpec) -> None:
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = spec.min_pixels
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = spec.max_pixels
    size = getattr(ip, "size", None)
    if isinstance(size, dict) and "shortest_edge" in size:
        ip.size = {"shortest_edge": spec.min_pixels, "longest_edge": spec.max_pixels}


def load_processor(spec: ModelSpec, cfg: Cfg, padding_side: str = "left") -> Any:
    from transformers import AutoProcessor

    kwargs = {"cache_dir": cfg.paths.hf_cache} if cfg.paths.get("hf_cache") else {}
    processor = AutoProcessor.from_pretrained(model_source(spec), **kwargs)
    configure_image_processor(processor, spec)
    processor.tokenizer.padding_side = padding_side
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def load_model(spec: ModelSpec, cfg: Cfg, adapter: str | Path | None = None, merge: bool = True,
               training: bool = False) -> Any:
    import torch

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as AutoVLM

    dtype = torch_dtype(spec.dtype)
    kwargs: dict = {"attn_implementation": spec.attn_implementation}
    kwargs["dtype" if _new_dtype_kwarg() else "torch_dtype"] = dtype
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if cfg.paths.get("hf_cache"):
        kwargs["cache_dir"] = cfg.paths.hf_cache
    if spec.load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute, bnb_4bit_use_double_quant=True
        )
    model = AutoVLM.from_pretrained(model_source(spec), **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=training)
        if merge and not training and not spec.load_in_4bit:
            model = model.merge_and_unload()
    return model


def accepts_kwarg(model: Any, name: str) -> bool:
    import inspect

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        return name in inspect.signature(base.forward).parameters
    except (TypeError, ValueError):
        return False
