from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import PipelineConfig


@dataclass(slots=True)
class LoadedRuntime:
    role: str
    model: Any
    tokenizer: Any
    primary_device: str


class StagedModelManager:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.active: LoadedRuntime | None = None

    def unload(self) -> None:
        if self.active is not None:
            del self.active.model
            del self.active.tokenizer
            self.active = None
        self._cleanup_memory()

    @contextmanager
    def use_socratic(self, mode: str = "inference") -> Iterator[LoadedRuntime]:
        runtime = self._load_socratic(mode)
        try:
            yield runtime
        finally:
            self.unload()

    @contextmanager
    def use_red(self, mode: str = "generation") -> Iterator[LoadedRuntime]:
        runtime = self._load_red(mode)
        try:
            yield runtime
        finally:
            self.unload()

    @contextmanager
    def use_judge(self, allow_fallback: bool = False) -> Iterator[LoadedRuntime]:
        runtime = self._load_judge(allow_fallback=allow_fallback)
        try:
            yield runtime
        finally:
            self.unload()

    def _load_socratic(self, mode: str) -> LoadedRuntime:
        self.unload()
        torch = _require_torch()
        transformers = _require_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.socratic.model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.socratic.model_path,
            torch_dtype=_dtype_for_name(torch, self.config.socratic.dtype),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.to(torch.device(self.config.socratic.train_device))
        model.config.use_cache = mode != "train"
        if mode == "train" and self.config.socratic.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        self.active = LoadedRuntime("socratic", model, tokenizer, self.config.socratic.train_device)
        return self.active

    def _load_red(self, mode: str) -> LoadedRuntime:
        self.unload()
        torch = _require_torch()
        transformers = _require_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.red.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "left"
        quantization_config = _build_quantization_config(transformers, self.config.red.quantization, compute_dtype=_dtype_for_name(torch, self.config.red.dtype))
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.red.model_name,
            torch_dtype=_dtype_for_name(torch, self.config.red.dtype),
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map={"": self.config.red.train_device},
            trust_remote_code=True,
        )
        model.config.use_cache = mode != "train"
        if mode == "train":
            peft = _require_peft()
            model = peft.prepare_model_for_kbit_training(model)
            model.gradient_checkpointing_enable()
        self.active = LoadedRuntime("red", model, tokenizer, self.config.red.train_device)
        return self.active

    def _load_judge(self, allow_fallback: bool) -> LoadedRuntime:
        self.unload()
        torch = _require_torch()
        transformers = _require_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.judge.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "left"

        kwargs: dict[str, Any] = {
            "torch_dtype": _dtype_for_name(torch, self.config.judge.dtype),
            "low_cpu_mem_usage": True,
            "device_map": "balanced",
            "max_memory": self._max_memory_map(),
            "offload_folder": str(Path(self.config.paths.offload_dir) / "judge"),
            "trust_remote_code": True,
        }
        if allow_fallback and self.config.judge.quantization_fallback:
            kwargs["quantization_config"] = _build_quantization_config(transformers, self.config.judge.quantization_fallback, compute_dtype=_dtype_for_name(torch, self.config.judge.dtype))
        model = transformers.AutoModelForCausalLM.from_pretrained(self.config.judge.model_name, **kwargs)
        self.active = LoadedRuntime("judge", model, tokenizer, _primary_device(model))
        return self.active

    def _cleanup_memory(self) -> None:
        gc.collect()
        try:
            torch = _require_torch()
        except RuntimeError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def _max_memory_map(self) -> dict[int | str, str]:
        per_gpu = f"{int(self.config.hardware.gpu_vram_gb * self.config.hardware.gpu_utilization_target)}GiB"
        mapping: dict[int | str, str] = {index: per_gpu for index in range(self.config.hardware.gpu_count)}
        mapping["cpu"] = f"{int(self.config.hardware.cpu_offload_budget_gb)}GiB"
        return mapping


def generate_texts(
    runtime: LoadedRuntime,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
) -> list[str]:
    torch = _require_torch()
    tokenizer = runtime.tokenizer
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    encoded = {name: tensor.to(runtime.primary_device) for name, tensor in encoded.items()}
    do_sample = temperature > 0.0
    outputs = runtime.model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    attention_mask = encoded["attention_mask"]
    prompt_lengths = attention_mask.sum(dim=1).tolist()
    repeated_lengths: list[int] = []
    for length in prompt_lengths:
        repeated_lengths.extend([int(length)] * num_return_sequences)
    generations: list[str] = []
    for output, prompt_length in zip(outputs, repeated_lengths, strict=False):
        generated_tokens = output[prompt_length:]
        generations.append(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return generations


def _primary_device(model: Any) -> str:
    if hasattr(model, "hf_device_map"):
        return str(next(iter(model.hf_device_map.values())))
    return str(next(model.parameters()).device)


def _build_quantization_config(transformers: Any, mode: str, compute_dtype: Any) -> Any:
    mode = mode.lower()
    if mode == "4bit":
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if mode == "8bit":
        return transformers.BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization mode: {mode}")


def _dtype_for_name(torch_module: Any, name: str) -> Any:
    normalized = name.lower()
    if normalized == "bfloat16":
        return torch_module.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch_module.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for staged model loading.") from error
    return torch


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as error:
        raise RuntimeError("transformers is required for staged model loading.") from error
    return transformers


def _require_peft() -> Any:
    try:
        import peft
    except ImportError as error:
        raise RuntimeError("peft is required for LoRA training.") from error
    return peft
