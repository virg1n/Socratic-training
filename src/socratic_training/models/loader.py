from __future__ import annotations

import gc
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Literal, Optional, Tuple

from socratic_training.config import JudgeModelConfig, RedModelConfig, SocraticModelConfig
from socratic_training.memory import get_gpu_info
from pathlib import Path


Quantization = Literal["none", "4bit", "8bit"]


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device_map: Optional[Dict[str, Any]] = None


def _is_accelerate_dispatched(model: Any) -> bool:
    # When device_map="auto" is used, transformers/accelerate attach hooks and set hf_device_map.
    try:
        return bool(getattr(model, "hf_device_map", None))
    except Exception:
        return False


def _torch_cleanup() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _max_memory_map(
    safety_margin: float = 0.90,
    cpu_gib: int = 110,
    *,
    allowed_gpus: Optional[Tuple[int, ...]] = None,
) -> Optional[Dict[str, str]]:
    gpus = get_gpu_info()
    if not gpus:
        return None
    # accelerate expects GPU keys as integers (device indices), not "cuda:0" strings.
    allowed = {int(x) for x in allowed_gpus} if allowed_gpus is not None else None
    max_mem: Dict[object, str] = {}
    for g in gpus:
        idx = int(g.idx)
        if allowed is not None and idx not in allowed:
            max_mem[idx] = "0GiB"
        else:
            max_mem[idx] = f"{int(g.total_gb * safety_margin)}GiB"
    max_mem["cpu"] = f"{cpu_gib}GiB"
    return max_mem  # type: ignore[return-value]


def _resolve_torch_dtype(dtype: str):
    import torch

    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unknown torch_dtype: {dtype}")


def _build_bnb_config(quantization: Quantization):
    from transformers import BitsAndBytesConfig

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _common_lora_targets() -> Tuple[str, ...]:
    # Works for most Qwen/LLaMA-like architectures.
    return (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def _infer_lora_targets(model: Any) -> Tuple[str, ...]:
    wanted = set(_common_lora_targets())
    found = set()
    for name, _ in model.named_modules():
        suffix = name.split(".")[-1]
        if suffix in wanted:
            found.add(suffix)
    if found:
        # Keep stable order.
        return tuple(x for x in _common_lora_targets() if x in found)
    # Fall back to common targets; PEFT will error if none match, which is better than silently doing nothing.
    return _common_lora_targets()


@contextmanager
def load_socratic(cfg: SocraticModelConfig, *, for_training: bool) -> Generator[LoadedModel, None, None]:
    """
    Loads Socratic model. If `for_training` and cfg.train_lora is True, attaches LoRA adapters.
    """
    _torch_cleanup()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = _resolve_torch_dtype(cfg.torch_dtype)
    max_memory = _max_memory_map(allowed_gpus=tuple(cfg.allowed_gpus) if cfg.allowed_gpus else None)

    tokenizer = AutoTokenizer.from_pretrained(cfg.path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.path,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        max_memory=max_memory,
        trust_remote_code=True,
    )

    model = base
    # Optional "full fine-tune" checkpoint: load state_dict from adapter_dir when configured.
    # This provides a lightweight checkpoint format (weights only, no optimizer).
    if (not cfg.train_lora) and getattr(cfg, "save_mode", None) == "state_dict":
        state_path = Path(cfg.adapter_dir) / getattr(cfg, "state_dict_name", "socratic_state_dict.pt")
        if state_path.exists():
            try:
                sd = torch.load(str(state_path), map_location="cpu")
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    warnings.warn(
                        f"Loaded Socratic state_dict with missing={len(missing)} unexpected={len(unexpected)} keys."
                    )
            except Exception as e:
                warnings.warn(f"Failed to load Socratic state_dict checkpoint {state_path}: {e}")
    if cfg.train_lora:
        try:
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PEFT is required for LoRA training. Install extras: pip install -e '.[gpu]'") from e

        adapter_dir = Path(cfg.adapter_dir)
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=for_training)
        else:
            targets = _infer_lora_targets(base)
            lora = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=list(targets),
            )
            model = get_peft_model(base, lora)

        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    if for_training:
        model.train()
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    try:
        yield LoadedModel(model=model, tokenizer=tokenizer, device_map=getattr(model, "hf_device_map", None))
    finally:
        try:
            if not _is_accelerate_dispatched(model):
                model.to("cpu")
        except Exception:
            pass
        del model
        del tokenizer
        _torch_cleanup()


@contextmanager
def load_red(cfg: RedModelConfig, *, for_training: bool) -> Generator[LoadedModel, None, None]:
    """
    Loads Red model in quantized mode (default 4-bit). When training, caller is expected
    to attach LoRA adapters (or call helper in training script).
    """
    _torch_cleanup()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    qcfg = _build_bnb_config(cfg.quantization)
    max_memory = _max_memory_map(allowed_gpus=tuple(cfg.allowed_gpus) if cfg.allowed_gpus else None)

    tokenizer = AutoTokenizer.from_pretrained(cfg.path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.path,
        device_map="auto" if torch.cuda.is_available() else None,
        max_memory=max_memory,
        quantization_config=qcfg,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = base
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    except Exception:
        LoraConfig = None  # type: ignore[assignment]
        PeftModel = None  # type: ignore[assignment]
        get_peft_model = None  # type: ignore[assignment]
        prepare_model_for_kbit_training = None  # type: ignore[assignment]

    adapter_dir = Path(cfg.adapter_dir)
    if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
        if PeftModel is None:  # pragma: no cover
            raise RuntimeError("PEFT is required to load Red adapters. Install extras: pip install -e '.[gpu]'")
        if for_training and prepare_model_for_kbit_training is not None and cfg.quantization in {"4bit", "8bit"}:
            base = prepare_model_for_kbit_training(base)
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=for_training)
    elif for_training:
        if get_peft_model is None or LoraConfig is None:  # pragma: no cover
            raise RuntimeError("PEFT is required for LoRA training. Install extras: pip install -e '.[gpu]'")
        if prepare_model_for_kbit_training is not None and cfg.quantization in {"4bit", "8bit"}:
            base = prepare_model_for_kbit_training(base)
        targets = _infer_lora_targets(base)
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(targets),
        )
        model = get_peft_model(base, lora)

    if for_training:
        model.train()
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    try:
        yield LoadedModel(model=model, tokenizer=tokenizer, device_map=getattr(model, "hf_device_map", None))
    finally:
        try:
            if not _is_accelerate_dispatched(model):
                model.to("cpu")
        except Exception:
            pass
        del model
        del tokenizer
        _torch_cleanup()


@contextmanager
def load_judge(cfg: JudgeModelConfig) -> Generator[LoadedModel, None, None]:
    """
    Loads Judge model frozen for inference.

    By default loads full bf16/fp16 weights. If OOM, caller may retry using cfg.quantization_fallback.
    """
    _torch_cleanup()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = _resolve_torch_dtype(cfg.torch_dtype)
    max_memory = _max_memory_map(allowed_gpus=tuple(cfg.allowed_gpus) if cfg.allowed_gpus else None)

    tokenizer = AutoTokenizer.from_pretrained(cfg.path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    qcfg = None
    if cfg.quantization_fallback and cfg.quantization_fallback != "none":
        warnings.warn(f"Judge quantization fallback enabled: {cfg.quantization_fallback}")
        qcfg = _build_bnb_config(cfg.quantization_fallback)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.path,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        max_memory=max_memory,
        quantization_config=qcfg,
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        yield LoadedModel(model=model, tokenizer=tokenizer, device_map=getattr(model, "hf_device_map", None))
    finally:
        try:
            if not _is_accelerate_dispatched(model):
                model.to("cpu")
        except Exception:
            pass
        del model
        del tokenizer
        _torch_cleanup()
