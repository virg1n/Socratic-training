from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

from socratic_training.config import AppConfig
from socratic_training.curriculum import Curriculum


@dataclass(frozen=True)
class GpuInfo:
    idx: int
    total_gb: float


def _bytes_per_param(quantization: Literal["none", "4bit", "8bit"], dtype_bytes: int) -> float:
    """
    Conservative approximation of *weight storage* bytes per parameter.

    - "none": standard dense weights in dtype_bytes
    - "8bit": ~1 byte + scales overhead
    - "4bit": ~0.5 byte + scales/zeros overhead
    """
    if quantization == "none":
        return float(dtype_bytes)
    if quantization == "8bit":
        return 1.15  # overhead for scales/zeros
    if quantization == "4bit":
        return 0.62  # overhead for scales/zeros
    raise ValueError(f"Unknown quantization: {quantization}")


def _dtype_bytes(dtype: str) -> int:
    if dtype == "float16":
        return 2
    if dtype == "bfloat16":
        return 2
    if dtype == "float32":
        return 4
    raise ValueError(f"Unknown dtype: {dtype}")


def _infer_params_b(model_path: str, role_default_b: float) -> float:
    """
    Best-effort parameter-count inference (billions).

    Important: do NOT regex-scan the full filesystem path because directory names may contain
    tokens like "24b" (e.g. course names) that are unrelated to model size. We prefer:
      1) local config.json estimate
      2) regex on *short* tail name (repo name / last path components)
      3) role default
    """
    import re

    def _try_regex(s: str) -> Optional[float]:
        m = re.search(r"(\d+(?:\.\d+)?)\s*B\b", s, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def _estimate_from_config(cfg: dict) -> Optional[float]:
        # Generic-ish transformer param estimate from config.json (no weights needed).
        def _get_int(*keys: str) -> Optional[int]:
            for k in keys:
                v = cfg.get(k)
                if isinstance(v, int):
                    return v
            return None

        hidden = _get_int("hidden_size", "n_embd", "d_model")
        layers = _get_int("num_hidden_layers", "n_layer", "num_layers")
        intermediate = _get_int("intermediate_size", "n_inner", "ffn_dim", "d_ff")
        vocab = _get_int("vocab_size")
        if not (hidden and layers and intermediate and vocab):
            return None

        tie = cfg.get("tie_word_embeddings", True)
        if not isinstance(tie, bool):
            tie = True

        embed = vocab * hidden
        lm_head = 0 if tie else vocab * hidden
        per_layer_attn = 4 * hidden * hidden
        per_layer_mlp = 3 * hidden * intermediate
        norms = 2 * layers * hidden + hidden
        total = embed + lm_head + layers * (per_layer_attn + per_layer_mlp) + norms
        return float(total) / 1e9

    p = Path(model_path)
    if p.exists() and p.is_dir():
        cfg_path = p / "config.json"
        if cfg_path.exists():
            try:
                obj = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    est = _estimate_from_config(obj)
                    if est is not None and est > 0:
                        return est
            except Exception:
                pass

        parts = list(p.parts)
        tail = "/".join(parts[-2:]) if len(parts) >= 2 else p.name
        est = _try_regex(tail)
        if est is not None:
            return est
        return role_default_b

    # HF repo id: only look at the repo name portion.
    name = model_path.split("/")[-1]
    est = _try_regex(name)
    if est is not None:
        return est
    return role_default_b


def estimate_weights_gb(*, params_b: float, quantization: Literal["none", "4bit", "8bit"], dtype: str) -> float:
    bytes_per = _bytes_per_param(quantization, _dtype_bytes(dtype))
    return (params_b * 1e9 * bytes_per) / 1e9


def estimate_kv_cache_gb(
    *,
    params_b: float,
    dtype: str,
    batch_size: int,
    seq_len: int,
) -> float:
    """
    Rough KV cache estimate (per *whole model*, not per-GPU), based on a rule-of-thumb
    that scales with parameter count and context length.

    This intentionally overestimates slightly to provide a safety buffer.
    """
    dtype_b = _dtype_bytes(dtype)
    # Empirical-ish constant: ~0.035–0.04 GB per (1B params, batch=1, seq=1024, fp16/bf16).
    gb_per_b_per_1024 = 0.04 * (dtype_b / 2.0)
    return params_b * gb_per_b_per_1024 * batch_size * (seq_len / 1024.0)


def estimate_inference_gb_total(
    *,
    params_b: float,
    quantization: Literal["none", "4bit", "8bit"],
    dtype: str,
    batch_size: int,
    seq_len: int,
) -> float:
    w = estimate_weights_gb(params_b=params_b, quantization=quantization, dtype=dtype)
    kv = estimate_kv_cache_gb(params_b=params_b, dtype=dtype, batch_size=batch_size, seq_len=seq_len)
    # Overhead: activations, allocator fragmentation, temporary buffers.
    overhead = 0.20 * w + 0.10 * kv
    return w + kv + overhead


def estimate_socratic_train_gb_per_gpu(
    *,
    params_b: float,
    dtype: str,
    train_lora: bool,
    micro_batch_size: int,
    seq_len: int,
) -> float:
    """
    Very conservative *per-GPU* estimate for data-parallel training.

    If you do full fine-tuning with Adam, memory can be near:
      weights + grads + Adam states (+ optional master weights).
    For LoRA training, only a small fraction is trainable; we approximate that.
    """
    dtype_b = _dtype_bytes(dtype)
    weights = estimate_weights_gb(params_b=params_b, quantization="none", dtype=dtype)

    if train_lora:
        # Crude approximation: 0.8% trainable params (varies by target modules + rank).
        trainable_fraction = 0.008
        trainable_params_b = params_b * trainable_fraction
        # LoRA weights in bf16/fp16, grads same, Adam states fp32.
        lora_train = (trainable_params_b * 1e9 * (dtype_b + dtype_b + 8.0)) / 1e9
        optim_and_grads = lora_train
    else:
        # Full fine-tune: bf16 weights + bf16 grads + fp32 Adam m/v.
        optim_and_grads = (params_b * 1e9 * (dtype_b + dtype_b + 8.0)) / 1e9

    # Activations scale with batch and seq; estimate as a fraction of weights.
    # With checkpointing, activations drop substantially; still keep conservative.
    act = weights * 0.35 * max(1.0, micro_batch_size) * (seq_len / 1024.0)

    overhead = 0.15 * weights
    return weights + optim_and_grads + act + overhead


def estimate_qlora_train_gb_per_gpu(
    *,
    params_b: float,
    base_quantization: Literal["none", "4bit", "8bit"],
    compute_dtype: str,
    micro_batch_size: int,
    seq_len: int,
) -> float:
    """
    Conservative estimate for QLoRA-style training:
    - Base weights are quantized and frozen.
    - Trainable params are adapters only (assume ~0.8%).
    """
    dtype_b = _dtype_bytes(compute_dtype)
    base_weights = estimate_weights_gb(params_b=params_b, quantization=base_quantization, dtype=compute_dtype)

    trainable_fraction = 0.008
    trainable_params_b = params_b * trainable_fraction
    adapter_train = (trainable_params_b * 1e9 * (dtype_b + dtype_b + 8.0)) / 1e9

    act = base_weights * 0.35 * max(1.0, micro_batch_size) * (seq_len / 1024.0)
    overhead = 0.20 * base_weights
    return base_weights + adapter_train + act + overhead


def get_gpu_info() -> Tuple[GpuInfo, ...]:
    try:
        import torch
    except Exception:
        return ()
    if not torch.cuda.is_available():
        return ()
    gpus = []
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        gpus.append(GpuInfo(idx=i, total_gb=prop.total_memory / (1024**3)))
    return tuple(gpus)


@dataclass
class PreflightReport:
    ok: bool
    warnings: Tuple[str, ...]
    estimates_gb: Dict[str, float]
    suggested_updates: Dict[str, object]


def _fits_per_gpu(total_gb: float, gpus: Tuple[GpuInfo, ...], safety_margin: float) -> bool:
    if not gpus:
        return True
    max_per_gpu = max(g.total_gb for g in gpus) * safety_margin
    # If the model is sharded over multiple GPUs, total_gb is spread out.
    # Assume we can shard evenly across all GPUs unless the model fits on 1 GPU.
    if total_gb <= max_per_gpu:
        return True
    shard_cap = sum(g.total_gb for g in gpus) * safety_margin
    return total_gb <= shard_cap


def preflight_and_autoscale(cfg: AppConfig, *, curriculum: Curriculum, dry_run: bool) -> PreflightReport:
    """
    Estimates memory for each scenario and (optionally) mutates cfg to safer defaults.
    """
    gpus = get_gpu_info()
    warnings_out = []
    estimates: Dict[str, float] = {}
    updates: Dict[str, object] = {}

    # Defaults based on roles (can be overridden later with explicit params).
    socratic_params_b = float(cfg.models.socratic.params_b or _infer_params_b(cfg.models.socratic.path, role_default_b=4.0))
    red_params_b = float(cfg.models.red.params_b or _infer_params_b(cfg.models.red.path, role_default_b=32.0))
    judge_params_b = float(cfg.models.judge.params_b or _infer_params_b(cfg.models.judge.path, role_default_b=32.0))
    estimates["socratic_params_b"] = socratic_params_b
    estimates["red_params_b"] = red_params_b
    estimates["judge_params_b"] = judge_params_b

    # Inference sequences are prompt+gen; prompt length varies, use conservative fixed prompt sizes.
    red_seq = 600 + cfg.generation.red_max_new_tokens
    soc_seq = 600 + cfg.generation.socratic_max_new_tokens
    judge_seq = 900 + cfg.generation.judge_max_new_tokens

    red_inf = estimate_inference_gb_total(
        params_b=red_params_b,
        quantization=cfg.models.red.quantization,
        dtype="bfloat16",
        batch_size=1,
        seq_len=red_seq,
    )
    estimates["red_inference_total_gb"] = red_inf

    soc_inf = estimate_inference_gb_total(
        params_b=socratic_params_b,
        quantization="none",
        dtype=cfg.models.socratic.torch_dtype,
        batch_size=1,
        seq_len=soc_seq,
    )
    estimates["socratic_inference_total_gb"] = soc_inf

    judge_inf = estimate_inference_gb_total(
        params_b=judge_params_b,
        quantization="none",
        dtype=cfg.models.judge.torch_dtype,
        batch_size=1,
        seq_len=judge_seq,
    )
    estimates["judge_inference_total_gb"] = judge_inf

    soc_train = estimate_socratic_train_gb_per_gpu(
        params_b=socratic_params_b,
        dtype=cfg.models.socratic.torch_dtype,
        train_lora=cfg.models.socratic.train_lora,
        micro_batch_size=cfg.training.grpo.micro_batch_size,
        seq_len=soc_seq,
    )
    estimates["socratic_train_per_gpu_gb"] = soc_train

    red_train = estimate_qlora_train_gb_per_gpu(
        params_b=red_params_b,
        base_quantization=cfg.models.red.quantization,
        compute_dtype="bfloat16",
        micro_batch_size=1,
        seq_len=red_seq,
    )
    estimates["red_lora_train_per_gpu_gb"] = red_train

    ok = True
    margin = cfg.memory.safety_margin

    def _warn(msg: str) -> None:
        nonlocal ok
        warnings_out.append(msg)
        ok = False

    def _subset(allowed: Optional[Tuple[int, ...]]) -> Tuple[GpuInfo, ...]:
        if not gpus:
            return ()
        if not allowed:
            return gpus
        wanted = {int(x) for x in allowed}
        sub = tuple(g for g in gpus if int(g.idx) in wanted)
        return sub or gpus

    red_gpus = _subset(tuple(cfg.models.red.allowed_gpus) if cfg.models.red.allowed_gpus else None)
    soc_gpus = _subset(tuple(cfg.models.socratic.allowed_gpus) if cfg.models.socratic.allowed_gpus else None)
    judge_gpus = _subset(tuple(cfg.models.judge.allowed_gpus) if cfg.models.judge.allowed_gpus else None)

    # Check each stage separately (staged execution means these are NOT concurrent).
    for stage_name, total_gb, stage_gpus in [
        ("Red (generation)", red_inf, red_gpus),
        ("Socratic (hint gen)", soc_inf, soc_gpus),
        ("Judge (scoring)", judge_inf, judge_gpus),
    ]:
        if not _fits_per_gpu(total_gb, stage_gpus, margin):
            _warn(
                f"{stage_name} estimate {total_gb:.1f}GB exceeds available GPU memory "
                f"(safety_margin={margin:.2f}). Enable more sharding/offload, or reduce lengths."
            )

    if soc_gpus:
        max_per_gpu = max(g.total_gb for g in soc_gpus) * margin
        if soc_train > max_per_gpu:
            _warn(
                f"Socratic training estimate {soc_train:.1f}GB per GPU exceeds ~{max_per_gpu:.1f}GB. "
                f"Use LoRA ({cfg.models.socratic.train_lora=}), reduce seq_len, or use sharded optimizer."
            )
    if red_gpus:
        max_per_gpu = max(g.total_gb for g in red_gpus) * margin
        if red_train > max_per_gpu:
            _warn(
                f"Red LoRA training estimate {red_train:.1f}GB per GPU exceeds ~{max_per_gpu:.1f}GB. "
                "Reduce sequence length or use more aggressive offload/sharding."
            )

    # Auto-reduce knobs if configured.
    if cfg.memory.auto_reduce and warnings_out:
        # Mutate cfg in-place to safer defaults, but never reduce below mins.
        def _reduce_tokens(path: str, current: int) -> int:
            return max(cfg.memory.min_max_new_tokens, int(math.floor(current * 0.75)))

        def _effective_floor(current: int, configured_floor: int) -> int:
            current = max(1, int(current))
            configured_floor = max(1, int(configured_floor))
            return min(current, configured_floor)

        def _reduce_count(current: int, configured_floor: int) -> int:
            current = max(1, int(current))
            floor = _effective_floor(current, configured_floor)
            if current <= floor:
                return current
            return max(floor, int(math.floor(current * 0.75)))

        # Prioritize reducing the longest generations first.
        cfg.generation.judge_max_new_tokens = _reduce_tokens(
            "generation.judge_max_new_tokens", cfg.generation.judge_max_new_tokens
        )
        cfg.generation.red_max_new_tokens = _reduce_tokens(
            "generation.red_max_new_tokens", cfg.generation.red_max_new_tokens
        )
        cfg.generation.socratic_max_new_tokens = _reduce_tokens(
            "generation.socratic_max_new_tokens", cfg.generation.socratic_max_new_tokens
        )

        # Reduce generations (num_hints/tasks) if still too heavy.
        cfg.generation.socratic_num_hints = _reduce_count(cfg.generation.socratic_num_hints, cfg.memory.min_num_hints)
        cfg.generation.red_num_tasks = _reduce_count(cfg.generation.red_num_tasks, cfg.memory.min_red_num_tasks)

        # Keep GRPO group_size aligned with number of hints we generate.
        cfg.training.grpo.group_size = min(cfg.training.grpo.group_size, cfg.generation.socratic_num_hints)

        updates = {
            "generation.judge_max_new_tokens": cfg.generation.judge_max_new_tokens,
            "generation.red_max_new_tokens": cfg.generation.red_max_new_tokens,
            "generation.socratic_max_new_tokens": cfg.generation.socratic_max_new_tokens,
            "generation.socratic_num_hints": cfg.generation.socratic_num_hints,
            "generation.red_num_tasks": cfg.generation.red_num_tasks,
            "training.grpo.group_size": cfg.training.grpo.group_size,
        }

        warnings.warn(
            "Preflight detected a heavy configuration; auto-reduced generation lengths/counts. "
            f"Updates: {updates}"
        )

    # Curriculum sanity: ensure requested buckets exist at runtime; here just check that curriculum loaded.
    if not curriculum.topics:
        _warn("Curriculum is empty; Red will have no allowed buckets.")

    if dry_run:
        for w in warnings_out:
            warnings.warn(w)

    return PreflightReport(ok=ok, warnings=tuple(warnings_out), estimates_gb=estimates, suggested_updates=updates)
