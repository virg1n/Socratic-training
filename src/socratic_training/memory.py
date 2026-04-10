from __future__ import annotations

from dataclasses import replace

from .config import PipelineConfig
from .types import ScenarioEstimate


def estimate_model_weights_gb(params_billions: float, bits: int, quantization_overhead: float = 1.0) -> float:
    bytes_per_param = bits / 8.0
    return params_billions * 1_000_000_000 * bytes_per_param * quantization_overhead / (1024**3)


def estimate_activation_gb(
    hidden_size: int,
    num_layers: int,
    sequence_length: int,
    batch_size: int,
    bytes_per_value: int = 2,
    checkpointing: bool = True,
) -> float:
    multiplier = 4.0 if checkpointing else 9.0
    return hidden_size * num_layers * sequence_length * batch_size * bytes_per_value * multiplier / (1024**3)


def estimate_kv_cache_gb(hidden_size: int, num_layers: int, sequence_length: int, batch_size: int, bytes_per_value: int = 2) -> float:
    return hidden_size * num_layers * sequence_length * batch_size * bytes_per_value * 2 / (1024**3)


def preflight_and_tune(config: PipelineConfig) -> tuple[PipelineConfig, list[ScenarioEstimate]]:
    tuned = config
    scenarios = _estimate_scenarios(tuned)
    warnings: list[str] = []

    while any(not scenario.fits for scenario in scenarios):
        changed = False

        if not scenarios[0].fits and tuned.socratic.max_hint_tokens > 96:
            tuned = replace(
                tuned,
                socratic=replace(
                    tuned.socratic,
                    max_hint_tokens=max(96, int(tuned.socratic.max_hint_tokens * 0.8)),
                    group_size=max(4, min(tuned.socratic.group_size, 6)),
                ),
            )
            warnings.append("Reduced Socratic hint length for a safer GRPO rollout footprint.")
            changed = True

        if not scenarios[1].fits and tuned.red.max_new_tokens > 640:
            tuned = replace(
                tuned,
                red=replace(
                    tuned.red,
                    max_new_tokens=max(640, int(tuned.red.max_new_tokens * 0.8)),
                    generation_batch_size=max(1, tuned.red.generation_batch_size - 1),
                ),
            )
            warnings.append("Reduced Red generation length and batch size to keep QLoRA generation safe.")
            changed = True

        if not scenarios[2].fits and tuned.red.train_batch_size > 1:
            tuned = replace(tuned, red=replace(tuned.red, train_batch_size=1, gradient_accumulation_steps=tuned.red.gradient_accumulation_steps * 2))
            warnings.append("Collapsed Red LoRA train batch size to 1 and increased accumulation.")
            changed = True

        if not scenarios[3].fits and tuned.judge.max_prompt_tokens > 512:
            tuned = replace(
                tuned,
                judge=replace(
                    tuned.judge,
                    max_prompt_tokens=max(512, int(tuned.judge.max_prompt_tokens * 0.8)),
                    max_new_tokens=max(160, int(tuned.judge.max_new_tokens * 0.8)),
                ),
            )
            warnings.append("Reduced Judge prompt and response length for staged rubric evaluation.")
            changed = True
        elif not scenarios[3].fits and tuned.judge.quantization_fallback:
            warnings.append(f"Judge full-precision estimate is too heavy; enable fallback quantization ({tuned.judge.quantization_fallback}) if required.")
            break

        if not changed:
            break
        scenarios = _estimate_scenarios(tuned)

    for scenario in scenarios:
        scenario.warnings.extend(warnings)
    return tuned, scenarios


def _estimate_scenarios(config: PipelineConfig) -> list[ScenarioEstimate]:
    return [
        estimate_socratic_training(config),
        estimate_red_generation(config),
        estimate_red_training(config),
        estimate_judge_inference(config),
    ]


def estimate_socratic_training(config: PipelineConfig) -> ScenarioEstimate:
    weights = estimate_model_weights_gb(config.socratic.params_billions, 16)
    gradients = weights
    activations = estimate_activation_gb(
        hidden_size=config.socratic.hidden_size,
        num_layers=config.socratic.num_hidden_layers,
        sequence_length=config.socratic.max_prompt_tokens + config.socratic.max_hint_tokens,
        batch_size=config.socratic.micro_batch_size,
        checkpointing=config.socratic.use_gradient_checkpointing,
    )
    optimizer_cpu = weights * 4.0 if config.socratic.paged_optimizer else weights * 1.5
    gpu_total = weights + gradients + activations + 4.0
    cpu_total = optimizer_cpu + 6.0
    gpu_limit = config.hardware.gpu_vram_gb * config.hardware.gpu_utilization_target
    cpu_limit = min(config.hardware.cpu_offload_budget_gb, config.hardware.system_ram_gb * 0.8)
    warnings = []
    if gpu_total > gpu_limit:
        warnings.append("Socratic training estimate exceeds single-GPU target; reduce tokens or use adapter training.")
    if cpu_total > cpu_limit:
        warnings.append("Socratic paged optimizer exceeds configured CPU offload budget.")
    return ScenarioEstimate("socratic_training", round(gpu_total, 2), round(cpu_total, 2), gpu_total <= gpu_limit, cpu_total <= cpu_limit, warnings)


def estimate_red_generation(config: PipelineConfig) -> ScenarioEstimate:
    weights = estimate_model_weights_gb(config.red.params_billions, 4, quantization_overhead=1.20)
    kv_cache = estimate_kv_cache_gb(
        hidden_size=config.red.hidden_size,
        num_layers=config.red.num_hidden_layers,
        sequence_length=config.red.max_prompt_tokens + config.red.max_new_tokens,
        batch_size=config.red.generation_batch_size,
    )
    gpu_total = weights + kv_cache + 3.0
    gpu_limit = config.hardware.gpu_vram_gb * config.hardware.gpu_utilization_target
    return ScenarioEstimate("red_generation", round(gpu_total, 2), 6.0, gpu_total <= gpu_limit, True, [])


def estimate_red_training(config: PipelineConfig) -> ScenarioEstimate:
    weights = estimate_model_weights_gb(config.red.params_billions, 4, quantization_overhead=1.25)
    activations = estimate_activation_gb(
        hidden_size=config.red.hidden_size,
        num_layers=config.red.num_hidden_layers,
        sequence_length=config.red.max_prompt_tokens,
        batch_size=config.red.train_batch_size,
        checkpointing=True,
    )
    lora_states = 2.0
    gpu_total = weights + activations + lora_states + 4.0
    gpu_limit = config.hardware.gpu_vram_gb * config.hardware.gpu_utilization_target
    cpu_total = 8.0
    return ScenarioEstimate("red_lora_training", round(gpu_total, 2), cpu_total, gpu_total <= gpu_limit, True, [])


def estimate_judge_inference(config: PipelineConfig) -> ScenarioEstimate:
    weights = estimate_model_weights_gb(config.judge.params_billions, 16)
    shard_count = max(2, config.hardware.gpu_count)
    per_gpu_weights = weights / shard_count
    kv_cache = estimate_kv_cache_gb(
        hidden_size=config.judge.hidden_size,
        num_layers=config.judge.num_hidden_layers,
        sequence_length=config.judge.max_prompt_tokens + config.judge.max_new_tokens,
        batch_size=1,
    )
    gpu_total = per_gpu_weights + kv_cache + 4.0
    gpu_limit = config.hardware.gpu_vram_gb * config.hardware.gpu_utilization_target
    cpu_total = 8.0
    warnings = []
    if gpu_total > gpu_limit:
        warnings.append("Judge full-precision inference is too heavy even when sharded; enable fallback quantization.")
    return ScenarioEstimate("judge_inference", round(gpu_total, 2), cpu_total, gpu_total <= gpu_limit, True, warnings)
