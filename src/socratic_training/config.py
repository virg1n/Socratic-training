from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomllib


@dataclass(slots=True)
class HardwareConfig:
    gpu_count: int = 4
    gpu_vram_gb: float = 48.0
    system_ram_gb: float = 130.0
    gpu_utilization_target: float = 0.90
    cpu_offload_budget_gb: float = 96.0


@dataclass(slots=True)
class SocraticConfig:
    model_path: str = "custom/path/to/model"
    params_billions: float = 4.0
    hidden_size: int = 2560
    num_hidden_layers: int = 32
    dtype: str = "bfloat16"
    train_mode: str = "full"
    train_device: str = "cuda:0"
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_prompt_tokens: int = 512
    max_hint_tokens: int = 160
    group_size: int = 4
    rollout_temperature: float = 0.8
    top_p: float = 0.95
    clip_range: float = 0.2
    kl_beta: float = 0.02
    epochs_per_iteration: int = 1
    max_grad_norm: float = 1.0
    use_gradient_checkpointing: bool = True
    paged_optimizer: bool = True
    hard_example_threshold: float = 0.45


@dataclass(slots=True)
class LoraConfigData:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


@dataclass(slots=True)
class RedConfig:
    model_name: str = "Qwen/Qwen3-32B"
    params_billions: float = 32.0
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    quantization: str = "4bit"
    dtype: str = "bfloat16"
    train_device: str = "cuda:0"
    max_prompt_tokens: int = 640
    max_new_tokens: int = 900
    candidate_count: int = 10
    generation_batch_size: int = 2
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    dpo_learning_rate: float = 1e-4
    lora: LoraConfigData = field(default_factory=LoraConfigData)
    sft_after_iterations: int = 4
    dpo_after_iterations: int = 8


@dataclass(slots=True)
class JudgeConfig:
    model_name: str = "Qwen/Qwen3-32B"
    params_billions: float = 32.0
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    dtype: str = "bfloat16"
    quantization_fallback: str = "8bit"
    max_prompt_tokens: int = 768
    max_new_tokens: int = 256
    pairwise_ranking: bool = True
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "did_not_give_final_answer": 0.32,
            "pedagogical_value": 0.20,
            "bug_localization_help": 0.16,
            "curriculum_alignment": 0.12,
            "beginner_friendliness": 0.10,
            "actionability_of_hint": 0.10,
        }
    )
    disclosure_penalty: float = 1.5


@dataclass(slots=True)
class ValidationConfig:
    timeout_seconds: int = 4
    memory_limit_mb: int = 512
    similarity_threshold: float = 0.85
    min_statement_chars: int = 40
    max_statement_chars: int = 700
    max_solution_lines: int = 120
    allowed_imports: list[str] = field(default_factory=lambda: ["math", "itertools", "functools", "collections", "string", "re"])


@dataclass(slots=True)
class PathsConfig:
    curriculum_file: str = "curriculum.txt"
    output_dir: str = "artifacts"
    hard_buffer_file: str = "artifacts/hard_examples.jsonl"
    coverage_log_file: str = "artifacts/curriculum_coverage.jsonl"
    performance_log_file: str = "artifacts/socratic_metrics.jsonl"
    checkpoints_dir: str = "checkpoints"
    offload_dir: str = "offload"


@dataclass(slots=True)
class PipelineConfig:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    socratic: SocraticConfig = field(default_factory=SocraticConfig)
    red: RedConfig = field(default_factory=RedConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    seed: int = 42
    log_level: str = "INFO"
    max_valid_tasks_per_iteration: int = 4
    validate_topic_match_with_keywords: bool = True

    @classmethod
    def from_toml(cls, path: str | Path) -> "PipelineConfig":
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineConfig":
        return cls(
            hardware=HardwareConfig(**payload.get("hardware", {})),
            socratic=SocraticConfig(**payload.get("socratic", {})),
            red=RedConfig(
                **{
                    **payload.get("red", {}),
                    "lora": LoraConfigData(**payload.get("red", {}).get("lora", {})),
                }
            ),
            judge=JudgeConfig(**payload.get("judge", {})),
            validation=ValidationConfig(**payload.get("validation", {})),
            paths=PathsConfig(**payload.get("paths", {})),
            seed=payload.get("seed", 42),
            log_level=payload.get("log_level", "INFO"),
            max_valid_tasks_per_iteration=payload.get("max_valid_tasks_per_iteration", 4),
            validate_topic_match_with_keywords=payload.get("validate_topic_match_with_keywords", True),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
