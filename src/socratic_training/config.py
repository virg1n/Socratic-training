from __future__ import annotations

from typing import List, Literal, Optional

try:  # pydantic v2 provides a v1 compat layer
    from pydantic.v1 import BaseModel, Field
except Exception:  # pragma: no cover
    from pydantic import BaseModel, Field  # type: ignore[no-redef]


class SocraticModelConfig(BaseModel):
    path: str
    params_b: Optional[float] = Field(
        default=None,
        description="Optional explicit parameter-count estimate (billions). Overrides path/config inference for preflight.",
    )
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    train_lora: bool = True
    adapter_dir: str = "runs/socratic_lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    allowed_gpus: Optional[List[int]] = Field(
        default=None,
        description="Optional list of CUDA device indices to allow for this model (single-node). "
        "When set, other GPUs will be excluded from HF device_map sharding.",
    )
    save_mode: Literal["pretrained", "state_dict", "none"] = Field(
        default="pretrained",
        description="How to save Socratic after GRPO updates: 'pretrained' (HF save_pretrained), "
        "'state_dict' (torch.save state_dict), or 'none'.",
    )
    max_saved_checkpoints: int = Field(
        default=2,
        ge=1,
        description="Maximum number of Socratic checkpoints to keep on disk. "
        "For save_mode='state_dict' this keeps the latest plus one '.prev' backup when set to 2.",
    )
    state_dict_name: str = Field(
        default="socratic_state_dict.pt",
        description="Filename used when save_mode='state_dict' (saved under adapter_dir).",
    )


class RedModelConfig(BaseModel):
    path: str
    params_b: Optional[float] = Field(
        default=None,
        description="Optional explicit parameter-count estimate (billions). Overrides path inference for preflight.",
    )
    quantization: Literal["none", "4bit", "8bit"] = "4bit"
    adapter_dir: str = "runs/red_lora"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    allowed_gpus: Optional[List[int]] = Field(
        default=None,
        description="Optional list of CUDA device indices to allow for this model (single-node).",
    )
    save_adapters: bool = Field(
        default=True,
        description="Whether to save Red adapters at the end of SFT/DPO training runs.",
    )


class JudgeModelConfig(BaseModel):
    path: str
    params_b: Optional[float] = Field(
        default=None,
        description="Optional explicit parameter-count estimate (billions). Overrides path inference for preflight.",
    )
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    quantization_fallback: Optional[Literal["none", "4bit", "8bit"]] = Field(
        default=None,
        description="Optional fallback if full BF16/FP16 judge inference is too heavy.",
    )
    allowed_gpus: Optional[List[int]] = Field(
        default=None,
        description="Optional list of CUDA device indices to allow for this model (single-node).",
    )


class ModelsConfig(BaseModel):
    socratic: SocraticModelConfig
    red: RedModelConfig
    judge: JudgeModelConfig


class GenerationConfig(BaseModel):
    red_num_tasks: int = 10
    red_max_new_tokens: int = 900

    socratic_num_hints: int = 6
    socratic_max_new_tokens: int = 220

    judge_max_new_tokens: int = 700


class ValidationConfig(BaseModel):
    python_timeout_s: float = 2.5
    max_tests: int = 12
    min_tests: int = 6
    require_buggy_passes_some: bool = True


class GRPOConfig(BaseModel):
    group_size: int = 6
    ppo_clip: float = 0.2
    lr: float = 2.0e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    micro_batch_size: int = 1
    grad_accum_steps: int = 8
    epochs_per_iter: int = 1
    beta_answer_dump_penalty: float = 3.0
    use_gradient_checkpointing: bool = True
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"


class TrainingConfig(BaseModel):
    grpo: GRPOConfig


class LoggingConfig(BaseModel):
    out_dir: str = "runs"
    jsonl_path: str = "runs/events.jsonl"
    hard_buffer_path: str = "runs/hard_buffer.jsonl"
    red_dpo_pairs_path: str = "runs/red_dpo_pairs.jsonl"


class MemoryConfig(BaseModel):
    safety_margin: float = 0.90
    auto_reduce: bool = True
    min_max_new_tokens: int = 96
    min_num_hints: int = 4
    min_red_num_tasks: int = 6


class ExecutionConfig(BaseModel):
    keep_socratic_loaded: bool = Field(
        default=False,
        description="If true, keep Socratic model loaded across multiple iterations (used by run-loop).",
    )
    keep_judge_loaded: bool = Field(
        default=False,
        description="If true, keep Judge model loaded across multiple iterations (used by run-loop).",
    )
    keep_red_loaded: bool = Field(
        default=False,
        description="If true, keep Red model loaded across multiple iterations (used by run-loop).",
    )
    reload_every_iters: int = Field(
        default=0,
        ge=0,
        description="If >0, force model reload every N iterations in run-loop (even when keep_* flags are true).",
    )


class AppConfig(BaseModel):
    curriculum_path: str = "curriculum.txt"
    models: ModelsConfig
    generation: GenerationConfig = GenerationConfig()
    validation: ValidationConfig = ValidationConfig()
    training: TrainingConfig = TrainingConfig(grpo=GRPOConfig())
    logging: LoggingConfig = LoggingConfig()
    memory: MemoryConfig = MemoryConfig()
    execution: ExecutionConfig = ExecutionConfig()
