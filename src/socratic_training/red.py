from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .model_manager import StagedModelManager, generate_texts
from .prompts import red_buggy_solution_prompt, red_dpo_prompt, red_sft_prompt, red_task_generation_prompt
from .types import CurriculumBucket, HardExample, TaskExample, TaskTest, TaskValidationResult
from .utils import extract_first_json


class RedGenerator:
    def __init__(self, config: PipelineConfig, models: StagedModelManager) -> None:
        self.config = config
        self.models = models

    def generate_tasks(self, bucket: CurriculumBucket, existing_tasks: list[TaskExample]) -> list[TaskExample]:
        prompt = red_task_generation_prompt(bucket, self.config.red.candidate_count, [task.statement for task in existing_tasks])
        with self.models.use_red(mode="generation") as runtime:
            output = generate_texts(
                runtime,
                [prompt],
                max_new_tokens=self.config.red.max_new_tokens,
                temperature=0.9,
                top_p=0.92,
            )[0]
            payload = extract_first_json(output)
            if not isinstance(payload, list):
                raise ValueError("Red task generation did not return a JSON array.")
            tasks = [self._task_from_payload(item) for item in payload[: self.config.red.candidate_count]]
            buggy_prompts = [red_buggy_solution_prompt(task, bucket) for task in tasks]
            buggy_outputs = generate_texts(
                runtime,
                buggy_prompts,
                max_new_tokens=min(384, self.config.red.max_new_tokens),
                temperature=0.7,
                top_p=0.9,
            )

        enriched: list[TaskExample] = []
        for task, output in zip(tasks, buggy_outputs, strict=False):
            payload = extract_first_json(output)
            task.buggy_solution = str(payload["buggy_solution"]).strip()
            task.bug_explanation = str(payload.get("bug_explanation", "")).strip()
            enriched.append(task)
        return enriched

    def _task_from_payload(self, payload: dict[str, Any]) -> TaskExample:
        tests = payload.get("tests", [])
        normalized_tests: list[dict[str, str]] = []
        for index, test in enumerate(tests):
            if isinstance(test, str):
                normalized_tests.append({"name": f"test_{index}", "code": test})
            else:
                normalized_tests.append({"name": test.get("name", f"test_{index}"), "code": test["code"]})
        return TaskExample(
            topic=str(payload["topic"]),
            difficulty=str(payload["difficulty"]),
            statement=str(payload["statement"]).strip(),
            canonical_solution=str(payload["canonical_solution"]).strip(),
            buggy_solution=str(payload.get("buggy_solution", "")).strip(),
            bug_explanation=str(payload.get("bug_explanation", "")).strip(),
            tests=[TaskTest(**item) for item in normalized_tests],
            expected_learning_objectives=[str(item) for item in payload.get("expected_learning_objectives", [])],
        )


class RedTrainer:
    def __init__(self, config: PipelineConfig, models: StagedModelManager) -> None:
        self.config = config
        self.models = models
        self.sft_records: list[dict[str, str]] = []
        self.dpo_records: list[dict[str, str]] = []

    def record_feedback(self, bucket: CurriculumBucket, validations: list[TaskValidationResult], hard_examples: list[HardExample]) -> None:
        accepted = [result.task for result in validations if result.accepted]
        rejected = [result.task for result in validations if not result.accepted]
        for task in accepted:
            self.sft_records.append({"text": red_sft_prompt(bucket, task) + "\n" + _task_to_json(task)})
        if not accepted or not rejected:
            return
        chosen_tasks = [example.task for example in hard_examples] or accepted[:1]
        for chosen in chosen_tasks:
            for rejected_task in rejected[:3]:
                self.dpo_records.append(
                    {
                        "prompt": red_dpo_prompt(bucket),
                        "chosen": _task_to_json(chosen),
                        "rejected": _task_to_json(rejected_task),
                    }
                )

    def maybe_train(self, iteration: int) -> None:
        if iteration % self.config.red.sft_after_iterations == 0 and self.sft_records:
            self._run_sft()
        if iteration % self.config.red.dpo_after_iterations == 0 and self.dpo_records:
            self._run_dpo()

    def _run_sft(self) -> None:
        datasets = _require_module("datasets")
        peft = _require_module("peft")
        trl = _require_module("trl")

        dataset = datasets.Dataset.from_list(self.sft_records)
        checkpoint_dir = Path(self.config.paths.checkpoints_dir) / "red" / "latest"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with self.models.use_red(mode="train") as runtime:
            lora_config = peft.LoraConfig(
                task_type=peft.TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.red.lora.r,
                lora_alpha=self.config.red.lora.alpha,
                lora_dropout=self.config.red.lora.dropout,
                target_modules=self.config.red.lora.target_modules,
            )
            runtime.model.enable_input_require_grads()
            sft_config = trl.SFTConfig(
                output_dir=str(checkpoint_dir),
                dataset_text_field="text",
                per_device_train_batch_size=self.config.red.train_batch_size,
                gradient_accumulation_steps=self.config.red.gradient_accumulation_steps,
                learning_rate=self.config.red.learning_rate,
                max_length=self.config.red.max_prompt_tokens + self.config.red.max_new_tokens,
                num_train_epochs=1,
                logging_steps=5,
                save_strategy="no",
                bf16=self.config.red.dtype == "bfloat16",
                report_to="none",
            )
            trainer = trl.SFTTrainer(
                model=runtime.model,
                args=sft_config,
                train_dataset=dataset,
                processing_class=runtime.tokenizer,
                peft_config=lora_config,
            )
            trainer.train()
            trainer.model.save_pretrained(checkpoint_dir)
            runtime.tokenizer.save_pretrained(checkpoint_dir)
        self.sft_records.clear()

    def _run_dpo(self) -> None:
        datasets = _require_module("datasets")
        peft = _require_module("peft")
        trl = _require_module("trl")

        dataset = datasets.Dataset.from_list(self.dpo_records)
        checkpoint_dir = Path(self.config.paths.checkpoints_dir) / "red" / "latest"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with self.models.use_red(mode="train") as runtime:
            lora_config = peft.LoraConfig(
                task_type=peft.TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.red.lora.r,
                lora_alpha=self.config.red.lora.alpha,
                lora_dropout=self.config.red.lora.dropout,
                target_modules=self.config.red.lora.target_modules,
            )
            dpo_config = trl.DPOConfig(
                output_dir=str(checkpoint_dir),
                per_device_train_batch_size=self.config.red.train_batch_size,
                gradient_accumulation_steps=self.config.red.gradient_accumulation_steps,
                learning_rate=self.config.red.dpo_learning_rate,
                max_prompt_length=self.config.red.max_prompt_tokens,
                max_length=self.config.red.max_prompt_tokens + self.config.red.max_new_tokens,
                num_train_epochs=1,
                save_strategy="no",
                bf16=self.config.red.dtype == "bfloat16",
                report_to="none",
            )
            trainer = trl.DPOTrainer(
                model=runtime.model,
                args=dpo_config,
                processing_class=runtime.tokenizer,
                train_dataset=dataset,
                peft_config=lora_config,
            )
            trainer.train()
            trainer.model.save_pretrained(checkpoint_dir)
            runtime.tokenizer.save_pretrained(checkpoint_dir)
        self.dpo_records.clear()


def _task_to_json(task: TaskExample) -> str:
    import json

    return json.dumps(task.to_dict(), indent=2, ensure_ascii=False)


def _require_module(name: str):
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(f"{name} is required for Red training.") from error
