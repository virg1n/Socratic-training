# Socratic Training

Sequential post-training stack for a Python tutoring model with three staged roles:

- `Socratic`: your ~4B SFT model, updated with staged GRPO.
- `Red`: `Qwen/Qwen3-32B` in 4-bit with LoRA, used for curriculum-bounded hard-example generation.
- `Judge`: frozen `Qwen/Qwen3-32B`, loaded only for rubric scoring.

The project is designed for a single Linux workstation with `4 x RTX 6000 Ada 48 GB`, ~`130 GB` RAM, and CUDA. It explicitly avoids impossible configurations by:

- loading only one role at a time,
- estimating memory before work starts,
- shrinking lengths / batch sizes when scenarios look unsafe,
- using CPU offload budgets and explicit unload cleanup.

## Architecture

1. Red generates `10` candidate tasks for one curriculum bucket.
2. Red generates one buggy solution per task.
3. Validator rejects syntax errors, unsafe code, off-topic tasks, trivial tasks, duplicates, and broken test packages.
4. Socratic samples `4-8` hints per valid task.
5. Judge scores hints with rubric subscores and ranking.
6. Socratic updates with a staged GRPO-style objective.
7. Poor-performing valid tasks go into the hard-example buffer.
8. Red is periodically refreshed with SFT and DPO on accepted hard examples and rejected counterexamples.

## Repository Layout

- `src/socratic_training/curriculum.py`: human-editable curriculum parser.
- `src/socratic_training/memory.py`: scenario estimator and safety tuning.
- `src/socratic_training/model_manager.py`: staged load/unload manager with cache cleanup.
- `src/socratic_training/validators.py`: task validation and sandboxed test execution.
- `src/socratic_training/socratic.py`: hint rollout and GRPO optimizer.
- `src/socratic_training/red.py`: curriculum-constrained task generation and Red training.
- `src/socratic_training/judge.py`: rubric scoring and disclosure penalties.
- `src/socratic_training/pipeline.py`: end-to-end sequential loop.

## Curriculum Format

`curriculum.txt` is the only source of allowed topics. The parser expects:

- `TOPIC:` to start a topic block,
- `SUBTOPIC:` to start a subtopic bucket,
- `OBJECTIVES:` / `KEYWORDS:` / `FORBIDDEN:` / `DIFFICULTIES:` as either comma-separated values or bullet lists,
- `---` to separate topic blocks.

Example:

```text
TOPIC: functions
DESCRIPTION: defining and calling small functions
DIFFICULTIES:
- beginner
- easy
FORBIDDEN:
- decorators
- recursion
SUBTOPIC: parameters and return values
OBJECTIVES:
- define a function with parameters
- return a computed value
KEYWORDS:
- def
- return
- argument
SUBTOPIC: tracing function calls
OBJECTIVES:
- trace a value through a function call
- predict the returned result
KEYWORDS: call, return, parameter
---
TOPIC: conditionals
DESCRIPTION: simple boolean branching
DIFFICULTIES: beginner, easy
FORBIDDEN:
- pattern matching
- walrus operator
SUBTOPIC: if else branches
OBJECTIVES:
- compare values with if and else
- follow one branch based on a condition
KEYWORDS:
- if
- else
- comparison
```

## Config

Default workstation settings live in `configs/workstation.toml`.

Important defaults:

- Socratic training stays conservative: short prompt/hint windows, micro-batch `1`, gradient checkpointing, paged optimizer.
- Judge loads separately with `device_map="balanced"` and CPU offload budget.
- Red uses 4-bit QLoRA for both generation and adapter training.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Preflight:

```bash
socratic-training --config configs/workstation.toml preflight
```

Run one iteration:

```bash
socratic-training --config configs/workstation.toml run-iteration --topic functions --subtopic "parameters and return values"
```

## Notes

- `Judge` is frozen and never trained.
- `Red` is not unconstrained adversarial generation; prompts explicitly bind it to one curriculum bucket.
- Validator requires canonical solutions to pass all tests and buggy solutions to fail at least one but not all tests.
- Hard examples are logged in `artifacts/hard_examples.jsonl`.
- Coverage and Socratic performance are logged per bucket in `artifacts/`.
