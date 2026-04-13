# Socratic Training (single-node)

Production-oriented training pipeline for a **Socratic hinting LLM** using **GRPO**, with a curriculum-constrained **Red** task generator and a frozen **Judge**.

Key properties:
- **Single node** (Linux + CUDA target), **staged model loading** (never keep Socratic + Red + Judge on GPU at the same time).
- **Curriculum is the only allowed source of topics** for Red.
- Strong **task validation** (syntax, tests, buggy-vs-correct behavior, curriculum alignment).
- Judge provides **rubric subscores** and a weighted reward with a **hard penalty for answer dumping**.

## Quick start

1) Install (example)
```bash
pip install -e ".[gpu,train]"
```

2) Edit `curriculum.txt` (see format below).

3) Run one iteration (generates tasks → validates → Socratic hints → Judge → GRPO update)
```bash
socratic-train run-iteration --config configs/default.yaml --topic "loops" --difficulty "easy"
```

4) Preflight (estimates memory + auto-reduces lengths/batches if needed)
```bash
socratic-train preflight --config configs/default.yaml
```

Train Red adapters (optional, periodic):
```bash
socratic-train train-red-sft --config configs/default.yaml
socratic-train train-red-dpo --config configs/default.yaml
```

Debug Red generation/validation (no Socratic/Judge load):
```bash
socratic-train debug-red --config configs/default.yaml --topic "loops" --difficulty "easy"
```

## Curriculum format (`curriculum.txt`)

The parser expects **topic blocks** separated by `---` and fields like `TOPIC:`, `DIFFICULTIES:`, `FORBIDDEN:`, `SUBTOPIC:`, `OBJECTIVES:`, `KEYWORDS:`.

Example:
```text
TOPIC: loops
DESCRIPTION: beginner loop tracing and simple accumulation
DIFFICULTIES:
- beginner
- easy
FORBIDDEN:
- generators
- async iteration
SUBTOPIC: for loops over ranges
OBJECTIVES:
- iterate with range
- accumulate a running total
KEYWORDS:
- for
- range
---
```

## Project layout

- `src/socratic_training/curriculum.py` – curriculum parser + bucket queries
- `src/socratic_training/memory.py` – VRAM/RAM estimator + safety auto-scaling
- `src/socratic_training/models/loader.py` – staged load/unload, offload, quant, LoRA
- `src/socratic_training/red/` – curriculum-constrained task generation
- `src/socratic_training/validation/` – schema + sandboxed execution validator
- `src/socratic_training/socratic/` – hint generation + disclosure guards
- `src/socratic_training/judge/` – rubric + scoring + ranking
- `src/socratic_training/rl/grpo.py` – minimal GRPO implementation
- `src/socratic_training/buffers/` – hard-example buffer for Red SFT/DPO

## Notes

- This repo targets a workstation with **4× RTX 6000 Ada (48GB each)** and **~130GB RAM**.
- Judge and Red are large; staged loading and CPU offload are required for many configurations.
- Validation executes generated code with strict timeouts; keep the curriculum constrained to safe, beginner Python topics.
