from __future__ import annotations

from dataclasses import replace

from .config import PipelineConfig
from .model_manager import StagedModelManager, generate_texts
from .prompts import judge_ranking_prompt, judge_single_hint_prompt
from .types import HintCandidate, JudgeEvaluation, JudgeSubscores, TaskExample
from .utils import extract_first_json, normalize_text


class JudgeClient:
    def __init__(self, config: PipelineConfig, models: StagedModelManager) -> None:
        self.config = config
        self.models = models

    def evaluate_group(self, task: TaskExample, candidates: list[HintCandidate]) -> list[JudgeEvaluation]:
        prompts = [judge_single_hint_prompt(task, candidate.hint) for candidate in candidates]
        evaluations: list[JudgeEvaluation] = []
        with self.models.use_judge(allow_fallback=True) as runtime:
            outputs = generate_texts(
                runtime,
                prompts,
                max_new_tokens=self.config.judge.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            for candidate, output in zip(candidates, outputs, strict=False):
                evaluations.append(self._parse_single(candidate, output))

            if self.config.judge.pairwise_ranking and len(candidates) > 1:
                ranking_output = generate_texts(
                    runtime,
                    [judge_ranking_prompt(task, candidates)],
                    max_new_tokens=self.config.judge.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                )[0]
                ranking = self._parse_ranking(ranking_output, candidates)
                evaluations = self._apply_ranking(evaluations, ranking)

        evaluations.sort(key=lambda item: item.rank or 999)
        if all(evaluation.rank == 0 for evaluation in evaluations):
            evaluations.sort(key=lambda item: item.final_reward, reverse=True)
            for index, evaluation in enumerate(evaluations, start=1):
                evaluation.rank = index
        return evaluations

    def _parse_single(self, candidate: HintCandidate, raw_output: str) -> JudgeEvaluation:
        payload = extract_first_json(raw_output)
        subscores = JudgeSubscores(
            did_not_give_final_answer=float(payload["did_not_give_final_answer"]),
            pedagogical_value=float(payload["pedagogical_value"]),
            bug_localization_help=float(payload["bug_localization_help"]),
            curriculum_alignment=float(payload["curriculum_alignment"]),
            beginner_friendliness=float(payload["beginner_friendliness"]),
            actionability_of_hint=float(payload["actionability_of_hint"]),
        )
        disclosure = disclosure_penalty(candidate.hint, candidate.task, self.config.judge.disclosure_penalty)
        final_reward = subscores.weighted_sum(self.config.judge.weights) - disclosure
        if subscores.did_not_give_final_answer < 0.25:
            final_reward -= self.config.judge.disclosure_penalty
        return JudgeEvaluation(
            candidate=candidate,
            subscores=subscores,
            rationale=str(payload.get("rationale", "")).strip(),
            disclosure_penalty=disclosure,
            final_reward=final_reward,
        )

    def _parse_ranking(self, raw_output: str, candidates: list[HintCandidate]) -> list[int]:
        try:
            payload = extract_first_json(raw_output)
            ranking = [int(item) for item in payload["ranking"]]
        except Exception:
            return [candidate.index for candidate in candidates]
        expected = {candidate.index for candidate in candidates}
        if set(ranking) != expected:
            return [candidate.index for candidate in candidates]
        return ranking

    def _apply_ranking(self, evaluations: list[JudgeEvaluation], ranking: list[int]) -> list[JudgeEvaluation]:
        by_index = {evaluation.candidate.index: evaluation for evaluation in evaluations}
        updated: list[JudgeEvaluation] = []
        for rank, index in enumerate(ranking, start=1):
            updated.append(replace(by_index[index], rank=rank, final_reward=by_index[index].final_reward + (len(ranking) - rank) * 0.02))
        return updated


def disclosure_penalty(hint: str, task: TaskExample, base_penalty: float) -> float:
    normalized_hint = normalize_text(hint)
    normalized_solution = normalize_text(task.canonical_solution)
    if not normalized_hint:
        return base_penalty
    penalty = 0.0
    if "```" in hint or "def solve" in hint.lower():
        penalty += base_penalty
    if normalized_solution and normalized_solution[:120] and normalized_solution[:120] in normalized_hint:
        penalty += base_penalty
    if "return " in hint.lower() and "you should" in hint.lower():
        penalty += base_penalty * 0.5
    if len(hint.splitlines()) > 6:
        penalty += base_penalty * 0.25
    return penalty
