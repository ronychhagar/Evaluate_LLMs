"""Evaluation runner and report types."""

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from llm_eval.dataset import Dataset
from llm_eval.models import Generation, Model
from llm_eval.prompts import PromptTemplate
from llm_eval.retrievers import NoOpRetriever, Retriever

QualityEvaluator = Callable[[str, str], float]


def normalized_exact_match(expected: str, actual: str) -> float:
    """Return 1 for equal whitespace-normalized, case-insensitive strings."""
    return float(" ".join(expected.casefold().split()) == " ".join(actual.casefold().split()))


@dataclass(frozen=True)
class Result:
    """Per-example output and measured evaluation values."""

    dataset: str
    example_index: int
    prompt: str
    model: str
    retriever: str
    expected_output: str
    output: str
    quality: float
    latency_seconds: float
    cost: float
    input_tokens: int
    output_tokens: int
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class Summary:
    """Aggregate metrics for one prompt/model/retriever combination."""

    prompt: str
    model: str
    retriever: str
    examples: int
    mean_quality: float
    mean_latency_seconds: float
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int


@dataclass(frozen=True)
class EvaluationReport:
    """Detailed results and grouped aggregate metrics."""

    dataset: str
    results: tuple[Result, ...]
    summaries: tuple[Summary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "results": [asdict(result) for result in self.results],
            "summaries": [asdict(summary) for summary in self.summaries],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        if not self.results:
            raise ValueError("Cannot write CSV for a report with no results.")
        with Path(path).open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(asdict(self.results[0])))
            writer.writeheader()
            writer.writerows(asdict(result) for result in self.results)


class Evaluator:
    """Run every prompt/model/retriever combination against one dataset."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        prompts: Iterable[PromptTemplate],
        models: Iterable[Model],
        retrievers: Iterable[Retriever] | None = None,
        quality_evaluators: Mapping[str, QualityEvaluator] | None = None,
    ) -> None:
        self.dataset = dataset
        self.prompts = tuple(prompts)
        self.models = tuple(models)
        self.retrievers = tuple(retrievers) if retrievers is not None else (NoOpRetriever(),)
        if not self.prompts:
            raise ValueError("Provide at least one prompt.")
        if not self.models:
            raise ValueError("Provide at least one model.")
        if not self.retrievers:
            raise ValueError("Provide at least one retriever.")
        self.quality_evaluators = dict(
            {"exact_match": normalized_exact_match}
            if quality_evaluators is None
            else quality_evaluators
        )
        if not self.quality_evaluators:
            raise ValueError("Provide at least one quality evaluator.")

    def run(self) -> EvaluationReport:
        results: list[Result] = []
        for retriever in self.retrievers:
            for prompt_template in self.prompts:
                for model in self.models:
                    for index, example in enumerate(self.dataset.examples):
                        context = retriever.retrieve(example.input)
                        prompt = prompt_template.render(example.input, context)
                        started = time.perf_counter()
                        generation = model.generate(prompt)
                        latency = time.perf_counter() - started
                        if not isinstance(generation, Generation):
                            raise TypeError(
                                f"Model {model.name!r} must return a Generation instance."
                            )
                        quality = sum(
                            evaluator(example.expected_output, generation.text)
                            for evaluator in self.quality_evaluators.values()
                        ) / len(self.quality_evaluators)
                        cost = (
                            generation.input_tokens * model.input_cost_per_million
                            + generation.output_tokens * model.output_cost_per_million
                        ) / 1_000_000
                        results.append(
                            Result(
                                dataset=self.dataset.name,
                                example_index=index,
                                prompt=prompt_template.name,
                                model=model.name,
                                retriever=retriever.name,
                                expected_output=example.expected_output,
                                output=generation.text,
                                quality=quality,
                                latency_seconds=latency,
                                cost=cost,
                                input_tokens=generation.input_tokens,
                                output_tokens=generation.output_tokens,
                                metadata=example.metadata,
                            )
                        )
        return EvaluationReport(
            dataset=self.dataset.name,
            results=tuple(results),
            summaries=self._summarize(results),
        )

    @staticmethod
    def _summarize(results: list[Result]) -> tuple[Summary, ...]:
        grouped: dict[tuple[str, str, str], list[Result]] = {}
        for result in results:
            grouped.setdefault((result.prompt, result.model, result.retriever), []).append(
                result
            )
        summaries = []
        for (prompt, model, retriever), group in grouped.items():
            count = len(group)
            summaries.append(
                Summary(
                    prompt=prompt,
                    model=model,
                    retriever=retriever,
                    examples=count,
                    mean_quality=sum(item.quality for item in group) / count,
                    mean_latency_seconds=sum(
                        item.latency_seconds for item in group
                    )
                    / count,
                    total_cost=sum(item.cost for item in group),
                    total_input_tokens=sum(item.input_tokens for item in group),
                    total_output_tokens=sum(item.output_tokens for item in group),
                )
            )
        return tuple(summaries)
