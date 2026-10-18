import json
from types import SimpleNamespace

import pytest

from llm_eval import (
    AnthropicModel,
    AzureOpenAIModel,
    Dataset,
    Evaluator,
    Example,
    Generation,
    OpenAICompatibleModel,
    PromptTemplate,
)
from llm_eval.engine import normalized_exact_match


class FakeModel:
    name = "fake"
    input_cost_per_million = 2.0
    output_cost_per_million = 4.0

    def generate(self, prompt: str) -> Generation:
        if "Question: " in prompt:
            answer = prompt.split("Question: ", maxsplit=1)[1].splitlines()[0]
        else:
            answer = prompt
        return Generation(
            text=answer,
            input_tokens=100,
            output_tokens=20,
        )


class FakeRetriever:
    name = "knowledge-base"

    def retrieve(self, query: str) -> str:
        return f"context for {query}"


def test_evaluator_runs_all_combinations_and_calculates_metrics() -> None:
    dataset = Dataset(
        [Example(input="What is 2+2?", expected_output="What is 2+2?")],
        name="arithmetic",
    )
    report = Evaluator(
        dataset=dataset,
        prompts=[PromptTemplate("question", "Question: {input}\n{context}")],
        models=[FakeModel()],
        retrievers=[FakeRetriever()],
    ).run()

    assert len(report.results) == 1
    result = report.results[0]
    assert result.retriever == "knowledge-base"
    assert result.quality == 1.0
    assert result.cost == pytest.approx(0.00028)
    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert len(report.summaries) == 1
    assert report.summaries[0].mean_quality == 1.0
    assert report.summaries[0].total_cost == pytest.approx(0.00028)


def test_report_serializes_to_json() -> None:
    dataset = Dataset([Example(input="x", expected_output="x")])
    report = Evaluator(
        dataset=dataset,
        prompts=[PromptTemplate("default", "{input}")],
        models=[FakeModel()],
    ).run()

    serialized = json.loads(report.to_json())

    assert serialized["dataset"] == "dataset"
    assert serialized["summaries"][0]["examples"] == 1


def test_dataset_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Dataset([])


def test_evaluator_requires_each_experiment_dimension() -> None:
    dataset = Dataset([Example(input="x", expected_output="x")])

    with pytest.raises(ValueError, match="at least one prompt"):
        Evaluator(dataset=dataset, prompts=[], models=[FakeModel()])

    with pytest.raises(ValueError, match="at least one quality evaluator"):
        Evaluator(
            dataset=dataset,
            prompts=[PromptTemplate("default", "{input}")],
            models=[FakeModel()],
            quality_evaluators={},
        )


def test_default_quality_is_case_and_whitespace_insensitive() -> None:
    assert normalized_exact_match("  The Answer ", "the answer") == 1.0
    assert normalized_exact_match("answer", "different") == 0.0


def test_anthropic_adapter_normalizes_response() -> None:
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                content=[SimpleNamespace(type="text", text="hello")],
                usage=SimpleNamespace(input_tokens=3, output_tokens=2),
            )
        )
    )

    generation = AnthropicModel("claude-test", client=client).generate("prompt")

    assert generation == Generation("hello", input_tokens=3, output_tokens=2)


@pytest.mark.parametrize("model_type", [AzureOpenAIModel, OpenAICompatibleModel])
def test_openai_adapters_normalize_response(model_type: type) -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="hello"),
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
                )
            )
        )
    )
    if model_type is AzureOpenAIModel:
        model = model_type("deployment", azure_endpoint="https://example.test", client=client)
    else:
        model = model_type("model", client=client)

    generation = model.generate("prompt")

    assert generation == Generation("hello", input_tokens=3, output_tokens=2)
