# LLM Eval

A small, reusable Python library for comparing prompts, models, and retrievers
against the same dataset. It records per-example quality, latency, token usage,
and estimated cost, then aggregates the results by experiment configuration.

## Features

- Compare prompt templates, models, and retrievers on a shared set of examples.
- Provider adapters for Anthropic, Azure OpenAI, and OpenAI-compatible chat
  completion endpoints.
- Pluggable retrievers and quality scoring functions.
- Per-example results and aggregate summaries, exportable as JSON or CSV.
- No provider SDK is required unless that provider adapter is used.

## Install

Install the provider support you need:

```shell
pip install -e ".[anthropic,openai]"
```

For development and tests:

```shell
pip install -e ".[dev]"
pytest
```

## Quick start

```python
from llm_eval import Dataset, Evaluator, Example, OpenAICompatibleModel, PromptTemplate

dataset = Dataset(
    [
        Example(input="What is the capital of France?", expected_output="Paris"),
        Example(input="What is 2 + 2?", expected_output="4"),
    ],
    name="general-knowledge",
)

model = OpenAICompatibleModel(
    model="gpt-4o-mini",
    api_key="YOUR_API_KEY",
    input_cost_per_million=0.15,
    output_cost_per_million=0.60,
)
report = Evaluator(
    dataset=dataset,
    prompts=[
        PromptTemplate("concise", "Answer briefly: {input}"),
        PromptTemplate("explain", "Answer and explain: {input}"),
    ],
    models=[model],
).run()

print(report.to_json())
report.write_csv("results.csv")
```

The API key can also be supplied through the standard `OPENAI_API_KEY`
environment variable. For an OpenAI-compatible endpoint, pass `base_url`.
For Azure OpenAI, use `AzureOpenAIModel` with a deployment name,
`azure_endpoint`, and `AZURE_OPENAI_API_KEY` (or an explicit `api_key`).
`AnthropicModel` accepts an explicit key or the standard `ANTHROPIC_API_KEY`
environment variable.

## Compare retrievers

Pass retrievers implementing `name` and `retrieve(query) -> str`. Their output
is supplied to the prompt template's `{context}` field. If retrievers are
omitted, the evaluator runs a single no-retrieval baseline.

## Quality and cost

The default quality score is normalized exact match (case- and
whitespace-insensitive). Supply `quality_evaluators` as a mapping of names to
functions accepting `(expected_output, actual_output)` and returning a numeric
score; the reported quality is their arithmetic mean.

Cost is estimated from model-provided token counts and the configured
`input_cost_per_million` and `output_cost_per_million` rates. Rates default to
zero, so configure them for meaningful cost reports. Latency measures the
model generation call and excludes retrieval.

## Offline example

```shell
python examples/basic.py
```

