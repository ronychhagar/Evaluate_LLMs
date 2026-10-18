"""Run a small offline evaluation to verify the package."""

from llm_eval import Dataset, Evaluator, Example, Generation, PromptTemplate


class EchoModel:
    name = "echo"
    input_cost_per_million = 0.0
    output_cost_per_million = 0.0

    def generate(self, prompt: str) -> Generation:
        answer = "Paris" if "capital of France" in prompt else "4"
        return Generation(text=answer, input_tokens=8, output_tokens=2)


dataset = Dataset(
    [
        Example(input="What is the capital of France?", expected_output="Paris"),
        Example(input="What is 2 + 2?", expected_output="4"),
    ],
    name="quickstart",
)
report = Evaluator(
    dataset=dataset,
    prompts=[PromptTemplate("concise", "Answer briefly: {input}")],
    models=[EchoModel()],
).run()

print(report.to_json())
