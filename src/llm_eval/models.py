"""Provider-neutral model interface and supported provider adapters."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Generation:
    """Text and usage information returned by a model."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class Model(Protocol):
    """Interface required by the evaluation engine."""

    name: str
    input_cost_per_million: float
    output_cost_per_million: float

    def generate(self, prompt: str) -> Generation:
        """Generate a response to a rendered prompt."""


class AnthropicModel:
    """Adapter for Anthropic's Messages API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        max_tokens: int = 1024,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ImportError(
                    "Anthropic support requires the optional dependency. "
                    "Install it with `pip install llm-eval[anthropic]`."
                ) from exc
            client = Anthropic(api_key=api_key)
        self.client = client
        self.name = model
        self.max_tokens = max_tokens
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

    def generate(self, prompt: str) -> Generation:
        response = self.client.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return Generation(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class OpenAICompatibleModel:
    """Adapter for OpenAI and OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenAI-compatible support requires the optional dependency. "
                    "Install it with `pip install llm-eval[openai]`."
                ) from exc
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.name = model
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

    def generate(self, prompt: str) -> Generation:
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = response.choices[0]
        usage = response.usage
        return Generation(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage is not None else 0,
            output_tokens=usage.completion_tokens if usage is not None else 0,
        )


class AzureOpenAIModel:
    """Adapter for Azure OpenAI chat-completions deployments."""

    def __init__(
        self,
        deployment: str,
        *,
        azure_endpoint: str,
        api_key: str | None = None,
        api_version: str = "2024-06-01",
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import AzureOpenAI
            except ImportError as exc:
                raise ImportError(
                    "Azure OpenAI support requires the optional dependency. "
                    "Install it with `pip install llm-eval[openai]`."
                ) from exc
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        self.client = client
        self.name = deployment
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

    def generate(self, prompt: str) -> Generation:
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = response.choices[0]
        usage = response.usage
        return Generation(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage is not None else 0,
            output_tokens=usage.completion_tokens if usage is not None else 0,
        )

