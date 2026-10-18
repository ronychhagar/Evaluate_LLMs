"""Prompt template support."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """A named prompt with ``{input}`` and optional ``{context}`` fields."""

    name: str
    template: str

    def render(self, input: str, context: str = "") -> str:
        return self.template.format(input=input, context=context)

