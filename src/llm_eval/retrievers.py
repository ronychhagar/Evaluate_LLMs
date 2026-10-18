"""Retriever interface and default no-retrieval implementation."""

from typing import Protocol


class Retriever(Protocol):
    """Interface for components that provide context for an input."""

    name: str

    def retrieve(self, query: str) -> str:
        """Return context relevant to the query."""


class NoOpRetriever:
    """Retriever that provides no context."""

    name = "none"

    def retrieve(self, query: str) -> str:
        return ""

