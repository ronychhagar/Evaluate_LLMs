"""Dataset structures used by evaluations."""

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Example:
    """One input and its expected answer."""

    input: str
    expected_output: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Dataset:
    """A named collection of examples shared across experiment variants."""

    examples: tuple[Example, ...]
    name: str = "dataset"

    def __init__(self, examples: Iterable[Example], name: str = "dataset") -> None:
        collected = tuple(examples)
        if not collected:
            raise ValueError("A dataset must contain at least one example.")
        object.__setattr__(self, "examples", collected)
        object.__setattr__(self, "name", name)

    def __len__(self) -> int:
        return len(self.examples)

