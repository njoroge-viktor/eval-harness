"""Core domain types.

Deliberately dependency-free: the whole library runs on the standard library so
it can be dropped into an annotation pipeline without dragging in a dependency
tree, and so CI stays fast.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    """Outcome of a pairwise comparison between two model responses."""

    A = "a"
    B = "b"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class Criterion:
    """One scored dimension of a rubric.

    Attributes:
        name: Stable identifier, used as the key in submitted scores.
        weight: Relative contribution to the weighted total. Must be positive.
        scale: Inclusive ``(low, high)`` bounds a score must fall within.
        description: Human-readable guidance shown to annotators.
    """

    name: str
    weight: float = 1.0
    scale: tuple[int, int] = (1, 5)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("criterion name must be non-empty")
        if self.weight <= 0:
            raise ValueError(f"{self.name}: weight must be positive, got {self.weight}")
        low, high = self.scale
        if low >= high:
            raise ValueError(f"{self.name}: scale must be increasing, got {self.scale}")

    def validate_score(self, value: float) -> None:
        """Raise :class:`ValueError` if ``value`` falls outside the scale."""
        low, high = self.scale
        if not low <= value <= high:
            raise ValueError(
                f"{self.name}: score {value} outside scale [{low}, {high}]"
            )


@dataclass(frozen=True, slots=True)
class Annotation:
    """One annotator's scores for one item."""

    item_id: str
    annotator_id: str
    scores: Mapping[str, float]
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id must be non-empty")
        if not self.annotator_id:
            raise ValueError("annotator_id must be non-empty")
        if not self.scores:
            raise ValueError(f"{self.item_id}: at least one score is required")


@dataclass(frozen=True, slots=True)
class Comparison:
    """One annotator's preference between two responses to the same prompt."""

    item_id: str
    annotator_id: str
    response_a: str
    response_b: str
    verdict: Verdict
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.response_a == self.response_b:
            raise ValueError(
                f"{self.item_id}: cannot compare a response against itself"
            )


@dataclass(frozen=True, slots=True)
class ScoredItem:
    """Result of applying a rubric to one annotation."""

    item_id: str
    annotator_id: str
    weighted_score: float
    normalized: float
    per_criterion: Mapping[str, float] = field(default_factory=dict)
