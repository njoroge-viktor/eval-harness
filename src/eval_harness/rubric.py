"""Weighted multi-dimensional rubric scoring."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .models import Annotation, Criterion, ScoredItem


@dataclass(frozen=True, slots=True)
class Rubric:
    """A weighted set of scoring criteria.

    A rubric turns an annotator's raw per-criterion scores into a single
    comparable number, normalised to ``[0, 1]`` so rubrics with different
    scales and weights can be compared directly.
    """

    name: str
    criteria: Sequence[Criterion]

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError(f"{self.name}: rubric needs at least one criterion")
        seen = [c.name for c in self.criteria]
        duplicates = {n for n in seen if seen.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"{self.name}: duplicate criteria {sorted(duplicates)}"
            )

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)

    def score(self, annotation: Annotation, *, strict: bool = True) -> ScoredItem:
        """Apply the rubric to one annotation.

        Args:
            annotation: The annotator's raw scores.
            strict: When true, every criterion must be present and in range.
                When false, missing criteria are skipped and the weighting is
                renormalised over those that were supplied — useful for partial
                review passes.

        Returns:
            The weighted total plus its ``[0, 1]`` normalisation.

        Raises:
            ValueError: If required criteria are missing, unknown criteria were
                submitted, or a score falls outside its scale.
        """
        known = {c.name: c for c in self.criteria}

        unknown = set(annotation.scores) - set(known)
        if unknown:
            raise ValueError(
                f"{annotation.item_id}: unknown criteria {sorted(unknown)}"
            )

        if strict:
            missing = set(known) - set(annotation.scores)
            if missing:
                raise ValueError(
                    f"{annotation.item_id}: missing criteria {sorted(missing)}"
                )

        weighted = 0.0
        applied_weight = 0.0
        per_criterion: dict[str, float] = {}

        for name, raw in annotation.scores.items():
            criterion = known[name]
            criterion.validate_score(raw)

            low, high = criterion.scale
            # Normalise onto [0, 1] so criteria with different scales combine
            # sensibly, then weight.
            unit = (raw - low) / (high - low)
            contribution = unit * criterion.weight

            weighted += contribution
            applied_weight += criterion.weight
            per_criterion[name] = unit

        if applied_weight == 0:  # pragma: no cover - guarded by __post_init__
            raise ValueError(f"{annotation.item_id}: no criteria applied")

        return ScoredItem(
            item_id=annotation.item_id,
            annotator_id=annotation.annotator_id,
            weighted_score=weighted,
            normalized=weighted / applied_weight,
            per_criterion=per_criterion,
        )

    def score_all(
        self, annotations: Iterable[Annotation], *, strict: bool = True
    ) -> list[ScoredItem]:
        """Score many annotations, preserving input order."""
        return [self.score(a, strict=strict) for a in annotations]

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Rubric:
        """Build a rubric from parsed JSON.

        Expected shape::

            {"name": "code-quality",
             "criteria": [{"name": "correctness", "weight": 3,
                           "scale": [1, 5], "description": "..."}]}
        """
        name = payload.get("name")
        if not isinstance(name, str):
            raise ValueError("rubric 'name' must be a string")

        raw_criteria = payload.get("criteria")
        if not isinstance(raw_criteria, list):
            raise ValueError("rubric 'criteria' must be a list")

        criteria: list[Criterion] = []
        for entry in raw_criteria:
            if not isinstance(entry, dict):
                raise ValueError("each criterion must be an object")
            scale = entry.get("scale", [1, 5])
            if not isinstance(scale, (list, tuple)) or len(scale) != 2:
                raise ValueError(
                    f"{entry.get('name')}: 'scale' must be a two-element list"
                )
            criteria.append(
                Criterion(
                    name=str(entry["name"]),
                    weight=float(entry.get("weight", 1.0)),
                    scale=(int(scale[0]), int(scale[1])),
                    description=str(entry.get("description", "")),
                )
            )

        return cls(name=name, criteria=criteria)
