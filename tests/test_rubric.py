"""Rubric scoring."""

from __future__ import annotations

import pytest

from eval_harness.models import Annotation, Criterion
from eval_harness.rubric import Rubric


@pytest.fixture
def rubric() -> Rubric:
    return Rubric(
        name="code-quality",
        criteria=[
            Criterion("correctness", weight=3.0, scale=(1, 5)),
            Criterion("clarity", weight=1.0, scale=(1, 5)),
        ],
    )


class TestCriterion:
    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Criterion("")

    def test_rejects_non_positive_weight(self) -> None:
        with pytest.raises(ValueError, match="weight must be positive"):
            Criterion("x", weight=0)

    def test_rejects_inverted_scale(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            Criterion("x", scale=(5, 1))

    def test_validates_score_bounds(self) -> None:
        c = Criterion("x", scale=(1, 5))
        c.validate_score(1)
        c.validate_score(5)
        with pytest.raises(ValueError, match="outside scale"):
            c.validate_score(6)


class TestRubricConstruction:
    def test_rejects_empty_criteria(self) -> None:
        with pytest.raises(ValueError, match="at least one criterion"):
            Rubric(name="empty", criteria=[])

    def test_rejects_duplicate_criteria(self) -> None:
        with pytest.raises(ValueError, match="duplicate criteria"):
            Rubric(
                name="dupes",
                criteria=[Criterion("a"), Criterion("a")],
            )

    def test_total_weight(self, rubric: Rubric) -> None:
        assert rubric.total_weight == 4.0


class TestScoring:
    def test_top_marks_normalise_to_one(self, rubric: Rubric) -> None:
        scored = rubric.score(
            Annotation("t1", "ann-a", {"correctness": 5, "clarity": 5})
        )
        assert scored.normalized == pytest.approx(1.0)

    def test_bottom_marks_normalise_to_zero(self, rubric: Rubric) -> None:
        scored = rubric.score(
            Annotation("t1", "ann-a", {"correctness": 1, "clarity": 1})
        )
        assert scored.normalized == pytest.approx(0.0)

    def test_hand_computed_weighting(self, rubric: Rubric) -> None:
        # correctness 4 on [1,5] -> unit 0.75, weight 3 -> 2.25
        # clarity     2 on [1,5] -> unit 0.25, weight 1 -> 0.25
        # weighted = 2.50; normalised = 2.50 / 4 = 0.625
        scored = rubric.score(
            Annotation("t1", "ann-a", {"correctness": 4, "clarity": 2})
        )
        assert scored.weighted_score == pytest.approx(2.5)
        assert scored.normalized == pytest.approx(0.625)

    def test_weighting_actually_favours_the_heavier_criterion(
        self, rubric: Rubric
    ) -> None:
        strong_where_it_counts = rubric.score(
            Annotation("t1", "a", {"correctness": 5, "clarity": 1})
        )
        strong_where_it_does_not = rubric.score(
            Annotation("t2", "a", {"correctness": 1, "clarity": 5})
        )
        assert strong_where_it_counts.normalized > strong_where_it_does_not.normalized

    def test_strict_mode_requires_every_criterion(self, rubric: Rubric) -> None:
        with pytest.raises(ValueError, match="missing criteria"):
            rubric.score(Annotation("t1", "ann-a", {"correctness": 4}))

    def test_lenient_mode_renormalises_over_supplied_criteria(
        self, rubric: Rubric
    ) -> None:
        scored = rubric.score(
            Annotation("t1", "ann-a", {"correctness": 4}), strict=False
        )
        # Only correctness applied: 0.75 * 3 = 2.25, renormalised over weight 3.
        assert scored.weighted_score == pytest.approx(2.25)
        assert scored.normalized == pytest.approx(0.75)

    def test_rejects_unknown_criteria(self, rubric: Rubric) -> None:
        with pytest.raises(ValueError, match="unknown criteria"):
            rubric.score(
                Annotation("t1", "a", {"correctness": 4, "clarity": 3, "vibes": 5})
            )

    def test_rejects_out_of_scale_score(self, rubric: Rubric) -> None:
        with pytest.raises(ValueError, match="outside scale"):
            rubric.score(Annotation("t1", "a", {"correctness": 9, "clarity": 3}))

    def test_score_all_preserves_order(self, rubric: Rubric) -> None:
        annotations = [
            Annotation("t1", "a", {"correctness": 5, "clarity": 5}),
            Annotation("t2", "b", {"correctness": 1, "clarity": 1}),
        ]
        assert [s.item_id for s in rubric.score_all(annotations)] == ["t1", "t2"]

    def test_criteria_with_different_scales_combine_fairly(self) -> None:
        # A 1-10 criterion must not dominate a 1-5 one purely through range.
        mixed = Rubric(
            name="mixed",
            criteria=[
                Criterion("wide", weight=1.0, scale=(1, 10)),
                Criterion("narrow", weight=1.0, scale=(1, 5)),
            ],
        )
        both_at_top = mixed.score(Annotation("t", "a", {"wide": 10, "narrow": 5}))
        assert both_at_top.normalized == pytest.approx(1.0)


class TestFromDict:
    def test_round_trip(self) -> None:
        rubric = Rubric.from_dict(
            {
                "name": "safety",
                "criteria": [
                    {"name": "harmlessness", "weight": 2, "scale": [1, 7]},
                    {"name": "honesty"},
                ],
            }
        )
        assert rubric.name == "safety"
        assert rubric.criteria[0].scale == (1, 7)
        assert rubric.criteria[1].weight == 1.0  # default

    def test_rejects_missing_name(self) -> None:
        with pytest.raises(ValueError, match="'name' must be a string"):
            Rubric.from_dict({"criteria": []})

    def test_rejects_non_list_criteria(self) -> None:
        with pytest.raises(ValueError, match="'criteria' must be a list"):
            Rubric.from_dict({"name": "x", "criteria": {}})

    def test_rejects_malformed_scale(self) -> None:
        with pytest.raises(ValueError, match="two-element list"):
            Rubric.from_dict({"name": "x", "criteria": [{"name": "a", "scale": [1]}]})


class TestAnnotationValidation:
    def test_rejects_empty_item_id(self) -> None:
        with pytest.raises(ValueError, match="item_id"):
            Annotation("", "a", {"x": 1})

    def test_rejects_empty_annotator_id(self) -> None:
        with pytest.raises(ValueError, match="annotator_id"):
            Annotation("t", "", {"x": 1})

    def test_rejects_empty_scores(self) -> None:
        with pytest.raises(ValueError, match="at least one score"):
            Annotation("t", "a", {})
