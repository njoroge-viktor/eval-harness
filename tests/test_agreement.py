"""Agreement coefficients, checked against hand-computed values.

Each non-trivial expectation carries the arithmetic that produced it, so a
future change that shifts a number has to argue with the derivation rather
than just re-baselining the test.
"""

from __future__ import annotations

import pytest

from eval_harness.agreement import (
    cohens_kappa,
    fleiss_kappa,
    krippendorffs_alpha,
    percent_agreement,
    scotts_pi,
)


class TestPercentAgreement:
    def test_identical_sequences(self) -> None:
        assert percent_agreement(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_total_disagreement(self) -> None:
        assert percent_agreement(["a", "a"], ["b", "b"]) == 0.0

    def test_partial(self) -> None:
        assert percent_agreement(["a", "b", "c", "d"], ["a", "b", "x", "y"]) == 0.5

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            percent_agreement(["a"], ["a", "b"])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            percent_agreement([], [])


class TestCohensKappa:
    def test_perfect_agreement(self) -> None:
        assert cohens_kappa([1, 2, 3, 1], [1, 2, 3, 1]) == pytest.approx(1.0)

    def test_chance_level_is_zero(self) -> None:
        # po = 2/4 = 0.5
        # pe = (2/4)(2/4) + (2/4)(2/4) = 0.5
        # kappa = (0.5 - 0.5) / (1 - 0.5) = 0
        assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)

    def test_hand_computed(self) -> None:
        # a = [1,1,1,0], b = [1,1,0,0]
        # po = 3/4 = 0.75
        # pe = (3/4)(2/4) + (1/4)(2/4) = 0.375 + 0.125 = 0.5
        # kappa = (0.75 - 0.5) / 0.5 = 0.5
        assert cohens_kappa([1, 1, 1, 0], [1, 1, 0, 0]) == pytest.approx(0.5)

    def test_systematic_disagreement_is_negative(self) -> None:
        assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) < 0

    def test_single_category_used_throughout(self) -> None:
        # pe == 1: the coefficient is undefined, and perfect observed
        # agreement is reported as 1.0 rather than a division by zero.
        assert cohens_kappa(["a", "a"], ["a", "a"]) == 1.0


class TestScottsPi:
    def test_perfect_agreement(self) -> None:
        assert scotts_pi([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_pools_marginals_unlike_cohen(self) -> None:
        # Raters with different marginals: pi and kappa must diverge.
        a = [1, 1, 1, 1, 0, 0]
        b = [1, 1, 0, 0, 0, 0]
        assert scotts_pi(a, b) != pytest.approx(cohens_kappa(a, b))


class TestFleissKappa:
    def test_perfect_agreement(self) -> None:
        assert fleiss_kappa([["a", "a", "a"], ["b", "b", "b"]]) == pytest.approx(1.0)

    def test_hand_computed(self) -> None:
        # item 1 = [A,A,A]: agreeing pairs 3*2 = 6, P_1 = 6/(3*2) = 1.0
        # item 2 = [A,A,B]: agreeing pairs 2*1 = 2, P_2 = 2/6 = 1/3
        # P_bar = (1 + 1/3) / 2 = 2/3
        # totals A=5, B=1 over 6 ratings -> P_e = (5/6)^2 + (1/6)^2 = 26/36
        # kappa = (2/3 - 26/36) / (1 - 26/36) = -0.2
        ratings = [["A", "A", "A"], ["A", "A", "B"]]
        assert fleiss_kappa(ratings) == pytest.approx(-0.2)

    def test_equals_scotts_pi_for_two_raters(self) -> None:
        # A known identity: with exactly two ratings per item, Fleiss' kappa
        # reduces to Scott's pi. Two independent code paths, one answer.
        a = [1, 1, 2, 3, 2, 1, 3, 3]
        b = [1, 2, 2, 3, 1, 1, 3, 2]
        paired = [[x, y] for x, y in zip(a, b, strict=True)]
        assert fleiss_kappa(paired) == pytest.approx(scotts_pi(a, b))

    def test_rejects_ragged_input(self) -> None:
        with pytest.raises(ValueError, match="same number of ratings"):
            fleiss_kappa([["a", "a"], ["b"]])

    def test_rejects_single_rating_per_item(self) -> None:
        with pytest.raises(ValueError, match="at least two ratings"):
            fleiss_kappa([["a"], ["b"]])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            fleiss_kappa([])


class TestKrippendorffsAlpha:
    def test_perfect_agreement(self) -> None:
        units = {"u1": {"a": "x", "b": "x"}, "u2": {"a": "y", "b": "y"}}
        assert krippendorffs_alpha(units) == pytest.approx(1.0)

    def test_hand_computed_chance_level(self) -> None:
        # u1 = {x, x}, u2 = {x, y}; each unit has m = 2 so weight = 1.
        # coincidences: (x,x) = 2, (x,y) = 1, (y,x) = 1
        # marginals: n_x = 3, n_y = 1, n = 4
        # D_o = 1 + 1 = 2
        # D_e = (1/3) * (3*1 + 1*3) = 2
        # alpha = 1 - 2/2 = 0
        units = {"u1": {"a": "x", "b": "x"}, "u2": {"a": "x", "b": "y"}}
        assert krippendorffs_alpha(units) == pytest.approx(0.0)

    def test_tolerates_uneven_rater_coverage(self) -> None:
        # u1 has three raters, u2 two, u3 only one (carries no pair info).
        # u1: 3 agreeing pairs at weight 1/2 -> (x,x) = 3
        # u2: 1 agreeing pair at weight 1   -> (y,y) = 2
        # No mismatched pairs, so D_o = 0 and alpha = 1 regardless of D_e.
        units = {
            "u1": {"a": "x", "b": "x", "c": "x"},
            "u2": {"a": "y", "b": "y"},
            "u3": {"a": "x"},
        }
        assert krippendorffs_alpha(units) == pytest.approx(1.0)

    def test_accepts_a_sequence_of_units(self) -> None:
        as_mapping = {"u1": {"a": "x", "b": "x"}, "u2": {"a": "x", "b": "y"}}
        as_sequence = list(as_mapping.values())
        assert krippendorffs_alpha(as_sequence) == pytest.approx(
            krippendorffs_alpha(as_mapping)
        )

    def test_single_label_everywhere_is_perfect(self) -> None:
        units = {"u1": {"a": "x", "b": "x"}, "u2": {"a": "x", "b": "x"}}
        assert krippendorffs_alpha(units) == 1.0

    def test_systematic_disagreement_is_negative(self) -> None:
        units = {
            "u1": {"a": "x", "b": "y"},
            "u2": {"a": "y", "b": "x"},
            "u3": {"a": "x", "b": "y"},
            "u4": {"a": "y", "b": "x"},
        }
        assert krippendorffs_alpha(units) < 0

    def test_requires_a_multiply_rated_item(self) -> None:
        with pytest.raises(ValueError, match="2\\+ ratings"):
            krippendorffs_alpha({"u1": {"a": "x"}})
