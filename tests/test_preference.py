"""Pairwise preference aggregation and Bradley-Terry fitting."""

from __future__ import annotations

import pytest

from eval_harness.models import Comparison, Verdict
from eval_harness.preference import bradley_terry, is_connected, rank, win_counts


def cmp(a: str, b: str, verdict: Verdict, item: str = "p1") -> Comparison:
    return Comparison(
        item_id=item, annotator_id="ann-a", response_a=a, response_b=b, verdict=verdict
    )


class TestComparisonValidation:
    def test_rejects_self_comparison(self) -> None:
        with pytest.raises(ValueError, match="against itself"):
            cmp("model-x", "model-x", Verdict.A)


class TestWinCounts:
    def test_counts_decisive_verdicts(self) -> None:
        wins, models = win_counts([cmp("a", "b", Verdict.A), cmp("a", "b", Verdict.B)])
        assert wins[("a", "b")] == 1.0
        assert wins[("b", "a")] == 1.0
        assert models == {"a", "b"}

    def test_ties_split_half_each_way(self) -> None:
        wins, _ = win_counts([cmp("a", "b", Verdict.TIE)])
        assert wins[("a", "b")] == 0.5
        assert wins[("b", "a")] == 0.5


class TestConnectivity:
    def test_connected_graph(self) -> None:
        wins, models = win_counts([cmp("a", "b", Verdict.A), cmp("b", "c", Verdict.A)])
        assert is_connected(wins, models)

    def test_disconnected_graph(self) -> None:
        wins, models = win_counts([cmp("a", "b", Verdict.A), cmp("c", "d", Verdict.A)])
        assert not is_connected(wins, models)


class TestBradleyTerry:
    def test_even_record_gives_equal_strength(self) -> None:
        strengths = bradley_terry([cmp("a", "b", Verdict.A), cmp("a", "b", Verdict.B)])
        assert strengths["a"] == pytest.approx(0.5, abs=1e-6)
        assert strengths["b"] == pytest.approx(0.5, abs=1e-6)

    def test_two_model_case_matches_closed_form(self) -> None:
        # For two models the MLE is exact: P(a beats b) = s_a / (s_a + s_b)
        # equals the observed win rate. Three wins from four -> s_a = 0.75.
        comparisons = [
            cmp("a", "b", Verdict.A),
            cmp("a", "b", Verdict.A),
            cmp("a", "b", Verdict.A),
            cmp("a", "b", Verdict.B),
        ]
        strengths = bradley_terry(comparisons)
        assert strengths["a"] == pytest.approx(0.75, abs=1e-6)
        assert strengths["b"] == pytest.approx(0.25, abs=1e-6)

    def test_strengths_sum_to_one(self) -> None:
        strengths = bradley_terry(
            [cmp("a", "b", Verdict.A), cmp("b", "c", Verdict.A), cmp("a", "c", Verdict.A)]
        )
        assert sum(strengths.values()) == pytest.approx(1.0)

    def test_respects_transitive_ordering(self) -> None:
        strengths = bradley_terry(
            [cmp("a", "b", Verdict.A), cmp("b", "c", Verdict.A), cmp("a", "c", Verdict.A)]
        )
        assert strengths["a"] > strengths["b"] > strengths["c"]

    def test_accounts_for_opponent_quality(self) -> None:
        # Both 'strong' and 'padded' win twice, but 'strong' beat the model
        # that beat everyone else. Win rate alone cannot tell them apart;
        # Bradley-Terry can.
        comparisons = [
            cmp("strong", "elite", Verdict.A),
            cmp("strong", "elite", Verdict.A),
            cmp("elite", "weak", Verdict.A),
            cmp("elite", "weak", Verdict.A),
            cmp("padded", "weak", Verdict.A),
            cmp("padded", "weak", Verdict.A),
            cmp("strong", "padded", Verdict.TIE),
        ]
        strengths = bradley_terry(comparisons)
        assert strengths["strong"] > strengths["padded"]

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="no comparisons"):
            bradley_terry([])

    def test_rejects_disconnected_graph(self) -> None:
        with pytest.raises(ValueError, match="disconnected"):
            bradley_terry([cmp("a", "b", Verdict.A), cmp("c", "d", Verdict.A)])


class TestRank:
    def test_orders_by_strength(self) -> None:
        standings = rank(
            [cmp("a", "b", Verdict.A), cmp("b", "c", Verdict.A), cmp("a", "c", Verdict.A)]
        )
        assert [s.model for s in standings] == ["a", "b", "c"]

    def test_win_rate_counts_ties_as_half(self) -> None:
        standings = {s.model: s for s in rank([cmp("a", "b", Verdict.TIE)])}
        assert standings["a"].win_rate == pytest.approx(0.5)
        assert standings["a"].ties == 1
        assert standings["a"].wins == pytest.approx(0.0)

    def test_records_are_discrete_after_tie_adjustment(self) -> None:
        standings = {
            s.model: s
            for s in rank(
                [
                    cmp("a", "b", Verdict.A),
                    cmp("a", "b", Verdict.B),
                    cmp("a", "b", Verdict.TIE),
                ]
            )
        }
        a = standings["a"]
        assert (a.wins, a.losses, a.ties) == (1.0, 1.0, 1)
        assert a.matches == 3.0

    def test_degrades_gracefully_on_disconnected_graph(self) -> None:
        # Rather than refusing to report, strengths come back as 0.0 to signal
        # they are unfit, while win rates remain meaningful.
        standings = rank([cmp("a", "b", Verdict.A), cmp("c", "d", Verdict.A)])
        assert len(standings) == 4
        assert all(s.strength == 0.0 for s in standings)
        by_model = {s.model: s for s in standings}
        assert by_model["a"].win_rate == 1.0
        assert by_model["b"].win_rate == 0.0


class TestPriorSmoothing:
    def test_winless_model_keeps_a_finite_strength(self) -> None:
        # Under plain MLE a model that never wins collapses to zero, which is
        # correct but useless on a leaderboard. A prior keeps it ordered.
        comparisons = [
            cmp("a", "c", Verdict.A),
            cmp("b", "c", Verdict.A),
            cmp("a", "b", Verdict.A),
        ]
        unsmoothed = bradley_terry(comparisons, prior=0.0)
        smoothed = bradley_terry(comparisons, prior=0.5)

        assert unsmoothed["c"] == pytest.approx(0.0, abs=1e-9)
        assert smoothed["c"] > 0.0
        assert smoothed["a"] > smoothed["b"] > smoothed["c"]

    def test_prior_shrinks_toward_parity(self) -> None:
        comparisons = [cmp("a", "b", Verdict.A) for _ in range(4)]
        heavy = bradley_terry(comparisons, prior=2.0)
        light = bradley_terry(comparisons, prior=0.0)
        # Both rank 'a' first, but the smoothed gap is narrower.
        assert light["a"] > heavy["a"]
        assert heavy["a"] > heavy["b"]

    def test_prior_preserves_symmetry(self) -> None:
        strengths = bradley_terry(
            [cmp("a", "b", Verdict.A), cmp("a", "b", Verdict.B)], prior=1.0
        )
        assert strengths["a"] == pytest.approx(strengths["b"])

    def test_rejects_negative_prior(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            bradley_terry([cmp("a", "b", Verdict.A)], prior=-1.0)

    def test_rank_smooths_by_default(self) -> None:
        standings = {
            s.model: s
            for s in rank([cmp("a", "c", Verdict.A), cmp("b", "c", Verdict.A),
                           cmp("a", "b", Verdict.A)])
        }
        assert standings["c"].strength > 0.0
        assert standings["c"].win_rate == 0.0
