"""Turning pairwise preferences into a ranking.

Win rate answers "how often did this model win?". Bradley-Terry answers the
question that actually matters for a reward model: "how strong is this model,
given *who* it was up against?" A model that only ever faced weak opponents
should not outrank one that beat strong ones.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .models import Comparison, Verdict


@dataclass(frozen=True, slots=True)
class Standing:
    """One model's aggregate record."""

    model: str
    wins: float
    losses: float
    ties: int
    strength: float

    @property
    def matches(self) -> float:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        """Ties count as half a win, the usual convention for preference data."""
        if self.matches == 0:
            return 0.0
        return (self.wins + 0.5 * self.ties) / self.matches


def win_counts(
    comparisons: Iterable[Comparison],
) -> tuple[dict[tuple[str, str], float], set[str]]:
    """Build the pairwise win matrix.

    Returns:
        ``(wins, models)`` where ``wins[(i, j)]`` is how often ``i`` was
        preferred over ``j``. A tie contributes 0.5 to each direction.
    """
    wins: defaultdict[tuple[str, str], float] = defaultdict(float)
    models: set[str] = set()

    for c in comparisons:
        models.update((c.response_a, c.response_b))
        if c.verdict is Verdict.A:
            wins[(c.response_a, c.response_b)] += 1.0
        elif c.verdict is Verdict.B:
            wins[(c.response_b, c.response_a)] += 1.0
        else:
            wins[(c.response_a, c.response_b)] += 0.5
            wins[(c.response_b, c.response_a)] += 0.5

    return dict(wins), models


def is_connected(wins: Mapping[tuple[str, str], float], models: set[str]) -> bool:
    """Whether every model is reachable from every other through comparisons.

    Bradley-Terry strengths are only identifiable on a connected comparison
    graph. On a disconnected one, two clusters can be scaled independently and
    the fit is meaningless — so we check rather than silently return numbers.
    """
    if len(models) <= 1:
        return True

    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for (i, j), count in wins.items():
        if count > 0:
            adjacency[i].add(j)
            adjacency[j].add(i)

    start = next(iter(models))
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)

    return seen == models


def bradley_terry(
    comparisons: Sequence[Comparison],
    *,
    prior: float = 0.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-9,
) -> dict[str, float]:
    """Fit Bradley-Terry strengths by minorization-maximization.

    The model says ``P(i beats j) = s_i / (s_i + s_j)``. Strengths are returned
    normalised to sum to 1, so they read as a share of total ability.

    Args:
        comparisons: Pairwise verdicts. Ties count as half a win each way.
        prior: Pseudo-wins added in both directions for every pair that
            actually met, i.e. Laplace smoothing on the win matrix. With
            ``prior=0`` this is the plain maximum-likelihood fit, under which a
            model that never won has no finite strength and collapses to zero.
            A small prior (0.5 is a reasonable default for leaderboards) turns
            the fit into a MAP estimate and keeps every strength finite and
            ordered. It shrinks all strengths toward parity, so use 0 when you
            want the unbiased fit and a prior when you want a usable ranking.
        max_iterations: Cap on MM iterations.
        tolerance: Stop once the largest strength moves less than this.

    Returns:
        ``{model: strength}``, summing to 1.

    Raises:
        ValueError: If there are no comparisons, the prior is negative, or the
            comparison graph is disconnected so strengths are not identifiable.
    """
    if not comparisons:
        raise ValueError("no comparisons supplied")
    if prior < 0:
        raise ValueError(f"prior must be non-negative, got {prior}")

    wins, models = win_counts(comparisons)

    if not is_connected(wins, models):
        raise ValueError(
            "comparison graph is disconnected; Bradley-Terry strengths are not "
            "identifiable. Add comparisons bridging the clusters."
        )

    ordered = sorted(models)

    # Matches played between each unordered pair, counted once. The prior adds
    # a notional win each way, so it only applies to pairs that actually met.
    pair_totals: dict[tuple[str, str], float] = {}
    for i, j in ((i, j) for i in ordered for j in ordered if i < j):
        played = wins.get((i, j), 0.0) + wins.get((j, i), 0.0)
        if played > 0:
            pair_totals[(i, j)] = played + 2 * prior

    smoothed_wins = dict(wins)
    if prior > 0:
        for i, j in pair_totals:
            smoothed_wins[(i, j)] = wins.get((i, j), 0.0) + prior
            smoothed_wins[(j, i)] = wins.get((j, i), 0.0) + prior

    total_wins = {
        m: sum(smoothed_wins.get((m, o), 0.0) for o in ordered) for m in ordered
    }

    strengths = {m: 1.0 / len(ordered) for m in ordered}

    for _ in range(max_iterations):
        updated: dict[str, float] = {}

        for m in ordered:
            denominator = 0.0
            for (i, j), played in pair_totals.items():
                if m == i:
                    other = j
                elif m == j:
                    other = i
                else:
                    continue
                denominator += played / (strengths[m] + strengths[other])

            # A model that never won gets a floor rather than collapsing to
            # zero, which would make the normalisation unstable.
            updated[m] = total_wins[m] / denominator if denominator > 0 else 1e-12

        total = sum(updated.values())
        updated = {m: v / total for m, v in updated.items()}

        shift = max(abs(updated[m] - strengths[m]) for m in ordered)
        strengths = updated
        if shift < tolerance:
            break

    return strengths


def rank(comparisons: Sequence[Comparison], *, prior: float = 0.5) -> list[Standing]:
    """Aggregate comparisons into a ranked table, strongest first.

    Defaults to a smoothed fit (``prior=0.5``) because a leaderboard showing a
    winless model at exactly zero strength is not actionable. Pass ``prior=0``
    for the unbiased maximum-likelihood strengths.
    """
    wins, models = win_counts(comparisons)

    try:
        strengths = bradley_terry(comparisons, prior=prior)
    except ValueError:
        # Disconnected graph: fall back to win rate alone rather than refusing
        # to report anything. Strength is reported as 0.0 to signal it is unfit.
        strengths = dict.fromkeys(models, 0.0)

    tie_counts: defaultdict[str, int] = defaultdict(int)
    for c in comparisons:
        if c.verdict is Verdict.TIE:
            tie_counts[c.response_a] += 1
            tie_counts[c.response_b] += 1

    standings = []
    for m in models:
        raw_wins = sum(wins.get((m, o), 0.0) for o in models if o != m)
        raw_losses = sum(wins.get((o, m), 0.0) for o in models if o != m)
        ties = tie_counts[m]
        # Ties were folded into both directions at 0.5; peel them back out so
        # wins/losses/ties read as discrete counts.
        standings.append(
            Standing(
                model=m,
                wins=raw_wins - 0.5 * ties,
                losses=raw_losses - 0.5 * ties,
                ties=ties,
                strength=strengths.get(m, 0.0),
            )
        )

    return sorted(standings, key=lambda s: (-s.strength, -s.win_rate, s.model))
