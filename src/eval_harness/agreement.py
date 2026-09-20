"""Inter-annotator agreement statistics.

Implements the coefficients an annotation QA lead actually needs:

============================  ==========  ==============  ================
Coefficient                   Raters      Missing data    Chance model
============================  ==========  ==============  ================
Percent agreement             2+          no              none
Cohen's kappa                 exactly 2   no              per-rater
Scott's pi                    exactly 2   no              pooled
Fleiss' kappa                 2+ (fixed)  no              pooled
Krippendorff's alpha          2+ (any)    yes             pooled
============================  ==========  ==============  ================

Cohen's kappa assumes each rater has their own marginal distribution; Scott's
pi and Fleiss' kappa pool the marginals. Krippendorff's alpha is the one to
reach for when raters did not all see every item, which is the normal case
for a real annotation queue.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence
from itertools import combinations
from typing import Any

Label = Hashable


def _require_equal_length(a: Sequence[Label], b: Sequence[Label]) -> None:
    if len(a) != len(b):
        raise ValueError(f"rating sequences differ in length: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("rating sequences must be non-empty")


def percent_agreement(a: Sequence[Label], b: Sequence[Label]) -> float:
    """Raw proportion of items two raters labelled identically.

    Reported alongside a chance-corrected coefficient, never instead of one:
    with skewed categories it can look excellent while agreement is no better
    than guessing.
    """
    _require_equal_length(a, b)
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def cohens_kappa(a: Sequence[Label], b: Sequence[Label]) -> float:
    """Cohen's kappa for exactly two raters over the same items.

    Returns:
        Agreement above chance: 1.0 is perfect, 0.0 is chance-level, and
        negative values mean systematic disagreement.

    Raises:
        ValueError: If the sequences differ in length or are empty.
    """
    _require_equal_length(a, b)
    n = len(a)

    observed = percent_agreement(a, b)

    count_a = Counter(a)
    count_b = Counter(b)
    # Each rater keeps their own marginal distribution — this is what separates
    # Cohen's kappa from Scott's pi.
    expected = sum((count_a[k] / n) * (count_b[k] / n) for k in set(count_a) | set(count_b))

    return _chance_correct(observed, expected)


def scotts_pi(a: Sequence[Label], b: Sequence[Label]) -> float:
    """Scott's pi: as Cohen's kappa, but with marginals pooled across raters."""
    _require_equal_length(a, b)
    n = len(a)

    observed = percent_agreement(a, b)

    pooled = Counter(a) + Counter(b)
    total = 2 * n
    expected = sum((count / total) ** 2 for count in pooled.values())

    return _chance_correct(observed, expected)


def fleiss_kappa(ratings: Sequence[Sequence[Label]]) -> float:
    """Fleiss' kappa for any fixed number of raters per item.

    Args:
        ratings: One sequence of labels per item. Every item must carry the
            same number of ratings, though not necessarily from the same
            raters — Fleiss' kappa is defined over counts, not identities.

    Returns:
        Chance-corrected agreement.

    Raises:
        ValueError: If items carry differing numbers of ratings, or fewer
            than two.
    """
    if not ratings:
        raise ValueError("ratings must be non-empty")

    widths = {len(r) for r in ratings}
    if len(widths) != 1:
        raise ValueError(
            f"every item needs the same number of ratings, saw {sorted(widths)}"
        )

    n_raters = widths.pop()
    if n_raters < 2:
        raise ValueError("Fleiss' kappa needs at least two ratings per item")

    n_items = len(ratings)
    categories = sorted({label for row in ratings for label in row}, key=repr)

    # P_i: proportion of rater *pairs* on item i that agree.
    agreement_per_item = []
    category_totals: Counter[Label] = Counter()

    for row in ratings:
        counts = Counter(row)
        category_totals.update(counts)
        pairs = sum(c * (c - 1) for c in counts.values())
        agreement_per_item.append(pairs / (n_raters * (n_raters - 1)))

    observed = sum(agreement_per_item) / n_items

    total_ratings = n_items * n_raters
    expected = sum(
        (category_totals[c] / total_ratings) ** 2 for c in categories
    )

    return _chance_correct(observed, expected)


def krippendorffs_alpha(
    ratings: Mapping[Any, Mapping[Any, Label]] | Sequence[Mapping[Any, Label]],
) -> float:
    """Krippendorff's alpha for nominal data, tolerating missing ratings.

    Unlike the kappa family this does not require every rater to have seen
    every item, which is what makes it usable on a real review queue where
    coverage is uneven.

    Args:
        ratings: Either a mapping of ``item_id -> {rater_id: label}``, or a
            sequence of such per-item mappings. Item and rater identifiers are
            never inspected, only the labels, so their types are unconstrained.
            Items rated by fewer than two raters carry no pairwise information
            and are skipped.

    Returns:
        1.0 for perfect agreement, 0.0 for chance-level, negative for
        systematic disagreement.

    Raises:
        ValueError: If no item has at least two ratings.
    """
    units = list(ratings.values()) if isinstance(ratings, Mapping) else list(ratings)
    usable = [u for u in units if len(u) >= 2]

    if not usable:
        raise ValueError("Krippendorff's alpha needs at least one item with 2+ ratings")

    # Coincidence matrix: every ordered pair of ratings within a unit, weighted
    # by 1/(m_u - 1) so units with more raters do not dominate.
    coincidence: defaultdict[tuple[Label, Label], float] = defaultdict(float)

    for unit in usable:
        labels = list(unit.values())
        m = len(labels)
        weight = 1.0 / (m - 1)
        for x, y in combinations(labels, 2):
            coincidence[(x, y)] += weight
            coincidence[(y, x)] += weight

    marginals: defaultdict[Label, float] = defaultdict(float)
    for (x, _), value in coincidence.items():
        marginals[x] += value
    total = sum(marginals.values())

    if total <= 1:
        raise ValueError("not enough paired ratings to estimate alpha")

    # Nominal metric: disagreement is 1 for any mismatched pair, 0 otherwise.
    observed_disagreement = sum(
        value for (x, y), value in coincidence.items() if x != y
    )
    expected_disagreement = sum(
        marginals[x] * marginals[y]
        for x in marginals
        for y in marginals
        if x != y
    ) / (total - 1)

    if expected_disagreement == 0:
        # Every rating is the same label; there is nothing to disagree about.
        return 1.0

    return 1.0 - (observed_disagreement / expected_disagreement)


def _chance_correct(observed: float, expected: float) -> float:
    """Apply the ``(po - pe) / (1 - pe)`` correction, guarding pe == 1."""
    if expected >= 1.0:
        # Only one category was ever used: agreement is certain, so the
        # coefficient is undefined. Perfect observed agreement reports as 1.
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)
