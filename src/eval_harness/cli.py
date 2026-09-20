"""Command line interface.

Three subcommands mirroring the three questions an eval lead asks of a batch:
how good is it, do the annotators agree, and which model won.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .agreement import fleiss_kappa, krippendorffs_alpha
from .loaders import LoadError, load_annotations, load_comparisons, load_rubric
from .preference import rank
from .rubric import Rubric


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-harness",
        description="Rubric scoring, agreement statistics and preference ranking.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="apply a rubric to an annotation batch")
    score.add_argument("--rubric", required=True, type=Path, help="rubric JSON file")
    score.add_argument("--annotations", required=True, type=Path, help="annotations JSONL")
    score.add_argument(
        "--lenient",
        action="store_true",
        help="allow partial annotations, renormalising over supplied criteria",
    )
    score.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    agree = sub.add_parser("agreement", help="inter-annotator agreement for a batch")
    agree.add_argument("--annotations", required=True, type=Path)
    agree.add_argument(
        "--criterion",
        required=True,
        help="which rubric criterion to measure agreement on",
    )
    agree.add_argument("--json", action="store_true")

    ranking = sub.add_parser("rank", help="rank models from pairwise comparisons")
    ranking.add_argument("--comparisons", required=True, type=Path)
    ranking.add_argument("--json", action="store_true")

    return parser


def _cmd_score(args: argparse.Namespace) -> int:
    rubric = Rubric.from_dict(load_rubric(args.rubric))
    annotations = load_annotations(args.annotations)
    scored = rubric.score_all(annotations, strict=not args.lenient)

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "item_id": s.item_id,
                        "annotator_id": s.annotator_id,
                        "weighted_score": round(s.weighted_score, 4),
                        "normalized": round(s.normalized, 4),
                        "per_criterion": {k: round(v, 4) for k, v in s.per_criterion.items()},
                    }
                    for s in scored
                ],
                indent=2,
            )
        )
        return 0

    print(f"rubric: {rubric.name}  ({len(rubric.criteria)} criteria)")
    print(f"{'item':<20} {'annotator':<14} {'weighted':>9} {'normalised':>11}")
    print("-" * 57)
    for s in scored:
        print(
            f"{s.item_id:<20} {s.annotator_id:<14} "
            f"{s.weighted_score:>9.3f} {s.normalized:>11.3f}"
        )

    if scored:
        mean = sum(s.normalized for s in scored) / len(scored)
        print("-" * 57)
        print(f"{'mean normalised':<35} {mean:>21.3f}")
    return 0


def _cmd_agreement(args: argparse.Namespace) -> int:
    annotations = load_annotations(args.annotations)

    by_item: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for a in annotations:
        if args.criterion not in a.scores:
            continue
        by_item[a.item_id][a.annotator_id] = a.scores[args.criterion]

    if not by_item:
        print(
            f"no annotations carry criterion {args.criterion!r}",
            file=sys.stderr,
        )
        return 2

    alpha = krippendorffs_alpha(by_item)

    # Fleiss' kappa needs a fixed number of raters per item; only report it
    # when the batch actually has complete coverage.
    widths = {len(v) for v in by_item.values()}
    kappa: float | None = None
    if len(widths) == 1 and widths != {1}:
        kappa = fleiss_kappa([list(v.values()) for v in by_item.values()])

    covered = sum(1 for v in by_item.values() if len(v) >= 2)
    payload = {
        "criterion": args.criterion,
        "items": len(by_item),
        "items_with_multiple_raters": covered,
        "krippendorff_alpha": round(alpha, 4),
        "fleiss_kappa": round(kappa, 4) if kappa is not None else None,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"criterion              {payload['criterion']}")
    print(f"items                  {payload['items']}")
    print(f"items with 2+ raters   {payload['items_with_multiple_raters']}")
    print(f"Krippendorff's alpha   {payload['krippendorff_alpha']:.4f}")
    if kappa is None:
        print("Fleiss' kappa          n/a (uneven rater coverage)")
    else:
        print(f"Fleiss' kappa          {kappa:.4f}")
    return 0


def _cmd_rank(args: argparse.Namespace) -> int:
    comparisons = load_comparisons(args.comparisons)
    standings = rank(comparisons)

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "model": s.model,
                        "wins": s.wins,
                        "losses": s.losses,
                        "ties": s.ties,
                        "win_rate": round(s.win_rate, 4),
                        "strength": round(s.strength, 6),
                    }
                    for s in standings
                ],
                indent=2,
            )
        )
        return 0

    print(f"{'model':<24} {'W':>6} {'L':>6} {'T':>4} {'win%':>7} {'strength':>10}")
    print("-" * 61)
    for s in standings:
        print(
            f"{s.model:<24} {s.wins:>6.1f} {s.losses:>6.1f} {s.ties:>4} "
            f"{s.win_rate * 100:>6.1f}% {s.strength:>10.4f}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "score": _cmd_score,
        "agreement": _cmd_agreement,
        "rank": _cmd_rank,
    }

    try:
        return handlers[args.command](args)
    except LoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
